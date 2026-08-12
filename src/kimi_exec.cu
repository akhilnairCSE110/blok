#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ugds.h>

void ck(cudaError_t e, const char *m);
[[noreturn]] void die(const std::string &m);

struct Tensor {
  std::string name, role, slot, dtype, shape, file;
  std::uint64_t off = 0, bytes = 0, align = 4096, data = 0, data_bytes = 0;
  int layer = -1, expert = -1;
};

struct Args {
  std::string index, prompt_tokens;
  std::uint64_t tokens = 0;
};

struct RuntimeIndex {
  std::unordered_map<std::string, Tensor> tensors;
};

struct Generation {
  std::vector<std::uint32_t> ids;
  bool eos = false;
};

struct FileExtent {
  std::uint64_t logical = 0, bytes = 0, device = 0;
};

struct Io {
  std::unordered_map<std::string, std::vector<FileExtent>> file_extents;
  std::string map_device;
  bool has_kv_scratch = false;
  std::uint64_t kv_base = 0, kv_bytes = 0;
  int fd = -1;
  uGDSHandle_t fh = nullptr;
  Io() {
    const char *dev = std::getenv("BLOK_UGDS_DEVICE"), *map = std::getenv("BLOK_UGDS_MAP");
    if (!dev || !map) die("BLOK_UGDS_DEVICE and BLOK_UGDS_MAP are required");
    ug(uGDSDriverOpen(), "uGDSDriverOpen");
    fd = open(dev, O_RDWR);
    if (fd < 0) die(std::string("open uGDS device failed: ") + std::strerror(errno));
    uGDSDescr_t d = {};
    d.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
    d.handle.fd = fd;
    ug(uGDSHandleRegister(&fh, &d), "uGDSHandleRegister");
    std::ifstream in(map);
    if (!in) die("open BLOK_UGDS_MAP failed");
    std::string line;
    bool header = false;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      std::string tag, file;
      if (!(ss >> tag)) continue;
      if (tag == "blok-ugds-map-v1") { header = true; continue; }
      if (tag == "block_size") {
        std::uint64_t size;
        if (!(ss >> size) || size != 4096) die("BLOK_UGDS_MAP block size must be 4096");
        continue;
      }
      if (tag == "device") {
        if (!(ss >> map_device)) die("bad BLOK_UGDS_MAP device line");
        continue;
      }
      if (tag == "kv_scratch") {
        if (!(ss >> kv_base >> kv_bytes)) die("bad BLOK_UGDS_MAP kv_scratch line");
        has_kv_scratch = true;
        continue;
      }
      if (tag == "file") {
        FileExtent e;
        if (!(ss >> file >> e.logical >> e.bytes >> e.device)) die("bad BLOK_UGDS_MAP file extent line");
        if (e.bytes == 0 || e.device % 4096 != 0 || e.logical % 4096 != 0 || e.bytes % 4096 != 0)
          die("BLOK_UGDS_MAP extents must be non-empty and 4096-byte aligned");
        file_extents[file].push_back(e);
      } else die("unknown BLOK_UGDS_MAP line: " + tag);
    }
    if (!header || map_device.empty() || !has_kv_scratch || file_extents.empty())
      die("BLOK_UGDS_MAP is incomplete");
    if (!kv_bytes || kv_base % 4096 || kv_bytes % 4096) die("BLOK_UGDS_MAP has invalid KV scratch");
    if (map_device != dev) die("BLOK_UGDS_DEVICE does not match BLOK_UGDS_MAP device");
  }
  ~Io() {
    if (fh) uGDSHandleDeregister(fh);
    if (fd >= 0) close(fd);
    uGDSDriverClose();
  }
  static void ug(uGDSError_t e, const char *m) {
    if (e.err != UGDS_SUCCESS) die(std::string(m) + ": " + uGDS_status_error(e.err));
  }
  void read_file_range(const std::string &file, std::uint64_t logical, std::uint64_t bytes, void *dst) {
    if (bytes == 0) return;
    if (logical % 4096 != 0 || bytes % 4096 != 0) die("uGDS reads must be 4096-byte aligned");
    auto found = file_extents.find(file);
    if (found == file_extents.end()) die("missing uGDS map entry: " + file);
    std::uint64_t done = 0;
    while (done < bytes) {
      std::uint64_t at = logical + done;
      const FileExtent *hit = nullptr;
      for (const auto &e : found->second) {
        if (at >= e.logical && at < e.logical + e.bytes) {
          hit = &e;
          break;
        }
      }
      if (!hit) die("uGDS map does not cover requested file range: " + file);
      std::uint64_t in_extent = at - hit->logical;
      std::uint64_t chunk = std::min(bytes - done, hit->bytes - in_extent);
      ssize_t n = uGDSRead(fh, dst, chunk, (off_t)(hit->device + in_extent), (off_t)done);
      if (n != (ssize_t)chunk) die("short extent uGDS read");
      done += chunk;
    }
  }
};

struct DeviceTensor {
  void *base = nullptr;
  std::uint64_t bytes = 0, skip = 0;
  DeviceTensor() = default;
  DeviceTensor(const DeviceTensor &) = delete;
  DeviceTensor &operator=(const DeviceTensor &) = delete;
  DeviceTensor(DeviceTensor &&o) noexcept : base(o.base), bytes(o.bytes), skip(o.skip) { o.base = nullptr; }
  DeviceTensor &operator=(DeviceTensor &&o) noexcept {
    if (this != &o) {
      if (base) cudaFree(base);
      base = o.base; bytes = o.bytes; skip = o.skip; o.base = nullptr;
    }
    return *this;
  }
  ~DeviceTensor() { if (base) cudaFree(base); }
  void reserve(std::uint64_t n) {
    if (bytes >= n) return;
    if (base) cudaFree(base);
    ck(cudaMalloc(&base, n), "cudaMalloc tensor");
    bytes = n;
  }
  __nv_bfloat16 *bf16() const { return reinterpret_cast<__nv_bfloat16 *>(static_cast<char *>(base) + skip); }
  std::uint32_t *u32() const { return reinterpret_cast<std::uint32_t *>(static_cast<char *>(base) + skip); }
};

struct Buf {
  float *p = nullptr;
  Buf() = default;
  Buf(const Buf &) = delete;
  Buf &operator=(const Buf &) = delete;
  Buf(Buf &&o) noexcept : p(o.p) { o.p = nullptr; }
  Buf &operator=(Buf &&o) noexcept {
    if (this != &o) { if (p) cudaFree(p); p = o.p; o.p = nullptr; }
    return *this;
  }
  ~Buf() { if (p) cudaFree(p); }
  void make(std::uint64_t n) { ck(cudaMalloc(&p, n * sizeof(float)), "cudaMalloc buf"); }
};

constexpr int LAYERS = 61, FIRST_DENSE = 1;
constexpr int HIDDEN = 7168, HEADS = 64, QK_NOPE = 128, QK_ROPE = 64, V_HEAD = 128;
constexpr int Q_RANK = 1536, KV_RANK = 512, KV_A = 576, EXPERTS = 384, TOPK = 8;
constexpr int DENSE = 18432, MOE = 2048, SHARED = 2048, VOCAB = 163840;
constexpr std::uint64_t MAX_CONTEXT = 262144;
constexpr int I4_GROUP = 32, I4_PER_WORD = 8, QK_HEAD = QK_NOPE + QK_ROPE, K_DIM = HEADS * QK_HEAD, V_DIM = HEADS * V_HEAD;
constexpr int HEAD_TILE = 256, EXPERT_TILE = 64, HIDDEN_TILE = 64, KV_TILE = 64;
constexpr float RMS_EPS = 1.0e-5f, ROPE_THETA = 50000.0f, ROUTED_SCALE = 2.827f;
constexpr float YARN_FACTOR = 64.0f, YARN_BETA_FAST = 32.0f, YARN_BETA_SLOW = 1.0f;
constexpr float YARN_ORIGINAL_CTX = 4096.0f, YARN_MSCALE_ALL_DIM = 1.0f;
constexpr std::uint32_t EOS_IM_END = 163586;

std::uint64_t u64(const char *s, const char *n);

constexpr std::uint64_t MAX_IO_BYTES = 4ull * 1024 * 1024;

struct KvCache {
  int lock_fd = -1;
  std::uint64_t k_layer_bytes = 0, v_layer_bytes = 0, ugds_base = 0;
  Buf kt, vt, score, hmax, hsum;
  Io &io;
  KvCache(Io &io_ref, std::uint64_t seq, int tile) : k_layer_bytes(seq * K_DIM * sizeof(float)), v_layer_bytes(seq * V_DIM * sizeof(float)), io(io_ref) {
    const char *base = std::getenv("BLOK_KV_UGDS_BASE"), *cap = std::getenv("BLOK_KV_UGDS_BYTES");
    if (!base || !cap) die("BLOK_KV_UGDS_BASE and BLOK_KV_UGDS_BYTES are required");
    ugds_base = u64(base, "BLOK_KV_UGDS_BASE");
    std::string lock_path = std::string("/tmp/blok-kv-") + std::to_string(ugds_base) + ".lock";
    lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) die("KV uGDS region is already reserved: " + lock_path);
    std::uint64_t capacity = u64(cap, "BLOK_KV_UGDS_BYTES");
    if (LAYERS * (k_layer_bytes + v_layer_bytes) > capacity) die("BLOK_KV_UGDS_BYTES is too small for requested KV schedule");
    if (capacity != io.kv_bytes || ugds_base != io.kv_base) die("KV environment does not match BLOK_UGDS_MAP");
    kt.make((std::uint64_t)tile * K_DIM); vt.make((std::uint64_t)tile * V_DIM); score.make((std::uint64_t)HEADS * tile);
    hmax.make(HEADS); hsum.make(HEADS);
  }
  ~KvCache() {
    if (lock_fd >= 0) { flock(lock_fd, LOCK_UN); close(lock_fd); }
  }
  std::uint64_t off(int layer, bool value, int pos = 0) const {
    return ugds_base + (std::uint64_t)layer * (k_layer_bytes + v_layer_bytes) + (value ? k_layer_bytes : 0) +
           (std::uint64_t)pos * (value ? V_DIM : K_DIM) * sizeof(float);
  }
  void store(int layer, int pos, const float *kp, const float *vp) {
    ck(cudaDeviceSynchronize(), "sync before KV write");
    Io::ug(uGDSBufRegister(kp, K_DIM * sizeof(float), 0), "uGDSBufRegister KV K");
    ssize_t nk = uGDSWrite(io.fh, kp, K_DIM * sizeof(float), (off_t)off(layer, false, pos), 0);
    Io::ug(uGDSBufDeregister(kp), "uGDSBufDeregister KV K");
    if (nk != (ssize_t)(K_DIM * sizeof(float))) die("short uGDS KV K write");
    Io::ug(uGDSBufRegister(vp, V_DIM * sizeof(float), 0), "uGDSBufRegister KV V");
    ssize_t nv = uGDSWrite(io.fh, vp, V_DIM * sizeof(float), (off_t)off(layer, true, pos), 0);
    Io::ug(uGDSBufDeregister(vp), "uGDSBufDeregister KV V");
    if (nv != (ssize_t)(V_DIM * sizeof(float))) die("short uGDS KV V write");
  }
  void load_tile(int layer, bool value, int start, int n) {
    ck(cudaDeviceSynchronize(), "sync before KV read");
    int dim = value ? V_DIM : K_DIM;
    float *dst = value ? vt.p : kt.p;
    std::uint64_t bytes = (std::uint64_t)n * dim * sizeof(float);
    Io::ug(uGDSBufRegister(dst, bytes, 0), "uGDSBufRegister KV tile");
    ssize_t got = uGDSRead(io.fh, dst, bytes, (off_t)off(layer, value, start), 0);
    Io::ug(uGDSBufDeregister(dst), "uGDSBufDeregister KV tile");
    if (got != (ssize_t)bytes) die("short uGDS KV tile read");
  }
};

__device__ float bf(const __nv_bfloat16 *p) { return __bfloat162float(*p); }
__device__ float bf(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

__global__ void bf16_to_f32_k(const __nv_bfloat16 *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = bf(x + i);
}

__global__ void add_k(float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = bf(a[i] + b[i]);
}

__global__ void silu_mul_k(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = bf(bf(a[i] / (1.0f + expf(-a[i]))) * b[i]);
}

__global__ void rmsnorm_k(const float *x, const __nv_bfloat16 *w, float *y, int n) {
  __shared__ float s[256];
  float v = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) v += x[i] * x[i];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  float scale = rsqrtf(s[0] / n + RMS_EPS);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = bf(bf(x[i] * scale) * bf(w + i));
}

__global__ void matvec_bf16_k(const __nv_bfloat16 *w, const float *x, float *y, int rows, int cols) {
  __shared__ float s[256];
  int r = blockIdx.x;
  float v = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) v += bf(w + (unsigned long long)r * cols + c) * x[c];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0 && r < rows) y[r] = bf(s[0]);
}

__global__ void matvec_bf16_f32_k(const __nv_bfloat16 *w, const float *x, float *y, int cols) {
  __shared__ float s[256];
  int r = blockIdx.x;
  float v = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) v += bf(w + (unsigned long long)r * cols + c) * x[c];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) y[r] = s[0];
}

__global__ void matvec_i4_bf16_scale_k(const std::uint32_t *w, const __nv_bfloat16 *scale, const float *x, float *y, int rows, int cols) {
  __shared__ float s[256];
  int r = blockIdx.x;
  float v = 0;
  int groups = cols / I4_GROUP, packed_cols = cols / I4_PER_WORD;
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    float sc = bf(scale + (unsigned long long)r * groups + g);
    int base = g * I4_GROUP;
    for (int p = 0; p < I4_GROUP / I4_PER_WORD; ++p) {
      std::uint32_t word = w[(unsigned long long)r * packed_cols + g * (I4_GROUP / I4_PER_WORD) + p];
      for (int lane = 0; lane < I4_PER_WORD; ++lane) {
        int q = (int)((word >> (4 * lane)) & 0xf) - 8;
        v += x[base + p * I4_PER_WORD + lane] * bf((float)q * sc);
      }
    }
  }
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0 && r < rows) y[r] = bf(s[0]);
}

__device__ float yarn_correction_dim(float rotations, float dim) {
  return dim * logf(YARN_ORIGINAL_CTX / (rotations * 2.0f * CUDART_PI_F)) / (2.0f * logf(ROPE_THETA));
}

__device__ float yarn_ramp(float low, float high, float at) {
  if (low == high) high += 0.001f;
  return fminf(1.0f, fmaxf(0.0f, (at - low) / (high - low)));
}

__device__ float yarn_inv_freq(int pair, int rotary) {
  float dim = (float)rotary;
  float freq_extra = powf(ROPE_THETA, -2.0f * (float)pair / dim);
  float freq_inter = freq_extra / YARN_FACTOR;
  float low = floorf(yarn_correction_dim(YARN_BETA_FAST, dim));
  float high = ceilf(yarn_correction_dim(YARN_BETA_SLOW, dim));
  low = fmaxf(low, 0.0f);
  high = fminf(high, dim - 1.0f);
  float inv_freq_mask = 1.0f - yarn_ramp(low, high, (float)pair);
  return freq_inter * (1.0f - inv_freq_mask) + freq_extra * inv_freq_mask;
}

__global__ void rope_k(const float *q, const float *k, float *qr, float *kr, int pos, int heads, int stride, int rotary) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int rotary_pairs = heads * rotary / 2;
  if (i >= rotary_pairs) return;
  int half = rotary / 2, h = i / half, p = i % half, base = h * stride + QK_NOPE;
  float freq = yarn_inv_freq(p, rotary), c = bf(cosf(pos * freq)), s = bf(sinf(pos * freq));
  float q0 = q[base + p * 2], q1 = q[base + p * 2 + 1], k0 = k[base + p * 2], k1 = k[base + p * 2 + 1];
  qr[base + p] = bf(bf(q0 * c) - bf(q1 * s));
  qr[base + half + p] = bf(bf(q1 * c) + bf(q0 * s));
  kr[base + p] = bf(bf(k0 * c) - bf(k1 * s));
  kr[base + half + p] = bf(bf(k1 * c) + bf(k0 * s));
}

__global__ void k_rope_fill_k(float *k, const float *rope) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HEADS * QK_ROPE) k[(i / QK_ROPE) * (QK_NOPE + QK_ROPE) + QK_NOPE + i % QK_ROPE] = rope[i % QK_ROPE];
}

__global__ void split_kv_k(const float *kv, float *k, float *v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= HEADS * (QK_NOPE + V_HEAD)) return;
  int h = i / (QK_NOPE + V_HEAD), d = i % (QK_NOPE + V_HEAD);
  if (d < QK_NOPE) k[h * QK_HEAD + d] = kv[i];
  else v[h * V_HEAD + d - QK_NOPE] = kv[i];
}

__global__ void add_bf16_k(float *x, const __nv_bfloat16 *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += bf(b + i);
}

__global__ void sigmoid_k(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void router_topk_k(const float *score, const float *select_score, std::uint16_t *expert, float *weight) {
  __shared__ float s[384];
  int tid = threadIdx.x;
  if (tid >= EXPERTS) return;
  s[tid] = select_score[tid];
  __syncthreads();
  if (tid) return;
  float sum = 0;
  for (int k = 0; k < TOPK; ++k) {
    int id = 0;
    float best = -CUDART_INF_F;
    for (int i = 0; i < EXPERTS; ++i) if (s[i] > best) best = s[i], id = i;
    expert[k] = (std::uint16_t)id; weight[k] = score[id]; sum += score[id]; s[id] = -CUDART_INF_F;
  }
  for (int k = 0; k < TOPK; ++k) weight[k] = weight[k] / (sum + 1.0e-20f) * ROUTED_SCALE;
}

__global__ void expert_accum_k(const float *y, const float *w, int slot, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += y[i] * w[slot];
}

__global__ void round_bf16_k(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = bf(x[i]);
}

__global__ void fill_k(float *x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

__global__ void attn_tile_score_k(const float *q, const float *k, float *score, int n) {
  __shared__ float s[256];
  int h = blockIdx.x, t = blockIdx.y;
  if (t >= n) return;
  float v = 0;
  for (int i = threadIdx.x; i < QK_HEAD; i += blockDim.x)
    v += q[h * QK_HEAD + i] * k[(unsigned long long)t * K_DIM + h * QK_HEAD + i];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (!threadIdx.x) {
    float mscale = 0.1f * YARN_MSCALE_ALL_DIM * logf(YARN_FACTOR) + 1.0f;
    score[h * n + t] = bf(bf(s[0]) * rsqrtf((float)QK_HEAD) * mscale * mscale);
  }
}

__global__ void attn_tile_max_k(const float *score, float *mx, int n) {
  __shared__ float s[256];
  int h = blockIdx.x;
  float v = -CUDART_INF_F;
  for (int t = threadIdx.x; t < n; t += blockDim.x) v = fmaxf(v, score[h * n + t]);
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + d]);
    __syncthreads();
  }
  if (!threadIdx.x) mx[h] = fmaxf(mx[h], s[0]);
}

__global__ void attn_tile_sum_k(const float *score, const float *mx, float *sum, int n) {
  __shared__ float s[256];
  int h = blockIdx.x;
  float value = 0;
  for (int t = threadIdx.x; t < n; t += blockDim.x) value += expf(score[h * n + t] - mx[h]);
  s[threadIdx.x] = value;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (!threadIdx.x) sum[h] += s[0];
}

__global__ void attn_tile_accum_k(const float *score, const float *v, const float *mx, const float *sum, float *out, int n) {
  int h = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= V_HEAD) return;
  float acc = 0, m = mx[h];
  for (int t = 0; t < n; ++t) {
    float p = bf(expf(score[h * n + t] - m) / sum[h]);
    acc += p * v[(unsigned long long)t * V_DIM + h * V_HEAD + d];
  }
  out[h * V_HEAD + d] += acc;
}

__global__ void attn_finish_k(float *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V_DIM) out[i] = bf(out[i]);
}

__global__ void argmax_k(const float *x, std::uint32_t *id, int n) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float best = -CUDART_INF_F;
  int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) if (x[i] > best) best = x[i], bi = i;
  sv[threadIdx.x] = best; si[threadIdx.x] = bi;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d && sv[threadIdx.x + d] > sv[threadIdx.x]) {
      sv[threadIdx.x] = sv[threadIdx.x + d];
      si[threadIdx.x] = si[threadIdx.x + d];
    }
    __syncthreads();
  }
  if (!threadIdx.x) *id = si[0];
}

[[noreturn]] void die(const std::string &m) {
  std::cerr << m << "\n";
  std::exit(1);
}

void ck(cudaError_t e, const char *m) {
  if (e != cudaSuccess) die(std::string(m) + ": " + cudaGetErrorString(e));
}

std::uint64_t u64(const char *s, const char *n) {
  char *end = nullptr;
  errno = 0;
  auto v = std::strtoull(s, &end, 10);
  if (errno || end == s || *end) die(std::string("bad ") + n);
  return v;
}

int i32(const std::string &s, const char *n) {
  char *end = nullptr;
  errno = 0;
  auto v = std::strtol(s.c_str(), &end, 10);
  if (errno || end == s.c_str() || *end) die(std::string("bad ") + n);
  return static_cast<int>(v);
}

int shape_dim(const Tensor &t, int dim) {
  int at = 0;
  std::size_t begin = 0;
  for (;;) {
    auto end = t.shape.find('x', begin);
    if (at == dim) return i32(t.shape.substr(begin, end == std::string::npos ? end : end - begin), "shape");
    if (end == std::string::npos) die("shape rank too small: " + t.name);
    begin = end + 1;
    ++at;
  }
}

Args args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) die("usage: blok-kimi-exec --index runtime-index.blok --prompt-tokens ids --tokens n");
    std::string f = argv[i];
    if (f == "--index") a.index = argv[i + 1];
    else if (f == "--prompt-tokens") a.prompt_tokens = argv[i + 1];
    else if (f == "--tokens") a.tokens = u64(argv[i + 1], "tokens");
    else die("unknown flag: " + f);
  }
  if (a.index.empty() || a.prompt_tokens.empty() || a.tokens == 0) die("bad args");
  return a;
}

std::vector<std::uint32_t> token_ids(const std::string &text) {
  std::vector<std::uint32_t> out;
  std::istringstream in(text);
  std::string value;
  while (std::getline(in, value, ',')) {
    auto id = u64(value.c_str(), "prompt token");
    if (id >= VOCAB) die("prompt token outside vocabulary");
    out.push_back((std::uint32_t)id);
  }
  if (out.empty()) die("prompt token list is empty");
  return out;
}

RuntimeIndex runtime_index(const std::string &path) {
  std::ifstream in(path);
  if (!in) die("missing runtime-index.blok; run scripts/model_fetch.py kimi-k2.6 materialize");
  RuntimeIndex out;
  std::string line;
  if (!std::getline(in, line) || line != "blok-runtime-index-v1") die("bad runtime index header");
  while (std::getline(in, line)) {
    if (!line.starts_with("tensor ")) die("bad runtime index line");
    std::istringstream ss(line);
    std::string tag, layer, expert;
    Tensor t;
    if (!(ss >> tag >> t.name >> t.role >> layer >> expert >> t.slot >> t.dtype >> t.shape >> t.off >>
          t.bytes >> t.align >> t.data >> t.data_bytes >> t.file)) die("incomplete runtime tensor line");
    std::string extra;
    if (ss >> extra) die("runtime tensor path cannot contain whitespace");
    t.layer = i32(layer, "layer");
    t.expert = i32(expert, "expert");
    std::string key = layer + ":" + expert + ":" + t.role + ":" + t.slot;
    if (!t.file.empty() && !out.tensors.emplace(key, std::move(t)).second) die("duplicate runtime tensor key: " + key);
  }
  if (out.tensors.empty()) die("runtime index has no file-backed tensors");
  return out;
}

const Tensor &need(const RuntimeIndex &rt, int layer, const std::string &role, const std::string &slot, int expert = -1) {
  std::string key = std::to_string(layer) + ":" + std::to_string(expert) + ":" + role + ":" + slot;
  auto found = rt.tensors.find(key);
  if (found == rt.tensors.end()) die("missing runtime tensor: " + key);
  return found->second;
}

const Tensor &bf16(const RuntimeIndex &rt, int layer, const std::string &role, const std::string &slot,
                   int rows, int cols = 0, int expert = -1) {
  const Tensor &t = need(rt, layer, role, slot, expert);
  std::string shape = std::to_string(rows) + (cols ? "x" + std::to_string(cols) : "");
  std::uint64_t bytes = (std::uint64_t)rows * (cols ? cols : 1) * sizeof(__nv_bfloat16);
  if (t.dtype != "bf16" || t.shape != shape || t.data_bytes != bytes) die("bad bf16 tensor contract: " + t.name);
  return t;
}

void validate_i4_pair(const Tensor &w, const Tensor &scale, int rows, int cols) {
  if (w.dtype != "i32") die("routed expert weight must be packed i32: " + w.name);
  if (scale.dtype != "bf16") die("routed expert scale must be bf16: " + scale.name);
  if (w.shape != std::to_string(rows) + "x" + std::to_string(cols / I4_PER_WORD) ||
      w.data_bytes != (std::uint64_t)rows * cols / I4_PER_WORD * sizeof(std::uint32_t)) die("bad routed expert packed shape: " + w.name);
  if (scale.shape != std::to_string(rows) + "x" + std::to_string(cols / I4_GROUP) ||
      scale.data_bytes != (std::uint64_t)rows * cols / I4_GROUP * sizeof(__nv_bfloat16)) die("bad routed expert scale shape: " + scale.name);
}

DeviceTensor load_tensor(Io &io, const Tensor &t) {
  if (t.dtype != "bf16" && t.dtype != "i32") die("only bf16 and i32 tensors are executable");
  if (t.data < t.off || t.data + t.data_bytes > t.off + t.bytes) die("tensor data range escapes aligned read");
  if (t.bytes > MAX_IO_BYTES) die("refusing oversized tensor read; use a tiled loader for " + t.name);
  DeviceTensor d;
  d.bytes = t.bytes;
  d.skip = t.data - t.off;
  ck(cudaMalloc(&d.base, t.bytes), "cudaMalloc tensor");
  Io::ug(uGDSBufRegister(d.base, t.bytes, 0), "uGDSBufRegister");
  io.read_file_range(t.file, t.off, t.bytes, d.base);
  Io::ug(uGDSBufDeregister(d.base), "uGDSBufDeregister");
  return d;
}

void load_tensor_slice_into(Io &io, const Tensor &t, std::uint64_t byte_start, std::uint64_t byte_len, DeviceTensor &d) {
  if (t.data + byte_start + byte_len > t.data + t.data_bytes) die("tensor slice out of range: " + t.name);
  std::uint64_t file_off = t.data + byte_start, aligned = file_off / t.align * t.align;
  d.skip = file_off - aligned;
  std::uint64_t bytes = ((d.skip + byte_len + t.align - 1) / t.align) * t.align;
  if (aligned < t.off || aligned + bytes > t.off + t.bytes) die("aligned tensor slice escapes materialized read range: " + t.name);
  if (bytes > MAX_IO_BYTES) die("refusing oversized tensor slice read: " + t.name);
  d.reserve(bytes);
  Io::ug(uGDSBufRegister(d.base, bytes, 0), "uGDSBufRegister slice");
  io.read_file_range(t.file, aligned, bytes, d.base);
  Io::ug(uGDSBufDeregister(d.base), "uGDSBufDeregister slice");
}

DeviceTensor load_tensor_slice(Io &io, const Tensor &t, std::uint64_t byte_start, std::uint64_t byte_len) {
  DeviceTensor d;
  load_tensor_slice_into(io, t, byte_start, byte_len, d);
  return d;
}

int dtype_bytes(const Tensor &t) {
  if (t.dtype == "bf16") return sizeof(__nv_bfloat16);
  if (t.dtype == "i32") return 4;
  die("unsupported sliced dtype: " + t.dtype);
}

DeviceTensor load_rows(Io &io, const Tensor &t, int row, int rows, int row_bytes) {
  return load_tensor_slice(io, t, (std::uint64_t)row * row_bytes, (std::uint64_t)rows * row_bytes);
}

void embed_token(Io &io, const Tensor &emb, std::uint32_t id, float *x) {
  if (id >= VOCAB) die("token id outside embedding table");
  auto row = load_rows(io, emb, (int)id, 1, HIDDEN * sizeof(__nv_bfloat16));
  bf16_to_f32_k<<<(HIDDEN + 255) / 256, 256>>>(row.bf16(), x, HIDDEN);
}

void matvec_bf16_rows(Io &io, const Tensor &w, const float *x, float *y, int row, int rows, int cols) {
  auto tile = load_rows(io, w, row, rows, cols * dtype_bytes(w));
  matvec_bf16_k<<<rows, 256>>>(tile.bf16(), x, y + row, rows, cols);
}

void matvec_bf16_f32_rows(Io &io, const Tensor &w, const float *x, float *y, int rows, int cols) {
  for (int row = 0; row < rows; row += EXPERT_TILE) {
    int n = std::min(EXPERT_TILE, rows - row);
    auto tile = load_rows(io, w, row, n, cols * dtype_bytes(w));
    matvec_bf16_f32_k<<<n, 256>>>(tile.bf16(), x, y + row, cols);
  }
}

void matvec_bf16_all_rows(Io &io, const Tensor &w, const float *x, float *y, int rows, int cols, int tile_rows) {
  for (int r = 0; r < rows; r += tile_rows) matvec_bf16_rows(io, w, x, y, r, std::min(tile_rows, rows - r), cols);
}

void bf16_mlp(Io &io, const Tensor &gw, const Tensor &uw, const Tensor &dw, const float *x, float *down, float *gate, float *up, float *mid, float *tmp, int ffn) {
  for (int r = 0; r < ffn; r += EXPERT_TILE) {
    int rows = std::min(EXPERT_TILE, ffn - r);
    matvec_bf16_rows(io, gw, x, gate, r, rows, HIDDEN);
    matvec_bf16_rows(io, uw, x, up, r, rows, HIDDEN);
    silu_mul_k<<<(rows + 255) / 256, 256>>>(gate + r, up + r, mid + r, rows);
  }
  for (int r = 0; r < HIDDEN; r += HIDDEN_TILE) {
    int rows = std::min(HIDDEN_TILE, HIDDEN - r);
    matvec_bf16_rows(io, dw, mid, tmp, r, rows, ffn);
    add_k<<<(rows + 255) / 256, 256>>>(down + r, tmp + r, rows);
  }
}

void matvec_i4_slice(const DeviceTensor &w, const DeviceTensor &scale, const float *x, float *y, int rows, int cols) {
  matvec_i4_bf16_scale_k<<<rows, 256>>>(w.u32(), scale.bf16(), x, y, rows, cols);
}

void tiled_attention(KvCache &kv, int layer, int seq, int tile, const float *q, float *out) {
  fill_k<<<1, 256>>>(kv.hmax.p, -CUDART_INF_F, HEADS);
  ck(cudaMemset(out, 0, V_DIM * sizeof(float)), "clear attention out");
  for (int start = 0; start < seq; start += tile) {
    int n = std::min(tile, seq - start);
    kv.load_tile(layer, false, start, n);
    attn_tile_score_k<<<dim3(HEADS, n), 256>>>(q, kv.kt.p, kv.score.p, n);
    attn_tile_max_k<<<HEADS, 256>>>(kv.score.p, kv.hmax.p, n);
  }
  ck(cudaMemset(kv.hsum.p, 0, HEADS * sizeof(float)), "clear attention sum");
  for (int start = 0; start < seq; start += tile) {
    int n = std::min(tile, seq - start);
    kv.load_tile(layer, false, start, n);
    attn_tile_score_k<<<dim3(HEADS, n), 256>>>(q, kv.kt.p, kv.score.p, n);
    attn_tile_sum_k<<<HEADS, 256>>>(kv.score.p, kv.hmax.p, kv.hsum.p, n);
  }
  for (int start = 0; start < seq; start += tile) {
    int n = std::min(tile, seq - start);
    kv.load_tile(layer, false, start, n);
    attn_tile_score_k<<<dim3(HEADS, n), 256>>>(q, kv.kt.p, kv.score.p, n);
    kv.load_tile(layer, true, start, n);
    attn_tile_accum_k<<<dim3(HEADS, (V_HEAD + 255) / 256), 256>>>(kv.score.p, kv.vt.p, kv.hmax.p, kv.hsum.p, out, n);
  }
  attn_finish_k<<<(V_DIM + 255) / 256, 256>>>(out);
}

std::uint32_t sample_lm_head(Io &io, const Tensor &head, const float *x, float *logits, std::uint32_t *did) {
  float best = -std::numeric_limits<float>::infinity();
  std::uint32_t token = 0;
  DeviceTensor h;
  for (int row = 0; row < VOCAB; row += HEAD_TILE) {
    int rows = std::min(HEAD_TILE, VOCAB - row);
    load_tensor_slice_into(io, head, (std::uint64_t)row * HIDDEN * sizeof(__nv_bfloat16), (std::uint64_t)rows * HIDDEN * sizeof(__nv_bfloat16), h);
    matvec_bf16_k<<<rows, 256>>>(h.bf16(), x, logits, rows, HIDDEN);
    argmax_k<<<1, 256>>>(logits, did, rows);
    std::uint32_t local = 0;
    ck(cudaMemcpy(&local, did, sizeof(local), cudaMemcpyDeviceToHost), "cudaMemcpy lm head sample");
    float value = 0;
    ck(cudaMemcpy(&value, logits + local, sizeof(value), cudaMemcpyDeviceToHost), "cudaMemcpy lm head score");
    if (value > best) { best = value; token = (std::uint32_t)row + local; }
  }
  return token;
}

void validate_forward_contract(const RuntimeIndex &rt) {
  for (const auto &[key, t] : rt.tensors) {
    (void)key;
    if (t.slot.empty() || t.dtype.empty() || t.shape.empty()) die("runtime index has incomplete tensor metadata");
    if (t.layer >= LAYERS || t.expert >= EXPERTS) die("runtime index has out-of-range Kimi coordinates");
    if (t.align != 4096 || t.off % t.align || t.bytes % t.align || t.data < t.off ||
        t.data_bytes > t.bytes || t.data - t.off > t.bytes - t.data_bytes)
      die("runtime index has invalid tensor extent: " + t.name);
  }
  for (int i = 0; i < LAYERS; ++i) {
    bf16(rt, i, "attention_resident", "q_a_proj", Q_RANK, HIDDEN);
    bf16(rt, i, "attention_resident", "q_a_layernorm", Q_RANK);
    bf16(rt, i, "attention_resident", "q_b_proj", K_DIM, Q_RANK);
    bf16(rt, i, "attention_resident", "kv_a_proj_with_mqa", KV_A, HIDDEN);
    bf16(rt, i, "attention_resident", "kv_a_layernorm", KV_RANK);
    bf16(rt, i, "attention_resident", "kv_b_proj", HEADS * (QK_NOPE + V_HEAD), KV_RANK);
    bf16(rt, i, "attention_resident", "o_proj", HIDDEN, V_DIM);
    bf16(rt, i, "resident", "input_layernorm", HIDDEN);
    bf16(rt, i, "resident", "post_attention_layernorm", HIDDEN);
    if (i < FIRST_DENSE) {
      bf16(rt, i, "dense_ffn_rowcol", "gate_proj", DENSE, HIDDEN);
      bf16(rt, i, "dense_ffn_rowcol", "up_proj", DENSE, HIDDEN);
      bf16(rt, i, "dense_ffn_rowcol", "down_proj", HIDDEN, DENSE);
      continue;
    }
    bf16(rt, i, "router", "gate", EXPERTS, HIDDEN);
    const Tensor &bias = need(rt, i, "router", "e_score_correction_bias");
    if (bias.dtype != "bf16" || bias.shape != std::to_string(EXPERTS) ||
        bias.data_bytes != EXPERTS * sizeof(__nv_bfloat16))
      die("bad router correction bias: " + bias.name);
    bf16(rt, i, "shared_expert_resident", "gate_proj", SHARED, HIDDEN);
    bf16(rt, i, "shared_expert_resident", "up_proj", SHARED, HIDDEN);
    bf16(rt, i, "shared_expert_resident", "down_proj", HIDDEN, SHARED);
    for (int e = 0; e < EXPERTS; ++e) {
      validate_i4_pair(need(rt, i, "routed_expert", "gate_proj.weight_packed", e), need(rt, i, "routed_expert", "gate_proj.weight_scale", e), MOE, HIDDEN);
      validate_i4_pair(need(rt, i, "routed_expert", "up_proj.weight_packed", e), need(rt, i, "routed_expert", "up_proj.weight_scale", e), MOE, HIDDEN);
      validate_i4_pair(need(rt, i, "routed_expert", "down_proj.weight_packed", e), need(rt, i, "routed_expert", "down_proj.weight_scale", e), HIDDEN, MOE);
    }
  }
  bf16(rt, -1, "resident", "embed_tokens", VOCAB, HIDDEN);
  bf16(rt, -1, "resident", "norm", HIDDEN);
  bf16(rt, -1, "resident", "lm_head", VOCAB, HIDDEN);
}

Generation generate(Io &io, const RuntimeIndex &rt, const Args &a) {
  auto input = token_ids(a.prompt_tokens);
  auto made = std::vector<std::uint32_t>{};
  if (input.empty()) die("tokenizer produced no prompt tokens");
  if (a.tokens > MAX_CONTEXT || input.size() > MAX_CONTEXT - a.tokens) die("requested sequence exceeds Kimi context");
  const Tensor &emb = need(rt, -1, "resident", "embed_tokens");
  auto final_norm = load_tensor(io, need(rt, -1, "resident", "norm"));
  const Tensor &head = need(rt, -1, "resident", "lm_head");
  std::uint32_t *did = nullptr;
  std::uint16_t *dex = nullptr;
  ck(cudaMalloc(&did, sizeof(std::uint32_t)), "cudaMalloc sample");
  ck(cudaMalloc(&dex, TOPK * sizeof(std::uint16_t)), "cudaMalloc experts");
  Buf x, n, qa, q, q_rot, kva, kv, k, k_rot, v, av, router, route_select, ew, gate, up, mid, down, logits;
  x.make(HIDDEN); n.make(HIDDEN); qa.make(Q_RANK); q.make(HEADS * (QK_NOPE + QK_ROPE));
  q_rot.make(HEADS * (QK_NOPE + QK_ROPE));
  kva.make(KV_A); kv.make(HEADS * (QK_NOPE + V_HEAD)); k.make(HEADS * (QK_NOPE + QK_ROPE));
  k_rot.make(HEADS * (QK_NOPE + QK_ROPE));
  v.make(HEADS * V_HEAD); av.make(HEADS * V_HEAD);
  router.make(EXPERTS); route_select.make(EXPERTS); ew.make(TOPK);
  gate.make(DENSE); up.make(DENSE); mid.make(DENSE); down.make(HIDDEN); logits.make(HEAD_TILE);
  KvCache kv_cache(io, input.size() + a.tokens, KV_TILE);
  auto run = [&](std::uint32_t id, int pos, bool sample) {
    embed_token(io, emb, id, x.p);
    for (int l = 0; l < LAYERS; ++l) {
      auto ln0 = load_tensor(io, need(rt, l, "resident", "input_layernorm"));
      const Tensor &qaw = need(rt, l, "attention_resident", "q_a_proj");
      auto qaln = load_tensor(io, need(rt, l, "attention_resident", "q_a_layernorm"));
      const Tensor &qbw = need(rt, l, "attention_resident", "q_b_proj");
      const Tensor &kvaw = need(rt, l, "attention_resident", "kv_a_proj_with_mqa");
      auto kvaln = load_tensor(io, need(rt, l, "attention_resident", "kv_a_layernorm"));
      const Tensor &kvbw = need(rt, l, "attention_resident", "kv_b_proj");
      const Tensor &ow = need(rt, l, "attention_resident", "o_proj");
      rmsnorm_k<<<1, 256>>>(x.p, ln0.bf16(), n.p, HIDDEN);
      matvec_bf16_all_rows(io, qaw, n.p, qa.p, Q_RANK, HIDDEN, EXPERT_TILE);
      rmsnorm_k<<<1, 256>>>(qa.p, qaln.bf16(), qa.p, Q_RANK);
      matvec_bf16_all_rows(io, qbw, qa.p, q.p, K_DIM, Q_RANK, EXPERT_TILE);
      matvec_bf16_all_rows(io, kvaw, n.p, kva.p, KV_A, HIDDEN, EXPERT_TILE);
      rmsnorm_k<<<1, 256>>>(kva.p, kvaln.bf16(), kva.p, KV_RANK);
      matvec_bf16_all_rows(io, kvbw, kva.p, kv.p, HEADS * (QK_NOPE + V_HEAD), KV_RANK, EXPERT_TILE);
      split_kv_k<<<(HEADS * (QK_NOPE + V_HEAD) + 255) / 256, 256>>>(kv.p, k.p, v.p);
      k_rope_fill_k<<<(HEADS * QK_ROPE + 255) / 256, 256>>>(k.p, kva.p + KV_RANK);
      ck(cudaMemcpy(q_rot.p, q.p, K_DIM * sizeof(float), cudaMemcpyDeviceToDevice), "copy q before RoPE");
      ck(cudaMemcpy(k_rot.p, k.p, K_DIM * sizeof(float), cudaMemcpyDeviceToDevice), "copy k before RoPE");
      rope_k<<<(HEADS * QK_ROPE / 2 + 255) / 256, 256>>>(q.p, k.p, q_rot.p, k_rot.p, pos, HEADS, QK_HEAD, QK_ROPE);
      kv_cache.store(l, pos, k_rot.p, v.p);
      tiled_attention(kv_cache, l, pos + 1, KV_TILE, q_rot.p, av.p);
      matvec_bf16_all_rows(io, ow, av.p, down.p, HIDDEN, HEADS * V_HEAD, HIDDEN_TILE);
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
      auto ln1 = load_tensor(io, need(rt, l, "resident", "post_attention_layernorm"));
      rmsnorm_k<<<1, 256>>>(x.p, ln1.bf16(), n.p, HIDDEN);
      ck(cudaMemset(down.p, 0, HIDDEN * sizeof(float)), "cudaMemset ffn");
      if (l < FIRST_DENSE) {
        const Tensor &gw = need(rt, l, "dense_ffn_rowcol", "gate_proj"), &uw = need(rt, l, "dense_ffn_rowcol", "up_proj"),
                     &dw = need(rt, l, "dense_ffn_rowcol", "down_proj");
        bf16_mlp(io, gw, uw, dw, n.p, down.p, gate.p, up.p, mid.p, av.p, shape_dim(gw, 0));
      } else {
        const Tensor &rw = need(rt, l, "router", "gate");
        auto rb = load_tensor(io, need(rt, l, "router", "e_score_correction_bias"));
        matvec_bf16_f32_rows(io, rw, n.p, router.p, EXPERTS, HIDDEN);
        sigmoid_k<<<(EXPERTS + 255) / 256, 256>>>(router.p, EXPERTS);
        ck(cudaMemcpy(route_select.p, router.p, EXPERTS * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy router scores");
        add_bf16_k<<<(EXPERTS + 255) / 256, 256>>>(route_select.p, rb.bf16(), EXPERTS);
        router_topk_k<<<1, EXPERTS>>>(router.p, route_select.p, dex, ew.p);
        std::uint16_t ex[TOPK];
        ck(cudaMemcpy(ex, dex, sizeof(ex), cudaMemcpyDeviceToHost), "cudaMemcpy experts");
        for (int slot = 0; slot < TOPK; ++slot) {
          int expert = ex[slot];
          const Tensor &gwm = need(rt, l, "routed_expert", "gate_proj.weight_packed", expert);
          const Tensor &gsm = need(rt, l, "routed_expert", "gate_proj.weight_scale", expert);
          const Tensor &uwm = need(rt, l, "routed_expert", "up_proj.weight_packed", expert);
          const Tensor &usm = need(rt, l, "routed_expert", "up_proj.weight_scale", expert);
          const Tensor &dwm = need(rt, l, "routed_expert", "down_proj.weight_packed", expert);
          const Tensor &dsm = need(rt, l, "routed_expert", "down_proj.weight_scale", expert);
          for (int r = 0; r < MOE; r += EXPERT_TILE) {
            int rows = std::min(EXPERT_TILE, MOE - r);
            auto gw = load_rows(io, gwm, r, rows, HIDDEN / I4_PER_WORD * dtype_bytes(gwm));
            auto gs = load_rows(io, gsm, r, rows, HIDDEN / I4_GROUP * dtype_bytes(gsm));
            auto uw = load_rows(io, uwm, r, rows, HIDDEN / I4_PER_WORD * dtype_bytes(uwm));
            auto us = load_rows(io, usm, r, rows, HIDDEN / I4_GROUP * dtype_bytes(usm));
            matvec_i4_slice(gw, gs, n.p, gate.p + r, rows, HIDDEN);
            matvec_i4_slice(uw, us, n.p, up.p + r, rows, HIDDEN);
            silu_mul_k<<<(rows + 255) / 256, 256>>>(gate.p + r, up.p + r, mid.p + r, rows);
          }
          for (int r = 0; r < HIDDEN; r += HIDDEN_TILE) {
            int rows = std::min(HIDDEN_TILE, HIDDEN - r);
            auto dw = load_rows(io, dwm, r, rows, MOE / I4_PER_WORD * dtype_bytes(dwm));
            auto ds = load_rows(io, dsm, r, rows, MOE / I4_GROUP * dtype_bytes(dsm));
            matvec_i4_slice(dw, ds, mid.p, av.p + r, rows, MOE);
            expert_accum_k<<<(rows + 255) / 256, 256>>>(av.p + r, ew.p, slot, down.p + r, rows);
          }
        }
        round_bf16_k<<<(HIDDEN + 255) / 256, 256>>>(down.p, HIDDEN);
        const Tensor &sgw = need(rt, l, "shared_expert_resident", "gate_proj");
        const Tensor &suw = need(rt, l, "shared_expert_resident", "up_proj");
        const Tensor &sdw = need(rt, l, "shared_expert_resident", "down_proj");
        bf16_mlp(io, sgw, suw, sdw, n.p, down.p, gate.p, up.p, mid.p, av.p, shape_dim(sgw, 0));
      }
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
    }
    if (!sample) return std::uint32_t{0};
    rmsnorm_k<<<1, 256>>>(x.p, final_norm.bf16(), n.p, HIDDEN);
    std::uint32_t next = sample_lm_head(io, head, n.p, logits.p, did);
    ck(cudaDeviceSynchronize(), "decode sync");
    return next;
  };
  std::uint32_t next = input[0];
  for (std::size_t i = 0; i < input.size(); ++i) next = run(input[i], (int)i, i + 1 == input.size());
  for (std::uint64_t i = 0; i < a.tokens && next != EOS_IM_END; ++i) {
    made.push_back(next);
    if (i + 1 < a.tokens) next = run(next, (int)(input.size() + i), true);
  }
  cudaFree(did); cudaFree(dex);
  return {std::move(made), next == EOS_IM_END};
}

int main(int argc, char **argv) {
  Args a = args(argc, argv);
  auto rt = runtime_index(a.index);
  validate_forward_contract(rt);
  Io io;
  auto gen = generate(io, rt, a);
  std::cout << "{\"status\":\"ok\",\"token_ids\":[";
  for (std::size_t i = 0; i < gen.ids.size(); ++i) std::cout << (i ? "," : "") << gen.ids[i];
  std::cout << "],\"finish_reason\":\"" << (gen.eos ? "eos" : "length") << "\"}\n";
}
