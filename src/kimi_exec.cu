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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef BLOK_HAVE_UGDS
#include <ugds.h>
#else
#error "blok-kimi-exec requires uGDS; host-bounced fallback paths are intentionally disabled"
#endif

void ck(cudaError_t e, const char *m);
[[noreturn]] void die(const std::string &m);

struct Tensor {
  std::string name, role, slot, dtype, shape, file;
  std::uint64_t off = 0, bytes = 0, align = 4096, data = 0, data_bytes = 0;
  int layer = -1, expert = -1;
};

struct Args {
  std::string manifest, prompt;
  std::uint64_t tokens = 0, topk = 8;
};

struct RuntimeIndex {
  std::string tokenizer_blok;
  std::vector<Tensor> tensors;
};

struct Generation {
  std::string text;
  std::uint64_t tokens = 0;
};

struct Io {
  std::unordered_map<std::string, std::uint64_t> base;
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
    std::string file;
    std::uint64_t off;
    while (in >> file >> off) base[file] = off;
  }
  ~Io() {
    if (fh) uGDSHandleDeregister(fh);
    if (fd >= 0) close(fd);
    uGDSDriverClose();
  }
  static void ug(uGDSError_t e, const char *m) {
    if (e.err != UGDS_SUCCESS) die(std::string(m) + ": " + uGDS_status_error(e.err));
  }
};

struct Tokenizer {
  std::unordered_map<std::string, std::uint32_t> ids;
  std::unordered_map<std::string, int> merges;
  std::unordered_map<std::uint32_t, std::string> text;
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
  float *f32() const { return reinterpret_cast<float *>(static_cast<char *>(base) + skip); }
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
constexpr int Q_RANK = 1536, KV_RANK = 512, KV_A = 576, EXPERTS = 384, TOPK = 8, MOE = 2048, SHARED = 18432, VOCAB = 163840;
constexpr int I4_GROUP = 32, I4_PER_WORD = 8, QK_HEAD = QK_NOPE + QK_ROPE, K_DIM = HEADS * QK_HEAD, V_DIM = HEADS * V_HEAD;
constexpr int HEAD_TILE = 256, EXPERT_TILE = 64, HIDDEN_TILE = 64, KV_TILE = 64;
constexpr float RMS_EPS = 1.0e-5f, ROPE_THETA = 50000.0f, ROUTED_SCALE = 2.827f;
constexpr std::uint32_t EOS_IM_END = 163586;

std::uint64_t u64(const char *s, const char *n);

constexpr std::uint64_t MAX_IO_BYTES = 4ull * 1024 * 1024;

struct KvCache {
  int lock_fd = -1;
  std::uint64_t k_layer_bytes = 0, v_layer_bytes = 0, ugds_base = 0;
  Buf kt, vt, score, hmax, hsum;
  Io &io;
  KvCache(Io &io_ref, std::uint64_t seq, int tile) : k_layer_bytes(seq * K_DIM * sizeof(float)), v_layer_bytes(seq * V_DIM * sizeof(float)), io(io_ref) {
    const char *base = std::getenv("BLOK_KV_UGDS_BASE");
    if (!base) die("BLOK_KV_UGDS_BASE is required for uGDS-backed KV cache");
    ugds_base = u64(base, "BLOK_KV_UGDS_BASE");
    std::string lock_path = std::string("/tmp/blok-kv-") + std::to_string(ugds_base) + ".lock";
    lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) die("KV uGDS region is already reserved: " + lock_path);
    if (const char *cap = std::getenv("BLOK_KV_UGDS_BYTES")) {
      if (LAYERS * (k_layer_bytes + v_layer_bytes) > u64(cap, "BLOK_KV_UGDS_BYTES")) die("BLOK_KV_UGDS_BYTES is too small for requested KV schedule");
    }
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

__global__ void bf16_to_f32_k(const __nv_bfloat16 *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = bf(x + i);
}

__global__ void add_k(float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void silu_mul_k(const float *a, const float *b, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * (1.0f / (1.0f + expf(-a[i]))) * b[i];
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
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * scale * bf(w + i);
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
  if (threadIdx.x == 0 && r < rows) y[r] = s[0];
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
        v += x[base + p * I4_PER_WORD + lane] * ((float)q * sc);
      }
    }
  }
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0 && r < rows) y[r] = s[0];
}

__global__ void matvec_i4_f32_scale_k(const std::uint32_t *w, const float *scale, const float *x, float *y, int rows, int cols) {
  __shared__ float s[256];
  int r = blockIdx.x;
  float v = 0;
  int groups = cols / I4_GROUP, packed_cols = cols / I4_PER_WORD;
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    float sc = scale[(unsigned long long)r * groups + g];
    int base = g * I4_GROUP;
    for (int p = 0; p < I4_GROUP / I4_PER_WORD; ++p) {
      std::uint32_t word = w[(unsigned long long)r * packed_cols + g * (I4_GROUP / I4_PER_WORD) + p];
      for (int lane = 0; lane < I4_PER_WORD; ++lane) {
        int q = (int)((word >> (4 * lane)) & 0xf) - 8;
        v += x[base + p * I4_PER_WORD + lane] * ((float)q * sc);
      }
    }
  }
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0 && r < rows) y[r] = s[0];
}

__global__ void rope_k(float *q, float *k, int pos, int heads, int stride, int rotary) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = heads * rotary / 2;
  if (i >= pairs) return;
  int h = i / (rotary / 2), p = i % (rotary / 2), base = h * stride + p * 2;
  float freq = powf(ROPE_THETA, -2.0f * p / rotary), c = cosf(pos * freq), s = sinf(pos * freq);
  float q0 = q[base], q1 = q[base + 1], k0 = k[base], k1 = k[base + 1];
  q[base] = q0 * c - q1 * s;
  q[base + 1] = q0 * s + q1 * c;
  k[base] = k0 * c - k1 * s;
  k[base + 1] = k0 * s + k1 * c;
}

__global__ void k_rope_fill_k(float *k, const float *rope) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HEADS * QK_ROPE) k[(i / QK_ROPE) * (QK_NOPE + QK_ROPE) + QK_NOPE + i % QK_ROPE] = rope[i % QK_ROPE];
}

__global__ void add_bf16_k(float *x, const __nv_bfloat16 *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += bf(b + i);
}

__global__ void add_f32_k(float *x, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += b[i];
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
    float best = -1;
    for (int i = 0; i < EXPERTS; ++i) if (s[i] > best) best = s[i], id = i;
    expert[k] = (std::uint16_t)id; weight[k] = score[id]; sum += score[id]; s[id] = -1;
  }
  for (int k = 0; k < TOPK; ++k) weight[k] = weight[k] / sum * ROUTED_SCALE;
}

__global__ void expert_accum_k(const float *y, const float *w, int slot, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += y[i] * w[slot];
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
    score[h * n + t] = s[0] * rsqrtf((float)QK_HEAD);
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

__global__ void attn_tile_accum_k(const float *score, const float *v, const float *mx, float *sum, float *out, int n) {
  int h = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= V_HEAD) return;
  float acc = 0, s = 0, m = mx[h];
  for (int t = 0; t < n; ++t) {
    float p = expf(score[h * n + t] - m);
    acc += p * v[(unsigned long long)t * V_DIM + h * V_HEAD + d];
    if (d == 0) s += p;
  }
  out[h * V_HEAD + d] += acc;
  if (d == 0) atomicAdd(sum + h, s);
}

__global__ void attn_finish_k(float *out, const float *sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < V_DIM) out[i] /= sum[i / V_HEAD];
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

std::string hex(const std::string &s) {
  std::string o;
  for (std::size_t i = 0; i + 1 < s.size(); i += 2) o.push_back((char)std::strtoul(s.substr(i, 2).c_str(), nullptr, 16));
  return o;
}

Args args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) die("usage: blok-kimi-exec --manifest m --prompt p --tokens n --router-top-k 8");
    std::string f = argv[i];
    if (f == "--manifest") a.manifest = argv[i + 1];
    else if (f == "--prompt") a.prompt = argv[i + 1];
    else if (f == "--tokens") a.tokens = u64(argv[i + 1], "tokens");
    else if (f == "--router-top-k") a.topk = u64(argv[i + 1], "router-top-k");
    else die("unknown flag: " + f);
  }
  if (a.manifest.empty() || a.prompt.empty() || a.tokens == 0 || a.topk != TOPK) die("bad args");
  return a;
}

Tokenizer tokenizer(const std::string &path) {
  std::ifstream in(path);
  if (!in) die("open tokenizer.blok failed: " + path);
  Tokenizer t;
  std::string tag, a, b, c;
  in >> tag;
  if (tag != "blok-tokenizer-v1") die("bad tokenizer.blok");
  while (in >> tag >> a >> b) {
    if (tag == "tok") {
      auto s = hex(b);
      auto id = (std::uint32_t)std::stoul(a);
      t.ids[s] = id; t.text[id] = s;
    } else if (tag == "merge" && in >> c) {
      t.merges[hex(b) + hex(c)] = std::stoi(a);
    }
  }
  if (t.ids.empty()) die("empty tokenizer");
  return t;
}

std::vector<std::uint32_t> encode(const Tokenizer &t, const std::string &s) {
  std::vector<std::string> pieces;
  for (unsigned char c : s) pieces.emplace_back(1, (char)c);
  for (;;) {
    int at = -1, rank = 1 << 30;
    for (int i = 0; i + 1 < (int)pieces.size(); ++i) {
      auto m = t.merges.find(pieces[i] + pieces[i + 1]);
      if (m != t.merges.end() && m->second < rank) rank = m->second, at = i;
    }
    if (at < 0) break;
    pieces[at] += pieces[at + 1];
    pieces.erase(pieces.begin() + at + 1);
  }
  std::vector<std::uint32_t> out;
  for (auto &p : pieces) {
    auto it = t.ids.find(p);
    if (it == t.ids.end()) die("prompt contains token missing from tokenizer vocab");
    out.push_back(it->second);
  }
  return out;
}

std::string decode(const Tokenizer &t, const std::vector<std::uint32_t> &ids) {
  std::string out;
  for (auto id : ids) {
    auto it = t.text.find(id);
    if (it == t.text.end()) die("sampled token missing from tokenizer vocab");
    out += it->second;
  }
  return out;
}

RuntimeIndex runtime_index(const std::string &manifest) {
  std::filesystem::path path = std::filesystem::path(manifest).parent_path() / "runtime-index.blok";
  std::ifstream in(path);
  if (!in) die("missing runtime-index.blok; run scripts/model_fetch.py kimi-k2.6 materialize");
  RuntimeIndex out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("tokenizer_blok ")) {
      out.tokenizer_blok = line.substr(15);
      continue;
    }
    if (!line.starts_with("tensor ")) continue;
    std::istringstream ss(line);
    std::string tag, layer, expert;
    Tensor t;
    ss >> tag >> t.name >> t.role >> layer >> expert >> t.slot >> t.dtype >> t.shape >> t.off >>
        t.bytes >> t.align >> t.data >> t.data_bytes >> t.file;
    t.layer = i32(layer, "layer");
    t.expert = i32(expert, "expert");
    if (!t.file.empty()) out.tensors.push_back(t);
  }
  if (out.tensors.empty()) die("runtime index has no file-backed tensors");
  if (out.tokenizer_blok.empty() || !std::filesystem::is_regular_file(out.tokenizer_blok)) die("tokenizer.blok missing from runtime index");
  return out;
}

const Tensor *find(const std::vector<Tensor> &ts, const std::string &role, const std::string &part = "") {
  for (const auto &t : ts)
    if (t.role == role && (part.empty() || t.name.find(part) != std::string::npos)) return &t;
  return nullptr;
}

bool has_layer_slot(const std::vector<Tensor> &ts, int layer, const std::string &slot) {
  for (const auto &t : ts)
    if (t.layer == layer && t.slot.find(slot) != std::string::npos) return true;
  return false;
}

const Tensor &need(const std::vector<Tensor> &ts, const std::string &part) {
  auto *t = find(ts, "resident", part);
  if (!t) die("missing tensor: " + part);
  return *t;
}

const Tensor &need(const std::vector<Tensor> &ts, int layer, const std::string &slot, int expert = -1) {
  for (const auto &t : ts)
    if (t.layer == layer && t.slot.find(slot) != std::string::npos && t.name.ends_with(".weight") &&
        (expert < 0 ? t.expert < 0 : t.expert == expert))
      return t;
  die("missing layer tensor: " + std::to_string(layer) + " " + slot);
}

const Tensor &need_role(const std::vector<Tensor> &ts, int layer, const std::string &role, const std::string &slot) {
  for (const auto &t : ts)
    if (t.layer == layer && t.role == role && t.expert < 0 && t.slot.find(slot) != std::string::npos) return t;
  die("missing layer tensor: " + std::to_string(layer) + " " + role + " " + slot);
}

const Tensor &need_routed_weight(const std::vector<Tensor> &ts, int layer, const std::string &proj, int expert) {
  std::string suffix = ".mlp.experts." + std::to_string(expert) + "." + proj + ".weight";
  for (const auto &t : ts)
    if (t.layer == layer && t.role == "routed_expert" && t.expert == expert && t.name.ends_with(suffix)) return t;
  die("missing routed expert weight: " + std::to_string(layer) + " " + proj + " expert " + std::to_string(expert));
}

const Tensor &need_routed_scale(const std::vector<Tensor> &ts, int layer, const std::string &proj, int expert) {
  std::string suffix = ".mlp.experts." + std::to_string(expert) + "." + proj + ".weight_scale";
  for (const auto &t : ts)
    if (t.layer == layer && t.role == "routed_expert" && t.expert == expert && t.name.ends_with(suffix)) return t;
  die("missing routed expert scale: " + std::to_string(layer) + " " + proj + " expert " + std::to_string(expert));
}

void validate_i4_pair(const Tensor &w, const Tensor &scale, int rows, int cols) {
  if (w.dtype != "i32") die("routed expert weight must be packed i32: " + w.name);
  if (scale.dtype != "bf16" && scale.dtype != "f32") die("routed expert scale must be bf16 or f32: " + scale.name);
  if (shape_dim(w, 0) != rows || shape_dim(w, 1) != cols / I4_PER_WORD) die("bad routed expert packed shape: " + w.name);
  if (shape_dim(scale, 0) != rows || shape_dim(scale, 1) != cols / I4_GROUP) die("bad routed expert scale shape: " + scale.name);
}

DeviceTensor load_tensor(Io &io, const Tensor &t) {
  if (t.dtype != "bf16" && t.dtype != "f32" && t.dtype != "i32") die("only bf16, f32, and i32 tensors are executable");
  if (t.data < t.off || t.data + t.data_bytes > t.off + t.bytes) die("tensor data range escapes aligned read");
  if (t.bytes > MAX_IO_BYTES) die("refusing oversized tensor read; use a tiled loader for " + t.name);
  DeviceTensor d;
  d.bytes = t.bytes;
  d.skip = t.data - t.off;
  ck(cudaMalloc(&d.base, t.bytes), "cudaMalloc tensor");
  auto it = io.base.find(t.file);
  if (it == io.base.end()) die("missing uGDS map entry: " + t.file);
  Io::ug(uGDSBufRegister(d.base, t.bytes, 0), "uGDSBufRegister");
  ssize_t n = uGDSRead(io.fh, d.base, t.bytes, (off_t)(it->second + t.off), 0);
  Io::ug(uGDSBufDeregister(d.base), "uGDSBufDeregister");
  if (n != (ssize_t)t.bytes) die("short uGDS read");
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
  auto it = io.base.find(t.file);
  if (it == io.base.end()) die("missing uGDS map entry: " + t.file);
  Io::ug(uGDSBufRegister(d.base, bytes, 0), "uGDSBufRegister slice");
  ssize_t n = uGDSRead(io.fh, d.base, bytes, (off_t)(it->second + aligned), 0);
  Io::ug(uGDSBufDeregister(d.base), "uGDSBufDeregister slice");
  if (n != (ssize_t)bytes) die("short uGDS slice read");
}

DeviceTensor load_tensor_slice(Io &io, const Tensor &t, std::uint64_t byte_start, std::uint64_t byte_len) {
  DeviceTensor d;
  load_tensor_slice_into(io, t, byte_start, byte_len, d);
  return d;
}

int dtype_bytes(const Tensor &t) {
  if (t.dtype == "bf16") return sizeof(__nv_bfloat16);
  if (t.dtype == "f32" || t.dtype == "i32") return 4;
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

void matvec_i4_slice(const Tensor &smeta, const DeviceTensor &w, const DeviceTensor &scale, const float *x, float *y, int rows, int cols) {
  if (smeta.dtype == "f32") matvec_i4_f32_scale_k<<<rows, 256>>>(w.u32(), scale.f32(), x, y, rows, cols);
  else matvec_i4_bf16_scale_k<<<rows, 256>>>(w.u32(), scale.bf16(), x, y, rows, cols);
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
    kv.load_tile(layer, true, start, n);
    attn_tile_accum_k<<<dim3(HEADS, (V_HEAD + 255) / 256), 256>>>(kv.score.p, kv.vt.p, kv.hmax.p, kv.hsum.p, out, n);
  }
  attn_finish_k<<<(V_DIM + 255) / 256, 256>>>(out, kv.hsum.p);
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
  const auto &ts = rt.tensors;
  bool attn[LAYERS] = {}, router[LAYERS] = {};
  const char *attn_slots[] = {"q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj", "kv_a_layernorm", "kv_b_proj", "o_proj"};
  for (const auto &t : ts) {
    if (t.slot.empty() || t.dtype.empty() || t.shape.empty()) die("runtime index has incomplete tensor metadata");
    if (t.layer >= LAYERS || t.expert >= EXPERTS) die("runtime index has out-of-range Kimi coordinates");
    if (t.role == "attention_resident" && t.layer >= 0) attn[t.layer] = true;
    if (t.role == "router" && t.layer >= 0) router[t.layer] = true;
  }
  for (int i = 0; i < LAYERS; ++i) {
    if (!attn[i]) die("missing layer attention tensors");
    for (auto slot : attn_slots)
      if (!has_layer_slot(ts, i, slot)) die("missing layer MLA projection tensor");
    if (!has_layer_slot(ts, i, "input_layernorm")) die("missing layer input_layernorm");
    if (!has_layer_slot(ts, i, "post_attention_layernorm")) die("missing layer post_attention_layernorm");
    if (i < FIRST_DENSE) {
      need(ts, i, "gate_proj"); need(ts, i, "up_proj"); need(ts, i, "down_proj");
      continue;
    }
    if (!router[i]) die("missing layer router tensors");
    need_role(ts, i, "router", "gate");
    need_role(ts, i, "router", "e_score_correction_bias");
    need_role(ts, i, "shared_expert_resident", "gate_proj");
    need_role(ts, i, "shared_expert_resident", "up_proj");
    need_role(ts, i, "shared_expert_resident", "down_proj");
    for (int e = 0; e < EXPERTS; ++e) {
      validate_i4_pair(need_routed_weight(ts, i, "gate_proj", e), need_routed_scale(ts, i, "gate_proj", e), MOE, HIDDEN);
      validate_i4_pair(need_routed_weight(ts, i, "up_proj", e), need_routed_scale(ts, i, "up_proj", e), MOE, HIDDEN);
      validate_i4_pair(need_routed_weight(ts, i, "down_proj", e), need_routed_scale(ts, i, "down_proj", e), HIDDEN, MOE);
    }
  }
  if (!find(ts, "resident", "embed_tokens")) die("missing token embedding");
  if (!find(ts, "resident", "norm.weight")) die("missing final norm");
  if (!find(ts, "resident", "lm_head")) die("missing lm_head");
}

std::string json(const std::string &s) {
  std::string o = "\"";
  const char *hex = "0123456789abcdef";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += static_cast<char>(c);
    } else if (c == '\n') {
      o += "\\n";
    } else if (c == '\r') {
      o += "\\r";
    } else if (c == '\t') {
      o += "\\t";
    } else if (c < 0x20) {
      o += "\\u00";
      o += hex[c >> 4];
      o += hex[c & 0xf];
    } else {
      o += static_cast<char>(c);
    }
  }
  return o + "\"";
}

Generation generate(Io &io, const RuntimeIndex &rt, const Args &a) {
  auto tokz = tokenizer(rt.tokenizer_blok);
  auto input = encode(tokz, a.prompt), made = std::vector<std::uint32_t>{};
  if (input.empty()) die("tokenizer produced no prompt tokens");
  const Tensor &emb = need(rt.tensors, "embed_tokens");
  auto final_norm = load_tensor(io, need(rt.tensors, "norm.weight"));
  const Tensor &head = need(rt.tensors, "lm_head");
  std::uint32_t *did = nullptr;
  std::uint16_t *dex = nullptr;
  ck(cudaMalloc(&did, sizeof(std::uint32_t)), "cudaMalloc sample");
  ck(cudaMalloc(&dex, TOPK * sizeof(std::uint16_t)), "cudaMalloc experts");
  Buf x, n, qa, q, kva, kv, k, v, av, router, route_select, ew, gate, up, mid, down, logits;
  x.make(HIDDEN); n.make(HIDDEN); qa.make(Q_RANK); q.make(HEADS * (QK_NOPE + QK_ROPE));
  kva.make(KV_A); kv.make(HEADS * (QK_NOPE + V_HEAD)); k.make(HEADS * (QK_NOPE + QK_ROPE));
  v.make(HEADS * V_HEAD); av.make(HEADS * V_HEAD);
  router.make(EXPERTS); route_select.make(EXPERTS); ew.make(TOPK);
  gate.make(SHARED); up.make(SHARED); mid.make(SHARED); down.make(HIDDEN); logits.make(HEAD_TILE);
  KvCache kv_cache(io, input.size() + a.tokens + 1, KV_TILE);
  auto run = [&](std::uint32_t id, int pos) {
    embed_token(io, emb, id, x.p);
    for (int l = 0; l < LAYERS; ++l) {
      auto ln0 = load_tensor(io, need(rt.tensors, l, "input_layernorm"));
      const Tensor &qaw = need(rt.tensors, l, "q_a_proj");
      auto qaln = load_tensor(io, need(rt.tensors, l, "q_a_layernorm"));
      const Tensor &qbw = need(rt.tensors, l, "q_b_proj");
      const Tensor &kvaw = need(rt.tensors, l, "kv_a_proj");
      auto kvaln = load_tensor(io, need(rt.tensors, l, "kv_a_layernorm"));
      const Tensor &kvbw = need(rt.tensors, l, "kv_b_proj");
      const Tensor &ow = need(rt.tensors, l, "o_proj");
      rmsnorm_k<<<1, 256>>>(x.p, ln0.bf16(), n.p, HIDDEN);
      matvec_bf16_all_rows(io, qaw, n.p, qa.p, Q_RANK, HIDDEN, EXPERT_TILE);
      rmsnorm_k<<<1, 256>>>(qa.p, qaln.bf16(), qa.p, Q_RANK);
      matvec_bf16_all_rows(io, qbw, qa.p, q.p, K_DIM, Q_RANK, EXPERT_TILE);
      matvec_bf16_all_rows(io, kvaw, n.p, kva.p, KV_A, HIDDEN, EXPERT_TILE);
      rmsnorm_k<<<1, 256>>>(kva.p, kvaln.bf16(), kva.p, KV_RANK);
      matvec_bf16_all_rows(io, kvbw, kva.p, kv.p, HEADS * (QK_NOPE + V_HEAD), KV_RANK, EXPERT_TILE);
      ck(cudaMemcpy(k.p, kv.p, HEADS * QK_NOPE * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy k nope");
      k_rope_fill_k<<<(HEADS * QK_ROPE + 255) / 256, 256>>>(k.p, kva.p + KV_RANK);
      ck(cudaMemcpy(v.p, kv.p + HEADS * QK_NOPE, HEADS * V_HEAD * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy v");
      rope_k<<<(HEADS * QK_ROPE / 2 + 255) / 256, 256>>>(q.p, k.p, pos, HEADS, QK_NOPE + QK_ROPE, QK_ROPE);
      kv_cache.store(l, pos, k.p, v.p);
      tiled_attention(kv_cache, l, pos + 1, KV_TILE, q.p, av.p);
      matvec_bf16_all_rows(io, ow, av.p, down.p, HIDDEN, HEADS * V_HEAD, HIDDEN_TILE);
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
      auto ln1 = load_tensor(io, need(rt.tensors, l, "post_attention_layernorm"));
      rmsnorm_k<<<1, 256>>>(x.p, ln1.bf16(), n.p, HIDDEN);
      ck(cudaMemset(down.p, 0, HIDDEN * sizeof(float)), "cudaMemset ffn");
      if (l < FIRST_DENSE) {
        const Tensor &gw = need(rt.tensors, l, "gate_proj"), &uw = need(rt.tensors, l, "up_proj"), &dw = need(rt.tensors, l, "down_proj");
        bf16_mlp(io, gw, uw, dw, n.p, down.p, gate.p, up.p, mid.p, av.p, shape_dim(gw, 0));
      } else {
        const Tensor &rw = need_role(rt.tensors, l, "router", "gate");
        auto rb = load_tensor(io, need_role(rt.tensors, l, "router", "e_score_correction_bias"));
        matvec_bf16_all_rows(io, rw, n.p, router.p, EXPERTS, HIDDEN, EXPERT_TILE);
        sigmoid_k<<<(EXPERTS + 255) / 256, 256>>>(router.p, EXPERTS);
        ck(cudaMemcpy(route_select.p, router.p, EXPERTS * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy router scores");
        if (rb.dtype == "f32") add_f32_k<<<(EXPERTS + 255) / 256, 256>>>(route_select.p, rb.f32(), EXPERTS);
        else add_bf16_k<<<(EXPERTS + 255) / 256, 256>>>(route_select.p, rb.bf16(), EXPERTS);
        router_topk_k<<<1, EXPERTS>>>(router.p, route_select.p, dex, ew.p);
        std::uint16_t ex[TOPK];
        ck(cudaMemcpy(ex, dex, sizeof(ex), cudaMemcpyDeviceToHost), "cudaMemcpy experts");
        for (int i = 0; i < TOPK; ++i) {
          const Tensor &gwm = need_routed_weight(rt.tensors, l, "gate_proj", ex[i]);
          const Tensor &gsm = need_routed_scale(rt.tensors, l, "gate_proj", ex[i]);
          const Tensor &uwm = need_routed_weight(rt.tensors, l, "up_proj", ex[i]);
          const Tensor &usm = need_routed_scale(rt.tensors, l, "up_proj", ex[i]);
          const Tensor &dwm = need_routed_weight(rt.tensors, l, "down_proj", ex[i]);
          const Tensor &dsm = need_routed_scale(rt.tensors, l, "down_proj", ex[i]);
          for (int r = 0; r < MOE; r += EXPERT_TILE) {
            int rows = std::min(EXPERT_TILE, MOE - r);
            auto gw = load_rows(io, gwm, r, rows, HIDDEN / I4_PER_WORD * dtype_bytes(gwm));
            auto gs = load_rows(io, gsm, r, rows, HIDDEN / I4_GROUP * dtype_bytes(gsm));
            auto uw = load_rows(io, uwm, r, rows, HIDDEN / I4_PER_WORD * dtype_bytes(uwm));
            auto us = load_rows(io, usm, r, rows, HIDDEN / I4_GROUP * dtype_bytes(usm));
            matvec_i4_slice(gsm, gw, gs, n.p, gate.p + r, rows, HIDDEN);
            matvec_i4_slice(usm, uw, us, n.p, up.p + r, rows, HIDDEN);
            silu_mul_k<<<(rows + 255) / 256, 256>>>(gate.p + r, up.p + r, mid.p + r, rows);
          }
          for (int r = 0; r < HIDDEN; r += HIDDEN_TILE) {
            int rows = std::min(HIDDEN_TILE, HIDDEN - r);
            auto dw = load_rows(io, dwm, r, rows, MOE / I4_PER_WORD * dtype_bytes(dwm));
            auto ds = load_rows(io, dsm, r, rows, MOE / I4_GROUP * dtype_bytes(dsm));
            matvec_i4_slice(dsm, dw, ds, mid.p, av.p + r, rows, MOE);
            expert_accum_k<<<(rows + 255) / 256, 256>>>(av.p + r, ew.p, i, down.p + r, rows);
          }
        }
        const Tensor &sgw = need_role(rt.tensors, l, "shared_expert_resident", "gate_proj");
        const Tensor &suw = need_role(rt.tensors, l, "shared_expert_resident", "up_proj");
        const Tensor &sdw = need_role(rt.tensors, l, "shared_expert_resident", "down_proj");
        bf16_mlp(io, sgw, suw, sdw, n.p, down.p, gate.p, up.p, mid.p, av.p, shape_dim(sgw, 0));
      }
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
    }
    rmsnorm_k<<<1, 256>>>(x.p, final_norm.bf16(), n.p, HIDDEN);
    std::uint32_t next = sample_lm_head(io, head, n.p, logits.p, did);
    ck(cudaDeviceSynchronize(), "decode sync");
    return next;
  };
  std::uint32_t next = input[0];
  for (std::size_t i = 0; i < input.size(); ++i) next = run(input[i], (int)i);
  for (std::uint64_t i = 0; i < a.tokens && next != EOS_IM_END; ++i) {
    made.push_back(next);
    next = run(next, (int)(input.size() + i));
  }
  cudaFree(did); cudaFree(dex);
  return {decode(tokz, made), (std::uint64_t)made.size()};
}

int main(int argc, char **argv) {
  Args a = args(argc, argv);
  auto rt = runtime_index(a.manifest);
  validate_forward_contract(rt);
  Io io;
  auto gen = generate(io, rt, a);
  std::cout << "{\"protocol\":\"blok-kimi-exec-v1\",\"status\":\"ok\",\"io_mode\":\"ugds\",\"text\":" << json(gen.text)
            << ",\"tokens\":" << gen.tokens << ",\"predicted_tps\":null,\"watts\":null}\n";
}
