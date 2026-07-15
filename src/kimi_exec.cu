#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef BLOK_HAVE_UGDS
#include <ugds.h>
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

struct Io {
  std::unordered_map<std::string, std::uint64_t> base;
#ifdef BLOK_HAVE_UGDS
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
#endif
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
  __nv_bfloat16 *bf16() const { return reinterpret_cast<__nv_bfloat16 *>(static_cast<char *>(base) + skip); }
  float *f32() const { return reinterpret_cast<float *>(static_cast<char *>(base) + skip); }
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
constexpr int Q_RANK = 1536, KV_RANK = 512, KV_A = 576, EXPERTS = 384, TOPK = 8, MOE = 2048, VOCAB = 163840;
constexpr float RMS_EPS = 1.0e-5f, ROPE_THETA = 50000.0f, ROUTED_SCALE = 2.827f;
constexpr std::uint32_t EOS_IM_END = 163586;

__device__ float bf(const __nv_bfloat16 *p) { return __bfloat162float(*p); }

__global__ void embed_k(const __nv_bfloat16 *e, const std::uint32_t *tok, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HIDDEN) x[i] = bf(e + (unsigned long long)*tok * HIDDEN + i);
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

__global__ void kv_store_k(const float *k, const float *v, float *kc, float *vc, int pos, int kd, int vd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kd) kc[(unsigned long long)pos * kd + i] = k[i];
  if (i < vd) vc[(unsigned long long)pos * vd + i] = v[i];
}

__global__ void k_rope_fill_k(float *k, const float *rope) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HEADS * QK_ROPE) k[(i / QK_ROPE) * (QK_NOPE + QK_ROPE) + QK_NOPE + i % QK_ROPE] = rope[i % QK_ROPE];
}

__global__ void attn_score_k(const float *q, const float *kc, float *score, int seq, int dim) {
  __shared__ float s[256];
  int t = blockIdx.x;
  float v = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) v += q[i] * kc[(unsigned long long)t * dim + i];
  s[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  if (!threadIdx.x && t < seq) score[t] = s[0] * rsqrtf((float)dim);
}

__global__ void softmax_k(float *x, int n) {
  __shared__ float s[256];
  float m = -CUDART_INF_F;
  for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
  s[threadIdx.x] = m;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + d]);
    __syncthreads();
  }
  float sum = 0, mx = s[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(x[i] - mx);
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x + d];
    __syncthreads();
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = expf(x[i] - mx) / s[0];
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

__global__ void router_topk_k(const float *score, std::uint16_t *expert, float *weight) {
  __shared__ float s[384];
  int tid = threadIdx.x;
  if (tid >= EXPERTS) return;
  s[tid] = score[tid];
  __syncthreads();
  if (tid) return;
  float sum = 0;
  for (int k = 0; k < TOPK; ++k) {
    int id = 0;
    float best = -1;
    for (int i = 0; i < EXPERTS; ++i) if (s[i] > best) best = s[i], id = i;
    expert[k] = (std::uint16_t)id; weight[k] = best; sum += best; s[id] = -1;
  }
  for (int k = 0; k < TOPK; ++k) weight[k] = weight[k] / sum * ROUTED_SCALE;
}

__global__ void expert_accum_k(const float *y, const float *w, int slot, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += y[i] * w[slot];
}

__global__ void attn_value_k(const float *p, const float *v, float *out, int seq, int dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  float x = 0;
  for (int t = 0; t < seq; ++t) x += p[t] * v[(unsigned long long)t * dim + d];
  out[d] = x;
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

int max_mlp_width(const std::vector<Tensor> &ts) {
  int width = MOE;
  for (const auto &t : ts)
    if (t.slot.find("gate_proj") != std::string::npos || t.slot.find("up_proj") != std::string::npos)
      width = std::max(width, shape_dim(t, 0));
  return width;
}

std::string hex(const std::string &s) {
  std::string o;
  for (std::size_t i = 0; i + 1 < s.size(); i += 2) o.push_back((char)std::strtoul(s.substr(i, 2).c_str(), nullptr, 16));
  return o;
}

Args args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) die("usage: blok-kimi-exec --manifest m --prompt p --tokens n --top-k 8");
    std::string f = argv[i];
    if (f == "--manifest") a.manifest = argv[i + 1];
    else if (f == "--prompt") a.prompt = argv[i + 1];
    else if (f == "--tokens") a.tokens = u64(argv[i + 1], "tokens");
    else if (f == "--top-k") a.topk = u64(argv[i + 1], "top-k");
    else die("unknown flag: " + f);
  }
  if (a.manifest.empty() || a.prompt.empty() || a.tokens == 0 || a.topk != 8) die("bad args");
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

void read_direct(const Tensor &t, void *dst) {
  int fd = open(t.file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) die(std::string("O_DIRECT open failed: ") + std::strerror(errno));
  auto *p = static_cast<std::uint8_t *>(dst);
  std::uint64_t done = 0;
  while (done < t.bytes) {
    ssize_t n = pread(fd, p + done, t.bytes - done, t.off + done);
    if (n <= 0) die(std::string("O_DIRECT read failed: ") + std::strerror(errno));
    done += static_cast<std::uint64_t>(n);
  }
  close(fd);
}

const Tensor *find(const std::vector<Tensor> &ts, const std::string &role, const std::string &part = "") {
  for (const auto &t : ts)
    if (t.role == role && (part.empty() || t.name.find(part) != std::string::npos)) return &t;
  return nullptr;
}

std::uint64_t count(const std::vector<Tensor> &ts, const std::string &role) {
  std::uint64_t n = 0;
  for (const auto &t : ts)
    if (t.role == role) ++n;
  return n;
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
    if (t.layer == layer && t.slot.find(slot) != std::string::npos && (expert < 0 ? t.expert < 0 : t.expert == expert)) return t;
  die("missing layer tensor: " + std::to_string(layer) + " " + slot);
}

const Tensor *maybe(const std::vector<Tensor> &ts, int layer, const std::string &slot, int expert = -1) {
  for (const auto &t : ts)
    if (t.layer == layer && t.slot.find(slot) != std::string::npos && (expert < 0 ? t.expert < 0 : t.expert == expert)) return &t;
  return nullptr;
}

const Tensor &need_role(const std::vector<Tensor> &ts, int layer, const std::string &role, const std::string &slot) {
  for (const auto &t : ts)
    if (t.layer == layer && t.role == role && t.expert < 0 && t.slot.find(slot) != std::string::npos) return t;
  die("missing layer tensor: " + std::to_string(layer) + " " + role + " " + slot);
}

DeviceTensor load_tensor(Io &io, const Tensor &t) {
  if (t.dtype != "bf16" && t.dtype != "f32") die("only bf16 and f32 tensors are executable");
  if (t.data < t.off || t.data + t.data_bytes > t.off + t.bytes) die("tensor data range escapes aligned read");
  DeviceTensor d;
  d.bytes = t.bytes;
  d.skip = t.data - t.off;
  ck(cudaMalloc(&d.base, t.bytes), "cudaMalloc tensor");
#ifdef BLOK_HAVE_UGDS
  auto it = io.base.find(t.file);
  if (it == io.base.end()) die("missing uGDS map entry: " + t.file);
  Io::ug(uGDSBufRegister(d.base, t.bytes, 0), "uGDSBufRegister");
  ssize_t n = uGDSRead(io.fh, d.base, t.bytes, (off_t)(it->second + t.off), 0);
  Io::ug(uGDSBufDeregister(d.base), "uGDSBufDeregister");
  if (n != (ssize_t)t.bytes) die("short uGDS read");
#else
  (void)io;
  void *host = nullptr;
  if (posix_memalign(&host, t.align, t.bytes)) die("posix_memalign failed");
  read_direct(t, host);
  ck(cudaMemcpy(d.base, host, t.bytes, cudaMemcpyHostToDevice), "cudaMemcpy tensor");
  free(host);
#endif
  return d;
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
      if (!maybe(ts, i, "gate_proj") || !maybe(ts, i, "up_proj") || !maybe(ts, i, "down_proj"))
        die("missing dense first-layer MLP tensors");
      continue;
    }
    if (!router[i]) die("missing layer router tensors");
    need_role(ts, i, "router", "gate");
    need_role(ts, i, "router", "e_score_correction_bias");
    need_role(ts, i, "shared_expert_resident", "gate_proj");
    need_role(ts, i, "shared_expert_resident", "up_proj");
    need_role(ts, i, "shared_expert_resident", "down_proj");
  }
  if (count(ts, "attention_resident") == 0) die("missing attention tensors");
  if (count(ts, "router") == 0) die("missing router tensors");
  if (count(ts, "routed_expert") == 0) die("missing routed expert tensors");
  if (count(ts, "shared_expert_resident") == 0) die("missing shared expert tensors");
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

std::string generate(Io &io, const RuntimeIndex &rt, const Args &a) {
  auto tokz = tokenizer(rt.tokenizer_blok);
  auto input = encode(tokz, a.prompt), made = std::vector<std::uint32_t>{};
  if (input.empty()) die("tokenizer produced no prompt tokens");
  int mlp_width = max_mlp_width(rt.tensors);
  auto emb = load_tensor(io, need(rt.tensors, "embed_tokens"));
  auto final_norm = load_tensor(io, need(rt.tensors, "norm.weight"));
  auto head = load_tensor(io, need(rt.tensors, "lm_head"));
  std::uint32_t *dtok = nullptr, *did = nullptr;
  std::uint16_t *dex = nullptr;
  ck(cudaMalloc(&dtok, sizeof(std::uint32_t)), "cudaMalloc token");
  ck(cudaMalloc(&did, sizeof(std::uint32_t)), "cudaMalloc sample");
  ck(cudaMalloc(&dex, TOPK * sizeof(std::uint16_t)), "cudaMalloc experts");
  Buf x, n, qa, q, kva, kv, k, v, av, score, router, ew, gate, up, mid, down, logits;
  x.make(HIDDEN); n.make(HIDDEN); qa.make(Q_RANK); q.make(HEADS * (QK_NOPE + QK_ROPE));
  kva.make(KV_A); kv.make(HEADS * (QK_NOPE + V_HEAD)); k.make(HEADS * (QK_NOPE + QK_ROPE));
  v.make(HEADS * V_HEAD); av.make(HEADS * V_HEAD); score.make(input.size() + a.tokens + 1);
  router.make(EXPERTS); ew.make(TOPK); gate.make(mlp_width); up.make(mlp_width); mid.make(mlp_width); down.make(HIDDEN); logits.make(VOCAB);
  std::vector<Buf> kc(LAYERS), vc(LAYERS);
  for (int l = 0; l < LAYERS; ++l) {
    kc[l].make((input.size() + a.tokens + 1) * HEADS * (QK_NOPE + QK_ROPE));
    vc[l].make((input.size() + a.tokens + 1) * HEADS * V_HEAD);
  }
  auto run = [&](std::uint32_t id, int pos) {
    ck(cudaMemcpy(dtok, &id, sizeof(id), cudaMemcpyHostToDevice), "cudaMemcpy token");
    embed_k<<<(HIDDEN + 255) / 256, 256>>>(emb.bf16(), dtok, x.p);
    for (int l = 0; l < LAYERS; ++l) {
      auto ln0 = load_tensor(io, need(rt.tensors, l, "input_layernorm"));
      auto qaw = load_tensor(io, need(rt.tensors, l, "q_a_proj"));
      auto qaln = load_tensor(io, need(rt.tensors, l, "q_a_layernorm"));
      auto qbw = load_tensor(io, need(rt.tensors, l, "q_b_proj"));
      auto kvaw = load_tensor(io, need(rt.tensors, l, "kv_a_proj"));
      auto kvaln = load_tensor(io, need(rt.tensors, l, "kv_a_layernorm"));
      auto kvbw = load_tensor(io, need(rt.tensors, l, "kv_b_proj"));
      auto ow = load_tensor(io, need(rt.tensors, l, "o_proj"));
      rmsnorm_k<<<1, 256>>>(x.p, ln0.bf16(), n.p, HIDDEN);
      matvec_bf16_k<<<Q_RANK, 256>>>(qaw.bf16(), n.p, qa.p, Q_RANK, HIDDEN);
      rmsnorm_k<<<1, 256>>>(qa.p, qaln.bf16(), qa.p, Q_RANK);
      matvec_bf16_k<<<HEADS * (QK_NOPE + QK_ROPE), 256>>>(qbw.bf16(), qa.p, q.p, HEADS * (QK_NOPE + QK_ROPE), Q_RANK);
      matvec_bf16_k<<<KV_A, 256>>>(kvaw.bf16(), n.p, kva.p, KV_A, HIDDEN);
      rmsnorm_k<<<1, 256>>>(kva.p, kvaln.bf16(), kva.p, KV_RANK);
      matvec_bf16_k<<<HEADS * (QK_NOPE + V_HEAD), 256>>>(kvbw.bf16(), kva.p, kv.p, HEADS * (QK_NOPE + V_HEAD), KV_RANK);
      ck(cudaMemcpy(k.p, kv.p, HEADS * QK_NOPE * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy k nope");
      k_rope_fill_k<<<(HEADS * QK_ROPE + 255) / 256, 256>>>(k.p, kva.p + KV_RANK);
      ck(cudaMemcpy(v.p, kv.p + HEADS * QK_NOPE, HEADS * V_HEAD * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy v");
      rope_k<<<(HEADS * QK_ROPE / 2 + 255) / 256, 256>>>(q.p, k.p, pos, HEADS, QK_NOPE + QK_ROPE, QK_ROPE);
      kv_store_k<<<(HEADS * (QK_NOPE + QK_ROPE) + 255) / 256, 256>>>(
          k.p, v.p, kc[l].p, vc[l].p, pos, HEADS * (QK_NOPE + QK_ROPE), HEADS * V_HEAD);
      attn_score_k<<<pos + 1, 256>>>(q.p, kc[l].p, score.p, pos + 1, HEADS * (QK_NOPE + QK_ROPE));
      softmax_k<<<1, 256>>>(score.p, pos + 1);
      attn_value_k<<<(HEADS * V_HEAD + 255) / 256, 256>>>(score.p, vc[l].p, av.p, pos + 1, HEADS * V_HEAD);
      matvec_bf16_k<<<HIDDEN, 256>>>(ow.bf16(), av.p, down.p, HIDDEN, HEADS * V_HEAD);
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
      auto ln1 = load_tensor(io, need(rt.tensors, l, "post_attention_layernorm"));
      rmsnorm_k<<<1, 256>>>(x.p, ln1.bf16(), n.p, HIDDEN);
      ck(cudaMemset(down.p, 0, HIDDEN * sizeof(float)), "cudaMemset ffn");
      if (l < FIRST_DENSE) {
        auto gw = load_tensor(io, need(rt.tensors, l, "gate_proj"));
        auto uw = load_tensor(io, need(rt.tensors, l, "up_proj"));
        auto dw = load_tensor(io, need(rt.tensors, l, "down_proj"));
        int ffn = shape_dim(need(rt.tensors, l, "gate_proj"), 0);
        matvec_bf16_k<<<ffn, 256>>>(gw.bf16(), n.p, gate.p, ffn, HIDDEN);
        matvec_bf16_k<<<ffn, 256>>>(uw.bf16(), n.p, up.p, ffn, HIDDEN);
        silu_mul_k<<<(ffn + 255) / 256, 256>>>(gate.p, up.p, mid.p, ffn);
        matvec_bf16_k<<<HIDDEN, 256>>>(dw.bf16(), mid.p, down.p, HIDDEN, ffn);
      } else {
        auto rw = load_tensor(io, need_role(rt.tensors, l, "router", "gate"));
        auto rb = load_tensor(io, need_role(rt.tensors, l, "router", "e_score_correction_bias"));
        matvec_bf16_k<<<EXPERTS, 256>>>(rw.bf16(), n.p, router.p, EXPERTS, HIDDEN);
        sigmoid_k<<<(EXPERTS + 255) / 256, 256>>>(router.p, EXPERTS);
        if (rb.dtype == "f32") add_f32_k<<<(EXPERTS + 255) / 256, 256>>>(router.p, rb.f32(), EXPERTS);
        else add_bf16_k<<<(EXPERTS + 255) / 256, 256>>>(router.p, rb.bf16(), EXPERTS);
        router_topk_k<<<1, EXPERTS>>>(router.p, dex, ew.p);
        std::uint16_t ex[TOPK];
        ck(cudaMemcpy(ex, dex, sizeof(ex), cudaMemcpyDeviceToHost), "cudaMemcpy experts");
        for (int i = 0; i < TOPK; ++i) {
          auto gw = load_tensor(io, need(rt.tensors, l, "gate_proj", ex[i]));
          auto uw = load_tensor(io, need(rt.tensors, l, "up_proj", ex[i]));
          auto dw = load_tensor(io, need(rt.tensors, l, "down_proj", ex[i]));
          int ffn = shape_dim(need(rt.tensors, l, "gate_proj", ex[i]), 0);
          matvec_bf16_k<<<ffn, 256>>>(gw.bf16(), n.p, gate.p, ffn, HIDDEN);
          matvec_bf16_k<<<ffn, 256>>>(uw.bf16(), n.p, up.p, ffn, HIDDEN);
          silu_mul_k<<<(ffn + 255) / 256, 256>>>(gate.p, up.p, mid.p, ffn);
          matvec_bf16_k<<<HIDDEN, 256>>>(dw.bf16(), mid.p, av.p, HIDDEN, ffn);
          expert_accum_k<<<(HIDDEN + 255) / 256, 256>>>(av.p, ew.p, i, down.p, HIDDEN);
        }
        auto sgw = load_tensor(io, need_role(rt.tensors, l, "shared_expert_resident", "gate_proj"));
        auto suw = load_tensor(io, need_role(rt.tensors, l, "shared_expert_resident", "up_proj"));
        auto sdw = load_tensor(io, need_role(rt.tensors, l, "shared_expert_resident", "down_proj"));
        int ffn = shape_dim(need_role(rt.tensors, l, "shared_expert_resident", "gate_proj"), 0);
        matvec_bf16_k<<<ffn, 256>>>(sgw.bf16(), n.p, gate.p, ffn, HIDDEN);
        matvec_bf16_k<<<ffn, 256>>>(suw.bf16(), n.p, up.p, ffn, HIDDEN);
        silu_mul_k<<<(ffn + 255) / 256, 256>>>(gate.p, up.p, mid.p, ffn);
        matvec_bf16_k<<<HIDDEN, 256>>>(sdw.bf16(), mid.p, av.p, HIDDEN, ffn);
        add_k<<<(HIDDEN + 255) / 256, 256>>>(down.p, av.p, HIDDEN);
      }
      add_k<<<(HIDDEN + 255) / 256, 256>>>(x.p, down.p, HIDDEN);
    }
    rmsnorm_k<<<1, 256>>>(x.p, final_norm.bf16(), n.p, HIDDEN);
    matvec_bf16_k<<<VOCAB, 256>>>(head.bf16(), n.p, logits.p, VOCAB, HIDDEN);
    argmax_k<<<1, 256>>>(logits.p, did, VOCAB);
    std::uint32_t next = 0;
    ck(cudaMemcpy(&next, did, sizeof(next), cudaMemcpyDeviceToHost), "cudaMemcpy sample");
    ck(cudaDeviceSynchronize(), "decode sync");
    return next;
  };
  std::uint32_t next = input[0];
  for (std::size_t i = 0; i < input.size(); ++i) next = run(input[i], (int)i);
  for (std::uint64_t i = 0; i < a.tokens && next != EOS_IM_END; ++i) {
    made.push_back(next);
    next = run(next, (int)(input.size() + i));
  }
  cudaFree(dtok); cudaFree(did); cudaFree(dex);
  return decode(tokz, made);
}

int main(int argc, char **argv) {
  Args a = args(argc, argv);
  auto rt = runtime_index(a.manifest);
  validate_forward_contract(rt);
  Io io;
  auto text = generate(io, rt, a);
  std::cout << "{\"status\":\"ok\",\"text\":" << json(text) << ",\"tokens\":" << a.tokens << ",\"predicted_tps\":0,\"watts\":null}\n";
}
