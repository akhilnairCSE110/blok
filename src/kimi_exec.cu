#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef BLOK_HAVE_UGDS
#include <ugds.h>
#endif

struct Tensor {
  std::string name, role, slot, dtype, shape, file;
  std::uint64_t off = 0, bytes = 0, align = 4096;
  int layer = -1, expert = -1;
};

struct Args {
  std::string manifest, prompt;
  std::uint64_t tokens = 0, topk = 8;
};

struct RuntimeIndex {
  std::string tokenizer;
  std::vector<Tensor> tensors;
};

constexpr int HIDDEN = 7168, HEADS = 64, QK_NOPE = 128, QK_ROPE = 64, V_HEAD = 128;
constexpr int Q_RANK = 1536, KV_RANK = 512, EXPERTS = 384, TOPK = 8, MOE = 2048, VOCAB = 163840;

__device__ float bf(const __nv_bfloat16 *p) { return __bfloat162float(*p); }

__global__ void embed_k(const __nv_bfloat16 *e, const std::uint32_t *tok, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HIDDEN) x[i] = bf(e + (unsigned long long)*tok * HIDDEN + i);
}

__global__ void add_k(float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void scale_add_k(float *a, const float *b, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i] * s;
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
  float scale = rsqrtf(s[0] / n + 1.0e-6f);
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
  float freq = powf(1000000.0f, -2.0f * p / rotary), c = cosf(pos * freq), s = sinf(pos * freq);
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

__global__ void router_topk_k(const float *logits, std::uint16_t *expert, float *weight) {
  __shared__ float s[384];
  int tid = threadIdx.x;
  s[tid] = 1.0f / (1.0f + expf(-logits[tid]));
  __syncthreads();
  if (tid) return;
  float sum = 0;
  for (int k = 0; k < 8; ++k) {
    int id = 0;
    float best = -1;
    for (int i = 0; i < 384; ++i) if (s[i] > best) best = s[i], id = i;
    expert[k] = (std::uint16_t)id; weight[k] = best; sum += best; s[id] = -1;
  }
  for (int k = 0; k < 8; ++k) weight[k] = weight[k] / sum * 2.827f;
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

RuntimeIndex runtime_index(const std::string &manifest) {
  std::filesystem::path path = std::filesystem::path(manifest).parent_path() / "runtime-index.blok";
  std::ifstream in(path);
  if (!in) die("missing runtime-index.blok; run scripts/model_fetch.py kimi-k2.6 materialize");
  RuntimeIndex out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.starts_with("tokenizer ")) {
      out.tokenizer = line.substr(10);
      continue;
    }
    if (!line.starts_with("tensor ")) continue;
    std::istringstream ss(line);
    std::string tag, layer, expert;
    Tensor t;
    ss >> tag >> t.name >> t.role >> layer >> expert >> t.slot >> t.dtype >> t.shape >> t.off >> t.bytes >> t.align >> t.file;
    t.layer = i32(layer, "layer");
    t.expert = i32(expert, "expert");
    if (!t.file.empty()) out.tensors.push_back(t);
  }
  if (out.tensors.empty()) die("runtime index has no file-backed tensors");
  if (out.tokenizer.empty() || !std::filesystem::is_regular_file(out.tokenizer)) die("tokenizer.json missing from runtime index");
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
    if (t.layer == layer && t.slot == slot) return true;
  return false;
}

void stage_tensor(const Tensor &t) {
  void *host = nullptr, *dev = nullptr;
  if (posix_memalign(&host, t.align, t.bytes)) die("posix_memalign failed");
  read_direct(t, host);
  ck(cudaMalloc(&dev, t.bytes), "cudaMalloc");
  ck(cudaMemcpy(dev, host, t.bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  ck(cudaFree(dev), "cudaFree");
  free(host);
}

void validate_forward_contract(const RuntimeIndex &rt) {
  const auto &ts = rt.tensors;
  bool attn[61] = {}, router[61] = {};
  const char *attn_slots[] = {"q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj"};
  for (const auto &t : ts) {
    if (t.slot.empty() || t.dtype.empty() || t.shape.empty()) die("runtime index has incomplete tensor metadata");
    if (t.layer >= 61 || t.expert >= 384) die("runtime index has out-of-range Kimi coordinates");
    if (t.role == "attention_resident" && t.layer >= 0) attn[t.layer] = true;
    if (t.role == "router" && t.layer >= 0) router[t.layer] = true;
  }
  for (int i = 0; i < 61; ++i) {
    if (!attn[i]) die("missing layer attention tensors");
    if (!router[i]) die("missing layer router tensors");
    for (auto slot : attn_slots)
      if (!has_layer_slot(ts, i, slot)) die("missing layer MLA projection tensor");
  }
  if (count(ts, "attention_resident") == 0) die("missing attention tensors");
  if (count(ts, "router") == 0) die("missing router tensors");
  if (count(ts, "routed_expert") == 0) die("missing routed expert tensors");
  if (!find(ts, "resident", "embed_tokens")) die("missing token embedding");
  if (!find(ts, "resident", "norm.weight")) die("missing final norm");
  if (!find(ts, "resident", "lm_head")) die("missing lm_head");
}

void decode_kernel_skeleton() {
  float *x = nullptr, *n = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *av = nullptr, *score = nullptr;
  float *gate = nullptr, *up = nullptr, *mid = nullptr, *down = nullptr, *logits = nullptr;
  __nv_bfloat16 *w = nullptr;
  std::uint32_t *tok = nullptr, *id = nullptr;
  std::uint16_t *experts = nullptr;
  float *ew = nullptr;
  ck(cudaMalloc(&x, HIDDEN * sizeof(float)), "cudaMalloc x");
  ck(cudaMalloc(&n, HIDDEN * sizeof(float)), "cudaMalloc norm");
  ck(cudaMalloc(&q, HEADS * (QK_NOPE + QK_ROPE) * sizeof(float)), "cudaMalloc q");
  ck(cudaMalloc(&k, HEADS * (QK_NOPE + QK_ROPE) * sizeof(float)), "cudaMalloc k");
  ck(cudaMalloc(&v, HEADS * V_HEAD * sizeof(float)), "cudaMalloc v");
  ck(cudaMalloc(&av, HEADS * V_HEAD * sizeof(float)), "cudaMalloc av");
  ck(cudaMalloc(&score, 16 * sizeof(float)), "cudaMalloc score");
  ck(cudaMalloc(&gate, MOE * sizeof(float)), "cudaMalloc gate");
  ck(cudaMalloc(&up, MOE * sizeof(float)), "cudaMalloc up");
  ck(cudaMalloc(&mid, MOE * sizeof(float)), "cudaMalloc mid");
  ck(cudaMalloc(&down, HIDDEN * sizeof(float)), "cudaMalloc down");
  ck(cudaMalloc(&logits, VOCAB * sizeof(float)), "cudaMalloc logits");
  ck(cudaMalloc(&w, (unsigned long long)VOCAB * HIDDEN * sizeof(__nv_bfloat16)), "cudaMalloc weights");
  ck(cudaMalloc(&tok, sizeof(std::uint32_t)), "cudaMalloc tok");
  ck(cudaMalloc(&id, sizeof(std::uint32_t)), "cudaMalloc id");
  ck(cudaMalloc(&experts, TOPK * sizeof(std::uint16_t)), "cudaMalloc experts");
  ck(cudaMalloc(&ew, TOPK * sizeof(float)), "cudaMalloc expert weights");
  ck(cudaMemset(w, 0, (unsigned long long)VOCAB * HIDDEN * sizeof(__nv_bfloat16)), "cudaMemset weights");
  ck(cudaMemset(tok, 0, sizeof(std::uint32_t)), "cudaMemset tok");
  ck(cudaMemset(x, 0, HIDDEN * sizeof(float)), "cudaMemset x");
  embed_k<<<(HIDDEN + 255) / 256, 256>>>(w, tok, x);
  for (int layer = 0; layer < 61; ++layer) {
    rmsnorm_k<<<1, 256>>>(x, w, n, HIDDEN);
    matvec_bf16_k<<<HEADS * (QK_NOPE + QK_ROPE), 256>>>(w, n, q, HEADS * (QK_NOPE + QK_ROPE), HIDDEN);
    matvec_bf16_k<<<HEADS * (QK_NOPE + QK_ROPE), 256>>>(w, n, k, HEADS * (QK_NOPE + QK_ROPE), HIDDEN);
    matvec_bf16_k<<<HEADS * V_HEAD, 256>>>(w, n, v, HEADS * V_HEAD, HIDDEN);
    rope_k<<<(HEADS * QK_ROPE / 2 + 255) / 256, 256>>>(q, k, layer, HEADS, QK_NOPE + QK_ROPE, QK_ROPE);
    kv_store_k<<<(HEADS * (QK_NOPE + QK_ROPE) + 255) / 256, 256>>>(k, v, k, v, 0, HEADS * (QK_NOPE + QK_ROPE), HEADS * V_HEAD);
    attn_score_k<<<1, 256>>>(q, k, score, 1, HEADS * (QK_NOPE + QK_ROPE));
    softmax_k<<<1, 256>>>(score, 1);
    attn_value_k<<<(HEADS * V_HEAD + 255) / 256, 256>>>(score, v, av, 1, HEADS * V_HEAD);
    matvec_bf16_k<<<HIDDEN, 256>>>(w, av, down, HIDDEN, HEADS * V_HEAD);
    add_k<<<(HIDDEN + 255) / 256, 256>>>(x, down, HIDDEN);
    rmsnorm_k<<<1, 256>>>(x, w, n, HIDDEN);
    matvec_bf16_k<<<EXPERTS, 256>>>(w, n, gate, EXPERTS, HIDDEN);
    router_topk_k<<<1, EXPERTS>>>(gate, experts, ew);
    matvec_bf16_k<<<MOE, 256>>>(w, n, gate, MOE, HIDDEN);
    matvec_bf16_k<<<MOE, 256>>>(w, n, up, MOE, HIDDEN);
    silu_mul_k<<<(MOE + 255) / 256, 256>>>(gate, up, mid, MOE);
    matvec_bf16_k<<<HIDDEN, 256>>>(w, mid, down, HIDDEN, MOE);
    expert_accum_k<<<(HIDDEN + 255) / 256, 256>>>(down, ew, 0, x, HIDDEN);
  }
  rmsnorm_k<<<1, 256>>>(x, w, n, HIDDEN);
  matvec_bf16_k<<<VOCAB, 256>>>(w, n, logits, VOCAB, HIDDEN);
  argmax_k<<<1, 256>>>(logits, id, VOCAB);
  ck(cudaGetLastError(), "decode kernel launch");
  ck(cudaDeviceSynchronize(), "decode kernel sync");
  for (void *p : {x, n, q, k, v, av, score, gate, up, mid, down, logits, w, tok, id, experts, ew}) ck(cudaFree(p), "cudaFree decode");
}

int main(int argc, char **argv) {
  Args a = args(argc, argv);
  auto rt = runtime_index(a.manifest);
  validate_forward_contract(rt);
  if (std::getenv("BLOK_KERNEL_SELFTEST")) decode_kernel_skeleton();
  stage_tensor(*find(rt.tensors, "resident", "embed_tokens"));
  stage_tensor(*find(rt.tensors, "attention_resident"));
  stage_tensor(*find(rt.tensors, "router"));
  die("forward kernels incomplete: tokenizer, MLA attention, expert GEMM, lm_head, and sampler must emit real tokens");
}
