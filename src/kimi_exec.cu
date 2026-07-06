#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef BLOK_HAVE_UGDS
#include <ugds.h>
#endif

struct Tensor {
  std::string name, role, file;
  std::uint64_t off = 0, bytes = 0, align = 4096;
};

struct Args {
  std::string manifest, prompt;
  std::uint64_t tokens = 0, topk = 8;
};

__global__ __launch_bounds__(384, 1) void router_topk(const float *__restrict__ logits,
                                                      std::uint16_t *__restrict__ expert,
                                                      float *__restrict__ weight) {
  __shared__ float s[384];
  const int tid = threadIdx.x;
  s[tid] = 1.0F / (1.0F + expf(-logits[tid]));
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0F;
    for (int k = 0; k < 8; ++k) {
      int id = 0;
      float best = -1.0F;
      for (int i = 0; i < 384; ++i)
        if (s[i] > best) best = s[i], id = i;
      expert[k] = static_cast<std::uint16_t>(id);
      weight[k] = best;
      sum += best;
      s[id] = -1.0F;
    }
    for (int k = 0; k < 8; ++k) weight[k] = weight[k] / sum * 2.827F;
  }
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

std::vector<Tensor> manifest(const std::string &path) {
  std::ifstream in(path);
  if (!in) die("open manifest failed: " + path);
  std::vector<Tensor> out;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.starts_with("tensor ")) continue;
    std::istringstream ss(line);
    std::string tag, dtype, shape;
    Tensor t;
    ss >> tag >> t.name >> t.role >> dtype >> shape >> t.off >> t.bytes >> t.align >> t.file;
    if (!t.file.empty()) out.push_back(t);
  }
  if (out.empty()) die("manifest has no file-backed tensors");
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

std::string js(const std::string &s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

int main(int argc, char **argv) {
  Args a = args(argc, argv);
  auto ts = manifest(a.manifest);
  const Tensor *first = nullptr;
  for (const auto &t : ts)
    if (t.role == "router" || t.role == "attention_resident" || t.role == "resident") {
      first = &t;
      break;
    }
  if (!first) first = &ts[0];
  void *host = nullptr;
  if (posix_memalign(&host, first->align, first->bytes)) die("posix_memalign failed");
  read_direct(*first, host);
  void *dev = nullptr;
  ck(cudaMalloc(&dev, first->bytes), "cudaMalloc");
  ck(cudaMemcpy(dev, host, first->bytes, cudaMemcpyHostToDevice), "cudaMemcpy");

  float *logits = nullptr, *weights = nullptr;
  std::uint16_t *experts = nullptr;
  ck(cudaMalloc(&logits, 384 * sizeof(float)), "cudaMalloc logits");
  ck(cudaMalloc(&weights, 8 * sizeof(float)), "cudaMalloc weights");
  ck(cudaMalloc(&experts, 8 * sizeof(std::uint16_t)), "cudaMalloc experts");
  ck(cudaMemset(logits, 0, 384 * sizeof(float)), "cudaMemset logits");
  router_topk<<<1, 384>>>(logits, experts, weights);
  ck(cudaGetLastError(), "router_topk");
  ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  die("decode kernels incomplete: tokenizer, MLA attention, expert GEMM, sampler not wired");
}
