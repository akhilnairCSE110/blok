#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef BLOK_HAVE_UGDS
#include <ugds.h>
#endif

namespace {

struct Args {
  std::string backend = "ugds";
  std::string file;
  std::string device = "/dev/ugds_drv0";
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 4096;
};

__global__ __launch_bounds__(256, 2) void blok_byte_probe_kernel(const uint4 *__restrict__ input,
                                                                 std::uint32_t *out,
                                                                 std::uint64_t lanes) {
  const std::uint64_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= lanes) {
    return;
  }
  std::uint32_t a;
  std::uint32_t b;
  std::uint32_t c;
  std::uint32_t d;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "l"(input + i));
  atomicXor(out, a ^ b ^ c ^ d);
}

__global__ __launch_bounds__(256, 2) void blok_neuron_dot_kernel(const float *__restrict__ x,
                                                                 const float *__restrict__ rows,
                                                                 float *__restrict__ y,
                                                                 std::uint32_t hidden) {
  extern __shared__ float partial[];
  const std::uint32_t neuron = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  float acc = 0.0F;
  for (std::uint32_t i = tid; i < hidden; i += blockDim.x) {
    float a;
    float b;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(a) : "l"(x + i));
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(b) : "l"(rows + (neuron * hidden) + i));
    asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(acc) : "f"(a), "f"(b));
  }
  partial[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    y[neuron] = partial[0];
  }
}

__global__ __launch_bounds__(256, 2) void blok_rmsnorm_kernel(const float *__restrict__ x,
                                                              const float *__restrict__ w,
                                                              float *__restrict__ y,
                                                              std::uint32_t hidden, float eps) {
  extern __shared__ float s[];
  const std::uint32_t tid = threadIdx.x;
  float sum = 0.0F;
  for (std::uint32_t i = tid; i < hidden; i += blockDim.x) {
    const float v = x[i];
    sum += v * v;
  }
  s[tid] = sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf((s[0] / static_cast<float>(hidden)) + eps);
  for (std::uint32_t i = tid; i < hidden; i += blockDim.x) {
    y[i] = x[i] * inv * w[i];
  }
}

__global__ __launch_bounds__(256, 2) void blok_silu_gate_kernel(const float *__restrict__ gate,
                                                                const float *__restrict__ up,
                                                                float *__restrict__ y,
                                                                std::uint32_t n) {
  const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    const float g = gate[i];
    y[i] = (g / (1.0F + expf(-g))) * up[i];
  }
}

__global__ __launch_bounds__(384, 1) void blok_router_topk_kernel(const float *__restrict__ logits,
                                                                  std::uint16_t *__restrict__ expert,
                                                                  float *__restrict__ weight,
                                                                  std::uint32_t experts,
                                                                  std::uint32_t k,
                                                                  float scale) {
  extern __shared__ float scores[];
  const std::uint32_t tid = threadIdx.x;
  if (tid < experts) {
    const float x = logits[tid];
    scores[tid] = 1.0F / (1.0F + expf(-x));
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0F;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      float best = -1.0F;
      std::uint32_t id = 0;
      for (std::uint32_t i = 0; i < experts; ++i) {
        const float v = scores[i];
        if (v > best) {
          best = v;
          id = i;
        }
      }
      expert[slot] = static_cast<std::uint16_t>(id);
      weight[slot] = best;
      total += best;
      scores[id] = -1.0F;
    }
    if (total > 0.0F) {
      for (std::uint32_t slot = 0; slot < k; ++slot) {
        weight[slot] = (weight[slot] / total) * scale;
      }
    }
  }
}

__global__ __launch_bounds__(256, 2) void blok_int4_group_matvec_kernel(
    const float *__restrict__ x, const std::uint8_t *__restrict__ packed,
    const float *__restrict__ scale, float *__restrict__ y, std::uint32_t rows,
    std::uint32_t cols) {
  extern __shared__ float partial[];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }
  float acc = 0.0F;
  const std::uint32_t packed_row = row * ((cols + 1) >> 1);
  for (std::uint32_t c = tid; c < cols; c += blockDim.x) {
    const std::uint8_t byte = packed[packed_row + (c >> 1)];
    const std::int8_t q = static_cast<std::int8_t>((c & 1) ? (byte >> 4) : (byte & 15)) - 8;
    acc += x[c] * static_cast<float>(q) * scale[(row * ((cols + 31) >> 5)) + (c >> 5)];
  }
  partial[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    y[row] = partial[0];
  }
}

__global__ __launch_bounds__(256, 2) void blok_rmsnorm_int4_matvec_kernel(
    const float *__restrict__ x, const float *__restrict__ norm_w,
    const std::uint8_t *__restrict__ packed, const float *__restrict__ scale,
    float *__restrict__ y, std::uint32_t rows, std::uint32_t cols, float eps) {
  extern __shared__ float s[];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (row >= rows) {
    return;
  }
  float ss = 0.0F;
  for (std::uint32_t c = tid; c < cols; c += blockDim.x) {
    const float v = x[c];
    ss += v * v;
  }
  s[tid] = ss;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf((s[0] / static_cast<float>(cols)) + eps);
  float acc = 0.0F;
  const std::uint32_t packed_row = row * ((cols + 1) >> 1);
  for (std::uint32_t c = tid; c < cols; c += blockDim.x) {
    const std::uint8_t byte = packed[packed_row + (c >> 1)];
    const std::int8_t q = static_cast<std::int8_t>((c & 1) ? (byte >> 4) : (byte & 15)) - 8;
    const float a = x[c] * inv * norm_w[c];
    acc += a * static_cast<float>(q) * scale[(row * ((cols + 31) >> 5)) + (c >> 5)];
  }
  s[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    y[row] = s[0];
  }
}

__global__ __launch_bounds__(256, 2) void blok_moe_silu_weighted_sum_kernel(
    const float *__restrict__ gate, const float *__restrict__ up, const float *__restrict__ down,
    const float *__restrict__ weight, float *__restrict__ y, std::uint32_t k,
    std::uint32_t hidden) {
  const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= hidden) {
    return;
  }
  float acc = 0.0F;
  for (std::uint32_t e = 0; e < k; ++e) {
    const float g = gate[(e * hidden) + i];
    const float u = up[(e * hidden) + i];
    acc += weight[e] * (g / (1.0F + expf(-g))) * u * down[(e * hidden) + i];
  }
  y[i] += acc;
}

[[noreturn]] void die(const std::string &message) {
  std::cerr << "blok-byte-probe: " << message << "\n";
  std::exit(1);
}

void cuda_ok(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    die(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

std::uint64_t parse_u64(const char *text, const char *name) {
  char *end = nullptr;
  errno = 0;
  const auto value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    die(std::string("invalid ") + name);
  }
  return value;
}

Args parse(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      die("usage: blok-byte-probe --file <path> --offset <bytes> --bytes <bytes> [--alignment <bytes>]");
    }
    const std::string flag = argv[i];
    if (flag == "--file") {
      args.file = argv[i + 1];
    } else if (flag == "--backend") {
      args.backend = argv[i + 1];
    } else if (flag == "--device") {
      args.device = argv[i + 1];
    } else if (flag == "--offset") {
      args.offset = parse_u64(argv[i + 1], "offset");
    } else if (flag == "--bytes") {
      args.bytes = parse_u64(argv[i + 1], "bytes");
    } else if (flag == "--alignment") {
      args.alignment = parse_u64(argv[i + 1], "alignment");
    } else {
      die("unknown flag: " + flag);
    }
  }
  if (args.bytes == 0 || (args.backend == "odirect" && args.file.empty())) {
    die("nonzero --bytes is required; --file is required for --backend odirect");
  }
  if (args.backend != "ugds" && args.backend != "odirect") {
    die("--backend must be ugds or odirect");
  }
  if ((args.alignment & (args.alignment - 1)) != 0 || args.alignment < 4096) {
    die("--alignment must be a power of two and at least 4096");
  }
  if ((args.offset % args.alignment) != 0 || (args.bytes % args.alignment) != 0 ||
      (args.bytes % sizeof(uint4)) != 0) {
    die("offset and bytes must satisfy O_DIRECT and uint4 alignment");
  }
  return args;
}

std::uint32_t checksum_cpu(const void *data, std::uint64_t bytes) {
  const auto *p = static_cast<const std::uint32_t *>(data);
  std::uint32_t x = 0;
  for (std::uint64_t i = 0; i < bytes / sizeof(std::uint32_t); ++i) {
    x ^= p[i];
  }
  return x;
}

void read_direct(const Args &args, void *dst) {
  const int fd = open(args.file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) {
    die(std::string("open O_DIRECT failed: ") + std::strerror(errno));
  }
  auto *p = static_cast<std::uint8_t *>(dst);
  std::uint64_t done = 0;
  while (done < args.bytes) {
    const ssize_t n = pread(fd, p + done, args.bytes - done, args.offset + done);
    if (n < 0) {
      const int saved = errno;
      close(fd);
      die(std::string("pread O_DIRECT failed: ") + std::strerror(saved));
    }
    if (n == 0) {
      close(fd);
      die("short direct read");
    }
    done += static_cast<std::uint64_t>(n);
  }
  close(fd);
}

void read_ugds(const Args &args, void *device_ptr) {
#ifdef BLOK_HAVE_UGDS
  const int fd = open(args.device.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    die(std::string("open uGDS device failed: ") + std::strerror(errno));
  }
  cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize before uGDS");
  if (uGDSDriverOpen() != 0) {
    close(fd);
    die("uGDSDriverOpen failed");
  }
  uGDSDescr_t desc = {};
  desc.type = UGDS_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  uGDSHandle_t handle = {};
  if (uGDSHandleRegister(&handle, &desc) != 0) {
    uGDSDriverClose();
    close(fd);
    die("uGDSHandleRegister failed");
  }
  if (uGDSBufRegister(device_ptr, args.bytes, 0) != 0) {
    uGDSHandleDeregister(handle);
    uGDSDriverClose();
    close(fd);
    die("uGDSBufRegister failed");
  }
  if (uGDSRead(handle, device_ptr, args.bytes, args.offset, 0) != static_cast<ssize_t>(args.bytes)) {
    uGDSBufDeregister(device_ptr);
    uGDSHandleDeregister(handle);
    uGDSDriverClose();
    close(fd);
    die("uGDSRead failed");
  }
  uGDSBufDeregister(device_ptr);
  uGDSHandleDeregister(handle);
  uGDSDriverClose();
  close(fd);
#else
  (void)args;
  (void)device_ptr;
  die("uGDS backend requested but ugds.h/libugds were not found at build time");
#endif
}

std::string json_escape(const std::string &s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  out += '"';
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  const Args args = parse(argc, argv);

  void *device = nullptr;
  std::uint32_t *device_out = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_ok(cudaMalloc(&device, args.bytes), "cudaMalloc input");
  cuda_ok(cudaMalloc(&device_out, sizeof(std::uint32_t)), "cudaMalloc output");
  cuda_ok(cudaMemset(device_out, 0, sizeof(std::uint32_t)), "cudaMemset output");
  cuda_ok(cudaEventCreate(&start), "cudaEventCreate start");
  cuda_ok(cudaEventCreate(&stop), "cudaEventCreate stop");
  if (args.backend == "ugds") {
    read_ugds(args, device);
  } else {
    void *host = nullptr;
    if (posix_memalign(&host, args.alignment, args.bytes) != 0) {
      die("posix_memalign failed");
    }
    read_direct(args, host);
    cuda_ok(cudaMemcpy(device, host, args.bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    std::free(host);
  }

  constexpr int threads = 256;
  const std::uint64_t lanes = args.bytes / sizeof(uint4);
  const auto blocks = static_cast<unsigned>((lanes + threads - 1) / threads);
  cuda_ok(cudaEventRecord(start), "cudaEventRecord start");
  blok_byte_probe_kernel<<<blocks, threads>>>(static_cast<const uint4 *>(device), device_out, lanes);
  cuda_ok(cudaGetLastError(), "blok_byte_probe_kernel launch");
  cuda_ok(cudaEventRecord(stop), "cudaEventRecord stop");
  cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float kernel_ms = 0.0F;
  std::uint32_t gpu = 0;
  void *verify = nullptr;
  if (posix_memalign(&verify, args.alignment, args.bytes) != 0) {
    die("verify posix_memalign failed");
  }
  cuda_ok(cudaEventElapsedTime(&kernel_ms, start, stop), "cudaEventElapsedTime");
  cuda_ok(cudaMemcpy(&gpu, device_out, sizeof(gpu), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  cuda_ok(cudaMemcpy(verify, device, args.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy verify");
  const std::uint32_t cpu = checksum_cpu(verify, args.bytes);

  std::cout << "{\"schema\":\"blok.byte_probe.v0\",\"backend\":" << json_escape(args.backend) << ","
            << "\"file\":" << json_escape(args.file) << ",\"offset\":" << args.offset
            << ",\"bytes\":" << args.bytes << ",\"alignment\":" << args.alignment
            << ",\"target_sm\":\"sm_120\",\"kernel\":\"blok_byte_probe_kernel\","
            << "\"threads\":" << threads << ",\"blocks\":" << blocks
            << ",\"cpu_checksum\":" << cpu << ",\"gpu_checksum\":" << gpu
            << ",\"verified\":" << (cpu == gpu ? "true" : "false")
            << ",\"kernel_ms\":" << kernel_ms << "}\n";

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(device_out);
  cudaFree(device);
  std::free(verify);
  return cpu == gpu ? 0 : 2;
}
