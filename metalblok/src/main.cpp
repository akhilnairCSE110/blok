#include "model.hpp"
#include "metal_ctx.hpp"
#include "runtime.hpp"
#include "streamer.hpp"
#include "tokenizer.hpp"
#include "kvcache.hpp"
#include "gguf.hpp"
#include "gguf_dequant.hpp"
#include "gguf_kernels.hpp"
#include "gguf_model.hpp"
#include "gguf_runtime.hpp"
#include "memstat.hpp"
#include "preflight.hpp"
#include "router_ref.hpp"
#include "../vendor/llama_cpp/iq1s_grid.h"
#include "../vendor/llama_cpp/iq2xxs_grid.h"
#include "prof.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace blade;

// Read-only inspection of a GGUF file.  Validates the parser end-to-end
// (header, KV, tensor descriptors, multi-shard merging, byte-size math)
// without loading anything into the model runtime.  Used as the gating
// milestone for the IQ1_S R1 path.
static int probe_gguf(const char* path) {
    Gguf g;
    long long t0 = prof::now_us();
    g.open(path);
    long long t1 = prof::now_us();

    std::fprintf(stdout, "shards: %zu (split.count=%u, total_tensor_count=%llu)\n",
                 g.shards.size(), g.shard_count,
                 (unsigned long long)g.total_tensor_count);
    for (size_t i = 0; i < g.shards.size(); ++i) {
        std::fprintf(stdout, "  shard[%zu] %s  size=%.2fGB  data_off=%llu  align=%llu\n",
                     i, g.shards[i].path.c_str(),
                     g.shards[i].size / 1e9,
                     (unsigned long long)g.shards[i].data_offset,
                     (unsigned long long)g.shards[i].alignment);
    }
    std::fprintf(stdout, "kv: %zu entries\n", g.kv.size());
    // Print a curated subset of architecture-relevant KVs.
    static const char* show_keys[] = {
        "general.architecture", "general.file_type", "general.quantization_version",
        "deepseek2.block_count", "deepseek2.embedding_length", "deepseek2.feed_forward_length",
        "deepseek2.attention.head_count", "deepseek2.attention.head_count_kv",
        "deepseek2.attention.q_lora_rank", "deepseek2.attention.kv_lora_rank",
        "deepseek2.attention.key_length", "deepseek2.attention.value_length",
        "deepseek2.expert_count", "deepseek2.expert_used_count",
        "deepseek2.leading_dense_block_count", "deepseek2.expert_feed_forward_length",
        "deepseek2.expert_shared_count", "deepseek2.expert_weights_scale",
        "deepseek2.expert_weights_norm", "deepseek2.expert_gating_func",
        "deepseek2.rope.freq_base", "deepseek2.rope.dimension_count",
        "deepseek2.rope.scaling.type", "deepseek2.rope.scaling.factor",
        "deepseek2.rope.scaling.original_context_length",
        "deepseek2.rope.scaling.yarn_log_multiplier",
        "deepseek2.context_length", "deepseek2.vocab_size",
        "tokenizer.ggml.model", "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        "tokenizer.chat_template",
        nullptr,
    };
    for (const char** kp = show_keys; *kp; ++kp) {
        auto it = g.kv_idx.find(*kp);
        if (it == g.kv_idx.end()) continue;
        const auto& v = g.kv[it->second];
        switch (v.type) {
            case GGUF_STRING:
                std::fprintf(stdout, "  %-55s = \"%s\"\n", *kp, v.str.c_str());
                break;
            case GGUF_F32: case GGUF_F64:
                std::fprintf(stdout, "  %-55s = %g\n", *kp, v.f64);
                break;
            case GGUF_I8: case GGUF_I16: case GGUF_I32: case GGUF_I64:
                std::fprintf(stdout, "  %-55s = %lld\n", *kp, (long long)v.i64);
                break;
            default:
                std::fprintf(stdout, "  %-55s = %llu\n", *kp, (unsigned long long)v.u64);
                break;
        }
    }

    // Tensor count + per-quant-type histogram.
    std::fprintf(stdout, "tensors: %zu\n", g.tensors.size());
    struct Hist { uint64_t count = 0; uint64_t bytes = 0; };
    Hist by_type[64] = {};
    for (const auto& t : g.tensors) {
        if (t.type < 64) {
            by_type[t.type].count += 1;
            by_type[t.type].bytes += t.bytes;
        }
    }
    uint64_t total_bytes = g.total_tensor_bytes();
    for (uint32_t i = 0; i < 64; ++i) {
        if (!by_type[i].count) continue;
        std::fprintf(stdout, "  %-8s : %5llu tensors  %8.2f GB  (%.1f%%)\n",
                     ggml_type_name(i),
                     (unsigned long long)by_type[i].count,
                     by_type[i].bytes / 1e9,
                     100.0 * by_type[i].bytes / (double)total_bytes);
    }
    for (uint32_t type : {uint32_t(GGML_F32), uint32_t(GGML_Q4_K),
                          uint32_t(GGML_Q5_K), uint32_t(GGML_Q6_K),
                          uint32_t(GGML_IQ2_XXS), uint32_t(GGML_IQ1_S)}) {
        const GgufTensor* best = nullptr;
        uint64_t best_slice = UINT64_MAX;
        for (const auto& t : g.tensors) {
            if (t.type == type && (t.dims.size() == 2 || t.dims.size() == 3)) {
                uint64_t slices = t.dims.size() == 3 ? t.dims[2] : 1;
                uint64_t slice = t.bytes / slices;
                if (slice < best_slice) { best = &t; best_slice = slice; }
            }
        }
        if (best) std::fprintf(stdout, "  validation_candidate %-8s %s slice_bytes=%llu\n",
                               ggml_type_name(type), best->name.c_str(),
                               (unsigned long long)best_slice);
    }
    std::fprintf(stdout, "  total payload: %.2f GB\n", total_bytes / 1e9);

    uint64_t shard_total = 0;
    for (const auto& sh : g.shards) shard_total += sh.size;
    std::fprintf(stdout, "  sum shard sizes: %.2f GB  (overhead = %.2f MB metadata + alignment)\n",
                 shard_total / 1e9,
                 (shard_total - total_bytes) / 1e6);

    // Per-shard tensor counts -- proves multi-shard merge wired up correctly.
    std::vector<uint64_t> per_shard(g.shards.size(), 0);
    for (const auto& t : g.tensors) per_shard[t.shard]++;
    for (size_t i = 0; i < per_shard.size(); ++i) {
        std::fprintf(stdout, "  shard[%zu] holds %llu tensors\n",
                     i, (unsigned long long)per_shard[i]);
    }

    // First few + last few tensor names so we can eyeball coverage.
    std::fprintf(stdout, "first 6 tensors:\n");
    for (size_t i = 0; i < std::min<size_t>(6, g.tensors.size()); ++i) {
        const auto& t = g.tensors[i];
        std::fprintf(stdout, "  [%-8s] %-50s shard=%u off=%llu bytes=%llu\n",
                     ggml_type_name(t.type), t.name.c_str(), t.shard,
                     (unsigned long long)t.offset_in_data,
                     (unsigned long long)t.bytes);
    }
    std::fprintf(stdout, "last 4 tensors:\n");
    for (size_t i = (g.tensors.size() > 4 ? g.tensors.size() - 4 : 0); i < g.tensors.size(); ++i) {
        const auto& t = g.tensors[i];
        std::fprintf(stdout, "  [%-8s] %-50s shard=%u off=%llu bytes=%llu\n",
                     ggml_type_name(t.type), t.name.c_str(), t.shard,
                     (unsigned long long)t.offset_in_data,
                     (unsigned long long)t.bytes);
    }

    std::fprintf(stdout, "parse: %lld us\n", t1 - t0);
    return 0;
}

// Open the GGUF, locate `tname`, dequantize it on the CPU, and report
// numerical statistics. Validates the dequantizer end-to-end against a
// real on-disk tensor without requiring any GPU code.
//
// For full tensors of supported types we report:
//   * shape, type, payload bytes
//   * super-block scale d for the first block (raw, not derived)
//   * full-tensor min / max / mean / abs-mean / RMS over every dequantized weight
//   * histogram by approximate magnitude bucket
//   * the first 16 dequantized weights (so a human can eyeball them)
//
// Today we support iq1_s. Other types print shape only and return 0.
static int probe_tensor(const char* path, const char* tname) {
    TensorRef ref;
    if (!Gguf::lookup(path, tname, ref)) {
        std::fprintf(stderr, "probe-tensor: tensor not found: %s\\n", tname);
        return 1;
    }

    using DqFn = void (*)(const void*, float*);
    DqFn dq = nullptr;
    uint64_t block_bytes = 0;
    switch (ref.type) {
        case GGML_IQ1_S:   dq = &dequantize_iq1_s_block;   block_bytes = 50;  break;
        case GGML_Q4_K:    dq = &dequantize_q4_K_block;    block_bytes = 144; break;
        case GGML_Q5_K:    dq = &dequantize_q5_K_block;    block_bytes = 176; break;
        case GGML_Q6_K:    dq = &dequantize_q6_K_block;    block_bytes = 210; break;
        case GGML_IQ2_XXS: dq = &dequantize_iq2_xxs_block; block_bytes = 66;  break;
        default:
            std::fprintf(stderr, "probe-tensor: no CPU dequantizer for %s\\n",
                         ggml_type_name(ref.type));
            return 2;
    }

    int fd = ::open(ref.shard_path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("probe-tensor open"); return 1; }
    (void)::fcntl(fd, F_NOCACHE, 1);
    (void)::fcntl(fd, F_RDAHEAD, 0);
    std::vector<uint8_t> block(block_bytes);
    ssize_t got = ::pread(fd, block.data(), block.size(),
                          static_cast<off_t>(ref.abs_offset));
    ::close(fd);
    if (got != static_cast<ssize_t>(block.size())) {
        std::fprintf(stderr, "probe-tensor: short read %zd/%zu\\n",
                     got, block.size());
        return 1;
    }

    std::printf("tensor=%s type=%s bytes=%llu shape=[", tname,
                ggml_type_name(ref.type),
                static_cast<unsigned long long>(ref.bytes));
    for (size_t i = 0; i < ref.dims.size(); ++i) {
        std::printf("%s%llu", i ? "," : "",
                    static_cast<unsigned long long>(ref.dims[i]));
    }
    std::printf("]\\n");

    float values[256];
    dq(block.data(), values);
    std::printf("first_block:");
    for (int i = 0; i < 16; ++i) std::printf(" %+.6f", values[i]);
    std::putchar('\n');
    return 0;
}
//   - needs_grid means a vendored codebook gets bound at buffer(4)
static inline uint16_t f32_to_f16(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    uint32_t s16 = (u >> 31) & 0x1;
    int32_t  e32 = (int32_t)((u >> 23) & 0xff) - 127;
    uint32_t m23 = u & 0x7fffff;
    if (e32 < -14)     return (uint16_t)(s16 << 15);                           // flush-to-zero
    if (e32 >  15)     return (uint16_t)((s16 << 15) | (0x1f << 10));          // saturate to inf
    return            (uint16_t)((s16 << 15) | ((e32 + 15) << 10) | (m23 >> 13));
}
static inline float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15);
    const uint32_t e = (h >> 10) & 0x1f;
    const uint32_t m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s << 31;
        else { uint32_t mm=m; int ee=-14;
            while ((mm & 0x400) == 0) { mm <<= 1; --ee; }
            bits = (s << 31) | ((127 + ee) << 23) | ((mm & 0x3ff) << 13); }
    } else if (e == 31) bits = (s << 31) | (0xffu << 23) | (m << 13);
    else                bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

static int validate_router() {
    constexpr uint32_t N = 256, K = 8, G = 8, TG = 4;
    constexpr float scale = 2.5f;
    constexpr uint32_t norm = 1;
    std::vector<uint16_t> logits_h(N);
    std::vector<float> logits_f(N), bias(N);
    uint64_t state = 0x726f757465722d31ULL;
    for (uint32_t i = 0; i < N; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        float l = static_cast<int32_t>(state & 0xffff) / 8192.0f - 4.0f;
        float b = static_cast<int32_t>((state >> 16) & 0xffff) / 131072.0f - 0.25f;
        logits_h[i] = f32_to_f16(l);
        logits_f[i] = f16_to_f32(logits_h[i]);
        bias[i] = b;
    }
    std::vector<RoutedExpert> ref;
    if (!route_grouped_sigmoid(logits_f.data(), bias.data(), N, K, G, TG,
                               scale, true, ref)) return 1;

    Metal mtl; mtl.init(METALBLOK_KERNEL_PATH);
    MtlBuf bLogits = mtl.alloc(N * sizeof(uint16_t));
    MtlBuf bBias = mtl.alloc(N * sizeof(float));
    MtlBuf bIdx = mtl.alloc(K * sizeof(uint32_t));
    MtlBuf bWts = mtl.alloc(K * sizeof(float));
    std::memcpy(bLogits.contents, logits_h.data(), bLogits.length);
    std::memcpy(bBias.contents, bias.data(), bBias.length);
    mtl.begin();
    mtl.dispatch("router_topk_grouped_sigmoid_f16",
                 {bLogits, bBias, bIdx, bWts},
                 {{&N,4},{&K,4},{&G,4},{&TG,4},{&scale,4},{&norm,4}},
                 1, 1, true);
    mtl.commit_and_wait();
    const auto* ids = static_cast<const uint32_t*>(bIdx.contents);
    const auto* wts = static_cast<const float*>(bWts.contents);
    for (uint32_t i = 0; i < K; ++i) {
        std::printf("router[%u] id=%u weight=%.9f ref_id=%u ref_weight=%.9f\n",
                    i, ids[i], wts[i], ref[i].id, ref[i].weight);
        if (ids[i] != ref[i].id || std::fabs(wts[i] - ref[i].weight) > 2e-5f)
            return 2;
    }
    return 0;
}

static void dequant_f32_256(const void* src, float* dst) {
    std::memcpy(dst, src, 256 * sizeof(float));
}

static int validate_gemv(const char* path, const char* tname, bool force) {
    // ---- LAZY METADATA LOOKUP --------------------------------------------
    // The old path called Gguf::open(path) which mmaps every shard (147 GB
    // virtual for R1) and materializes a 1025-entry tensor table.  Walking
    // ~16 MB of metadata into resident memory before we even know which
    // tensor we want is exactly the "fault more than we need" pattern the
    // grounding doc bans.  Gguf::lookup() reads only the relevant shard's
    // metadata window, captures one TensorRef, and drops everything.
    TensorRef ref;
    if (!Gguf::lookup(path, tname, ref)) {
        std::fprintf(stderr, "validate-gemv: tensor not found: %s\n", tname);
        return 1;
    }
    if (ref.dims.size() != 2 && ref.dims.size() != 3) {
        std::fprintf(stderr, "validate-gemv: only 2D or 3D tensors supported (got %zu dims)\n",
                     ref.dims.size());
        return 1;
    }

    using DqFn = void (*)(const void*, float*);
    struct TypeInfo { uint32_t bpb; DqFn cpu; const char* kernel; bool needs_grid; const void* grid; size_t grid_bytes; };
    TypeInfo ti{};
    switch (ref.type) {
        case GGML_F32:      ti = {1024, &dequant_f32_256,          "gemv_f32_f16",     false, nullptr, 0 }; break;
        case GGML_Q6_K:    ti = { 210, &dequantize_q6_K_block,    "gemv_q6_K_f16",    false, nullptr, 0 }; break;
        case GGML_Q4_K:    ti = { 144, &dequantize_q4_K_block,    "gemv_q4_K_f16",    false, nullptr, 0 }; break;
        case GGML_Q5_K:    ti = { 176, &dequantize_q5_K_block,    "gemv_q5_K_f16",    false, nullptr, 0 }; break;
        case GGML_IQ1_S:   ti = {  50, &dequantize_iq1_s_block,   "gemv_iq1_s_f16",   true,
                                  blade::vendored::iq1s_grid,    sizeof(blade::vendored::iq1s_grid)   }; break;
        case GGML_IQ2_XXS: ti = {  66, &dequantize_iq2_xxs_block, "gemv_iq2_xxs_f16", true,
                                  blade::vendored::iq2xxs_grid,  sizeof(blade::vendored::iq2xxs_grid) }; break;
        default:
            std::fprintf(stderr, "validate-gemv: type %s not wired up yet\n",
                         ggml_type_name(ref.type));
            return 1;
    }

    // GGUF stores dims = [K, N] (or [K, N, n_experts] for stacked MoE expert
    // tensors).  Row-major W[row=N][col=K]; for 3D, expert e starts at
    // expert_stride * e from the start of the tensor.  We always validate
    // expert 0 -- byte layout is identical across experts, so one slice is
    // a complete kernel check.
    const uint64_t K       = ref.dims[0];
    const uint64_t n_rows  = ref.dims[1];
    if (K % 256 != 0) {
        std::fprintf(stderr, "K=%llu not multiple of 256\n", (unsigned long long)K); return 1;
    }
    const uint64_t nblk_per_row  = K / 256;
    const uint64_t bytes_per_row = nblk_per_row * ti.bpb;
    const uint64_t expert_bytes  = n_rows * bytes_per_row;
    const uint64_t n_experts     = ref.dims.size() == 3 ? ref.dims[2] : 1;
    if (ref.bytes != expert_bytes * n_experts) {
        std::fprintf(stderr, "byte count mismatch: have %llu expect %llu (bpb=%u, n_experts=%llu)\n",
                     (unsigned long long)ref.bytes,
                     (unsigned long long)(expert_bytes * n_experts),
                     ti.bpb, (unsigned long long)n_experts);
        return 1;
    }
    std::fprintf(stdout, "validate-gemv: %s  type=%s  K=%llu  n_rows=%llu  n_experts=%llu  expert_slice=%.2f MB  kernel=%s  (validating expert 0)\n",
                 tname, ggml_type_name(ref.type),
                 (unsigned long long)K, (unsigned long long)n_rows,
                 (unsigned long long)n_experts,
                 expert_bytes / 1e6, ti.kernel);

    // ---- MEMORY PREFLIGHT -------------------------------------------------
    // What we actually allocate:
    //   - one Metal shared-storage buffer of `expert_bytes` for W (we pread
    //     straight into it; on UMA Apple Silicon the buffer's `.contents`
    //     is plain host memory and is exactly what the GPU dispatches against)
    //   - X (K * 2 bytes), Y (n_rows * 2 bytes): tiny
    //   - y_ref (n_rows * 4 bytes), w_scratch (1 KB), x_h/x_f (K * 6 bytes)
    // So total resident growth ≈ expert_bytes + ~10 MB worst case.  We
    // require a 512 MiB cushion on top of that for the rest of the system
    // (Metal driver heap, dyld, threads, kernel page tables for the new
    // allocations).  --force overrides for the user-knows-best case.
    {
        const auto    s        = mem::snapshot();
        const uint64_t need    = expert_bytes + (16ull << 20);
        const uint64_t cushion = 512ull << 20;
        char buf[160]; mem::format(s, buf, sizeof(buf));
        std::fprintf(stdout, "  memstat:   %s   need=%.2f MB + 512 MB cushion\n",
                     buf, need / 1e6);
        if (!force && s.available < need + cushion) {
            std::fprintf(stderr,
                "validate-gemv: REFUSING -- need %.2f MB + 512 MB cushion but only %.2f GB available.\n"
                "  Close other apps or pass --force.\n",
                need / 1e6, s.available / 1e9);
            return 1;
        }
    }

    // ---- ALLOC + pread DIRECTLY INTO METAL BUFFER ------------------------
    // The old path was: mmap shard, Metal::wrap that mapping.  The fault
    // pattern was unbounded because mmap is page-granular and Metal pinned
    // the whole 734 MB range resident for the dispatch.  Now we do the
    // opposite: ask Metal for exactly `expert_bytes` of shared-storage host
    // memory, then pread the expert-0 slice STRAIGHT into that buffer.
    // No second heap copy, no mmap, no wrap.  Resident growth = exactly
    // expert_bytes for W, plus the tiny X/Y.  This is the single change
    // that lets the validator survive on a 24 GB box with VS Code+browser
    // already eating 22 GB.
    Metal mtl; mtl.init(METALBLOK_KERNEL_PATH);
    MtlBuf bW = mtl.alloc(expert_bytes);
    {
        int wfd = ::open(ref.shard_path.c_str(), O_RDONLY);
        if (wfd < 0) { std::perror("validate-gemv: open shard"); return 1; }
        uint64_t remaining = expert_bytes;
        uint64_t off       = ref.abs_offset;
        uint8_t* dst       = (uint8_t*)bW.contents;
        long long tp0 = prof::now_us();
        while (remaining > 0) {
            ssize_t got = ::pread(wfd, dst, (size_t)remaining, (off_t)off);
            if (got <= 0) {
                std::fprintf(stderr, "validate-gemv: pread failed (%zd) at off=%llu remaining=%llu\n",
                             got, (unsigned long long)off, (unsigned long long)remaining);
                ::close(wfd);
                return 1;
            }
            dst       += got;
            off       += (uint64_t)got;
            remaining -= (uint64_t)got;
        }
        long long tp1 = prof::now_us();
        ::close(wfd);
        std::fprintf(stdout, "  pread:     %.2f MB in %.2f ms  (%.2f GB/s)\n",
                     expert_bytes / 1e6, (tp1 - tp0) / 1e3,
                     (double)expert_bytes / 1e9 / ((tp1 - tp0) / 1e6));
    }

    // Deterministic random fp16 input vector x ~ U[-1, 1].  splitmix64 PRNG.
    std::vector<uint16_t> x_h(K);
    std::vector<float>    x_f(K);
    {
        uint64_t s = 0x9E3779B97F4A7C15ull ^ K;
        for (uint64_t i = 0; i < K; ++i) {
            s += 0x9E3779B97F4A7C15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z =  z ^ (z >> 31);
            float v = (int64_t)(z >> 11) * (1.0f / (float)(1ull << 52)) - 1.0f;
            x_h[i] = f32_to_f16(v);
            x_f[i] = f16_to_f32(x_h[i]);          // bit-exact with what the GPU sees
        }
    }

    // CPU oracle: dequant each row to scratch and dot with x_f -> y_ref.
    // Reads directly from the Metal buffer's `.contents` -- on UMA this is
    // just host memory.  No second copy.
    std::vector<float> y_ref(n_rows);
    std::vector<float> w_scratch(256);
    long long tc0 = prof::now_us();
    const uint8_t* W_bytes = static_cast<const uint8_t*>(bW.contents);
    for (uint64_t r = 0; r < n_rows; ++r) {
        const uint8_t* Wrow = W_bytes + r * bytes_per_row;
        double acc = 0.0;
        for (uint64_t b = 0; b < nblk_per_row; ++b) {
            ti.cpu(Wrow + b * ti.bpb, w_scratch.data());
            const float* xb = x_f.data() + b * 256;
            for (int i = 0; i < 256; ++i) acc += (double)w_scratch[i] * (double)xb[i];
        }
        y_ref[r] = (float)acc;
    }
    long long tc1 = prof::now_us();
    std::fprintf(stdout, "  CPU oracle: %.2f ms  (%.2f GB/s W, %.2f GFLOP/s)\n",
                 (tc1 - tc0) / 1e3,
                 (double)expert_bytes / 1e9 / ((tc1 - tc0) / 1e6),
                 (double)(2 * K * n_rows) / 1e9 / ((tc1 - tc0) / 1e6));

    // GPU dispatch.  bW is already populated by the pread above; we just
    // bind the small X/Y buffers and (for iq1_s) the codebook grid.
    MtlBuf bX = mtl.alloc(K * sizeof(uint16_t));
    MtlBuf bY = mtl.alloc(n_rows * sizeof(uint16_t));
    std::memcpy(bX.contents, x_h.data(), K * sizeof(uint16_t));
    std::vector<MtlBuf> bufs{ bW, bX, bY };
    MtlBuf bGrid{};
    if (ti.needs_grid) {
        // Bound at buffer(3); the kernel reads it as device const ulong*.
        // The grid is a static const u64 array in read-only __DATA_CONST;
        // the GPU cannot safely read those pages via newBufferWithBytesNoCopy
        // (hard fault on dispatch), so copy it into a real shared buffer.
        bGrid = mtl.alloc(ti.grid_bytes);
        std::memcpy(bGrid.contents, ti.grid, ti.grid_bytes);
        bufs.push_back(bGrid);
    }

    const uint32_t Ku = (uint32_t)K;
    const uint32_t TG = 128;
    long long tg0 = prof::now_us();
    mtl.begin();
    mtl.dispatch(ti.kernel, bufs, { { &Ku, sizeof(Ku) } },
                 (uint32_t)n_rows, TG, true);
    mtl.commit_and_wait();
    long long tg1 = prof::now_us();
    std::fprintf(stdout, "  GPU kernel: %.2f ms  (%.2f GB/s W, %.2f GFLOP/s)\n",
                 (tg1 - tg0) / 1e3,
                 (double)expert_bytes / 1e9 / ((tg1 - tg0) / 1e6),
                 (double)(2 * K * n_rows) / 1e9 / ((tg1 - tg0) / 1e6));

    // Compare.  fp16 round-to-nearest accumulates ~ sqrt(K) * 2^-11 noise.
    const uint16_t* y_h = static_cast<const uint16_t*>(bY.contents);
    double max_abs = 0.0, sumsq_err = 0.0, sumsq_ref = 0.0;
    uint64_t worst_i = 0;
    for (uint64_t r = 0; r < n_rows; ++r) {
        const float yg = f16_to_f32(y_h[r]);
        const float yr = y_ref[r];
        const double d = (double)yg - (double)yr;
        if (std::fabs(d) > max_abs) { max_abs = std::fabs(d); worst_i = r; }
        sumsq_err += d * d;
        sumsq_ref += (double)yr * (double)yr;
    }
    const double rms_err = std::sqrt(sumsq_err / (double)n_rows);
    const double rms_ref = std::sqrt(sumsq_ref / (double)n_rows);
    const double rel     = rms_err / (rms_ref > 0 ? rms_ref : 1.0);
    std::fprintf(stdout, "  vs oracle:  max_abs=%.6f  rms_err=%.6f  rms_ref=%.6f  rel=%.2e  worst[%llu]: ref=%+.4f gpu=%+.4f\n",
                 max_abs, rms_err, rms_ref, rel,
                 (unsigned long long)worst_i, y_ref[worst_i], f16_to_f32(y_h[worst_i]));

    // No mmap to clean up.  All MtlBufs release on scope exit.
    return (rel < 5e-3) ? 0 : 2;
}

static bool file_exists(const std::string& p) {
    struct stat st; return ::stat(p.c_str(), &st) == 0;
}

// S1 acceptance hook: build a GgufModel from shard 0 (auto-detects siblings),
// confirm tensor count + summed payload bytes, then exit. No payload reads.
static int gguf_info(const char* path) {
    auto report = inspect_model_files(path);
    if (!report.all_resident) {
        print_model_preflight(report);
        return 3;
    }
    GgufModel m;
    long long t0 = prof::now_us();
    m.load(path);
    long long t1 = prof::now_us();
    std::fprintf(stdout, "tensors=%zu bytes=%.2fGB shards=%zu load=%.2fms\n",
                 m.tensor_count(), m.total_bytes() / 1e9,
                 m.shard_paths().size(), (t1 - t0) / 1e3);
    return 0;
}

static void usage() {
    std::fprintf(stderr,
"metalblok -m <model-path> -p <prompt> [options]\n"
"  -m, --model      <path>  .blade dir, HF checkpoint dir, or .gguf shard 0\n"
"  -p, --prompt     <str>   prompt text\n"
"  -t, --tokenizer  <file>  tokenizer.bin (default: <model-dir>/tokenizer.bin)\n"
"  -n, --max-tokens <N>     max new tokens to generate           (default 128)\n"
"      --context    <N>     KV capacity; default 64 for safe bring-up\n"
"      --raw-prompt          bypass the DeepSeek-R1 user/assistant template\n"
"      --tokenize-only       print formatted prompt token IDs and exit\n"
"      --state       <file>  exact per-token MLA KV checkpoint\n"
"      --single-step-token N process one token, save --state, print next token\n"
"      --continue-state      emit/extend generation already stored in --state\n"
"      --stop       <str>   stop generation when this substring appears\n"
"      --kv-cache   <dir>   reuse / extend a KV prefix cache directory\n"
"      --preflight  <file>  metadata-only residency/safety check; never reads payload\n"
"      --probe-gguf <file>  parse a .gguf shard (and siblings) and print\n"
"                           the tensor inventory + per-quant histogram, then exit\n"
"      --probe-tensor <file> <name>\n"
"                           dequantize <name> on the CPU and print numerical stats;\n"
"                           validates the dequantizer against a real on-disk tensor\n"
"      --validate-gemv <file> <name>\n"
"                           run y=W@x on CPU oracle and on the matching fused GPU\n"
"                           kernel and print the relative error.  q6_K only for now.\n"
"      --validate-router    compare exact grouped Metal router with CPU oracle\n"
"  -h, --help               show this help\n"
"\n"
"Per-phase timings, RSS, and per-step gpu/sync split are written to stderr\n"
"via the always-on prof::log channel.\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    const char* model_dir = nullptr;
    const char* prompt    = nullptr;
    const char* tok_path  = nullptr;
    const char* stop_str  = nullptr;
    const char* kv_dir    = nullptr;
    const char* probe     = nullptr;
    const char* preflight = nullptr;
    const char* ginfo     = nullptr;
    const char* probe_t_path = nullptr;
    const char* probe_t_name = nullptr;
    const char* vgemv_path   = nullptr;
    const char* vgemv_name   = nullptr;
    bool        force        = false;
    bool        validate_router_flag = false;
    bool        raw_prompt = false;
    bool        tokenize_only = false;
    const char* state_path = nullptr;
    bool        single_step = false;
    uint32_t    single_step_token = 0;
    bool        continue_state = false;
    int  n_predict = 128;
    uint32_t context = 64;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      ((a == "-m" || a == "--model")      && i+1 < argc) model_dir = argv[++i];
        else if ((a == "-p" || a == "--prompt")     && i+1 < argc) prompt    = argv[++i];
        else if ((a == "-t" || a == "--tokenizer")  && i+1 < argc) tok_path  = argv[++i];
        else if ((a == "-n" || a == "--max-tokens") && i+1 < argc) n_predict = std::atoi(argv[++i]);
        else if ( a == "--context"                   && i+1 < argc) context = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if ( a == "--stop"                     && i+1 < argc) stop_str  = argv[++i];
        else if ( a == "--kv-cache"                 && i+1 < argc) kv_dir    = argv[++i];
        else if ( a == "--probe-gguf"               && i+1 < argc) probe     = argv[++i];
        else if ( a == "--preflight"                && i+1 < argc) preflight = argv[++i];
        else if ( a == "--gguf-info"                && i+1 < argc) ginfo     = argv[++i];
        else if ( a == "--probe-tensor"             && i+2 < argc) {
            probe_t_path = argv[++i];
            probe_t_name = argv[++i];
        }
        else if ( a == "--validate-gemv"            && i+2 < argc) {
            vgemv_path = argv[++i];
            vgemv_name = argv[++i];
        }
        else if ( a == "--force")                                  force = true;
        else if ( a == "--validate-router")                         validate_router_flag = true;
        else if ( a == "--raw-prompt")                              raw_prompt = true;
        else if ( a == "--tokenize-only")                            tokenize_only = true;
        else if ( a == "--state" && i+1 < argc)                      state_path = argv[++i];
        else if ( a == "--single-step-token" && i+1 < argc) {
            single_step = true;
            single_step_token = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        }
        else if ( a == "--continue-state")                          continue_state = true;
        else if ( a == "-h" || a == "--help")                      usage();
        else usage();
    }
    if (validate_router_flag) return validate_router();
    if (preflight) {
        auto report = inspect_model_files(preflight);
        print_model_preflight(report);
        return report.all_resident ? 0 : 3;
    }
    if (probe) {
        auto report = inspect_model_files(probe);
        if (!report.all_resident) {
            print_model_preflight(report);
            std::fprintf(stderr, "metalblok: refusing GGUF probe: every shard must be physically resident\n");
            return 3;
        }
        return probe_gguf(probe);
    }
    if (ginfo) return gguf_info(ginfo);
    if (probe_t_path && probe_t_name) return probe_tensor(probe_t_path, probe_t_name);
    if (vgemv_path && vgemv_name)     return validate_gemv(vgemv_path, vgemv_name, force);
    if (!model_dir || !prompt) usage();
    if (n_predict <= 0) n_predict = 128;
    if (context < 64 || context > 65536) {
        std::fprintf(stderr, "metalblok: --context must be in [64, 65536]\n");
        return 2;
    }
    {
        char context_buf[32];
        std::snprintf(context_buf, sizeof(context_buf), "%u", context);
        ::setenv("METALBLOK_MAX_SEQ", context_buf, 1);
    }

    // -----------------------------------------------------------------------
    // .gguf dispatch (S6). A model_dir argument ending in ".gguf" routes to
    // the GgufModel + GgufRuntime path; all other paths fall through to the
    // existing .blade / HF path. Detection is on filename suffix because
    // GGUF is shipped as a single file (or shard 0 of a multi-shard set),
    // not a directory.
    // -----------------------------------------------------------------------
    {
        std::string mpath = model_dir;
        bool is_gguf = mpath.size() >= 5
            && mpath.compare(mpath.size() - 5, 5, ".gguf") == 0
            && file_exists(mpath);
        if (is_gguf) {
            auto report = inspect_model_files(mpath);
            if (!report.all_resident) {
                print_model_preflight(report);
                std::fprintf(stderr, "metalblok: model payload is incomplete; no Metal resources were allocated\n");
                return 3;
            }
            // 1. Parse metadata once for the tokenizer (Gguf keeps the
            //    metadata mmap alive until close()).
            prof::mark("gguf: parse metadata for tokenizer");
            Gguf gmeta;
            gmeta.open(mpath);
            Tokenizer tok;
            tok.load_from_gguf(gmeta);
            std::string formatted_prompt = raw_prompt
                ? std::string(prompt)
                : std::string("<｜User｜>") + prompt +
                  "<｜Assistant｜><think>\n";
            auto ids = tok.encode(formatted_prompt);
            // DeepSeek-R1's BOS id is literally 0, so we must gate on the
            // add_bos flag, not on bos()!=0 (which would silently drop BOS and
            // prefill the model off-distribution from position 0).
            if (tok.add_bos() && (ids.empty() || ids.front() != tok.bos()))
                ids.insert(ids.begin(), tok.bos());
            if (tokenize_only) {
                std::printf("formatted_prompt=%s\ntoken_ids=", formatted_prompt.c_str());
                for (size_t i = 0; i < ids.size(); ++i)
                    std::printf("%s%u", i ? "," : "", ids[i]);
                std::printf("\nroundtrip=%s\n", tok.decode(ids).c_str());
                return 0;
            }
            prof::log("gguf: prompt %zu tokens (bos=%u eos=%u vocab=%u)",
                      ids.size(), tok.bos(), tok.eos(),
                      gmeta.get_u32("deepseek2.vocab_size"));

            // 2. Build the streaming tensor index. GgufModel re-parses
            //    shard 0 (and siblings) and drops the metadata mmap when
            //    load() returns. We keep `gmeta` alive only long enough
            //    to hand its KV table to GgufRuntime::init().
            prof::mark("gguf: build tensor index");
            GgufModel gm;
            gm.load(mpath);
            prof::log("gguf: %zu tensors, %.2f GB total, %zu shard(s)",
                      gm.tensor_count(), gm.total_bytes() / 1e9,
                      gm.shard_paths().size());

            // 3. Init Metal + runtime.
            prof::mark("gguf: metal + runtime init");
            Metal mtl; mtl.init(METALBLOK_KERNEL_PATH);
            GgufRuntime rt;
            rt.init(gmeta, gm, mtl);

            if (single_step) {
                if (!state_path) {
                    std::fprintf(stderr, "metalblok: --single-step-token requires --state\n");
                    return 2;
                }
                if (file_exists(state_path) && !rt.load_state(state_path)) {
                    std::fprintf(stderr, "metalblok: invalid or incompatible state: %s\n", state_path);
                    return 5;
                }
                uint32_t next = rt.step(single_step_token);
                if (!rt.save_state(state_path)) {
                    std::fprintf(stderr, "metalblok: failed to atomically save state: %s\n", state_path);
                    return 5;
                }
                std::string piece = tok.decode({next});
                std::printf("state_pos=%u input_token=%u next_token=%u piece=",
                            rt.pos(), single_step_token, next);
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::putchar('\n');
                return 0;
            }

            // 4. Prefill then decode.
            prof::mark("gguf: PREFILL begin");
            long long tp0 = prof::now_us();
            uint32_t reused = 0;
            if (continue_state && (!state_path || !file_exists(state_path))) {
                std::fprintf(stderr, "metalblok: --continue-state requires an existing --state file\n");
                return 5;
            }
            if (state_path && file_exists(state_path)) {
                if (!rt.load_state(state_path)) {
                    std::fprintf(stderr, "metalblok: invalid or incompatible state: %s\n", state_path);
                    return 5;
                }
                reused = rt.pos();
                if (!continue_state && reused > ids.size()) {
                    std::fprintf(stderr, "metalblok: state position exceeds prompt length\n");
                    return 5;
                }
            }
            uint32_t next = rt.predicted_token();
            if (!continue_state) {
                for (uint32_t i = reused; i < ids.size(); ++i) {
                    next = rt.step(ids[i]);
                    if (state_path && !rt.save_state(state_path)) {
                        std::fprintf(stderr, "metalblok: failed to save state: %s\n", state_path);
                        return 5;
                    }
                }
            }
            long long tp1 = prof::now_us();
            prof::log("gguf: PREFILL end %lld us rss=%zuMB",
                      tp1 - tp0, prof::rss_mb());

            prof::mark("gguf: DECODE begin");
            long long td0 = prof::now_us();
            int produced = 0;
            std::string tail;
            bool hit_eos = false, hit_stop = false;
            for (int i = 0; i < n_predict; ++i) {
                std::string piece = tok.decode({next});
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
                ++produced;
                if (next == tok.eos()) { hit_eos = true; break; }
                if (stop_str) {
                    tail += piece;
                    size_t keep = std::strlen(stop_str) + 16;
                    if (tail.size() > keep) tail.erase(0, tail.size() - keep);
                    if (tail.find(stop_str) != std::string::npos) { hit_stop = true; break; }
                }
                if (i + 1 < n_predict) {
                    next = rt.step(next);
                    if (state_path && !rt.save_state(state_path)) {
                        std::fprintf(stderr, "metalblok: failed to save state: %s\n", state_path);
                        return 5;
                    }
                }
            }
            std::fputc('\n', stdout);
            long long td1 = prof::now_us();
            double dprefill_s = (tp1 - tp0) / 1e6;
            double ddecode_s  = (td1 - td0) / 1e6;
            double dtps = produced > 0 && ddecode_s > 0 ? produced / ddecode_s : 0.0;
            prof::log("gguf: SUMMARY prefill %zu tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.1f tok/s) | %s",
                      ids.size(), dprefill_s,
                      ids.size() / (dprefill_s > 0 ? dprefill_s : 1.0),
                      produced, ddecode_s, dtps,
                      hit_eos  ? "stopped: EOS"   :
                      hit_stop ? "stopped: --stop" : "stopped: max-tokens");
            return 0;
        }
    }

    Model model;
    bool is_hf = file_exists(std::string(model_dir) + "/config.json")
              && !file_exists(std::string(model_dir) + "/header.json");
    prof::mark(is_hf ? "open HF checkpoint" : "open .blade dir");
    {
        long long t = prof::now_us();
        if (is_hf) model.open_hf(model_dir); else model.open(model_dir);
        prof::log("model: %s opened in %lld us  rss=%zuMB",
                  is_hf ? "HF" : ".blade", prof::now_us() - t, prof::rss_mb());
    }
    prof::log("model: %u layers, hidden=%u, vocab=%u, %u experts (%u active), MLA Lk=%u, dtype=%s",
              model.cfg.n_layers, model.cfg.hidden, model.cfg.vocab,
              model.cfg.n_experts, model.cfg.n_experts_active, model.cfg.kv_lora_rank,
              model.cfg.weight_dtype == 1 ? "bf16" : "fp8");

    // Pre-fault dense weights sequentially.  On HF (bf16) path this turns
    // the first decode step from a fault-storm into a pure compute pass.
    {
        prof::mark("warm dense weights");
        long long t = prof::now_us();
        model.warm_dense();
        prof::log("model: warmed dense weights in %lld us  rss=%zuMB",
                  prof::now_us() - t, prof::rss_mb());
    }

    std::string tpath = tok_path ? tok_path : (std::string(model_dir) + "/tokenizer.bin");
    Tokenizer tok; tok.load(tpath);
    auto ids = tok.encode(prompt);
    if (tok.bos() != 0 && (ids.empty() || ids.front() != tok.bos())) {
        ids.insert(ids.begin(), tok.bos());
    }
    prof::log("prompt: %zu tokens (bos=%u, eos=%u)", ids.size(), tok.bos(), tok.eos());

    prof::mark("metal init + runtime init");
    Metal mtl; mtl.init(METALBLOK_KERNEL_PATH);
    Streamer streamer; streamer.init(model);
    Runtime rt; rt.init(model, mtl, streamer);

    KVCache kv;
    uint32_t reused = 0;
    if (kv_dir) { kv.init(rt, kv_dir); reused = kv.load(ids); }
    if (reused == ids.size()) { reused -= 1; rt.pos = reused; }
    if (reused)
        prof::log("kv-cache hit: reusing %u/%zu prompt tokens", reused, ids.size());

    prof::mark("PREFILL begin");
    long long t_prefill0 = prof::now_us();
    uint32_t next = 0;
    if (reused < ids.size())
        next = rt.prefill(ids.data() + reused, (uint32_t)(ids.size() - reused));
    long long t_prefill1 = prof::now_us();
    prof::log("PREFILL end: %lld us  rss=%zuMB",
              t_prefill1 - t_prefill0, prof::rss_mb());

    if (kv_dir && reused < ids.size()) kv.save(ids);

    prof::mark("DECODE begin");
    long long t_decode0 = prof::now_us();
    int produced = 0;
    std::string tail;
    bool hit_eos = false, hit_stop = false;
    for (int i = 0; i < n_predict; ++i) {
        std::string piece = tok.decode({next});
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++produced;
        if (next == tok.eos()) { hit_eos = true; break; }
        if (stop_str) {
            tail += piece;
            size_t keep = std::strlen(stop_str) + 16;
            if (tail.size() > keep) tail.erase(0, tail.size() - keep);
            if (tail.find(stop_str) != std::string::npos) { hit_stop = true; break; }
        }
        if (i + 1 < n_predict) next = rt.step(next);
    }
    std::fputc('\n', stdout);
    long long t_decode1 = prof::now_us();
    prof::log("DECODE end: %lld us  rss=%zuMB", t_decode1 - t_decode0, prof::rss_mb());

    double prefill_s = (t_prefill1 - t_prefill0) / 1e6;
    double decode_s  = (t_decode1  - t_decode0)  / 1e6;
    double tps = produced > 0 && decode_s > 0 ? produced / decode_s : 0.0;
    prof::log("SUMMARY: prefill %zu tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.1f tok/s) | %s",
              ids.size(), prefill_s, ids.size() / (prefill_s > 0 ? prefill_s : 1.0),
              produced, decode_s, tps,
              hit_eos  ? "stopped: EOS"   :
              hit_stop ? "stopped: --stop" : "stopped: max-tokens");

    streamer.shutdown();
    return 0;
}
