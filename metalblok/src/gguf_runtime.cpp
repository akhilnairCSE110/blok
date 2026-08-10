// src/gguf_runtime.cpp
// ---------------------------------------------------------------------------
// GgufRuntime implementation. See gguf_runtime.hpp for context, memory
// budget, and the per-tensor streaming primitive.
//
// Anti-drift: this file does NOT include runtime.hpp / mla.cpp / moe.cpp.
// The decode loop is implemented independently against the same Metal
// kernels they use (rms_norm_f16, mla_q_split_rope, mla_kv_split_rope,
// mla_attn_decode_f16, swiglu_f16, router_topk_f16, axpy_f16, argmax_f16,
// and the gemv_<type>_f16 family).
//
// Conventions inherited from ops.hpp:
//   TG_GEMV = 1024 (used by gemv_f16_f16 path -- resident activations)
//   TG_RED  = 256  (rms_norm, mla_kv_split_rope, mla_attn_decode_f16)
//   TG_ELT  = 256  (axpy, swiglu, mla_q_split_rope)
// For GGUF k-quant / i-quant gemv we use TG=128, matching validate_gemv in
// main.cpp (already a validated path -- changing it would re-open numerics).
// ---------------------------------------------------------------------------
#include "gguf_runtime.hpp"

#include "gguf_dequant.hpp"
#include "gguf_kernels.hpp"
#include "ops.hpp"
#include "prof.hpp"
#include "memstat.hpp"

#include "../vendor/llama_cpp/iq1s_grid.h"
#include "../vendor/llama_cpp/iq2xxs_grid.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>

namespace blade {

namespace {

[[noreturn]] void die(const char* m) {
    std::fprintf(stderr, "gguf_runtime: %s\n", m);
    std::abort();
}

// Convert one f32 to IEEE-754 binary16 (round-to-nearest, ties-to-even).
// Used for resident-norm conversion; not on the per-token hot path.
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp >= 31) {
        if (mant == 0 && ((x >> 23) & 0xff) != 0xff) return uint16_t(sign | 0x7c00);
        return uint16_t(sign | 0x7c00 | (mant ? 1 : 0));
    }
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = (mant >> (shift - 1)) & 1;
        uint32_t out = (mant >> shift) + round_bit;
        return uint16_t(sign | out);
    }
    uint32_t round_bit = (mant >> 12) & 1;
    uint32_t out_mant  = (mant + (round_bit ? 0x1000 : 0)) >> 13;
    return uint16_t(sign | ((uint32_t)exp << 10) | (out_mant & 0x3ff));
}

// Round n_elems up to a whole number of 256-element super-blocks. Most
// dequant-friendly K dims are already multiples of 256; the rest abort.
inline uint32_t require_block_aligned(uint32_t K, const char* what) {
    if (K == 0 || (K & 255u) != 0u) {
        std::fprintf(stderr, "gguf_runtime: %s K=%u not multiple of 256\n", what, K);
        std::abort();
    }
    return K / 256u;
}

// Pick the per-block byte count and CPU dequant function for one ggml type.
struct CpuDqInfo {
    uint32_t bpb       = 0;        // bytes / 256-weight block
    void   (*fn)(const void*, float*) = nullptr;
    bool     is_f32    = false;    // f32 tensors are passed through verbatim
    bool     is_f16    = false;    // f16 tensors are reinterpreted
};
inline CpuDqInfo cpu_dq_info(uint32_t t) {
    CpuDqInfo r{};
    switch (t) {
        case GGML_F32:     r.is_f32 = true;                                 break;
        case GGML_F16:     r.is_f16 = true;                                 break;
        case GGML_Q4_K:    r.bpb=144; r.fn=&dequantize_q4_K_block;          break;
        case GGML_Q5_K:    r.bpb=176; r.fn=&dequantize_q5_K_block;          break;
        case GGML_Q6_K:    r.bpb=210; r.fn=&dequantize_q6_K_block;          break;
        case GGML_IQ1_S:   r.bpb= 50; r.fn=&dequantize_iq1_s_block;         break;
        case GGML_IQ2_XXS: r.bpb= 66; r.fn=&dequantize_iq2_xxs_block;       break;
        default: break;
    }
    return r;
}

// Compute the per-row payload size for a 2D quantized weight: K cols of a
// type with `bpb` bytes per 256-element block.
inline uint64_t bytes_per_row(uint32_t K, uint32_t bpb_or_zero, uint32_t ggml_type) {
    if (ggml_type == GGML_F32) return (uint64_t)K * 4;
    if (ggml_type == GGML_F16) return (uint64_t)K * 2;
    return (uint64_t)(K / 256u) * (uint64_t)bpb_or_zero;
}

} // namespace

// ---------------------------------------------------------------------------
// init / config / verify (cold path)
// ---------------------------------------------------------------------------

void GgufRuntime::init(const Gguf& g, GgufModel& gm, Metal& mtl) {
    g_   = &g;
    gm_  = &gm;
    mtl_ = &mtl;
    pos_ = 0;
    parse_config_(g);
    verify_tensor_table_(gm);

    // Explicit override; the default stays at 64 for safe bring-up.
    if (const char* s = std::getenv("METALBLOK_MAX_SEQ")) {
        unsigned long v = std::strtoul(s, nullptr, 10);
        if (v >= 64 && v <= 65536) cfg_.max_seq = (uint32_t)v;
    }
    if (cfg_.max_seq > cfg_.context_length && cfg_.context_length > 0)
        cfg_.max_seq = cfg_.context_length;

    // Full pre-allocation ledger. Absorbed MLA is retained lazily but reaches
    // this exact total after all layers. The largest transient is the untied
    // output head. Keep 3 GiB reclaimable host headroom on this 24 GB target.
    const uint64_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint64_t absorbed = uint64_t(cfg_.n_layers) * cfg_.n_heads *
        cfg_.kv_lora_rank * (Dn + cfg_.value_length) * 2ULL;
    const uint64_t kv = uint64_t(cfg_.n_layers) * cfg_.max_seq *
        (cfg_.kv_lora_rank + cfg_.rope_dim) * 2ULL;
    const uint64_t scores = uint64_t(cfg_.n_heads) * cfg_.max_seq * 4ULL;
    const auto* output = gm.find("output.weight");
    const uint64_t largest_transient = output ? output->nbytes : 0;
    const uint64_t runtime_margin = 256ULL << 20;
    const uint64_t estimated = absorbed + kv + scores + largest_transient +
                               runtime_margin;
    const uint64_t host_reserve = 3ULL << 30;
    const auto memory = mem::snapshot();
    prof::log("memory-ledger: absorbed=%.2fGB kv=%.2fMB transient=%.2fMB "
              "margin=%.2fMB estimated=%.2fGB available=%.2fGB reserve=%.2fGB",
              absorbed / 1e9, kv / 1e6, largest_transient / 1e6,
              runtime_margin / 1e6, estimated / 1e9,
              memory.available / 1e9, host_reserve / 1e9);
    if (memory.available < estimated + host_reserve) {
        std::fprintf(stderr,
            "metalblok: memory safety gate refused startup: need %.2f GB "
            "runtime + %.2f GB host reserve; %.2f GB available\n",
            estimated / 1e9, host_reserve / 1e9, memory.available / 1e9);
        std::exit(4);
    }

    alloc_activations_();
    alloc_kv_cache_();
    load_resident_norms_();
    build_grids_();

    prof::log("gguf_runtime: init n_layers=%u hidden=%u vocab=%u "
              "n_experts=%u active=%u kv_lora=%u rope_dim=%u max_seq=%u rss=%zuMB",
              cfg_.n_layers, cfg_.hidden, cfg_.vocab,
              cfg_.n_experts, cfg_.n_experts_active,
              cfg_.kv_lora_rank, cfg_.rope_dim, cfg_.max_seq, prof::rss_mb());
}

void GgufRuntime::parse_config_(const Gguf& g) {
    cfg_.n_layers         = g.get_u32("deepseek2.block_count");
    cfg_.hidden           = g.get_u32("deepseek2.embedding_length");
    cfg_.n_heads          = g.get_u32("deepseek2.attention.head_count");
    cfg_.q_lora_rank      = g.get_u32("deepseek2.attention.q_lora_rank");
    cfg_.kv_lora_rank     = g.get_u32("deepseek2.attention.kv_lora_rank");
    cfg_.key_length       = g.get_u32("deepseek2.attention.key_length");
    cfg_.value_length     = g.get_u32("deepseek2.attention.value_length");
    cfg_.rope_dim         = g.get_u32("deepseek2.rope.dimension_count");
    cfg_.rope_freq_base   = g.get_f32_or("deepseek2.rope.freq_base", 10000.0f);
    cfg_.n_experts        = g.get_u32("deepseek2.expert_count");
    cfg_.n_experts_active = g.get_u32("deepseek2.expert_used_count");
    cfg_.n_dense_layers   = g.get_u32("deepseek2.leading_dense_block_count");
    cfg_.expert_ffn       = g.get_u32("deepseek2.expert_feed_forward_length");
    cfg_.n_shared         = g.get_u32_or("deepseek2.expert_shared_count", 1);
    cfg_.context_length   = g.get_u32_or("deepseek2.context_length", 4096);
    cfg_.vocab            = g.get_u32("deepseek2.vocab_size");
    cfg_.rms_eps          = g.get_f32_or("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
    // YaRN attention scale. DeepSeek V2/V3/R1 softmax_scale =
    //   (1/sqrt(Dn+Dr)) * yarn_mscale
    // with    yarn_mscale = (1 + log_mult * ln(factor))^2
    // (matches model.cpp:307-324 derivation from HF rope_scaling). The GGUF
    // stores `yarn_log_multiplier` = 0.1 * mscale_all_dim and
    // `scaling.factor` directly. If either is absent, mscale stays at 1.0
    // (correct for prompts inside the trained window). $BLADE_YARN_MSCALE
    // overrides for explicit experimentation.
    cfg_.yarn_mscale = 1.0f;
    {
        float factor   = g.get_f32_or("deepseek2.rope.scaling.factor", 0.0f);
        float log_mult = g.get_f32_or("deepseek2.rope.scaling.yarn_log_multiplier", 0.0f);
        if (factor > 1.0f && log_mult > 0.0f) {
            double m = 1.0 + (double)log_mult * std::log((double)factor);
            cfg_.yarn_mscale = (float)(m * m);
        }
    }
    if (const char* s = std::getenv("BLADE_YARN_MSCALE")) {
        float v = std::strtof(s, nullptr);
        if (v > 0.0f && v < 8.0f) cfg_.yarn_mscale = v;
    }

    // MoE routing parameters. R1 declares sigmoid gating (==2), a routed
    // scaling factor of 2.5, and weight normalization. Defaults keep the
    // legacy softmax path for models that omit these keys.
    cfg_.expert_gating_func   = g.get_u32_or("deepseek2.expert_gating_func", 1);
    cfg_.expert_weights_scale = g.get_f32_or("deepseek2.expert_weights_scale", 1.0f);
    cfg_.expert_weights_norm  =
        g.get_u32_or("deepseek2.expert_weights_norm", 0) ? 1u : 0u;

    if (cfg_.n_layers == 0 || cfg_.hidden == 0 || cfg_.vocab == 0)
        die("config: zero-sized core dim");
    if (cfg_.n_dense_layers > cfg_.n_layers)
        die("config: leading_dense_block_count > block_count");
    if (cfg_.kv_lora_rank == 0 || cfg_.q_lora_rank == 0)
        die("config: zero LoRA rank (not an MLA model?)");
    if (cfg_.key_length <= cfg_.rope_dim)
        die("config: key_length <= rope_dim (no nope split)");
    if (cfg_.n_experts != 256 || cfg_.n_experts_active != 8)
        die("config: this correctness target requires DeepSeek-R1 256/top-8 routing");
    if (cfg_.n_experts % cfg_.n_expert_groups != 0)
        die("config: expert groups do not divide expert count");
}

static void require_(const GgufModel& gm, const std::string& name,
                     uint32_t expect_K, uint32_t expect_N) {
    const GgufTensorEntry* e = gm.find(name);
    if (!e) {
        std::fprintf(stderr, "gguf_runtime: tensor missing: %s\n", name.c_str());
        std::abort();
    }
    if (expect_K && e->shape[0] != expect_K) {
        std::fprintf(stderr, "gguf_runtime: %s: K=%llu expected %u\n",
                     name.c_str(), (unsigned long long)e->shape[0], expect_K);
        std::abort();
    }
    if (expect_N && e->n_dims >= 2 && e->shape[1] != expect_N) {
        std::fprintf(stderr, "gguf_runtime: %s: N=%llu expected %u\n",
                     name.c_str(), (unsigned long long)e->shape[1], expect_N);
        std::abort();
    }
}

void GgufRuntime::verify_tensor_table_(const GgufModel& gm) const {
    require_(gm, "token_embd.weight",  cfg_.hidden, cfg_.vocab);
    require_(gm, "output.weight",      cfg_.hidden, cfg_.vocab);
    require_(gm, "output_norm.weight", cfg_.hidden, 0);

    char buf[128];
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_norm.weight", L);
        require_(gm, buf, cfg_.hidden, 0);

        std::snprintf(buf, sizeof(buf), "blk.%u.attn_q_a.weight", L);
        require_(gm, buf, cfg_.hidden, cfg_.q_lora_rank);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_q_a_norm.weight", L);
        require_(gm, buf, cfg_.q_lora_rank, 0);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_q_b.weight", L);
        require_(gm, buf, cfg_.q_lora_rank, cfg_.n_heads * cfg_.key_length);

        std::snprintf(buf, sizeof(buf), "blk.%u.attn_kv_a_mqa.weight", L);
        require_(gm, buf, cfg_.hidden, cfg_.kv_lora_rank + cfg_.rope_dim);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_kv_a_norm.weight", L);
        require_(gm, buf, cfg_.kv_lora_rank, 0);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_kv_b.weight", L);
        const uint32_t nope = cfg_.key_length - cfg_.rope_dim;
        require_(gm, buf, cfg_.kv_lora_rank,
                 cfg_.n_heads * (nope + cfg_.value_length));

        std::snprintf(buf, sizeof(buf), "blk.%u.attn_output.weight", L);
        require_(gm, buf, cfg_.n_heads * cfg_.value_length, cfg_.hidden);

        std::snprintf(buf, sizeof(buf), "blk.%u.ffn_norm.weight", L);
        require_(gm, buf, cfg_.hidden, 0);

        if (L < cfg_.n_dense_layers) {
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_gate.weight", L);
            require_(gm, buf, cfg_.hidden, 0);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_up.weight", L);
            require_(gm, buf, cfg_.hidden, 0);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_down.weight", L);
            require_(gm, buf, 0, cfg_.hidden);
        } else {
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_gate_inp.weight", L);
            require_(gm, buf, cfg_.hidden, cfg_.n_experts);
            std::snprintf(buf, sizeof(buf), "blk.%u.exp_probs_b.bias", L);
            if (!gm.find(buf))
                std::fprintf(stderr, "gguf_runtime: note: %s absent\n", buf);

            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_gate_shexp.weight", L);
            require_(gm, buf, cfg_.hidden, 0);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_up_shexp.weight", L);
            require_(gm, buf, cfg_.hidden, 0);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_down_shexp.weight", L);
            require_(gm, buf, 0, cfg_.hidden);

            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_gate_exps.weight", L);
            require_(gm, buf, cfg_.hidden, cfg_.expert_ffn);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_up_exps.weight", L);
            require_(gm, buf, cfg_.hidden, cfg_.expert_ffn);
            std::snprintf(buf, sizeof(buf), "blk.%u.ffn_down_exps.weight", L);
            require_(gm, buf, cfg_.expert_ffn, cfg_.hidden);
        }
    }
    prof::log("gguf_runtime: tensor table verified (%zu tensors)",
              gm.tensor_count());
}

// ---------------------------------------------------------------------------
// Activation / KV / norm / grid setup (cold path)
// ---------------------------------------------------------------------------

void GgufRuntime::alloc_activations_() {
    auto fp16 = [&](size_t n){ return mtl_->alloc(n * 2); };
    const uint32_t H  = cfg_.hidden;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    const uint32_t Fe = cfg_.expert_ffn;
    const uint32_t Fs = Fe * cfg_.n_shared;
    // Dense FFN size: for R1 the 3 dense layers use ffn_down with N=hidden,
    // K = innermost of ffn_down (per-tensor; we don't pre-know it from cfg).
    // The activation buffers must be large enough for max(Fs, dense F).
    // Probe the dense layer to determine: ffn_gate.weight has shape[1] = F.
    uint32_t F_dense = 0;
    if (cfg_.n_dense_layers > 0) {
        const GgufTensorEntry* e = gm_->find("blk.0.ffn_gate.weight");
        if (e && e->n_dims >= 2) F_dense = (uint32_t)e->shape[1];
    }
    const uint32_t F_max = std::max(F_dense, Fs);

    x_         = fp16(H);
    x_norm_    = fp16(H);
    q_a_       = fp16(Hi);
    q_a_n_     = fp16(Hi);
    q_full_    = fp16((size_t)HE * (Dn + Dr));
    q_nope_    = fp16((size_t)HE * Dn);
    q_rope_    = fp16((size_t)HE * Dr);
    kv_a_      = fp16((size_t)Lk + Dr);
    q_eff_     = fp16((size_t)HE * Lk);
    o_lat_     = fp16((size_t)HE * Lk);
    o_full_    = fp16((size_t)HE * Dv);
    attn_out_  = fp16(H);
    ffn_gate_  = fp16(F_max);
    ffn_up_    = fp16(F_max);
    ffn_act_   = fp16(F_max);
    ffn_out_   = fp16(H);
    expert_tmp_= fp16(H);
    router_log_= fp16(cfg_.n_experts);
    router_idx_= mtl_->alloc((size_t)cfg_.n_experts_active * 4);
    router_wts_= mtl_->alloc((size_t)cfg_.n_experts_active * 4);
    logits_    = fp16(cfg_.vocab);
    next_tok_  = mtl_->alloc(4);

    // Zero bias for V3-style router fallback.
    zero_bias_ = mtl_->alloc((size_t)cfg_.n_experts * 4);
    std::memset(zero_bias_.contents, 0, (size_t)cfg_.n_experts * 4);
}

void GgufRuntime::alloc_kv_cache_() {
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Dr = cfg_.rope_dim;
    auto fp16 = [&](size_t n){ return mtl_->alloc(n * 2); };
    c_kv_   = fp16((size_t)cfg_.n_layers * cfg_.max_seq * Lk);
    k_rope_ = fp16((size_t)cfg_.n_layers * cfg_.max_seq * Dr);
    scores_ = mtl_->alloc((size_t)HE * cfg_.max_seq * 4);
    prof::log("gguf_runtime: KV cache c_kv=%.2fMB k_rope=%.2fMB scores=%.2fMB",
              c_kv_.length / 1e6, k_rope_.length / 1e6, scores_.length / 1e6);
}

uint64_t GgufRuntime::cpu_dequant_to_f16_(const GgufTensorEntry& e, uint16_t* dst) {
    const CpuDqInfo info = cpu_dq_info(e.type);
    if (!info.is_f32 && !info.is_f16 && info.fn == nullptr) {
        std::fprintf(stderr, "gguf_runtime: cpu_dequant: unsupported type %u\n", e.type);
        std::abort();
    }
    // Read payload into transient scratch.
    if (dequant_scratch_bytes_.size() < e.nbytes) dequant_scratch_bytes_.resize(e.nbytes);
    std::atomic<bool> done{false};
    gm_->ring().submit(e.shard, e.abs_offset, e.nbytes,
                       dequant_scratch_bytes_.data(), &done);
    PreadRing::wait(&done);

    const uint64_t n_elems =
        (e.n_dims >= 1 ? e.shape[0] : 1) *
        (e.n_dims >= 2 ? e.shape[1] : 1) *
        (e.n_dims >= 3 ? e.shape[2] : 1) *
        (e.n_dims >= 4 ? e.shape[3] : 1);

    if (info.is_f16) {
        // Direct copy as fp16 -> fp16.
        std::memcpy(dst, dequant_scratch_bytes_.data(), (size_t)n_elems * 2);
        return n_elems;
    }
    if (info.is_f32) {
        const float* src = (const float*)dequant_scratch_bytes_.data();
        for (uint64_t i = 0; i < n_elems; ++i) dst[i] = f32_to_f16(src[i]);
        return n_elems;
    }
    // Block-quant path: walk in 256-element groups.
    if ((n_elems & 255ull) != 0) die("cpu_dequant: total elements not multiple of 256");
    const uint64_t n_blocks = n_elems / 256ull;
    if (dequant_scratch_f32_.size() < 256) dequant_scratch_f32_.resize(256);
    float* tmp = dequant_scratch_f32_.data();
    const uint8_t* src = dequant_scratch_bytes_.data();
    for (uint64_t b = 0; b < n_blocks; ++b) {
        info.fn(src + b * info.bpb, tmp);
        for (uint32_t i = 0; i < 256; ++i) dst[b * 256 + i] = f32_to_f16(tmp[i]);
    }
    return n_elems;
}

void GgufRuntime::load_resident_norms_() {
    lw_.resize(cfg_.n_layers);
    auto load_into = [&](const std::string& name, MtlBuf& dst, uint64_t n) {
        const GgufTensorEntry* e = gm_->find(name);
        if (!e) { std::fprintf(stderr, "norm missing: %s\n", name.c_str()); std::abort(); }
        dst = mtl_->alloc(n * 2);
        cpu_dequant_to_f16_(*e, (uint16_t*)dst.contents);
    };

    char buf[128];
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_norm.weight", L);
        load_into(buf, lw_[L].attn_norm, cfg_.hidden);
        std::snprintf(buf, sizeof(buf), "blk.%u.ffn_norm.weight", L);
        load_into(buf, lw_[L].ffn_norm,  cfg_.hidden);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_q_a_norm.weight", L);
        load_into(buf, lw_[L].q_a_norm,  cfg_.q_lora_rank);
        std::snprintf(buf, sizeof(buf), "blk.%u.attn_kv_a_norm.weight", L);
        load_into(buf, lw_[L].kv_a_norm, cfg_.kv_lora_rank);
    }
    load_into("output_norm.weight", output_norm_b_, cfg_.hidden);
    prof::log("gguf_runtime: resident norms loaded (%u layers, %zu MB)",
              cfg_.n_layers,
              (size_t)(cfg_.hidden * 4 + cfg_.q_lora_rank + cfg_.kv_lora_rank)
                  * cfg_.n_layers / (1024 * 1024));
}

void GgufRuntime::build_grids_() {
    // Copy the vendored codebook arrays into Metal-owned shared-storage
    // buffers. The arrays live in read-only __DATA_CONST; wrapping those
    // pages with newBufferWithBytesNoCopy yields a buffer the GPU cannot
    // safely read (hard fault on first dispatch). A real alloc() buffer is
    // CPU+GPU coherent and writable, so we memcpy once at init.
    iq1s_grid_b_ = mtl_->alloc(sizeof(blade::vendored::iq1s_grid));
    std::memcpy(iq1s_grid_b_.contents, blade::vendored::iq1s_grid,
                sizeof(blade::vendored::iq1s_grid));
    iq2xxs_grid_b_ = mtl_->alloc(sizeof(blade::vendored::iq2xxs_grid));
    std::memcpy(iq2xxs_grid_b_.contents, blade::vendored::iq2xxs_grid,
                sizeof(blade::vendored::iq2xxs_grid));
}

// ---------------------------------------------------------------------------
// Absorbed MLA: build per-head W_uk / W_uv from attn_kv_b, lazily per layer.
//
// attn_kv_b in R1 is q4_K with shape [Lk=512, HE*(nope+Dv)=128*(128+128)=32768].
// llama.cpp's deepseek2 convention packs per head as [nope || v_dim] along
// the N axis, contiguous across heads.
//
// Absorbed form:
//   W_uk[h, k, dn]  -- for each head h, the nope-half projection from latent k
//                      to nope-dim dn. Shape [HE, Lk, Dn]. Sourced from rows
//                      [0..nope) of head h's slice of attn_kv_b (transposed:
//                      attn_kv_b[k, h*(nope+dv) + dn]).
//   W_uv[h, dv, k]  -- per-head value-up projection. Shape [HE, Dv, Lk].
//                      Sourced from rows [nope..nope+Dv) of head h's slice,
//                      transposed so K is innermost (matches gemv layout
//                      "row reads K contiguous elements").
//
// Layout for gemv_f16_f16:
//   W_uk: stored row-major [HE*Lk rows, Dn cols]. Row r=(h*Lk + k) holds
//         entries for head h, latent k, varying over nope dim.
//         Grouped: group_size = Lk so all rows of one head share the same
//         q_nope[h*Dn..(h+1)*Dn) input slice.
//   W_uv: stored row-major [HE*Dv rows, Lk cols]. Row r=(h*Dv + dv) holds
//         entries for head h, value dim dv, varying over latent k.
//         Grouped: group_size = Dv so all rows of one head share the same
//         o_lat[h*Lk..(h+1)*Lk) input slice.
//
// Build: CPU-dequant attn_kv_b into a [Lk * HE * (nope+Dv)] f32 buffer
// (innermost = Lk by ggml convention, then we read it as src[k, n] where
// n = h*(nope+Dv) + (offset in head)). Transpose into the row-major shapes
// described above.
// ---------------------------------------------------------------------------
void GgufRuntime::build_absorbed_kv_b_(uint32_t L) {
    if (lw_[L].absorbed_built) return;

    const uint32_t HE   = cfg_.n_heads;
    const uint32_t Lk   = cfg_.kv_lora_rank;
    const uint32_t Dn   = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dv   = cfg_.value_length;
    const uint32_t Nper = Dn + Dv;
    const uint32_t Ntot = HE * Nper;

    char nbuf[64];
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_kv_b.weight", L);
    const GgufTensorEntry* e = gm_->find(nbuf);
    if (!e) { std::fprintf(stderr, "missing %s\n", nbuf); std::abort(); }

    // 1. CPU-dequant the whole tensor into an f32 scratch.
    // GGUF convention: ne[0] is innermost.  attn_kv_b has shape [Lk, Ntot]
    // meaning the WEIGHT is [N rows, K cols] with K=Lk (rows dotted against
    // the Lk-length latent input).  Flat layout therefore is
    //   src_flat[n * Lk + k]
    // for output row n (n = h*Nper + d) and latent column k.
    std::vector<float> src_f32((size_t)Lk * Ntot);
    {
        // Reuse cpu_dequant_to_f16_ shape but materialize as f32.
        if (dequant_scratch_bytes_.size() < e->nbytes)
            dequant_scratch_bytes_.resize(e->nbytes);
        std::atomic<bool> done{false};
        gm_->ring().submit(e->shard, e->abs_offset, e->nbytes,
                           dequant_scratch_bytes_.data(), &done);
        PreadRing::wait(&done);

        const CpuDqInfo info = cpu_dq_info(e->type);
        if (info.fn == nullptr) {
            std::fprintf(stderr, "attn_kv_b unexpected type %u\n", e->type);
            std::abort();
        }
        const uint64_t total = (uint64_t)Lk * Ntot;
        if ((total & 255ull) != 0) die("attn_kv_b: not block-aligned");
        const uint64_t n_blocks = total / 256ull;
        std::vector<float> tmp(256);
        const uint8_t* sb = dequant_scratch_bytes_.data();
        for (uint64_t b = 0; b < n_blocks; ++b) {
            info.fn(sb + b * info.bpb, tmp.data());
            std::memcpy(src_f32.data() + b * 256, tmp.data(), 256 * sizeof(float));
        }
    }

    // 2. Allocate the two device buffers and fill via transpose.
    lw_[L].w_uk_b = mtl_->alloc((size_t)HE * Lk * Dn * 2);
    lw_[L].w_uv_b = mtl_->alloc((size_t)HE * Dv * Lk * 2);
    uint16_t* w_uk = (uint16_t*)lw_[L].w_uk_b.contents;
    uint16_t* w_uv = (uint16_t*)lw_[L].w_uv_b.contents;

    // Source: W_kv_b[h*(Dn+Dv)+d, k] = src_f32[(h*Nper + d) * Lk + k]
    //
    // Standard MLA absorption (no transpose; what we want is the row that
    // pairs each (h, k) with its nope-d coefficient, since q_eff[h,k] dots
    // q_nope[h, :Dn] with W_kv_b[h, dn=0..Dn, k]):
    //   W_uk[h, k, dn] = W_kv_b[h, dn, k] = src_f32[(h*Nper + dn) * Lk + k]
    //   W_uv[h, dv, k] = W_kv_b[h, Dn+dv, k] = src_f32[(h*Nper + Dn+dv) * Lk + k]
    //
    // Output layouts (row-major for gemv_f16_f16):
    //   w_uk_b[ (h*Lk + k) * Dn + dn ]
    //   w_uv_b[ (h*Dv + dv) * Lk + k ]
    for (uint32_t h = 0; h < HE; ++h) {
        const uint32_t base = h * Nper;
        for (uint32_t k = 0; k < Lk; ++k) {
            uint16_t* dst_uk = w_uk + ((size_t)h * Lk + k) * Dn;
            for (uint32_t dn = 0; dn < Dn; ++dn) {
                dst_uk[dn] = f32_to_f16(
                    src_f32[(size_t)(base + dn) * Lk + k]);
            }
        }
        for (uint32_t dv = 0; dv < Dv; ++dv) {
            uint16_t* dst_uv = w_uv + ((size_t)h * Dv + dv) * Lk;
            for (uint32_t k = 0; k < Lk; ++k) {
                dst_uv[k] = f32_to_f16(
                    src_f32[(size_t)(base + Dn + dv) * Lk + k]);
            }
        }
    }

    lw_[L].absorbed_built = true;
    prof::log("gguf_runtime: layer %u absorbed w_uk/w_uv built (%.2f MB)",
              L, (lw_[L].w_uk_b.length + lw_[L].w_uv_b.length) / 1e6);
}

// ---------------------------------------------------------------------------
// Streaming primitives (hot path)
// ---------------------------------------------------------------------------

void GgufRuntime::stream_gemv_(const std::string& name,
                               const MtlBuf& bX, const MtlBuf& bY,
                               uint32_t K, uint32_t N,
                               uint32_t slice_idx, uint64_t slice_stride)
{
    const GgufTensorEntry* e = gm_->find(name);
    if (!e) { std::fprintf(stderr, "stream_gemv: missing %s\n", name.c_str()); std::abort(); }

    const char* kn = kernel_name_for(e->type);
    if (!kn) { std::fprintf(stderr, "stream_gemv: type %u unsupported\n", e->type); std::abort(); }

    // Determine the slice byte count + offset.
    uint64_t off = e->abs_offset;
    uint64_t nbytes_slice = e->nbytes;
    if (slice_stride > 0) {
        off          += (uint64_t)slice_idx * slice_stride;
        nbytes_slice  = slice_stride;
    }

    // Allocate transient device buffer + pread directly into it.
    MtlBuf bW = mtl_->alloc(nbytes_slice);
    std::atomic<bool> done{false};
    gm_->ring().submit(e->shard, off, nbytes_slice, bW.contents, &done);
    PreadRing::wait(&done);

    std::vector<MtlBuf> bufs{ bW, bX, bY };
    if (e->type == GGML_IQ1_S)   bufs.push_back(iq1s_grid_b_);
    if (e->type == GGML_IQ2_XXS) bufs.push_back(iq2xxs_grid_b_);

    const uint32_t Ku = K;
    const uint32_t TG = 128;
    mtl_->begin();
    mtl_->dispatch(kn, bufs, { { &Ku, sizeof(Ku) } }, N, TG, true);
    mtl_->commit_and_wait();
    mtl_->release(bW);
}

void GgufRuntime::grouped_gemv_f16_(const MtlBuf& bW, const MtlBuf& bX, const MtlBuf& bY,
                                    uint32_t N, uint32_t K, uint32_t group)
{
    // Dispatch in its own cmdbuf -- the streaming gemv's also commit, so
    // we keep one-dispatch-per-commit ordering consistent.
    mtl_->begin();
    mtl_->dispatch("gemv_f16_f16", { bW, bX, bY },
                   { { &K, 4 }, { &group, 4 } },
                   N, TG_GEMV, true);
    mtl_->commit_and_wait();
}

// ---------------------------------------------------------------------------
// Embedding lookup: pread + dequant one row of token_embd.
// ---------------------------------------------------------------------------
void GgufRuntime::embed_lookup_(uint32_t tok) {
    const GgufTensorEntry* e = gm_->find("token_embd.weight");
    if (!e) die("token_embd missing");
    if (tok >= cfg_.vocab) die("token id out of range");

    const uint32_t H = cfg_.hidden;
    const uint64_t per_row = bytes_per_row(H, bytes_per_block(e->type), e->type);
    const uint64_t off     = e->abs_offset + (uint64_t)tok * per_row;

    if (dequant_scratch_bytes_.size() < per_row) dequant_scratch_bytes_.resize(per_row);
    std::atomic<bool> done{false};
    gm_->ring().submit(e->shard, off, per_row, dequant_scratch_bytes_.data(), &done);
    PreadRing::wait(&done);

    uint16_t* dst = (uint16_t*)x_.contents;
    const CpuDqInfo info = cpu_dq_info(e->type);
    if (info.is_f16) {
        std::memcpy(dst, dequant_scratch_bytes_.data(), (size_t)H * 2);
    } else if (info.is_f32) {
        const float* src = (const float*)dequant_scratch_bytes_.data();
        for (uint32_t i = 0; i < H; ++i) dst[i] = f32_to_f16(src[i]);
    } else if (info.fn) {
        require_block_aligned(H, "token_embd");
        const uint32_t n_blocks = H / 256;
        if (dequant_scratch_f32_.size() < 256) dequant_scratch_f32_.resize(256);
        float* tmp = dequant_scratch_f32_.data();
        const uint8_t* src = dequant_scratch_bytes_.data();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            info.fn(src + b * info.bpb, tmp);
            for (uint32_t i = 0; i < 256; ++i) dst[b * 256 + i] = f32_to_f16(tmp[i]);
        }
    } else {
        die("token_embd: unsupported type");
    }
}

// ---------------------------------------------------------------------------
// MLA decode attention for ONE token at the GGUF path. Mirrors mla.cpp
// Runtime::mla_attn (mla.cpp:225-256) but every weight gemv streams the
// quantized weight from disk on demand; the absorbed per-head W_uk/W_uv
// are taken from the resident f16 cache built by build_absorbed_kv_b_().
// ---------------------------------------------------------------------------
void GgufRuntime::mla_attn_(uint32_t L) {
    build_absorbed_kv_b_(L);
    const auto& w  = lw_[L];
    const uint32_t H  = cfg_.hidden;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    uint32_t pos_u = pos_, L_u = L, ms = cfg_.max_seq, T = pos_ + 1;
    float th  = cfg_.rope_freq_base;
    float eps = cfg_.rms_eps;
    float scale = (1.0f / std::sqrt((float)(Dn + Dr))) * cfg_.yarn_mscale;

    char nbuf[64];

    // attn_norm * x -> x_norm   (resident gain)
    mtl_->begin();
    rmsnorm(*mtl_, x_, w.attn_norm, x_norm_, H, eps);
    mtl_->commit_and_wait();

    // Q LoRA: q_a = W_q_a @ x_norm; q_a_n = rmsnorm(q_a, q_a_norm);
    // q_full = W_q_b @ q_a_n
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_q_a.weight", L);
    stream_gemv_(nbuf, x_norm_, q_a_, H, Hi);

    mtl_->begin();
    rmsnorm(*mtl_, q_a_, w.q_a_norm, q_a_n_, Hi, eps);
    mtl_->commit_and_wait();

    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_q_b.weight", L);
    stream_gemv_(nbuf, q_a_n_, q_full_, Hi, HE * (Dn + Dr));

    // Split Q into nope / rope halves; rope half rotates by `pos`.
    // GGUF (llama.cpp/Unsloth) stores rope dims in NEOX split-half layout,
    // so use the _neox variant (NOT the interleaved .blade kernel).
    mtl_->begin();
    mtl_->dispatch("mla_q_split_rope_neox", { q_full_, q_nope_, q_rope_ },
                   { {&Dn,4},{&Dr,4},{&pos_u,4},{&th,4} },
                   HE, TG_ELT, true);
    mtl_->commit_and_wait();

    // Absorbed Q->latent: q_eff = W_uk @ q_nope  (per-head grouped GEMV).
    grouped_gemv_f16_(w.w_uk_b, q_nope_, q_eff_, HE * Lk, Dn, Lk);

    // KV-A MQA: kv_a = W_kv_a @ x_norm  -> split + norm + rope into KV cache.
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_kv_a_mqa.weight", L);
    stream_gemv_(nbuf, x_norm_, kv_a_, H, Lk + Dr);

    mtl_->begin();
    mtl_->dispatch("mla_kv_split_rope_neox",
                   { kv_a_, w.kv_a_norm, c_kv_, k_rope_ },
                   { {&Lk,4},{&Dr,4},{&L_u,4},{&pos_u,4},{&ms,4},{&eps,4},{&th,4} },
                   1, TG_RED, true);
    mtl_->commit_and_wait();

    // Score + softmax + value: o_lat[h, :Lk] = attn(q_eff, q_rope, c_kv, k_rope).
    mtl_->begin();
    mtl_->dispatch("mla_attn_decode_f16",
                   { q_eff_, q_rope_, c_kv_, k_rope_, o_lat_, scores_ },
                   { {&HE,4},{&Lk,4},{&Dr,4},{&L_u,4},{&T,4},{&ms,4},{&scale,4} },
                   HE, TG_RED, true);
    mtl_->commit_and_wait();

    // Absorbed latent->value: o_full = W_uv @ o_lat (per-head grouped GEMV).
    grouped_gemv_f16_(w.w_uv_b, o_lat_, o_full_, HE * Dv, Lk, Dv);

    // attn_out = W_o @ o_full
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_output.weight", L);
    stream_gemv_(nbuf, o_full_, attn_out_, HE * Dv, H);
}

// ---------------------------------------------------------------------------
// Dense FFN (R1: layers 0..n_dense_layers-1, i.e. layers 0..2).
// ---------------------------------------------------------------------------
void GgufRuntime::ffn_dense_(uint32_t L) {
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden;

    // Probe F from blk.L.ffn_gate.weight.shape[1].
    char nbuf[64];
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate.weight", L);
    const GgufTensorEntry* eg = gm_->find(nbuf);
    if (!eg || eg->n_dims < 2) die("ffn_gate shape");
    const uint32_t F = (uint32_t)eg->shape[1];

    mtl_->begin();
    rmsnorm(*mtl_, x_, w.ffn_norm, x_norm_, H, cfg_.rms_eps);
    mtl_->commit_and_wait();

    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate.weight", L);
    stream_gemv_(nbuf, x_norm_, ffn_gate_, H, F);
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_up.weight", L);
    stream_gemv_(nbuf, x_norm_, ffn_up_,   H, F);

    mtl_->begin();
    swiglu(*mtl_, ffn_gate_, ffn_up_, ffn_act_, F);
    mtl_->commit_and_wait();

    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_down.weight", L);
    stream_gemv_(nbuf, ffn_act_, ffn_out_, F, H);
}

// ---------------------------------------------------------------------------
// MoE FFN (R1: layers n_dense_layers..n_layers-1).
//
// Routing mode for R1 (V3 family, norm_topk_prob=true) = 0 in router_topk_f16.
// If the per-expert bias `exp_probs_b.bias` is present we bind it; otherwise
// the kernel sees zero_bias_ and the same softmax+norm path is exercised.
// ---------------------------------------------------------------------------
void GgufRuntime::ffn_moe_(uint32_t L) {
    const auto& w = lw_[L];
    const uint32_t H  = cfg_.hidden;
    const uint32_t Fe = cfg_.expert_ffn;
    const uint32_t Fs = Fe * cfg_.n_shared;
    const uint32_t K  = cfg_.n_experts_active;
    const uint32_t Ne = cfg_.n_experts;

    char nbuf[64];

    mtl_->begin();
    rmsnorm(*mtl_, x_, w.ffn_norm, x_norm_, H, cfg_.rms_eps);
    mtl_->commit_and_wait();

    // Defensive zero: routed experts accumulate into ffn_out_ via axpy. If
    // n_shared > 0 the shared-expert down-projection overwrites this, but we
    // zero unconditionally so the path is correct for any future n_shared=0
    // config or partial shared-expert failure.
    std::memset(ffn_out_.contents, 0, (size_t)H * 2);

    if (cfg_.n_shared > 0) {
        // Shared expert: gate, up, swiglu, down. Result lands in ffn_out_.
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate_shexp.weight", L);
        stream_gemv_(nbuf, x_norm_, ffn_gate_, H, Fs);
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_up_shexp.weight", L);
        stream_gemv_(nbuf, x_norm_, ffn_up_,   H, Fs);
        mtl_->begin();
        swiglu(*mtl_, ffn_gate_, ffn_up_, ffn_act_, Fs);
        mtl_->commit_and_wait();
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_down_shexp.weight", L);
        stream_gemv_(nbuf, ffn_act_, ffn_out_, Fs, H);
    }

    // Router: logits = router_inp @ x_norm.
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate_inp.weight", L);
    stream_gemv_(nbuf, x_norm_, router_log_, H, Ne);

    // Bias buffer for the router.
    MtlBuf bias_buf = zero_bias_;
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.exp_probs_b.bias", L);
    const GgufTensorEntry* be = gm_->find(nbuf);
    MtlBuf bias_owned{};
    bool bias_is_owned = false;
    if (be) {
        // Bias is f32 [Ne]; CPU-load once per dispatch (one-time per layer).
        if (be->type != GGML_F32) {
            std::fprintf(stderr, "exp_probs_b.bias not f32 (%u)\n", be->type);
            std::abort();
        }
        if (be->nbytes != (uint64_t)Ne * 4) die("exp_probs_b.bias size");
        bias_owned = mtl_->alloc(be->nbytes);
        std::atomic<bool> done{false};
        gm_->ring().submit(be->shard, be->abs_offset, be->nbytes,
                           bias_owned.contents, &done);
        PreadRing::wait(&done);
        bias_buf = bias_owned;
        bias_is_owned = true;
    }

    uint32_t mode = 0u;   // R1: V3 norm_topk_prob path
    mtl_->begin();
    if (cfg_.expert_gating_func == 2) {
        // R1 sigmoid gating: prob=sigmoid(logit), select by prob+bias,
        // weights = normalized sigmoid probs * routed_scaling_factor.
        float scale = cfg_.expert_weights_scale;
        uint32_t norm = cfg_.expert_weights_norm;
        const uint32_t groups = cfg_.n_expert_groups;
        const uint32_t top_groups = cfg_.n_limited_groups;
        mtl_->dispatch("router_topk_grouped_sigmoid_f16",
                       { router_log_, bias_buf, router_idx_, router_wts_ },
                       { {&Ne,4},{&K,4},{&groups,4},{&top_groups,4},
                         {&scale,4},{&norm,4} },
                       1, 1, true);
    } else {
        mtl_->dispatch("router_topk_f16",
                       { router_log_, bias_buf, router_idx_, router_wts_ },
                       { {&Ne,4},{&K,4},{&mode,4} },
                       1, 1, true);
    }
    mtl_->commit_and_wait();   // make router_idx_/router_wts_ host-visible
    if (bias_is_owned) mtl_->release(bias_owned);

    const uint32_t* idxs = (const uint32_t*)router_idx_.contents;
    const float*    wts  = (const float*)   router_wts_.contents;

    // Per-expert stride for the stacked exps tensors. shape[0]=K (=H or Fe),
    // shape[1]=N (=Fe or H), shape[2]=Ne. nbytes/Ne = one expert slice.
    auto per_expert_stride = [&](const std::string& name) -> uint64_t {
        const GgufTensorEntry* e = gm_->find(name);
        if (!e) { std::fprintf(stderr, "missing %s\n", name.c_str()); std::abort(); }
        if (e->n_dims != 3 || e->shape[2] != Ne) {
            std::fprintf(stderr, "%s not 3D or wrong n_experts\n", name.c_str());
            std::abort();
        }
        if ((e->nbytes % Ne) != 0) die("stacked exps nbytes not divisible by Ne");
        return e->nbytes / Ne;
    };
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate_exps.weight", L);
    const uint64_t st_gate = per_expert_stride(nbuf);
    std::string n_gate(nbuf);
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_up_exps.weight", L);
    const uint64_t st_up   = per_expert_stride(nbuf);
    std::string n_up(nbuf);
    std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_down_exps.weight", L);
    const uint64_t st_down = per_expert_stride(nbuf);
    std::string n_down(nbuf);

    for (uint32_t k = 0; k < K; ++k) {
        uint32_t e = idxs[k];
        float    a = wts[k];

        stream_gemv_(n_gate, x_norm_, ffn_gate_, H,  Fe, e, st_gate);
        stream_gemv_(n_up,   x_norm_, ffn_up_,   H,  Fe, e, st_up);

        mtl_->begin();
        swiglu(*mtl_, ffn_gate_, ffn_up_, ffn_act_, Fe);
        mtl_->commit_and_wait();

        stream_gemv_(n_down, ffn_act_, expert_tmp_, Fe, H, e, st_down);

        mtl_->begin();
        axpy(*mtl_, ffn_out_, expert_tmp_, a, H);
        mtl_->commit_and_wait();
    }
}

// ---------------------------------------------------------------------------
// One full forward pass at the current pos_. Reused by prefill and step.
// At entry pos_ = position of THIS token in the sequence (0-based).
// On exit pos_ has been advanced by one and next_tok_.contents holds the
// argmax sample at pos_-1.
// ---------------------------------------------------------------------------
void GgufRuntime::forward_one_(uint32_t tok) {
    // 1. Embedding.
    embed_lookup_(tok);

    // 2. 61 transformer blocks. Each block:
    //      attn_out = MLA(x);  x += attn_out
    //      ffn_out  = FFN(x);  x += ffn_out
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        mla_attn_(L);
        mtl_->begin();
        axpy(*mtl_, x_, attn_out_, 1.0f, cfg_.hidden);
        mtl_->commit_and_wait();
        if (L < cfg_.n_dense_layers) ffn_dense_(L); else ffn_moe_(L);
        mtl_->begin();
        axpy(*mtl_, x_, ffn_out_, 1.0f, cfg_.hidden);
        mtl_->commit_and_wait();
    }

    // 3. Final norm + lm_head (tied to token_embd) + greedy argmax.
    mtl_->begin();
    rmsnorm(*mtl_, x_, output_norm_b_, x_norm_, cfg_.hidden, cfg_.rms_eps);
    mtl_->commit_and_wait();

    // lm_head: R1 ships a dedicated `output.weight` (NOT tied to token_embd).
    // Using token_embd here projects through the IQ1_S embedding instead of
    // the trained output head -> wrong logits -> incoherent tokens. Fall back
    // to token_embd only if output.weight is genuinely absent (tied models).
    stream_gemv_("output.weight", x_norm_, logits_, cfg_.hidden, cfg_.vocab);

    uint32_t V = cfg_.vocab;
    mtl_->begin();
    mtl_->dispatch("argmax_f16", { logits_, next_tok_ },
                   { { &V, sizeof(uint32_t) } },
                   1, 1024, true);
    mtl_->commit_and_wait();

    pos_++;
}

uint32_t GgufRuntime::prefill(const uint32_t* ids, uint32_t n) {
    if (n == 0) die("prefill: empty prompt");
    long long t0 = prof::now_us();
    for (uint32_t i = 0; i < n; ++i) {
        if (pos_ >= cfg_.max_seq) die("prefill: pos exceeded max_seq");
        forward_one_(ids[i]);
    }
    long long us = prof::now_us() - t0;
    prof::log("gguf_runtime: prefill %u tok in %lld us (%.2f tok/s) rss=%zuMB",
              n, us, n * 1e6 / (us ? us : 1), prof::rss_mb());
    return *(uint32_t*)next_tok_.contents;
}

uint32_t GgufRuntime::step(uint32_t prev_id) {
    if (pos_ >= cfg_.max_seq) die("step: pos exceeded max_seq");
    long long t0 = prof::now_us();
    forward_one_(prev_id);
    long long us = prof::now_us() - t0;
    prof::log("gguf_runtime: decode pos=%u: %lld us rss=%zuMB",
              pos_, us, prof::rss_mb());
    return *(uint32_t*)next_tok_.contents;
}

namespace {
struct StateHeader {
    char magic[8];
    uint32_t version;
    uint32_t n_layers;
    uint32_t max_seq;
    uint32_t kv_rank;
    uint32_t rope_dim;
    uint32_t pos;
    uint32_t next_token;
};
struct StateHeaderV1 {
    char magic[8];
    uint32_t version;
    uint32_t n_layers;
    uint32_t max_seq;
    uint32_t kv_rank;
    uint32_t rope_dim;
    uint32_t pos;
};
}

bool GgufRuntime::save_state(const std::string& path) const {
    if (pos_ > cfg_.max_seq) return false;
    const std::string partial = path + ".partial";
    FILE* f = std::fopen(partial.c_str(), "wb");
    if (!f) return false;
    StateHeader h{{'M','B','L','K','S','T','A','T'}, 2, cfg_.n_layers,
                  cfg_.max_seq, cfg_.kv_lora_rank, cfg_.rope_dim, pos_,
                  predicted_token()};
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    const auto* kv = static_cast<const uint16_t*>(c_kv_.contents);
    const auto* rope = static_cast<const uint16_t*>(k_rope_.contents);
    for (uint32_t l = 0; ok && l < cfg_.n_layers; ++l) {
        const uint64_t kv_off = uint64_t(l) * cfg_.max_seq * cfg_.kv_lora_rank;
        const uint64_t r_off = uint64_t(l) * cfg_.max_seq * cfg_.rope_dim;
        ok = std::fwrite(kv + kv_off, sizeof(uint16_t),
                         uint64_t(pos_) * cfg_.kv_lora_rank, f) ==
             uint64_t(pos_) * cfg_.kv_lora_rank;
        if (ok) ok = std::fwrite(rope + r_off, sizeof(uint16_t),
                                  uint64_t(pos_) * cfg_.rope_dim, f) ==
                     uint64_t(pos_) * cfg_.rope_dim;
    }
    if (ok) ok = std::fflush(f) == 0;
    if (ok) ok = ::fsync(::fileno(f)) == 0;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) { ::unlink(partial.c_str()); return false; }
    if (::rename(partial.c_str(), path.c_str()) != 0) {
        ::unlink(partial.c_str());
        return false;
    }
    return true;
}

bool GgufRuntime::load_state(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    StateHeader h{};
    StateHeaderV1 old{};
    bool ok = std::fread(&old, sizeof(old), 1, f) == 1 &&
              std::memcmp(old.magic, "MBLKSTAT", 8) == 0 &&
              (old.version == 1 || old.version == 2);
    if (ok) {
        std::memcpy(h.magic, old.magic, sizeof(old.magic));
        h.version = old.version;
        h.n_layers = old.n_layers;
        h.max_seq = old.max_seq;
        h.kv_rank = old.kv_rank;
        h.rope_dim = old.rope_dim;
        h.pos = old.pos;
        if (old.version == 2)
            ok = std::fread(&h.next_token, sizeof(h.next_token), 1, f) == 1;
    }
    ok = ok &&
              h.n_layers == cfg_.n_layers && h.max_seq == cfg_.max_seq &&
              h.kv_rank == cfg_.kv_lora_rank && h.rope_dim == cfg_.rope_dim &&
              h.pos <= cfg_.max_seq;
    auto* kv = static_cast<uint16_t*>(c_kv_.contents);
    auto* rope = static_cast<uint16_t*>(k_rope_.contents);
    for (uint32_t l = 0; ok && l < cfg_.n_layers; ++l) {
        const uint64_t kv_off = uint64_t(l) * cfg_.max_seq * cfg_.kv_lora_rank;
        const uint64_t r_off = uint64_t(l) * cfg_.max_seq * cfg_.rope_dim;
        ok = std::fread(kv + kv_off, sizeof(uint16_t),
                        uint64_t(h.pos) * cfg_.kv_lora_rank, f) ==
             uint64_t(h.pos) * cfg_.kv_lora_rank;
        if (ok) ok = std::fread(rope + r_off, sizeof(uint16_t),
                                 uint64_t(h.pos) * cfg_.rope_dim, f) ==
                     uint64_t(h.pos) * cfg_.rope_dim;
    }
    if (ok) {
        int extra = std::fgetc(f);
        ok = extra == EOF;
    }
    std::fclose(f);
    if (ok) {
        pos_ = h.pos;
        *static_cast<uint32_t*>(next_tok_.contents) = h.next_token;
    }
    return ok;
}

} // namespace blade
