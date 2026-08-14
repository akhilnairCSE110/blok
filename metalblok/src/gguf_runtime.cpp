// src/gguf_runtime.cpp
// ---------------------------------------------------------------------------
// Native GGUF/Metal implementation; independent of the legacy BF16/FP8
// runtime. Activations and reductions are FP32, exact KV storage is FP16,
// and one 32-lane SIMD-group computes each quantized output row.
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
#include <algorithm>

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

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int shift = 0;
            while ((mant & 0x400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x3ffu;
            bits = sign | uint32_t(127 - 14 - shift) << 23 | mant << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 112u) << 23 | mant << 13;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
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

inline MtlBuf view(const MtlBuf& buffer, size_t byte_offset, size_t byte_length) {
    if (byte_offset > buffer.length || byte_length > buffer.length - byte_offset)
        die("buffer view out of range");
    MtlBuf result = buffer;
    result.offset += byte_offset;
    result.length = byte_length;
    result.contents = static_cast<uint8_t*>(buffer.contents) + byte_offset;
    return result;
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
    trace_ = std::getenv("METALBLOK_TRACE") != nullptr;
    profile_layers_ = std::getenv("METALBLOK_PROFILE_LAYERS") != nullptr;
    compact_mla_ = std::getenv("METALBLOK_MLA") != nullptr;
    validate_mla_ = std::getenv("METALBLOK_VALIDATE_MLA") != nullptr;
    if (validate_mla_ && !compact_mla_) die("MLA validation requires compact MLA");
    parse_config_(g);
    verify_tensor_table_(gm);

    // The direct native binary defaults to 64; the public runner always sets
    // this from --context before initialization.
    if (const char* s = std::getenv("METALBLOK_MAX_SEQ")) {
        unsigned long v = std::strtoul(s, nullptr, 10);
        if (v >= 64 && v <= cfg_.context_length) cfg_.max_seq = (uint32_t)v;
    }
    if (cfg_.max_seq > cfg_.context_length && cfg_.context_length > 0)
        cfg_.max_seq = cfg_.context_length;

    // Full pre-allocation ledger. Default mode retains expanded per-head K/V.
    // --mla keeps the latent plus the immutable compressed KV-B tensors.
    const uint64_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint64_t kv_width = compact_mla_
        ? cfg_.kv_lora_rank + cfg_.rope_dim
        : cfg_.n_heads * (Dn + cfg_.value_length) + cfg_.rope_dim;
    const uint64_t kv = uint64_t(cfg_.n_layers) * cfg_.max_seq * kv_width * 2ULL;
    uint64_t absorbed = 0;
    if (compact_mla_) {
        char kv_name[96];
        for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
            std::snprintf(kv_name, sizeof(kv_name), "blk.%u.attn_kv_b.weight", L);
            absorbed += gm.find(kv_name)->nbytes;
        }
    }
    const uint64_t scores = uint64_t(cfg_.n_heads) * cfg_.max_seq * 4ULL;
    const auto align16k = [](uint64_t n) { return (n + 16383) & ~uint64_t(16383); };
    const uint64_t output_head = gm.find("output.weight")->nbytes;
    uint64_t layer_stage = 0;
    uint64_t fixed_payload = 0;
    uint64_t largest_transient = 0;
    char fixed_name[96];
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        uint64_t layer_bytes = 0;
        for (const char* suffix : {"attn_q_a.weight", "attn_q_b.weight",
                                   "attn_kv_a_mqa.weight", "attn_kv_b.weight",
                                   "attn_output.weight"}) {
            if (compact_mla_ && std::strcmp(suffix, "attn_kv_b.weight") == 0)
                continue;
            std::snprintf(fixed_name, sizeof(fixed_name), "blk.%u.%s", L, suffix);
            layer_bytes += align16k(gm.find(fixed_name)->nbytes);
        }
        const char* ffn_suffixes[3] = {
            L < cfg_.n_dense_layers ? "ffn_gate.weight" : "ffn_gate_shexp.weight",
            L < cfg_.n_dense_layers ? "ffn_up.weight" : "ffn_up_shexp.weight",
            L < cfg_.n_dense_layers ? "ffn_down.weight" : "ffn_down_shexp.weight",
        };
        for (const char* suffix : ffn_suffixes) {
            std::snprintf(fixed_name, sizeof(fixed_name), "blk.%u.%s", L, suffix);
            layer_bytes += align16k(gm.find(fixed_name)->nbytes);
        }
        if (L >= cfg_.n_dense_layers) {
            std::snprintf(fixed_name, sizeof(fixed_name), "blk.%u.ffn_gate_inp.weight", L);
            layer_bytes += align16k(gm.find(fixed_name)->nbytes);
            for (const char* suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                       "ffn_down_exps.weight"}) {
                std::snprintf(fixed_name, sizeof(fixed_name), "blk.%u.%s", L, suffix);
                const auto* entry = gm.find(fixed_name);
                largest_transient = std::max(largest_transient,
                                             entry->nbytes / cfg_.n_experts);
            }
        }
        layer_stage = std::max(layer_stage, layer_bytes);
        fixed_payload += layer_bytes;
    }
    const uint64_t fixed = output_head + 2 * layer_stage;
    uint32_t dense_ffn = 0;
    if (cfg_.n_dense_layers) {
        const auto* entry = gm.find("blk.0.ffn_gate.weight");
        dense_ffn = entry ? static_cast<uint32_t>(entry->shape[1]) : 0;
    }
    const uint64_t batch_elems = uint64_t(kPrefillBatch) * (
        4ULL * cfg_.hidden + 2ULL * cfg_.q_lora_rank +
        uint64_t(cfg_.n_heads) * (3ULL * Dn + 2ULL * cfg_.rope_dim +
                                  2ULL * cfg_.value_length) +
        2ULL * cfg_.kv_lora_rank + cfg_.rope_dim +
        3ULL * std::max(dense_ffn, cfg_.expert_ffn * cfg_.n_shared) +
        cfg_.n_experts + uint64_t(cfg_.n_experts_active) * cfg_.hidden);
    const uint64_t compact_batch = compact_mla_ ? uint64_t(kPrefillBatch) *
        2ULL * cfg_.n_heads * cfg_.kv_lora_rank : 0;
    const uint64_t batch_scratch = (batch_elems + compact_batch) * sizeof(float) +
        uint64_t(kPrefillBatch) * cfg_.n_experts_active *
            (sizeof(uint32_t) + sizeof(float));
    const uint64_t runtime_margin = 32ULL << 20;
    const uint64_t expert_batch = largest_transient * cfg_.n_experts_active * 3;
    const uint64_t base_estimated = fixed + kv + absorbed + scores + expert_batch +
                                    batch_scratch + runtime_margin;
    const uint64_t host_reserve = cfg_.max_seq <= 2048 ? 1ULL << 30 : 2ULL << 30;
    const auto memory = mem::snapshot();
    // In compact mode the Q6_K absorption removes 3.25 GB of avoidable
    // resident expansion. Spend that headroom on the deterministic fixed
    // stream: every cached byte removes one SSD byte from every decode step.
    uint64_t desired_cache = compact_mla_ ? fixed_payload : 256ULL << 20;
    if (const char* value = std::getenv("METALBLOK_FIXED_CACHE_MB")) {
        char* end = nullptr;
        const unsigned long mb = std::strtoul(value, &end, 10);
        if (!end || *end != '\0' || mb > 12288)
            die("METALBLOK_FIXED_CACHE_MB must be in [0,12288]");
        desired_cache = uint64_t(mb) << 20;
    }
    const uint64_t cache_guard = 2ULL << 30;
    const uint64_t cache_headroom = memory.available >
        base_estimated + host_reserve + cache_guard
        ? memory.available - base_estimated - host_reserve - cache_guard : 0;
    fixed_cache_budget_ = std::min(desired_cache, cache_headroom);
    const uint64_t estimated = base_estimated + fixed_cache_budget_;
    prof::log("memory-ledger: mode=%s output=%.2fGB layer_slabs=%.2fGB fixed_cache=%.2fGB "
        "kv=%.2fGB absorbed=%.2fGB expert_batch=%.2fMB batch=%.2fMB margin=%.2fMB "
        "fixed_set=%.2fGB estimated=%.2fGB available=%.2fGB reserve=%.2fGB "
        "cache_guard=%.2fGB",
              compact_mla_ ? "mla" : "expanded", output_head / 1e9,
              2 * layer_stage / 1e9, fixed_cache_budget_ / 1e9, kv / 1e9,
              absorbed / 1e9, expert_batch / 1e6,
              batch_scratch / 1e6,
              runtime_margin / 1e6, fixed_payload / 1e9,
              estimated / 1e9, memory.available / 1e9,
              host_reserve / 1e9, cache_guard / 1e9);
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
    load_fixed_projections_();
    if (compact_mla_) build_absorbed_kv_();
    alloc_expert_slots_();
    build_grids_();

    prof::log("gguf_runtime: init mode=%s n_layers=%u hidden=%u vocab=%u "
              "n_experts=%u active=%u kv_lora=%u rope_dim=%u max_seq=%u rss=%zuMB",
              compact_mla_ ? "mla" : "expanded", cfg_.n_layers, cfg_.hidden, cfg_.vocab,
              cfg_.n_experts, cfg_.n_experts_active,
              cfg_.kv_lora_rank, cfg_.rope_dim, cfg_.max_seq, prof::rss_mb());
}

void GgufRuntime::set_sampling(float temperature, float top_p, uint64_t seed) {
    if (!std::isfinite(temperature) || temperature < 0.0f || temperature > 2.0f)
        die("temperature must be in [0, 2]");
    if (!std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f)
        die("top-p must be in (0, 1]");
    temperature_ = temperature;
    top_p_ = top_p;
    seed_ = seed;
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
    // DeepSeek applies mscale^2 to QK and YaRN interpolation to rotary
    // frequencies. The latter is implemented in the pinned consecutive-pair
    // kernels; this value is only the former softmax multiplier.
    cfg_.yarn_mscale = 1.0f;
    {
        float factor   = g.get_f32_or("deepseek2.rope.scaling.factor", 0.0f);
        float log_mult = g.get_f32_or("deepseek2.rope.scaling.yarn_log_multiplier", 0.0f);
        const uint32_t original =
            g.get_u32_or("deepseek2.rope.scaling.original_context_length", 0);
        if (std::fabs(factor - 40.0f) > 1e-6f ||
            std::fabs(log_mult - 0.1f) > 1e-6f || original != 4096)
            die("config: checkpoint YaRN contract differs from pinned R1");
        if (factor > 1.0f && log_mult > 0.0f) {
            double m = 1.0 + (double)log_mult * std::log((double)factor);
            cfg_.yarn_mscale = (float)(m * m);
        }
    }
    // R1 declares sigmoid gating (==2), scale 2.5, and normalized top-k.
    // Defaults make omitted metadata fail the pinned-contract check below.
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
    if (cfg_.n_layers != 61 || cfg_.hidden != 7168 || cfg_.n_heads != 128 ||
        cfg_.q_lora_rank != 1536 || cfg_.kv_lora_rank != 512 ||
        cfg_.key_length != 192 || cfg_.value_length != 128 || cfg_.rope_dim != 64 ||
        cfg_.n_dense_layers != 3 || cfg_.expert_ffn != 2048 || cfg_.n_shared != 1 ||
        cfg_.vocab != 129280 || cfg_.context_length != 163840)
        die("config: checkpoint dimensions differ from pinned DeepSeek-R1-671B");
    if (cfg_.expert_gating_func != 2 || cfg_.expert_weights_norm != 1 ||
        std::fabs(cfg_.expert_weights_scale - 2.5f) > 1e-6f)
        die("config: checkpoint router differs from pinned noaux_tc contract");
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
            require_(gm, buf, cfg_.n_experts, 0);

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
    auto fp32 = [&](size_t n){ return mtl_->alloc(n * 4); };
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

    x_         = fp32(H);
    x_norm_    = fp32(H);
    q_a_       = fp32(Hi);
    q_a_n_     = fp32(Hi);
    q_full_    = fp32((size_t)HE * (Dn + Dr));
    q_nope_    = fp32((size_t)HE * Dn);
    q_rope_    = fp32((size_t)HE * Dr);
    if (compact_mla_) q_eff_ = fp32((size_t)HE * Lk);
    kv_a_      = fp32((size_t)Lk + Dr);
    kv_lat_    = fp32(Lk);
    kv_full_   = fp32((size_t)HE * (Dn + Dv));
    if (compact_mla_) o_lat_ = fp32((size_t)HE * Lk);
    if (validate_mla_) o_lat_ref_ = fp32((size_t)HE * Lk);
    o_full_    = fp32((size_t)HE * Dv);
    attn_out_  = fp32(H);
    ffn_gate_  = fp32(F_max);
    ffn_up_    = fp32(F_max);
    ffn_act_   = fp32(F_max);
    ffn_out_   = fp32(H);
    expert_tmp_= fp32(H);
    router_log_= fp32(cfg_.n_experts);
    router_idx_= mtl_->alloc((size_t)cfg_.n_experts_active * 4);
    router_wts_= mtl_->alloc((size_t)cfg_.n_experts_active * 4);
    logits_    = fp32(cfg_.vocab);
    next_tok_  = mtl_->alloc(4);

    const size_t B = kPrefillBatch;
    x_b_       = fp32(B * H);
    xn_b_      = fp32(B * H);
    qa_b_      = fp32(B * Hi);
    qan_b_     = fp32(B * Hi);
    qf_b_      = fp32(B * HE * (Dn + Dr));
    qn_b_      = fp32(B * HE * Dn);
    qr_b_      = fp32(B * HE * Dr);
    if (compact_mla_) qeff_b_ = fp32(B * HE * Lk);
    kva_b_     = fp32(B * (Lk + Dr));
    kvlat_b_   = fp32(B * Lk);
    kvfull_b_  = fp32(B * HE * (Dn + Dv));
    if (compact_mla_) olat_b_ = fp32(B * HE * Lk);
    ofull_b_   = fp32(B * HE * Dv);
    attnout_b_ = fp32(B * H);
    fg_b_      = fp32(B * F_max);
    fu_b_      = fp32(B * F_max);
    fa_b_      = fp32(B * F_max);
    fo_b_      = fp32(B * H);
    rlog_b_    = fp32(B * cfg_.n_experts);
    ridx_b_    = mtl_->alloc(B * cfg_.n_experts_active * sizeof(uint32_t));
    rwts_b_    = fp32(B * cfg_.n_experts_active);
    routed_b_  = fp32(B * cfg_.n_experts_active * H);

    if (compact_mla_) {
        constexpr size_t kAttentionBlocks = 32;
        attn_partials_ = fp32(size_t(HE) * kAttentionBlocks * Lk);
        attn_stats_ = fp32(size_t(HE) * kAttentionBlocks * 2);
    }

}

void GgufRuntime::alloc_kv_cache_() {
    const uint32_t HE = cfg_.n_heads;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length, Dr = cfg_.rope_dim;
    if (compact_mla_) c_kv_.resize(cfg_.n_layers);
    else {
        k_nope_.resize(cfg_.n_layers);
        v_cache_.resize(cfg_.n_layers);
    }
    k_rope_.resize(cfg_.n_layers);
    const size_t key_bytes = (size_t)cfg_.max_seq * HE * Dn * 2;
    const size_t val_bytes = (size_t)cfg_.max_seq * HE * Dv * 2;
    const size_t rope_bytes = (size_t)cfg_.max_seq * Dr * 2;
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        if (compact_mla_) c_kv_[L] = mtl_->alloc_lazy(
            (size_t)cfg_.max_seq * cfg_.kv_lora_rank * 2);
        else {
            k_nope_[L] = mtl_->alloc_lazy(key_bytes);
            v_cache_[L] = mtl_->alloc_lazy(val_bytes);
        }
        k_rope_[L] = mtl_->alloc_lazy(rope_bytes);
    }
    scores_ = mtl_->alloc((size_t)HE * cfg_.max_seq * 4);
    if (compact_mla_)
        prof::log("gguf_runtime: MLA cache latent=%.3fGB rope=%.3fGB scores=%.2fMB",
                  uint64_t(cfg_.max_seq) * cfg_.kv_lora_rank * 2 * cfg_.n_layers / 1e9,
                  rope_bytes * cfg_.n_layers / 1e9, scores_.length / 1e6);
    else
        prof::log("gguf_runtime: exact KV cache k=%.3fGB v=%.3fGB rope=%.2fMB scores=%.2fMB",
                  key_bytes * cfg_.n_layers / 1e9,
                  val_bytes * cfg_.n_layers / 1e9,
                  rope_bytes * cfg_.n_layers / 1e6, scores_.length / 1e6);
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

void GgufRuntime::build_absorbed_kv_() {
    const uint32_t HE = cfg_.n_heads;
    const uint32_t Lk = cfg_.kv_lora_rank;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length;
    const long long started = prof::now_us();
    uint64_t source_bytes = 0;
    char name[96];

    for (uint32_t layer = 0; layer < cfg_.n_layers; ++layer) {
        std::snprintf(name, sizeof(name), "blk.%u.attn_kv_b.weight", layer);
        const auto* entry = gm_->find(name);
        if (!entry || entry->shape[0] != Lk ||
            entry->shape[1] != HE * (Dn + Dv))
            die("invalid KV-B tensor for MLA absorption");
        if (entry->type != GGML_Q6_K)
            die("--mla requires the DeepSeek-R1 UD-IQ1_S Q6_K KV-B tensors");

        auto& resident = lw_[layer];
        resident.kv_b.buffer = mtl_->alloc(entry->nbytes);
        std::atomic<bool> done{false};
        gm_->ring().submit(entry->shard, entry->abs_offset, entry->nbytes,
                           resident.kv_b.buffer.contents, &done);
        PreadRing::wait(&done);
        source_bytes += entry->nbytes;
        resident.kv_b.resident = true;
    }
    const long long elapsed = prof::now_us() - started;
    prof::log("gguf_runtime: MLA resident Q6_K KV-B loaded %.3fGB in %.3fs",
              source_bytes / 1e9, elapsed / 1e6);
}

void GgufRuntime::load_resident_norms_() {
    lw_.resize(cfg_.n_layers);
    auto load_into = [&](const std::string& name, MtlBuf& dst, uint64_t n) {
        const GgufTensorEntry* e = gm_->find(name);
        if (!e) { std::fprintf(stderr, "norm missing: %s\n", name.c_str()); std::abort(); }
        if (e->type != GGML_F32 || e->nbytes != n * sizeof(float))
            die("exact R1 contract requires f32 norm weights");
        dst = mtl_->alloc(e->nbytes);
        std::atomic<bool> done{false};
        gm_->ring().submit(e->shard, e->abs_offset, e->nbytes, dst.contents, &done);
        PreadRing::wait(&done);
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
        if (L >= cfg_.n_dense_layers) {
            std::snprintf(buf, sizeof(buf), "blk.%u.exp_probs_b.bias", L);
            const auto* e = gm_->find(buf);
            if (!e || e->type != GGML_F32 ||
                e->nbytes != uint64_t(cfg_.n_experts) * 4)
                die("router bias must be f32[n_experts]");
            lw_[L].router_bias = mtl_->alloc(e->nbytes);
            std::atomic<bool> done{false};
            gm_->ring().submit(e->shard, e->abs_offset, e->nbytes,
                               lw_[L].router_bias.contents, &done);
            PreadRing::wait(&done);
        }
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
// Streaming primitives (hot path)
// ---------------------------------------------------------------------------

GgufRuntime::LoadedWeight GgufRuntime::load_weight_(
    const std::string& name, uint32_t K, uint32_t N,
    uint32_t slice_idx, uint64_t slice_stride) {
    const auto* entry = gm_->find(name);
    if (!entry || !kernel_name_for(entry->type)) {
        std::fprintf(stderr, "load_weight: missing or unsupported %s\n", name.c_str());
        std::abort();
    }
    uint64_t offset = entry->abs_offset;
    uint64_t bytes = entry->nbytes;
    if (slice_stride) {
        offset += uint64_t(slice_idx) * slice_stride;
        bytes = slice_stride;
    }
    LoadedWeight weight;
    if (expert_slot_cursor_ >= kExpertSlots ||
        bytes > expert_slots_[expert_slot_cursor_].length)
        die("expert staging arena exhausted");
    weight.buffer = expert_slots_[expert_slot_cursor_];
    weight.entry = entry;
    weight.ready = &expert_ready_[expert_slot_cursor_++];
    weight.ready->store(false, std::memory_order_relaxed);
    weight.K = K;
    weight.N = N;
    gm_->ring().submit(entry->shard, offset, bytes, weight.buffer.contents,
                       weight.ready, true);
    step_model_bytes_ += bytes;
    ++step_reads_;
    return weight;
}

void GgufRuntime::wait_weight_(const LoadedWeight& weight) {
    const long long start = prof::now_us();
    PreadRing::wait(weight.ready);
    step_io_wait_us_ += prof::now_us() - start;
}

void GgufRuntime::dispatch_weight_(const LoadedWeight& weight,
                                   const MtlBuf& x, const MtlBuf& y) {
    const char* kernel = kernel_name_for(weight.entry->type);
    if (weight.entry->type == GGML_IQ1_S)
        mtl_->dispatch(kernel, {weight.buffer, x, y, iq1s_grid_b_},
                       {{&weight.K, sizeof(weight.K)}}, weight.N, 32, true);
    else if (weight.entry->type == GGML_IQ2_XXS)
        mtl_->dispatch(kernel, {weight.buffer, x, y, iq2xxs_grid_b_},
                       {{&weight.K, sizeof(weight.K)}}, weight.N, 32, true);
    else
        mtl_->dispatch(kernel, {weight.buffer, x, y},
                       {{&weight.K, sizeof(weight.K)}}, weight.N, 32, true);
}

void GgufRuntime::release_weight_(LoadedWeight& weight) {
    weight.ready = nullptr;
}

GgufRuntime::Projection GgufRuntime::load_projection_(
    const std::string& name, uint32_t K, uint32_t N) {
    const auto* entry = gm_->find(name);
    if (!entry || !kernel_name_for(entry->type)) {
        std::fprintf(stderr, "load_projection: missing or unsupported %s\n", name.c_str());
        std::abort();
    }
    Projection projection{mtl_->alloc(entry->nbytes), entry, K, N};
    std::atomic<bool> done{false};
    gm_->ring().submit(entry->shard, entry->abs_offset, entry->nbytes,
                       projection.buffer.contents, &done);
    PreadRing::wait(&done);
    return projection;
}

void GgufRuntime::dispatch_projection_(const Projection& projection,
                                       const MtlBuf& x, const MtlBuf& y) {
    const char* kernel = kernel_name_for(projection.entry->type);
    if (projection.entry->type == GGML_Q6_K && projection.K == 2048 &&
        projection.N % 4 == 0)
        mtl_->dispatch("gemv_q6_K_f32_r4", {projection.buffer, x, y},
                       {{&projection.K, sizeof(projection.K)}}, projection.N / 4,
                       32, true);
    else if (projection.entry->type == GGML_IQ1_S)
        mtl_->dispatch(kernel, {projection.buffer, x, y, iq1s_grid_b_},
                       {{&projection.K, sizeof(projection.K)}}, projection.N, 32, true);
    else if (projection.entry->type == GGML_IQ2_XXS)
        mtl_->dispatch(kernel, {projection.buffer, x, y, iq2xxs_grid_b_},
                       {{&projection.K, sizeof(projection.K)}}, projection.N, 32, true);
    else
        mtl_->dispatch(kernel, {projection.buffer, x, y},
                       {{&projection.K, sizeof(projection.K)}}, projection.N, 32, true);
}

void GgufRuntime::dispatch_projection_batch_(const Projection& projection,
                                             const MtlBuf& x, const MtlBuf& y,
                                             uint32_t count) {
    if (count == 1) {
        dispatch_projection_(projection, x, y);
        return;
    }
    const uint32_t K = projection.K, N = projection.N;
    if (projection.entry->type == GGML_IQ1_S) {
        mtl_->dispatch2d("gemv_iq1_s_f32_b",
                         {projection.buffer, x, y, iq1s_grid_b_},
                         {{&K,4},{&N,4}}, N, count, 32);
        return;
    }
    for (uint32_t i = 0; i < count; ++i)
        dispatch_projection_(projection,
                             view(x, size_t(i) * K * sizeof(float),
                                  size_t(K) * sizeof(float)),
                             view(y, size_t(i) * N * sizeof(float),
                                  size_t(N) * sizeof(float)));
}

void GgufRuntime::load_fixed_projections_() {
    const uint32_t H = cfg_.hidden;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads;
    const uint32_t Hi = cfg_.q_lora_rank;
    const uint32_t Lk = cfg_.kv_lora_rank;
    const auto align16k = [](uint64_t n) { return (n + 16383) & ~uint64_t(16383); };
    uint64_t largest_layer = 0;
    char name[96];
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        uint64_t layer_bytes = 0;
        auto index = [&](Projection& dst, const char* suffix, uint32_t K, uint32_t N,
                         bool streamed = true) {
            std::snprintf(name, sizeof(name), "blk.%u.%s", L, suffix);
            const auto* entry = gm_->find(name);
            if (!entry || !kernel_name_for(entry->type))
                die("fixed projection is missing or unsupported");
            dst = {{}, entry, K, N, false};
            if (streamed) layer_bytes += align16k(entry->nbytes);
        };
        index(lw_[L].q_a, "attn_q_a.weight", H, Hi);
        index(lw_[L].q_b, "attn_q_b.weight", Hi, HE * (Dn + Dr));
        index(lw_[L].kv_a, "attn_kv_a_mqa.weight", H, Lk + Dr);
        index(lw_[L].kv_b, "attn_kv_b.weight", Lk, HE * (Dn + Dv),
              !compact_mla_);
        index(lw_[L].attn_output, "attn_output.weight", HE * Dv, H);
        if (L < cfg_.n_dense_layers) {
            std::snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", L);
            const uint32_t F = static_cast<uint32_t>(gm_->find(name)->shape[1]);
            index(lw_[L].ffn_gate, "ffn_gate.weight", H, F);
            index(lw_[L].ffn_up, "ffn_up.weight", H, F);
            index(lw_[L].ffn_down, "ffn_down.weight", F, H);
        } else {
            const uint32_t F = cfg_.expert_ffn * cfg_.n_shared;
            index(lw_[L].ffn_gate, "ffn_gate_shexp.weight", H, F);
            index(lw_[L].ffn_up, "ffn_up_shexp.weight", H, F);
            index(lw_[L].ffn_down, "ffn_down_shexp.weight", F, H);
            index(lw_[L].router, "ffn_gate_inp.weight", H, cfg_.n_experts);
        }
        largest_layer = std::max(largest_layer, layer_bytes);
    }
    std::vector<Projection*> candidates;
    candidates.reserve(cfg_.n_layers * 9);
    for (auto& layer : lw_) {
        for (Projection* projection : {
                 &layer.q_a, &layer.q_b, &layer.kv_a, &layer.kv_b,
                 &layer.attn_output, &layer.ffn_gate, &layer.ffn_up,
                 &layer.ffn_down, &layer.router})
            if (projection->entry && (!compact_mla_ || projection != &layer.kv_b))
                candidates.push_back(projection);
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Projection* a, const Projection* b) {
                         return a->entry->nbytes < b->entry->nbytes;
                     });
    uint64_t cache_bytes = 0;
    size_t cache_count = 0;
    for (const Projection* projection : candidates) {
        const uint64_t bytes = align16k(projection->entry->nbytes);
        if (bytes > fixed_cache_budget_ - cache_bytes) break;
        cache_bytes += bytes;
        ++cache_count;
    }
    if (cache_bytes) {
        fixed_cache_ = mtl_->alloc(cache_bytes);
        auto ready = std::make_unique<std::atomic<bool>[]>(cache_count);
        size_t offset = 0;
        for (size_t i = 0; i < cache_count; ++i) {
            auto& projection = *candidates[i];
            projection.buffer = view(fixed_cache_, offset, projection.entry->nbytes);
            projection.resident = true;
            ready[i].store(false, std::memory_order_relaxed);
            gm_->ring().submit(projection.entry->shard, projection.entry->abs_offset,
                               projection.entry->nbytes, projection.buffer.contents,
                               &ready[i]);
            offset += align16k(projection.entry->nbytes);
        }
        for (size_t i = 0; i < cache_count; ++i) PreadRing::wait(&ready[i]);
    }
    // Cache selection can make the conservative pre-ledger slab size obsolete.
    // Allocate only what the largest remaining streamed layer actually needs.
    uint64_t streamed_layer = 0;
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const auto& layer = lw_[L];
        uint64_t bytes = 0;
        for (const Projection* projection : {
                 &layer.q_a, &layer.q_b, &layer.kv_a, &layer.kv_b,
                 &layer.attn_output, &layer.ffn_gate, &layer.ffn_up,
                 &layer.ffn_down, &layer.router})
            if (projection->entry && !projection->resident &&
                (!compact_mla_ || projection != &layer.kv_b))
                bytes += align16k(projection->entry->nbytes);
        streamed_layer = std::max(streamed_layer, bytes);
    }
    streamed_layer = std::max<uint64_t>(streamed_layer, 16384);
    for (auto& slab : layer_stage_) slab = mtl_->alloc(streamed_layer);
    output_projection_ = load_projection_("output.weight", H, cfg_.vocab);
    prof::log("gguf_runtime: fixed tier output=%.3fGB cache=%.3fGB/%zu projections "
              "layer_slabs=2x%.3fGB",
              output_projection_.buffer.length / 1e9, cache_bytes / 1e9, cache_count,
              streamed_layer / 1e9);
}

void GgufRuntime::begin_stage_layer_(uint32_t L, uint32_t slot) {
    if (L >= cfg_.n_layers) die("stage layer out of range");
    if (slot >= layer_stage_.size()) die("stage slot out of range");
    auto& layer = lw_[L];
    std::array<Projection*, 9> projections = {
        &layer.q_a, &layer.q_b, &layer.kv_a, &layer.kv_b,
        &layer.attn_output, &layer.ffn_gate, &layer.ffn_up,
        &layer.ffn_down, &layer.router,
    };
    const uint32_t count = L < cfg_.n_dense_layers ? 8 : 9;
    size_t offset = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& projection = *projections[i];
        if (projection.resident) {
            layer_ready_[slot][i].store(true, std::memory_order_relaxed);
            continue;
        }
        offset = (offset + 16383) & ~size_t(16383);
        if (offset > layer_stage_[slot].length ||
            projection.entry->nbytes > layer_stage_[slot].length - offset)
            die("layer staging slab overflow");
        projection.buffer = view(layer_stage_[slot], offset, projection.entry->nbytes);
        layer_ready_[slot][i].store(false, std::memory_order_relaxed);
        gm_->ring().submit(projection.entry->shard, projection.entry->abs_offset,
                           projection.entry->nbytes, projection.buffer.contents,
                           &layer_ready_[slot][i]);
        step_model_bytes_ += projection.entry->nbytes;
        ++step_reads_;
        offset += projection.entry->nbytes;
    }
    layer_stage_started_[slot] = prof::now_us();
}

void GgufRuntime::wait_stage_layer_(uint32_t L, uint32_t slot) {
    const uint32_t count = L < cfg_.n_dense_layers ? 8 : 9;
    const long long started = prof::now_us();
    for (uint32_t i = 0; i < count; ++i)
        PreadRing::wait(&layer_ready_[slot][i]);
    const long long now = prof::now_us();
    step_io_wait_us_ += now - started;
    if (profile_layers_)
        prof::log("layer-stage pos=%u layer=%u slot=%u span_us=%lld blocked_us=%lld",
                  pos_, L, slot, now - layer_stage_started_[slot], now - started);
}

void GgufRuntime::alloc_expert_slots_() {
    uint64_t largest = 0;
    char name[96];
    for (uint32_t L = cfg_.n_dense_layers; L < cfg_.n_layers; ++L) {
        for (const char* suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                   "ffn_down_exps.weight"}) {
            std::snprintf(name, sizeof(name), "blk.%u.%s", L, suffix);
            largest = std::max(largest, gm_->find(name)->nbytes / cfg_.n_experts);
        }
    }
    expert_slot_bytes_ = largest;
    for (uint32_t i = 0; i < kDecodeExpertSlots; ++i)
        expert_slots_[i] = mtl_->alloc(largest);
    prof::log("gguf_runtime: expert arena %u x %.3f MB = %.3f MB",
              kDecodeExpertSlots, largest / 1e6,
              kDecodeExpertSlots * largest / 1e6);
}

// ---------------------------------------------------------------------------
// Embedding lookup: pread + dequant one row of token_embd.
// ---------------------------------------------------------------------------
void GgufRuntime::embed_lookup_(uint32_t tok) {
    embed_lookup_into_(tok, x_);
}

void GgufRuntime::embed_lookup_into_(uint32_t tok, const MtlBuf& output) {
    const GgufTensorEntry* e = gm_->find("token_embd.weight");
    if (!e) die("token_embd missing");
    if (tok >= cfg_.vocab) die("token id out of range");

    const uint32_t H = cfg_.hidden;
    const uint64_t per_row = bytes_per_row(H, bytes_per_block(e->type), e->type);
    const uint64_t off     = e->abs_offset + (uint64_t)tok * per_row;

    if (dequant_scratch_bytes_.size() < per_row) dequant_scratch_bytes_.resize(per_row);
    std::atomic<bool> done{false};
    gm_->ring().submit(e->shard, off, per_row, dequant_scratch_bytes_.data(), &done);
    const long long io_start = prof::now_us();
    PreadRing::wait(&done);
    step_io_wait_us_ += prof::now_us() - io_start;
    step_model_bytes_ += per_row;
    ++step_reads_;

    if (output.length < size_t(H) * sizeof(float)) die("embedding output too small");
    float* dst = static_cast<float*>(output.contents);
    const CpuDqInfo info = cpu_dq_info(e->type);
    if (info.is_f16) {
        const uint16_t* src = static_cast<const uint16_t*>(
            static_cast<const void*>(dequant_scratch_bytes_.data()));
        for (uint32_t i = 0; i < H; ++i) dst[i] = f16_to_f32(src[i]);
    } else if (info.is_f32) {
        const float* src = (const float*)dequant_scratch_bytes_.data();
        std::memcpy(dst, src, (size_t)H * sizeof(float));
    } else if (info.fn) {
        require_block_aligned(H, "token_embd");
        const uint32_t n_blocks = H / 256;
        if (dequant_scratch_f32_.size() < 256) dequant_scratch_f32_.resize(256);
        float* tmp = dequant_scratch_f32_.data();
        const uint8_t* src = dequant_scratch_bytes_.data();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            info.fn(src + b * info.bpb, tmp);
            std::memcpy(dst + b * 256, tmp, 256 * sizeof(float));
        }
    } else {
        die("token_embd: unsupported type");
    }
}

// ---------------------------------------------------------------------------
// Exact legacy DeepSeek attention. This checkpoint has a combined wkv_b
// tensor, so it is expanded before fp16 K/V caching exactly like llama.cpp's
// non-MLA compatibility graph. Algebraic absorption is not interchangeable
// after the cache's fp16 rounding and fails token parity on this GGUF.
// ---------------------------------------------------------------------------
void GgufRuntime::mla_attn_compact_(uint32_t L) {
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden, Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim, Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    const uint32_t layer = 0, ms = cfg_.max_seq, T = pos_ + 1;
    const uint32_t position = pos_;
    const float theta = cfg_.rope_freq_base, eps = cfg_.rms_eps;
    const float scale = cfg_.yarn_mscale / std::sqrt(float(Dn + Dr));

    mtl_->begin();
    rmsnorm_f32(*mtl_, x_, w.attn_norm, x_norm_, H, eps);
    dispatch_projection_(w.q_a, x_norm_, q_a_);
    rmsnorm_f32(*mtl_, q_a_, w.q_a_norm, q_a_n_, Hi, eps);
    dispatch_projection_(w.q_b, q_a_n_, q_full_);
    mtl_->dispatch("mla_q_split_rope_r1", {q_full_, q_nope_, q_rope_},
                   {{&Dn,4},{&Dr,4},{&position,4},{&theta,4}}, HE, TG_ELT, true);
    mtl_->dispatch("mla_q6_kv_b_query_r1", {w.kv_b.buffer, q_nope_, q_eff_},
                   {{&Lk,4},{&Dn,4},{&Dv,4}}, HE * Lk, 32, true);

    dispatch_projection_(w.kv_a, x_norm_, kv_a_);
    mtl_->dispatch("mla_kv_norm_rope_store_r1",
                   {kv_a_, w.kv_a_norm, c_kv_[L], k_rope_[L]},
                   {{&Lk,4},{&Dr,4},{&position,4},{&ms,4},{&eps,4},{&theta,4}},
                   1, TG_RED, true);
    if (T >= 1024) {
        constexpr uint32_t blocks = 32;
        if (validate_mla_ && T == 1024)
            mtl_->dispatch("mla_attn_decode_f32",
                           {q_eff_, q_rope_, c_kv_[L], k_rope_[L],
                            o_lat_ref_, scores_},
                           {{&HE,4},{&Lk,4},{&Dr,4},{&layer,4},{&T,4},{&ms,4},{&scale,4}},
                           HE, TG_RED, true);
        mtl_->dispatch2d("mla_attn_online_r1_pass1",
                         {q_eff_, q_rope_, c_kv_[L], k_rope_[L],
                          attn_partials_, attn_stats_},
                         {{&T,4},{&ms,4},{&scale,4}}, blocks, HE, 32);
        mtl_->dispatch("mla_attn_online_r1_pass2",
                       {attn_partials_, attn_stats_, o_lat_}, {}, HE, 32, true);
        if (validate_mla_ && T == 1024) {
            mtl_->flush();
            validate_mla_output_(L, T, o_lat_ref_, o_lat_);
        }
    } else {
        mtl_->dispatch("mla_attn_decode_f32",
                       {q_eff_, q_rope_, c_kv_[L], k_rope_[L], o_lat_, scores_},
                       {{&HE,4},{&Lk,4},{&Dr,4},{&layer,4},{&T,4},{&ms,4},{&scale,4}},
                       HE, TG_RED, true);
    }
    mtl_->dispatch("mla_q6_kv_b_value_r1", {w.kv_b.buffer, o_lat_, o_full_},
                   {{&Lk,4},{&Dn,4},{&Dv,4}}, HE * Dv, 32, true);
    dispatch_projection_(w.attn_output, o_full_, attn_out_);
    axpy_f32(*mtl_, x_, attn_out_, 1.0f, H);
}

void GgufRuntime::mla_attn_(uint32_t L) {
    if (compact_mla_) { mla_attn_compact_(L); return; }
    const auto& w  = lw_[L];
    const uint32_t H  = cfg_.hidden;
    const uint32_t Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim;
    const uint32_t Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    uint32_t pos_u = pos_, cache_layer = 0, ms = cfg_.max_seq, T = pos_ + 1;
    float th  = cfg_.rope_freq_base;
    float eps = cfg_.rms_eps;
    float scale = (1.0f / std::sqrt((float)(Dn + Dr))) * cfg_.yarn_mscale;

    // Current-layer weights are immutable and ready in staged/cached shared
    // buffers. Metal preserves dispatch order, including the cache writes.
    mtl_->begin();
    rmsnorm_f32(*mtl_, x_, w.attn_norm, x_norm_, H, eps);
    dispatch_projection_(w.q_a, x_norm_, q_a_);
    rmsnorm_f32(*mtl_, q_a_, w.q_a_norm, q_a_n_, Hi, eps);
    dispatch_projection_(w.q_b, q_a_n_, q_full_);
    mtl_->dispatch("mla_q_split_rope_r1", { q_full_, q_nope_, q_rope_ },
                   { {&Dn,4},{&Dr,4},{&pos_u,4},{&th,4} },
                   HE, TG_ELT, true);
    dispatch_projection_(w.kv_a, x_norm_, kv_a_);
    mtl_->dispatch("mha_kv_norm_rope_r1",
                   { kv_a_, w.kv_a_norm, kv_lat_, k_rope_[L] },
                   { {&Lk,4},{&Dr,4},{&cache_layer,4},{&pos_u,4},{&ms,4},{&eps,4},{&th,4} },
                   1, TG_RED, true);
    dispatch_projection_(w.kv_b, kv_lat_, kv_full_);
    mtl_->dispatch("mha_kv_store_f16",
                   { kv_full_, k_nope_[L], v_cache_[L] },
                   { {&HE,4},{&Dn,4},{&Dv,4},{&cache_layer,4},{&pos_u,4},{&ms,4} },
                   HE, TG_ELT, true);
    mtl_->dispatch("mha_attn_decode_f32",
                   { q_nope_, q_rope_, k_nope_[L], k_rope_[L], v_cache_[L], o_full_, scores_ },
                   { {&HE,4},{&Dn,4},{&Dr,4},{&Dv,4},{&cache_layer,4},{&T,4},{&ms,4},{&scale,4} },
                   HE, TG_RED, true);
    dispatch_projection_(w.attn_output, o_full_, attn_out_);
    axpy_f32(*mtl_, x_, attn_out_, 1.0f, H);
    // The following FFN is on the same dependency chain and closes this
    // command buffer, avoiding a host round-trip at every layer boundary.
}

// ---------------------------------------------------------------------------
// Dense FFN (R1: layers 0..n_dense_layers-1, i.e. layers 0..2).
// ---------------------------------------------------------------------------
void GgufRuntime::ffn_dense_(uint32_t L) {
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden;

    const uint32_t F = w.ffn_gate.N;

    rmsnorm_f32(*mtl_, x_, w.ffn_norm, x_norm_, H, cfg_.rms_eps);
    dispatch_projection_(w.ffn_gate, x_norm_, ffn_gate_);
    dispatch_projection_(w.ffn_up, x_norm_, ffn_up_);
    swiglu_f32(*mtl_, ffn_gate_, ffn_up_, ffn_act_, F);
    dispatch_projection_(w.ffn_down, ffn_act_, ffn_out_);
    axpy_f32(*mtl_, x_, ffn_out_, 1.0f, H);
    mtl_->commit_and_wait();
}

// ---------------------------------------------------------------------------
// MoE FFN (R1: layers n_dense_layers..n_layers-1).
//
// The pinned R1 checkpoint uses exact grouped sigmoid/noaux_tc routing.
// ---------------------------------------------------------------------------
void GgufRuntime::ffn_moe_(uint32_t L) {
    const auto& w = lw_[L];
    const uint32_t H  = cfg_.hidden;
    const uint32_t Fe = cfg_.expert_ffn;
    const uint32_t Fs = w.ffn_gate.N;
    const uint32_t K  = cfg_.n_experts_active;
    const uint32_t Ne = cfg_.n_experts;

    // Router is the serial discovery point. Its staged/cached matrix and
    // resident bias publish top-8 before any routed expert read is issued.
    rmsnorm_f32(*mtl_, x_, w.ffn_norm, x_norm_, H, cfg_.rms_eps);
    dispatch_projection_(w.router, x_norm_, router_log_);
    float scale = cfg_.expert_weights_scale;
    uint32_t norm = cfg_.expert_weights_norm;
    const uint32_t groups = cfg_.n_expert_groups;
    const uint32_t top_groups = cfg_.n_limited_groups;
    mtl_->dispatch("router_topk_grouped_sigmoid_f32",
                   { router_log_, w.router_bias, router_idx_, router_wts_ },
                   { {&Ne,4},{&K,4},{&groups,4},{&top_groups,4},
                     {&scale,4},{&norm,4} },
                   1, 1, true);
    mtl_->commit_and_wait();
    const long long route_gpu_us = mtl_->last_gpu_us();

    if (trace_ && L == 3) {
        trace_buffer_("ffn_inp", L, x_, H);
        trace_buffer_("ffn_norm", L, x_norm_, H);
        trace_buffer_("router_logits", L, router_log_, Ne);
    }

    const uint32_t* idxs = (const uint32_t*)router_idx_.contents;
    const float*    wts  = (const float*)   router_wts_.contents;
    if (trace_) {
        for (uint32_t k = 0; k < K; ++k)
            prof::log("trace router pos=%u layer=%u rank=%u expert=%u weight=%.8g",
                      pos_, L, k, idxs[k], wts[k]);
    }

    char nbuf[64];
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

    // Queue all 24 exact expert slices. The shard worker reads while the GPU
    // evaluates the always-active shared expert.
    expert_slot_cursor_ = 0;
    std::vector<LoadedWeight> experts;
    experts.reserve(K * 3);
    for (uint32_t k = 0; k < K; ++k) {
        experts.push_back(load_weight_(n_gate, H, Fe, idxs[k], st_gate));
        experts.push_back(load_weight_(n_up, H, Fe, idxs[k], st_up));
        experts.push_back(load_weight_(n_down, Fe, H, idxs[k], st_down));
    }

    mtl_->begin();
    dispatch_projection_(w.ffn_gate, x_norm_, ffn_gate_);
    dispatch_projection_(w.ffn_up, x_norm_, ffn_up_);
    swiglu_f32(*mtl_, ffn_gate_, ffn_up_, ffn_act_, Fs);
    dispatch_projection_(w.ffn_down, ffn_act_, ffn_out_);
    mtl_->commit_and_wait();
    const long long shared_gpu_us = mtl_->last_gpu_us();
    if (trace_ && L == 3) trace_buffer_("ffn_shared", L, ffn_out_, H);

    uint32_t ready_after_shared = 0;
    for (const auto& expert : experts)
        ready_after_shared += expert.ready->load(std::memory_order_acquire);
    long long group_wait_us[2]{}, group_gpu_us[2]{};
    constexpr uint32_t group_size = 4;
    for (uint32_t base = 0; base < K; base += group_size) {
        const uint32_t group = base / group_size;
        const uint32_t end = std::min(K, base + group_size);
        const long long wait_started = prof::now_us();
        for (uint32_t k = base; k < end; ++k) {
            wait_weight_(experts[k * 3]);
            wait_weight_(experts[k * 3 + 1]);
            wait_weight_(experts[k * 3 + 2]);
        }
        group_wait_us[group] = prof::now_us() - wait_started;
        mtl_->begin();
        for (uint32_t k = base; k < end; ++k) {
            const auto& gate = experts[k * 3];
            const auto& up = experts[k * 3 + 1];
            const auto& down = experts[k * 3 + 2];
            if (gate.entry->type == GGML_IQ1_S &&
                up.entry->type == GGML_IQ1_S) {
                mtl_->dispatch("expert_gate_up_swiglu_iq1_s",
                               {gate.buffer, up.buffer, x_norm_, ffn_act_, iq1s_grid_b_},
                               {{&H,4}}, Fe, 32, true);
            } else {
                dispatch_weight_(gate, x_norm_, ffn_gate_);
                dispatch_weight_(up, x_norm_, ffn_up_);
                swiglu_f32(*mtl_, ffn_gate_, ffn_up_, ffn_act_, Fe);
            }
            if (down.entry->type == GGML_IQ1_S) {
                mtl_->dispatch("expert_down_accum_iq1_s_r4",
                               {down.buffer, ffn_act_, ffn_out_, iq1s_grid_b_},
                               {{&Fe,4},{&wts[k],4}}, H / 4, 32, true);
            } else {
                dispatch_weight_(down, ffn_act_, expert_tmp_);
                axpy_f32(*mtl_, ffn_out_, expert_tmp_, wts[k], H);
            }
        }
        if (end == K) axpy_f32(*mtl_, x_, ffn_out_, 1.0f, H);
        mtl_->commit_and_wait();
        group_gpu_us[group] = mtl_->last_gpu_us();
        for (uint32_t k = base; k < end; ++k) {
            release_weight_(experts[k * 3]);
            release_weight_(experts[k * 3 + 1]);
            release_weight_(experts[k * 3 + 2]);
        }
    }
    if (profile_layers_)
        prof::log("moe-pipeline pos=%u layer=%u ready_after_shared=%u/24 "
                  "route_gpu_us=%lld shared_gpu_us=%lld group_size=4 "
                  "g0_wait_us=%lld g0_gpu_us=%lld g1_wait_us=%lld g1_gpu_us=%lld",
                  pos_, L, ready_after_shared, route_gpu_us, shared_gpu_us,
                  group_wait_us[0], group_gpu_us[0],
                  group_wait_us[1], group_gpu_us[1]);
    if (trace_ && L == 3) trace_buffer_("ffn_total", L, ffn_out_, H);
}

void GgufRuntime::attention_batch_compact_(uint32_t L, uint32_t count) {
    const long long started = prof::now_us();
    const int dispatches_before = mtl_->step_dispatches;
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden, Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim, Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    const uint32_t layer = 0, ms = cfg_.max_seq;
    const float theta = cfg_.rope_freq_base, eps = cfg_.rms_eps;
    const float scale = cfg_.yarn_mscale / std::sqrt(float(Dn + Dr));
    auto row = [](const MtlBuf& b, uint32_t i, size_t n) {
        return view(b, size_t(i) * n * sizeof(float), n * sizeof(float));
    };

    mtl_->begin();
    rmsnorm_f32(*mtl_, x_b_, w.attn_norm, xn_b_, H, eps, count);
    dispatch_projection_batch_(w.q_a, xn_b_, qa_b_, count);
    rmsnorm_f32(*mtl_, qa_b_, w.q_a_norm, qan_b_, Hi, eps, count);
    dispatch_projection_batch_(w.q_b, qan_b_, qf_b_, count);
    dispatch_projection_batch_(w.kv_a, xn_b_, kva_b_, count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t position = pos_ + i;
        mtl_->dispatch("mla_q_split_rope_r1",
                       {row(qf_b_, i, size_t(HE) * (Dn + Dr)),
                        row(qn_b_, i, size_t(HE) * Dn),
                        row(qr_b_, i, size_t(HE) * Dr)},
                       {{&Dn,4},{&Dr,4},{&position,4},{&theta,4}},
                       HE, TG_ELT, true);
        mtl_->dispatch("mla_q6_kv_b_query_r1",
                       {w.kv_b.buffer, row(qn_b_, i, size_t(HE) * Dn),
                        row(qeff_b_, i, size_t(HE) * Lk)},
                       {{&Lk,4},{&Dn,4},{&Dv,4}}, HE * Lk, 32, true);
        mtl_->dispatch("mla_kv_norm_rope_store_r1",
                       {row(kva_b_, i, Lk + Dr), w.kv_a_norm,
                        c_kv_[L], k_rope_[L]},
                       {{&Lk,4},{&Dr,4},{&position,4},{&ms,4},{&eps,4},{&theta,4}},
                       1, TG_RED, true);
    }
    if (pos_ + count < 1024) {
        mtl_->dispatch2d("mla_attn_prefill_r1_q4",
                         {qeff_b_, qr_b_, c_kv_[L], k_rope_[L], olat_b_},
                         {{&HE,4},{&count,4},{&pos_,4},{&ms,4},{&scale,4}},
                         HE, (count + 3) / 4, TG_RED);
    } else for (uint32_t i = 0; i < count; ++i) {
        const uint32_t T = pos_ + i + 1;
        constexpr uint32_t blocks = 32;
        if (validate_mla_ && T == 1024)
            mtl_->dispatch("mla_attn_decode_f32",
                           {row(qeff_b_, i, size_t(HE) * Lk),
                            row(qr_b_, i, size_t(HE) * Dr), c_kv_[L],
                            k_rope_[L], o_lat_ref_, scores_},
                           {{&HE,4},{&Lk,4},{&Dr,4},{&layer,4},{&T,4},{&ms,4},{&scale,4}},
                           HE, TG_RED, true);
        mtl_->dispatch2d("mla_attn_online_r1_pass1",
                         {row(qeff_b_, i, size_t(HE) * Lk),
                          row(qr_b_, i, size_t(HE) * Dr), c_kv_[L], k_rope_[L],
                          attn_partials_, attn_stats_},
                         {{&T,4},{&ms,4},{&scale,4}}, blocks, HE, 32);
        mtl_->dispatch("mla_attn_online_r1_pass2",
                       {attn_partials_, attn_stats_, row(olat_b_, i, size_t(HE) * Lk)},
                       {}, HE, 32, true);
        if (validate_mla_ && T == 1024) {
            mtl_->flush();
            validate_mla_output_(L, T, o_lat_ref_,
                                 row(olat_b_, i, size_t(HE) * Lk));
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        mtl_->dispatch("mla_q6_kv_b_value_r1",
                       {w.kv_b.buffer, row(olat_b_, i, size_t(HE) * Lk),
                        row(ofull_b_, i, size_t(HE) * Dv)},
                       {{&Lk,4},{&Dn,4},{&Dv,4}}, HE * Dv, 32, true);
    }
    dispatch_projection_batch_(w.attn_output, ofull_b_, attnout_b_, count);
    axpy_f32(*mtl_, x_b_, attnout_b_, 1.0f, count * H);
    mtl_->commit_and_wait();
    if (profile_layers_)
        prof::log("prefill-stage layer=%u name=attention tokens=%u wall_us=%lld "
                  "gpu_us=%lld dispatches=%d", L, count, prof::now_us() - started,
                  mtl_->last_gpu_us(), mtl_->step_dispatches - dispatches_before);
}

void GgufRuntime::validate_mla_output_(uint32_t layer, uint32_t context,
                                       const MtlBuf& reference,
                                       const MtlBuf& candidate) const {
    const size_t count = size_t(cfg_.n_heads) * cfg_.kv_lora_rank;
    const auto* ref = static_cast<const float*>(reference.contents);
    const auto* got = static_cast<const float*>(candidate.contents);
    double diff2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t worst = 0;
    uint32_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            ++nonfinite;
            continue;
        }
        const float error = std::abs(got[i] - ref[i]);
        const float relative = error / std::max(std::abs(ref[i]), 1e-6f);
        if (error > max_abs) { max_abs = error; worst = i; }
        max_rel = std::max(max_rel, relative);
        diff2 += double(error) * error;
        ref2 += double(ref[i]) * ref[i];
        got2 += double(got[i]) * got[i];
        dot += double(ref[i]) * got[i];
    }
    const double rms = std::sqrt(diff2 / count);
    const double relative_l2 = std::sqrt(diff2 / std::max(ref2, 1e-30));
    const double cosine = dot / std::sqrt(std::max(ref2 * got2, 1e-30));
    prof::log("accuracy-mla layer=%u context=%u count=%zu max_abs=%.9g "
              "max_rel=%.9g rms=%.9g relative_l2=%.9g cosine=%.12g "
              "worst=%zu ref=%.9g candidate=%.9g nonfinite=%u",
              layer, context, count, max_abs, max_rel, rms, relative_l2,
              cosine, worst, ref[worst], got[worst], nonfinite);
    if (nonfinite) die("nonfinite MLA validation output");
}

void GgufRuntime::attention_batch_(uint32_t L, uint32_t count) {
    if (compact_mla_) { attention_batch_compact_(L, count); return; }
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden, Dn = cfg_.key_length - cfg_.rope_dim;
    const uint32_t Dr = cfg_.rope_dim, Dv = cfg_.value_length;
    const uint32_t HE = cfg_.n_heads, Lk = cfg_.kv_lora_rank, Hi = cfg_.q_lora_rank;
    const uint32_t cache_layer = 0, ms = cfg_.max_seq;
    const float theta = cfg_.rope_freq_base, eps = cfg_.rms_eps;
    const float scale = cfg_.yarn_mscale / std::sqrt(float(Dn + Dr));
    auto row = [](const MtlBuf& b, uint32_t i, size_t n) {
        return view(b, size_t(i) * n * sizeof(float), n * sizeof(float));
    };

    mtl_->begin();
    rmsnorm_f32(*mtl_, x_b_, w.attn_norm, xn_b_, H, eps, count);
    dispatch_projection_batch_(w.q_a, xn_b_, qa_b_, count);
    rmsnorm_f32(*mtl_, qa_b_, w.q_a_norm, qan_b_, Hi, eps, count);
    dispatch_projection_batch_(w.q_b, qan_b_, qf_b_, count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t position = pos_ + i;
        mtl_->dispatch("mla_q_split_rope_r1",
                       {row(qf_b_, i, size_t(HE) * (Dn + Dr)),
                        row(qn_b_, i, size_t(HE) * Dn),
                        row(qr_b_, i, size_t(HE) * Dr)},
                       {{&Dn,4},{&Dr,4},{&position,4},{&theta,4}},
                       HE, TG_ELT, true);
    }
    dispatch_projection_batch_(w.kv_a, xn_b_, kva_b_, count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t position = pos_ + i;
        mtl_->dispatch("mha_kv_norm_rope_r1",
                       {row(kva_b_, i, Lk + Dr), w.kv_a_norm,
                        row(kvlat_b_, i, Lk), k_rope_[L]},
                       {{&Lk,4},{&Dr,4},{&cache_layer,4},{&position,4},
                        {&ms,4},{&eps,4},{&theta,4}}, 1, TG_RED, true);
    }
    dispatch_projection_batch_(w.kv_b, kvlat_b_, kvfull_b_, count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t position = pos_ + i;
        mtl_->dispatch("mha_kv_store_f16",
                       {row(kvfull_b_, i, size_t(HE) * (Dn + Dv)),
                        k_nope_[L], v_cache_[L]},
                       {{&HE,4},{&Dn,4},{&Dv,4},{&cache_layer,4},
                        {&position,4},{&ms,4}}, HE, TG_ELT, true);
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t T = pos_ + i + 1;
        mtl_->dispatch("mha_attn_decode_f32",
                       {row(qn_b_, i, size_t(HE) * Dn),
                        row(qr_b_, i, size_t(HE) * Dr), k_nope_[L],
                        k_rope_[L], v_cache_[L],
                        row(ofull_b_, i, size_t(HE) * Dv), scores_},
                       {{&HE,4},{&Dn,4},{&Dr,4},{&Dv,4},{&cache_layer,4},
                        {&T,4},{&ms,4},{&scale,4}}, HE, TG_RED, true);
    }
    dispatch_projection_batch_(w.attn_output, ofull_b_, attnout_b_, count);
    axpy_f32(*mtl_, x_b_, attnout_b_, 1.0f, count * H);
    mtl_->commit_and_wait();
}

void GgufRuntime::dense_batch_(uint32_t L, uint32_t count) {
    const long long started = prof::now_us();
    const int dispatches_before = mtl_->step_dispatches;
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden, F = w.ffn_gate.N;
    mtl_->begin();
    rmsnorm_f32(*mtl_, x_b_, w.ffn_norm, xn_b_, H, cfg_.rms_eps, count);
    dispatch_projection_batch_(w.ffn_gate, xn_b_, fg_b_, count);
    dispatch_projection_batch_(w.ffn_up, xn_b_, fu_b_, count);
    swiglu_f32(*mtl_, fg_b_, fu_b_, fa_b_, count * F);
    dispatch_projection_batch_(w.ffn_down, fa_b_, fo_b_, count);
    axpy_f32(*mtl_, x_b_, fo_b_, 1.0f, count * H);
    mtl_->commit_and_wait();
    if (profile_layers_)
        prof::log("prefill-stage layer=%u name=dense tokens=%u wall_us=%lld "
                  "gpu_us=%lld dispatches=%d", L, count, prof::now_us() - started,
                  mtl_->last_gpu_us(), mtl_->step_dispatches - dispatches_before);
}

void GgufRuntime::moe_batch_(uint32_t L, uint32_t count) {
    const long long fixed_started = prof::now_us();
    const int fixed_dispatches = mtl_->step_dispatches;
    const auto& w = lw_[L];
    const uint32_t H = cfg_.hidden, Fe = cfg_.expert_ffn;
    const uint32_t Fs = w.ffn_gate.N, K = cfg_.n_experts_active;
    const uint32_t Ne = cfg_.n_experts;
    auto row = [](const MtlBuf& b, uint32_t i, size_t n) {
        return view(b, size_t(i) * n * sizeof(float), n * sizeof(float));
    };

    mtl_->begin();
    rmsnorm_f32(*mtl_, x_b_, w.ffn_norm, xn_b_, H, cfg_.rms_eps, count);
    dispatch_projection_batch_(w.router, xn_b_, rlog_b_, count);
    const float route_scale = cfg_.expert_weights_scale;
    const uint32_t route_norm = cfg_.expert_weights_norm;
    const uint32_t groups = cfg_.n_expert_groups, top_groups = cfg_.n_limited_groups;
    for (uint32_t i = 0; i < count; ++i)
        mtl_->dispatch("router_topk_grouped_sigmoid_f32",
                       {row(rlog_b_, i, Ne), w.router_bias,
                        view(ridx_b_, size_t(i) * K * 4, size_t(K) * 4),
                        view(rwts_b_, size_t(i) * K * 4, size_t(K) * 4)},
                       {{&Ne,4},{&K,4},{&groups,4},{&top_groups,4},
                        {&route_scale,4},{&route_norm,4}}, 1, 1, true);
    dispatch_projection_batch_(w.ffn_gate, xn_b_, fg_b_, count);
    dispatch_projection_batch_(w.ffn_up, xn_b_, fu_b_, count);
    swiglu_f32(*mtl_, fg_b_, fu_b_, fa_b_, count * Fs);
    dispatch_projection_batch_(w.ffn_down, fa_b_, fo_b_, count);
    mtl_->commit_and_wait();
    const long long fixed_wall_us = prof::now_us() - fixed_started;
    const long long fixed_gpu_us = mtl_->last_gpu_us();
    const int fixed_dispatch_count = mtl_->step_dispatches - fixed_dispatches;

    constexpr size_t assignments = size_t(kPrefillBatch) * 256;
    std::array<uint32_t, assignments> tokens{};
    std::array<uint32_t, assignments> slots{};
    std::array<uint32_t, 256> counts{}, active{};
    uint32_t active_count = 0;
    const auto* ids = static_cast<const uint32_t*>(ridx_b_.contents);
    for (uint32_t token = 0; token < count; ++token) {
        for (uint32_t rank = 0; rank < K; ++rank) {
            const uint32_t expert = ids[token * K + rank];
            if (expert >= Ne) die("batch router produced invalid expert");
            if (counts[expert]++ == 0) active[active_count++] = expert;
            const size_t dst = size_t(expert) * kPrefillBatch + counts[expert] - 1;
            tokens[dst] = token;
            slots[dst] = token * K + rank;
        }
    }

    char name[64];
    auto tensor = [&](const char* suffix) {
        std::snprintf(name, sizeof(name), "blk.%u.%s", L, suffix);
        return std::string(name);
    };
    const std::string gate_name = tensor("ffn_gate_exps.weight");
    const std::string up_name = tensor("ffn_up_exps.weight");
    const std::string down_name = tensor("ffn_down_exps.weight");
    auto stride = [&](const std::string& tensor_name) {
        const auto* entry = gm_->find(tensor_name);
        if (!entry || entry->n_dims != 3 || entry->shape[2] != Ne ||
            entry->nbytes % Ne) die("invalid stacked expert bank");
        return entry->nbytes / Ne;
    };
    const uint64_t gate_stride = stride(gate_name), up_stride = stride(up_name);
    const uint64_t down_stride = stride(down_name);

    std::array<std::array<LoadedWeight, 24>, 2> loaded{};
    auto queue_group = [&](uint32_t base, uint32_t bank) {
        const uint32_t group_count = std::min(8u, active_count - base);
        expert_slot_cursor_ = bank * kDecodeExpertSlots;
        for (uint32_t j = 0; j < group_count; ++j) {
            const uint32_t expert = active[base + j];
            loaded[bank][j * 3] = load_weight_(gate_name, H, Fe, expert, gate_stride);
            loaded[bank][j * 3 + 1] = load_weight_(up_name, H, Fe, expert, up_stride);
            loaded[bank][j * 3 + 2] = load_weight_(down_name, Fe, H, expert, down_stride);
        }
    };
    long long pipeline_wait_us = 0, pipeline_gpu_us = 0;
    queue_group(0, 0);
    for (uint32_t base = 0; base < active_count; base += 8) {
        const uint32_t bank = (base / 8) & 1u;
        const uint32_t group_count = std::min(8u, active_count - base);
        const long long wait_started = prof::now_us();
        for (uint32_t j = 0; j < group_count * 3; ++j)
            wait_weight_(loaded[bank][j]);
        pipeline_wait_us += prof::now_us() - wait_started;
        if (base + 8 < active_count) queue_group(base + 8, bank ^ 1u);
        mtl_->begin();
        for (uint32_t j = 0; j < group_count; ++j) {
            const uint32_t expert = active[base + j], n = counts[expert];
            const size_t offset = size_t(expert) * kPrefillBatch;
            auto& gate = loaded[bank][j * 3];
            auto& up = loaded[bank][j * 3 + 1];
            auto& down = loaded[bank][j * 3 + 2];
            if (gate.entry->type == GGML_IQ1_S && up.entry->type == GGML_IQ1_S) {
                mtl_->dispatch2d("expert_gate_up_swiglu_iq1_s_b",
                                 {gate.buffer, up.buffer, xn_b_, fa_b_, iq1s_grid_b_},
                                 {{tokens.data() + offset, size_t(n) * 4},
                                  {&H,4},{&Fe,4}}, Fe, n, 32);
            } else {
                for (uint32_t item = 0; item < n; ++item) {
                    const auto input = row(xn_b_, tokens[offset + item], H);
                    dispatch_weight_(gate, input, row(fg_b_, item, Fe));
                    dispatch_weight_(up, input, row(fu_b_, item, Fe));
                    swiglu_f32(*mtl_, row(fg_b_, item, Fe), row(fu_b_, item, Fe),
                               row(fa_b_, item, Fe), Fe);
                }
            }
            if (down.entry->type == GGML_IQ1_S) {
                mtl_->dispatch2d("expert_down_iq1_s_b_r4",
                                 {down.buffer, fa_b_, routed_b_, iq1s_grid_b_},
                                 {{slots.data() + offset, size_t(n) * 4},
                                  {&Fe,4},{&H,4}}, H / 4, n, 32);
            } else {
                for (uint32_t item = 0; item < n; ++item)
                    dispatch_weight_(down, row(fa_b_, item, Fe),
                                     row(routed_b_, slots[offset + item], H));
            }
        }
        mtl_->commit_and_wait();
        pipeline_gpu_us += mtl_->last_gpu_us();
        for (uint32_t j = 0; j < group_count * 3; ++j)
            release_weight_(loaded[bank][j]);
    }
    mtl_->begin();
    mtl_->dispatch("moe_residual_merge_f32", {x_b_, fo_b_, routed_b_, rwts_b_},
                   {{&H,4},{&K,4}}, count * H, TG_ELT, false);
    mtl_->commit_and_wait();
    const long long merge_gpu_us = mtl_->last_gpu_us();
    prof::log("prefill-moe layer=%u tokens=%u expert_union=%u selections=%u "
              "fixed_wall_us=%lld fixed_gpu_us=%lld fixed_dispatches=%d "
              "pipeline_wait_us=%lld pipeline_gpu_us=%lld merge_gpu_us=%lld",
              L, count, active_count, count * K,
              fixed_wall_us, fixed_gpu_us, fixed_dispatch_count,
              pipeline_wait_us, pipeline_gpu_us, merge_gpu_us);
}

void GgufRuntime::prefill_chunk_(const uint32_t* ids, uint32_t count) {
    if (!count || count > kPrefillBatch || pos_ + count > cfg_.max_seq)
        die("invalid prefill tile");
    const long long started = prof::now_us();
    const uint64_t available_before = mem::snapshot().available;
    mtl_->reset_step_stats();
    gm_->ring().reset_stats();
    step_model_bytes_ = step_reads_ = step_allocations_ = 0;
    step_io_wait_us_ = 0;
    for (uint32_t i = 0; i < count; ++i)
        embed_lookup_into_(ids[i], view(x_b_, size_t(i) * cfg_.hidden * 4,
                                        size_t(cfg_.hidden) * 4));
    begin_stage_layer_(0, 0);
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const long long layer_started = prof::now_us();
        const long long gpu_before = mtl_->step_gpu_us;
        const int cmdbufs_before = mtl_->step_cmdbufs;
        const int dispatches_before = mtl_->step_dispatches;
        const auto io_before = profile_layers_ ? gm_->ring().stats() : PreadRing::Stats{};
        const uint32_t slot = L & 1u;
        wait_stage_layer_(L, slot);
        if (L + 1 < cfg_.n_layers) begin_stage_layer_(L + 1, slot ^ 1u);
        attention_batch_(L, count);
        if (L < cfg_.n_dense_layers) dense_batch_(L, count);
        else moe_batch_(L, count);
        if (profile_layers_) {
            const auto io_after = gm_->ring().stats();
            prof::log("prefill-layer begin=%u layer=%u tokens=%u wall_us=%lld "
                      "gpu_us=%lld nvme_bytes=%llu nvme_reads=%llu cmdbufs=%d "
                      "dispatches=%d",
                      pos_, L, count, prof::now_us() - layer_started,
                      mtl_->step_gpu_us - gpu_before,
                      static_cast<unsigned long long>(io_after.bytes - io_before.bytes),
                      static_cast<unsigned long long>(io_after.requests - io_before.requests),
                      mtl_->step_cmdbufs - cmdbufs_before,
                      mtl_->step_dispatches - dispatches_before);
        }
    }
    const auto last_x = view(x_b_, size_t(count - 1) * cfg_.hidden * 4,
                             size_t(cfg_.hidden) * 4);
    mtl_->begin();
    rmsnorm_f32(*mtl_, last_x, output_norm_b_, x_norm_, cfg_.hidden, cfg_.rms_eps);
    dispatch_projection_(output_projection_, x_norm_, logits_);
    mtl_->commit_and_wait();
    pos_ += count - 1;
    *static_cast<uint32_t*>(next_tok_.contents) = sample_logits_();
    ++pos_;
    const auto io = gm_->ring().stats();
    const long long elapsed = prof::now_us() - started;
    const int64_t memory_delta = int64_t(available_before) -
                                 int64_t(mem::snapshot().available);
    prof::log("prefill-tile begin=%u tokens=%u wall_us=%lld tok_s=%.3f "
              "gpu_us=%lld io_wait_us=%lld nvme_bytes=%llu nvme_reads=%llu "
              "urgent_bytes=%llu urgent_reads=%llu nvme_gbps=%.3f "
              "io_service_us=%llu io_max_us=%llu io_peak=%llu "
              "cmdbufs=%d dispatches=%d allocations=%llu available_delta=%lld",
              pos_ - count, count, elapsed, count * 1e6 / double(elapsed),
              mtl_->step_gpu_us, step_io_wait_us_,
              static_cast<unsigned long long>(io.bytes),
              static_cast<unsigned long long>(io.requests),
              static_cast<unsigned long long>(io.urgent_bytes),
              static_cast<unsigned long long>(io.urgent_requests),
              io.elapsed_us ? double(io.bytes) / (io.elapsed_us * 1e3) : 0.0,
              static_cast<unsigned long long>(io.service_us),
              static_cast<unsigned long long>(io.max_service_us),
              static_cast<unsigned long long>(io.peak_outstanding),
              mtl_->step_cmdbufs,
              mtl_->step_dispatches,
              static_cast<unsigned long long>(step_allocations_),
              static_cast<long long>(memory_delta));
}

// ---------------------------------------------------------------------------
// One full forward pass at the current pos_. Reused by prefill and step.
// At entry pos_ = position of THIS token in the sequence (0-based).
// On exit pos_ has been advanced by one and next_tok_.contents holds the
// argmax sample at pos_-1.
// ---------------------------------------------------------------------------
void GgufRuntime::forward_one_(uint32_t tok) {
    const long long started = prof::now_us();
    const auto memory_before = mem::snapshot();
    mtl_->reset_step_stats();
    gm_->ring().reset_stats();
    step_model_bytes_ = 0;
    step_reads_ = 0;
    step_allocations_ = 0;
    step_io_wait_us_ = 0;

    // 1. Embedding.
    embed_lookup_(tok);

    // 2. 61 transformer blocks. Each block:
    //      attn_out = MLA(x);  x += attn_out
    //      ffn_out  = FFN(x);  x += ffn_out
    begin_stage_layer_(0, 0);
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        const long long layer_started = prof::now_us();
        const long long io_wait_before = step_io_wait_us_;
        const long long gpu_before = mtl_->step_gpu_us;
        const int cmdbufs_before = mtl_->step_cmdbufs;
        const int dispatches_before = mtl_->step_dispatches;
        const auto io_before = profile_layers_ ? gm_->ring().stats() : PreadRing::Stats{};
        const uint32_t slot = L & 1u;
        wait_stage_layer_(L, slot);
        if (L + 1 < cfg_.n_layers) begin_stage_layer_(L + 1, slot ^ 1u);
        const long long fixed_done = prof::now_us();
        const long long fixed_wait = step_io_wait_us_;
        const auto fixed_io = profile_layers_ ? gm_->ring().stats() : PreadRing::Stats{};
        mla_attn_(L);
        const long long attention_done = prof::now_us();
        const long long attention_gpu = mtl_->step_gpu_us;
        if (L < cfg_.n_dense_layers) ffn_dense_(L); else ffn_moe_(L);
        if (profile_layers_) {
            const auto final_io = gm_->ring().stats();
            prof::log(
                "layer-metrics pos=%u layer=%u wall_us=%lld fixed_wall_us=%lld "
                "fixed_io_wait_us=%lld fixed_bytes=%llu fixed_reads=%llu "
                "attention_wall_us=%lld attention_gpu_us=%lld ffn_wall_us=%lld "
                "ffn_gpu_us=%lld expert_io_wait_us=%lld expert_bytes=%llu "
                "expert_reads=%llu cmdbufs=%d dispatches=%d",
                pos_, L, prof::now_us() - layer_started,
                fixed_done - layer_started, fixed_wait - io_wait_before,
                static_cast<unsigned long long>(fixed_io.bytes - io_before.bytes),
                static_cast<unsigned long long>(fixed_io.requests - io_before.requests),
                attention_done - fixed_done, attention_gpu - gpu_before,
                prof::now_us() - attention_done,
                mtl_->step_gpu_us - attention_gpu,
                step_io_wait_us_ - fixed_wait,
                static_cast<unsigned long long>(final_io.bytes - fixed_io.bytes),
                static_cast<unsigned long long>(final_io.requests - fixed_io.requests),
                mtl_->step_cmdbufs - cmdbufs_before,
                mtl_->step_dispatches - dispatches_before);
        }
        if (trace_) trace_hidden_(L);
    }

    // Final norm and the always-resident dedicated output projection.
    mtl_->begin();
    rmsnorm_f32(*mtl_, x_, output_norm_b_, x_norm_, cfg_.hidden, cfg_.rms_eps);
    dispatch_projection_(output_projection_, x_norm_, logits_);
    mtl_->commit_and_wait();

    *static_cast<uint32_t*>(next_tok_.contents) = sample_logits_();

    pos_++;
    const long long elapsed = prof::now_us() - started;
    const auto memory_after = mem::snapshot();
    const auto io = gm_->ring().stats();
    const double io_gbps = io.elapsed_us > 0
        ? double(io.bytes) / (io.elapsed_us * 1e3) : 0.0;
    const int64_t memory_delta = int64_t(memory_before.available) -
                                 int64_t(memory_after.available);
    const uint64_t kv_width = compact_mla_
        ? uint64_t(cfg_.kv_lora_rank) + cfg_.rope_dim
        : uint64_t(cfg_.n_heads) *
              ((cfg_.key_length - cfg_.rope_dim) + cfg_.value_length) +
              cfg_.rope_dim;
    const uint64_t kv_bytes = uint64_t(cfg_.n_layers) * kv_width * sizeof(uint16_t);
    prof::log(
        "metrics pos=%u wall_us=%lld gpu_us=%lld io_wait_us=%lld model_bytes=%llu "
        "nvme_bytes=%llu nvme_span_us=%llu nvme_gbps=%.3f useful_reads=%llu nvme_reads=%llu "
        "urgent_bytes=%llu urgent_reads=%llu io_service_us=%llu io_max_us=%llu io_peak=%llu "
        "cmdbufs=%d dispatches=%d allocations=%llu kv_bytes=%llu available_delta=%lld "
        "pageouts=%llu compressions=%llu decompressions=%llu swapins=%llu swapouts=%llu",
        pos_, elapsed, mtl_->step_gpu_us, step_io_wait_us_,
        static_cast<unsigned long long>(step_model_bytes_),
        static_cast<unsigned long long>(io.bytes),
        static_cast<unsigned long long>(io.elapsed_us), io_gbps,
        static_cast<unsigned long long>(step_reads_),
        static_cast<unsigned long long>(io.requests),
        static_cast<unsigned long long>(io.urgent_bytes),
        static_cast<unsigned long long>(io.urgent_requests),
        static_cast<unsigned long long>(io.service_us),
        static_cast<unsigned long long>(io.max_service_us),
        static_cast<unsigned long long>(io.peak_outstanding),
        mtl_->step_cmdbufs,
        mtl_->step_dispatches,
        static_cast<unsigned long long>(step_allocations_),
        static_cast<unsigned long long>(kv_bytes),
        static_cast<long long>(memory_delta),
        static_cast<unsigned long long>(memory_after.pageouts - memory_before.pageouts),
        static_cast<unsigned long long>(memory_after.compressions - memory_before.compressions),
        static_cast<unsigned long long>(memory_after.decompressions - memory_before.decompressions),
        static_cast<unsigned long long>(memory_after.swapins - memory_before.swapins),
        static_cast<unsigned long long>(memory_after.swapouts - memory_before.swapouts));
}

uint32_t GgufRuntime::sample_logits_() {
    const auto* raw = static_cast<const float*>(logits_.contents);
    if (temperature_ == 0.0f) {
        uint32_t best = 0;
        float best_value = -INFINITY;
        float second_value = -INFINITY;
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < cfg_.vocab; ++i) {
            const float value = raw[i];
            if (!std::isfinite(value)) { ++nonfinite; continue; }
            if (value > best_value) {
                second_value = best_value;
                best = i;
                best_value = value;
            } else if (value > second_value) {
                second_value = value;
            }
        }
        if (nonfinite) die("nonfinite output logits");
        prof::log("sample pos=%u mode=greedy token=%u logit=%.6g "
                  "runner_up=%.6g margin=%.6g nonfinite=0",
                  pos_, best, best_value, second_value, best_value - second_value);
        if (trace_) {
            std::array<std::pair<float, uint32_t>, 8> top{};
            for (auto& item : top) item = {-INFINITY, 0};
            for (uint32_t token = 0; token < cfg_.vocab; ++token) {
                const float value = raw[token];
                size_t rank = top.size();
                while (rank && value > top[rank - 1].first) --rank;
                if (rank == top.size()) continue;
                for (size_t j = top.size() - 1; j > rank; --j) top[j] = top[j - 1];
                top[rank] = {value, token};
            }
            for (size_t rank = 0; rank < top.size(); ++rank)
                prof::log("trace logit pos=%u rank=%zu token=%u value=%.7g",
                          pos_, rank, top[rank].second, top[rank].first);
        }
        return best;
    }

    struct Candidate { float probability; uint32_t token; };
    std::vector<Candidate> candidates;
    candidates.reserve(cfg_.vocab);
    float max_logit = -INFINITY;
    for (uint32_t i = 0; i < cfg_.vocab; ++i)
        max_logit = std::max(max_logit, raw[i]);
    double sum = 0.0;
    for (uint32_t i = 0; i < cfg_.vocab; ++i) {
        const float p = std::exp((raw[i] - max_logit) / temperature_);
        candidates.push_back({p, i});
        sum += p;
    }
    for (auto& candidate : candidates) candidate.probability /= float(sum);
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.probability > b.probability;
    });
    double nucleus = 0.0;
    size_t count = 0;
    do { nucleus += candidates[count++].probability; }
    while (count < candidates.size() && nucleus < top_p_);
    const uint64_t random = splitmix64(seed_ ^ uint64_t(pos_));
    const double unit = double(random >> 11) * 0x1.0p-53;
    const double target = unit * nucleus;
    double cumulative = 0.0;
    uint32_t chosen = candidates[count - 1].token;
    for (size_t i = 0; i < count; ++i) {
        cumulative += candidates[i].probability;
        if (target <= cumulative) { chosen = candidates[i].token; break; }
    }
    prof::log("sample pos=%u temp=%.3f top_p=%.3f nucleus=%zu token=%u",
              pos_, temperature_, top_p_, count, chosen);
    if (trace_) {
        const size_t show = std::min<size_t>(8, candidates.size());
        for (size_t i = 0; i < show; ++i)
            prof::log("trace logit rank=%zu token=%u p=%.7f", i,
                      candidates[i].token, candidates[i].probability);
    }
    return chosen;
}

void GgufRuntime::trace_buffer_(const char* name, uint32_t layer,
                                const MtlBuf& buffer, uint32_t size) const {
    const auto* raw = static_cast<const float*>(buffer.contents);
    double sum2 = 0.0;
    float lo = INFINITY, hi = -INFINITY;
    uint64_t hash = 1469598103934665603ULL;
    uint32_t nonfinite = 0;
    for (uint32_t i = 0; i < size; ++i) {
        const float value = raw[i];
        if (!std::isfinite(value)) ++nonfinite;
        else { sum2 += double(value) * value; lo = std::min(lo, value); hi = std::max(hi, value); }
        uint32_t bits;
        std::memcpy(&bits, raw + i, sizeof(bits));
        hash = (hash ^ bits) * 1099511628211ULL;
    }
    prof::log("trace %s pos=%u layer=%u rms=%.7g min=%.7g max=%.7g nonfinite=%u hash=%016llx",
              name, pos_, layer, std::sqrt(sum2 / size), lo, hi, nonfinite,
              static_cast<unsigned long long>(hash));
}

void GgufRuntime::trace_hidden_(uint32_t layer) const {
    trace_buffer_("hidden", layer, x_, cfg_.hidden);
}

uint32_t GgufRuntime::prefill(const uint32_t* ids, uint32_t n) {
    if (n == 0) die("prefill: empty prompt");
    if (n > 1) {
        for (uint32_t i = kDecodeExpertSlots; i < kExpertSlots; ++i)
            expert_slots_[i] = mtl_->alloc(expert_slot_bytes_);
        prof::log("gguf_runtime: prefill expert bank +%.3f MB",
                  kDecodeExpertSlots * expert_slot_bytes_ / 1e6);
    }
    long long t0 = prof::now_us();
    for (uint32_t i = 0; i < n;) {
        const uint32_t count = std::min(kPrefillBatch, n - i);
        if (count == 1) forward_one_(ids[i]);
        else prefill_chunk_(ids + i, count);
        i += count;
    }
    if (n > 1)
        for (uint32_t i = kDecodeExpertSlots; i < kExpertSlots; ++i)
            mtl_->release(expert_slots_[i]);
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
    StateHeader h{{'M','B','L','K','S','T','A','T'}, compact_mla_ ? 4u : 3u,
                  cfg_.n_layers,
                  cfg_.max_seq, cfg_.kv_lora_rank, cfg_.rope_dim, pos_,
                  predicted_token()};
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1;
    const uint64_t key_width = uint64_t(cfg_.n_heads) *
        (cfg_.key_length - cfg_.rope_dim);
    const uint64_t val_width = uint64_t(cfg_.n_heads) * cfg_.value_length;
    for (uint32_t l = 0; ok && l < cfg_.n_layers; ++l) {
        const auto* rope = static_cast<const uint16_t*>(k_rope_[l].contents);
        if (compact_mla_) {
            const auto* latent = static_cast<const uint16_t*>(c_kv_[l].contents);
            ok = std::fwrite(latent, sizeof(uint16_t),
                             uint64_t(pos_) * cfg_.kv_lora_rank, f) ==
                 uint64_t(pos_) * cfg_.kv_lora_rank;
        } else {
            const auto* key = static_cast<const uint16_t*>(k_nope_[l].contents);
            const auto* val = static_cast<const uint16_t*>(v_cache_[l].contents);
            ok = std::fwrite(key, sizeof(uint16_t), uint64_t(pos_) * key_width, f) ==
                 uint64_t(pos_) * key_width;
            if (ok) ok = std::fwrite(val, sizeof(uint16_t),
                                      uint64_t(pos_) * val_width, f) ==
                         uint64_t(pos_) * val_width;
        }
        if (ok) ok = std::fwrite(rope, sizeof(uint16_t),
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
    const uint32_t expected_version = compact_mla_ ? 4u : 3u;
    bool ok = std::fread(&old, sizeof(old), 1, f) == 1 &&
              std::memcmp(old.magic, "MBLKSTAT", 8) == 0 &&
              old.version == expected_version;
    if (ok) {
        std::memcpy(h.magic, old.magic, sizeof(old.magic));
        h.version = old.version;
        h.n_layers = old.n_layers;
        h.max_seq = old.max_seq;
        h.kv_rank = old.kv_rank;
        h.rope_dim = old.rope_dim;
        h.pos = old.pos;
        ok = std::fread(&h.next_token, sizeof(h.next_token), 1, f) == 1;
    }
    ok = ok &&
              h.n_layers == cfg_.n_layers && h.max_seq == cfg_.max_seq &&
              h.kv_rank == cfg_.kv_lora_rank && h.rope_dim == cfg_.rope_dim &&
              h.pos <= cfg_.max_seq;
    const uint64_t key_width = uint64_t(cfg_.n_heads) *
        (cfg_.key_length - cfg_.rope_dim);
    const uint64_t val_width = uint64_t(cfg_.n_heads) * cfg_.value_length;
    for (uint32_t l = 0; ok && l < cfg_.n_layers; ++l) {
        auto* rope = static_cast<uint16_t*>(k_rope_[l].contents);
        if (compact_mla_) {
            auto* latent = static_cast<uint16_t*>(c_kv_[l].contents);
            ok = std::fread(latent, sizeof(uint16_t),
                            uint64_t(h.pos) * cfg_.kv_lora_rank, f) ==
                 uint64_t(h.pos) * cfg_.kv_lora_rank;
        } else {
            auto* key = static_cast<uint16_t*>(k_nope_[l].contents);
            auto* val = static_cast<uint16_t*>(v_cache_[l].contents);
            ok = std::fread(key, sizeof(uint16_t), uint64_t(h.pos) * key_width, f) ==
                 uint64_t(h.pos) * key_width;
            if (ok) ok = std::fread(val, sizeof(uint16_t),
                                     uint64_t(h.pos) * val_width, f) ==
                        uint64_t(h.pos) * val_width;
        }
        if (ok) ok = std::fread(rope, sizeof(uint16_t),
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
