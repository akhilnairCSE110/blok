// blade on-disk format. Single source of truth.
//
// A .blade model is a directory with three files:
//   header.json     - human-readable config (see below)
//   hot.bin         - all always-resident weights, one tensor after another,
//                     16-byte aligned. mlock'd at startup.
//   experts.bin     - all routed-expert weights, one expert after another,
//                     each expert = w_gate ‖ w_up ‖ w_down (block-FP8, 128B blocks).
//                     Whole file is mmap'd PROT_READ MAP_SHARED, paged on demand.
//   experts.idx     - 16 B per (layer,expert): u64 offset, u32 nbytes, u32 _pad.
//                     Layer L expert E lives at index ((L - first_moe_layer) * n_experts + E).
//
// All multibyte ints little-endian. All tensors row-major.
//
// hot.bin tensor order is fixed and matches HotLayout below; the converter
// MUST emit them in this order so we can resolve pointers by simple offset
// arithmetic without any per-tensor dictionary lookups at runtime.

#pragma once
#include <cstdint>
#include <cstddef>

namespace blade {

// FP8 E4M3 block-quant: 128 contiguous fp8 values share one fp32 scale.
// Block stride in bytes = 128 + 4 = 132. We round-up tensor sizes to a
// multiple of 128 elements at conversion time so block math is trivial.
struct Fp8Block { uint8_t q[128]; float scale; } __attribute__((packed));
static_assert(sizeof(Fp8Block) == 132);

inline size_t fp8_blocks(size_t n_elems) { return (n_elems + 127) / 128; }
inline size_t fp8_bytes (size_t n_elems) { return fp8_blocks(n_elems) * sizeof(Fp8Block); }

struct Config {
    // Architecture (Kimi K2 / DeepSeek-V3 family)
    uint32_t n_layers;          // 61
    uint32_t n_dense_layers;    // 1   (layers [0, n_dense_layers) are dense FFN, rest MoE)
    uint32_t hidden;            // 7168
    uint32_t vocab;             // 163840
    uint32_t n_heads;           // 64    (query heads)
    uint32_t head_dim_qk_nope;  // 128   (non-rope part of Q/K)
    uint32_t head_dim_qk_rope;  // 64    (decoupled rope part)
    uint32_t head_dim_v;        // 128
    uint32_t kv_lora_rank;      // 512   (latent KV dim)
    uint32_t q_lora_rank;       // 1536  -- 0 means "no Q-LoRA: use direct q_proj"
    uint32_t ffn_dense;         // 18432 (intermediate of the single dense FFN)
    uint32_t n_experts;         // 384
    uint32_t n_experts_active;  // 8
    uint32_t n_shared_experts;  // 1
    uint32_t expert_ffn;        // 2048
    float    rope_theta;        // 50000.0
    float    rms_eps;           // 1e-6
    uint32_t max_seq;           // 1<<20 = 1048576
    // Flags (0/1). Default 1 for both to match Kimi K2 / DeepSeek-V3 behaviour.
    uint32_t tied_embed;        // 1: lm_head weight == embedding (Kimi K2/V3)
                                // 0: separate lm_head tensor stored after final_norm (V2-Lite)
    uint32_t has_router_bias;   // 1: per-expert FP32 bias after router weight (V3)
                                // 0: no bias (V2 uses plain softmax routing)
    uint32_t weight_dtype;      // 0 = block-FP8 (.blade format)
                                // 1 = bf16  (raw safetensors mmap path)
    // YaRN attention scale multiplier: softmax_scale = base/sqrt(d) * yarn_mscale.
    // 1.0 for non-YaRN models (Kimi K2 / V3 with default rope).  V2-Lite uses
    // YaRN with factor=40, mscale=mscale_all_dim=0.707, applied as
    //   yarn_mscale = ( 1 + 0.1*log(factor)*mscale )^2  (squared because
    //   DeepSeek-V2 applies mscale_all_dim to softmax_scale twice).
    float yarn_mscale;
};

// Convenience helpers for the bf16 path.
inline size_t bf16_bytes(size_t n_elems) { return n_elems * 2; }
// Generic: number of bytes for a weight tensor of n_elems given dtype.
inline size_t weight_bytes(size_t n_elems, uint32_t dtype) {
    return dtype == 1 ? bf16_bytes(n_elems) : fp8_bytes(n_elems);
}

// Index entry for one routed expert.
struct ExpertIdx { uint64_t offset; uint32_t nbytes; uint32_t _pad; };

// Per-expert tensor sizes (FP8 bytes). Computed from Config at load.
struct ExpertSizes {
    size_t gate;     // expert_ffn * hidden
    size_t up;       // expert_ffn * hidden
    size_t down;     // hidden     * expert_ffn
    size_t total;    // sum of the three (in bytes, fp8-block-padded)
};
inline ExpertSizes expert_sizes(const Config& c) {
    ExpertSizes s;
    s.gate = fp8_bytes((size_t)c.expert_ffn * c.hidden);
    s.up   = s.gate;
    s.down = fp8_bytes((size_t)c.hidden * c.expert_ffn);
    s.total = s.gate + s.up + s.down;
    return s;
}

} // namespace blade
