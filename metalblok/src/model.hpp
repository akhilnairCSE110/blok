// Model: mmap'd weight files + named pointers into them.
#pragma once
#include "format.hpp"
#include <string>
#include <vector>

namespace blade {

// Per-layer hot pointers. All point into hot.bin (RAM-resident, mlock'd).
// Tensors are stored as block-FP8 (Fp8Block array) unless noted as fp32/fp16.
struct LayerHot {
    // Norms (fp16, hidden floats each).
    const uint16_t* attn_norm;
    const uint16_t* ffn_norm;

    // MLA projections (block-FP8).
    // Q path is one of two shapes:
    //   q_lora_rank > 0 :  w_q_a + q_a_norm + w_q_b   (Kimi K2 / DeepSeek-V3)
    //   q_lora_rank == 0:  w_q only [HE*(Dn+Dr), H]    (DeepSeek-V2-Lite)
    const Fp8Block* w_q_a;       // [q_lora_rank, hidden]
    const uint16_t* q_a_norm;    // fp16 [q_lora_rank]
    const Fp8Block* w_q_b;       // [n_heads*(head_dim_qk_nope+head_dim_qk_rope), q_lora_rank]
    const Fp8Block* w_q;         // [n_heads*(head_dim_qk_nope+head_dim_qk_rope), hidden]  (no-LoRA)
    const Fp8Block* w_kv_a;      // [kv_lora_rank + head_dim_qk_rope, hidden]
    const uint16_t* kv_a_norm;   // fp16 [kv_lora_rank]
    // MLA absorption split (converter pre-splits W_kv_b):
    const Fp8Block* w_uk;        // [n_heads * kv_lora_rank, head_dim_qk_nope]   K-up, transposed for absorb
    const Fp8Block* w_uv;        // [n_heads * head_dim_v, kv_lora_rank]         V-up
    const Fp8Block* w_o;         // [hidden, n_heads*head_dim_v]

    // FFN: either dense (layer<n_dense_layers) or MoE.
    // Dense FFN (only valid for dense layers):
    const Fp8Block* w_gate_dense; const Fp8Block* w_up_dense; const Fp8Block* w_down_dense;

    // MoE shared expert (always active on MoE layers):
    const Fp8Block* w_gate_shared; const Fp8Block* w_up_shared; const Fp8Block* w_down_shared;

    // Router (block-FP8 [n_experts, hidden]) and per-expert bias (fp32, [n_experts]).
    // router_bias is nullptr when cfg.has_router_bias == 0 (V2-style routing).
    const Fp8Block* w_router;
    const float*    router_bias;
};

class Model {
public:
    Config       cfg;
    ExpertSizes  esz;

    // Always-resident.
    const Fp8Block* w_embed;          // [vocab, hidden]
    const uint16_t* final_norm;       // fp16 [hidden]
    // lm_head: when cfg.tied_embed == 1, equals w_embed (no extra storage).
    //          when cfg.tied_embed == 0, distinct [vocab, hidden] tensor in hot.bin.
    const Fp8Block* lm_head;
    std::vector<LayerHot> layers;

    // Routed experts: mmap'd file + index table.
    const uint8_t*   experts_base;    // mmap of experts.bin (whole file, PROT_READ)
    size_t           experts_len;
    const ExpertIdx* expert_idx;      // mmap of experts.idx
    int              experts_fd;

    // Raw bf16 path (Model::open_hf): per-expert sub-tensor pointers into the
    // mmap'd safetensors shards. Three pointers per (moe_layer, expert).
    // Empty when using the .blade FP8 path.
    struct RawExpert {
        const void* gate;     // bf16 [Fe, H]
        const void* up;       // bf16 [Fe, H]
        const void* down;     // bf16 [H, Fe]
    };
    std::vector<RawExpert> raw_experts;   // [(n_layers - n_dense_layers) * n_experts]

    // Open a .blade directory. Throws via abort() with message on error.
    void open(const std::string& dir);
    // Open a HuggingFace checkpoint dir directly (config.json + safetensors).
    // Sets cfg.weight_dtype = 1. No on-disk conversion.
    void open_hf(const std::string& dir);

    // Sequentially page-fault every dense (non-expert) weight into the kernel
    // page cache. One byte read per 16 KiB page, in tensor order, lets the
    // kernel coalesce contiguous pages into bulk readahead -- much faster
    // than letting decode hit them cold one cache line at a time. No-op on
    // the FP8 path (hot.bin is mlock'd) and on first-run safetensors that
    // exceed RAM (the kernel will evict, that's fine; we paid for streaming).
    void warm_dense();

    // Owned heap arenas populated by open_hf():
    //   norm_arena   - bf16-to-fp16 converted RMSNorm gains.
    //   uk_uv_arena  - per-layer pre-split W_uk / W_uv (bf16, transposed).
    // Lifetimes match the Model. Released by ~Model.
    void* norm_arena  = nullptr;
    void* uk_uv_arena = nullptr;
    ~Model();

    // Pointer to expert E in MoE layer L. Triggers a page fault if not resident.
    const Fp8Block* expert_ptr(uint32_t moe_layer, uint32_t expert) const {
        const auto& e = expert_idx[(size_t)moe_layer * cfg.n_experts + expert];
        return reinterpret_cast<const Fp8Block*>(experts_base + e.offset);
    }
};

} // namespace blade
