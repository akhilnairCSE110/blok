// Runtime: persistent activation/KV buffers and per-layer block runners.
#pragma once
#include "model.hpp"
#include "metal_ctx.hpp"
#include <vector>
#include <atomic>
#include <thread>

namespace blade {

class Streamer;   // fwd

class Runtime {
public:
    Model*  M = nullptr;
    Metal*  G = nullptr;
    Streamer* S = nullptr;

    // Persistent device buffers. Allocated once, reused every step.
    MtlBuf x;          // [H] fp16    residual stream
    MtlBuf x_norm;     // [H] fp16
    MtlBuf q_a;        // [q_lora_rank] fp16
    MtlBuf q_a_n;      // [q_lora_rank] fp16
    MtlBuf q_full;     // [n_heads*(Dn+Dr)] fp16
    MtlBuf q_nope_buf; // [n_heads*Dn] fp16   (output of mla_q_split_rope)
    MtlBuf q_rope_buf; // [n_heads*Dr] fp16   (output of mla_q_split_rope)
    MtlBuf kv_a;       // [Lk + Dr] fp16
    MtlBuf q_eff;      // [n_heads*Lk] fp16   q absorbed through W_uk
    MtlBuf o_lat;      // [n_heads*Lk] fp16
    MtlBuf o_full;     // [n_heads*Dv] fp16
    MtlBuf attn_out;   // [H] fp16
    MtlBuf ffn_gate;   // [max(ffn_dense, expert_ffn*n_shared)] fp16
    MtlBuf ffn_up;     // same
    MtlBuf ffn_act;    // gate*silu(up)
    MtlBuf ffn_out;    // [H] fp16  (shared expert output, then routed-expert accumulator)
    MtlBuf expert_tmp; // [H] fp16  (one routed expert's down-proj output)
    MtlBuf router_log; // [n_experts] fp16
    MtlBuf router_idx; // [K] uint32   -- read on host between cmdbufs
    MtlBuf router_wts; // [K] fp32     -- read on host between cmdbufs
    MtlBuf logits;     // [vocab] fp16
    MtlBuf next_tok;   // [1] uint32

    // Batched prefill scratch.  Sized for PREFILL_B prompt tokens; weights
    // load ONCE per chunk -> arithmetic intensity scales B-fold -> compute
    // bound.  Decode (B=1) uses the non-batched scratch above.
    //
    // PREFILL_B sized for MoE amortization: with B=64, K=6 active, 64 experts,
    // average tokens-per-chosen-expert = 6 -> 6x weight reuse on routed
    // experts (vs 32x on shared/router/attention paths).  Larger B helps
    // routed amortization but costs more threadgroup mem in attention prefill
    // and more scratch RAM.
    static constexpr uint32_t PREFILL_B = 64;
    MtlBuf tokens_b;   // [B] uint32
    MtlBuf x_b;        // [B, H] fp16  residual
    MtlBuf xn_b;       // [B, H] fp16  norm out
    MtlBuf qa_b;       // [B, Hi]
    MtlBuf qan_b;      // [B, Hi]
    MtlBuf qf_b;       // [B, HE*(Dn+Dr)]
    MtlBuf qn_b;       // [B, HE*Dn]
    MtlBuf qr_b;       // [B, HE*Dr]
    MtlBuf kva_b;      // [B, Lk+Dr]
    MtlBuf qeff_b;     // [B, HE*Lk]
    MtlBuf olat_b;     // [B, HE*Lk]
    MtlBuf ofull_b;    // [B, HE*Dv]
    MtlBuf ao_b;       // [B, H]   attn out
    MtlBuf fg_b;       // [B, F_max]
    MtlBuf fu_b;       // [B, F_max]
    MtlBuf fa_b;       // [B, F_max]
    MtlBuf fo_b;       // [B, H]   ffn out
    MtlBuf rlog_b;     // [B, Ne]  router logits
    MtlBuf ridx_b;     // [B, K]   uint32  (router top-K)
    MtlBuf rwts_b;     // [B, K]   fp32
    MtlBuf bkt_idx;    // [B] uint32  per-expert token bucket (gather idx)
    MtlBuf bkt_wts;    // [B] fp32    per-expert token weights

    // KV cache (latent). Sized for max_seq tokens. Mmap-backed so it can grow
    // beyond RAM; the working window stays hot.
    MtlBuf c_kv;       // [n_layers, max_seq, Lk] fp16
    MtlBuf k_rope;     // [n_layers, max_seq, Dr] fp16
    // Scratch for attention scores: [n_heads, max_seq] fp32
    MtlBuf scores;

    // Wrapped (no-copy) buffers for hot weight tensors. Built per-layer on
    // demand and cached. Wrapping is cheap; the underlying memory is mmap'd.
    struct LayerWrap {
        MtlBuf attn_norm, ffn_norm;
        MtlBuf w_q_a, q_a_norm, w_q_b;       // valid when q_lora_rank > 0
        MtlBuf w_q;                           // valid when q_lora_rank == 0
        MtlBuf w_kv_a, kv_a_norm, w_uk, w_uv, w_o;
        MtlBuf w_gate_dense, w_up_dense, w_down_dense;
        MtlBuf w_gate_shared, w_up_shared, w_down_shared;
        MtlBuf w_router; MtlBuf router_bias;   // router_bias may alias zero_bias_b
    };
    std::vector<LayerWrap> lw;
    MtlBuf w_embed_b, final_norm_b, lm_head_b;
    // Zero buffer of size [n_experts] fp32, used for V2 router (no bias).
    MtlBuf zero_bias_b;

    // One MtlBuf per (moe_layer, expert) sub-tensor, pre-wrapped at init.
    // Wrapping is metadata only; pages remain unfaulted until the GPU touches
    // them.  Layout: index = (moe_layer * n_experts + expert) * 3 + {0:gate,
    // 1:up, 2:down}.  We index via expert_w(L,e,sub).
    std::vector<MtlBuf> expert_bufs;
    inline const MtlBuf& expert_w(uint32_t moe_layer, uint32_t expert, uint32_t sub) const {
        return expert_bufs[((size_t)moe_layer * M->cfg.n_experts + expert) * 3 + sub];
    }

    uint32_t pos = 0;       // current sequence length (= position of next token)

    // Kernel-name dispatch table. Set in init() based on cfg.weight_dtype:
    //   weight_dtype == 0 (FP8 .blade): "gemv_fp8_f16", etc.
    //   weight_dtype == 1 (raw bf16):   "gemv_bf16_f16", etc.
    const char* k_gemv  = "gemv_fp8_f16";
    const char* k_embed = "embed_lookup_fp8";

    void init(Model& m, Metal& g, Streamer& s);

    // Forward pass for a single token. Updates kv cache at index `pos`.
    // Returns next sampled token (greedy).
    uint32_t step(uint32_t token);

    // Batched prefill: consumes ids[0..n), advances pos by n, returns the
    // greedy next-token argmax over the LAST position's logits.  Internally
    // chunks at PREFILL_B.  Same KV slots single-token would have written,
    // so the on-disk KV cache contract is preserved.
    uint32_t prefill(const uint32_t* ids, uint32_t n);

private:
    void block(uint32_t L);       // one transformer block
    void mla_attn(uint32_t L);
    void ffn_dense(uint32_t L);
    void ffn_moe(uint32_t L);
    // Batched counterparts.  B is the chunk size (<= PREFILL_B).
    void prefill_chunk(const uint32_t* ids, uint32_t B);
    void block_b(uint32_t L, uint32_t B);
    void mla_attn_b(uint32_t L, uint32_t B);
    void ffn_dense_b(uint32_t L, uint32_t B);
    void ffn_moe_b(uint32_t L, uint32_t B);
};

} // namespace blade
