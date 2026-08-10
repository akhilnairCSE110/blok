// src/gguf_runtime.hpp
// ---------------------------------------------------------------------------
// GgufRuntime: streaming inference engine for GGUF-quantized weights.
//
// Sibling to Runtime (runtime.hpp). The existing Runtime/MLA/MoE pipeline
// (mla.cpp, moe.cpp) is hard-wired to bf16/fp8 weights wrapped as MtlBufs
// over mmap'd .blade or HF checkpoints. R1 IQ1_S cannot use that pipeline:
//
//   1. Weights are k-quant / i-quant blocks, not bf16/fp8.
//   2. Total 140 GB does not fit anywhere near the 24 GB UMA budget, so
//      mmap of payload is BANNED (IO_PROBE_FINDINGS.md). All weight access
//      is per-tensor pread into a small set of reused MtlBufs.
//   3. The per-token graph dispatches the GGUF gemv kernels
//      (kernel_name_for() in gguf_kernels.hpp), not gemv_bf16_f16 /
//      gemv_fp8_f16.
//
// Anti-drift (mixie_plan.txt:113-114): do NOT modify streamer.cpp,
// kvcache.cpp, mla.cpp, moe.cpp. The streaming decoder lives entirely in
// this file + gguf_runtime.cpp.
//
// Memory budget (24 GB UMA target):
//
//   Activations (f16) -- always resident, ~few MB total:
//     x, x_norm                                              28 KB
//     q_a, q_a_n        [Hi=1536]                             6 KB
//     q_full            [HE*(Dn+Dr)=128*192]                 50 KB
//     q_nope, q_rope    [HE*Dn=16K, HE*Dr=8K]                48 KB
//     kv_a              [Lk+Dr=576]                          1.2 KB
//     q_eff, o_lat      [HE*Lk=64K]                          256 KB
//     o_full            [HE*Dv=16K]                           32 KB
//     attn_out          [H]                                   14 KB
//     ffn_gate/up/act   [max(F_dense, F_exp)=2048]           12 KB
//     ffn_out, expert_tmp [H]                                 28 KB
//     router_log/idx/wts [Ne=256, K=8]                       0.6 KB + 64 B
//     logits            [V=129280]                            253 KB
//
//   KV cache (f16, max_seq positions, all 61 layers):
//     c_kv   [n_layers * max_seq * Lk]    @ max_seq=4096    240 MB
//     k_rope [n_layers * max_seq * Dr]    @ max_seq=4096     30 MB
//     scores [HE * max_seq]                                  16 MB (fp32)
//
//   Persistent per-layer (f16) -- ~2 GB total at HE=128:
//     attn_norm, ffn_norm, q_a_norm, kv_a_norm
//                                          14+14+3+1 KB/layer
//     w_uk [HE, Lk, Dn]                    16 MB/layer
//     w_uv [HE, Dv, Lk]                    16 MB/layer
//     -> ~2 GB across 61 layers
//
//   Streaming weights -- allocated/freed per dispatch, transient:
//     q_b              [HE*(Dn+Dr) * Hi]    @ q4_K ≈   20 MB / layer
//     q_a              [Hi * H]             @ q4_K ≈    6 MB / layer
//     kv_a_mqa         [(Lk+Dr) * H]        @ q4_K ≈    2.3 MB / layer
//     attn_output      [H * (HE*Dv)]        @ q4_K ≈   65 MB / layer
//     ffn dense gate/up/down                @ q4_K ≈   3 * 20 MB / dense layer
//     ffn shared gate/up/down               @ q4_K small (Fs=2048)
//     router_inp       [Ne * H]             @ q4_K ≈    1.3 MB
//     expert gate/up   [Fe * H]             @ iq1_s ≈  90 KB / expert
//     expert down      [H * Fe]             @ iq1_s ≈  90 KB / expert
//     token_embd row   [H]                  one row's payload ≈ 1.4 KB
//
//   Peak transient ≈ size of attn_output weight (~65 MB) at any moment.
//   Total RAM footprint ≈ 2 GB resident + ~280 MB KV + ~70 MB peak transient
//   = well under the 24 GB budget.
//
// Per-tensor streaming primitive:
//
//     // Streaming GEMV for one weight tensor.
//     //   W is a 2D row-major weight (rows = output dim, cols = K).
//     //   x is the f16 activation vector of length K, already in `bX`.
//     //   y is the f16 output vector of length n_rows, will be written to `bY`.
//     // 1. Look up the entry: const GgufTensorEntry* e = gm_->find(name);
//     // 2. Pick the kernel:   const char* kn = kernel_name_for(e->type);
//     // 3. Allocate a one-shot weight buffer:
//     //        MtlBuf bW = mtl_->alloc(e->nbytes);
//     // 4. Submit pread directly into bW.contents via the live PreadRing:
//     //        std::atomic<bool> done{false};
//     //        gm_->ring().submit(e->shard, e->abs_offset, e->nbytes,
//     //                           bW.contents, &done);
//     //        PreadRing::wait(&done);
//     // 5. Bind buffers: { bW, bX, bY } and (iq1_s/iq2_xxs) bGrid at buf(4).
//     // 6. Dispatch with K as the only byte arg (GGUF kernels), or
//     //    {K, group_size} for gemv_f16_f16.
//     // 7. commit_and_wait + release(bW).
// ---------------------------------------------------------------------------
#pragma once

#include "gguf.hpp"
#include "gguf_model.hpp"
#include "metal_ctx.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace blade {

// R1 / DeepSeek-V2 architecture parameters parsed from GGUF KV metadata.
// All fields are read by GgufRuntime::init() from the well-known
// deepseek2.* keys (see main.cpp probe_gguf show_keys for the full list).
struct GgufConfig {
    uint32_t n_layers          = 0;   // deepseek2.block_count             (61)
    uint32_t hidden            = 0;   // deepseek2.embedding_length        (7168)
    uint32_t n_heads           = 0;   // deepseek2.attention.head_count    (128)
    uint32_t q_lora_rank       = 0;   // deepseek2.attention.q_lora_rank   (1536)
    uint32_t kv_lora_rank      = 0;   // deepseek2.attention.kv_lora_rank  (512)
    uint32_t key_length        = 0;   // deepseek2.attention.key_length    (192 = 128 nope + 64 rope)
    uint32_t value_length      = 0;   // deepseek2.attention.value_length  (128)
    uint32_t rope_dim          = 0;   // deepseek2.rope.dimension_count    (64)
    float    rope_freq_base    = 0.f; // deepseek2.rope.freq_base          (10000)
    uint32_t n_experts         = 0;   // deepseek2.expert_count            (256)
    uint32_t n_experts_active  = 0;   // deepseek2.expert_used_count       (8)
    uint32_t n_expert_groups   = 8;   // DeepSeek-R1 noaux_tc n_group
    uint32_t n_limited_groups  = 4;   // DeepSeek-R1 noaux_tc topk_group
    uint32_t n_dense_layers    = 0;   // deepseek2.leading_dense_block_count (3)
    uint32_t expert_ffn        = 0;   // deepseek2.expert_feed_forward_length (2048)
    uint32_t n_shared          = 0;   // deepseek2.expert_shared_count     (1)
    uint32_t context_length    = 0;   // deepseek2.context_length
    uint32_t vocab             = 0;   // deepseek2.vocab_size              (129280)
    // RMSNorm epsilon. R1 default = 1e-6 (matches deepseek2/llama convention).
    float    rms_eps           = 1e-6f;
    // Correctness-first default. Larger contexts require explicit opt-in and
    // must pass the runtime memory ledger before any Metal allocation.
    uint32_t max_seq           = 64;
    // YaRN attention scaling. Read from deepseek2.rope.scaling.yarn_log_multiplier
    // if present; treated as 1.0 (= base rope) otherwise. R1 trained with
    // YaRN but is functional with mscale=1.0 for short prompts; we keep this
    // conservative default and only multiply the attention scale by it.
    float    yarn_mscale       = 1.0f;
    // MoE routing (DeepSeek-V3 / R1). expert_gating_func: 1=softmax, 2=sigmoid.
    // expert_weights_scale = routed_scaling_factor (2.5 for R1).
    // expert_weights_norm  = normalize the K chosen weights to sum 1.
    uint32_t expert_gating_func   = 1;     // deepseek2.expert_gating_func
    float    expert_weights_scale = 1.0f;  // deepseek2.expert_weights_scale
    uint32_t expert_weights_norm  = 0;     // deepseek2.expert_weights_norm
};

class GgufRuntime {
public:
    // Take ownership of references; caller keeps them alive for the lifetime
    // of this object. `gm` must already be load()ed (its PreadRing is live).
    void init(const Gguf& g, GgufModel& gm, Metal& mtl);

    // Run prefill on the prompt tokens. Returns the id sampled at the last
    // prompt position (the first generated token). Decode (B=1) loop reused
    // -- each token does a full forward pass and advances pos_.
    uint32_t prefill(const uint32_t* ids, uint32_t n);

    // Decode one token: feed `prev_id`, run all 61 layers, sample, return id.
    uint32_t step(uint32_t prev_id);

    // Crash-safe diagnostic checkpoint for long correctness runs. Stores only
    // exact MLA KV state and position; weights remain in GGUF and are rebuilt.
    bool save_state(const std::string& path) const;
    bool load_state(const std::string& path);
    uint32_t predicted_token() const {
        return next_tok_.contents ? *static_cast<const uint32_t*>(next_tok_.contents) : 0;
    }

    const GgufConfig& cfg() const { return cfg_; }
    uint32_t          pos() const { return pos_; }

private:
    // -----------------------------------------------------------------------
    // Cold-path init.
    // -----------------------------------------------------------------------
    void parse_config_(const Gguf& g);
    void verify_tensor_table_(const GgufModel& gm) const;
    void alloc_activations_();
    void alloc_kv_cache_();
    void load_resident_norms_();
    void build_grids_();

    // -----------------------------------------------------------------------
    // Streaming primitives (hot path).
    // -----------------------------------------------------------------------

    // Dequant one tensor (any quant type) to f16 host memory at `dst`. Reads
    // the payload via PreadRing into a temp aligned scratch, then walks blocks.
    // Sized for tensors small enough to fit a transient scratch (norms, biases,
    // attn_kv_b at ~9 MB/layer). Returns the number of fp16 elements written.
    uint64_t cpu_dequant_to_f16_(const GgufTensorEntry& e, uint16_t* dst);

    // One streaming GEMV: pread the tensor identified by `name` into a
    // freshly-allocated MtlBuf, dispatch the appropriate GGUF kernel, release.
    // For 3D tensors (expert stacks) pass slice_idx (0 to n_experts-1) and
    // the per-slice byte stride. For 2D pass slice_idx=0, stride=0.
    //
    // K is the innermost dim (cols), N the number of output rows. TG=128
    // matches the validated path in validate_gemv (main.cpp:480-495).
    void stream_gemv_(const std::string& name,
                      const MtlBuf& bX, const MtlBuf& bY,
                      uint32_t K, uint32_t N,
                      uint32_t slice_idx = 0, uint64_t slice_stride = 0);

    // f16 grouped GEMV against a resident weight buffer (e.g. absorbed
    // W_uk / W_uv). Uses ops.hpp helpers via gemv_f16_f16.
    void grouped_gemv_f16_(const MtlBuf& bW, const MtlBuf& bX, const MtlBuf& bY,
                           uint32_t N, uint32_t K, uint32_t group);

    // Build the absorbed per-head W_uk / W_uv for layer L from the GGUF
    // attn_kv_b tensor (quantized, stacked [Lk, HE*(Dn+Dv)]). Done lazily on
    // first use of the layer. The result lives in lw_[L].w_uk_b / w_uv_b
    // for the lifetime of GgufRuntime.
    void build_absorbed_kv_b_(uint32_t L);

    // Resolve token_embd row for `tok` into x_b (f16 [hidden]). One row =
    // hidden / 256 IQ1_S blocks (28 blocks * 50 bytes = 1400 bytes for R1).
    // Reads payload via PreadRing, dequants on CPU into x_b.contents.
    void embed_lookup_(uint32_t tok);

    // The forward graph for ONE token. Used by both prefill and step.
    void forward_one_(uint32_t tok);
    void mla_attn_(uint32_t L);
    void ffn_dense_(uint32_t L);
    void ffn_moe_(uint32_t L);

    // -----------------------------------------------------------------------
    // State.
    // -----------------------------------------------------------------------
    const Gguf*  g_   = nullptr;
    GgufModel*   gm_  = nullptr;
    Metal*       mtl_ = nullptr;
    GgufConfig   cfg_{};
    uint32_t     pos_ = 0;       // KV-cache position counter

    // Per-layer resident f16 buffers.
    struct LayerResident {
        MtlBuf attn_norm;   // [H]   f16
        MtlBuf ffn_norm;    // [H]   f16
        MtlBuf q_a_norm;    // [Hi]  f16
        MtlBuf kv_a_norm;   // [Lk]  f16
        // Absorbed MLA projections, built lazily by build_absorbed_kv_b_().
        // w_uk_b: [HE, Lk, Dn] grouped GEMV input.  Rows = HE*Lk, group = Lk.
        // w_uv_b: [HE, Dv, Lk] grouped GEMV input.  Rows = HE*Dv, group = Dv.
        MtlBuf w_uk_b;
        MtlBuf w_uv_b;
        bool   absorbed_built = false;
    };
    std::vector<LayerResident> lw_;

    // Global resident: output norm (f16).
    MtlBuf output_norm_b_;

    // Activations / scratch (all f16 unless noted).
    MtlBuf x_, x_norm_;
    MtlBuf q_a_, q_a_n_;
    MtlBuf q_full_, q_nope_, q_rope_;
    MtlBuf kv_a_;
    MtlBuf q_eff_, o_lat_, o_full_, attn_out_;
    MtlBuf ffn_gate_, ffn_up_, ffn_act_, ffn_out_, expert_tmp_;
    MtlBuf router_log_;     // [Ne]    f16
    MtlBuf router_idx_;     // [K]     u32
    MtlBuf router_wts_;     // [K]     f32
    MtlBuf logits_;         // [V]     f16
    MtlBuf next_tok_;       // [1]     u32

    // KV cache + scores scratch.
    MtlBuf c_kv_;           // [n_layers, max_seq, Lk]   f16
    MtlBuf k_rope_;         // [n_layers, max_seq, Dr]   f16
    MtlBuf scores_;         // [HE, max_seq]             f32

    // Zero-bias buffer for V3-style routing when exp_probs_b is absent.
    MtlBuf zero_bias_;      // [Ne]    f32

    // Vendored codebook grids wrapped as MtlBufs (zero-copy, point into
    // .text). Bound at buffer(4) for iq1_s / iq2_xxs gemv dispatches.
    MtlBuf iq1s_grid_b_;
    MtlBuf iq2xxs_grid_b_;

    // Scratch for CPU dequant of one tensor at a time (sized to the largest
    // tensor we dequant on CPU = attn_kv_b at ~9.4 MB / layer).
    std::vector<uint8_t> dequant_scratch_bytes_;
    std::vector<float>   dequant_scratch_f32_;
};

} // namespace blade
