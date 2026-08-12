// Exact DeepSeek-R1 UD-IQ1_S execution from a three-shard GGUF. Quantized
// weights remain compressed until Metal GEMV. Fixed per-layer projections
// stream through two 16-KiB-aligned slabs; exact router-selected experts use
// a 24-slot urgent arena. The output head, norms, codebooks, and a bounded
// subset of small fixed projections remain resident.
//
// Activations and reductions are FP32. The v3 checkpoint stores the combined
// attn_kv_b graph's expanded FP16 non-RoPE K, V, and shared RoPE K exactly.
// This costs 4,005,504 bytes per committed sequence position. Real-arithmetic
// latent absorption is deliberately not used because its finite-precision
// reassociation failed the token-parity release contract for this checkpoint.
#pragma once

#include "gguf.hpp"
#include "gguf_model.hpp"
#include "metal_ctx.hpp"

#include <atomic>
#include <array>
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
    // Pinned R1 routing metadata; initialization rejects non-sigmoid paths.
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

    // Run layer-major prefill in tiles of at most 128 tokens. Returns the ID
    // sampled at the final prompt position (the first generated token).
    uint32_t prefill(const uint32_t* ids, uint32_t n);

    // Decode one token: feed `prev_id`, run all 61 layers, sample, return id.
    uint32_t step(uint32_t prev_id);

    // DeepSeek-R1 recommends temperature 0.6. A zero temperature preserves
    // the greedy parity mode. Sampling is position-derived, so a restored
    // checkpoint produces the same continuation for the same seed.
    void set_sampling(float temperature, float top_p, uint64_t seed);

    // Crash-safe continuation checkpoint. Stores exact expanded K/V/RoPE
    // prefixes, position, and the pending predicted token. Weights stay GGUF.
    bool save_state(const std::string& path) const;
    bool load_state(const std::string& path);
    uint32_t predicted_token() const {
        return next_tok_.contents ? *static_cast<const uint32_t*>(next_tok_.contents) : 0;
    }

    const GgufConfig& cfg() const { return cfg_; }
    uint32_t          pos() const { return pos_; }
    static constexpr uint32_t kPrefillBatch = 128;
    static constexpr uint32_t prefill_batch_size() { return kPrefillBatch; }

private:
    // -----------------------------------------------------------------------
    // Cold-path init.
    // -----------------------------------------------------------------------
    void parse_config_(const Gguf& g);
    void verify_tensor_table_(const GgufModel& gm) const;
    void alloc_activations_();
    void alloc_kv_cache_();
    void load_resident_norms_();
    void load_fixed_projections_();
    void begin_stage_layer_(uint32_t layer, uint32_t slot);
    void wait_stage_layer_(uint32_t layer, uint32_t slot);
    void alloc_expert_slots_();
    void build_grids_();

    // -----------------------------------------------------------------------
    // Streaming primitives (hot path).
    // -----------------------------------------------------------------------

    // Dequant one tensor (any quant type) to f16 host memory at `dst`. Reads
    // the payload via PreadRing into a temp aligned scratch, then walks blocks.
    // Sized for tensors small enough to fit a transient scratch (norms, biases,
    // attn_kv_b at ~9 MB/layer). Returns the number of fp16 elements written.
    uint64_t cpu_dequant_to_f16_(const GgufTensorEntry& e, uint16_t* dst);

    struct LoadedWeight {
        MtlBuf buffer{};
        const GgufTensorEntry* entry = nullptr;
        std::atomic<bool>* ready = nullptr;
        uint32_t K = 0, N = 0;
    };
    struct Projection {
        MtlBuf buffer{};
        const GgufTensorEntry* entry = nullptr;
        uint32_t K = 0, N = 0;
        bool resident = false;
    };
    LoadedWeight load_weight_(const std::string& name, uint32_t K, uint32_t N,
                              uint32_t slice_idx = 0, uint64_t slice_stride = 0);
    void wait_weight_(const LoadedWeight& weight);
    void dispatch_weight_(const LoadedWeight& weight,
                          const MtlBuf& x, const MtlBuf& y);
    void release_weight_(LoadedWeight& weight);
    Projection load_projection_(const std::string& name, uint32_t K, uint32_t N);
    void dispatch_projection_(const Projection& projection,
                              const MtlBuf& x, const MtlBuf& y);

    // Resolve one quantized embedding row into an FP32 activation. The pinned
    // Q4_K embedding row is 4,032 stored bytes.
    void embed_lookup_(uint32_t tok);
    void embed_lookup_into_(uint32_t tok, const MtlBuf& dst);

    // The forward graph for ONE token. Used by both prefill and step.
    void forward_one_(uint32_t tok);
    void prefill_chunk_(const uint32_t* ids, uint32_t count);
    void attention_batch_(uint32_t layer, uint32_t count);
    void dense_batch_(uint32_t layer, uint32_t count);
    void moe_batch_(uint32_t layer, uint32_t count);
    void mla_attn_(uint32_t L);
    void ffn_dense_(uint32_t L);
    void ffn_moe_(uint32_t L);
    uint32_t sample_logits_();
    void trace_hidden_(uint32_t layer) const;
    void trace_buffer_(const char* name, uint32_t layer,
                       const MtlBuf& buffer, uint32_t size) const;

    // -----------------------------------------------------------------------
    // State.
    // -----------------------------------------------------------------------
    const Gguf*  g_   = nullptr;
    GgufModel*   gm_  = nullptr;
    Metal*       mtl_ = nullptr;
    GgufConfig   cfg_{};
    uint32_t     pos_ = 0;       // KV-cache position counter
    float        temperature_ = 0.0f;
    float        top_p_ = 0.95f;
    uint64_t     seed_ = 3407;
    bool         trace_ = false;
    bool         profile_layers_ = false;

    // Per-layer resident FP32 weights plus projection descriptors.
    struct LayerResident {
        MtlBuf attn_norm;   // [H]   f32
        MtlBuf ffn_norm;    // [H]   f32
        MtlBuf q_a_norm;    // [Hi]  f32
        MtlBuf kv_a_norm;   // [Lk]  f32
        MtlBuf router_bias; // [Ne]  f32; sparse layers only
        Projection q_a, q_b, kv_a, kv_b, attn_output;
        Projection ffn_gate, ffn_up, ffn_down;
        Projection router;
    };
    std::vector<LayerResident> lw_;
    Projection output_projection_;
    MtlBuf fixed_cache_;
    uint64_t fixed_cache_budget_ = 0;
    std::array<MtlBuf, 2> layer_stage_{};
    std::array<std::array<std::atomic<bool>, 9>, 2> layer_ready_{};
    std::array<long long, 2> layer_stage_started_{};
    static constexpr uint32_t kExpertSlots = 24;
    std::array<MtlBuf, kExpertSlots> expert_slots_{};
    std::array<std::atomic<bool>, kExpertSlots> expert_ready_{};
    uint32_t expert_slot_cursor_ = 0;

    // Global resident output norm (f32).
    MtlBuf output_norm_b_;

    // Activations and scratch are FP32 unless noted.
    MtlBuf x_, x_norm_;
    MtlBuf q_a_, q_a_n_;
    MtlBuf q_full_, q_nope_, q_rope_;
    MtlBuf kv_a_, kv_lat_, kv_full_;
    MtlBuf o_full_, attn_out_;
    MtlBuf ffn_gate_, ffn_up_, ffn_act_, ffn_out_, expert_tmp_;
    MtlBuf router_log_;     // [Ne]    f32
    MtlBuf router_idx_;     // [K]     u32
    MtlBuf router_wts_;     // [K]     f32
    MtlBuf logits_;         // [V]     f32
    MtlBuf next_tok_;       // [1]     u32

    // DeepSeek-R1 layer-major prefill tile. At 128 tokens nearly every
    // expert selected by the tile is loaded once instead of once per token.
    MtlBuf x_b_, xn_b_, qa_b_, qan_b_, qf_b_, qn_b_, qr_b_;
    MtlBuf kva_b_, kvlat_b_, kvfull_b_, ofull_b_, attnout_b_;
    MtlBuf fg_b_, fu_b_, fa_b_, fo_b_, rlog_b_, ridx_b_, rwts_b_, routed_b_;

    // KV cache + scores scratch.
    // Separate resources prevent Metal from making a multi-GB monolithic
    // cache resident whenever one layer binds it.
    std::vector<MtlBuf> k_nope_;  // each [max_seq, HE, Dn] f16
    std::vector<MtlBuf> v_cache_; // each [max_seq, HE, Dv] f16
    std::vector<MtlBuf> k_rope_;  // each [max_seq, Dr]     f16
    MtlBuf scores_;         // [HE, max_seq]             f32

    // Vendored codebook grids copied once into Metal-owned shared buffers.
    MtlBuf iq1s_grid_b_;
    MtlBuf iq2xxs_grid_b_;

    // Scratch for CPU dequant of one tensor at a time (sized to the largest
    // tensor we dequant on CPU = attn_kv_b at ~9.4 MB / layer).
    std::vector<uint8_t> dequant_scratch_bytes_;
    std::vector<float>   dequant_scratch_f32_;

    uint64_t step_model_bytes_ = 0;
    uint64_t step_reads_ = 0;
    uint64_t step_allocations_ = 0;
    long long step_io_wait_us_ = 0;
};

} // namespace blade
