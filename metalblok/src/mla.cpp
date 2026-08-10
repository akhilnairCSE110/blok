#include "runtime.hpp"
#include "streamer.hpp"
#include "ops.hpp"
#include "prof.hpp"
#include <cstring>
#include <cmath>

namespace blade {

void Runtime::init(Model& m, Metal& g, Streamer& s) {
    M = &m; G = &g; S = &s;
    const auto& c = m.cfg;

    // Pick kernel variants based on weight dtype.
    if (c.weight_dtype == 1) {
        k_gemv  = "gemv_bf16_f16";
        k_embed = "embed_lookup_bf16";
    } else {
        k_gemv  = "gemv_fp8_f16";
        k_embed = "embed_lookup_fp8";
    }
    auto wb = [&](size_t n){ return weight_bytes(n, c.weight_dtype); };

    const uint32_t H  = c.hidden;
    const uint32_t Dn = c.head_dim_qk_nope, Dr = c.head_dim_qk_rope, Dv = c.head_dim_v;
    const uint32_t HE = c.n_heads, Lk = c.kv_lora_rank, Hi = c.q_lora_rank;
    const uint32_t F_dense = c.ffn_dense;
    const uint32_t F_exp   = c.expert_ffn;
    const uint32_t F_max   = std::max(F_dense, F_exp * c.n_shared_experts);

    auto fp16 = [&](size_t n){ return g.alloc(n * 2); };
    x          = fp16(H);
    x_norm     = fp16(H);
    // Q-LoRA scratch only used when q_lora_rank > 0; allocate min size to keep
    // descriptors valid (Metal rejects zero-byte buffers).
    q_a        = fp16(Hi > 0 ? Hi : 1);
    q_a_n      = fp16(Hi > 0 ? Hi : 1);
    q_full     = fp16((size_t)HE * (Dn + Dr));
    q_nope_buf = fp16((size_t)HE * Dn);
    q_rope_buf = fp16((size_t)HE * Dr);
    kv_a       = fp16((size_t)Lk + Dr);
    q_eff      = fp16((size_t)HE * Lk);
    o_lat      = fp16((size_t)HE * Lk);
    o_full     = fp16((size_t)HE * Dv);
    attn_out   = fp16(H);
    ffn_gate   = fp16(F_max);
    ffn_up     = fp16(F_max);
    ffn_act    = fp16(F_max);
    ffn_out    = fp16(H);
    expert_tmp = fp16(H);
    router_log = fp16(c.n_experts);
    router_idx = g.alloc((size_t)c.n_experts_active * 4);
    router_wts = g.alloc((size_t)c.n_experts_active * 4);
    logits     = fp16(c.vocab);
    next_tok   = g.alloc(4);

    // Batched prefill scratch.  Sized for B = PREFILL_B; each is loaded
    // ONCE per chunk by every weight matmul -> arithmetic intensity B-fold.
    const uint32_t B = PREFILL_B;
    tokens_b = g.alloc((size_t)B * 4);
    x_b      = fp16((size_t)B * H);
    xn_b     = fp16((size_t)B * H);
    qa_b     = fp16((size_t)B * (Hi > 0 ? Hi : 1));
    qan_b    = fp16((size_t)B * (Hi > 0 ? Hi : 1));
    qf_b     = fp16((size_t)B * HE * (Dn + Dr));
    qn_b     = fp16((size_t)B * HE * Dn);
    qr_b     = fp16((size_t)B * HE * Dr);
    kva_b    = fp16((size_t)B * (Lk + Dr));
    qeff_b   = fp16((size_t)B * HE * Lk);
    olat_b   = fp16((size_t)B * HE * Lk);
    ofull_b  = fp16((size_t)B * HE * Dv);
    ao_b     = fp16((size_t)B * H);
    fg_b     = fp16((size_t)B * F_max);
    fu_b     = fp16((size_t)B * F_max);
    fa_b     = fp16((size_t)B * F_max);
    fo_b     = fp16((size_t)B * H);
    rlog_b   = fp16((size_t)B * c.n_experts);
    ridx_b   = g.alloc((size_t)B * c.n_experts_active * 4);
    rwts_b   = g.alloc((size_t)B * c.n_experts_active * 4);
    // bkt_*: per-expert bucket of (token_idx, weight) for grouped MoE.  Sized
    // for ALL experts at once and written CPU-side BEFORE any dispatch so
    // each per-expert dispatch (encoded then later executed) reads its own
    // slot.  Otherwise: setBuffer captures a reference, not contents -> all
    // dispatches would see only the LAST iteration's data at GPU run time.
    bkt_idx  = g.alloc((size_t)c.n_experts * B * 4);
    bkt_wts  = g.alloc((size_t)c.n_experts * B * 4);

    // KV caches.  Latent c_kv [n_layers, max_seq, Lk] fp16.
    c_kv   = fp16((size_t)c.n_layers * c.max_seq * Lk);
    k_rope = fp16((size_t)c.n_layers * c.max_seq * Dr);
    // Score scratch [HE, max_seq] fp32 — reused across layers.
    scores = g.alloc((size_t)HE * c.max_seq * 4);

    // Wrap embedding + final norm + per-layer hot weights as zero-copy buffers.
    w_embed_b    = g.wrap(m.w_embed,    wb((size_t)c.vocab * H));
    final_norm_b = g.wrap(m.final_norm, (size_t)H * 2);
    if (c.tied_embed) {
        lm_head_b = w_embed_b;
    } else {
        lm_head_b = g.wrap(m.lm_head, wb((size_t)c.vocab * H));
    }
    // Zero bias for V2-style routing (no per-expert correction).
    zero_bias_b = g.alloc((size_t)c.n_experts * sizeof(float));
    std::memset(zero_bias_b.contents, 0, (size_t)c.n_experts * sizeof(float));

    lw.resize(c.n_layers);
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const auto& src = m.layers[L];
        auto& d = lw[L];
        d.attn_norm = g.wrap(src.attn_norm, H * 2);
        d.ffn_norm  = g.wrap(src.ffn_norm,  H * 2);
        if (Hi > 0) {
            d.w_q_a    = g.wrap(src.w_q_a,    wb((size_t)Hi * H));
            d.q_a_norm = g.wrap(src.q_a_norm, Hi * 2);
            d.w_q_b    = g.wrap(src.w_q_b,    wb((size_t)HE * (Dn + Dr) * Hi));
        } else {
            d.w_q      = g.wrap(src.w_q,      wb((size_t)HE * (Dn + Dr) * H));
        }
        d.w_kv_a    = g.wrap(src.w_kv_a,    wb((size_t)(Lk + Dr) * H));
        d.kv_a_norm = g.wrap(src.kv_a_norm, Lk * 2);
        d.w_uk      = g.wrap(src.w_uk,      wb((size_t)HE * Lk * Dn));
        d.w_uv      = g.wrap(src.w_uv,      wb((size_t)HE * Dv * Lk));
        d.w_o       = g.wrap(src.w_o,       wb((size_t)H * HE * Dv));
        if (L < c.n_dense_layers) {
            d.w_gate_dense = g.wrap(src.w_gate_dense, wb((size_t)F_dense * H));
            d.w_up_dense   = g.wrap(src.w_up_dense,   wb((size_t)F_dense * H));
            d.w_down_dense = g.wrap(src.w_down_dense, wb((size_t)H * F_dense));
        } else {
            d.w_gate_shared = g.wrap(src.w_gate_shared, wb((size_t)F_exp * c.n_shared_experts * H));
            d.w_up_shared   = g.wrap(src.w_up_shared,   wb((size_t)F_exp * c.n_shared_experts * H));
            d.w_down_shared = g.wrap(src.w_down_shared, wb((size_t)H * F_exp * c.n_shared_experts));
            d.w_router      = g.wrap(src.w_router,      wb((size_t)c.n_experts * H));
            if (c.has_router_bias) {
                d.router_bias = g.wrap(src.router_bias, (size_t)c.n_experts * 4);
            } else {
                d.router_bias = zero_bias_b;
            }
        }
    }

    // Pre-wrap every routed expert's three sub-tensors (gate, up, down).
    // Two source layouts:
    //   FP8 (.blade):  experts.bin + experts.idx -> (offset, nbytes) per expert.
    //   bf16 (HF):     Model::open_hf populated Model::raw_experts with three
    //                  pointers (gate/up/down) each pointing into a mmap'd
    //                  safetensors shard.  No nbytes table; sizes computed
    //                  from cfg directly.
    size_t n_moe = c.n_layers - c.n_dense_layers;
    expert_bufs.resize(n_moe * c.n_experts * 3);
    if (c.weight_dtype == 1) {
        const size_t bytes_gu = wb((size_t)c.expert_ffn * H);
        const size_t bytes_dn = wb((size_t)H * c.expert_ffn);
        for (uint32_t ml = 0; ml < n_moe; ++ml) {
            for (uint32_t e = 0; e < c.n_experts; ++e) {
                const auto& re = m.raw_experts[(size_t)ml * c.n_experts + e];
                size_t i = ((size_t)ml * c.n_experts + e) * 3;
                expert_bufs[i + 0] = g.wrap(re.gate, bytes_gu);
                expert_bufs[i + 1] = g.wrap(re.up,   bytes_gu);
                expert_bufs[i + 2] = g.wrap(re.down, bytes_dn);
            }
        }
    } else {
        for (uint32_t ml = 0; ml < n_moe; ++ml) {
            for (uint32_t e = 0; e < c.n_experts; ++e) {
                const auto& ent = m.expert_idx[(size_t)ml * c.n_experts + e];
                const uint8_t* base = m.experts_base + ent.offset;
                size_t off = 0;
                size_t i = ((size_t)ml * c.n_experts + e) * 3;
                expert_bufs[i + 0] = g.wrap(base + off, m.esz.gate); off += m.esz.gate;
                expert_bufs[i + 1] = g.wrap(base + off, m.esz.up);   off += m.esz.up;
                expert_bufs[i + 2] = g.wrap(base + off, m.esz.down);
            }
        }
    }
    pos = 0;
    prof::log("runtime: %zu expert sub-tensors wrapped, rss=%zuMB",
              expert_bufs.size(), prof::rss_mb());
}

uint32_t Runtime::step(uint32_t token) {
    const auto& c = M->cfg;
    long long t0 = prof::now_us();
    G->reset_step_stats();
    G->begin();
    uint32_t H = c.hidden;
    G->dispatch(k_embed, {w_embed_b, x},
                {{&token, sizeof(uint32_t)}, {&H, sizeof(uint32_t)}},
                H, TG_ELT, /*one_tg_per_grid_x=*/false);

    // Seed the sliding-window: we're about to enter block(0), so warm 0 and 1.
    // block(L) itself will then advance the window by hinting L+1.
    S->prefetch_layer(0);
    if (c.n_layers > 1) S->prefetch_layer(1);

    for (uint32_t L = 0; L < c.n_layers; ++L) block(L);

    // Final norm + lm_head + greedy argmax.
    rmsnorm(*G, x, final_norm_b, x_norm, H, c.rms_eps);
    gemv(*G, k_gemv, lm_head_b, x_norm, logits, c.vocab, H);
    uint32_t V = c.vocab;
    G->dispatch("argmax_f16", {logits, next_tok}, {{&V, sizeof(uint32_t)}},
                1, 1024, /*one_tg_per_grid_x=*/true);
    G->commit_and_wait();
    pos++;
    long long us = prof::now_us() - t0;
    long long sync_us = us - G->step_gpu_us;
    prof::log("decode pos=%u: %lld us  gpu=%lld (%d%%)  sync=%lld  cmd=%d  rss=%zuMB",
              pos, us, G->step_gpu_us,
              (int)(100 * G->step_gpu_us / (us ? us : 1)),
              sync_us, G->step_cmdbufs, prof::rss_mb());
    return *(uint32_t*)next_tok.contents;
}

void Runtime::block(uint32_t L) {
    // Sliding-window prefetch of layer L+1's weights from NVMe.
    if (L + 1 < M->cfg.n_layers) S->prefetch_layer(L + 1);
    mla_attn(L);
    axpy(*G, x, attn_out, 1.0f, M->cfg.hidden);
    if (L < M->cfg.n_dense_layers) ffn_dense(L);
    else                           ffn_moe(L);
    axpy(*G, x, ffn_out, 1.0f, M->cfg.hidden);
}

// MLA decode attention.  T = pos+1 KV positions; q is the single new token.
void Runtime::mla_attn(uint32_t L) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H  = c.hidden;
    const uint32_t Dn = c.head_dim_qk_nope, Dr = c.head_dim_qk_rope, Dv = c.head_dim_v;
    const uint32_t HE = c.n_heads, Lk = c.kv_lora_rank, Hi = c.q_lora_rank;
    uint32_t pos_u = pos, L_u = L, ms = c.max_seq, T = pos + 1;
    float th = c.rope_theta, eps = c.rms_eps;
    float scale = (1.0f / std::sqrt((float)(Dn + Dr))) * c.yarn_mscale;

    rmsnorm(*G, x, w.attn_norm, x_norm, H, c.rms_eps);
    if (Hi > 0) {
        gemv   (*G, k_gemv, w.w_q_a, x_norm, q_a, Hi, H);
        rmsnorm(*G, q_a, w.q_a_norm, q_a_n, Hi, c.rms_eps);
        gemv   (*G, k_gemv, w.w_q_b, q_a_n, q_full, HE * (Dn + Dr), Hi);
    } else {
        gemv   (*G, k_gemv, w.w_q,   x_norm, q_full, HE * (Dn + Dr), H);
    }
    G->dispatch("mla_q_split_rope", {q_full, q_nope_buf, q_rope_buf},
                {{&Dn,4},{&Dr,4},{&pos_u,4},{&th,4}}, HE, TG_ELT, true);
    gemv(*G, k_gemv, w.w_uk,   q_nope_buf, q_eff, HE * Lk, Dn, Lk);  // per-head W_uk
    gemv(*G, k_gemv, w.w_kv_a, x_norm,     kv_a,  Lk + Dr, H);
    G->dispatch("mla_kv_split_rope", {kv_a, w.kv_a_norm, c_kv, k_rope},
                {{&Lk,4},{&Dr,4},{&L_u,4},{&pos_u,4},{&ms,4},{&eps,4},{&th,4}},
                1, TG_RED, true);
    G->dispatch("mla_attn_decode_f16",
                {q_eff, q_rope_buf, c_kv, k_rope, o_lat, scores},
                {{&HE,4},{&Lk,4},{&Dr,4},{&L_u,4},{&T,4},{&ms,4},{&scale,4}},
                HE, TG_RED, true);
    gemv(*G, k_gemv, w.w_uv, o_lat,  o_full,   HE * Dv, Lk, Dv);     // per-head W_uv
    gemv(*G, k_gemv, w.w_o,  o_full, attn_out, H,       HE * Dv);
}

void Runtime::ffn_dense(uint32_t L) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H = c.hidden, F = c.ffn_dense;
    rmsnorm(*G, x, w.ffn_norm, x_norm, H, c.rms_eps);
    gemv(*G, k_gemv, w.w_gate_dense, x_norm, ffn_gate, F, H);
    gemv(*G, k_gemv, w.w_up_dense,   x_norm, ffn_up,   F, H);
    swiglu(*G, ffn_gate, ffn_up, ffn_act, F);
    gemv(*G, k_gemv, w.w_down_dense, ffn_act, ffn_out, H, F);
}

// ============================================================================
// Batched prefill.  Loads each weight matrix ONCE per B-token chunk via
// gemm_bf16_f16 (BT=8).  Decode GEMV path untouched.  bf16 only.
// ============================================================================

void Runtime::mla_attn_b(uint32_t L, uint32_t B) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H  = c.hidden;
    const uint32_t Dn = c.head_dim_qk_nope, Dr = c.head_dim_qk_rope, Dv = c.head_dim_v;
    const uint32_t HE = c.n_heads, Lk = c.kv_lora_rank, Hi = c.q_lora_rank;

    rmsnorm(*G, x_b, w.attn_norm, xn_b, H, c.rms_eps, B);
    if (Hi > 0) {
        gemm(*G, w.w_q_a, xn_b, qa_b, Hi, H, B);
        rmsnorm(*G, qa_b, w.q_a_norm, qan_b, Hi, c.rms_eps, B);
        gemm(*G, w.w_q_b, qan_b, qf_b, HE * (Dn + Dr), Hi, B);
    } else {
        gemm(*G, w.w_q, xn_b, qf_b, HE * (Dn + Dr), H, B);
    }
    uint32_t pos_u = pos, L_u = L, ms = c.max_seq;
    float th = c.rope_theta, eps = c.rms_eps;
    float scale = (1.0f / std::sqrt((float)(Dn + Dr))) * c.yarn_mscale;
    G->dispatch("mla_q_split_rope_b", {qf_b, qn_b, qr_b},
                {{&Dn,4},{&Dr,4},{&HE,4},{&pos_u,4},{&th,4}},
                B * HE, TG_ELT, true);
    gemm(*G, w.w_uk, qn_b, qeff_b, HE * Lk, Dn, B, Lk);   // per-head W_uk
    gemm(*G, w.w_kv_a, xn_b, kva_b, Lk + Dr, H, B);
    G->dispatch("mla_kv_split_rope_b", {kva_b, w.kv_a_norm, c_kv, k_rope},
                {{&Lk,4},{&Dr,4},{&L_u,4},{&pos_u,4},{&ms,4},{&eps,4},{&th,4}},
                B, TG_RED, true);
    G->dispatch("mla_attn_prefill_f16",
                {qeff_b, qr_b, c_kv, k_rope, olat_b},
                {{&HE,4},{&Lk,4},{&Dr,4},{&L_u,4},{&pos_u,4},{&ms,4},{&scale,4}},
                B * HE, TG_RED, true);
    gemm(*G, w.w_uv, olat_b,  ofull_b, HE * Dv, Lk,      B, Dv);  // per-head W_uv
    gemm(*G, w.w_o,  ofull_b, ao_b,    H,       HE * Dv, B);
}

void Runtime::ffn_dense_b(uint32_t L, uint32_t B) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H = c.hidden, F = c.ffn_dense;
    rmsnorm(*G, x_b, w.ffn_norm, xn_b, H, c.rms_eps, B);
    gemm(*G, w.w_gate_dense, xn_b, fg_b, F, H, B);
    gemm(*G, w.w_up_dense,   xn_b, fu_b, F, H, B);
    swiglu(*G, fg_b, fu_b, fa_b, B * F);
    gemm(*G, w.w_down_dense, fa_b, fo_b, H, F, B);
}

void Runtime::block_b(uint32_t L, uint32_t B) {
    if (L + 1 < M->cfg.n_layers) S->prefetch_layer(L + 1);
    mla_attn_b(L, B);
    axpy(*G, x_b, ao_b, 1.0f, B * M->cfg.hidden);
    if (L < M->cfg.n_dense_layers) ffn_dense_b(L, B);
    else                           ffn_moe_b(L, B);
    axpy(*G, x_b, fo_b, 1.0f, B * M->cfg.hidden);
}

void Runtime::prefill_chunk(const uint32_t* ids, uint32_t B) {
    const auto& c = M->cfg;
    uint32_t H = c.hidden;
    std::memcpy(tokens_b.contents, ids, (size_t)B * 4);
    G->begin();
    G->dispatch("embed_lookup_bf16_b", {w_embed_b, x_b, tokens_b},
                {{&B, sizeof(uint32_t)}, {&H, sizeof(uint32_t)}},
                B * H, TG_ELT, /*one_tg_per_grid_x=*/false);
    S->prefetch_layer(0);
    if (c.n_layers > 1) S->prefetch_layer(1);
    for (uint32_t L = 0; L < c.n_layers; ++L) block_b(L, B);

    // LM head only on the LAST position; intermediate logits unused.
    MtlBuf last_x = x_b; last_x.offset = x_b.offset + (size_t)(B - 1) * H * 2;
    rmsnorm(*G, last_x, final_norm_b, x_norm, H, c.rms_eps);
    gemv(*G, k_gemv, lm_head_b, x_norm, logits, c.vocab, H);
    uint32_t V = c.vocab;
    G->dispatch("argmax_f16", {logits, next_tok}, {{&V,4}}, 1, 1024, true);
    G->commit_and_wait();
    pos += B;
}

uint32_t Runtime::prefill(const uint32_t* ids, uint32_t n) {
    // Batched path requires bf16 weights (gemm_bf16_f16) and amortizes per-
    // expert dispatch overhead when avg tokens/chosen-expert is high enough.
    // Crossover = 4 * Ne / K_active (V2-Lite: 42 tok); below it, per-token
    // decode wins.  FP8 weights also fall through.
    const auto& c = M->cfg;
    uint32_t crossover = 4u * c.n_experts / std::max<uint32_t>(c.n_experts_active, 1u);
    if (c.weight_dtype != 1 || n < crossover) {
        uint32_t next = 0;
        for (uint32_t i = 0; i < n; ++i) next = step(ids[i]);
        return next;
    }
    for (uint32_t i = 0; i < n; ) {
        uint32_t B = std::min<uint32_t>(PREFILL_B, n - i);
        char what[64];
        std::snprintf(what, sizeof(what), "prefill chunk pos=%u B=%u", pos, B);
        prof::mark(what);
        G->reset_step_stats();
        long long t0 = prof::now_us();
        prefill_chunk(ids + i, B);
        long long us = prof::now_us() - t0;
        prof::log("prefill chunk pos=%u B=%u: %lld us  gpu=%lld (%d%%)  cmd=%d  rss=%zuMB  tok/s=%.1f",
                  pos - B, B, us, G->step_gpu_us,
                  (int)(100 * G->step_gpu_us / (us ? us : 1)),
                  G->step_cmdbufs, prof::rss_mb(), B * 1e6 / us);
        i += B;
    }
    return *(uint32_t*)next_tok.contents;
}

} // namespace blade

