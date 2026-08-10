#include "runtime.hpp"
#include "streamer.hpp"
#include "ops.hpp"
#include <cstring>

namespace blade {

// Decode-time MoE: ffn_norm -> shared FFN -> router -> top-K -> flush ->
// host reads chosen experts -> per-expert FFN (3 GEMVs + axpy).  One flush
// per MoE layer is the dominant decode-time stall (~2/3 of total step time
// on DeepSeek-V2-Lite per measurement).  Removing it requires a kernel that
// reads router_idx on device AND dereferences expert weight pointers on
// device.  Apple Metal 3 supports raw-VA pointer derefs only for buffers
// passed to useResources:.  Empirical finding: useResources: cost scales
// linearly with #buffers (4992 bufs -> 2.4s/step), and shard-collapsed VAs
// don't get honored for cross-buffer derefs.  Solution path is argument
// buffers; tracked separately.  This decode loop is the proven baseline.
void Runtime::ffn_moe(uint32_t L) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H = c.hidden, Fe = c.expert_ffn;
    const uint32_t Fs = Fe * c.n_shared_experts;
    const uint32_t K = c.n_experts_active, Ne = c.n_experts;
    const uint32_t ml = L - c.n_dense_layers;

    rmsnorm(*G, x, w.ffn_norm, x_norm, H, c.rms_eps);
    gemv(*G, k_gemv, w.w_gate_shared, x_norm, ffn_gate, Fs, H);
    gemv(*G, k_gemv, w.w_up_shared,   x_norm, ffn_up,   Fs, H);
    swiglu(*G, ffn_gate, ffn_up, ffn_act, Fs);
    gemv(*G, k_gemv, w.w_down_shared, ffn_act, ffn_out, H, Fs);
    gemv(*G, k_gemv, w.w_router,      x_norm, router_log, Ne, H);

    uint32_t mode = c.has_router_bias ? 0u : 1u;
    G->dispatch("router_topk_f16",
                {router_log, w.router_bias, router_idx, router_wts},
                {{&Ne,4},{&K,4},{&mode,4}}, 1, 1, true);
    G->flush();   // make router results host-visible

    const uint32_t* idxs = (const uint32_t*)router_idx.contents;
    const float*    wts  = (const float*)   router_wts.contents;
    for (uint32_t k = 0; k < K; ++k) { S->prefetch(ml, idxs[k]); S->touch(ml, idxs[k]); }

    for (uint32_t k = 0; k < K; ++k) {
        uint32_t e = idxs[k];
        gemv(*G, k_gemv, expert_w(ml,e,0), x_norm, ffn_gate, Fe, H);
        gemv(*G, k_gemv, expert_w(ml,e,1), x_norm, ffn_up,   Fe, H);
        swiglu(*G, ffn_gate, ffn_up, ffn_act, Fe);
        gemv(*G, k_gemv, expert_w(ml,e,2), ffn_act, expert_tmp, H, Fe);
        axpy(*G, ffn_out, expert_tmp, wts[k], H);
    }
}

// Prefill MoE: group B tokens by chosen expert (CPU bucket build), then
// per-expert gather -> 3 GEMMs -> scatter-add.  Each expert's weights load
// ONCE per chunk regardless of how many tokens chose it.
void Runtime::ffn_moe_b(uint32_t L, uint32_t B) {
    const auto& c = M->cfg;
    const auto& w = lw[L];
    const uint32_t H = c.hidden, Fe = c.expert_ffn;
    const uint32_t Fs = Fe * c.n_shared_experts;
    const uint32_t K = c.n_experts_active, Ne = c.n_experts;
    const uint32_t ml = L - c.n_dense_layers;

    rmsnorm(*G, x_b, w.ffn_norm, xn_b, H, c.rms_eps, B);
    gemm(*G, w.w_gate_shared, xn_b, fg_b, Fs, H, B);
    gemm(*G, w.w_up_shared,   xn_b, fu_b, Fs, H, B);
    swiglu(*G, fg_b, fu_b, fa_b, B * Fs);
    gemm(*G, w.w_down_shared, fa_b, fo_b, H, Fs, B);
    gemm(*G, w.w_router,      xn_b, rlog_b, Ne, H, B);

    uint32_t mode = c.has_router_bias ? 0u : 1u;
    G->dispatch("router_topk_f16_b",
                {rlog_b, w.router_bias, ridx_b, rwts_b},
                {{&Ne,4},{&K,4},{&mode,4},{&B,4}}, B, 1, true);
    G->flush();   // ridx/rwts host-visible

    // Static-local: zero stack, single runtime thread.  MAX_NE covers every
    // supported model (Kimi K2 = 384).  Flat [Ne, PREFILL_B] layout because
    // each per-expert dispatch (encoded now, executed later) needs its own
    // slot — setBuffer captures by reference, not value.
    constexpr uint32_t MAX_NE = 512;
    static uint32_t cnt[MAX_NE];
    static uint32_t bk_tok[MAX_NE * Runtime::PREFILL_B];
    static float    bk_w  [MAX_NE * Runtime::PREFILL_B];
    if (Ne > MAX_NE) std::abort();

    const uint32_t* idxs = (const uint32_t*)ridx_b.contents;
    const float*    wts  = (const float*)   rwts_b.contents;
    std::memset(cnt, 0, Ne * sizeof(uint32_t));
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t e = idxs[(size_t)b * K + k];
            uint32_t p = cnt[e]++;
            bk_tok[e * Runtime::PREFILL_B + p] = b;
            bk_w  [e * Runtime::PREFILL_B + p] = wts[(size_t)b * K + k];
        }
    for (uint32_t e = 0; e < Ne; ++e) if (cnt[e]) { S->prefetch(ml, e); S->touch(ml, e); }
    std::memcpy(bkt_idx.contents, bk_tok, (size_t)Ne * Runtime::PREFILL_B * 4);
    std::memcpy(bkt_wts.contents, bk_w,   (size_t)Ne * Runtime::PREFILL_B * 4);

    // ao_b reused as scratch — its post-attn value was already added to x_b.
    for (uint32_t e = 0; e < Ne; ++e) {
        uint32_t n_e = cnt[e]; if (!n_e) continue;
        size_t boff = (size_t)e * Runtime::PREFILL_B * 4;
        MtlBuf bidx = bkt_idx; bidx.offset += boff;
        MtlBuf bwts = bkt_wts; bwts.offset += boff;
        G->dispatch("gather_rows_f16", {xn_b, ao_b, bidx},
                    {{&n_e,4},{&H,4}}, n_e * H, TG_ELT, false);
        gemm(*G, expert_w(ml,e,0), ao_b, fg_b, Fe, H, n_e);
        gemm(*G, expert_w(ml,e,1), ao_b, fu_b, Fe, H, n_e);
        swiglu(*G, fg_b, fu_b, fa_b, n_e * Fe);
        gemm(*G, expert_w(ml,e,2), fa_b, ao_b, H, Fe, n_e);
        G->dispatch("scatter_add_weighted_f16", {fo_b, ao_b, bidx, bwts},
                    {{&n_e,4},{&H,4}}, n_e * H, TG_ELT, false);
    }
}

} // namespace blade
