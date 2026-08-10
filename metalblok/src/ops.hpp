// Shared dispatch helpers.  One definition, used by mla.cpp and moe.cpp.
// Each helper is one Metal dispatch.  Decode (B=1) and prefill (B>1) share
// the same elementwise kernels (axpy_f16, swiglu_f16) — total-N is total-N.
#pragma once
#include "metal_ctx.hpp"

namespace blade {

inline constexpr uint32_t TG_GEMV = 1024;
inline constexpr uint32_t TG_RED  = 256;
inline constexpr uint32_t TG_ELT  = 256;

// Grouped GEMV: row r reads x[(r/group)*K .. +K].  group=M -> plain GEMV.
inline void gemv(Metal& g, const char* k, const MtlBuf& W, const MtlBuf& x, const MtlBuf& y,
                 uint32_t M, uint32_t K, uint32_t group) {
    g.dispatch(k, {W,x,y}, {{&K,4},{&group,4}}, M, TG_GEMV, true);
}
inline void gemv(Metal& g, const char* k, const MtlBuf& W, const MtlBuf& x, const MtlBuf& y,
                 uint32_t M, uint32_t K) { gemv(g,k,W,x,y,M,K,M); }

// Batched GEMM (bf16 weights only).  Tiled simdgroup-matrix MMA: 32x32 output
// tile per TG (16 simdgroups doing bf16 8*8*8 mma).  Scales compute-bound with
// B; W is read once per 32-M-row tile, so weight bandwidth amortizes ~32x.
// Constraint for grouped use: group_size must be a multiple of 32.
inline void gemm(Metal& g, const MtlBuf& W, const MtlBuf& x, const MtlBuf& y,
                 uint32_t M, uint32_t K, uint32_t B, uint32_t group) {
    constexpr uint32_t BM = 32, BN = 32, THREADS = 512;
    uint32_t gx = (M + BM - 1u) / BM;
    uint32_t gy = (B + BN - 1u) / BN;
    g.dispatch2d("gemm_bf16_f16_mma", {W,x,y},
                 {{&K,4},{&M,4},{&B,4},{&group,4}},
                 gx, gy, THREADS);
}
inline void gemm(Metal& g, const MtlBuf& W, const MtlBuf& x, const MtlBuf& y,
                 uint32_t M, uint32_t K, uint32_t B) { gemm(g,W,x,y,M,K,B,M); }

inline void rmsnorm(Metal& g, const MtlBuf& x, const MtlBuf& gain, const MtlBuf& y,
                    uint32_t H, float eps, uint32_t B = 1) {
    g.dispatch(B == 1 ? "rms_norm_f16" : "rms_norm_f16_b",
               {x, gain, y}, {{&H,4},{&eps,4}}, B, TG_RED, true);
}
inline void axpy(Metal& g, const MtlBuf& y, const MtlBuf& x, float a, uint32_t N) {
    g.dispatch("axpy_f16", {y,x}, {{&a,4},{&N,4}}, N, TG_ELT, false);
}
inline void swiglu(Metal& g, const MtlBuf& gate, const MtlBuf& up, const MtlBuf& y, uint32_t N) {
    g.dispatch("swiglu_f16", {gate,up,y}, {{&N,4}}, N, TG_ELT, false);
}

} // namespace blade
