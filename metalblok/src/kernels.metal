// blade Metal kernels.
//
// Conventions
//   - All matmuls are GEMV (decode batch=1).
//   - Weights: block-FP8 E4M3 with one fp32 scale per 128 elements (Fp8Block).
//   - Activations / KV cache: fp16.
//   - One threadgroup per output row for GEMV; TG_GEMV threads cooperate via
//     SIMD reduce -> threadgroup-scratch -> SIMD reduce.
//   - All shared/threadgroup reductions follow the same idiom (no atomics):
//        local = ...
//        local = simd_sum/max(local)
//        if (lane==0) tg_warp[warp] = local
//        barrier
//        if (warp==0) reduce tg_warp -> tg_scalar
//        barrier
//        v = tg_scalar
//
// All kernel uniforms are passed via setBytes (so the kernel signature lists
// them as `constant T&`).  All buffers are MTLResourceStorageModeShared.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// (TG_GEMV is set on the host side via dispatchThreadgroups; the kernels
//  read `threads_per_threadgroup` so they don't need a duplicate constant.)

// ---------- FP8 E4M3 dequant -----------------------------------------------
// E4M3: 1 sign, 4 exp (bias 7), 3 mantissa.  Subnormals when exp==0.
inline float fp8_e4m3_to_f32(uchar b) {
    uint s = (b >> 7) & 1u;
    uint e = (b >> 3) & 0xfu;
    uint m =  b       & 0x7u;
    float v = (e == 0)
        ? ldexp(float(m) / 8.0f, -6)                     // subnormal
        : ldexp(1.0f + float(m) / 8.0f, int(e) - 7);     // normal
    return s ? -v : v;
}

struct Fp8Block { uchar q[128]; float scale; };

// ---------- threadgroup reduce helpers -------------------------------------
// Reduce 'val' across all threads in the threadgroup; returns the result on
// every thread.  Uses up to 32 warps (1024 threads).
inline float tg_reduce_sum(float val, uint tid, uint tgs,
                           threadgroup float* scratch /*[32]*/) {
    val = simd_sum(val);
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) scratch[warp] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        uint nwarps = (tgs + 31u) >> 5;
        float v = (tid < nwarps) ? scratch[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
inline float tg_reduce_max(float val, uint tid, uint tgs,
                           threadgroup float* scratch /*[32]*/) {
    val = simd_max(val);
    uint lane = tid & 31u;
    uint warp = tid >> 5;
    if (lane == 0) scratch[warp] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        uint nwarps = (tgs + 31u) >> 5;
        float v = (tid < nwarps) ? scratch[tid] : -INFINITY;
        v = simd_max(v);
        if (tid == 0) scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

// ---------- RMSNorm --------------------------------------------------------
// y = x * rsqrt(mean(x^2) + eps) * gain.   x, y, gain all fp16, length H.
kernel void rms_norm_f16(
    device const half* x    [[buffer(0)]],
    device const half* gain [[buffer(1)]],
    device       half* y    [[buffer(2)]],
    constant uint&     H    [[buffer(3)]],
    constant float&    eps  [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < H; i += tgs) { float v = float(x[i]); acc += v*v; }
    float ss = tg_reduce_sum(acc, tid, tgs, scratch);
    float inv = rsqrt(ss / float(H) + eps);
    for (uint i = tid; i < H; i += tgs)
        y[i] = half(float(x[i]) * inv * float(gain[i]));
}

kernel void rms_norm_f32(
    device const float* x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device       float* y    [[buffer(2)]],
    constant uint&      H    [[buffer(3)]],
    constant float&     eps  [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < H; i += tgs) acc += x[i] * x[i];
    float inv = rsqrt(tg_reduce_sum(acc, tid, tgs, scratch) / float(H) + eps);
    for (uint i = tid; i < H; i += tgs) y[i] = x[i] * inv * gain[i];
}

// The same reduction tree as rms_norm_f32, with one independent threadgroup
// per prompt row. Batching changes submission only, never arithmetic order.
kernel void rms_norm_f32_b(
    device const float* x    [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device       float* y    [[buffer(2)]],
    constant uint&      H    [[buffer(3)]],
    constant float&     eps  [[buffer(4)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xb = x + (size_t)b * H;
    device float* yb = y + (size_t)b * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += tgs) acc += xb[i] * xb[i];
    float inv = rsqrt(tg_reduce_sum(acc, tid, tgs, scratch) / float(H) + eps);
    for (uint i = tid; i < H; i += tgs) yb[i] = xb[i] * inv * gain[i];
}

// ---------- FP8 GEMV: y[M] = W[M,K] @ x[ row/group_size , : ] -------------
// One TG per output row.  Pass group_size = M for a plain (non-grouped) GEMV
// (every row then reads x from offset 0).  Grouped form is used by the MLA
// per-head absorb projections (W_uk/W_uv).  K must be a multiple of 128.
kernel void gemv_fp8_f16(
    device const Fp8Block* W          [[buffer(0)]],
    device const half*     x          [[buffer(1)]],
    device       half*     y          [[buffer(2)]],
    constant uint&         K          [[buffer(3)]],
    constant uint&         group_size [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    uint nblk = K / 128;
    device const Fp8Block* Wr = W + (size_t)row * nblk;
    device const half*     xh = x + (size_t)(row / group_size) * K;
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        const device Fp8Block& blk = Wr[b];
        device const half* xb = xh + b * 128;
        float local = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 128; i += 8) {
            float a0 = fp8_e4m3_to_f32(blk.q[i+0]);
            float a1 = fp8_e4m3_to_f32(blk.q[i+1]);
            float a2 = fp8_e4m3_to_f32(blk.q[i+2]);
            float a3 = fp8_e4m3_to_f32(blk.q[i+3]);
            float a4 = fp8_e4m3_to_f32(blk.q[i+4]);
            float a5 = fp8_e4m3_to_f32(blk.q[i+5]);
            float a6 = fp8_e4m3_to_f32(blk.q[i+6]);
            float a7 = fp8_e4m3_to_f32(blk.q[i+7]);
            local += a0*float(xb[i+0]) + a1*float(xb[i+1])
                   + a2*float(xb[i+2]) + a3*float(xb[i+3])
                   + a4*float(xb[i+4]) + a5*float(xb[i+5])
                   + a6*float(xb[i+6]) + a7*float(xb[i+7]);
        }
        acc += local * blk.scale;
    }
    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = half(v);
}

// ---------- Q split + decoupled-rope RoPE (fused) --------------------------
// q_full layout per head: [Dn|Dr]  (Dr = head_dim_qk_rope, must be even).
//   q_nope_out[h, 0..Dn) = q_full[h, 0..Dn)
//   q_rope_out[h, 0..Dr) = RoPE(pos, q_full[h, Dn..Dn+Dr))
// One TG per head.
kernel void mla_q_split_rope(
    device const half* q_full     [[buffer(0)]],   // [HE, Dn+Dr]
    device       half* q_nope_out [[buffer(1)]],   // [HE, Dn]
    device       half* q_rope_out [[buffer(2)]],   // [HE, Dr]
    constant uint&  Dn            [[buffer(3)]],
    constant uint&  Dr            [[buffer(4)]],
    constant uint&  pos           [[buffer(5)]],
    constant float& theta         [[buffer(6)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const half* qf = q_full     + (size_t)h * (Dn + Dr);
    device       half* qn = q_nope_out + (size_t)h * Dn;
    device       half* qr = q_rope_out + (size_t)h * Dr;

    for (uint i = tid; i < Dn; i += tgs) qn[i] = qf[i];

    // DeepSeek's q_proj / q_b_proj output stores the rope dims in INTERLEAVED
    // pair format [a0,b0,a1,b1,...].  Apply pair rotation:
    //   new_a = a*cos - b*sin
    //   new_b = b*cos + a*sin
    // Q and K go through the same convention so dot product remains valid.
    uint half_dr = Dr / 2;
    for (uint i = tid; i < half_dr; i += tgs) {
        float freq = pow(theta, -2.0f * float(i) / float(Dr));
        float ang  = float(pos) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(qf[Dn + 2u*i]);
        float b = float(qf[Dn + 2u*i + 1u]);
        qr[2u*i]      = half(a*c - b*s);
        qr[2u*i + 1u] = half(b*c + a*s);
    }
}

// ---------- KV split + latent RMSNorm + decoupled K RoPE + KV-cache write --
// kv_a layout: [Lk | Dr] fp16 (output of W_kv_a GEMV).
//   c_kv_cache[layer, pos, :Lk] = RMSNorm(kv_a[:Lk]) * gain
//   k_rope_cache[layer, pos, :Dr] = RoPE(pos, kv_a[Lk:Lk+Dr])
// Single threadgroup; tg_reduce computes the RMS sum.
kernel void mla_kv_split_rope(
    device const half* kv_a         [[buffer(0)]],   // [Lk + Dr]
    device const half* kv_a_norm    [[buffer(1)]],   // [Lk] gain
    device       half* c_kv_cache   [[buffer(2)]],   // [n_layers, max_seq, Lk]
    device       half* k_rope_cache [[buffer(3)]],   // [n_layers, max_seq, Dr]
    constant uint&  Lk      [[buffer(4)]],
    constant uint&  Dr      [[buffer(5)]],
    constant uint&  L       [[buffer(6)]],
    constant uint&  pos     [[buffer(7)]],
    constant uint&  max_seq [[buffer(8)]],
    constant float& eps     [[buffer(9)]],
    constant float& theta   [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    // RMS sum over the latent half.
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) { float v = float(kv_a[i]); acc += v*v; }
    float ss = tg_reduce_sum(acc, tid, tgs, scratch);
    float inv = rsqrt(ss / float(Lk) + eps);

    // Write normed latent into c_kv_cache[L, pos, :].
    device half* cdst = c_kv_cache + ((size_t)L * max_seq + pos) * Lk;
    for (uint i = tid; i < Lk; i += tgs)
        cdst[i] = half(float(kv_a[i]) * inv * float(kv_a_norm[i]));

    // RoPE the decoupled K rope half into k_rope_cache[L, pos, :].
    // Same INTERLEAVED-pair convention as the Q-rope path above.
    device half* krdst = k_rope_cache + ((size_t)L * max_seq + pos) * Dr;
    uint half_dr = Dr / 2;
    for (uint i = tid; i < half_dr; i += tgs) {
        float freq = pow(theta, -2.0f * float(i) / float(Dr));
        float ang  = float(pos) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(kv_a[Lk + 2u*i]);
        float b = float(kv_a[Lk + 2u*i + 1u]);
        krdst[2u*i]      = half(a*c - b*s);
        krdst[2u*i + 1u] = half(b*c + a*s);
    }
}

// ---------- DeepSeek-R1 YaRN RoPE (consecutive pairs) ----------------------
// llama.cpp classifies DEEPSEEK2 as LLAMA_ROPE_TYPE_NORMAL: dimensions are
// paired consecutively (0,1), (2,3), ... . This is independent of GGUF's
// tensor layout. Split-half/NEOX pairing corrupts attention after position 0.
// The pinned R1 GGUF also declares YaRN factor=40, original_context=4096,
// beta_fast=32, beta_slow=1. For Dr=64 this gives correction pairs [10,23].
// ggml's angle is theta_extrap * (0.025 + 0.975*ramp), where ramp falls
// linearly from one to zero across that interval. Its RoPE magnitude is one:
// the DeepSeek attention mscale is applied once in the QK softmax scale.
inline float deepseek_r1_yarn_angle(uint pos, uint pair, uint Dr, float theta) {
    constexpr float freq_scale = 0.025f;
    constexpr float corr_low = 10.0f;
    constexpr float corr_high = 23.0f;
    float ramp = 1.0f - clamp((float(pair) - corr_low) /
                              (corr_high - corr_low), 0.0f, 1.0f);
    float extrap = float(pos) * pow(theta, -2.0f * float(pair) / float(Dr));
    return extrap * (freq_scale + (1.0f - freq_scale) * ramp);
}

kernel void mla_q_split_rope_r1(
    device const float* q_full     [[buffer(0)]],   // [HE, Dn+Dr]
    device       float* q_nope_out [[buffer(1)]],   // [HE, Dn]
    device       float* q_rope_out [[buffer(2)]],   // [HE, Dr]
    constant uint&  Dn            [[buffer(3)]],
    constant uint&  Dr            [[buffer(4)]],
    constant uint&  pos           [[buffer(5)]],
    constant float& theta         [[buffer(6)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const float* qf = q_full     + (size_t)h * (Dn + Dr);
    device       float* qn = q_nope_out + (size_t)h * Dn;
    device       float* qr = q_rope_out + (size_t)h * Dr;

    for (uint i = tid; i < Dn; i += tgs) qn[i] = qf[i];

    uint pairs = Dr / 2;
    for (uint i = tid; i < pairs; i += tgs) {
        float ang = deepseek_r1_yarn_angle(pos, i, Dr, theta);
        float c = cos(ang), s = sin(ang);
        float a = qf[Dn + 2u*i];
        float b = qf[Dn + 2u*i + 1u];
        qr[2u*i] = a*c - b*s;
        qr[2u*i + 1u] = b*c + a*s;
    }
}

kernel void mla_q_split_rope_r1_b(
    device const float* q_full [[buffer(0)]],
    device float* q_nope_out [[buffer(1)]],
    device float* q_rope_out [[buffer(2)]],
    constant uint& Dn [[buffer(3)]],
    constant uint& Dr [[buffer(4)]],
    constant uint& HE [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint h = group.x, token = group.y, position = pos + token;
    device const float* qf = q_full + ((size_t)token * HE + h) * (Dn + Dr);
    device float* qn = q_nope_out + ((size_t)token * HE + h) * Dn;
    device float* qr = q_rope_out + ((size_t)token * HE + h) * Dr;
    for (uint i = tid; i < Dn; i += 256) qn[i] = qf[i];
    for (uint i = tid; i < Dr / 2; i += 256) {
        const float angle = deepseek_r1_yarn_angle(position, i, Dr, theta);
        const float c = cos(angle), s = sin(angle);
        const float a = qf[Dn + 2u * i], b = qf[Dn + 2u * i + 1u];
        qr[2u * i] = a * c - b * s;
        qr[2u * i + 1u] = b * c + a * s;
    }
}

kernel void mla_kv_split_rope_neox(
    device const float* kv_a         [[buffer(0)]],   // [Lk + Dr]
    device const float* kv_a_norm    [[buffer(1)]],   // [Lk] gain
    device       half* c_kv_cache   [[buffer(2)]],   // [n_layers, max_seq, Lk]
    device       half* k_rope_cache [[buffer(3)]],   // [n_layers, max_seq, Dr]
    constant uint&  Lk      [[buffer(4)]],
    constant uint&  Dr      [[buffer(5)]],
    constant uint&  L       [[buffer(6)]],
    constant uint&  pos     [[buffer(7)]],
    constant uint&  max_seq [[buffer(8)]],
    constant float& eps     [[buffer(9)]],
    constant float& theta   [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) { float v = kv_a[i]; acc += v*v; }
    float ss = tg_reduce_sum(acc, tid, tgs, scratch);
    float inv = rsqrt(ss / float(Lk) + eps);

    device half* cdst = c_kv_cache + ((size_t)L * max_seq + pos) * Lk;
    for (uint i = tid; i < Lk; i += tgs)
        cdst[i] = half(kv_a[i] * inv * kv_a_norm[i]);

    device half* krdst = k_rope_cache + ((size_t)L * max_seq + pos) * Dr;
    uint m = Dr / 2;   // split-half pairing: (i, i+m)
    for (uint i = tid; i < m; i += tgs) {
        float ang = deepseek_r1_yarn_angle(pos, i, Dr, theta);
        float c = cos(ang), s = sin(ang);
        float a = kv_a[Lk + i];
        float b = kv_a[Lk + i + m];
        krdst[i]     = half(a*c - b*s);
        krdst[i + m] = half(b*c + a*s);
    }
}

// Legacy combined-wkv GGUF path. Keep the normalized latent in fp32 for the
// quantized wkv_b GEMV, while caching the shared rotary key in fp16.
kernel void mha_kv_norm_rope_r1(
    device const float* kv_a         [[buffer(0)]],
    device const float* kv_a_norm    [[buffer(1)]],
    device       float* kv_lat       [[buffer(2)]],
    device       half*  k_rope_cache [[buffer(3)]],
    constant uint&  Lk      [[buffer(4)]],
    constant uint&  Dr      [[buffer(5)]],
    constant uint&  L       [[buffer(6)]],
    constant uint&  pos     [[buffer(7)]],
    constant uint&  max_seq [[buffer(8)]],
    constant float& eps     [[buffer(9)]],
    constant float& theta   [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) acc += kv_a[i] * kv_a[i];
    float inv = rsqrt(tg_reduce_sum(acc, tid, tgs, scratch) / float(Lk) + eps);
    for (uint i = tid; i < Lk; i += tgs)
        kv_lat[i] = kv_a[i] * inv * kv_a_norm[i];

    device half* krdst = k_rope_cache + ((size_t)L * max_seq + pos) * Dr;
    uint pairs = Dr / 2;
    for (uint i = tid; i < pairs; i += tgs) {
        float ang = deepseek_r1_yarn_angle(pos, i, Dr, theta);
        float c = cos(ang), s = sin(ang);
        float a = kv_a[Lk + 2u*i], b = kv_a[Lk + 2u*i + 1u];
        krdst[2u*i] = half(a*c - b*s);
        krdst[2u*i + 1u] = half(b*c + a*s);
    }
}

// Compact --mla path. Normalize and commit the latent directly, avoiding the
// 32,768-wide KV-B expansion and its 4,005,504-byte/position expanded cache.
// This is a distinct opt-in graph; the default kernel above remains unchanged.
kernel void mla_kv_norm_rope_store_r1(
    device const float* kv_a         [[buffer(0)]],
    device const float* kv_a_norm    [[buffer(1)]],
    device       half*  c_kv_cache   [[buffer(2)]],
    device       half*  k_rope_cache [[buffer(3)]],
    constant uint& Lk      [[buffer(4)]],
    constant uint& Dr      [[buffer(5)]],
    constant uint& pos     [[buffer(6)]],
    constant uint& max_seq [[buffer(7)]],
    constant float& eps    [[buffer(8)]],
    constant float& theta  [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) acc += kv_a[i] * kv_a[i];
    float inv = rsqrt(tg_reduce_sum(acc, tid, tgs, scratch) / float(Lk) + eps);
    device half* latent = c_kv_cache + (size_t)pos * Lk;
    for (uint i = tid; i < Lk; i += tgs)
        latent[i] = half(kv_a[i] * inv * kv_a_norm[i]);

    device half* rope = k_rope_cache + (size_t)pos * Dr;
    for (uint pair = tid; pair < Dr / 2; pair += tgs) {
        float angle = deepseek_r1_yarn_angle(pos, pair, Dr, theta);
        float c = cos(angle), s = sin(angle);
        float a = kv_a[Lk + 2u * pair], b = kv_a[Lk + 2u * pair + 1u];
        rope[2u * pair] = half(a * c - b * s);
        rope[2u * pair + 1u] = half(b * c + a * s);
    }
}

kernel void mla_kv_norm_rope_store_r1_b(
    device const float* kv_a [[buffer(0)]],
    device const float* kv_a_norm [[buffer(1)]],
    device half* c_kv_cache [[buffer(2)]],
    device half* k_rope_cache [[buffer(3)]],
    constant uint& Lk [[buffer(4)]],
    constant uint& Dr [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    constant uint& max_seq [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint position = pos + token;
    device const float* input = kv_a + (size_t)token * (Lk + Dr);
    threadgroup float scratch[32];
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) acc += input[i] * input[i];
    const float inv = rsqrt(tg_reduce_sum(acc, tid, tgs, scratch) / float(Lk) + eps);
    device half* latent = c_kv_cache + (size_t)position * Lk;
    for (uint i = tid; i < Lk; i += tgs)
        latent[i] = half(input[i] * inv * kv_a_norm[i]);
    device half* rope = k_rope_cache + (size_t)position * Dr;
    for (uint pair = tid; pair < Dr / 2; pair += tgs) {
        const float angle = deepseek_r1_yarn_angle(position, pair, Dr, theta);
        const float c = cos(angle), s = sin(angle);
        const float a = input[Lk + 2u * pair], b = input[Lk + 2u * pair + 1u];
        rope[2u * pair] = half(a * c - b * s);
        rope[2u * pair + 1u] = half(b * c + a * s);
    }
}

kernel void mha_kv_store_f16(
    device const float* kv_full [[buffer(0)]], // [HE, Dn+Dv]
    device       half*  k_cache [[buffer(1)]], // [L,max_seq,HE,Dn]
    device       half*  v_cache [[buffer(2)]], // [L,max_seq,HE,Dv]
    constant uint& HE      [[buffer(3)]],
    constant uint& Dn      [[buffer(4)]],
    constant uint& Dv      [[buffer(5)]],
    constant uint& L       [[buffer(6)]],
    constant uint& pos     [[buffer(7)]],
    constant uint& max_seq [[buffer(8)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const float* src = kv_full + (size_t)h * (Dn + Dv);
    device half* kd = k_cache + (((size_t)L * max_seq + pos) * HE + h) * Dn;
    device half* vd = v_cache + (((size_t)L * max_seq + pos) * HE + h) * Dv;
    for (uint i = tid; i < Dn; i += tgs) kd[i] = half(src[i]);
    for (uint i = tid; i < Dv; i += tgs) vd[i] = half(src[Dn + i]);
}

// Exact MHA decode for the legacy combined-wkv graph. One threadgroup owns a
// head; global score scratch is reused across layers and tokens.
kernel void mha_attn_decode_f32(
    device const float* q_nope      [[buffer(0)]],
    device const float* q_rope      [[buffer(1)]],
    device const half*  k_cache     [[buffer(2)]],
    device const half*  k_rope      [[buffer(3)]],
    device const half*  v_cache     [[buffer(4)]],
    device       float* out         [[buffer(5)]],
    device       float* scores      [[buffer(6)]],
    constant uint& HE      [[buffer(7)]],
    constant uint& Dn      [[buffer(8)]],
    constant uint& Dr      [[buffer(9)]],
    constant uint& Dv      [[buffer(10)]],
    constant uint& L       [[buffer(11)]],
    constant uint& T       [[buffer(12)]],
    constant uint& max_seq [[buffer(13)]],
    constant float& scale  [[buffer(14)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    device const float* qn = q_nope + (size_t)h * Dn;
    device const float* qr = q_rope + (size_t)h * Dr;
    device float* sc = scores + (size_t)h * max_seq;
    const size_t layer_k = (size_t)L * max_seq * HE * Dn;
    const size_t layer_v = (size_t)L * max_seq * HE * Dv;
    const size_t layer_r = (size_t)L * max_seq * Dr;

    float local_max = -INFINITY;
    for (uint t = tid; t < T; t += tgs) {
        device const half* kh = k_cache + layer_k + ((size_t)t * HE + h) * Dn;
        device const half* kr = k_rope + layer_r + (size_t)t * Dr;
        float s = 0.0f;
        for (uint i = 0; i < Dn; ++i) s += qn[i] * float(kh[i]);
        for (uint i = 0; i < Dr; ++i) s += qr[i] * float(kr[i]);
        s *= scale;
        sc[t] = s;
        local_max = max(local_max, s);
    }
    float gmax = tg_reduce_max(local_max, tid, tgs, scratch);
    float local_sum = 0.0f;
    for (uint t = tid; t < T; t += tgs) {
        float e = exp(sc[t] - gmax);
        sc[t] = e;
        local_sum += e;
    }
    float inv = 1.0f / tg_reduce_sum(local_sum, tid, tgs, scratch);

    threadgroup float oacc[128];
    for (uint i = tid; i < Dv; i += tgs) oacc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint t = 0; t < T; ++t) {
        device const half* vh = v_cache + layer_v + ((size_t)t * HE + h) * Dv;
        float wt = sc[t];
        for (uint i = tid; i < Dv; i += tgs) oacc[i] += wt * float(vh[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* oh = out + (size_t)h * Dv;
    for (uint i = tid; i < Dv; i += tgs) oh[i] = oacc[i] * inv;
}

// ---------- MLA decode attention (one TG per head) -------------------------
// Inputs:
//   q_eff[h,:Lk]    -- q_nope absorbed through W_uk (= W_uk^T q_nope per head)
//   q_rope[h,:Dr]
//   c_kv  [L, t, :Lk]    KV cache (latent), t in [0..T)
//   k_rope[L, t, :Dr]
// Output:
//   o_lat[h,:Lk]    -- to be V-up'd by W_uv outside
//
// Three-pass softmax with global scratch[h, t] of size T fp32.
// Total bandwidth: ~2 * Lk * T fp16 reads of c_kv (shared across heads via
// L1/L2) plus ~2 * Dr * T fp16 reads of k_rope.  Memory-bound at long T.
kernel void mla_attn_decode_f32(
    device const float* q_eff        [[buffer(0)]],   // [HE, Lk]
    device const float* q_rope       [[buffer(1)]],   // [HE, Dr]
    device const half*  c_kv_cache   [[buffer(2)]],   // [n_layers, max_seq, Lk]
    device const half*  k_rope_cache [[buffer(3)]],   // [n_layers, max_seq, Dr]
    device       float* o_lat        [[buffer(4)]],   // [HE, Lk]
    device       float* scores       [[buffer(5)]],   // [HE, max_seq]
    constant uint&  HE      [[buffer(6)]],
    constant uint&  Lk      [[buffer(7)]],
    constant uint&  Dr      [[buffer(8)]],
    constant uint&  L       [[buffer(9)]],
    constant uint&  T       [[buffer(10)]],
    constant uint&  max_seq [[buffer(11)]],
    constant float& scale   [[buffer(12)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    device const float* qe = q_eff   + (size_t)h * Lk;
    device const float* qr = q_rope  + (size_t)h * Dr;
    device       float* sc = scores + (size_t)h * max_seq;
    device const half* ck0 = c_kv_cache   + (size_t)L * max_seq * Lk;
    device const half* kr0 = k_rope_cache + (size_t)L * max_seq * Dr;

    // ---- Pass 1: scores + global max
    float local_max = -INFINITY;
    for (uint t = tid; t < T; t += tgs) {
        device const half* ck = ck0 + (size_t)t * Lk;
        device const half* kr = kr0 + (size_t)t * Dr;
        float s = 0.0f;
        for (uint i = 0; i < Lk; ++i) s += qe[i] * float(ck[i]);
        for (uint i = 0; i < Dr; ++i) s += qr[i] * float(kr[i]);
        s *= scale;
        sc[t] = s;
        local_max = max(local_max, s);
    }
    float gmax = tg_reduce_max(local_max, tid, tgs, scratch);
    // sc[] writes from pass 1 are visible only to the same thread until a
    // device-memory barrier executes.  Pass 2 still reads strictly its own
    // stride so it doesn't need the barrier, but Pass 3 reads ALL sc[t]
    // across threads -- so the barrier MUST happen between Pass 2 and Pass 3.

    // ---- Pass 2: exp + sum  (each thread reads/writes only its own sc[t])
    float local_sum = 0.0f;
    for (uint t = tid; t < T; t += tgs) {
        float e = exp(sc[t] - gmax);
        sc[t] = e;
        local_sum += e;
    }
    float gsum = tg_reduce_sum(local_sum, tid, tgs, scratch);
    float inv = 1.0f / gsum;

    // ---- Pass 3: o_lat[h,i] = inv * sum_t sc[t] * c_kv[t,i]
    //
    // Loop-order matters here.  The naive (i outer, t inner) form has each
    // thread walk a stride-Lk column of c_kv, so a TG only touches half of
    // each c_kv[t,:] row before loop-i wraps -- and the 16 head-TGs each
    // hit memory in disjoint orders, defeating L2 reuse of c_kv (which is
    // shared across heads).  Swap to (t outer, i inner) and keep the per-i
    // accumulator in threadgroup memory: now each TG sweeps c_kv[t,:]
    // contiguously, and all 16 heads stream the same rows in the same
    // order, so the L2 serves them once.  Lk<=1024 covers V2-Lite (512),
    // V3, and Kimi K2; 4 KB of TG scratch.
    //
    // The mem_device fence is required for correctness: Pass 2 writes sc[]
    // (device memory) per-thread, Pass 3 reads all sc[t] across threads.
    threadgroup float oacc[1024];
    for (uint i = tid; i < Lk; i += tgs) oacc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint t = 0; t < T; ++t) {
        float wt = sc[t];
        for (uint i = tid; i < Lk; i += tgs)
            oacc[i] += wt * float(ck0[(size_t)t * Lk + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* oh = o_lat + (size_t)h * Lk;
    for (uint i = tid; i < Lk; i += tgs) oh[i] = oacc[i] * inv;
}

// Exact short-prefill specialization for the pinned R1 geometry. Four causal
// queries share every compact KV load, while each query retains the scalar
// dot-product order, 256-lane softmax reduction tree, and ascending-T value
// accumulation of mla_attn_decode_f32.
kernel void mla_attn_prefill_r1_q4(
    device const float* q_eff [[buffer(0)]],
    device const float* q_rope [[buffer(1)]],
    device const half* c_kv [[buffer(2)]],
    device const half* k_rope [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& HE [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& pos [[buffer(7)]],
    constant uint& max_seq [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint Lk = 512, Dr = 64, BQ = 4, MAX_T = 1024, tgs = 256;
    const uint head = group.x, b0 = group.y * BQ;
    device const half* ck0 = c_kv;
    device const half* kr0 = k_rope;
    threadgroup float scratch[32];
    threadgroup float scores[BQ * MAX_T];
    threadgroup float numerators[BQ * Lk];
    float inverse[BQ];
    float local_max[BQ];
#pragma unroll
    for (uint q = 0; q < BQ; ++q) local_max[q] = -INFINITY;

    const uint last_b = min(b0 + BQ, B) - 1;
    const uint max_T = pos + last_b + 1;
    for (uint t = tid; t < max_T; t += tgs) {
        device const half* ck = ck0 + (size_t)t * Lk;
        device const half* kr = kr0 + (size_t)t * Dr;
        float score[BQ] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = 0; i < Lk; ++i) {
            const float key = float(ck[i]);
#pragma unroll
            for (uint q = 0; q < BQ; ++q) {
                const uint b = b0 + q;
                if (b < B && t <= pos + b)
                    score[q] += q_eff[((size_t)b * HE + head) * Lk + i] * key;
            }
        }
        for (uint i = 0; i < Dr; ++i) {
            const float key = float(kr[i]);
#pragma unroll
            for (uint q = 0; q < BQ; ++q) {
                const uint b = b0 + q;
                if (b < B && t <= pos + b)
                    score[q] += q_rope[((size_t)b * HE + head) * Dr + i] * key;
            }
        }
#pragma unroll
        for (uint q = 0; q < BQ; ++q) {
            const uint b = b0 + q;
            if (b < B && t <= pos + b) {
                score[q] *= scale;
                scores[q * MAX_T + t] = score[q];
                local_max[q] = max(local_max[q], score[q]);
            }
        }
    }

#pragma unroll
    for (uint q = 0; q < BQ; ++q) {
        const uint b = b0 + q;
        if (b >= B) { inverse[q] = 0.0f; continue; }
        const uint T = pos + b + 1;
        threadgroup float* sc = scores + q * MAX_T;
        const float global_max = tg_reduce_max(local_max[q], tid, tgs, scratch);
        float local_sum = 0.0f;
        for (uint t = tid; t < T; t += tgs) {
            const float value = exp(sc[t] - global_max);
            sc[t] = value;
            local_sum += value;
        }
        inverse[q] = 1.0f / tg_reduce_sum(local_sum, tid, tgs, scratch);
    }

    for (uint q = 0; q < BQ; ++q)
        for (uint i = tid; i < Lk; i += tgs)
            numerators[q * Lk + i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < max_T; ++t) {
        device const half* cv = ck0 + (size_t)t * Lk;
        for (uint i = tid; i < Lk; i += tgs) {
            const float value = float(cv[i]);
#pragma unroll
            for (uint q = 0; q < BQ; ++q) {
                const uint b = b0 + q;
                if (b < B && t <= pos + b)
                    numerators[q * Lk + i] += scores[q * MAX_T + t] * value;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint q = 0; q < BQ; ++q) {
        const uint b = b0 + q;
        if (b >= B) continue;
        device float* dst = out + ((size_t)b * HE + head) * Lk;
        for (uint i = tid; i < Lk; i += tgs)
            dst[i] = numerators[q * Lk + i] * inverse[q];
    }
}

// DeepSeek-R1 long-context MLA. The pinned geometry (latent 512, RoPE 64)
// maps exactly to one SIMD-group: every lane keeps 16 latent values in
// registers, contributes two RoPE values, and reuses the latent load for the
// value accumulation. Thirty-two temporal blocks expose enough independent
// work to saturate M5 while bounding global scratch independently of context.
// Pass 1 uses the exact online-softmax recurrence per block:
//   m' = max(m,s), l' = exp(m-m')l + exp(s-m')
//   o' = exp(m-m')o + exp(s-m')v.
kernel void mla_attn_online_r1_pass1(
    device const float* q_eff [[buffer(0)]],
    device const float* q_rope [[buffer(1)]],
    device const half* c_kv [[buffer(2)]],
    device const half* k_rope [[buffer(3)]],
    device float* partials [[buffer(4)]],
    device float* stats [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& max_seq [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint Lk = 512;
    constexpr uint Dr = 64;
    constexpr uint blocks = 32;
    const uint block = group.x;
    const uint head = group.y;
    const uint begin = uint((ulong(T) * block) / blocks);
    const uint end = uint((ulong(T) * (block + 1)) / blocks);
    device const float* qe = q_eff + (size_t)head * Lk;
    device const float* qr = q_rope + (size_t)head * Dr;
    float out[16];
#pragma unroll
    for (uint j = 0; j < 16; ++j) out[j] = 0.0f;
    float block_max = -INFINITY;
    float block_sum = 0.0f;

    for (uint t = begin; t < end; ++t) {
        device const half* cv = c_kv + (size_t)t * Lk;
        device const half* rv = k_rope + (size_t)t * Dr;
        half values[16];
        float dot = qr[lane] * float(rv[lane])
                  + qr[lane + 32] * float(rv[lane + 32]);
#pragma unroll
        for (uint j = 0; j < 16; ++j) {
            const uint i = lane + j * 32;
            values[j] = cv[i];
            dot += qe[i] * float(values[j]);
        }
        const float score = simd_sum(dot) * scale;
        const float next_max = max(block_max, score);
        const float old_scale = exp(block_max - next_max);
        const float new_scale = exp(score - next_max);
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            out[j] = old_scale * out[j] + new_scale * float(values[j]);
        block_sum = old_scale * block_sum + new_scale;
        block_max = next_max;
    }

    device float* dst = partials + ((size_t)head * blocks + block) * Lk;
#pragma unroll
    for (uint j = 0; j < 16; ++j) dst[lane + j * 32] = out[j];
    if (lane == 0) {
        const size_t stat = ((size_t)head * blocks + block) * 2;
        stats[stat] = block_max;
        stats[stat + 1] = block_sum;
    }
}

// Merge the block-local (max, denominator, numerator) triples with the same
// log-sum-exp identity. One SIMD-group owns a head and writes 512 outputs.
kernel void mla_attn_online_r1_pass2(
    device const float* partials [[buffer(0)]],
    device const float* stats [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint Lk = 512;
    constexpr uint blocks = 32;
    const size_t stats_base = (size_t)head * blocks * 2;
    const float global_max = simd_max(stats[stats_base + lane * 2]);
    const float local_sum = stats[stats_base + lane * 2 + 1] *
                            exp(stats[stats_base + lane * 2] - global_max);
    const float denominator = simd_sum(local_sum);
    float merged[16];
#pragma unroll
    for (uint j = 0; j < 16; ++j) merged[j] = 0.0f;
    for (uint block = 0; block < blocks; ++block) {
        const float factor = exp(stats[stats_base + block * 2] - global_max);
        device const float* src = partials +
            ((size_t)head * blocks + block) * Lk;
#pragma unroll
        for (uint j = 0; j < 16; ++j)
            merged[j] += factor * src[lane + j * 32];
    }
    device float* dst = out + (size_t)head * Lk;
#pragma unroll
    for (uint j = 0; j < 16; ++j)
        dst[lane + j * 32] = merged[j] / denominator;
}

// Legacy .blade/HF path; the GGUF parity path above keeps activations f32.
kernel void mla_attn_decode_f16(
    device const half* q_eff [[buffer(0)]],
    device const half* q_rope [[buffer(1)]],
    device const half* c_kv_cache [[buffer(2)]],
    device const half* k_rope_cache [[buffer(3)]],
    device half* o_lat [[buffer(4)]],
    device float* scores [[buffer(5)]],
    constant uint& HE [[buffer(6)]], constant uint& Lk [[buffer(7)]],
    constant uint& Dr [[buffer(8)]], constant uint& L [[buffer(9)]],
    constant uint& T [[buffer(10)]], constant uint& max_seq [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const half* qe = q_eff + (size_t)h * Lk;
    device const half* qr = q_rope + (size_t)h * Dr;
    device float* sc = scores + (size_t)h * max_seq;
    device const half* ck0 = c_kv_cache + (size_t)L * max_seq * Lk;
    device const half* kr0 = k_rope_cache + (size_t)L * max_seq * Dr;
    float local_max = -INFINITY;
    for (uint t = tid; t < T; t += tgs) {
        device const half* ck = ck0 + (size_t)t * Lk;
        device const half* kr = kr0 + (size_t)t * Dr;
        float s = 0.0f;
        for (uint i = 0; i < Lk; ++i) s += float(qe[i]) * float(ck[i]);
        for (uint i = 0; i < Dr; ++i) s += float(qr[i]) * float(kr[i]);
        sc[t] = s * scale;
        local_max = max(local_max, sc[t]);
    }
    float gmax = tg_reduce_max(local_max, tid, tgs, scratch);
    float local_sum = 0.0f;
    for (uint t = tid; t < T; t += tgs) {
        sc[t] = exp(sc[t] - gmax);
        local_sum += sc[t];
    }
    float inv = 1.0f / tg_reduce_sum(local_sum, tid, tgs, scratch);
    threadgroup float oacc[1024];
    for (uint i = tid; i < Lk; i += tgs) oacc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint t = 0; t < T; ++t) {
        float wt = sc[t];
        for (uint i = tid; i < Lk; i += tgs)
            oacc[i] += wt * float(ck0[(size_t)t * Lk + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* oh = o_lat + (size_t)h * Lk;
    for (uint i = tid; i < Lk; i += tgs) oh[i] = half(oacc[i] * inv);
}

// ---------- SwiGLU epilogue ------------------------------------------------
kernel void swiglu_f16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device       half* y    [[buffer(2)]],
    constant uint& N        [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    float g = float(gate[i]);
    float s = g / (1.0f + exp(-g));
    y[i] = half(s * float(up[i]));
}

kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device       float* y    [[buffer(2)]],
    constant uint& N         [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < N) y[i] = (gate[i] / (1.0f + exp(-gate[i]))) * up[i];
}

// ---------- Router top-K (fp16 logits, fp32 bias) --------------------------
// Two routing modes:
//   mode=0 (DeepSeek-V3 / Kimi K2): top-K is selected on (logits + bias);
//     weights are softmax over the K chosen logits, then *renormalized*
//     (norm_topk_prob = true).
//   mode=1 (DeepSeek-V2 / V2-Lite, scoring_func=softmax, norm_topk_prob=false):
//     weights are the full-N softmax probabilities at the chosen indices,
//     NOT renormalized.  Bias is unused (we pass a zero buffer).
kernel void router_topk_f16(
    device const half*  logits [[buffer(0)]],   // [N]
    device const float* bias   [[buffer(1)]],   // [N]
    device       uint*  idx    [[buffer(2)]],   // [K]
    device       float* wts    [[buffer(3)]],   // [K]
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& mode        [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;
    float top_l[16]; uint top_i[16];
    for (uint k = 0; k < K; ++k) { top_l[k] = -INFINITY; top_i[k] = 0; }
    for (uint n = 0; n < N; ++n) {
        float v = float(logits[n]) + bias[n];
        if (v > top_l[K-1]) {
            uint p = K-1;
            while (p > 0 && top_l[p-1] < v) {
                top_l[p] = top_l[p-1]; top_i[p] = top_i[p-1]; --p;
            }
            top_l[p] = v; top_i[p] = n;
        }
    }
    if (mode == 0) {
        // V3: softmax over the K chosen un-biased logits, then renormalize.
        float m = -INFINITY;
        for (uint k = 0; k < K; ++k) {
            float v = float(logits[top_i[k]]);
            if (v > m) m = v;
        }
        float ex[16]; float s = 0.0f;
        for (uint k = 0; k < K; ++k) { ex[k] = exp(float(logits[top_i[k]]) - m); s += ex[k]; }
        for (uint k = 0; k < K; ++k) { idx[k] = top_i[k]; wts[k] = ex[k] / s; }
    } else {
        // V2: full-N softmax probabilities at the chosen indices (no renorm).
        float m = -INFINITY;
        for (uint n = 0; n < N; ++n) { float v = float(logits[n]); if (v > m) m = v; }
        float s = 0.0f;
        for (uint n = 0; n < N; ++n) s += exp(float(logits[n]) - m);
        for (uint k = 0; k < K; ++k) {
            idx[k] = top_i[k];
            wts[k] = exp(float(logits[top_i[k]]) - m) / s;
        }
    }
}

// ---------- Router top-K, sigmoid gating (DeepSeek-V3 / R1) ----------------
// R1 routing (llama.cpp deepseek2, expert_gating_func = sigmoid):
//   prob[e]      = sigmoid(logit[e])
//   sel_score[e] = prob[e] + bias[e]          // bias = e_score_correction_bias
//   choose top-K experts by sel_score
//   wt[k]        = prob[chosen]               // bias EXCLUDED from the weight
//   if norm:  wt[k] /= sum_k wt[k]            // norm_topk_prob / weights_norm
//   wt[k]       *= scale                      // routed_scaling_factor (2.5)
// This is selected over the softmax path (router_topk_f16) because R1's GGUF
// declares deepseek2.expert_gating_func == 2 (sigmoid). Using softmax here
// selects the wrong experts and mis-weights them -> incoherent tokens.
kernel void router_topk_sigmoid_f16(
    device const half*  logits [[buffer(0)]],   // [N]
    device const float* bias   [[buffer(1)]],   // [N]
    device       uint*  idx    [[buffer(2)]],   // [K]
    device       float* wts    [[buffer(3)]],   // [K]
    constant uint&  N          [[buffer(4)]],
    constant uint&  K          [[buffer(5)]],
    constant float& scale      [[buffer(6)]],
    constant uint&  norm       [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;
    float top_s[16]; uint top_i[16]; float top_p[16];
    for (uint k = 0; k < K; ++k) { top_s[k] = -INFINITY; top_i[k] = 0; top_p[k] = 0.0f; }
    for (uint n = 0; n < N; ++n) {
        float prob = 1.0f / (1.0f + exp(-float(logits[n])));
        float sel  = prob + bias[n];
        if (sel > top_s[K-1]) {
            uint p = K-1;
            while (p > 0 && top_s[p-1] < sel) {
                top_s[p] = top_s[p-1]; top_i[p] = top_i[p-1]; top_p[p] = top_p[p-1]; --p;
            }
            top_s[p] = sel; top_i[p] = n; top_p[p] = prob;
        }
    }
    float s = 0.0f;
    for (uint k = 0; k < K; ++k) s += top_p[k];
    float inv = (norm && s > 0.0f) ? (1.0f / s) : 1.0f;
    for (uint k = 0; k < K; ++k) {
        idx[k] = top_i[k];
        wts[k] = top_p[k] * inv * scale;
    }
}

// Exact DeepSeek-R1 noaux_tc router. Selection is group-limited:
// 256 experts -> 8 contiguous groups; score each group by its two highest
// corrected scores; retain 4 groups; then select 8 experts. Mixture weights
// use the uncorrected sigmoid probabilities.
kernel void router_topk_grouped_sigmoid_f32(
    device const float* logits    [[buffer(0)]],
    device const float* bias      [[buffer(1)]],
    device       uint*  idx       [[buffer(2)]],
    device       float* wts       [[buffer(3)]],
    constant uint&  N             [[buffer(4)]],
    constant uint&  K             [[buffer(5)]],
    constant uint&  n_groups      [[buffer(6)]],
    constant uint&  top_groups    [[buffer(7)]],
    constant float& scale         [[buffer(8)]],
    constant uint&  norm          [[buffer(9)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;
    if (N > 256 || K > 16 || n_groups > 8 || n_groups == 0 ||
        (N % n_groups) != 0 || top_groups > n_groups) return;
    logits += (size_t)token * N;
    idx += (size_t)token * K;
    wts += (size_t)token * K;

    float prob[256];
    float corrected[256];
    for (uint e = 0; e < N; ++e) {
        float p = 1.0f / (1.0f + exp(-logits[e]));
        prob[e] = p;
        corrected[e] = p + bias[e];
    }

    const uint per_group = N / n_groups;
    float group_score[8];
    uint group_id[8];
    for (uint g = 0; g < n_groups; ++g) {
        float first = -INFINITY;
        float second = -INFINITY;
        for (uint j = 0; j < per_group; ++j) {
            float v = corrected[g * per_group + j];
            if (v > first) { second = first; first = v; }
            else if (v > second) second = v;
        }
        float gs = first + second;
        uint p = g;
        while (p > 0 && (group_score[p-1] < gs ||
               (group_score[p-1] == gs && group_id[p-1] > g))) {
            group_score[p] = group_score[p-1];
            group_id[p] = group_id[p-1];
            --p;
        }
        group_score[p] = gs;
        group_id[p] = g;
    }

    bool keep[8];
    for (uint g = 0; g < n_groups; ++g) keep[g] = false;
    for (uint g = 0; g < top_groups; ++g) keep[group_id[g]] = true;

    float top_score[16];
    float top_prob[16];
    uint top_id[16];
    for (uint k = 0; k < K; ++k) {
        top_score[k] = -INFINITY;
        top_prob[k] = 0.0f;
        top_id[k] = 0;
    }
    for (uint e = 0; e < N; ++e) {
        if (!keep[e / per_group]) continue;
        float v = corrected[e];
        if (v > top_score[K-1] || (v == top_score[K-1] && e < top_id[K-1])) {
            uint p = K - 1;
            while (p > 0 && (top_score[p-1] < v ||
                   (top_score[p-1] == v && top_id[p-1] > e))) {
                top_score[p] = top_score[p-1];
                top_prob[p] = top_prob[p-1];
                top_id[p] = top_id[p-1];
                --p;
            }
            top_score[p] = v;
            top_prob[p] = prob[e];
            top_id[p] = e;
        }
    }

    float denom = 0.0f;
    for (uint k = 0; k < K; ++k) denom += top_prob[k];
    float inv = (norm && denom > 0.0f) ? (1.0f / denom) : 1.0f;
    for (uint k = 0; k < K; ++k) {
        idx[k] = top_id[k];
        wts[k] = top_prob[k] * inv * scale;
    }
}

// ---------- y += alpha * x  -----------------------------------------------
kernel void axpy_f16(
    device       half* y     [[buffer(0)]],
    device const half* x     [[buffer(1)]],
    constant float& alpha    [[buffer(2)]],
    constant uint&  N        [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    y[i] = half(float(y[i]) + alpha * float(x[i]));
}

kernel void axpy_f32(
    device       float* y     [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    constant float& alpha     [[buffer(2)]],
    constant uint& N          [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < N) y[i] += alpha * x[i];
}

// ---------- Embedding lookup (FP8 -> fp16) --------------------------------
kernel void embed_lookup_fp8(
    device const Fp8Block* W [[buffer(0)]],   // [vocab, H]
    device       half*     y [[buffer(1)]],   // [H]
    constant uint& token     [[buffer(2)]],
    constant uint& H         [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= H) return;
    uint nblk_per_row = H / 128;
    uint b = i / 128;
    uint k = i & 127;
    const device Fp8Block& blk = W[(size_t)token * nblk_per_row + b];
    y[i] = half(fp8_e4m3_to_f32(blk.q[k]) * blk.scale);
}

// ---------- Argmax over fp16 logits (greedy sampling) ---------------------
kernel void argmax_f16(
    device const half* logits [[buffer(0)]],
    device       uint* out    [[buffer(1)]],
    constant uint& N          [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float bestv[32]; threadgroup uint besti[32];
    float lv = -INFINITY; uint li = 0;
    for (uint i = tid; i < N; i += tgs) {
        float v = float(logits[i]);
        if (v > lv) { lv = v; li = i; }
    }
    // SIMD reduction with index
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(lv, off);
        uint  oi = simd_shuffle_down(li, off);
        if (ov > lv) { lv = ov; li = oi; }
    }
    uint lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) { bestv[warp] = lv; besti[warp] = li; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        uint nwarps = (tgs + 31u) >> 5;
        lv = (tid < nwarps) ? bestv[tid] : -INFINITY;
        li = (tid < nwarps) ? besti[tid] : 0u;
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(lv, off);
            uint  oi = simd_shuffle_down(li, off);
            if (ov > lv) { lv = ov; li = oi; }
        }
        if (tid == 0) *out = li;
    }
}

// ============================================================================
// BF16 weight path (read raw HuggingFace safetensors directly, no conversion).
// bfloat16 layout: 1 sign bit | 8 exp | 7 mantissa.  Top 16 bits of fp32.
// Conversion bf16 -> fp32 is just (uint32(bf) << 16) reinterpreted as float.
// ============================================================================
inline float bf16_to_f32(ushort b) {
    uint u = uint(b) << 16;
    return as_type<float>(u);
}

// y[M] = W[M,K] @ x[ row/group_size , : ], W stored as bf16 row-major.  Pass
// group_size = M for the plain (non-grouped) form.  Bandwidth-bound on M-series:
// each thread stride does two 4-wide bf16 + two 4-wide fp16 loads and folds
// them into the accumulator with two `dot4` SIMD ops.  K must be a multiple of 8.
kernel void gemv_bf16_f16(
    device const bfloat* W          [[buffer(0)]],
    device const half*   x          [[buffer(1)]],
    device       half*   y          [[buffer(2)]],
    constant uint&       K          [[buffer(3)]],
    constant uint&       group_size [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const bfloat4* W4 = (device const bfloat4*)(W + (size_t)row * K);
    device const half4*   x4 = (device const half4*)(x + (size_t)(row / group_size) * K);
    uint K8 = K >> 3;
    float acc = 0.0f;
    for (uint c = tid; c < K8; c += tgs) {
        bfloat4 wA = W4[c*2 + 0]; bfloat4 wB = W4[c*2 + 1];
        half4   xA = x4[c*2 + 0]; half4   xB = x4[c*2 + 1];
        acc += dot(float4(wA), float4(xA)) + dot(float4(wB), float4(xB));
    }
    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = half(v);
}

// Embedding lookup, bf16 weight -> fp16 hidden state.
kernel void embed_lookup_bf16(
    device const bfloat* W [[buffer(0)]],   // [vocab, H]
    device       half*   y [[buffer(1)]],   // [H]
    constant uint& token   [[buffer(2)]],
    constant uint& H       [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= H) return;
    y[i] = half(float(W[(size_t)token * H + i]));
}

// ============================================================================
// Batched prefill kernels.  Process B prompt tokens in one forward pass; each
// weight tensor is loaded ONCE per pass instead of once per token.  Arithmetic
// intensity scales with B -> push the system from bandwidth-bound (GEMV) into
// compute-bound (GEMM) territory.  Decode-time GEMV path is unchanged.
// ============================================================================

// y[B, M] = x[B, K] @ W[M, K]^T,  W bf16, x/y fp16.  K%8 == 0.
// Tile: each TG handles one row r and a batch tile [b0..b0+BT) of size BT=8.
// Grid: M * ceil(B/BT) threadgroups.  W[r,:] is read ONCE per TG and reused
// across BT batch lanes -- the whole point.
//
// group_size: if < M, the K-input for row r is read from
// x[b, (r/group_size)*K + 0..K], i.e. row r belongs to group g=r/group_size
// which selects which input slice each batch lane reads.  Pass group_size=M
// for the plain (non-grouped) form (every row reads x[b, 0..K]).  Used by
// MLA W_uk / W_uv to fold HE per-head GEMMs into a single dispatch over a
// [B, HE, Lk-or-Dn] activation tensor laid out as [B, HE*K_or_M].
kernel void gemm_bf16_f16(
    device const bfloat* W [[buffer(0)]],   // [M, K]
    device const half*   x [[buffer(1)]],   // [B, n_groups*K]  (n_groups=M/group_size)
    device       half*   y [[buffer(2)]],   // [B, M]
    constant uint& K          [[buffer(3)]],
    constant uint& M          [[buffer(4)]],
    constant uint& B          [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    constexpr uint BT = 8;
    uint nbt = (B + BT - 1u) / BT;
    uint r   = gid / nbt;
    uint b0  = (gid - r * nbt) * BT;
    if (r >= M) return;
    uint g     = r / group_size;
    uint x_row_stride = (M / group_size) * K;   // bytes-equiv: stride per batch lane

    device const bfloat4* W4 = (device const bfloat4*)(W + (size_t)r * K);
    uint K8 = K >> 3;
    float acc[BT];
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BT; ++b) acc[b] = 0.0f;

    for (uint c = tid; c < K8; c += tgs) {
        bfloat4 wA = W4[c*2 + 0];
        bfloat4 wB = W4[c*2 + 1];
        float4 wAf = float4(wA), wBf = float4(wB);
        #pragma clang loop unroll(full)
        for (uint b = 0; b < BT; ++b) {
            uint bi = b0 + b;
            if (bi >= B) continue;
            device const half4* xb = (device const half4*)(x + (size_t)bi * x_row_stride + (size_t)g * K);
            half4 xA = xb[c*2 + 0], xB = xb[c*2 + 1];
            acc[b] += dot(wAf, float4(xA)) + dot(wBf, float4(xB));
        }
    }
    threadgroup float scratch[32];
    #pragma clang loop unroll(full)
    for (uint b = 0; b < BT; ++b) {
        float v = tg_reduce_sum(acc[b], tid, tgs, scratch);
        uint bi = b0 + b;
        if (tid == 0 && bi < B) y[(size_t)bi * M + r] = half(v);
    }
}

// ============================================================================
// gemm_bf16_f16_mma: tiled prefill GEMM using simdgroup_matrix bf16 MMA.
//
// Math:  y[b, m] = sum_k W[m, k] * x[b, group(m)*K + k]
//   where group(m) = m / group_size  (group_size==M  ->  plain GEMM).
//
// Tile per threadgroup: 32 batch rows  x  32 M-cols  reduced over BK=8 K-cols.
// Layout:  16 simdgroups per TG arranged 4x4, each owning one 8x8 fp32 acc.
//   sg_row in [0,4)  -> batch tile [sg_row*8, sg_row*8+8)
//   sg_col in [0,4)  -> M-tile     [sg_col*8, sg_col*8+8)
// Threads/TG = 16*32 = 512.
//
// Per K-step (BK=8 columns of K):
//   1. Cooperatively load a [BM=32, BK=8] W tile to threadgroup memory.
//   2. Cooperatively load a [BN=32, BK=8] x tile (half->bfloat) to tg memory.
//   3. Each simdgroup loads its 8x8 view of x and W^T, runs one bf16 mma.
//
// Apple bf16 mma: 8*8*8 = 512 FMAs per simdgroup per cycle (peak).
// Achieved arithmetic intensity scales linearly in B until tg memory or
// register pressure binds; at BM=BN=32 we hit ~16 FMA per byte loaded
// from device memory, comfortably compute-bound for any K >= 64.
//
// Constraint: BM and BN must divide M and B respectively for full tiles.
// Edge tiles are masked at load (padded with 0) and at store (skipped).
// For grouped use (group_size < M), the tile must lie within ONE group, i.e.
// group_size % BM == 0. All current call sites (W_uk, W_uv) satisfy this.
// ============================================================================
kernel void gemm_bf16_f16_mma(
    device const bfloat* W [[buffer(0)]],   // [M, K]
    device const half*   x [[buffer(1)]],   // [B, n_groups*K]
    device       half*   y [[buffer(2)]],   // [B, M]
    constant uint& K          [[buffer(3)]],
    constant uint& M          [[buffer(4)]],
    constant uint& B          [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  sg_id   [[simdgroup_index_in_threadgroup]],
    uint  tid     [[thread_index_in_threadgroup]])
{
    constexpr uint BM = 32, BN = 32, BK = 8;
    constexpr uint THREADS = 512;

    uint bm0 = tg_id.x * BM;
    uint bn0 = tg_id.y * BN;
    if (bm0 >= M || bn0 >= B) return;

    uint sg_row = sg_id >> 2;       // 0..3, batch sub-tile
    uint sg_col = sg_id & 3u;       // 0..3, M sub-tile

    threadgroup bfloat A_tg[BM * BK];   // W tile: [m, k]   row-major, stride=BK
    threadgroup bfloat B_tg[BN * BK];   // x tile: [b, k]   row-major, stride=BK
    threadgroup float  C_tg[BN * BM];   // float accum scratch for store

    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint x_row_stride = (M / group_size) * K;
    uint x_col_off    = (bm0 / group_size) * K;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Load W tile [32 x 8] = 256 bfloats, 512 threads -> some idle.
        if (tid < BM * BK) {
            uint mi = tid / BK;
            uint ki = tid - mi * BK;
            uint mg = bm0 + mi;
            A_tg[tid] = (mg < M) ? W[(size_t)mg * K + (k0 + ki)] : bfloat(0);
        }
        // Load x tile [32 x 8] = 256 elements, half -> bfloat.
        if (tid < BN * BK) {
            uint bi = tid / BK;
            uint ki = tid - bi * BK;
            uint bg = bn0 + bi;
            B_tg[tid] = (bg < B)
                ? bfloat(float(x[(size_t)bg * x_row_stride + x_col_off + (k0 + ki)]))
                : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A = x sub-tile [8 batch x 8 k], B = W sub-tile [8 m x 8 k] viewed as [k x 8 m].
        simdgroup_matrix<bfloat, 8, 8> A_mma, B_mma;
        simdgroup_load(A_mma, B_tg, BK, ulong2(0, sg_row * 8), false);
        simdgroup_load(B_mma, A_tg, BK, ulong2(0, sg_col * 8), true);
        simdgroup_multiply_accumulate(C, A_mma, B_mma, C);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage C to threadgroup fp32 then cooperatively write to device fp16.
    simdgroup_store(C, C_tg, BM, ulong2(sg_col * 8, sg_row * 8), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < BN * BM; i += THREADS) {
        uint bi = i / BM;
        uint mi = i - bi * BM;
        uint bg = bn0 + bi;
        uint mg = bm0 + mi;
        if (bg < B && mg < M) y[(size_t)bg * M + mg] = half(C_tg[i]);
    }
}

// Batched RMSNorm: B independent rows.  Grid = B threadgroups.
kernel void rms_norm_f16_b(
    device const half*  x    [[buffer(0)]],   // [B, H]
    device const half*  gain [[buffer(1)]],   // [H]
    device       half*  y    [[buffer(2)]],   // [B, H]
    constant uint&  H        [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    device const half* xr = x + (size_t)b * H;
    device       half* yr = y + (size_t)b * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += tgs) { float v = float(xr[i]); acc += v*v; }
    float ss = tg_reduce_sum(acc, tid, tgs, scratch);
    float inv = rsqrt(ss / float(H) + eps);
    for (uint i = tid; i < H; i += tgs)
        yr[i] = half(float(xr[i]) * inv * float(gain[i]));
}

// Batched Q split + decoupled-rope.  One TG per (head, batch).
kernel void mla_q_split_rope_b(
    device const half* q_full     [[buffer(0)]],   // [B, HE, Dn+Dr]
    device       half* q_nope_out [[buffer(1)]],   // [B, HE, Dn]
    device       half* q_rope_out [[buffer(2)]],   // [B, HE, Dr]
    constant uint&  Dn   [[buffer(3)]],
    constant uint&  Dr   [[buffer(4)]],
    constant uint&  HE   [[buffer(5)]],
    constant uint&  pos  [[buffer(6)]],   // base; query position = pos + b
    constant float& theta[[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    uint b = gid / HE;
    uint h = gid - b * HE;
    uint p = pos + b;
    device const half* qf = q_full     + ((size_t)b * HE + h) * (Dn + Dr);
    device       half* qn = q_nope_out + ((size_t)b * HE + h) * Dn;
    device       half* qr = q_rope_out + ((size_t)b * HE + h) * Dr;
    for (uint i = tid; i < Dn; i += tgs) qn[i] = qf[i];
    uint half_dr = Dr / 2;
    for (uint i = tid; i < half_dr; i += tgs) {
        float freq = pow(theta, -2.0f * float(i) / float(Dr));
        float ang  = float(p) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(qf[Dn + 2u*i]);
        float bv= float(qf[Dn + 2u*i + 1u]);
        qr[2u*i]      = half(a*c - bv*s);
        qr[2u*i + 1u] = half(bv*c + a*s);
    }
}

// Batched KV split + RMSNorm + RoPE + cache write.  One TG per batch position;
// writes c_kv[L, pos+b, :] and k_rope[L, pos+b, :].
kernel void mla_kv_split_rope_b(
    device const half* kv_a         [[buffer(0)]],   // [B, Lk+Dr]
    device const half* kv_a_norm    [[buffer(1)]],   // [Lk]
    device       half* c_kv_cache   [[buffer(2)]],
    device       half* k_rope_cache [[buffer(3)]],
    constant uint& Lk      [[buffer(4)]],
    constant uint& Dr      [[buffer(5)]],
    constant uint& L       [[buffer(6)]],
    constant uint& pos     [[buffer(7)]],
    constant uint& max_seq [[buffer(8)]],
    constant float& eps    [[buffer(9)]],
    constant float& theta  [[buffer(10)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    device const half* kva = kv_a + (size_t)b * (Lk + Dr);
    uint p = pos + b;
    float acc = 0.0f;
    for (uint i = tid; i < Lk; i += tgs) { float v = float(kva[i]); acc += v*v; }
    float ss = tg_reduce_sum(acc, tid, tgs, scratch);
    float inv = rsqrt(ss / float(Lk) + eps);
    device half* cdst = c_kv_cache + ((size_t)L * max_seq + p) * Lk;
    for (uint i = tid; i < Lk; i += tgs)
        cdst[i] = half(float(kva[i]) * inv * float(kv_a_norm[i]));
    device half* krdst = k_rope_cache + ((size_t)L * max_seq + p) * Dr;
    uint half_dr = Dr / 2;
    for (uint i = tid; i < half_dr; i += tgs) {
        float freq = pow(theta, -2.0f * float(i) / float(Dr));
        float ang  = float(p) * freq;
        float c = cos(ang), s = sin(ang);
        float a = float(kva[Lk + 2u*i]);
        float bv= float(kva[Lk + 2u*i + 1u]);
        krdst[2u*i]      = half(a*c - bv*s);
        krdst[2u*i + 1u] = half(bv*c + a*s);
    }
}

// Batched MLA prefill attention (causal).  One TG per (head, batch).  Each
// query at position pos+b attends to keys [0, pos+b].  Same three-pass softmax
// as the decode kernel, but sc[] lives in threadgroup memory (fits at
// max_seq=4096), removing the device-memory scratch and its required fence.
kernel void mla_attn_prefill_f16(
    device const half*  q_eff        [[buffer(0)]],   // [B, HE, Lk]
    device const half*  q_rope       [[buffer(1)]],   // [B, HE, Dr]
    device const half*  c_kv_cache   [[buffer(2)]],
    device const half*  k_rope_cache [[buffer(3)]],
    device       half*  o_lat        [[buffer(4)]],   // [B, HE, Lk]
    constant uint& HE      [[buffer(5)]],
    constant uint& Lk      [[buffer(6)]],
    constant uint& Dr      [[buffer(7)]],
    constant uint& L       [[buffer(8)]],
    constant uint& pos     [[buffer(9)]],   // base; query positions = pos+b
    constant uint& max_seq [[buffer(10)]],
    constant float& scale  [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float scratch[32];
    threadgroup float oacc[1024];
    threadgroup float sc[4096];   // T <= max_seq; compile-time max.
    uint b = gid / HE;
    uint h = gid - b * HE;
    uint q_pos = pos + b;
    uint T = q_pos + 1u;

    device const half* qe = q_eff   + ((size_t)b * HE + h) * Lk;
    device const half* qr = q_rope  + ((size_t)b * HE + h) * Dr;
    device const half* ck0 = c_kv_cache   + (size_t)L * max_seq * Lk;
    device const half* kr0 = k_rope_cache + (size_t)L * max_seq * Dr;

    float local_max = -INFINITY;
    for (uint t = tid; t < T; t += tgs) {
        device const half* ck = ck0 + (size_t)t * Lk;
        device const half* kr = kr0 + (size_t)t * Dr;
        float s = 0.0f;
        for (uint i = 0; i < Lk; ++i) s += float(qe[i]) * float(ck[i]);
        for (uint i = 0; i < Dr; ++i) s += float(qr[i]) * float(kr[i]);
        s *= scale;
        sc[t] = s;
        local_max = max(local_max, s);
    }
    float gmax = tg_reduce_max(local_max, tid, tgs, scratch);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint t = tid; t < T; t += tgs) {
        float e = exp(sc[t] - gmax);
        sc[t] = e;
        local_sum += e;
    }
    float gsum = tg_reduce_sum(local_sum, tid, tgs, scratch);
    float inv = 1.0f / gsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < Lk; i += tgs) oacc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < T; ++t) {
        float wt = sc[t];
        for (uint i = tid; i < Lk; i += tgs)
            oacc[i] += wt * float(ck0[(size_t)t * Lk + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* oh = o_lat + ((size_t)b * HE + h) * Lk;
    for (uint i = tid; i < Lk; i += tgs) oh[i] = half(oacc[i] * inv);
}

// Batched embedding lookup.
kernel void embed_lookup_bf16_b(
    device const bfloat* W      [[buffer(0)]],
    device       half*   y      [[buffer(1)]],   // [B, H]
    device const uint*   tokens [[buffer(2)]],   // [B]
    constant uint& B            [[buffer(3)]],
    constant uint& H            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = B * H;
    if (gid >= total) return;
    uint b = gid / H;
    uint i = gid - b * H;
    y[(size_t)b * H + i] = half(float(W[(size_t)tokens[b] * H + i]));
}

// Batched router top-K.  One TG per batch row b; reuses scalar router logic.
kernel void router_topk_f16_b(
    device const half*  logits [[buffer(0)]],   // [B, N]
    device const float* bias   [[buffer(1)]],   // [N]
    device       uint*  idx    [[buffer(2)]],   // [B, K]
    device       float* wts    [[buffer(3)]],   // [B, K]
    constant uint& N           [[buffer(4)]],
    constant uint& K           [[buffer(5)]],
    constant uint& mode        [[buffer(6)]],
    constant uint& B           [[buffer(7)]],
    uint b   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0 || b >= B) return;
    device const half* lg = logits + (size_t)b * N;
    device       uint* ix = idx    + (size_t)b * K;
    device       float* wt= wts    + (size_t)b * K;
    float top_l[16]; uint top_i[16];
    for (uint k = 0; k < K; ++k) { top_l[k] = -INFINITY; top_i[k] = 0; }
    for (uint n = 0; n < N; ++n) {
        float v = float(lg[n]) + bias[n];
        if (v > top_l[K-1]) {
            uint p = K-1;
            while (p > 0 && top_l[p-1] < v) {
                top_l[p] = top_l[p-1]; top_i[p] = top_i[p-1]; --p;
            }
            top_l[p] = v; top_i[p] = n;
        }
    }
    if (mode == 0) {
        float m = -INFINITY;
        for (uint k = 0; k < K; ++k) { float v = float(lg[top_i[k]]); if (v > m) m = v; }
        float ex[16]; float s = 0.0f;
        for (uint k = 0; k < K; ++k) { ex[k] = exp(float(lg[top_i[k]]) - m); s += ex[k]; }
        for (uint k = 0; k < K; ++k) { ix[k] = top_i[k]; wt[k] = ex[k] / s; }
    } else {
        float m = -INFINITY;
        for (uint n = 0; n < N; ++n) { float v = float(lg[n]); if (v > m) m = v; }
        float s = 0.0f;
        for (uint n = 0; n < N; ++n) s += exp(float(lg[n]) - m);
        for (uint k = 0; k < K; ++k) { ix[k] = top_i[k]; wt[k] = exp(float(lg[top_i[k]]) - m) / s; }
    }
}

// Gather: y[i, :] = x[idx[i], :], for i in [0, n).  Used by grouped MoE to
// pack the input rows of one expert's tokens contiguously.
kernel void gather_rows_f16(
    device const half* x   [[buffer(0)]],
    device       half* y   [[buffer(1)]],
    device const uint* idx [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    constant uint& D       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = n * D;
    if (gid >= total) return;
    uint i = gid / D;
    uint d = gid - i * D;
    y[(size_t)i * D + d] = x[(size_t)idx[i] * D + d];
}

// Scatter-add weighted: y[idx[i], :] += w[i] * x[i, :], for i in [0, n).
// Within one dispatch, idx[] entries are guaranteed unique (each token appears
// at most once per expert's bucket), so no atomics required.
kernel void scatter_add_weighted_f16(
    device       half*  y   [[buffer(0)]],
    device const half*  x   [[buffer(1)]],
    device const uint*  idx [[buffer(2)]],
    device const float* w   [[buffer(3)]],
    constant uint& n        [[buffer(4)]],
    constant uint& D        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = n * D;
    if (gid >= total) return;
    uint i = gid / D;
    uint d = gid - i * D;
    float v = float(y[(size_t)idx[i] * D + d]) + w[i] * float(x[(size_t)i * D + d]);
    y[(size_t)idx[i] * D + d] = half(v);
}

// ---------------------------------------------------------------------------
// GGUF q6_K fused dequant-gemv.
//
// Block layout (210 bytes / 256 weights), exactly matching the bytes that
// llama.cpp / ggml writes to disk.  Mirrored verbatim from the CPU oracle in
// src/gguf_dequant.cpp.  We deliberately do NOT use a Metal struct here:
// q6_K is not 4-byte aligned (210 = 2·105) and a packed struct would force
// the compiler into per-byte loads regardless.  Indexed-byte access is
// idiomatic for k-quants in ggml-metal too.
//
//   uint8  ql[128]      offset   0
//   uint8  qh[64]       offset 128
//   int8   scales[16]   offset 192
//   uint16 d_half       offset 208
//
// One threadgroup per output row.  Each thread strides through the row's
// blocks (nblk = K/256), dequantizes its 256 weights inline, MACs against
// the input vector, and the threadgroup sums via the standard tg_reduce_sum.
//
// Performance notes:
//   - We never materialize fp16 weights: the dequant lives entirely in
//     registers.  Bandwidth = 210 B / 256 weights = 0.82 B/weight (vs 2 B/w
//     for fp16 stage), so the same kernel is 2.4x faster bandwidth-bound.
//   - Half-of-byte unpacking shares the byte load between q1/q3 and q2/q4
//     (same ql byte, different shift).  The 2-bit qh expansion likewise
//     shares one qh byte across all four sub-quads.

inline float q6_k_partial(device const uchar* Wrow,
                          device const float* x,
                          uint nblk, uint tid, uint tgs) {
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        device const uchar* blk = Wrow + (size_t)b * 210;
        device const uchar* ql_base = blk + 0;
        device const uchar* qh_base = blk + 128;
        device const char*  sc_base = (device const char*)(blk + 192);
        ushort d_h = ((device const ushort*)(blk + 208))[0];
        float d = float(as_type<half>(d_h));

        device const float* xb = x + (size_t)b * 256;

        // Two outer halves of 128 weights each.
        for (uint n = 0; n < 2; ++n) {
            device const uchar* ql = ql_base + n * 64;
            device const uchar* qh = qh_base + n * 32;
            device const char*  sc = sc_base + n * 8;
            device const float* xs = xb + n * 128;

            float local = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                uchar qhb = qh[l];
                uchar ql0 = ql[l +  0];
                uchar ql1 = ql[l + 32];
                int q1 = int((ql0 & 0x0F) | (((qhb >> 0) & 3) << 4)) - 32;
                int q2 = int((ql1 & 0x0F) | (((qhb >> 2) & 3) << 4)) - 32;
                int q3 = int((ql0 >>   4) | (((qhb >> 4) & 3) << 4)) - 32;
                int q4 = int((ql1 >>   4) | (((qhb >> 6) & 3) << 4)) - 32;
                uint is = l >> 4;
                local += float(sc[is + 0]) * float(q1) * xs[l +  0]
                       + float(sc[is + 2]) * float(q2) * xs[l + 32]
                       + float(sc[is + 4]) * float(q3) * xs[l + 64]
                       + float(sc[is + 6]) * float(q4) * xs[l + 96];
            }
            acc += d * local;
        }
    }
    return acc;
}

kernel void gemv_q6_K_f32(
    device const uchar*  W       [[buffer(0)]],   // raw bytes, 210 * nblk per row
    device const float*  x       [[buffer(1)]],   // length K
    device       float*  y       [[buffer(2)]],   // length n_rows
    constant uint&       K       [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 210;
    const float acc = q6_k_partial(Wrow, x, nblk, tid, tgs);

    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// R1 shared-expert down specialization: K=2048 is exactly eight Q6_K
// blocks. Four rows share one SIMD-group, eliminating the 24 idle lanes in
// the generic one-row mapping while reading the identical quantized bytes.
kernel void gemv_q6_K_f32_r4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint row4 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid & 7u;
    const uint row = row4 * 4u + (tid >> 3);
    constexpr uint nblk = 8;
    device const uchar* Wrow = W + (size_t)row * nblk * 210;
    float value = q6_k_partial(Wrow, x, nblk, lane, 8);
    value += simd_shuffle_down(value, 4);
    value += simd_shuffle_down(value, 2);
    value += simd_shuffle_down(value, 1);
    if (lane == 0) y[row] = value;
}

// Read one coefficient from a Q6_K row without expanding the tensor.  The
// R1 KV-B matrix is fixed at K=Lk=512, so its two absorbed MLA products can
// operate on the original 0.82-byte/weight representation held in UMA.
inline float q6_k_value(device const uchar* row, uint column) {
    device const uchar* blk = row + (size_t)(column >> 8) * 210;
    const uint index = column & 255u;
    const uint half_index = index >> 7;
    const uint within = index & 127u;
    const uint quarter = within >> 5;
    const uint lane = within & 31u;
    device const uchar* ql = blk + half_index * 64;
    const uchar qlb = ql[lane + ((quarter & 1u) << 5)];
    const uchar qhb = blk[128 + half_index * 32 + lane];
    const uint low = quarter < 2 ? (qlb & 15u) : (qlb >> 4);
    const uint high = (qhb >> (quarter * 2)) & 3u;
    const int q = int(low | (high << 4)) - 32;
    const char scale = ((device const char*)(blk + 192))[half_index * 8 +
                                                              quarter * 2 + (lane >> 4)];
    const ushort dh = ((device const ushort*)(blk + 208))[0];
    return float(as_type<half>(dh)) * float(scale) * float(q);
}

kernel void qmm_q6_K_f32_m32n32k128(
    device const uchar* W [[buffer(0)]],
    device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 32, TN = 32, TK = 128;
    threadgroup float weight_tile[TK * TN];
    using device_tensor = tensor<device float, dextents<int, 2>, tensor_inline>;
    using group_tensor = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
    device_tensor input(x, dextents<int, 2>(int(K), int(B)),
                        array<int, 2>({1, int(K)}));
    device_tensor output(y, dextents<int, 2>(int(N), int(B)),
                         array<int, 2>({1, int(N)}));
    const int out_col = int(group.x) * TN;
    const int out_row = int(group.y) * TM;
    auto output_tile = output.slice<TN, TM>(out_col, out_row);
    constexpr auto descriptor = matmul2d_descriptor(
        TM, TN, TK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto first_input = input.slice<TK, TM>(0, out_row);
    group_tensor weights(weight_tile, dextents<int, 2>(TN, TK),
                         array<int, 2>({1, TN}));
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < accumulator.get_capacity(); ++i)
        accumulator[i] = 0.0f;
    const uint row_bytes = (K >> 8) * 210;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        for (uint i = tid; i < TK * TN; i += 128) {
            const uint k = i / TN;
            const uint n = i - k * TN;
            weight_tile[i] = q6_k_value(W + (size_t)(out_col + n) * row_bytes,
                                        k0 + k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_tile = input.slice<TK, TM>(int(k0), out_row);
        operation.run(input_tile, weights, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    accumulator.store(output_tile);
}

// q_eff[h,k] = sum_d W_kv_b[h,d,k] * q_nope[h,d].  Reading the transposed
// product directly from compressed Q6_K removes the former 4.09 GB FP32
// expansion and its 4.09 GB/token UMA read.  One TG owns one (head,latent)
// output, so accumulation and reduction order are deterministic.
kernel void mla_q6_kv_b_query_r1(
    device const uchar* W      [[buffer(0)]],
    device const float* q      [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant uint& Lk          [[buffer(3)]],
    constant uint& Dn          [[buffer(4)]],
    constant uint& Dv          [[buffer(5)]],
    uint output [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint head = output / Lk;
    const uint latent = output - head * Lk;
    const uint row_bytes = (Lk >> 8) * 210;
    device const float* qh = q + (size_t)head * Dn;
    const uint row0 = head * (Dn + Dv);
    float acc = 0.0f;
    for (uint d = tid; d < Dn; d += tgs)
        acc += q6_k_value(W + (size_t)(row0 + d) * row_bytes, latent) * qh[d];
    threadgroup float scratch[32];
    const float result = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[output] = result;
}

kernel void mla_q6_kv_b_query_r1_b(
    device const uchar* W [[buffer(0)]],
    device const float* q [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& Lk [[buffer(3)]],
    constant uint& Dn [[buffer(4)]],
    constant uint& Dv [[buffer(5)]],
    constant uint& HE [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint output = group.x, token = group.y;
    const uint head = output / Lk, latent = output - head * Lk;
    const uint row_bytes = (Lk >> 8) * 210;
    device const float* qh = q + ((size_t)token * HE + head) * Dn;
    const uint row0 = head * (Dn + Dv);
    float acc = 0.0f;
    for (uint d = tid; d < Dn; d += 32)
        acc += q6_k_value(W + (size_t)(row0 + d) * row_bytes, latent) * qh[d];
    threadgroup float scratch[32];
    const float result = tg_reduce_sum(acc, tid, 32, scratch);
    if (tid == 0) y[(size_t)token * HE * Lk + output] = result;
}

// o_full[h,d] = sum_k W_kv_b[h,Dn+d,k] * o_lat[h,k].
kernel void mla_q6_kv_b_value_r1(
    device const uchar* W      [[buffer(0)]],
    device const float* o_lat  [[buffer(1)]],
    device       float* y      [[buffer(2)]],
    constant uint& Lk          [[buffer(3)]],
    constant uint& Dn          [[buffer(4)]],
    constant uint& Dv          [[buffer(5)]],
    uint output [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint head = output / Dv;
    const uint dim = output - head * Dv;
    const uint row_bytes = (Lk >> 8) * 210;
    device const uchar* wr = W + (size_t)(head * (Dn + Dv) + Dn + dim) * row_bytes;
    device const float* oh = o_lat + (size_t)head * Lk;
    float acc = 0.0f;
    for (uint k = tid; k < Lk; k += tgs) acc += q6_k_value(wr, k) * oh[k];
    threadgroup float scratch[32];
    const float result = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[output] = result;
}

// ---------------------------------------------------------------------------
// GGUF q4_K fused dequant-gemv.
//
// Block layout (144 bytes / 256 weights):
//   uint16 d            offset   0
//   uint16 dmin         offset   2
//   uint8  scales[12]   offset   4   (packed 6-bit scale & min, 8 sub-blocks)
//   uint8  qs[128]      offset  16   (4-bit weights, 32 lo + 32 hi per 64)
//
// CPU oracle: dequantize_q4_K_block in src/gguf_dequant.cpp.
// Mirrors ggml-quants.c::dequantize_row_q4_K verbatim.

inline void k_get_scale_min_k4(int j, device const uchar* q,
                               thread uchar& d, thread uchar& m)
{
    if (j < 4) {
        d = q[j]     & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

inline float q4_k_partial(device const uchar* Wrow,
                          device const float* x,
                          uint nblk, uint tid, uint tgs) {
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        device const uchar* blk = Wrow + (size_t)b * 144;
        ushort d_h    = ((device const ushort*)(blk + 0))[0];
        ushort dmin_h = ((device const ushort*)(blk + 2))[0];
        float d    = float(as_type<half>(d_h));
        float dmin = float(as_type<half>(dmin_h));

        device const uchar* scales = blk + 4;
        device const uchar* qs     = blk + 16;
        device const float* xb     = x   + (size_t)b * 256;

        // Four 64-weight strips per super-block (is = 0,2,4,6).
        for (uint strip = 0; strip < 4; ++strip) {
            uchar sc1, m1c, sc2, m2c;
            k_get_scale_min_k4(2 * strip + 0, scales, sc1, m1c);
            k_get_scale_min_k4(2 * strip + 1, scales, sc2, m2c);
            float d1 = d * float(sc1);
            float m1 = dmin * float(m1c);
            float d2 = d * float(sc2);
            float m2 = dmin * float(m2c);

            device const uchar* q  = qs + strip * 32;
            device const float* xs = xb + strip * 64;

            float local = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                uchar  qb = q[l];
                float  w_lo = d1 * float(qb & 0x0F) - m1;
                float  w_hi = d2 * float(qb >>   4) - m2;
                local += w_lo * xs[l] + w_hi * xs[l + 32];
            }
            acc += local;
        }
    }
    return acc;
}

inline float q4_k_value(device const uchar* row, uint column) {
    device const uchar* blk = row + (size_t)(column >> 8) * 144;
    const uint index = column & 255u;
    const uint strip = index >> 6;
    const uint lane = index & 31u;
    const uint high = (index >> 5) & 1u;
    uchar scale, minimum;
    k_get_scale_min_k4(2 * strip + high, blk + 4, scale, minimum);
    const uchar packed = blk[16 + strip * 32 + lane];
    const uint q = high ? packed >> 4 : packed & 15u;
    const float d = float(as_type<half>(((device const ushort*)blk)[0]));
    const float dmin = float(as_type<half>(((device const ushort*)(blk + 2))[0]));
    return d * float(scale) * float(q) - dmin * float(minimum);
}

// DeepSeek fixed-projection prefill: C[B,N] = X[B,K] W[N,K]^T.
// The native Q4_K tile is expanded only into 16 KiB of threadgroup memory,
// consumed immediately by MPP TensorOps, and overwritten for the next K tile.
// All extents are exact for the accepted 128-row V0 tile and model geometry.
kernel void qmm_q4_K_f32_m32n32k128(
    device const uchar* W [[buffer(0)]],
    device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 32, TN = 32, TK = 128;
    threadgroup float weight_tile[TK * TN];
    using device_tensor = tensor<device float, dextents<int, 2>, tensor_inline>;
    using group_tensor = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
    device_tensor input(x, dextents<int, 2>(int(K), int(B)),
                        array<int, 2>({1, int(K)}));
    device_tensor output(y, dextents<int, 2>(int(N), int(B)),
                         array<int, 2>({1, int(N)}));
    const int out_col = int(group.x) * TN;
    const int out_row = int(group.y) * TM;
    auto output_tile = output.slice<TN, TM>(out_col, out_row);
    constexpr auto descriptor = matmul2d_descriptor(
        TM, TN, TK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto first_input = input.slice<TK, TM>(0, out_row);
    group_tensor weights(weight_tile, dextents<int, 2>(TN, TK),
                         array<int, 2>({1, TN}));
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < accumulator.get_capacity(); ++i)
        accumulator[i] = 0.0f;
    const uint row_bytes = (K >> 8) * 144;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        for (uint i = tid; i < TK * TN; i += 128) {
            const uint k = i / TN;
            const uint n = i - k * TN;
            weight_tile[i] = q4_k_value(W + (size_t)(out_col + n) * row_bytes,
                                        k0 + k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_tile = input.slice<TK, TM>(int(k0), out_row);
        operation.run(input_tile, weights, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    accumulator.store(output_tile);
}

kernel void gemv_q4_K_f32(
    device const uchar*  W       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant uint&       K       [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 144;

    threadgroup float scratch[32];
    float v = tg_reduce_sum(q4_k_partial(Wrow, x, nblk, tid, tgs),
                            tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// ---------------------------------------------------------------------------
// GGUF IQ1_S fused dequant-gemv.
//
// Block layout (50 bytes / 256 weights):
//   uint16 d_half       offset  0
//   uint8  qs[32]       offset  2     (low 8 bits of grid index, per group of 8)
//   uint16 qh[8]        offset 34     (per sub-block: scale, sign, 4x hi-grid-bits)
//
// Per sub-block (8 sub-blocks of 32 weights):
//   s_bits = (qh >> 12) & 0x7;          // 3 bits
//   dl     = d * (2*s_bits + 1)
//   delta  = (qh & 0x8000) ? -0.125 : +0.125
//   for g in 0..4:
//     idx   = qs[sub*4 + g] | (((qh >> (3*g)) & 0x7) << 8)   // 11-bit
//     bytes = iq1s_grid[idx]                                  // u64 = 8 i8 in {-1,0,+1}
//     w[g*8 + j] = dl * (i8(bytes >> 8j) + delta)
//
// The grid is a trained codebook (vendored verbatim from llama.cpp); we pass
// it as buffer(4) -- 16 KiB constant data, lives in cache after warmup.
//
// CPU oracle: dequantize_iq1_s_block in src/gguf_dequant.cpp.

constant float IQ1S_DELTA = 0.125f;

inline float iq1s_partial(device const uchar* Wrow,
                          device const float* x,
                          device const ulong* grid,
                          uint nblk, uint tid, uint tgs) {
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        device const uchar* blk = Wrow + (size_t)b * 50;
        float d = float(as_type<half>(((device const ushort*)blk)[0]));
        device const uchar* qs = blk + 2;
        device const ushort* qh = (device const ushort*)(blk + 34);
        device const float* xb = x + (size_t)b * 256;
        for (uint sub = 0; sub < 8; ++sub) {
            ushort q = qh[sub];
            float dl = d * float(2u * uint((q >> 12) & 7u) + 1u);
            float delta = (q & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            for (uint g = 0; g < 4; ++g) {
                uint idx = uint(qs[sub * 4 + g]) | (uint((q >> (3 * g)) & 7u) << 8);
                ulong code = grid[idx];
                device const float* xv = xb + sub * 32 + g * 8;
                float local = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j)
                    local += (float(int(char((code >> (8 * j)) & 0xff))) + delta) * xv[j];
                acc += dl * local;
            }
        }
    }
    return acc;
}

kernel void gemv_iq1_s_f32(
    device const uchar*    W    [[buffer(0)]],  // 50 * nblk per row
    device const float*    x    [[buffer(1)]],
    device       float*    y    [[buffer(2)]],
    device const ulong*    grid [[buffer(3)]],  // 2048 entries
    constant uint&         K    [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;

    threadgroup float scratch[32];
    float v = tg_reduce_sum(iq1s_partial(Wrow, x, grid, nblk, tid, tgs),
                            tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

kernel void gemv_iq1_s_f32_b(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tgs = 32;
    const uint row = group.x, batch = group.y;
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;
    device const float* xb = x + (size_t)batch * K;
    threadgroup float scratch[32];
    const float value = tg_reduce_sum(iq1s_partial(Wrow, xb, grid, nblk, tid, tgs),
                                      tid, tgs, scratch);
    if (tid == 0) y[(size_t)batch * N + row] = value;
}

// DeepSeek-R1 routed SwiGLU: reuse x while computing the IQ1_S gate/up pair,
// then round at the same fp16 boundaries as the unfused graph.
kernel void expert_gate_up_swiglu_iq1_s(
    device const uchar* Wg [[buffer(0)]],
    device const uchar* Wu [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const ulong* grid [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint nblk = K / 256;
    const size_t stride = (size_t)nblk * 50;
    threadgroup float scratch[32];
    float gate = tg_reduce_sum(iq1s_partial(Wg + row * stride, x, grid, nblk, tid, tgs),
                               tid, tgs, scratch);
    float up = tg_reduce_sum(iq1s_partial(Wu + row * stride, x, grid, nblk, tid, tgs),
                             tid, tgs, scratch);
    if (tid == 0) {
        y[row] = (gate / (1.0f + exp(-gate))) * up;
    }
}

// DeepSeek-R1 routed down projection plus weighted expert accumulation.
kernel void expert_down_accum_iq1_s(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;
    threadgroup float scratch[32];
    float value = tg_reduce_sum(iq1s_partial(Wrow, x, grid, nblk, tid, tgs),
                                tid, tgs, scratch);
    if (tid == 0)
        y[row] += alpha * value;
}

// R1 routed-down specialization: Fe=2048 is exactly eight IQ1_S blocks.
// Four independent output rows share one 32-lane SIMD-group, so all lanes
// perform useful work while retaining the same 8-value reduction tree.
kernel void expert_down_accum_iq1_s_r4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    uint row4 [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid & 7u;
    const uint row = row4 * 4u + (tid >> 3);
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;
    float value = iq1s_partial(Wrow, x, grid, nblk, lane, 8);
    value += simd_shuffle_down(value, 4);
    value += simd_shuffle_down(value, 2);
    value += simd_shuffle_down(value, 1);
    if (lane == 0) y[row] += alpha * value;
}

// Blocked prefill variants. Assignments sharing one expert execute together,
// so its quantized rows remain hot while independent prompt positions fill
// the GPU. `tokens` indexes the layer-major activation tile; `slots` indexes
// [token, top-k-rank] output records and preserves the reference sum order.
kernel void expert_gate_up_swiglu_iq1_s_b(
    device const uchar* Wg [[buffer(0)]],
    device const uchar* Wu [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const ulong* grid [[buffer(4)]],
    constant uint* tokens [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    constant uint& Fe [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tgs = 32;
    const uint row = group.x, item = group.y;
    const uint nblk = H / 256;
    const size_t stride = (size_t)nblk * 50;
    device const float* input = x + (size_t)tokens[item] * H;
    threadgroup float scratch[32];
    float gate = tg_reduce_sum(iq1s_partial(Wg + row * stride, input, grid,
                                            nblk, tid, tgs), tid, tgs, scratch);
    float up = tg_reduce_sum(iq1s_partial(Wu + row * stride, input, grid,
                                          nblk, tid, tgs), tid, tgs, scratch);
    if (tid == 0) y[(size_t)item * Fe + row] =
        (gate / (1.0f + exp(-gate))) * up;
}

kernel void expert_down_iq1_s_b(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint* slots [[buffer(4)]],
    constant uint& Fe [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint tgs = 32;
    const uint row = group.x, item = group.y;
    const uint nblk = Fe / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;
    device const float* input = x + (size_t)item * Fe;
    threadgroup float scratch[32];
    float value = tg_reduce_sum(iq1s_partial(Wrow, input, grid, nblk, tid, tgs),
                                tid, tgs, scratch);
    if (tid == 0)
        y[(size_t)slots[item] * H + row] = value;
}

kernel void expert_down_iq1_s_b_r4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint* slots [[buffer(4)]],
    constant uint& Fe [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid & 7u;
    const uint row = group.x * 4u + (tid >> 3);
    const uint item = group.y;
    const uint nblk = Fe / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 50;
    device const float* input = x + (size_t)item * Fe;
    float value = iq1s_partial(Wrow, input, grid, nblk, lane, 8);
    value += simd_shuffle_down(value, 4);
    value += simd_shuffle_down(value, 2);
    value += simd_shuffle_down(value, 1);
    if (lane == 0) y[(size_t)slots[item] * H + row] = value;
}

kernel void moe_residual_merge_f32(
    device float* x [[buffer(0)]],
    device const float* shared [[buffer(1)]],
    device const float* routed [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    constant uint& H [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const uint token = gid / H, d = gid - token * H;
    float value = shared[gid];
    const size_t base = (size_t)token * K * H + d;
    for (uint rank = 0; rank < K; ++rank)
        value += weights[token * K + rank] * routed[base + (size_t)rank * H];
    x[gid] += value;
}

// ---------------------------------------------------------------------------
// GGUF q5_K fused dequant-gemv.
//
// Block layout (176 bytes / 256 weights):
//   uint16 d            offset   0
//   uint16 dmin         offset   2
//   uint8  scales[12]   offset   4   (packed 6-bit scale & min, 8 sub-blocks)
//   uint8  qh[32]       offset  16   (high bit per weight)
//   uint8  qs[128]      offset  48   (4-bit weights, 32 lo + 32 hi per 64)
//
// Same scale/min packing as q4_K; on top of each 4-bit nibble we OR in a
// 5th high bit from qh, selected by a per-strip mask that walks 0x01,0x02,
// 0x04,0x08 (low nibbles) and 0x02,0x04,0x08,0x10 -> stored as u1/u2 in
// the CPU oracle. Each weight = d * sc * (nib + (qh_bit ? 16 : 0)) - dmin*m.
//
// CPU oracle: dequantize_q5_K_block in src/gguf_dequant.cpp.

inline float q5_k_partial(device const uchar* Wrow,
                          device const float* x,
                          uint nblk, uint tid, uint tgs) {
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        device const uchar* blk = Wrow + (size_t)b * 176;
        ushort d_h    = ((device const ushort*)(blk + 0))[0];
        ushort dmin_h = ((device const ushort*)(blk + 2))[0];
        float d    = float(as_type<half>(d_h));
        float dmin = float(as_type<half>(dmin_h));

        device const uchar* scales = blk + 4;
        device const uchar* qh     = blk + 16;
        device const uchar* qs     = blk + 48;
        device const float* xb     = x   + (size_t)b * 256;

        uchar u1 = 1, u2 = 2;
        for (uint strip = 0; strip < 4; ++strip) {
            uchar sc1, m1c, sc2, m2c;
            k_get_scale_min_k4(2 * strip + 0, scales, sc1, m1c);
            k_get_scale_min_k4(2 * strip + 1, scales, sc2, m2c);
            float d1 = d * float(sc1);
            float m1 = dmin * float(m1c);
            float d2 = d * float(sc2);
            float m2 = dmin * float(m2c);

            device const uchar* q  = qs + strip * 32;
            device const float* xs = xb + strip * 64;

            float local = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                uchar  qb  = q[l];
                uchar  qhb = qh[l];
                float  w_lo = d1 * float((qb & 0x0F) + ((qhb & u1) ? 16 : 0)) - m1;
                float  w_hi = d2 * float((qb >>   4) + ((qhb & u2) ? 16 : 0)) - m2;
                local += w_lo * xs[l] + w_hi * xs[l + 32];
            }
            acc += local;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return acc;
}

inline float q5_k_value(device const uchar* row, uint column) {
    device const uchar* blk = row + (size_t)(column >> 8) * 176;
    const uint index = column & 255u;
    const uint strip = index >> 6;
    const uint lane = index & 31u;
    const uint high = (index >> 5) & 1u;
    uchar scale, minimum;
    k_get_scale_min_k4(2 * strip + high, blk + 4, scale, minimum);
    const uchar packed = blk[48 + strip * 32 + lane];
    const uint nibble = high ? packed >> 4 : packed & 15u;
    const uint mask = 1u << (2 * strip + high);
    const uint q = nibble + ((blk[16 + lane] & mask) ? 16u : 0u);
    const float d = float(as_type<half>(((device const ushort*)blk)[0]));
    const float dmin = float(as_type<half>(((device const ushort*)(blk + 2))[0]));
    return d * float(scale) * float(q) - dmin * float(minimum);
}

kernel void qmm_q5_K_f32_m32n32k128(
    device const uchar* W [[buffer(0)]],
    device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 32, TN = 32, TK = 128;
    threadgroup float weight_tile[TK * TN];
    using device_tensor = tensor<device float, dextents<int, 2>, tensor_inline>;
    using group_tensor = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
    device_tensor input(x, dextents<int, 2>(int(K), int(B)),
                        array<int, 2>({1, int(K)}));
    device_tensor output(y, dextents<int, 2>(int(N), int(B)),
                         array<int, 2>({1, int(N)}));
    const int out_col = int(group.x) * TN;
    const int out_row = int(group.y) * TM;
    auto output_tile = output.slice<TN, TM>(out_col, out_row);
    constexpr auto descriptor = matmul2d_descriptor(
        TM, TN, TK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto first_input = input.slice<TK, TM>(0, out_row);
    group_tensor weights(weight_tile, dextents<int, 2>(TN, TK),
                         array<int, 2>({1, TN}));
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < accumulator.get_capacity(); ++i)
        accumulator[i] = 0.0f;
    const uint row_bytes = (K >> 8) * 176;
    for (uint k0 = 0; k0 < K; k0 += TK) {
        for (uint i = tid; i < TK * TN; i += 128) {
            const uint k = i / TN;
            const uint n = i - k * TN;
            weight_tile[i] = q5_k_value(W + (size_t)(out_col + n) * row_bytes,
                                        k0 + k);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_tile = input.slice<TK, TM>(int(k0), out_row);
        operation.run(input_tile, weights, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    accumulator.store(output_tile);
}

kernel void gemv_q5_K_f32(
    device const uchar*  W       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant uint&       K       [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 176;

    threadgroup float scratch[32];
    float v = tg_reduce_sum(q5_k_partial(Wrow, x, nblk, tid, tgs),
                            tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// ---------------------------------------------------------------------------
// GGUF iq2_xxs fused dequant-gemv.
//
// Block layout (66 bytes / 256 weights):
//   uint16 d           offset 0
//   uint16 qs[16]      offset 2   (8 sub-blocks of 32 weights; each =
//                                   2 uint32: aux32[0]=4 grid-byte indices,
//                                   aux32[1]=4*7-bit signs (28b) + scale4(4b))
// Per sub-block:
//   db = d * (0.5 + (aux32[1] >> 28)) * 0.25
//   for g in 0..4:
//     grid_bytes = iq2xxs_grid[aux8[g]]                 // 8 bytes in [0..]
//     signs      = ksigns_iq2xs[(aux32[1] >> 7*g) & 0x7f]   // 8-bit mask
//     for j in 0..8:
//       sign = (signs & (1<<j)) ? -1 : +1
//       w    = db * grid_bytes[j] * sign
//
// We bind iq2xxs_grid at buffer(4) (256 * u64 = 2 KiB). ksigns_iq2xs and
// kmask_iq2xs are tiny enough to inline as `constant` arrays here.
//
// CPU oracle: dequantize_iq2_xxs_block in src/gguf_dequant.cpp.

constant uchar IQ2XXS_KMASK[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

constant uchar IQ2XXS_KSIGNS[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

inline float iq2_xxs_partial(device const uchar* Wrow,
                             device const float* x,
                             device const ulong* grid,
                             uint nblk, uint tid, uint tgs) {
    float acc = 0.0f;
    for (uint b = tid; b < nblk; b += tgs) {
        device const uchar*  blk = Wrow + (size_t)b * 66;
        ushort d_h = ((device const ushort*)(blk + 0))[0];
        float  d   = float(as_type<half>(d_h));
        device const uchar*  qs = blk + 2;
        device const float*  xb = x + (size_t)b * 256;

        for (uint sub = 0; sub < 8; ++sub) {
            device const uchar* sub_bytes = qs + 8 * sub;
            // qs starts at blk+2 (uint16-aligned only); a uint* cast would
            // be misaligned for every block whose row stride is 66 bytes.
            // Assemble the two uint32s byte-wise to stay safe.
            uint a0 = uint(sub_bytes[0])       | (uint(sub_bytes[1]) <<  8)
                    | (uint(sub_bytes[2]) <<16) | (uint(sub_bytes[3]) << 24);
            uint a1 = uint(sub_bytes[4])       | (uint(sub_bytes[5]) <<  8)
                    | (uint(sub_bytes[6]) <<16) | (uint(sub_bytes[7]) << 24);
            float db = d * (0.5f + float(a1 >> 28)) * 0.25f;

            device const float* xs = xb + sub * 32;
            for (uint g = 0; g < 4; ++g) {
                uint  gi    = (a0 >> (8 * g)) & 0xff;
                ulong gbits = grid[gi];
                uchar signs = IQ2XXS_KSIGNS[(a1 >> (7 * g)) & 0x7f];

                device const float* xg = xs + g * 8;
                float local = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; ++j) {
                    float gv = float((gbits >> (8 * j)) & 0xff);
                    float s  = (signs & IQ2XXS_KMASK[j]) ? -1.0f : +1.0f;
                    local += gv * s * xg[j];
                }
                acc += db * local;
            }
        }
    }
    return acc;
}

kernel void gemv_iq2_xxs_f32(
    device const uchar*    W    [[buffer(0)]],
    device const float*    x    [[buffer(1)]],
    device       float*    y    [[buffer(2)]],
    device const ulong*    grid [[buffer(3)]],
    constant uint&         K    [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    const uint nblk = K / 256;
    device const uchar* Wrow = W + (size_t)row * nblk * 66;
    threadgroup float scratch[32];
    float v = tg_reduce_sum(iq2_xxs_partial(Wrow, x, grid, nblk, tid, tgs),
                            tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// Layers 3..8 use IQ2_XXS for routed down. Fe=2048 is eight blocks, so four
// output rows occupy one SIMD-group exactly as in the proven IQ1_S mapping.
kernel void expert_down_iq2_xxs_b_r4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const ulong* grid [[buffer(3)]],
    constant uint* slots [[buffer(4)]],
    constant uint& Fe [[buffer(5)]],
    constant uint& H [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lane = tid & 7u;
    const uint row = group.x * 4u + (tid >> 3);
    const uint item = group.y;
    constexpr uint nblk = 8;
    device const uchar* Wrow = W + (size_t)row * nblk * 66;
    device const float* input = x + (size_t)item * Fe;
    float value = iq2_xxs_partial(Wrow, input, grid, nblk, lane, 8);
#pragma unroll
    for (uint packed_row = 0; packed_row < 4; ++packed_row) {
        const float packed = tid < 8
            ? simd_shuffle(value, packed_row * 8 + tid) : 0.0f;
        const float sum = simd_sum(packed);
        if (tid == 0)
            y[(size_t)slots[item] * H + group.x * 4 + packed_row] = sum;
    }
}

// ---------------------------------------------------------------------------
// gemv_f32_f16: fp32 weight matrix × fp16 vector -> fp16 vector.
// Used for the f32-stored tensors in the R1 IQ1_S mix (mixie_plan.txt:32-39):
//   361 tensors at 0.43 GB total -- norms, router (ffn_gate_inp), exp_probs_b
//   bias, output_norm. f32 is treated as block-of-1 element (4 bytes), so
//   bytes_per_block(GGML_F32)==4 in gguf_kernels.hpp lines up with `K * 4`
//   bytes per row here.
//
// Signature matches the quant gemv kernels in this file:
//   (W, x, y, K)  with K bound at buffer(3) as `constant uint&`.
// Dispatch: one threadgroup per output row, TG_GEMV threads cooperating via
// tg_reduce_sum. No grid table, no quant unpack -- just a plain dot product
// hoisted into the same calling convention so kernel_name_for(GGML_F32)
// returns a real PSO and metal_ctx.mm:81 stops aborting.
// ---------------------------------------------------------------------------
kernel void gemv_f32_f32(
    device const float*  W       [[buffer(0)]],   // K floats per row, row-major
    device const float*  x       [[buffer(1)]],
    device       float*  y       [[buffer(2)]],
    constant uint&       K       [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const float* Wrow = W + (size_t)row * (size_t)K;

    float acc = 0.0f;
    for (uint i = tid; i < K; i += tgs) {
        acc += Wrow[i] * x[i];
    }

    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// Exact batched router projection. Each (token,row) threadgroup executes the
// same 32-lane dot/reduction tree as gemv_f32_f32; only dispatch encoding is
// shared across the 128-token prefill tile.
kernel void gemv_f32_f32_b(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    device const float* wr = W + (size_t)group.x * K;
    device const float* xr = x + (size_t)group.y * K;
    float acc = 0.0f;
    for (uint i = tid; i < K; i += 32) acc += wr[i] * xr[i];
    threadgroup float scratch[32];
    const float v = tg_reduce_sum(acc, tid, 32, scratch);
    if (tid == 0) y[(size_t)group.y * N + group.x] = v;
}

kernel void gemv_f32_f32_grouped(
    device const float* W          [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device       float* y          [[buffer(2)]],
    constant uint& K               [[buffer(3)]],
    constant uint& group_size      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    device const float* wr = W + (size_t)row * K;
    device const float* xr = x + (size_t)(row / group_size) * K;
    float acc = 0.0f;
    for (uint i = tid; i < K; i += tgs) acc += wr[i] * xr[i];
    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

kernel void gemv_f16_f32_grouped(
    device const half* W          [[buffer(0)]],
    device const float* x         [[buffer(1)]],
    device       float* y         [[buffer(2)]],
    constant uint& K              [[buffer(3)]],
    constant uint& group_size     [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]) {
    device const half* wr = W + (size_t)row * K;
    device const float* xr = x + (size_t)(row / group_size) * K;
    float acc = 0.0f;
    for (uint i = tid; i < K; i += tgs) acc += float(wr[i]) * xr[i];
    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = v;
}

// ---------------------------------------------------------------------------
// gemv_f16_f16: fp16 weight matrix × fp16 vector -> fp16 vector.
// Grouped variant: `group_size` selects which x slice each row reads, matching
// the gemv_fp8_f16 signature in this file. Plain (non-grouped) GEMV when
// group_size == M (every row reads x[0..K)).
//
// Used by the GGUF R1 path for the absorbed per-head W_uk / W_uv projections.
// These are derived from the q4_K-quantized attn_kv_b.weight at first decode
// use (per-layer CPU dequant + per-head reshape into fp16) and held resident
// in fp16 -- ~32 MB per layer (16 MB W_uk + 16 MB W_uv at HE=128, Lk=512,
// Dn=Dv=128). Stays well under the 24 GB UMA budget when summed across all
// 61 layers (~2 GB), and avoids the much-worse alternative of a 16 GB
// per-head expanded KV cache.
//
// Signature uses the same (W, x, y, K, group) ABI as gemv_fp8_f16 and
// gemv_bf16_f16, so ops.hpp::gemv() drives it without modification.
// ---------------------------------------------------------------------------
kernel void gemv_f16_f16(
    device const half*   W          [[buffer(0)]],   // [M, K] row-major
    device const half*   x          [[buffer(1)]],   // [M/group_size * K]
    device       half*   y          [[buffer(2)]],   // [M]
    constant uint&       K          [[buffer(3)]],
    constant uint&       group_size [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const half* Wrow = W + (size_t)row * (size_t)K;
    device const half* xh   = x + (size_t)(row / group_size) * (size_t)K;

    float acc = 0.0f;
    for (uint i = tid; i < K; i += tgs) {
        acc += float(Wrow[i]) * float(xh[i]);
    }

    threadgroup float scratch[32];
    float v = tg_reduce_sum(acc, tid, tgs, scratch);
    if (tid == 0) y[row] = half(v);
}
