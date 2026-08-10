// CPU reference dequantizers. See gguf_dequant.hpp.
//
// IQ1_S decoder follows the format documented in
// vendor/llama_cpp/iq1s_grid.h (cross-checked against ggml-common.h).

#include "gguf_dequant.hpp"
#include "../vendor/llama_cpp/iq1s_grid.h"
#include "../vendor/llama_cpp/iq2xxs_grid.h"

#include <cstring>

namespace blade {

// IEEE-754 binary16 -> binary32. Exact for all 65536 inputs incl. subnormals,
// inf, NaN. fp16 subnormal value = mantissa * 2^-24, so for non-zero subnormals
// we left-shift the mantissa to normalize, decrementing the exponent each time
// (base = -14, the smallest fp16 normal exponent).
static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15);
    const uint32_t e = (uint32_t)(h >> 10) & 0x1f;
    const uint32_t m = (uint32_t)h         & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s << 31; }                 // ±0
        else {
            uint32_t mm = m;
            int ee = -14;
            while ((mm & 0x400) == 0) { mm <<= 1; --ee; }
            bits = (s << 31) | ((uint32_t)(127 + ee) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | (0xffu << 23) | (m << 13);   // inf / NaN
    } else {
        bits = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void dequantize_iq1_s_block(const void* block_v, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(block_v);

    // Layout (50 bytes total):
    //   uint16  d_half           offset 0
    //   uint8   qs[32]           offset 2
    //   uint16  qh[8]            offset 34
    uint16_t d_half;
    std::memcpy(&d_half, p, 2);
    const float d = fp16_to_fp32(d_half);

    const uint8_t* qs       = p + 2;
    const uint8_t* qh_bytes = p + 34;

    // 8 sub-blocks of 32 weights each. Reference dequantizer
    // (ggml-quants.c::dequantize_row_iq1_s):
    //
    //   dl    = d * (2*scale + 1)
    //   delta = (qh & 0x8000) ? -IQ1S_DELTA : +IQ1S_DELTA
    //   w     = dl * (grid_byte + delta)              // no separate ml offset
    //
    // The delta is a small (±0.125) jitter applied uniformly within the
    // sub-block; the sign bit of qh selects which side of the codebook
    // the block was assigned during quantization.
    for (int sub = 0; sub < 8; ++sub) {
        uint16_t qh;
        std::memcpy(&qh, qh_bytes + 2 * sub, 2);

        const uint32_t s_bits = (uint32_t)((qh >> 12) & 0x7);
        const float    dl     = d * (2.0f * (float)s_bits + 1.0f);
        const float    delta  = (qh & 0x8000u) ? -vendored::IQ1S_DELTA
                                               : +vendored::IQ1S_DELTA;

        // 4 groups of 8 weights per sub-block. The high-3-bit "page" of
        // the grid index for each group is packed into qh bits 0..11.
        for (int g = 0; g < 4; ++g) {
            const uint32_t lo  = qs[sub * 4 + g];
            const uint32_t hi  = (uint32_t)((qh >> (3 * g)) & 0x7);
            const uint32_t idx = lo | (hi << 8);

            const uint64_t bytes = vendored::iq1s_grid[idx];
            float* w = out + sub * 32 + g * 8;
            for (int j = 0; j < 8; ++j) {
                const int8_t sj = (int8_t)((bytes >> (8 * j)) & 0xff);  // -1, 0, or +1
                w[j] = dl * ((float)sj + delta);
            }
        }
    }
}

void dequantize_iq1_s(const void* blocks_v, std::size_t n_super_blocks, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(blocks_v);
    for (std::size_t b = 0; b < n_super_blocks; ++b) {
        dequantize_iq1_s_block(p + 50 * b, out + 256 * b);
    }
}

// ---------------------------------------------------------------------------
// k-quants helper: unpack 6-bit scale/min from the packed 12-byte array.
// Mirrors `get_scale_min_k4` in ggml-quants.c (line 818). `j` in [0..7].
static inline void get_scale_min_k4(int j, const uint8_t* q,
                                    uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j]     & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// q4_K  (144 bytes / 256 weights)  --  ggml-quants.c::dequantize_row_q4_K (1467)
//   uint16 d         offset   0
//   uint16 dmin      offset   2
//   uint8  scales[12] offset  4    (packed 6-bit scale & min, 8 sub-blocks)
//   uint8  qs[128]   offset  16    (4-bit weights, 32 lo + 32 hi per 64)
void dequantize_q4_K_block(const void* block_v, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(block_v);
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    p + 0, 2);
    std::memcpy(&dmin_h, p + 2, 2);
    const float d    = fp16_to_fp32(d_h);
    const float dmin = fp16_to_fp32(dmin_h);

    const uint8_t* scales = p + 4;
    const uint8_t* q      = p + 16;

    int is = 0;
    float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l) y[l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) y[l + 32] = d2 * (q[l] >>  4) - m2;
        y  += 64;
        q  += 32;
        is += 2;
    }
}

// ---------------------------------------------------------------------------
// q5_K  (176 bytes / 256 weights)  --  ggml-quants.c::dequantize_row_q5_K (1669)
//   uint16 d         offset   0
//   uint16 dmin      offset   2
//   uint8  scales[12] offset  4
//   uint8  qh[32]    offset  16    (high bit per weight)
//   uint8  qs[128]   offset  48    (low 4 bits per weight)
void dequantize_q5_K_block(const void* block_v, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(block_v);
    uint16_t d_h, dmin_h;
    std::memcpy(&d_h,    p + 0, 2);
    std::memcpy(&dmin_h, p + 2, 2);
    const float d    = fp16_to_fp32(d_h);
    const float dmin = fp16_to_fp32(dmin_h);

    const uint8_t* scales = p + 4;
    const uint8_t* qh     = p + 16;
    const uint8_t* ql     = p + 48;

    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l)
            y[l]      = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l)
            y[l + 32] = d2 * ((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        y  += 64;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// ---------------------------------------------------------------------------
// q6_K  (210 bytes / 256 weights)  --  ggml-quants.c::dequantize_row_q6_K (1877)
//   uint8  ql[128]      offset   0  (low 4 bits per weight)
//   uint8  qh[64]       offset 128  (high 2 bits per weight)
//   int8   scales[16]   offset 192  (per 16-weight sub-block scale)
//   uint16 d            offset 208
void dequantize_q6_K_block(const void* block_v, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(block_v);
    const uint8_t* ql_base = p + 0;
    const uint8_t* qh_base = p + 128;
    const int8_t*  sc_base = reinterpret_cast<const int8_t*>(p + 192);
    uint16_t d_h;
    std::memcpy(&d_h, p + 208, 2);
    const float d = fp16_to_fp32(d_h);

    float* y = out;
    const uint8_t* ql = ql_base;
    const uint8_t* qh = qh_base;
    const int8_t*  sc = sc_base;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ---------------------------------------------------------------------------
// iq2_xxs  (66 bytes / 256 weights)  --  ggml-quants.c::dequantize_row_iq2_xxs (2412)
//   uint16 d         offset   0
//   uint16 qs[16]    offset   2    (8 groups of 32 weights; each group =
//                                    8 bytes = 2*uint32 [aux32[0]=4 grid bytes,
//                                    aux32[1]=signs(28b)+scale4(4b)])
void dequantize_iq2_xxs_block(const void* block_v, float* out) {
    const uint8_t* p = static_cast<const uint8_t*>(block_v);
    uint16_t d_h;
    std::memcpy(&d_h, p + 0, 2);
    const float d = fp16_to_fp32(d_h);
    const uint8_t* qs = p + 2;

    float* y = out;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux32[2];
        std::memcpy(aux32, qs + 8 * ib32, 8);
        const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);  // 4 grid byte-indices
        const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

        for (int l = 0; l < 4; ++l) {
            const uint64_t  grid = vendored::iq2xxs_grid[aux8[l]];
            const uint8_t   signs = vendored::ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            const uint8_t*  g     = reinterpret_cast<const uint8_t*>(&grid);
            for (int j = 0; j < 8; ++j) {
                const float s = (signs & vendored::kmask_iq2xs[j]) ? -1.0f : +1.0f;
                y[j] = db * (float)g[j] * s;
            }
            y += 8;
        }
    }
}

} // namespace blade
