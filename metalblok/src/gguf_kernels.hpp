// SPDX-License-Identifier: MIT
// gguf_kernels.hpp — single source of truth for the per-GGUF-type Metal
// kernel dispatch on the R1 path. Header-only on purpose: callers (S2
// streaming fetcher, S6 .gguf dispatch, validate_gemv) just need the
// kernel symbol name and the bytes-per-block to size a Metal launch.
//
// Coverage = the 6 quant types present in DeepSeek-R1-UD-IQ1_S:
//   f32, q4_K, q5_K, q6_K, iq2_xxs, iq1_s
// (everything else returns nullptr / 0 so callers fail loud.)
//
// bytes_per_block values match the hardcoded table in main.cpp's
// validate_gemv() switch (q4_K=144, q5_K=176, q6_K=210, iq2_xxs=66,
// iq1_s=50). f32 is treated as block-of-1 element (4 bytes).
//
// Kernel names follow the gemv_<type>_f16 convention used by the
// already-validated kernels in src/kernels.metal. The q5_K and
// iq2_xxs entries are placeholders until S3/S4 land — looking them
// up before then is a programmer error, not a runtime fallback.

#pragma once

#include <cstdint>
#include "gguf.hpp"

namespace blade {

// Returns the Metal kernel function name for a GGUF f16 GEMV against a
// weight of the given ggml type, or nullptr if unsupported on the R1 path.
inline const char* kernel_name_for(uint32_t ggml_type) {
    switch (ggml_type) {
        case GGML_F32:     return "gemv_f32_f32";
        case GGML_Q4_K:    return "gemv_q4_K_f32";
        case GGML_Q5_K:    return "gemv_q5_K_f32";
        case GGML_Q6_K:    return "gemv_q6_K_f32";
        case GGML_IQ2_XXS: return "gemv_iq2_xxs_f32";
        case GGML_IQ1_S:   return "gemv_iq1_s_f32";
        default:           return nullptr;
    }
}

// Returns the size in bytes of one quant block for the given ggml type,
// or 0 if unsupported. f32 is reported as 4 (block = 1 element).
// Values match validate_gemv() in main.cpp.
inline uint32_t bytes_per_block(uint32_t ggml_type) {
    switch (ggml_type) {
        case GGML_F32:     return   4;
        case GGML_Q4_K:    return 144;
        case GGML_Q5_K:    return 176;
        case GGML_Q6_K:    return 210;
        case GGML_IQ2_XXS: return  66;
        case GGML_IQ1_S:   return  50;
        default:           return   0;
    }
}

} // namespace blade
