// CPU reference dequantizers for GGUF k-quant / i-quant weight blocks.
//
// Purpose: a byte-exact oracle that GPU kernels (still to be written) can
// be validated against. Performance is irrelevant here -- correctness is
// the entire product. We test these against real R1 tensors from the
// command line (`metalblok --probe-tensor <file> <name>`) and against vendored test
// vectors when we add them.
//
// These are clean-room implementations from the published format; only
// the trained LUTs (e.g. iq1s_grid) are vendored verbatim.

#pragma once
#include <cstddef>
#include <cstdint>

namespace blade {

// Dequantize one IQ1_S super-block (256 weights, 50 bytes input) into out[].
// `block` points at the block_iq1_s struct as laid out in the GGUF file:
//   bytes 0..1   : fp16 d            (little-endian half)
//   bytes 2..33  : uint8 qs[32]
//   bytes 34..49 : uint16 qh[8]      (little-endian)
// Writes 256 floats to out.
void dequantize_iq1_s_block(const void* block, float* out);

// Dequantize a full IQ1_S tensor (n_super_blocks * 256 weights).
// Convenience over the per-block call; same numerics.
void dequantize_iq1_s(const void* blocks, std::size_t n_super_blocks, float* out);

// One super-block (256 weights) dequantizers for the remaining types in
// the Unsloth UD-IQ1_S R1 release. Block byte sizes:
//   q4_K     = 144 bytes : d(fp16), dmin(fp16), scales[12], qs[128]
//   q5_K     = 176 bytes : d(fp16), dmin(fp16), scales[12], qh[32], qs[128]
//   q6_K     = 210 bytes : ql[128], qh[64], scales[16 i8], d(fp16)
//   iq2_xxs  =  66 bytes : d(fp16), qs[16 u16]
// All formulas match ggml/src/ggml-quants.c reference dequantizers
// (cross-checked at sparse-checkout commit bbeb89d).
void dequantize_q4_K_block   (const void* block, float* out);
void dequantize_q5_K_block   (const void* block, float* out);
void dequantize_q6_K_block   (const void* block, float* out);
void dequantize_iq2_xxs_block(const void* block, float* out);

} // namespace blade
