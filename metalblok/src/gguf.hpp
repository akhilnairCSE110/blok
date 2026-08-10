// GGUF v3 reader. Standalone: no llama.cpp / ggml dependency.
//
// Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
// We support what Unsloth's UD-IQ1_S DeepSeek-R1 release uses:
//   - GGUF v3 header
//   - all primitive KV value types
//   - string and primitive arrays (skipped if huge -- e.g. tokenizer.tokens)
//   - 6 tensor quant types: F32, Q4_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS
//   (other types parse fine and are exposed; we just don't compute their byte
//    size yet, so calling tensor_bytes() on them aborts.)
//
// Multi-shard: GGUF supports a 3-key split convention
//   split.no, split.count, split.tensors.count
// Caller passes shard 0 path; we autodetect siblings via the
// "<base>-NNNNN-of-MMMMM.gguf" suffix and merge tensor tables, with
// per-tensor shard-relative offsets. Payload bytes are never mapped.

#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace blade {

// GGML tensor type ids -- subset we care about.  Numeric values match
// llama.cpp / ggml so dumps line up with their tooling.
enum GgmlType : uint32_t {
    GGML_F32     = 0,
    GGML_F16     = 1,
    GGML_Q4_0    = 2,
    GGML_Q4_1    = 3,
    GGML_Q5_0    = 6,
    GGML_Q5_1    = 7,
    GGML_Q8_0    = 8,
    GGML_Q8_1    = 9,
    GGML_Q2_K    = 10,
    GGML_Q3_K    = 11,
    GGML_Q4_K    = 12,
    GGML_Q5_K    = 13,
    GGML_Q6_K    = 14,
    GGML_Q8_K    = 15,
    GGML_IQ2_XXS = 16,
    GGML_IQ2_XS  = 17,
    GGML_IQ3_XXS = 18,
    GGML_IQ1_S   = 19,
    GGML_IQ4_NL  = 20,
    GGML_IQ3_S   = 21,
    GGML_IQ2_S   = 22,
    GGML_IQ4_XS  = 23,
    GGML_BF16    = 30,
};

// Returns short name like "q4_K" or empty string for unknowns.
const char* ggml_type_name(uint32_t t);

// Returns elements per quant block (1 for f32/f16/bf16, 32 for q4_0/q8_0,
// 256 for k-quants and i-quants).  Returns 0 for unknown types.
uint32_t ggml_block_size(uint32_t t);

// Returns bytes per quant block.  Returns 0 for unknown types.
uint32_t ggml_type_bytes(uint32_t t);

// GGUF metadata value types (the wire-level enum -- different namespace
// from GGML tensor types).
enum GgufValueType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3,
    GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9,
    GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12,
};

// One key/value entry from the metadata block.  For arrays we store the
// element type + element count and a (start, end) byte range into the
// owning shard's mmap, so callers can decode lazily and we don't pay a
// 130k-element copy for tokenizer.tokens.
struct GgufKV {
    std::string   key;
    GgufValueType type;            // top-level type
    // For non-array primitives, the value is decoded into one of these.
    // Exactly one is meaningful, selected by `type`.
    uint64_t      u64 = 0;          // covers all int/bool widths
    int64_t       i64 = 0;
    double        f64 = 0.0;
    std::string   str;
    // For arrays.
    GgufValueType arr_type = GGUF_U8;
    uint64_t      arr_count = 0;
    const uint8_t* arr_data = nullptr;   // points into mmap
    uint64_t       arr_bytes = 0;        // total bytes in arr_data (0 for str arrays)
};

// One tensor descriptor + a resolved data pointer.
struct GgufTensor {
    std::string           name;
    uint32_t              type = 0;            // GgmlType
    std::vector<uint64_t> dims;                // row-major (ne[0] is innermost)
    uint64_t              offset_in_data = 0;  // offset within the shard's data segment
    const uint8_t*        data = nullptr;      // always null; payload uses pread
    uint64_t              bytes = 0;           // computed payload size
    uint32_t              shard = 0;           // index into Gguf::shards
};

struct GgufShard {
    std::string    path;
    int            fd = -1;
    const uint8_t* base = nullptr;       // mmap base
    size_t         size = 0;
    size_t         mapped_size = 0;      // bounded metadata window only
    uint64_t       data_offset = 0;      // start of tensor payload region
    uint64_t       alignment = 32;       // from general.alignment if present
};

// ---------------------------------------------------------------------------
// Lightweight tensor reference for the "declared working set" code path.
//
// The full `Gguf` object holds persistent mmaps of every shard (147 GB
// virtual for R1) plus a 1025-entry tensor table for the lifetime of the
// process.  That's fine for the full decode loop where we WILL touch most
// tensors, but it's catastrophic for one-shot tools (validator, dump,
// probe) that need exactly one tensor.
//
// `Gguf::lookup()` reads only the metadata window of the relevant shard
// (~16 MB for R1-IQ1_S), drops every mapping and allocation before
// returning, and hands back a TensorRef.  The caller pread()s `bytes`
// from `shard_path` at `abs_offset`.  No persistent state.
// ---------------------------------------------------------------------------
struct TensorRef {
    std::string           name;
    uint32_t              type = 0;       // GgmlType
    std::vector<uint64_t> dims;
    uint64_t              bytes = 0;
    std::string           shard_path;     // absolute path to the shard file
    uint64_t              abs_offset = 0; // byte offset within shard_path
};

class Gguf {
public:
    // Magic numbers as parsed from shard 0.
    uint32_t version = 0;
    uint64_t total_tensor_count = 0;     // from split.tensors.count if multi-shard
    uint32_t shard_count = 0;            // from split.count, defaults to 1

    std::vector<GgufShard>  shards;
    std::vector<GgufKV>     kv;
    std::vector<GgufTensor> tensors;

    // Index for O(1) lookup by tensor name.
    std::unordered_map<std::string, uint32_t> tensor_idx;

    // Index for O(1) lookup by KV key (points into kv[]).
    std::unordered_map<std::string, uint32_t> kv_idx;

    // Open `path` and (if it has split.no/split.count) all sibling shards.
    // Aborts on any I/O or format error -- this is a load-once cold path,
    // we don't try to be clever about recovery.
    void open(const std::string& path);

    // Release all mmaps + fds.
    void close();
    ~Gguf() { close(); }

    // Convenience accessors.  Abort if missing / wrong type.
    uint32_t    get_u32(const char* key) const;
    uint64_t    get_u64(const char* key) const;
    int32_t     get_i32(const char* key) const;
    float       get_f32(const char* key) const;
    bool        get_bool(const char* key) const;
    std::string get_string(const char* key) const;

    // Optional accessors: return def if key is missing.
    uint32_t    get_u32_or(const char* key, uint32_t def) const;
    float       get_f32_or(const char* key, float def) const;
    std::string get_string_or(const char* key, const char* def) const;

    // Tensor lookup; returns nullptr if absent.
    const GgufTensor* find(const std::string& name) const;

    // Number of bytes summed over all tensor payloads (for sanity vs file size).
    uint64_t total_tensor_bytes() const;

    // -----------------------------------------------------------------------
    // Lazy single-tensor lookup.  Does NOT instantiate a Gguf; reads only the
    // metadata window of the shard(s), drops the mapping on return.  Used by
    // tools/validators that want exactly one tensor and refuse to spend RAM
    // on the rest.  Multi-shard files are searched transparently.  Returns
    // false if `tname` is not present in any shard.
    // -----------------------------------------------------------------------
    static bool lookup(const std::string& path, const std::string& tname,
                       TensorRef& out);
};

} // namespace blade
