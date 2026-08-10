// GGUF v3 parser. See gguf.hpp for the contract.

#include "gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <regex>
#include <string>

namespace blade {

// ----------------------------------------------------------------------------
// type tables
// ----------------------------------------------------------------------------

const char* ggml_type_name(uint32_t t) {
    switch (t) {
        case GGML_F32:     return "f32";
        case GGML_F16:     return "f16";
        case GGML_Q4_0:    return "q4_0";
        case GGML_Q4_1:    return "q4_1";
        case GGML_Q5_0:    return "q5_0";
        case GGML_Q5_1:    return "q5_1";
        case GGML_Q8_0:    return "q8_0";
        case GGML_Q8_1:    return "q8_1";
        case GGML_Q2_K:    return "q2_K";
        case GGML_Q3_K:    return "q3_K";
        case GGML_Q4_K:    return "q4_K";
        case GGML_Q5_K:    return "q5_K";
        case GGML_Q6_K:    return "q6_K";
        case GGML_Q8_K:    return "q8_K";
        case GGML_IQ2_XXS: return "iq2_xxs";
        case GGML_IQ2_XS:  return "iq2_xs";
        case GGML_IQ3_XXS: return "iq3_xxs";
        case GGML_IQ1_S:   return "iq1_s";
        case GGML_IQ4_NL:  return "iq4_nl";
        case GGML_IQ3_S:   return "iq3_s";
        case GGML_IQ2_S:   return "iq2_s";
        case GGML_IQ4_XS:  return "iq4_xs";
        case GGML_BF16:    return "bf16";
        default:           return "";
    }
}

uint32_t ggml_block_size(uint32_t t) {
    switch (t) {
        case GGML_F32:     return 1;
        case GGML_F16:     return 1;
        case GGML_BF16:    return 1;
        // Legacy 32-element blocks (we don't currently load these but report size correctly).
        case GGML_Q4_0:    return 32;
        case GGML_Q4_1:    return 32;
        case GGML_Q5_0:    return 32;
        case GGML_Q5_1:    return 32;
        case GGML_Q8_0:    return 32;
        case GGML_Q8_1:    return 32;
        // K-quants and i-quants -- 256 elements per super-block.
        case GGML_Q2_K:    return 256;
        case GGML_Q3_K:    return 256;
        case GGML_Q4_K:    return 256;
        case GGML_Q5_K:    return 256;
        case GGML_Q6_K:    return 256;
        case GGML_Q8_K:    return 256;
        case GGML_IQ2_XXS: return 256;
        case GGML_IQ2_XS:  return 256;
        case GGML_IQ3_XXS: return 256;
        case GGML_IQ1_S:   return 256;
        case GGML_IQ4_NL:  return 32;
        case GGML_IQ3_S:   return 256;
        case GGML_IQ2_S:   return 256;
        case GGML_IQ4_XS:  return 256;
        default:           return 0;
    }
}

// Bytes per quant block, lifted from llama.cpp's ggml-common.h block_* sizeof:
//   block_q4_K  = 144  (2 fp16 d/dmin + 12 6-bit packed scales + 128 4-bit weights)
//   block_q5_K  = 176  (q4_K layout + 32 bytes of high-bit q5)
//   block_q6_K  = 210  (128 bytes ql 6-bit lo + 64 bytes qh hi-bits + 16 bytes scales + 2 bytes d)
//   block_iq1_s = 50   (fp16 d + 32 bytes qs + 8 uint16 qh)
//   block_iq2_xxs = 66 (fp16 d + 8 uint16 qs)
uint32_t ggml_type_bytes(uint32_t t) {
    switch (t) {
        case GGML_F32:     return 4;
        case GGML_F16:     return 2;
        case GGML_BF16:    return 2;
        case GGML_Q4_0:    return 18;   // fp16 d + 16 bytes
        case GGML_Q4_1:    return 20;   // 2 fp16 + 16 bytes
        case GGML_Q5_0:    return 22;   // fp16 d + 4 bytes qh + 16 bytes
        case GGML_Q5_1:    return 24;   // 2 fp16 + 4 bytes qh + 16 bytes
        case GGML_Q8_0:    return 34;   // fp16 d + 32 bytes
        case GGML_Q8_1:    return 36;   // 2 fp16 + 32 bytes
        case GGML_Q2_K:    return 84;   // 16 bytes scales + 64 bytes qs + 2 fp16
        case GGML_Q3_K:    return 110;  // 32 bytes hmask + 64 bytes qs + 12 bytes scales + 2 fp16
        case GGML_Q4_K:    return 144;
        case GGML_Q5_K:    return 176;
        case GGML_Q6_K:    return 210;
        case GGML_Q8_K:    return 292;  // 4 fp + 256 bytes qs + 16 i16 bsums
        case GGML_IQ2_XXS: return 66;
        case GGML_IQ2_XS:  return 74;   // fp16 + 64 bytes qs + 4 uint16 scales
        case GGML_IQ3_XXS: return 98;   // fp16 + 64 bytes qs + 32 bytes signs
        case GGML_IQ1_S:   return 50;
        case GGML_IQ4_NL:  return 18;   // fp16 + 16 bytes
        case GGML_IQ3_S:   return 110;  // fp16 + 64 + 4 + 32 + 8
        case GGML_IQ2_S:   return 82;   // fp16 + 64 + 16 + bits
        case GGML_IQ4_XS:  return 136;  // fp16 + 16 (qh) + 16 + 128
        default:           return 0;
    }
}

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

[[noreturn]] static void die(const char* msg, const std::string& ctx = "") {
    if (ctx.empty()) {
        std::fprintf(stderr, "[gguf] %s\n", msg);
    } else {
        std::fprintf(stderr, "[gguf] %s: %s\n", msg, ctx.c_str());
    }
    std::abort();
}

static const uint8_t* mmap_metadata_ro(const std::string& path, size_t& sz_out,
                                       size_t& mapped_out, int& fd_out) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("open failed", path);
    struct stat st;
    if (::fstat(fd, &st) < 0) die("fstat failed", path);
    constexpr size_t kMetadataLimit = 256ULL << 20;
    mapped_out = std::min<size_t>(static_cast<size_t>(st.st_size),
                                  kMetadataLimit);
    void* p = ::mmap(nullptr, mapped_out, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("mmap failed", path);
    ::madvise(p, mapped_out, MADV_SEQUENTIAL);
    sz_out = st.st_size;
    fd_out = fd;
    return (const uint8_t*)p;
}

// Cursor over an mmap'd shard.  All reads are bounds-checked; aborts on overflow.
struct Cur {
    const uint8_t* base;
    size_t         len;
    size_t         off = 0;

    void need(size_t n, const char* what) const {
        if (off + n > len) {
            std::fprintf(stderr, "[gguf] EOF reading %s at off=%zu need=%zu len=%zu\n",
                         what, off, n, len);
            std::abort();
        }
    }
    template <typename T>
    T pod(const char* what) {
        need(sizeof(T), what);
        T v;
        std::memcpy(&v, base + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    std::string str(const char* what) {
        uint64_t n = pod<uint64_t>(what);
        need(n, what);
        std::string s((const char*)(base + off), (size_t)n);
        off += n;
        return s;
    }
    // Skip n bytes (used to step over array payloads we don't decode here).
    void skip(uint64_t n, const char* what) {
        need(n, what);
        off += n;
    }
};

// Element size for primitive value types.  Returns 0 for STRING/ARRAY (variable).
static uint32_t gguf_value_size(GgufValueType t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;
    }
}

// Decode a single primitive value into the relevant union slot of `kv`.
static void decode_primitive(GgufValueType t, Cur& c, GgufKV& kv) {
    switch (t) {
        case GGUF_U8:   kv.u64 = c.pod<uint8_t>("u8");                       break;
        case GGUF_I8:   kv.i64 = c.pod<int8_t>("i8");                        break;
        case GGUF_U16:  kv.u64 = c.pod<uint16_t>("u16");                     break;
        case GGUF_I16:  kv.i64 = c.pod<int16_t>("i16");                      break;
        case GGUF_U32:  kv.u64 = c.pod<uint32_t>("u32");                     break;
        case GGUF_I32:  kv.i64 = c.pod<int32_t>("i32");                      break;
        case GGUF_F32:  kv.f64 = c.pod<float>("f32");                        break;
        case GGUF_BOOL: kv.u64 = c.pod<uint8_t>("bool") ? 1 : 0;             break;
        case GGUF_U64:  kv.u64 = c.pod<uint64_t>("u64");                     break;
        case GGUF_I64:  kv.i64 = c.pod<int64_t>("i64");                      break;
        case GGUF_F64:  kv.f64 = c.pod<double>("f64");                       break;
        case GGUF_STRING: kv.str = c.str("str");                              break;
        default: die("decode_primitive on non-primitive type");
    }
}

// Read one KV entry.  Updates the cursor.
static GgufKV read_kv(Cur& c) {
    GgufKV kv;
    kv.key  = c.str("kv.key");
    kv.type = (GgufValueType)c.pod<uint32_t>("kv.type");
    if (kv.type == GGUF_ARRAY) {
        kv.arr_type  = (GgufValueType)c.pod<uint32_t>("arr.type");
        kv.arr_count = c.pod<uint64_t>("arr.count");
        if (kv.arr_type == GGUF_STRING) {
            // Strings are variable-length; record byte range and skip.
            const uint8_t* start = c.base + c.off;
            for (uint64_t i = 0; i < kv.arr_count; ++i) {
                uint64_t slen = c.pod<uint64_t>("arr.str.len");
                c.skip(slen, "arr.str.data");
            }
            kv.arr_data  = start;
            kv.arr_bytes = (uint64_t)((c.base + c.off) - start);
        } else {
            uint32_t es = gguf_value_size(kv.arr_type);
            if (es == 0) die("nested arrays unsupported");
            kv.arr_data  = c.base + c.off;
            kv.arr_bytes = es * kv.arr_count;
            c.skip(kv.arr_bytes, "arr.data");
        }
    } else {
        decode_primitive(kv.type, c, kv);
    }
    return kv;
}

// ----------------------------------------------------------------------------
// shard discovery
// ----------------------------------------------------------------------------

// "DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"  ->  ("DeepSeek-R1-UD-IQ1_S", 1, 3)
// Returns false if path doesn't match the convention (single-shard file).
struct ShardInfo {
    std::string base;       // path prefix without "-NNNNN-of-MMMMM.gguf"
    uint32_t    no = 0;     // 1-based
    uint32_t    count = 0;
};
static bool parse_shard_name(const std::string& path, ShardInfo& out) {
    // Match "...-(\d{5})-of-(\d{5})\.gguf$"
    static const std::regex re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return false;
    out.base  = m[1].str();
    out.no    = (uint32_t)std::stoul(m[2].str());
    out.count = (uint32_t)std::stoul(m[3].str());
    return true;
}

static std::string shard_path(const std::string& base, uint32_t no, uint32_t count) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "-%05u-of-%05u.gguf", no, count);
    return base + buf;
}

// ----------------------------------------------------------------------------
// per-shard parse
// ----------------------------------------------------------------------------

// Reads header + KV + tensor descriptors.  Sets shard.data_offset to the
// (alignment-padded) start of the tensor payload region.  Appends KVs (only
// from shard 0 to avoid duplicates) and tensors (from every shard) into
// the parent Gguf.
//
// Returns the alignment used for this shard's tensor payloads.
static void parse_shard(Gguf& g, GgufShard& sh, bool is_first) {
    Cur c{ sh.base, sh.mapped_size };

    // Header.
    uint8_t magic[4];
    c.need(4, "magic");
    std::memcpy(magic, sh.base, 4);
    c.off = 4;
    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        die("not a GGUF file", sh.path);
    }
    uint32_t version = c.pod<uint32_t>("version");
    if (version != 3) {
        std::fprintf(stderr, "[gguf] unsupported version %u (only v3 is tested)\n", version);
        std::abort();
    }
    if (is_first) g.version = version;

    uint64_t n_tensors = c.pod<uint64_t>("n_tensors");
    uint64_t n_kv      = c.pod<uint64_t>("n_kv");

    // KV section.
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        GgufKV kv = read_kv(c);
        if (kv.key == "general.alignment" && kv.type == GGUF_U32) {
            alignment = kv.u64;
        }
        if (is_first) {
            g.kv_idx[kv.key] = (uint32_t)g.kv.size();
            g.kv.push_back(std::move(kv));
        }
        // Non-first shards: discard KV (they should agree; we don't double-check).
    }
    sh.alignment = alignment;

    // Tensor descriptors.  Collect first; data_offset patches happen after
    // we know where the payload region starts.
    std::vector<GgufTensor> local;
    local.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = c.str("tensor.name");
        uint32_t nd = c.pod<uint32_t>("tensor.n_dims");
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) {
            t.dims[d] = c.pod<uint64_t>("tensor.dim");
        }
        t.type           = c.pod<uint32_t>("tensor.type");
        t.offset_in_data = c.pod<uint64_t>("tensor.offset");
        local.push_back(std::move(t));
    }

    // Pad cursor to alignment boundary -- this is where payloads begin.
    sh.data_offset = (c.off + (alignment - 1)) & ~(alignment - 1);

    // Resolve absolute pointers + byte sizes.  Also bounds-check.
    for (auto& t : local) {
        uint32_t bs = ggml_block_size(t.type);
        uint32_t bb = ggml_type_bytes(t.type);
        if (bs == 0 || bb == 0) {
            std::fprintf(stderr, "[gguf] %s: unsupported type %u (%s)\n",
                         t.name.c_str(), t.type, ggml_type_name(t.type));
            std::abort();
        }
        // Total elements.  We currently require ne[0] (the innermost dim) to
        // be a multiple of the block size.  This holds for every tensor in
        // R1-IQ1_S (innermost is hidden=7168, expert_ffn=2048, kv_lora=512,
        // q_lora=1536, vocab=129280 -- all divisible by 256).
        if (t.dims.empty()) die("zero-dim tensor", t.name);
        if ((t.dims[0] % bs) != 0) {
            std::fprintf(stderr, "[gguf] %s: ne[0]=%llu not multiple of block=%u\n",
                         t.name.c_str(),
                         (unsigned long long)t.dims[0], bs);
            std::abort();
        }
        uint64_t blocks_per_row = t.dims[0] / bs;
        uint64_t row_bytes      = blocks_per_row * bb;
        uint64_t rows = 1;
        for (size_t d = 1; d < t.dims.size(); ++d) rows *= t.dims[d];
        t.bytes = row_bytes * rows;

        uint64_t abs = sh.data_offset + t.offset_in_data;
        if (abs + t.bytes > sh.size) {
            std::fprintf(stderr, "[gguf] %s: payload out of bounds (abs=%llu bytes=%llu shard=%zu)\n",
                         t.name.c_str(),
                         (unsigned long long)abs,
                         (unsigned long long)t.bytes,
                         sh.size);
            std::abort();
        }
        t.data  = nullptr;
        t.shard = (uint32_t)(&sh - g.shards.data());

        g.tensor_idx[t.name] = (uint32_t)g.tensors.size();
        g.tensors.push_back(std::move(t));
    }
}

// ----------------------------------------------------------------------------
// public API
// ----------------------------------------------------------------------------

void Gguf::open(const std::string& path) {
    // Open shard 0 first to read split.no/count.
    {
        GgufShard sh;
        sh.path = path;
        sh.base = mmap_metadata_ro(path, sh.size, sh.mapped_size, sh.fd);
        shards.push_back(sh);
    }
    parse_shard(*this, shards[0], /*is_first=*/true);

    // Determine how many shards.
    shard_count = get_u32_or("split.count", 1);
    total_tensor_count = get_u32_or("split.tensors.count",
                                    (uint32_t)tensors.size());

    if (shard_count == 1) return;

    // Multi-shard: derive sibling paths and parse each.
    ShardInfo si;
    if (!parse_shard_name(path, si)) {
        die("multi-shard GGUF but path doesn't match -NNNNN-of-MMMMM.gguf", path);
    }
    if (si.count != shard_count) {
        std::fprintf(stderr, "[gguf] split.count=%u disagrees with filename count=%u\n",
                     shard_count, si.count);
        std::abort();
    }
    if (si.no != 1) {
        die("please pass shard 1 (-00001-of-NNNNN.gguf)", path);
    }
    for (uint32_t k = 2; k <= shard_count; ++k) {
        GgufShard sh;
        sh.path = shard_path(si.base, k, shard_count);
        sh.base = mmap_metadata_ro(sh.path, sh.size, sh.mapped_size, sh.fd);
        shards.push_back(sh);
        parse_shard(*this, shards.back(), /*is_first=*/false);
    }
}

void Gguf::close() {
    for (auto& sh : shards) {
        if (sh.base) {
            ::madvise((void*)sh.base, sh.mapped_size, MADV_DONTNEED);
            ::munmap((void*)sh.base, sh.mapped_size);
        }
        if (sh.fd >= 0) ::close(sh.fd);
        sh.base = nullptr;
        sh.mapped_size = 0;
        sh.fd = -1;
    }
    shards.clear();
    tensors.clear();
    tensor_idx.clear();
    kv.clear();
    kv_idx.clear();
}

const GgufTensor* Gguf::find(const std::string& name) const {
    auto it = tensor_idx.find(name);
    if (it == tensor_idx.end()) return nullptr;
    return &tensors[it->second];
}

uint64_t Gguf::total_tensor_bytes() const {
    uint64_t s = 0;
    for (const auto& t : tensors) s += t.bytes;
    return s;
}

// ---------------------------------------------------------------------------
// Lazy single-tensor lookup -- no persistent state, no payload pages touched.
//
// For each shard we mmap only the metadata window (capped at 256 MiB; real
// R1-IQ1_S is ~16 MiB), walk it once with MADV_SEQUENTIAL, then MADV_DONTNEED
// + munmap on the way out.  Payload bytes are never read by this function;
// the caller is expected to pread(shard_path, abs_offset, bytes) the slice
// they actually need.
//
// Why mmap a window instead of pread'ing a buffer: GGUF KV arrays
// (tokenizer.tokens for R1 is 129280 strings) are easiest to skip with a
// random-access cursor.  A 256 MiB virtual mapping that we MADV_DONTNEED at
// the end touches at most ~16 MiB of resident memory and zero payload pages.
// ---------------------------------------------------------------------------
static bool lookup_in_shard(const std::string& path,
                            const std::string& tname,
                            TensorRef& out,
                            uint32_t* split_count_out /* nullable */) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("lookup: open failed", path);
    struct stat st;
    if (::fstat(fd, &st) < 0) die("lookup: fstat failed", path);

    // Map only the metadata window.  256 MiB is 16x our measured worst case.
    const size_t META_MAX = (size_t)256 << 20;
    size_t map_len = (size_t)std::min<uint64_t>(st.st_size, (uint64_t)META_MAX);
    void* mp = ::mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (mp == MAP_FAILED) die("lookup: mmap failed", path);
    ::madvise(mp, map_len, MADV_SEQUENTIAL);

    auto cleanup = [&](bool found) {
        ::madvise(mp, map_len, MADV_DONTNEED);
        ::munmap(mp, map_len);
        ::close(fd);
        return found;
    };

    Cur c{ (const uint8_t*)mp, map_len };

    // Header.
    const uint8_t* p = (const uint8_t*)mp;
    if (p[0] != 'G' || p[1] != 'G' || p[2] != 'U' || p[3] != 'F') {
        die("lookup: not a GGUF file", path);
    }
    c.off = 4;
    uint32_t version = c.pod<uint32_t>("version");
    if (version != 3) die("lookup: unsupported GGUF version");
    uint64_t n_tensors = c.pod<uint64_t>("n_tensors");
    uint64_t n_kv      = c.pod<uint64_t>("n_kv");

    // KV walk -- we only need general.alignment + split.count.  All other
    // entries are decoded and immediately discarded so the cursor advances.
    uint64_t alignment   = 32;
    uint32_t split_count = 1;
    for (uint64_t i = 0; i < n_kv; ++i) {
        GgufKV kv = read_kv(c);
        if (kv.key == "general.alignment") alignment = kv.u64;
        if (kv.key == "split.count")       split_count = (uint32_t)kv.u64;
    }
    if (split_count_out) *split_count_out = split_count;

    // Tensor descriptors.  Capture the matching one (if any), keep going to
    // advance the cursor past every descriptor (data_offset depends on the
    // total descriptor block length).
    bool                  found = false;
    uint32_t              found_type = 0;
    std::vector<uint64_t> found_dims;
    uint64_t              found_offset_in_data = 0;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        std::string name = c.str("tensor.name");
        uint32_t    nd   = c.pod<uint32_t>("tensor.n_dims");
        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d) dims[d] = c.pod<uint64_t>("tensor.dim");
        uint32_t ttype = c.pod<uint32_t>("tensor.type");
        uint64_t toff  = c.pod<uint64_t>("tensor.offset");
        if (!found && name == tname) {
            found = true;
            found_type           = ttype;
            found_dims           = std::move(dims);
            found_offset_in_data = toff;
        }
    }
    if (!found) return cleanup(false);

    // data_offset (where payloads begin in this shard) = aligned end of
    // descriptor block.
    uint64_t data_offset = (c.off + (alignment - 1)) & ~(alignment - 1);

    // Compute payload size.
    uint32_t bs = ggml_block_size(found_type);
    uint32_t bb = ggml_type_bytes(found_type);
    if (bs == 0 || bb == 0) die("lookup: unsupported tensor type");
    if (found_dims.empty() || (found_dims[0] % bs) != 0)
        die("lookup: bad tensor shape");
    uint64_t blocks_per_row = found_dims[0] / bs;
    uint64_t row_bytes      = blocks_per_row * bb;
    uint64_t rows = 1;
    for (size_t d = 1; d < found_dims.size(); ++d) rows *= found_dims[d];
    uint64_t nbytes = row_bytes * rows;

    uint64_t abs = data_offset + found_offset_in_data;
    if (abs + nbytes > (uint64_t)st.st_size)
        die("lookup: payload out of bounds");

    out.name       = tname;
    out.type       = found_type;
    out.dims       = std::move(found_dims);
    out.bytes      = nbytes;
    out.shard_path = path;
    out.abs_offset = abs;
    return cleanup(true);
}

bool Gguf::lookup(const std::string& path, const std::string& tname, TensorRef& out) {
    uint32_t split_count = 1;
    if (lookup_in_shard(path, tname, out, &split_count)) return true;
    if (split_count <= 1) return false;

    // Multi-shard: derive sibling paths and search each in turn.
    ShardInfo si;
    if (!parse_shard_name(path, si)) return false;
    if (si.count != split_count) {
        std::fprintf(stderr, "[gguf] lookup: split.count=%u disagrees with filename=%u\n",
                     split_count, si.count);
        std::abort();
    }
    if (si.no != 1) die("lookup: please pass shard 1");
    for (uint32_t k = 2; k <= split_count; ++k) {
        std::string sp = shard_path(si.base, k, split_count);
        if (lookup_in_shard(sp, tname, out, nullptr)) return true;
    }
    return false;
}

// --- KV accessors ---------------------------------------------------------

const GgufKV* find_kv(const Gguf& g, const char* key) {
    auto it = g.kv_idx.find(key);
    if (it == g.kv_idx.end()) return nullptr;
    return &g.kv[it->second];
}

uint32_t Gguf::get_u32(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type == GGUF_U32 || k->type == GGUF_U16 || k->type == GGUF_U8 ||
        k->type == GGUF_U64 || k->type == GGUF_BOOL) return (uint32_t)k->u64;
    if (k->type == GGUF_I32 || k->type == GGUF_I16 || k->type == GGUF_I8 ||
        k->type == GGUF_I64) return (uint32_t)k->i64;
    die("KV not integer-convertible", key);
}

uint64_t Gguf::get_u64(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type == GGUF_U32 || k->type == GGUF_U16 || k->type == GGUF_U8 ||
        k->type == GGUF_U64 || k->type == GGUF_BOOL) return k->u64;
    if (k->type == GGUF_I32 || k->type == GGUF_I16 || k->type == GGUF_I8 ||
        k->type == GGUF_I64) return (uint64_t)k->i64;
    die("KV not integer-convertible", key);
}

int32_t Gguf::get_i32(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type == GGUF_I32 || k->type == GGUF_I16 || k->type == GGUF_I8 ||
        k->type == GGUF_I64) return (int32_t)k->i64;
    if (k->type == GGUF_U32 || k->type == GGUF_U16 || k->type == GGUF_U8 ||
        k->type == GGUF_U64 || k->type == GGUF_BOOL) return (int32_t)k->u64;
    die("KV not integer-convertible", key);
}

float Gguf::get_f32(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type == GGUF_F32 || k->type == GGUF_F64) return (float)k->f64;
    die("KV not float-convertible", key);
}

bool Gguf::get_bool(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type != GGUF_BOOL) die("KV not bool", key);
    return k->u64 != 0;
}

std::string Gguf::get_string(const char* key) const {
    auto* k = find_kv(*this, key);
    if (!k) die("missing KV", key);
    if (k->type != GGUF_STRING) die("KV not string", key);
    return k->str;
}

uint32_t Gguf::get_u32_or(const char* key, uint32_t def) const {
    auto* k = find_kv(*this, key);
    if (!k) return def;
    if (k->type == GGUF_U32 || k->type == GGUF_U16 || k->type == GGUF_U8 ||
        k->type == GGUF_U64 || k->type == GGUF_BOOL) return (uint32_t)k->u64;
    if (k->type == GGUF_I32 || k->type == GGUF_I16 || k->type == GGUF_I8 ||
        k->type == GGUF_I64) return (uint32_t)k->i64;
    return def;
}

float Gguf::get_f32_or(const char* key, float def) const {
    auto* k = find_kv(*this, key);
    if (!k) return def;
    if (k->type == GGUF_F32 || k->type == GGUF_F64) return (float)k->f64;
    return def;
}

std::string Gguf::get_string_or(const char* key, const char* def) const {
    auto* k = find_kv(*this, key);
    if (!k || k->type != GGUF_STRING) return def;
    return k->str;
}

} // namespace blade
