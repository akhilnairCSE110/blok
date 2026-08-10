#include "model.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace blade {

[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "blade: %s: %s\n", msg, std::strerror(errno));
    std::abort();
}

// Tiny JSON-ish reader: looks up "key": <number>. Sufficient for header.json
// emitted by our own converter; we do NOT need a general JSON parser.
static double json_num(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) { std::fprintf(stderr, "missing %s\n", key); std::abort(); }
    p = s.find(':', p) + 1;
    return std::strtod(s.c_str() + p, nullptr);
}

// Same, but returns `def` if the key is absent (for backwards-compat flags).
static double json_num_default(const std::string& s, const char* key, double def) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return def;
    p = s.find(':', p) + 1;
    return std::strtod(s.c_str() + p, nullptr);
}

static std::string slurp(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die(path.c_str());
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::string s(n, '\0'); if (std::fread(s.data(), 1, n, f) != (size_t)n) die("fread");
    std::fclose(f); return s;
}

static const uint8_t* map_ro(const std::string& path, size_t& len_out, int* fd_out = nullptr) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die(path.c_str());
    struct stat st; if (::fstat(fd, &st) < 0) die("fstat");
    void* p = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("mmap");
    len_out = st.st_size;
    if (fd_out) *fd_out = fd; else ::close(fd);
    return (const uint8_t*)p;
}

// Cursor: walks hot.bin in the canonical tensor order set by the converter.
// 16-byte aligned. Returns typed pointers.
struct Cursor {
    const uint8_t* base; size_t off = 0; size_t len;
    template<class T> const T* take(size_t bytes) {
        off = (off + 15) & ~size_t{15};
        if (off + bytes > len) { std::fprintf(stderr, "hot.bin overflow\n"); std::abort(); }
        const T* p = reinterpret_cast<const T*>(base + off);
        off += bytes;
        return p;
    }
};

void Model::open(const std::string& dir) {
    // 1) header
    std::string hj = slurp(dir + "/header.json");
    cfg.n_layers          = (uint32_t)json_num(hj, "n_layers");
    cfg.n_dense_layers    = (uint32_t)json_num(hj, "n_dense_layers");
    cfg.hidden            = (uint32_t)json_num(hj, "hidden");
    cfg.vocab             = (uint32_t)json_num(hj, "vocab");
    cfg.n_heads           = (uint32_t)json_num(hj, "n_heads");
    cfg.head_dim_qk_nope  = (uint32_t)json_num(hj, "head_dim_qk_nope");
    cfg.head_dim_qk_rope  = (uint32_t)json_num(hj, "head_dim_qk_rope");
    cfg.head_dim_v        = (uint32_t)json_num(hj, "head_dim_v");
    cfg.kv_lora_rank      = (uint32_t)json_num(hj, "kv_lora_rank");
    cfg.q_lora_rank       = (uint32_t)json_num(hj, "q_lora_rank");
    cfg.ffn_dense         = (uint32_t)json_num(hj, "ffn_dense");
    cfg.n_experts         = (uint32_t)json_num(hj, "n_experts");
    cfg.n_experts_active  = (uint32_t)json_num(hj, "n_experts_active");
    cfg.n_shared_experts  = (uint32_t)json_num(hj, "n_shared_experts");
    cfg.expert_ffn        = (uint32_t)json_num(hj, "expert_ffn");
    cfg.rope_theta        = (float)   json_num(hj, "rope_theta");
    cfg.rms_eps           = (float)   json_num(hj, "rms_eps");
    cfg.max_seq           = (uint32_t)json_num(hj, "max_seq");
    cfg.tied_embed        = (uint32_t)json_num_default(hj, "tied_embed",      1.0);
    cfg.has_router_bias   = (uint32_t)json_num_default(hj, "has_router_bias", 1.0);
    cfg.weight_dtype      = 0;
    cfg.yarn_mscale       = 1.0f;
    esz = expert_sizes(cfg);

    // 2) hot.bin
    size_t hot_len; int hot_fd;
    const uint8_t* hot = map_ro(dir + "/hot.bin", hot_len, &hot_fd);
    // Lock into RAM (best-effort; will partially fail past RLIMIT_MEMLOCK).
    ::mlock(hot, hot_len);
    Cursor C{hot, 0, hot_len};

    const auto& c = cfg;
    const size_t H = c.hidden;
    const size_t qk = c.head_dim_qk_nope + c.head_dim_qk_rope;
    const size_t kv_a_out = c.kv_lora_rank + c.head_dim_qk_rope;

    w_embed    = C.take<Fp8Block>(fp8_bytes((size_t)c.vocab * H));
    final_norm = C.take<uint16_t>(H * 2);
    if (c.tied_embed) {
        lm_head = w_embed;
    } else {
        lm_head = C.take<Fp8Block>(fp8_bytes((size_t)c.vocab * H));
    }

    layers.resize(c.n_layers);
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        auto& h = layers[L];
        h.attn_norm = C.take<uint16_t>(H * 2);
        h.ffn_norm  = C.take<uint16_t>(H * 2);
        if (c.q_lora_rank > 0) {
            h.w_q_a    = C.take<Fp8Block>(fp8_bytes(c.q_lora_rank * H));
            h.q_a_norm = C.take<uint16_t>(c.q_lora_rank * 2);
            h.w_q_b    = C.take<Fp8Block>(fp8_bytes((size_t)c.n_heads * qk * c.q_lora_rank));
            h.w_q      = nullptr;
        } else {
            h.w_q_a = nullptr; h.q_a_norm = nullptr; h.w_q_b = nullptr;
            h.w_q   = C.take<Fp8Block>(fp8_bytes((size_t)c.n_heads * qk * H));
        }
        h.w_kv_a    = C.take<Fp8Block>(fp8_bytes(kv_a_out * H));
        h.kv_a_norm = C.take<uint16_t>(c.kv_lora_rank * 2);
        h.w_uk      = C.take<Fp8Block>(fp8_bytes((size_t)c.n_heads * c.kv_lora_rank * c.head_dim_qk_nope));
        h.w_uv      = C.take<Fp8Block>(fp8_bytes((size_t)c.n_heads * c.head_dim_v   * c.kv_lora_rank));
        h.w_o       = C.take<Fp8Block>(fp8_bytes(H * (size_t)c.n_heads * c.head_dim_v));

        if (L < c.n_dense_layers) {
            h.w_gate_dense = C.take<Fp8Block>(fp8_bytes((size_t)c.ffn_dense * H));
            h.w_up_dense   = C.take<Fp8Block>(fp8_bytes((size_t)c.ffn_dense * H));
            h.w_down_dense = C.take<Fp8Block>(fp8_bytes(H * (size_t)c.ffn_dense));
            h.w_gate_shared = h.w_up_shared = h.w_down_shared = nullptr;
            h.w_router = nullptr; h.router_bias = nullptr;
        } else {
            h.w_gate_dense = h.w_up_dense = h.w_down_dense = nullptr;
            h.w_gate_shared = C.take<Fp8Block>(fp8_bytes((size_t)c.expert_ffn * c.n_shared_experts * H));
            h.w_up_shared   = C.take<Fp8Block>(fp8_bytes((size_t)c.expert_ffn * c.n_shared_experts * H));
            h.w_down_shared = C.take<Fp8Block>(fp8_bytes(H * (size_t)c.expert_ffn * c.n_shared_experts));
            h.w_router      = C.take<Fp8Block>(fp8_bytes((size_t)c.n_experts * H));
            if (c.has_router_bias) {
                h.router_bias = C.take<float>(c.n_experts * 4);
            } else {
                h.router_bias = nullptr;
            }
        }
    }
    if (C.off > C.len) { std::fprintf(stderr, "hot.bin underrun\n"); std::abort(); }

    // 3) experts.bin (mmap, demand-paged)
    size_t enexp;
    experts_base = map_ro(dir + "/experts.bin", enexp, &experts_fd);
    experts_len  = enexp;
    // Tell the kernel: random access, do NOT prefetch entire file.
    ::madvise((void*)experts_base, experts_len, MADV_RANDOM);

    // 4) experts.idx
    size_t ilen;
    expert_idx = (const ExpertIdx*)map_ro(dir + "/experts.idx", ilen);
    size_t expected = (size_t)(c.n_layers - c.n_dense_layers) * c.n_experts * sizeof(ExpertIdx);
    if (ilen != expected) { std::fprintf(stderr, "experts.idx size %zu != %zu\n", ilen, expected); std::abort(); }
}

// ============================================================================
// Raw-HF (bf16) loader.  Mmap each .safetensors shard, parse its JSON header,
// build a tensor name -> (base, byte_offset, nbytes) map, then resolve every
// tensor we need into a LayerHot pointer.  No conversion to disk.
// ============================================================================
namespace {
struct TensorView { const uint8_t* base; size_t nbytes; };
using TensorMap = std::unordered_map<std::string, TensorView>;

// Skip whitespace.
inline void ws(const char*& p, const char* end) {
    while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
}
// Parse a JSON string into the input iterator. Returns std::string.
std::string parse_str(const char*& p, const char* end) {
    if (*p != '"') { std::fprintf(stderr, "expect \" got %c\n", *p); std::abort(); }
    ++p;
    std::string s;
    while (p < end && *p != '"') {
        if (*p == '\\' && p+1 < end) { s += p[1]; p += 2; }
        else                          { s += *p++; }
    }
    ++p; // closing quote
    return s;
}
// Skip a JSON value at p (object, array, string, number, bool, null).
void skip_value(const char*& p, const char* end) {
    ws(p, end);
    if (*p == '{' || *p == '[') {
        char open = *p, close = (open == '{') ? '}' : ']';
        int depth = 1; ++p;
        while (p < end && depth) {
            if (*p == '"') { parse_str(p, end); continue; }
            if (*p == open)  ++depth;
            if (*p == close) --depth;
            ++p;
        }
    } else if (*p == '"') { parse_str(p, end); }
    else { while (p < end && *p != ',' && *p != '}' && *p != ']') ++p; }
}

// Mmap one safetensors file, parse its header, append entries to `out`.
void parse_safetensors(const std::string& path, std::vector<const uint8_t*>& bases,
                       std::vector<size_t>& sizes, TensorMap& out) {
    size_t flen; const uint8_t* base = map_ro(path, flen);
    bases.push_back(base); sizes.push_back(flen);
    if (flen < 8) { std::fprintf(stderr, "%s: too small\n", path.c_str()); std::abort(); }
    uint64_t hlen = *(const uint64_t*)base;
    if (hlen + 8 > flen) { std::fprintf(stderr, "%s: bad header\n", path.c_str()); std::abort(); }
    const char* p   = (const char*)(base + 8);
    const char* end = p + hlen;
    const uint8_t* tensor_base = base + 8 + hlen;

    ws(p, end);
    if (*p != '{') { std::fprintf(stderr, "%s: expect {\n", path.c_str()); std::abort(); }
    ++p;
    while (true) {
        ws(p, end);
        if (*p == '}') break;
        std::string name = parse_str(p, end);
        ws(p, end); if (*p != ':') std::abort(); ++p; ws(p, end);
        if (name == "__metadata__") { skip_value(p, end); }
        else {
            // Walk the inner object looking for "data_offsets":[a,b]
            if (*p != '{') std::abort();
            ++p;
            uint64_t off_a = 0, off_b = 0;
            while (true) {
                ws(p, end);
                if (*p == '}') { ++p; break; }
                std::string k = parse_str(p, end);
                ws(p, end); if (*p != ':') std::abort(); ++p; ws(p, end);
                if (k == "data_offsets") {
                    if (*p != '[') std::abort(); ++p; ws(p, end);
                    off_a = std::strtoull(p, (char**)&p, 10);
                    ws(p, end); if (*p == ',') ++p; ws(p, end);
                    off_b = std::strtoull(p, (char**)&p, 10);
                    ws(p, end); if (*p == ']') ++p;
                } else {
                    skip_value(p, end);
                }
                ws(p, end); if (*p == ',') ++p;
            }
            out[name] = { tensor_base + off_a, (size_t)(off_b - off_a) };
        }
        ws(p, end); if (*p == ',') ++p;
    }
}
} // anonymous namespace

void Model::open_hf(const std::string& dir) {
    // 1) config.json -> Config
    std::string cj = slurp(dir + "/config.json");
    auto json_uint = [&](const char* k){ return (uint32_t)json_num_default(cj, k, 0.0); };
    auto json_uint_d = [&](const char* k, uint32_t d){ return (uint32_t)json_num_default(cj, k, (double)d); };
    auto json_flt = [&](const char* k, float d){ return (float)json_num_default(cj, k, (double)d); };

    cfg.n_layers          = (uint32_t)json_num(cj, "num_hidden_layers");
    cfg.n_dense_layers    = json_uint_d("first_k_dense_replace", 1);
    cfg.hidden            = (uint32_t)json_num(cj, "hidden_size");
    cfg.vocab             = (uint32_t)json_num(cj, "vocab_size");
    cfg.n_heads           = (uint32_t)json_num(cj, "num_attention_heads");
    cfg.head_dim_qk_nope  = (uint32_t)json_num(cj, "qk_nope_head_dim");
    cfg.head_dim_qk_rope  = (uint32_t)json_num(cj, "qk_rope_head_dim");
    cfg.head_dim_v        = (uint32_t)json_num(cj, "v_head_dim");
    cfg.kv_lora_rank      = (uint32_t)json_num(cj, "kv_lora_rank");
    cfg.q_lora_rank       = json_uint("q_lora_rank");      // 0 if null/missing
    cfg.ffn_dense         = (uint32_t)json_num(cj, "intermediate_size");
    cfg.n_experts         = (uint32_t)json_num(cj, "n_routed_experts");
    cfg.n_experts_active  = (uint32_t)json_num(cj, "num_experts_per_tok");
    cfg.n_shared_experts  = json_uint_d("n_shared_experts", 1);
    cfg.expert_ffn        = (uint32_t)json_num(cj, "moe_intermediate_size");
    cfg.rope_theta        = json_flt("rope_theta", 10000.0f);
    cfg.rms_eps           = json_flt("rms_norm_eps", 1e-6f);
    // Cap context at 4k for first run; KV cache scales linearly.
    cfg.max_seq           = 4096;
    // tied_embed: HF flag tie_word_embeddings (default true if missing).
    {
        size_t pp = cj.find("\"tie_word_embeddings\"");
        bool tied = true;
        if (pp != std::string::npos) {
            size_t cc = cj.find(':', pp) + 1;
            while (cc < cj.size() && (cj[cc]==' '||cj[cc]=='\t')) ++cc;
            tied = (cj.compare(cc, 4, "true") == 0);
        }
        cfg.tied_embed = tied ? 1 : 0;
    }
    cfg.has_router_bias = 0;   // detected below
    cfg.weight_dtype    = 1;   // bf16
    // YaRN attention-scale fixup.  Look for "rope_scaling": {... "type":"yarn",
    // "factor": F, "mscale_all_dim": M, ...}.  When present, the effective
    // softmax_scale = (1/sqrt(d)) * yarn_mscale where
    //   yarn_mscale = ( 1 + 0.1*log(factor)*mscale_all_dim )^2 .
    cfg.yarn_mscale = 1.0f;
    {
        size_t pp = cj.find("\"rope_scaling\"");
        if (pp != std::string::npos) {
            // Cheap scoped scan: find next "factor" and "mscale_all_dim" keys.
            auto local_num = [&](const char* k, double def) -> double {
                std::string pat = std::string("\"") + k + "\"";
                size_t q = cj.find(pat, pp);
                if (q == std::string::npos) return def;
                q = cj.find(':', q) + 1;
                return std::strtod(cj.c_str() + q, nullptr);
            };
            double factor = local_num("factor", 1.0);
            double mscale = local_num("mscale_all_dim", local_num("mscale", 1.0));
            if (factor > 1.0) {
                double m = 1.0 + 0.1 * std::log(factor) * mscale;
                cfg.yarn_mscale = (float)(m * m);
            }
        }
    }
    esz = expert_sizes(cfg);   // unused on this path but harmless

    // 2) Discover safetensors shards via the index file (if present).
    std::vector<std::string> shards;
    {
        std::string idx_path = dir + "/model.safetensors.index.json";
        FILE* f = std::fopen(idx_path.c_str(), "rb");
        if (f) {
            std::fclose(f);
            std::string ij = slurp(idx_path);
            // Pull every distinct value of the form "...safetensors". Each
            // appears as: "tensor.name": "model-XXXXX-of-YYYYY.safetensors"
            std::set<std::string> seen;
            const char* needle = ".safetensors";
            const size_t nlen  = std::strlen(needle);
            size_t p = 0;
            while ((p = ij.find(needle, p)) != std::string::npos) {
                // Step past needle then require closing quote.
                size_t q_end = p + nlen;
                if (q_end >= ij.size() || ij[q_end] != '"') { p = q_end; continue; }
                // Walk back to the opening quote.
                size_t q_start = ij.rfind('"', p);
                if (q_start == std::string::npos) break;
                seen.insert(ij.substr(q_start + 1, q_end - q_start - 1));
                p = q_end + 1;
            }
            for (auto& s : seen) shards.push_back(dir + "/" + s);
        } else {
            // Single-file checkpoint.
            shards.push_back(dir + "/model.safetensors");
        }
        if (shards.empty()) { std::fprintf(stderr, "no safetensors shards\n"); std::abort(); }
    }

    // 3) Parse each shard's header into one big tensor map.
    std::vector<const uint8_t*> shard_bases;
    std::vector<size_t>         shard_sizes;
    TensorMap tm;
    for (auto& s : shards) parse_safetensors(s, shard_bases, shard_sizes, tm);
    // Tell the kernel: we will read these shards in-bulk soon, but no readahead
    // beyond what we ask for (the access pattern jumps between many tensors).
    // RANDOM disables aggressive readahead; MoE expert pages are then driven
    // exclusively by the Streamer's per-expert WILLNEED hints.
    for (size_t i = 0; i < shard_bases.size(); ++i)
        ::madvise((void*)shard_bases[i], shard_sizes[i], MADV_RANDOM);

    auto need = [&](const std::string& n) -> TensorView {
        auto it = tm.find(n);
        if (it == tm.end()) {
            std::fprintf(stderr, "missing tensor: %s\n", n.c_str()); std::abort();
        }
        return it->second;
    };
    auto maybe = [&](const std::string& n) -> const TensorView* {
        auto it = tm.find(n);
        return it == tm.end() ? nullptr : &it->second;
    };

    // Norm-gain conversion arena.  rms_norm_f16 reads gains as half (fp16),
    // but HF safetensors store them as bf16.  Convert once at load via the
    // hardware bf16->fp32->fp16 path (Apple Clang's __fp16 = IEEE half with
    // RTNE rounding).  Total: O(layers * hidden * a-few) -- ~hundreds of KB.
    auto bf16_to_f16 = [](uint16_t b) -> uint16_t {
        uint32_t u = ((uint32_t)b) << 16;
        float f; std::memcpy(&f, &u, 4);
        __fp16 h = (__fp16)f;
        uint16_t out; std::memcpy(&out, &h, 2);
        return out;
    };

    // Count norms we need: per layer (attn, ffn, kv_a) + (q_a if Q-LoRA) + final.
    size_t H_sz = (size_t)cfg.hidden;
    size_t Lk_sz = (size_t)cfg.kv_lora_rank;
    size_t Hi_sz = (size_t)cfg.q_lora_rank;
    size_t norm_words = (size_t)cfg.n_layers * (2 * H_sz + Lk_sz + (Hi_sz > 0 ? Hi_sz : 0))
                      + H_sz;  // final
    static uint16_t* norm_arena = nullptr;
    norm_arena = (uint16_t*)std::aligned_alloc(64, ((norm_words * 2 + 63) & ~size_t{63}));
    if (!norm_arena) { std::fprintf(stderr, "norm arena alloc failed\n"); std::abort(); }
    size_t norm_cur = 0;
    auto take_norm = [&](size_t n) -> uint16_t* {
        uint16_t* p = norm_arena + norm_cur; norm_cur += n; return p;
    };
    auto convert_norm = [&](const std::string& name, size_t n) -> const uint16_t* {
        const uint16_t* src = (const uint16_t*)need(name).base;
        uint16_t* dst = take_norm(n);
        for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f16(src[i]);
        return dst;
    };

    // 4) Detect router bias (V3) from the first MoE layer.
    {
        std::string k = "model.layers." + std::to_string(cfg.n_dense_layers) +
                        ".mlp.gate.e_score_correction_bias";
        cfg.has_router_bias = maybe(k) ? 1 : 0;
    }

    // 5) Resolve top-level tensors.
    w_embed    = (const Fp8Block*)need("model.embed_tokens.weight").base;
    final_norm = convert_norm("model.norm.weight", H_sz);
    if (cfg.tied_embed) {
        lm_head = w_embed;
    } else {
        lm_head = (const Fp8Block*)need("lm_head.weight").base;
    }

    // 6) Per-layer hot tensors. For W_uk/W_uv we need to pre-split kv_b. We
    //    allocate a single arena and place each layer's split inside it.
    const uint32_t HE = cfg.n_heads, Lk = cfg.kv_lora_rank;
    const uint32_t Dn = cfg.head_dim_qk_nope, Dv = cfg.head_dim_v;
    size_t per_layer_uk = (size_t)HE * Lk * Dn * sizeof(uint16_t);
    size_t per_layer_uv = (size_t)HE * Dv * Lk * sizeof(uint16_t);
    size_t arena_bytes  = (size_t)cfg.n_layers * (per_layer_uk + per_layer_uv);
    static uint8_t* uk_uv_arena = nullptr;
    uk_uv_arena = (uint8_t*)std::aligned_alloc(64, (arena_bytes + 63) & ~size_t{63});
    if (!uk_uv_arena) { std::fprintf(stderr, "arena alloc failed\n"); std::abort(); }

    layers.resize(cfg.n_layers);
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        auto& h = layers[L];
        std::string P = "model.layers." + std::to_string(L) + ".";
        h.attn_norm = convert_norm(P + "input_layernorm.weight",          H_sz);
        h.ffn_norm  = convert_norm(P + "post_attention_layernorm.weight", H_sz);

        if (cfg.q_lora_rank > 0) {
            h.w_q_a    = (const Fp8Block*)need(P + "self_attn.q_a_proj.weight").base;
            h.q_a_norm = convert_norm(P + "self_attn.q_a_layernorm.weight", Hi_sz);
            h.w_q_b    = (const Fp8Block*)need(P + "self_attn.q_b_proj.weight").base;
            h.w_q      = nullptr;
        } else {
            h.w_q_a = nullptr; h.q_a_norm = nullptr; h.w_q_b = nullptr;
            h.w_q   = (const Fp8Block*)need(P + "self_attn.q_proj.weight").base;
        }

        h.w_kv_a    = (const Fp8Block*)need(P + "self_attn.kv_a_proj_with_mqa.weight").base;
        h.kv_a_norm = convert_norm(P + "self_attn.kv_a_layernorm.weight", Lk_sz);

        // kv_b in HF is bf16 [HE*(Dn+Dv), Lk] row-major. Reshape into
        //   uk[HE, Lk, Dn]  (transposed Dn<->Lk axes) and uv[HE, Dv, Lk].
        const uint16_t* kv_b = (const uint16_t*)need(P + "self_attn.kv_b_proj.weight").base;
        uint8_t* arena_layer = uk_uv_arena + (size_t)L * (per_layer_uk + per_layer_uv);
        uint16_t* uk_dst = (uint16_t*)arena_layer;
        uint16_t* uv_dst = (uint16_t*)(arena_layer + per_layer_uk);
        for (uint32_t hi = 0; hi < HE; ++hi) {
            const uint16_t* head_src = kv_b + (size_t)hi * (Dn + Dv) * Lk;
            // uk[hi, l, n] = src[hi, n, l]   (n in [0,Dn), l in [0,Lk))
            for (uint32_t n = 0; n < Dn; ++n) {
                for (uint32_t l = 0; l < Lk; ++l) {
                    uk_dst[((size_t)hi * Lk + l) * Dn + n] = head_src[(size_t)n * Lk + l];
                }
            }
            // uv[hi, v, l] = src[hi, Dn+v, l]  (v in [0,Dv), l in [0,Lk))
            for (uint32_t v = 0; v < Dv; ++v) {
                for (uint32_t l = 0; l < Lk; ++l) {
                    uv_dst[((size_t)hi * Dv + v) * Lk + l] = head_src[(size_t)(Dn + v) * Lk + l];
                }
            }
        }
        h.w_uk = (const Fp8Block*)uk_dst;
        h.w_uv = (const Fp8Block*)uv_dst;

        h.w_o = (const Fp8Block*)need(P + "self_attn.o_proj.weight").base;

        if (L < cfg.n_dense_layers) {
            h.w_gate_dense = (const Fp8Block*)need(P + "mlp.gate_proj.weight").base;
            h.w_up_dense   = (const Fp8Block*)need(P + "mlp.up_proj.weight").base;
            h.w_down_dense = (const Fp8Block*)need(P + "mlp.down_proj.weight").base;
            h.w_gate_shared = h.w_up_shared = h.w_down_shared = nullptr;
            h.w_router = nullptr; h.router_bias = nullptr;
        } else {
            h.w_gate_dense = h.w_up_dense = h.w_down_dense = nullptr;
            h.w_gate_shared = (const Fp8Block*)need(P + "mlp.shared_experts.gate_proj.weight").base;
            h.w_up_shared   = (const Fp8Block*)need(P + "mlp.shared_experts.up_proj.weight").base;
            h.w_down_shared = (const Fp8Block*)need(P + "mlp.shared_experts.down_proj.weight").base;
            h.w_router      = (const Fp8Block*)need(P + "mlp.gate.weight").base;
            if (cfg.has_router_bias) {
                h.router_bias = (const float*)need(P + "mlp.gate.e_score_correction_bias").base;
            } else {
                h.router_bias = nullptr;
            }
        }
    }

    // 7) Routed experts as raw bf16 pointers.
    size_t n_moe = cfg.n_layers - cfg.n_dense_layers;
    raw_experts.resize(n_moe * cfg.n_experts);
    for (uint32_t ml = 0; ml < n_moe; ++ml) {
        uint32_t L = cfg.n_dense_layers + ml;
        std::string base = "model.layers." + std::to_string(L) + ".mlp.experts.";
        for (uint32_t e = 0; e < cfg.n_experts; ++e) {
            std::string ep = base + std::to_string(e) + ".";
            auto& re = raw_experts[(size_t)ml * cfg.n_experts + e];
            re.gate = need(ep + "gate_proj.weight").base;
            re.up   = need(ep + "up_proj.weight").base;
            re.down = need(ep + "down_proj.weight").base;
        }
    }

    // The FP8 path's experts.bin/idx fields are unused on this path.
    experts_base = nullptr; experts_len = 0; expert_idx = nullptr; experts_fd = -1;
}

} // namespace blade

namespace blade {
Model::~Model() {
    if (norm_arena)  { std::free(norm_arena);  norm_arena  = nullptr; }
    if (uk_uv_arena) { std::free(uk_uv_arena); uk_uv_arena = nullptr; }
    // Mmap'd shards and experts.bin are intentionally NOT munmapped here:
    // their lifetime equals the process and the OS reaps them at exit.
}

// Page-fault every dense weight into the kernel page cache, in tensor order,
// one byte per 16 KiB page.  Sequential faults let the kernel issue bulk
// readahead I/O at near-NVMe peak; without this the first decode step
// fault-storms one cache line at a time and serialises against compute.
void Model::warm_dense() {
    if (cfg.weight_dtype != 1) return;       // FP8 path's hot.bin is mlock'd
    auto walk = [](const void* p, size_t bytes) {
        if (!p || bytes == 0) return uint64_t{0};
        const volatile uint8_t* base = (const volatile uint8_t*)p;
        uint64_t s = 0;
        for (size_t off = 0; off < bytes; off += 16384) s ^= base[off];
        return s;
    };
    auto wb = [&](size_t n) { return weight_bytes(n, cfg.weight_dtype); };
    const size_t H  = cfg.hidden;
    const size_t Lk = cfg.kv_lora_rank;
    const size_t qk = (size_t)cfg.head_dim_qk_nope + cfg.head_dim_qk_rope;
    volatile uint64_t sink = 0;
    sink ^= walk(w_embed, wb((size_t)cfg.vocab * H));
    if (!cfg.tied_embed) sink ^= walk(lm_head, wb((size_t)cfg.vocab * H));
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        const auto& h = layers[L];
        if (cfg.q_lora_rank > 0) {
            sink ^= walk(h.w_q_a, wb((size_t)cfg.q_lora_rank * H));
            sink ^= walk(h.w_q_b, wb((size_t)cfg.n_heads * qk * cfg.q_lora_rank));
        } else {
            sink ^= walk(h.w_q,   wb((size_t)cfg.n_heads * qk * H));
        }
        sink ^= walk(h.w_kv_a, wb((size_t)(Lk + cfg.head_dim_qk_rope) * H));
        sink ^= walk(h.w_o,    wb(H * (size_t)cfg.n_heads * cfg.head_dim_v));
        if (L < cfg.n_dense_layers) {
            sink ^= walk(h.w_gate_dense, wb((size_t)cfg.ffn_dense * H));
            sink ^= walk(h.w_up_dense,   wb((size_t)cfg.ffn_dense * H));
            sink ^= walk(h.w_down_dense, wb(H * (size_t)cfg.ffn_dense));
        } else {
            const size_t Fs = (size_t)cfg.expert_ffn * cfg.n_shared_experts;
            sink ^= walk(h.w_gate_shared, wb(Fs * H));
            sink ^= walk(h.w_up_shared,   wb(Fs * H));
            sink ^= walk(h.w_down_shared, wb(H * Fs));
            sink ^= walk(h.w_router,      wb((size_t)cfg.n_experts * H));
        }
    }
    (void)sink;
}
} // namespace blade
