#include "tokenizer.hpp"
#include "gguf.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <array>
#include <algorithm>

namespace blade {

[[noreturn]] static void die(const char* m) { std::fprintf(stderr, "tok: %s\n", m); std::abort(); }

// ---------------------------------------------------------------------------
// GPT-2 byte<->unicode bijection used by Llama/DeepSeek byte-level BPE vocabs.
// 188 "visible" bytes map to themselves; the remaining 68 bytes map to
// codepoints 256..323. Piece strings in the vocab are the UTF-8 of these
// codepoints. We build two lookup tables once at load time:
//   byte_to_utf8_[b] = UTF-8 string of the codepoint b maps to
//   utf8_to_byte_   = reverse map (multi-byte UTF-8 string -> original byte)
// ---------------------------------------------------------------------------
static bool is_printable_gpt2(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}
static std::string utf8_from_cp(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}
static const std::array<std::string, 256>& byte_to_utf8_table() {
    static std::array<std::string, 256> t;
    static bool init = false;
    if (!init) {
        uint32_t next_cp = 256;
        for (int b = 0; b < 256; ++b) {
            if (is_printable_gpt2(b)) t[b] = utf8_from_cp((uint32_t)b);
            else                       t[b] = utf8_from_cp(next_cp++);
        }
        init = true;
    }
    return t;
}
static const std::unordered_map<std::string, uint8_t>& utf8_to_byte_table() {
    static std::unordered_map<std::string, uint8_t> m;
    static bool init = false;
    if (!init) {
        const auto& fwd = byte_to_utf8_table();
        for (int b = 0; b < 256; ++b) m.emplace(fwd[b], (uint8_t)b);
        init = true;
    }
    return m;
}
static std::string remap_bytes_to_pieces(const std::string& raw) {
    const auto& fwd = byte_to_utf8_table();
    std::string out; out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += fwd[c];
    return out;
}
static std::string remap_piece_to_bytes(const std::string& piece) {
    // Walk UTF-8 codepoint by codepoint; map each to its original byte.
    const auto& rev = utf8_to_byte_table();
    std::string out; out.reserve(piece.size());
    size_t i = 0;
    while (i < piece.size()) {
        unsigned char c = (unsigned char)piece[i];
        size_t L = 1;
        if      ((c & 0x80) == 0x00) L = 1;
        else if ((c & 0xE0) == 0xC0) L = 2;
        else if ((c & 0xF0) == 0xE0) L = 3;
        else if ((c & 0xF8) == 0xF0) L = 4;
        if (i + L > piece.size()) { out.append(piece, i, std::string::npos); break; }
        auto it = rev.find(piece.substr(i, L));
        if (it != rev.end()) out += (char)it->second;
        else                 out.append(piece, i, L);   // pass through (special tokens etc)
        i += L;
    }
    return out;
}

void Tokenizer::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die(path.c_str());
    uint32_t n; if (std::fread(&n, 4, 1, f) != 1) die("hdr");
    pieces_.resize(n);
    ranks_.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t rank, len;
        if (std::fread(&rank, 4, 1, f) != 1) die("rank");
        if (std::fread(&len,  4, 1, f) != 1) die("len");
        std::string piece(len, '\0');
        if (std::fread(piece.data(), 1, len, f) != len) die("piece");
        pieces_[rank] = piece;
        ranks_.emplace(std::move(piece), rank);
    }
    std::fclose(f);
    // Look up the special tokens by their canonical piece strings.  Different
    // model families use different conventions; we try the common ones.
    // DeepSeek uses fullwidth bars and U+2581 LOWER ONE EIGHTH BLOCK as a
    // separator inside the special-token string -- spelled out byte-by-byte
    // to avoid hex-escape ambiguity (gcc treats \xef\xbd\x9c as one number).
    static const char DS_BOS[] = {
        '<', (char)0xef,(char)0xbd,(char)0x9c, 'b','e','g','i','n',
        (char)0xe2,(char)0x96,(char)0x81, 'o','f',
        (char)0xe2,(char)0x96,(char)0x81, 's','e','n','t','e','n','c','e',
        (char)0xef,(char)0xbd,(char)0x9c, '>', 0 };
    static const char DS_EOS[] = {
        '<', (char)0xef,(char)0xbd,(char)0x9c, 'e','n','d',
        (char)0xe2,(char)0x96,(char)0x81, 'o','f',
        (char)0xe2,(char)0x96,(char)0x81, 's','e','n','t','e','n','c','e',
        (char)0xef,(char)0xbd,(char)0x9c, '>', 0 };
    auto find = [&](std::initializer_list<const char*> names) -> uint32_t {
        for (auto* s : names) {
            auto it = ranks_.find(s);
            if (it != ranks_.end()) return it->second;
        }
        return 0u;
    };
    bos_ = find({DS_BOS, "[BOS]", "<s>", "<|begin_of_text|>"});
    eos_ = find({DS_EOS, "[EOS]", "</s>", "<|end_of_text|>", "<|eot_id|>"});
    for (uint32_t i = 0; i < pieces_.size(); ++i) {
        const auto& p = pieces_[i];
        if ((p.rfind("<｜", 0) == 0 && p.size() >= 6 && p.ends_with("｜>")) ||
            p == "<think>" || p == "</think>") {
            specials_.emplace_back(p, i);
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

// ---------------------------------------------------------------------------
// load_from_gguf: read the piece table out of the live Gguf KV metadata.
//
// GGUF wire format for a string array (gguf.cpp:222-230):
//   u32  arr_type == GGUF_STRING
//   u64  arr_count
//   repeat arr_count times:
//     u64    slen
//     u8[]   bytes (no NUL terminator)
// arr_data points at the first slen, arr_bytes covers the whole run.
//
// We copy bytes out (don't retain pointers into the mmap, since GgufModel
// will release the metadata mapping after its parse). Piece index == rank,
// per GGUF's spec for byte-level BPE tokenizers.
// ---------------------------------------------------------------------------
void Tokenizer::load_from_gguf(const Gguf& g) {
    auto it = g.kv_idx.find("tokenizer.ggml.tokens");
    if (it == g.kv_idx.end()) die("tokenizer.ggml.tokens missing");
    const GgufKV& kv = g.kv[it->second];
    if (kv.type != GGUF_ARRAY || kv.arr_type != GGUF_STRING)
        die("tokenizer.ggml.tokens not a STRING array");

    const uint64_t n = kv.arr_count;
    pieces_.resize(n);
    ranks_.reserve(n * 2);

    const uint8_t* p   = kv.arr_data;
    const uint8_t* end = kv.arr_data + kv.arr_bytes;
    for (uint64_t i = 0; i < n; ++i) {
        if ((size_t)(end - p) < sizeof(uint64_t)) die("tokens: truncated slen");
        uint64_t slen;
        std::memcpy(&slen, p, sizeof(uint64_t));
        p += sizeof(uint64_t);
        if ((uint64_t)(end - p) < slen) die("tokens: truncated piece");
        std::string piece((const char*)p, (size_t)slen);
        p += slen;
        pieces_[i] = piece;
        // For byte-level BPE: ranks_ maps piece bytes -> id == rank. If two
        // GGUF entries collide on bytes (shouldn't happen in valid vocabs)
        // we keep the first; emplace is no-op on existing key.
        ranks_.emplace(std::move(piece), (uint32_t)i);
    }
    if (p != end) {
        std::fprintf(stderr, "tok: warning: %lld trailing bytes in tokens array\n",
                     (long long)(end - p));
    }

    // Specials: prefer the explicit u32 KV ids when present (R1 has them).
    auto bos_it = g.kv_idx.find("tokenizer.ggml.bos_token_id");
    auto eos_it = g.kv_idx.find("tokenizer.ggml.eos_token_id");
    if (bos_it != g.kv_idx.end()) bos_ = (uint32_t)g.kv[bos_it->second].u64;
    if (eos_it != g.kv_idx.end()) eos_ = (uint32_t)g.kv[eos_it->second].u64;

    // Whether to prepend BOS. R1 sets add_bos_token=true with bos id 0, so we
    // cannot infer this from the id; default to true when a BOS id is present
    // (the GGUF convention) and override with the explicit flag if given.
    add_bos_ = (bos_it != g.kv_idx.end());
    auto add_bos_it = g.kv_idx.find("tokenizer.ggml.add_bos_token");
    if (add_bos_it != g.kv_idx.end())
        add_bos_ = g.kv[add_bos_it->second].u64 != 0;

    // GGUF token types follow llama.cpp's enum: 3=CONTROL, 4=USER_DEFINED.
    // These literals bypass byte remapping/BPE and are emitted atomically.
    auto type_it = g.kv_idx.find("tokenizer.ggml.token_type");
    if (type_it != g.kv_idx.end()) {
        const GgufKV& types = g.kv[type_it->second];
        if (types.type == GGUF_ARRAY && types.arr_count == n &&
            types.arr_type == GGUF_I32 && types.arr_bytes == n * sizeof(int32_t)) {
            for (uint64_t i = 0; i < n; ++i) {
                int32_t type;
                std::memcpy(&type, types.arr_data + i * sizeof(int32_t), sizeof(type));
                if (type == 3 || type == 4) specials_.emplace_back(pieces_[i], (uint32_t)i);
            }
        }
    }
    // Some quantizers omit token_type. Preserve explicit DeepSeek delimiters
    // as a deterministic fallback.
    if (specials_.empty()) {
        for (uint32_t i = 0; i < pieces_.size(); ++i) {
            const auto& piece = pieces_[i];
            if ((piece.rfind("<｜", 0) == 0 && piece.ends_with("｜>")) ||
                piece == "<think>" || piece == "</think>")
                specials_.emplace_back(piece, i);
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) {
                  if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
                  return a.second < b.second;
              });
    specials_.erase(std::unique(specials_.begin(), specials_.end(),
                                [](const auto& a, const auto& b) { return a.second == b.second; }),
                    specials_.end());
}

// Standard byte-level BPE merge: maintain a doubly-linked list of pieces
// (each starts as one byte), and at each iteration find the adjacent pair
// with lowest merge rank, merge it, repeat until no mergeable pair remains.
static std::vector<uint32_t> bpe_chunk(const std::string& chunk,
                                       const std::unordered_map<std::string, uint32_t>& ranks) {
    if (chunk.empty()) return {};
    // Each "node" is [byte_offset, byte_len, prev, next].
    size_t n = chunk.size();
    std::vector<int32_t> prev(n), next(n);
    std::vector<uint32_t> len(n, 1);
    for (size_t i = 0; i < n; ++i) { prev[i] = (int32_t)i - 1; next[i] = (int32_t)i + 1; }
    next[n-1] = -1;

    auto rank_of = [&](int32_t i) -> uint32_t {
        if (i < 0 || next[i] < 0) return std::numeric_limits<uint32_t>::max();
        std::string s(chunk.data() + i, len[i] + len[next[i]]);
        auto it = ranks.find(s);
        return it == ranks.end() ? std::numeric_limits<uint32_t>::max() : it->second;
    };
    while (true) {
        int32_t best = -1; uint32_t best_r = std::numeric_limits<uint32_t>::max();
        for (int32_t i = 0; i >= 0 && i < (int32_t)n; i = next[i]) {
            uint32_t r = rank_of(i);
            if (r < best_r) { best_r = r; best = i; }
        }
        if (best < 0) break;
        int32_t j = next[best];
        len[best] += len[j];
        next[best] = next[j];
        if (next[j] >= 0) prev[next[j]] = best;
    }
    std::vector<uint32_t> out;
    for (int32_t i = 0; i >= 0 && i < (int32_t)n; i = next[i]) {
        std::string s(chunk.data() + i, len[i]);
        auto it = ranks.find(s);
        if (it == ranks.end()) {
            // byte fallback: emit each byte as its own token if mapping exists.
            for (size_t k = 0; k < len[i]; ++k) {
                std::string b(1, chunk[i + k]);
                auto bit = ranks.find(b);
                if (bit != ranks.end()) out.push_back(bit->second);
            }
        } else out.push_back(it->second);
    }
    return out;
}

// Minimal pretokenizer: split on whitespace boundaries, keeping leading space
// with the following word (as GPT-style BPE expects). After splitting, each
// chunk is remapped through the GPT-2 byte<->unicode bijection so that the
// BPE search sees the same piece strings the GGUF vocab was trained on.
static void encode_plain(const std::string& s,
                         const std::unordered_map<std::string, uint32_t>& ranks,
                         std::vector<uint32_t>& ids) {
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t j = i;
        if (s[i] == ' ' && i + 1 < n && s[i+1] != ' ') {
            j = i + 1;
            while (j < n && s[j] != ' ') ++j;
        } else if (s[i] == ' ') {
            j = i;
            while (j < n && s[j] == ' ') ++j;
        } else {
            j = i;
            while (j < n && s[j] != ' ') ++j;
        }
        std::string chunk = remap_bytes_to_pieces(s.substr(i, j - i));
        auto t = bpe_chunk(chunk, ranks);
        ids.insert(ids.end(), t.begin(), t.end());
        i = j;
    }
}

std::vector<uint32_t> Tokenizer::encode(const std::string& s) const {
    std::vector<uint32_t> ids;
    size_t cursor = 0;
    while (cursor < s.size()) {
        size_t best_pos = std::string::npos;
        const std::pair<std::string, uint32_t>* best = nullptr;
        for (const auto& special : specials_) {
            size_t pos = s.find(special.first, cursor);
            if (pos < best_pos || (pos == best_pos && best &&
                                   special.first.size() > best->first.size())) {
                best_pos = pos;
                best = &special;
            }
        }
        if (!best) {
            encode_plain(s.substr(cursor), ranks_, ids);
            break;
        }
        if (best_pos > cursor)
            encode_plain(s.substr(cursor, best_pos - cursor), ranks_, ids);
        ids.push_back(best->second);
        cursor = best_pos + best->first.size();
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string s;
    for (uint32_t id : ids) {
        if (id >= pieces_.size()) continue;
        s += remap_piece_to_bytes(pieces_[id]);
    }
    return s;
}

} // namespace blade
