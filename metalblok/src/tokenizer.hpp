// Byte-level BPE tokenizer (tiktoken-style).  Supports any vocab whose pieces
// are stored as raw byte strings -- currently exercised on Kimi K2 (vocab
// 163840) and DeepSeek-V2-Lite (vocab 102400).  Special-token IDs (BOS/EOS)
// are auto-detected at load time from a small set of canonical piece names.
//
// File format we accept (`tokenizer.bin`, emitted by tools/dump_tokenizer.py
// for HF or tools/convert_kimi_k2.py for Kimi):
//   uint32  n_tokens
//   for i in 0..n_tokens-1:
//       uint32 rank        // BPE merge rank == token id
//       uint32 byte_len
//       byte[byte_len]     // raw bytes of the token piece
//
// Encode runs the standard tiktoken algorithm: split input by a minimal
// whitespace-aware pretokenizer, then for each chunk repeatedly merge the
// lowest-rank adjacent byte-pair until no merges remain.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace blade {

class Gguf;  // fwd-decl; defined in gguf.hpp

class Tokenizer {
public:
    void load(const std::string& path);

    // Build the tokenizer from a live Gguf object (metadata still mmapped).
    // Reads tokenizer.ggml.tokens (string array) for the piece table and
    // tokenizer.ggml.bos_token_id / eos_token_id (u32) for specials.
    // Merges are not required for byte-level BPE encode -- the piece-rank
    // map alone is sufficient, because for any chunk we always pick the
    // adjacent pair whose concatenation has the lowest rank in the piece
    // table, and a pair that is unmergeable is simply absent from ranks_.
    // The GGUF token order is itself the merge order, so piece index == rank.
    // Caller must keep the Gguf alive across this call (we copy out the
    // bytes; safe to close the Gguf after return).
    void load_from_gguf(const Gguf& g);

    std::vector<uint32_t> encode(const std::string& s) const;
    std::string           decode(const std::vector<uint32_t>& ids) const;
    uint32_t bos() const { return bos_; }
    uint32_t eos() const { return eos_; }
    // Whether a BOS token should be prepended to an encoded prompt. Read from
    // tokenizer.ggml.add_bos_token. NOTE: the BOS id itself can legitimately
    // be 0 (DeepSeek-R1: <｜begin▁of▁sentence｜> == id 0), so callers must use
    // this flag and must NOT treat bos()==0 as "no BOS".
    bool add_bos() const { return add_bos_; }

private:
    // pieces_[id] = the raw bytes for token id.
    std::vector<std::string> pieces_;
    // map from piece bytes -> rank (== id).
    std::unordered_map<std::string, uint32_t> ranks_;
    // Control/user-defined tokens must be recognized before byte-level BPE.
    // Each entry is (literal UTF-8 spelling, token id), longest first.
    std::vector<std::pair<std::string, uint32_t>> specials_;
    uint32_t bos_ = 0, eos_ = 0;
    bool     add_bos_ = false;
};

} // namespace blade
