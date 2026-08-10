// KV prefix cache.  MLA compresses K/V to (Lk + Dr) floats per token per
// layer (~31 KB/token on V2-Lite), so an entire system prompt's KV state
// fits comfortably on disk.  We hash the token-id sequence and save the
// live slice of c_kv / k_rope keyed by (length, FNV1a-64).  On the next
// run we scan the cache dir, find the longest cached prefix that matches
// the current prompt, and memcpy it back into the runtime's shared Metal
// buffers -- skipping prefill for those tokens.
//
// Aligned with KV-Cache Persistence (arxiv 2603.04428): treat NVMe as a
// content-addressable extension of the latent KV memory.
#pragma once
#include "runtime.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace blade {

class KVCache {
public:
    void init(Runtime& rt, const std::string& dir);
    // Returns # tokens loaded (== rt.pos after call), 0 on miss.
    uint32_t load(const std::vector<uint32_t>& token_ids);
    // Save current state for token_ids[0..rt.pos).  No-op if rt.pos == 0.
    void save(const std::vector<uint32_t>& token_ids);
private:
    Runtime* RT = nullptr;
    std::string dir_;
};

} // namespace blade
