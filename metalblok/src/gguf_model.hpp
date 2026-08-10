// src/gguf_model.hpp
// ---------------------------------------------------------------------------
// GgufModel: sibling to Model, owns a flat TensorIndex (name -> shard/off/
// bytes/type/dims) plus a long-lived blade::PreadRing across all shards.
//
// load() reads ONLY shard headers via blade::Gguf (which mmaps the metadata
// window). The mmaps are released before load() returns -- we keep only the
// resolved tensor table and the shard paths. Payload reads happen later via
// PreadRing::submit() into caller-supplied 16KB-aligned buffers, per the
// IO_PROBE_FINDINGS.md primitive (pread + F_NOCACHE = 5.8 GB/s, 0 pageouts).
//
// Never mmaps payload. Never reads payload in load().
// ---------------------------------------------------------------------------
#pragma once

#include "gguf.hpp"
#include "pread_ring.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace blade {

struct GgufTensorEntry {
    uint32_t                shard      = 0;   // index into shard_paths()
    uint64_t                abs_offset = 0;   // byte offset into that shard file
    uint64_t                nbytes     = 0;   // payload size
    uint32_t                type       = 0;   // GgmlType
    std::array<uint64_t, 4> shape      = {0, 0, 0, 0};  // GGUF order, [0]=innermost
    uint32_t                n_dims     = 0;
};

class GgufModel {
public:
    // Parse shard 0 (auto-detects siblings via filename split convention),
    // build the flat tensor index, then drop all mmaps. Opens a PreadRing
    // on the shard fds. Aborts on any I/O / format error -- this is a
    // cold load-once path.
    void load(const std::string& shard0_path);

    // Tear down PreadRing. Idempotent. Called by dtor.
    void close();
    ~GgufModel() { close(); }

    // O(1) lookup; nullptr if absent.
    const GgufTensorEntry* find(const std::string& name) const;

    size_t                          tensor_count() const { return tensors_.size(); }
    uint64_t                        total_bytes()  const { return total_bytes_; }
    const std::vector<std::string>& shard_paths()  const { return shard_paths_; }

    // Live ring; callers issue submit()/wait() against this for payload reads.
    PreadRing&       ring()       { return ring_; }
    const PreadRing& ring() const { return ring_; }

    GgufModel() = default;
    GgufModel(const GgufModel&)            = delete;
    GgufModel& operator=(const GgufModel&) = delete;

private:
    std::vector<GgufTensorEntry>              tensors_;
    std::unordered_map<std::string, uint32_t> by_name_;     // name -> tensors_ idx
    std::vector<std::string>                  shard_paths_;
    uint64_t                                  total_bytes_ = 0;
    PreadRing                                 ring_;
};

} // namespace blade
