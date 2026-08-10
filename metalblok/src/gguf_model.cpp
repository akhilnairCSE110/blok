// src/gguf_model.cpp
#include "gguf_model.hpp"

#include <cstdio>
#include <cstdlib>

namespace blade {

void GgufModel::load(const std::string& shard0_path) {
    // 1) Parse headers via Gguf (mmaps metadata window of each shard).
    //    Gguf::open auto-detects siblings via the split.no/split.count
    //    filename convention and merges the per-shard tensor tables.
    Gguf g;
    g.open(shard0_path);

    // 2) Snapshot what we need into our own structures, resolving each
    //    tensor's absolute byte offset within its shard file.
    shard_paths_.clear();
    shard_paths_.reserve(g.shards.size());
    std::vector<uint64_t> shard_data_off(g.shards.size(), 0);
    for (size_t i = 0; i < g.shards.size(); ++i) {
        shard_paths_.push_back(g.shards[i].path);
        shard_data_off[i] = g.shards[i].data_offset;
    }

    tensors_.clear();
    tensors_.reserve(g.tensors.size());
    by_name_.clear();
    by_name_.reserve(g.tensors.size() * 2);
    total_bytes_ = 0;

    for (const auto& t : g.tensors) {
        if (t.shard >= shard_data_off.size()) {
            std::fprintf(stderr, "[gguf_model] tensor %s has bad shard idx %u\n",
                         t.name.c_str(), t.shard);
            std::abort();
        }
        if (t.dims.size() > 4) {
            std::fprintf(stderr, "[gguf_model] tensor %s has %zu dims (max 4)\n",
                         t.name.c_str(), t.dims.size());
            std::abort();
        }
        GgufTensorEntry e{};
        e.shard      = t.shard;
        e.abs_offset = shard_data_off[t.shard] + t.offset_in_data;
        e.nbytes     = t.bytes;
        e.type       = t.type;
        e.n_dims     = (uint32_t)t.dims.size();
        for (size_t d = 0; d < t.dims.size(); ++d) e.shape[d] = t.dims[d];
        by_name_[t.name] = (uint32_t)tensors_.size();
        tensors_.push_back(e);
        total_bytes_ += t.bytes;
    }

    // 3) Drop all shard mmaps + fds. We keep only the resolved index above
    //    and the shard paths for the PreadRing.
    g.close();

    // 4) Spin up the async pread ring (1 worker thread per shard, each fd
    //    F_NOCACHE+F_RDAHEAD=0). This is the long-lived payload data path.
    if (!ring_.open(shard_paths_)) {
        std::fprintf(stderr, "[gguf_model] PreadRing::open failed: %s\n",
                     ring_.last_error().c_str());
        std::abort();
    }
}

void GgufModel::close() {
    ring_.close();
    tensors_.clear();
    by_name_.clear();
    shard_paths_.clear();
    total_bytes_ = 0;
}

const GgufTensorEntry* GgufModel::find(const std::string& name) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end()) return nullptr;
    return &tensors_[it->second];
}

} // namespace blade
