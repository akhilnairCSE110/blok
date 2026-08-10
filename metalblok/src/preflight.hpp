#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace blade {

struct ShardPreflight {
    std::string path;
    uint64_t logical_bytes = 0;
    uint64_t allocated_bytes = 0;
    uint32_t flags = 0;
    bool exists = false;
    bool regular = false;
    bool dataless = false;
    bool sparse = false;
    bool resident = false;
    std::string error;
};

struct ModelPreflight {
    std::vector<ShardPreflight> shards;
    uint64_t logical_bytes = 0;
    uint64_t allocated_bytes = 0;
    uint64_t missing_physical_bytes = 0;
    uint64_t free_bytes = 0;
    bool all_resident = false;
    bool target_manifest_match = false;
    std::string error;
};

// Metadata-only inspection. It calls lstat/statfs and never open/read/mmap.
ModelPreflight inspect_model_files(const std::string& first_shard);
void print_model_preflight(const ModelPreflight& report);

} // namespace blade
