#include "preflight.hpp"

#include <sys/mount.h>
#include <sys/stat.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <regex>

namespace blade {
namespace {

constexpr std::array<uint64_t, 3> kR1Iq1sSizes = {
    49349193664ULL, 49397904416ULL, 41484340384ULL
};

struct SplitName {
    std::string base;
    uint32_t first = 0;
    uint32_t count = 0;
};

bool split_name(const std::string& path, SplitName& out) {
    static const std::regex re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return false;
    out.base = m[1].str();
    out.first = static_cast<uint32_t>(std::stoul(m[2].str()));
    out.count = static_cast<uint32_t>(std::stoul(m[3].str()));
    return out.count > 0;
}

std::string shard_path(const SplitName& split, uint32_t one_based) {
    char suffix[40];
    std::snprintf(suffix, sizeof(suffix), "-%05u-of-%05u.gguf",
                  one_based, split.count);
    return split.base + suffix;
}

ShardPreflight inspect_one(const std::string& path) {
    ShardPreflight out;
    out.path = path;
    struct stat st {};
    if (::lstat(path.c_str(), &st) != 0) {
        out.error = std::string("lstat: ") + std::strerror(errno);
        return out;
    }
    out.exists = true;
    out.regular = S_ISREG(st.st_mode);
    out.logical_bytes = static_cast<uint64_t>(st.st_size);
    out.allocated_bytes = static_cast<uint64_t>(st.st_blocks) * 512ULL;
    out.flags = static_cast<uint32_t>(st.st_flags);
    out.dataless = (st.st_flags & SF_DATALESS) != 0;

    // Model weights are effectively incompressible. A large file with less
    // than 90% of its logical extent physically allocated is a sparse/cloud
    // object and must not be touched by the GGUF parser.
    out.sparse = out.logical_bytes > (16ULL << 20) &&
                 out.allocated_bytes < out.logical_bytes * 9ULL / 10ULL;
    out.resident = out.exists && out.regular && out.logical_bytes > 0 &&
                   !out.dataless && !out.sparse;
    if (!out.regular) out.error = "not a regular file";
    else if (out.dataless) out.error = "APFS dataless placeholder";
    else if (out.sparse) out.error = "payload is not physically resident";
    else if (out.logical_bytes == 0) out.error = "empty shard";
    return out;
}

} // namespace

ModelPreflight inspect_model_files(const std::string& first_shard) {
    ModelPreflight out;
    SplitName split;
    if (split_name(first_shard, split)) {
        if (split.first != 1) {
            out.error = "pass shard 00001, not a later shard";
            return out;
        }
        if (split.count > 128) {
            out.error = "unreasonable shard count";
            return out;
        }
        out.shards.reserve(split.count);
        for (uint32_t i = 1; i <= split.count; ++i) {
            out.shards.push_back(inspect_one(shard_path(split, i)));
        }
    } else {
        out.shards.push_back(inspect_one(first_shard));
    }

    out.all_resident = !out.shards.empty();
    for (const auto& shard : out.shards) {
        out.logical_bytes += shard.logical_bytes;
        out.allocated_bytes += shard.allocated_bytes;
        if (!shard.resident) {
            out.all_resident = false;
            out.missing_physical_bytes += shard.logical_bytes;
        }
    }

    struct statfs fs {};
    if (::statfs(first_shard.c_str(), &fs) == 0) {
        out.free_bytes = static_cast<uint64_t>(fs.f_bavail) *
                         static_cast<uint64_t>(fs.f_bsize);
    }

    if (out.shards.size() == kR1Iq1sSizes.size()) {
        out.target_manifest_match = true;
        for (size_t i = 0; i < kR1Iq1sSizes.size(); ++i) {
            if (out.shards[i].logical_bytes != kR1Iq1sSizes[i]) {
                out.target_manifest_match = false;
            }
        }
    }
    return out;
}

void print_model_preflight(const ModelPreflight& report) {
    std::printf("model_preflight all_resident=%s manifest=%s shards=%zu\n",
                report.all_resident ? "true" : "false",
                report.target_manifest_match ? "deepseek-r1-ud-iq1_s" : "unknown",
                report.shards.size());
    for (size_t i = 0; i < report.shards.size(); ++i) {
        const auto& s = report.shards[i];
        std::printf("shard[%zu] resident=%s dataless=%s sparse=%s "
                    "logical=%llu allocated=%llu path=%s",
                    i, s.resident ? "true" : "false",
                    s.dataless ? "true" : "false",
                    s.sparse ? "true" : "false",
                    static_cast<unsigned long long>(s.logical_bytes),
                    static_cast<unsigned long long>(s.allocated_bytes),
                    s.path.c_str());
        if (!s.error.empty()) std::printf(" error=\"%s\"", s.error.c_str());
        std::putchar('\n');
    }
    std::printf("logical_total=%llu allocated_total=%llu missing_physical=%llu "
                "free=%llu\n",
                static_cast<unsigned long long>(report.logical_bytes),
                static_cast<unsigned long long>(report.allocated_bytes),
                static_cast<unsigned long long>(report.missing_physical_bytes),
                static_cast<unsigned long long>(report.free_bytes));
    if (!report.error.empty()) std::printf("error=\"%s\"\n", report.error.c_str());
}

} // namespace blade
