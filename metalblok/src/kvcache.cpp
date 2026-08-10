#include "kvcache.hpp"
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace blade {

static constexpr uint32_t MAGIC   = 0x424c4b56u;   // 'BLKV'
static constexpr uint32_t VERSION = 1;

struct Header {
    uint32_t magic, version, n_tokens, n_layers, Lk, Dr;
    uint64_t hash;
};

// FNV-1a 64-bit over the token id stream.  hashes[i] = hash of token_ids[0..i).
// One linear pass, lets us answer "does length P match" in O(1) for any P.
static void prefix_hashes(const std::vector<uint32_t>& ids, std::vector<uint64_t>& out) {
    out.resize(ids.size() + 1);
    uint64_t h = 0xcbf29ce484222325ull;
    out[0] = h;
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t t = ids[i];
        for (int b = 0; b < 4; ++b) { h ^= (t >> (8*b)) & 0xff; h *= 0x100000001b3ull; }
        out[i + 1] = h;
    }
}

static bool parse_name(const char* name, uint32_t& n, uint64_t& h) {
    // Format: kv_<n>_<hash16hex>.bin
    return std::sscanf(name, "kv_%u_%16llx.bin", &n, (unsigned long long*)&h) == 2;
}

static std::string path_for(const std::string& dir, uint32_t n, uint64_t h) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/kv_%u_%016llx.bin", n, (unsigned long long)h);
    return dir + buf;
}

void KVCache::init(Runtime& rt, const std::string& dir) {
    RT = &rt;
    dir_ = dir;
    ::mkdir(dir_.c_str(), 0755);   // best-effort; ignore EEXIST
}

uint32_t KVCache::load(const std::vector<uint32_t>& ids) {
    if (!RT || ids.empty()) return 0;
    std::vector<uint64_t> h; prefix_hashes(ids, h);
    DIR* d = ::opendir(dir_.c_str());
    if (!d) return 0;
    uint32_t best_n = 0; uint64_t best_h = 0;
    for (dirent* e = ::readdir(d); e; e = ::readdir(d)) {
        uint32_t n; uint64_t hh;
        if (!parse_name(e->d_name, n, hh)) continue;
        if (n == 0 || n > ids.size() || n <= best_n) continue;
        if (h[n] == hh) { best_n = n; best_h = hh; }
    }
    ::closedir(d);
    if (best_n == 0) return 0;

    std::string p = path_for(dir_, best_n, best_h);
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    Header hdr{};
    if (::read(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)
        || hdr.magic != MAGIC || hdr.version != VERSION
        || hdr.n_tokens != best_n
        || hdr.n_layers != RT->M->cfg.n_layers
        || hdr.Lk != RT->M->cfg.kv_lora_rank
        || hdr.Dr != RT->M->cfg.head_dim_qk_rope) { ::close(fd); return 0; }

    const uint32_t L  = hdr.n_layers;
    const uint32_t Lk = hdr.Lk;
    const uint32_t Dr = hdr.Dr;
    const size_t   ms = RT->M->cfg.max_seq;
    uint8_t* ckv = (uint8_t*)RT->c_kv.contents;
    uint8_t* kr  = (uint8_t*)RT->k_rope.contents;
    bool ok = true;
    for (uint32_t l = 0; l < L && ok; ++l) {
        size_t off = (size_t)l * ms * Lk * 2;
        size_t bytes = (size_t)best_n * Lk * 2;
        ok = ::read(fd, ckv + off, bytes) == (ssize_t)bytes;
    }
    for (uint32_t l = 0; l < L && ok; ++l) {
        size_t off = (size_t)l * ms * Dr * 2;
        size_t bytes = (size_t)best_n * Dr * 2;
        ok = ::read(fd, kr + off, bytes) == (ssize_t)bytes;
    }
    ::close(fd);
    if (!ok) return 0;
    RT->pos = best_n;
    return best_n;
}

void KVCache::save(const std::vector<uint32_t>& ids) {
    if (!RT || RT->pos == 0 || RT->pos > ids.size()) return;
    std::vector<uint64_t> h; prefix_hashes(ids, h);
    const uint32_t n  = RT->pos;
    const uint64_t hh = h[n];
    std::string p = path_for(dir_, n, hh);
    if (::access(p.c_str(), F_OK) == 0) return;          // already cached

    std::string tmp = p + ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    const auto& c = RT->M->cfg;
    Header hdr{ MAGIC, VERSION, n, c.n_layers, c.kv_lora_rank, c.head_dim_qk_rope, hh };
    bool ok = ::write(fd, &hdr, sizeof(hdr)) == (ssize_t)sizeof(hdr);
    const uint8_t* ckv = (const uint8_t*)RT->c_kv.contents;
    const uint8_t* kr  = (const uint8_t*)RT->k_rope.contents;
    for (uint32_t l = 0; l < c.n_layers && ok; ++l) {
        size_t off = (size_t)l * c.max_seq * c.kv_lora_rank * 2;
        size_t bytes = (size_t)n * c.kv_lora_rank * 2;
        ok = ::write(fd, ckv + off, bytes) == (ssize_t)bytes;
    }
    for (uint32_t l = 0; l < c.n_layers && ok; ++l) {
        size_t off = (size_t)l * c.max_seq * c.head_dim_qk_rope * 2;
        size_t bytes = (size_t)n * c.head_dim_qk_rope * 2;
        ok = ::write(fd, kr + off, bytes) == (ssize_t)bytes;
    }
    ::close(fd);
    if (ok) ::rename(tmp.c_str(), p.c_str());
    else    ::unlink(tmp.c_str());
}

} // namespace blade
