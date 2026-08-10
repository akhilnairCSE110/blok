// tools/pread_ring_test.cpp
// ---------------------------------------------------------------------------
// Standalone validator for src/pread_ring.{hpp,cpp}.
//
// Mirrors the `random nocache` mode of tools/io_probe_v2: issues 480 reads of
// 4 MB at pseudo-random offsets, spread round-robin across the shards given
// on the command line. Each shard has its own pool of aligned 4 MB staging
// buffers; we keep the pool small (8 slots) so memory stays tiny and we
// force real overlap between producer and the per-shard workers.
//
// Gate (hard failure unless --no-gate):
//   - aggregate throughput >= 5.0 GB/s
//   - d_pageouts == 0 across the entire run
//
// Build: wired into CMakeLists.txt as the `pread_ring_test` target.
//
// Usage:
//   ./build/pread_ring_test <shard1> [<shard2> ...] [--reads N] [--no-gate]
// ---------------------------------------------------------------------------

#include "../src/pread_ring.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include <mach/mach.h>
#include <mach/mach_host.h>

static constexpr size_t kSlotBytes  = 4ull * 1024 * 1024;
static constexpr size_t kAlign      = 16ull * 1024;
static constexpr size_t kSlotsPerSh = 8;

struct VmSample {
    uint64_t pageouts = 0, purges = 0, compressor_pages = 0;
};
static void vm_sample(VmSample* s) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) != KERN_SUCCESS) {
        std::memset(s, 0, sizeof(*s)); return;
    }
    s->pageouts         = vm.pageouts;
    s->purges           = vm.purges;
    s->compressor_pages = vm.compressor_page_count;
}
static double now_s() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static uint64_t peak_rss() {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return (uint64_t)r.ru_maxrss; // Darwin: bytes
}
static uint64_t sm64(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t file_size(const char* p) {
    struct stat st{};
    if (::stat(p, &st) != 0) return 0;
    return (uint64_t)st.st_size;
}

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    size_t total_reads = 480;
    bool   no_gate     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--reads" && i + 1 < argc) { total_reads = (size_t)std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--no-gate")          { no_gate = true; }
        else if (a.rfind("--", 0) == 0)     { std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); return 2; }
        else                                { shards.push_back(a); }
    }
    if (shards.empty()) {
        std::fprintf(stderr,
            "usage: %s <shard1> [<shard2> ...] [--reads N] [--no-gate]\n", argv[0]);
        return 2;
    }

    std::vector<uint64_t> sizes(shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        sizes[i] = file_size(shards[i].c_str());
        if (sizes[i] < kSlotBytes * 2) {
            std::fprintf(stderr, "shard too small or missing: %s\n", shards[i].c_str());
            return 2;
        }
        std::printf("shard[%zu] %s (%.2f GB)\n",
                    i, shards[i].c_str(), sizes[i] / 1e9);
    }

    blade::PreadRing ring;
    if (!ring.open(shards)) {
        std::fprintf(stderr, "ring.open failed: %s\n", ring.last_error().c_str());
        return 1;
    }

    // Per-shard pool of aligned 4 MB buffers + matching done flags.
    const size_t S = shards.size();
    std::vector<void*>             bufs(S * kSlotsPerSh, nullptr);
    std::vector<std::atomic<bool>> dones(S * kSlotsPerSh);
    for (auto& d : dones) d.store(true, std::memory_order_relaxed); // free
    for (size_t i = 0; i < bufs.size(); ++i) {
        if (posix_memalign(&bufs[i], kAlign, kSlotBytes) != 0 || !bufs[i]) {
            std::fprintf(stderr, "posix_memalign failed\n");
            return 1;
        }
        // Touch once so the page is wired before we time.
        std::memset(bufs[i], 0, kSlotBytes);
    }

    // Round-robin a slot index per shard.
    std::vector<size_t> slot_cursor(S, 0);
    uint64_t rng = 0xB1ADE0FFEEull;

    VmSample vm_before{}; vm_sample(&vm_before);
    double   t0 = now_s();
    uint64_t cksum = 0;

    for (size_t i = 0; i < total_reads; ++i) {
        uint32_t sh = (uint32_t)(i % S);
        // Pick a 4 MB-aligned offset in [0, size - 4MB].
        uint64_t span = (sizes[sh] - kSlotBytes) / kSlotBytes;
        uint64_t off  = (sm64(&rng) % (span + 1)) * kSlotBytes;

        size_t slot_idx = sh * kSlotsPerSh + slot_cursor[sh];
        slot_cursor[sh] = (slot_cursor[sh] + 1) % kSlotsPerSh;

        // Recycle this slot: wait for any prior read into it to finish.
        blade::PreadRing::wait(&dones[slot_idx]);
        dones[slot_idx].store(false, std::memory_order_release);

        ring.submit(sh, off, kSlotBytes, bufs[slot_idx], &dones[slot_idx]);
    }

    // Drain everything.
    for (auto& d : dones) blade::PreadRing::wait(&d);

    double   dt = now_s() - t0;
    VmSample vm_after{}; vm_sample(&vm_after);

    // Touch one byte per slot to keep the optimizer honest.
    for (auto* b : bufs) cksum += static_cast<uint8_t*>(b)[0];

    uint64_t bytes = (uint64_t)total_reads * kSlotBytes;
    double   gbps  = bytes / 1e9 / dt;
    int64_t  d_po  = (int64_t)vm_after.pageouts - (int64_t)vm_before.pageouts;
    int64_t  d_pu  = (int64_t)vm_after.purges   - (int64_t)vm_before.purges;

    std::printf(
        "RESULT reads=%zu shards=%zu bytes=%.2fGB time=%.3fs throughput=%.2fGB/s "
        "d_pageouts=%+" PRId64 " d_purges=%+" PRId64 " peak_rss=%.2fMB cksum=0x%016" PRIx64 "\n",
        total_reads, S, bytes / 1e9, dt, gbps, d_po, d_pu,
        peak_rss() / (1024.0 * 1024.0), cksum);

    for (auto* b : bufs) free(b);
    ring.close();

    if (no_gate) return 0;
    int rc = 0;
    if (gbps < 5.0)  { std::fprintf(stderr, "FAIL: throughput %.2f GB/s < 5.0\n", gbps); rc = 4; }
    if (d_po  > 0)   { std::fprintf(stderr, "FAIL: d_pageouts=%" PRId64 "\n", d_po);     rc = 4; }
    return rc;
}
