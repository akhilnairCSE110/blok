// mem::free_bytes() — the ONLY source of "how much physical RAM can we touch."
//
// Used by:
//   - Streamer::init  to size the expert-LRU budget per the grounding-doc rule
//                     budget = min(10 GiB, max(512 MiB, free - 4 GiB))
//   - validate_gemv   to refuse if a single tensor slice doesn't fit in
//                     comfortable headroom
//   - (future)        Model::open_gguf, Runtime::init -- anywhere we want to
//                     declare a working set, we ask this first.
//
// The grounding doc is explicit: HARDCODED MEMORY BUDGETS ARE THE BUG.
// On a 24 GB box with VS Code + a browser open, "free" is < 100 MB; on a
// fresh boot it's > 20 GB. A constant in source code cannot serve both.
//
// Darwin: host_statistics64(HOST_VM_INFO64).
//   "free" = (free_count + speculative_count) * page_size
//   "available" = free + inactive + conservative active file-backed pages.
//     Model reads leave clean pages on the active external queue; excluding
//     them makes a warmed 420 GB checkpoint look like unreclaimable RAM.
// Linux:  /proc/meminfo MemAvailable (kernel-computed; kernel-correct).
//
// We expose two numbers because they answer different questions:
//   free_bytes()      -- "how much RAM has nothing in it right now"
//   available_bytes() -- "how much RAM could we get by evicting clean pages"
// The streamer cares about `available` (we'll evict clean file-backed pages);
// the validator cares about `free` (we want strict no-pressure headroom).

#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>

#if defined(__APPLE__)
  #include <mach/mach.h>
  #include <mach/mach_host.h>
  #include <sys/sysctl.h>
  #include <unistd.h>
#elif defined(__linux__)
  #include <cstdio>
  #include <cstring>
  #include <unistd.h>
#endif

namespace blade { namespace mem {

struct Snapshot {
    uint64_t total;       // physical RAM bytes
    uint64_t free;        // strictly unused
    uint64_t available;   // free + reclaimable (inactive/cached file pages)
    uint64_t page_size;
};

inline Snapshot snapshot() {
    Snapshot s{};
#if defined(__APPLE__)
    s.page_size = (uint64_t)::sysconf(_SC_PAGESIZE);
    // Total RAM: hw.memsize
    {
        int      mib[2] = { CTL_HW, HW_MEMSIZE };
        uint64_t mem    = 0;
        size_t   len    = sizeof(mem);
        if (::sysctl(mib, 2, &mem, &len, nullptr, 0) == 0) s.total = mem;
    }
    // VM stats
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &cnt) == KERN_SUCCESS) {
        const uint64_t pg = s.page_size;
        s.free      = (uint64_t)(vm.free_count + vm.speculative_count) * pg;
        // external_page_count spans page queues. Subtract every inactive and
        // speculative page (even internal ones) before adding the remainder;
        // this deliberately undercounts rather than double-counts reclaimable
        // active file cache. Wired, anonymous-active, and compressed pages are
        // never included.
        const uint64_t external = vm.external_page_count;
        const uint64_t counted = uint64_t(vm.inactive_count) + vm.speculative_count;
        const uint64_t active_file = external > counted ? external - counted : 0;
        s.available = s.free +
            (uint64_t(vm.inactive_count) + active_file) * pg;
    }
#elif defined(__linux__)
    s.page_size = (uint64_t)::sysconf(_SC_PAGESIZE);
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (f) {
        char  key[64]; uint64_t val_kb; char unit[16];
        while (std::fscanf(f, "%63s %llu %15s\n",
                           key, (unsigned long long*)&val_kb, unit) == 3) {
            const uint64_t b = val_kb * 1024ull;
            if      (std::strncmp(key, "MemTotal:",     9)  == 0) s.total     = b;
            else if (std::strncmp(key, "MemFree:",      8)  == 0) s.free      = b;
            else if (std::strncmp(key, "MemAvailable:", 13) == 0) s.available = b;
        }
        std::fclose(f);
    }
#endif
    if (s.available == 0) s.available = s.free;   // fallback
    return s;
}

inline uint64_t free_bytes()      { return snapshot().free; }
inline uint64_t available_bytes() { return snapshot().available; }
inline uint64_t total_bytes()     { return snapshot().total; }

// Convenience: format "free=X.XX GB / avail=X.XX GB / total=XX.XX GB"
// into a caller-provided buffer.  No allocation; safe in hot paths.
inline void format(const Snapshot& s, char* buf, size_t buflen) {
    std::snprintf(buf, buflen,
                  "free=%.2fGB avail=%.2fGB total=%.2fGB",
                  s.free / 1e9, s.available / 1e9, s.total / 1e9);
}

}} // namespace blade::mem
