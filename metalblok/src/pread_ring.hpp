// src/pread_ring.hpp
// ---------------------------------------------------------------------------
// Async pread ring: 1 worker thread per shard, each owning an F_NOCACHE fd
// and an SPSC request queue. Caller supplies the destination buffer; the
// worker pread()s directly into it and flips *done to true on completion.
//
// Design contract (matches IO_PROBE_FINDINGS.md: pread+F_NOCACHE into a
// reused aligned buffer = 5.8 GB/s, 0 pageouts):
//   - One producer thread per ring (the runtime decode loop).
//   - One consumer thread per shard (spawned by open()).
//   - dst MUST remain valid until *done is observed true.
//   - dst SHOULD be 16 KB aligned for top throughput on M-series.
//   - No exceptions, no RTTI (matches project-wide CMake flags).
// ---------------------------------------------------------------------------
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace blade {

class PreadRing {
public:
    // Per-shard outstanding request capacity. 64 lets the runtime overlap
    // many expert fetches without ever blocking on submit().
    static constexpr size_t kQueueCapacity = 64;

    // Open one fd per shard (O_RDONLY, then fcntl F_NOCACHE=1, F_RDAHEAD=0)
    // and spawn one worker thread per shard. On any failure, no fds remain
    // open, no threads remain running, and last_error() is populated.
    bool open(const std::vector<std::string>& shard_paths);

    // Stop all workers, join, close fds. Idempotent. Called by dtor.
    void close();

    // Enqueue a read. Blocks (spins) the producer iff the target shard's
    // ring is full (kQueueCapacity outstanding). *done must point to a
    // bool the caller has set to false; the worker stores true with
    // release ordering when the read finishes.
    void submit(uint32_t shard_idx,
                uint64_t offset,
                uint64_t nbytes,
                void*    dst,
                std::atomic<bool>* done);

    // Spin until *done == true (acquire). Caller must reset *done = false
    // before re-using it for another submit().
    static void wait(const std::atomic<bool>* done);

    struct Stats { uint64_t bytes, requests, elapsed_us; };
    void reset_stats();
    Stats stats() const;

    const std::string& last_error() const { return last_error_; }
    size_t             shard_count() const { return shards_.size(); }

    PreadRing() = default;
    ~PreadRing();
    PreadRing(const PreadRing&)            = delete;
    PreadRing& operator=(const PreadRing&) = delete;

private:
    struct Request {
        uint64_t           offset;
        uint64_t           nbytes;
        void*              dst;
        std::atomic<bool>* done;
    };

    struct Shard {
        PreadRing* owner = nullptr;
        int               fd = -1;
        std::string       path;
        Request           ring[kQueueCapacity];
        // SPSC indices. Producer publishes with head.store(release);
        // consumer reads with head.load(acquire). Mirror for tail.
        std::atomic<uint64_t> head{0};
        std::atomic<uint64_t> tail{0};
        std::atomic<bool>     stop{false};
        std::thread           worker;
    };

    static void worker_loop(Shard* s);

    std::vector<std::unique_ptr<Shard>> shards_;
    std::string                         last_error_;
    bool                                running_ = false;
    std::atomic<uint64_t> stat_bytes_{0};
    std::atomic<uint64_t> stat_requests_{0};
    std::atomic<uint64_t> stat_first_ns_{0};
    std::atomic<uint64_t> stat_last_ns_{0};
};

} // namespace blade
