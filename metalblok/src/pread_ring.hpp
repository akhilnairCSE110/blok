// src/pread_ring.hpp
// ---------------------------------------------------------------------------
// Async pread ring: a measured number of worker lanes per shard, each owning
// an F_NOCACHE fd and two SPSC request queues. Caller supplies the destination
// buffer; the worker pread()s directly into it and publishes completion.
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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace blade {

class PreadRing {
public:
    // Per-shard outstanding request capacity. 64 lets the runtime overlap
    // many expert fetches without ever blocking on submit().
    static constexpr size_t kQueueCapacity = 64;
    static constexpr size_t kDefaultLanesPerShard = 8;

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
                std::atomic<bool>* done,
                bool urgent = false);

    // Spin until *done == true (acquire). Caller must reset *done = false
    // before re-using it for another submit().
    static void wait(const std::atomic<bool>* done);

    struct Stats {
        uint64_t bytes, requests, elapsed_us;
        uint64_t urgent_bytes, urgent_requests;
        uint64_t service_us, max_service_us, peak_outstanding;
    };
    struct LaneStats {
        uint32_t shard, lane;
        uint64_t bytes, requests, urgent_bytes, urgent_requests;
        uint64_t service_us, max_service_us;
    };
    void reset_stats();
    Stats stats() const;
    std::vector<LaneStats> lane_stats() const;

    const std::string& last_error() const { return last_error_; }
    size_t             shard_count() const { return logical_shards_; }

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
        uint32_t          shard_idx = 0;
        uint32_t          lane_idx = 0;
        int               fd = -1;
        std::string       path;
        // Queue 0 is background fixed-weight prefetch; queue 1 is the exact
        // router-selected expert critical path. The worker drains urgent
        // work first but never interrupts an in-flight pread.
        Request           ring[2][kQueueCapacity];
        // SPSC indices. Producer publishes with head.store(release);
        // consumer reads with head.load(acquire). Mirror for tail.
        std::atomic<uint64_t> head[2]{};
        std::atomic<uint64_t> tail[2]{};
        std::atomic<bool>     stop{false};
        std::mutex            mutex;
        std::condition_variable wake;
        std::thread           worker;
        std::atomic<uint64_t> stat_bytes{0}, stat_requests{0};
        std::atomic<uint64_t> stat_urgent_bytes{0}, stat_urgent_requests{0};
        std::atomic<uint64_t> stat_service_ns{0}, stat_max_service_ns{0};
    };

    static void worker_loop(Shard* s);

    std::vector<std::unique_ptr<Shard>> shards_;
    std::vector<uint32_t>                lane_cursor_;
    size_t                               logical_shards_ = 0;
    size_t                               lanes_per_shard_ = kDefaultLanesPerShard;
    std::string                         last_error_;
    bool                                running_ = false;
    std::atomic<uint64_t> stat_bytes_{0};
    std::atomic<uint64_t> stat_requests_{0};
    std::atomic<uint64_t> stat_first_ns_{0};
    std::atomic<uint64_t> stat_last_ns_{0};
    std::atomic<uint64_t> stat_urgent_bytes_{0};
    std::atomic<uint64_t> stat_urgent_requests_{0};
    std::atomic<uint64_t> stat_service_ns_{0};
    std::atomic<uint64_t> stat_max_service_ns_{0};
    std::atomic<uint64_t> stat_outstanding_{0};
    std::atomic<uint64_t> stat_peak_outstanding_{0};
};

} // namespace blade
