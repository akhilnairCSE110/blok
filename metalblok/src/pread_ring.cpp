// src/pread_ring.cpp
#include "pread_ring.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace blade {

PreadRing::~PreadRing() { close(); }

bool PreadRing::open(const std::vector<std::string>& shard_paths) {
    if (running_) {
        last_error_ = "PreadRing::open called twice";
        return false;
    }
    if (shard_paths.empty()) {
        last_error_ = "PreadRing::open: no shards";
        return false;
    }

    shards_.reserve(shard_paths.size());
    for (const auto& p : shard_paths) {
        auto s = std::make_unique<Shard>();
        s->owner = this;
        s->path = p;
        s->fd = ::open(p.c_str(), O_RDONLY);
        if (s->fd < 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "open(%s) failed: %s", p.c_str(), std::strerror(errno));
            last_error_ = buf;
            shards_.clear();
            return false;
        }
        // Critical: defeat the unified buffer cache. Test D1-D4 in
        // IO_PROBE_FINDINGS.md proves this is the only safe mode on M5.
        if (::fcntl(s->fd, F_NOCACHE, 1) < 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "fcntl(F_NOCACHE) failed on %s: %s",
                          p.c_str(), std::strerror(errno));
            last_error_ = buf;
            ::close(s->fd);
            shards_.clear();
            return false;
        }
        // Disable read-ahead; we already issue large reads ourselves and
        // read-ahead just inflates the inactive queue.
        ::fcntl(s->fd, F_RDAHEAD, 0);

        shards_.push_back(std::move(s));
    }

    // Spawn workers only after all fds are good, so partial failure
    // cleanup above doesn't have to join threads.
    for (auto& s : shards_) {
        Shard* raw = s.get();
        raw->worker = std::thread(&PreadRing::worker_loop, raw);
    }
    running_ = true;
    return true;
}

void PreadRing::close() {
    if (!running_ && shards_.empty()) return;
    for (auto& s : shards_) {
        s->stop.store(true, std::memory_order_release);
    }
    for (auto& s : shards_) {
        if (s->worker.joinable()) s->worker.join();
        if (s->fd >= 0) { ::close(s->fd); s->fd = -1; }
    }
    shards_.clear();
    running_ = false;
}

void PreadRing::submit(uint32_t shard_idx,
                       uint64_t offset,
                       uint64_t nbytes,
                       void*    dst,
                       std::atomic<bool>* done) {
    Shard* s = shards_[shard_idx].get();
    // Wait for a free slot. Single-producer => only this thread writes head,
    // so head doesn't need atomic load with synchronization.
    uint64_t head = s->head.load(std::memory_order_relaxed);
    for (;;) {
        uint64_t tail = s->tail.load(std::memory_order_acquire);
        if (head - tail < kQueueCapacity) break;
        // Busy-wait; queue is sized so this is rare.
        std::this_thread::yield();
    }
    Request& r = s->ring[head % kQueueCapacity];
    r.offset = offset;
    r.nbytes = nbytes;
    r.dst    = dst;
    r.done   = done;
    s->head.store(head + 1, std::memory_order_release);
}

void PreadRing::wait(const std::atomic<bool>* done) {
    while (!done->load(std::memory_order_acquire)) {
        // Tight spin; the workload (per-layer expert fetches) is latency
        // sensitive enough that a syscall-based wait would dominate.
    }
}

void PreadRing::reset_stats() {
    stat_bytes_.store(0, std::memory_order_relaxed);
    stat_requests_.store(0, std::memory_order_relaxed);
    stat_first_ns_.store(0, std::memory_order_relaxed);
    stat_last_ns_.store(0, std::memory_order_relaxed);
}

PreadRing::Stats PreadRing::stats() const {
    const uint64_t first = stat_first_ns_.load(std::memory_order_relaxed);
    const uint64_t last = stat_last_ns_.load(std::memory_order_relaxed);
    return {stat_bytes_.load(std::memory_order_relaxed),
            stat_requests_.load(std::memory_order_relaxed),
            last > first ? (last - first) / 1000 : 0};
}

void PreadRing::worker_loop(Shard* s) {
    uint64_t tail = s->tail.load(std::memory_order_relaxed);
    for (;;) {
        uint64_t head = s->head.load(std::memory_order_acquire);
        if (tail == head) {
            if (s->stop.load(std::memory_order_acquire)) return;
            std::this_thread::yield();
            continue;
        }
        Request r = s->ring[tail % kQueueCapacity];
        const auto now_ns = [] {
            return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        };
        const uint64_t started = now_ns();
        uint64_t unset = 0;
        s->owner->stat_first_ns_.compare_exchange_strong(
            unset, started, std::memory_order_relaxed);

        // Loop pread to handle short reads (very rare with F_NOCACHE on a
        // local file, but cheap insurance).
        uint8_t* p     = static_cast<uint8_t*>(r.dst);
        uint64_t left  = r.nbytes;
        uint64_t off   = r.offset;
        while (left > 0) {
            ssize_t n = ::pread(s->fd, p, (size_t)left, (off_t)off);
            if (n > 0) {
                p    += n;
                off  += (uint64_t)n;
                left -= (uint64_t)n;
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            // Hard failure or EOF mid-read. We have no error channel per
            // request, so log to stderr and still flip done so the producer
            // doesn't deadlock. Higher layers should treat partial data as
            // a fatal model-load error and abort.
            std::fprintf(stderr,
                "pread_ring: %s: pread off=%llu left=%llu rc=%zd errno=%d\n",
                s->path.c_str(),
                (unsigned long long)off, (unsigned long long)left, n, errno);
            break;
        }
        s->owner->stat_bytes_.fetch_add(r.nbytes - left, std::memory_order_relaxed);
        s->owner->stat_requests_.fetch_add(1, std::memory_order_relaxed);
        const uint64_t finished = now_ns();
        uint64_t previous = s->owner->stat_last_ns_.load(std::memory_order_relaxed);
        while (previous < finished && !s->owner->stat_last_ns_.compare_exchange_weak(
                   previous, finished, std::memory_order_relaxed)) {}
        // Publish result to producer.
        r.done->store(true, std::memory_order_release);
        ++tail;
        s->tail.store(tail, std::memory_order_release);
    }
}

} // namespace blade
