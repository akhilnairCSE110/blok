// src/pread_ring.cpp
#include "pread_ring.hpp"

#include <cerrno>
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
        // Publish result to producer.
        r.done->store(true, std::memory_order_release);
        ++tail;
        s->tail.store(tail, std::memory_order_release);
    }
}

} // namespace blade
