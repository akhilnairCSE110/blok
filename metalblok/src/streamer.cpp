#include "streamer.hpp"
#include "memstat.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdio>

namespace blade {

// Compute the byte range for an expert.
//   FP8 path:  contiguous (offset, nbytes) from experts.idx.
//   bf16 path: the three sub-tensors (gate/up/down) live at separate offsets
//              inside two different mmap'd shards in the worst case, so we
//              walk all three pointers and madvise each.
struct ExpertSpan { const uint8_t* base; size_t nbytes; };
static inline void expert_spans(const Model& m, uint32_t moe_layer, uint32_t expert,
                                ExpertSpan out[3], uint32_t& count) {
    if (m.cfg.weight_dtype == 1) {
        const auto& re = m.raw_experts[(size_t)moe_layer * m.cfg.n_experts + expert];
        size_t bytes_gu = bf16_bytes((size_t)m.cfg.expert_ffn * m.cfg.hidden);
        size_t bytes_dn = bf16_bytes((size_t)m.cfg.hidden * m.cfg.expert_ffn);
        out[0] = { (const uint8_t*)re.gate, bytes_gu };
        out[1] = { (const uint8_t*)re.up,   bytes_gu };
        out[2] = { (const uint8_t*)re.down, bytes_dn };
        count = 3;
    } else {
        const auto& e = m.expert_idx[(size_t)moe_layer * m.cfg.n_experts + expert];
        out[0] = { m.experts_base + e.offset, e.nbytes };
        count = 1;
    }
}

static inline void madvise_span(const ExpertSpan& s, int advice) {
    if (!s.base || s.nbytes == 0) return;
    uintptr_t base = (uintptr_t)s.base;
    uintptr_t pg   = base & ~(uintptr_t)4095;
    size_t    span = (s.nbytes + (base - pg) + 4095) & ~size_t{4095};
    ::madvise((void*)pg, span, advice);
}

// All dense (non-expert) tensor spans for one transformer layer.  Order is
// the on-disk order from Model::open*; we hint them in a single sweep so the
// kernel can coalesce sequential pages into bulk reads -- the LLM-in-a-Flash
// row-column-bundling intuition without changing the on-disk layout.
//
// We deliberately omit the per-layer norm gains: those live in a tiny
// heap-allocated arena (Model::norm_arena) that's already RAM-resident, and
// they're <1 KB total per layer.  Likewise router_bias is fp32 [n_experts]
// and trivially small.
static inline uint32_t layer_dense_spans(const Model& m, uint32_t layer,
                                         ExpertSpan out[16]) {
    const auto& c = m.cfg;
    if (layer >= c.n_layers) return 0;
    const auto& h = m.layers[layer];
    auto wb = [&](size_t n) { return weight_bytes(n, c.weight_dtype); };
    const size_t H  = c.hidden;
    const size_t Lk = c.kv_lora_rank;
    const size_t qk = (size_t)c.head_dim_qk_nope + c.head_dim_qk_rope;
    uint32_t n = 0;
    auto add_span = [&](const void* p, size_t bytes) {
        if (p && bytes > 0) out[n++] = { (const uint8_t*)p, bytes };
    };
    // Q path: either Q-LoRA (a + b) or single q_proj.
    if (c.q_lora_rank > 0) {
        add_span(h.w_q_a, wb((size_t)c.q_lora_rank * H));
        add_span(h.w_q_b, wb((size_t)c.n_heads * qk * c.q_lora_rank));
    } else {
        add_span(h.w_q,   wb((size_t)c.n_heads * qk * H));
    }
    // KV down + per-head W_uk/W_uv + output projection.
    add_span(h.w_kv_a, wb((size_t)(Lk + c.head_dim_qk_rope) * H));
    // W_uk / W_uv live in our owned arena (ALWAYS bf16 on the HF path), so
    // their pages are RAM-resident too -- skip them.  On the FP8 path they're
    // in hot.bin which is mlock'd.  Either way, no value in WILLNEED'ing them.
    add_span(h.w_o,    wb(H * (size_t)c.n_heads * c.head_dim_v));
    // FFN: dense vs MoE.
    if (layer < c.n_dense_layers) {
        add_span(h.w_gate_dense, wb((size_t)c.ffn_dense * H));
        add_span(h.w_up_dense,   wb((size_t)c.ffn_dense * H));
        add_span(h.w_down_dense, wb(H * (size_t)c.ffn_dense));
    } else {
        const size_t Fs = (size_t)c.expert_ffn * c.n_shared_experts;
        add_span(h.w_gate_shared, wb(Fs * H));
        add_span(h.w_up_shared,   wb(Fs * H));
        add_span(h.w_down_shared, wb(H * Fs));
        add_span(h.w_router,      wb((size_t)c.n_experts * H));
    }
    return n;
}

void Streamer::init(Model& m) {
    M = &m;
    total_experts = (size_t)(m.cfg.n_layers - m.cfg.n_dense_layers) * m.cfg.n_experts;
    if (total_experts == 0) return;
    lru = (uint64_t*)std::calloc(total_experts, sizeof(uint64_t));
    // bf16 path: per-expert size = 3 * Fe*H*2.  FP8 path: m.esz.total.
    size_t per_expert = (m.cfg.weight_dtype == 1)
        ? (3 * bf16_bytes((size_t)m.cfg.expert_ffn * m.cfg.hidden))
        : m.esz.total;
    per_expert_bytes = per_expert;

    // Size the LRU byte budget from CURRENT free RAM, never a hardcoded
    // constant.  Grounding doc rule (FlexGen / LLM-in-a-Flash budget mgmt):
    //     budget = min(10 GiB, max(512 MiB, avail - 4 GiB))
    // The 4 GiB headroom is for activations + KV + hot-band; if the box has
    // less than (per_expert + 4 GiB) available we still allow a 512 MiB floor
    // so the worker can at least do single-shot prefetch + DONTNEED churn.
    const uint64_t avail    = mem::available_bytes();
    const uint64_t HEADROOM = 4ull << 30;
    const uint64_t FLOOR    = 512ull << 20;
    const uint64_t CEIL     = 10ull << 30;
    uint64_t b = (avail > HEADROOM) ? (avail - HEADROOM) : 0;
    if (b < FLOOR) b = FLOOR;
    if (b > CEIL)  b = CEIL;
    budget_bytes = (size_t)b;
    {
        char buf[128]; auto s = mem::snapshot(); mem::format(s, buf, sizeof(buf));
        std::fprintf(stderr,
            "[streamer] %s -> LRU budget %.2f GiB (per-expert %.2f MiB, %zu experts)\n",
            buf, budget_bytes / (double)(1ull<<30),
            per_expert_bytes / (double)(1ull<<20), total_experts);
    }

    worker = std::thread([this]{ run(); });
}

void Streamer::shutdown() {
    stop.store(true);
    if (worker.joinable()) worker.join();
    std::free(lru); lru = nullptr;
}

void Streamer::prefetch(uint32_t moe_layer, uint32_t expert) {
    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t n = (h + 1) % RING;
    if (n == tail.load(std::memory_order_acquire)) return;   // ring full -> drop
    ring[h] = {0, moe_layer, expert};
    head.store(n, std::memory_order_release);
}

void Streamer::prefetch_layer(uint32_t layer) {
    uint32_t h = head.load(std::memory_order_relaxed);
    uint32_t n = (h + 1) % RING;
    if (n == tail.load(std::memory_order_acquire)) return;
    ring[h] = {1, layer, 0};
    head.store(n, std::memory_order_release);
}

void Streamer::touch(uint32_t moe_layer, uint32_t expert) {
    if (!lru) return;
    size_t i = (size_t)moe_layer * M->cfg.n_experts + expert;
    lru[i] = clock.fetch_add(1, std::memory_order_relaxed) + 1;
}

void Streamer::willneed_layer(uint32_t layer) {
    ExpertSpan sp[16];
    uint32_t n = layer_dense_spans(*M, layer, sp);
    for (uint32_t s = 0; s < n; ++s) madvise_span(sp[s], MADV_WILLNEED);
}

void Streamer::run() {
    using namespace std::chrono;
    auto last_evict = steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        // Drain prefetch ring.
        bool did_work = false;
        while (true) {
            uint32_t t = tail.load(std::memory_order_relaxed);
            if (t == head.load(std::memory_order_acquire)) break;
            Job j = ring[t];
            tail.store((t + 1) % RING, std::memory_order_release);
            if (j.kind == 1) {
                willneed_layer(j.a);
            } else {
                ExpertSpan sp[3]; uint32_t nspans = 0;
                expert_spans(*M, j.a, j.b, sp, nspans);
                for (uint32_t s = 0; s < nspans; ++s) madvise_span(sp[s], MADV_WILLNEED);
            }
            did_work = true;
        }

        // Periodic LRU eviction (best-effort).
        if (steady_clock::now() - last_evict > milliseconds(250)) {
            last_evict = steady_clock::now();
            size_t resident = 0;
            for (size_t i = 0; i < total_experts; ++i)
                if (lru[i]) resident += per_expert_bytes;
            if (resident > budget_bytes) {
                uint64_t now = clock.load(std::memory_order_relaxed);
                uint64_t thresh = now > 256 ? now - 256 : 0;
                for (size_t i = 0; i < total_experts; ++i) {
                    if (lru[i] && lru[i] < thresh) {
                        uint32_t ml = (uint32_t)(i / M->cfg.n_experts);
                        uint32_t e  = (uint32_t)(i % M->cfg.n_experts);
                        ExpertSpan sp[3]; uint32_t nspans = 0;
                        expert_spans(*M, ml, e, sp, nspans);
                        for (uint32_t s = 0; s < nspans; ++s)
                            madvise_span(sp[s], MADV_DONTNEED);
                        lru[i] = 0;
                    }
                }
            }
        }
        if (!did_work) std::this_thread::sleep_for(microseconds(50));
    }
}

} // namespace blade

