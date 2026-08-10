// Streamer: background prefetch of routed-expert pages AND per-layer dense
// weights for the *next* layer while compute runs on the current one.
//
// LLM inference is fully deterministic given the prompt; for a single token
// we cannot know which experts the router will pick more than one block ahead.
// What we CAN do:
//  (a) the moment the router decides experts for layer L, issue
//      madvise(MADV_WILLNEED) on those experts' byte ranges so the kernel
//      starts faulting them in concurrently with the GPU's attention work for
//      layer L, hiding most of the SSD latency.
//  (b) maintain an LRU eviction list and madvise(MADV_DONTNEED) cold experts
//      to keep RAM pressure bounded.
//  (c) at the start of block(L), issue MADV_WILLNEED on every dense tensor of
//      layer L+1 (attn_norm, ffn_norm, all MLA matrices, dense FFN OR shared
//      FFN + router).  This is the FlexGen / TPI-LLM "sliding window memory
//      scheduler" -- a deterministic next-layer prefetch that perfectly
//      overlaps NVMe sequential reads with current-layer GPU compute.  Loosely
//      mirrors LLM-in-a-Flash row-column bundling: each tensor is a single
//      contiguous span, so the kernel's readahead issues one large request
//      aligned to flash erasure blocks.
//
// This is deliberately a tight, single-thread design: one worker that pulls
// (layer,expert) AND (layer-prefetch) jobs off a lock-free SPSC ring posted
// by the runtime.
#pragma once
#include "model.hpp"
#include <atomic>
#include <thread>
#include <cstdint>

namespace blade {

class Streamer {
public:
    void init(Model& m);
    void shutdown();
    // Post a routed-expert prefetch request. Non-blocking. moe_layer is
    // layer - n_dense_layers.
    void prefetch(uint32_t moe_layer, uint32_t expert);
    // Touch an expert: bump its LRU position. Called when actually used.
    void touch  (uint32_t moe_layer, uint32_t expert);
    // Sliding-window prefetch: ask the worker to MADV_WILLNEED every dense
    // tensor of `layer` (norms, MLA matrices, FFN/shared/router) so the pages
    // are streaming in by the time the runtime advances to that layer.
    // Idempotent and cheap: the kernel coalesces consecutive WILLNEED hints.
    void prefetch_layer(uint32_t layer);

private:
    Model* M = nullptr;
    static constexpr uint32_t RING = 1024;
    // Job tag: kind 0 = (moe_layer, expert), kind 1 = (layer, IGNORED).
    struct Job { uint32_t kind; uint32_t a; uint32_t b; };
    Job             ring[RING];
    std::atomic<uint32_t> head{0}, tail{0};
    std::thread     worker;
    std::atomic<bool> stop{false};

    // LRU: array of timestamps per (moe_layer, expert). Worker periodically
    // walks the array and DONTNEEDs the oldest experts past a budget.
    uint64_t* lru = nullptr;
    std::atomic<uint64_t> clock{0};
    size_t total_experts = 0;
    size_t per_expert_bytes = 0;
    // Expert-LRU byte budget. Computed in init() from mem::available_bytes()
    // per the grounding-doc rule:
    //     budget = min(10 GiB, max(512 MiB, avail - 4 GiB))
    // 4 GiB headroom covers activations (~2 GiB), KV cache (~2 GiB), and the
    // hot-band weights we mlock at startup.  A constant here was a bug:
    // on a 24 GB box already 22 GB-resident, 10 GiB would silently swap.
    size_t budget_bytes  = 0;

    void run();
    // Internal: walk a layer's hot tensors and madvise(WILLNEED) each span.
    void willneed_layer(uint32_t layer);
};

} // namespace blade

