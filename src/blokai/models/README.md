# Model Invariants

- Supported model: `moonshotai/Kimi-K2.6`.
- Text core: 61 layers, hidden 7168, 64 heads, MLA, 384 routed experts, top-8.
- Routers, attention, norms, shared experts are resident candidates.
- Routed experts are cache/stream candidates.
