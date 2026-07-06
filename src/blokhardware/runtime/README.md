# Runtime Invariants

- Kernel first.
- Router early.
- Group tokens by expert before expert GEMM.
- Stream cold experts into registered GPU slabs.
- Never reload all active weights per token if cache can avoid it.
