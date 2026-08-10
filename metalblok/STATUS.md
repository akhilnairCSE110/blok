# MetalBlok status

Completion requires a real prompt to generate repeatable text from the exact
three-shard DeepSeek-R1 checkpoint without crossing the memory ledger.

| Gate | Status | Evidence |
|---|---|---|
| Native build and runtime Metal compilation | In progress | — |
| Non-hydrating shard preflight | In progress | — |
| All three model shards resident and verified | Blocked: shard 3 is dataless | — |
| Six real GGUF quant types pass CPU/Metal parity | Pending | — |
| Exact grouped DeepSeek router passes parity | Pending | — |
| One real generated token | Pending | — |
| Five repeatable coherent tokens | Pending | — |
| 32-token safety soak | Pending | — |

No component-level success may change the project status to complete.
