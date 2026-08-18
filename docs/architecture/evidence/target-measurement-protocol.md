# Target measurement protocol

This protocol closes the evidence gap for the separate Linux/NVIDIA Kimi
track. It does not certify the active macOS MetalBlok runtime.

1. Record the commit, model manifest hash, kernel build, driver, CUDA version,
   GPU, RAM, NVMe devices, filesystem, and thermal/power configuration.
2. Run `scripts/check_hardware.sh`, then `scripts/check_kimi_contract.py` and
   retain their complete stdout and exit status.
3. Run the native one-token contract check from a cold process. Record the
   selected expert IDs, nonfinite count, token ID, logits, bytes read, GPU
   time, I/O wait, command count, allocations, peak memory, and power sample.
4. Repeat from the identical checkpoint. Acceptance requires identical token
   IDs and the numerical tolerance declared by the model contract; unexplained
   drift is failure.
5. Run at least 1,000 input and 1,000 output tokens. Preserve the prompt,
   decoded output, checkpoint, diagnostics, wall time, prefill/decode rates,
   effective NVMe bandwidth, KV bytes/token, peak memory, and joules/token
   when power telemetry is available.
6. Test interruption and exact continuation from the last committed state.
7. Label capacities that were only admitted or allocated as such. Do not call
   32K, 128K, or million-token execution proven without a completed run.

The release gate is correctness first, then lower end-to-end wall time from an
identical state. A faster result with different expert routing, logits, or
checkpoint bytes is rejected.
