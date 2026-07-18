# Kimi K2.6 Runtime Notes

Concise source-backed notes for Blok's text-only V0 executor.

## Official Shape And Deployment Context

- Model: `moonshotai/Kimi-K2.6`.
- Hugging Face model card: https://huggingface.co/moonshotai/Kimi-K2.6
- Current config: https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json
- The top-level architecture is `KimiK25ForConditionalGeneration`; the text tower uses `DeepseekV3ForCausalLM` with `model_type: kimi_k2`.
- Text summary from the model card: 61 layers, 1 dense layer, 64 attention heads, 384 experts, 8 selected experts per token, 1 shared expert, 7168 hidden size, 2048 per-expert MoE hidden size, 163840 vocab, 256K context, MLA attention.
- The model card recommends established engines first: vLLM, SGLang, and KTransformers. Blok's native path is therefore experimental until it passes token/logit parity.

## Config Values Blok Must Match

Source: https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json

- `hidden_size: 7168`
- `num_hidden_layers: 61`
- `first_k_dense_replace: 1`
- `num_attention_heads: 64`
- `q_lora_rank: 1536`
- `kv_lora_rank: 512`
- `qk_nope_head_dim: 128`
- `qk_rope_head_dim: 64`
- `v_head_dim: 128`
- `n_routed_experts: 384`
- `num_experts_per_tok: 8`
- `n_group: 1`
- `topk_group: 1`
- `topk_method: noaux_tc`
- `scoring_func: sigmoid`
- `norm_topk_prob: true`
- `routed_scaling_factor: 2.827`
- `rope_theta: 50000`
- `rope_scaling: yarn`, `factor: 64`, `original_max_position_embeddings: 4096`, `beta_fast: 32`, `beta_slow: 1`, `mscale: 1`, `mscale_all_dim: 1`

## Official Forward Semantics To Preserve

Reference DeepSeek/Kimi implementation: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/modeling_deepseek.py

- MLA uses `q_a_proj -> q_a_layernorm -> q_b_proj` and `kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj`.
- `kv_a_proj_with_mqa` outputs `kv_lora_rank + qk_rope_head_dim`; the last RoPE slice is shared across heads before rotation.
- RoPE applies DeepSeek YaRN frequencies. The rotary half is first reshaped from adjacent pairs into half/half layout before `rotate_half`.
- Attention scale starts at `q_head_dim ** -0.5`; when `mscale_all_dim` is set, it is multiplied by YaRN mscale squared.
- MoE routing for `noaux_tc` adds `e_score_correction_bias` for expert choice, selects top groups, gathers top-k experts from the uncorrected sigmoid score, normalizes top-k probabilities, then multiplies by `routed_scaling_factor`.
- For Kimi K2.6, `n_group=1` and `topk_group=1`, so group filtering collapses to global top-k but the corrected-vs-uncorrected score split still matters.

## Quantization Notes

- The official config describes `compressed-tensors`, `pack-quantized`, symmetric INT4, group size 32.
- The model card calls this "Native INT4 Quantization": https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/README.md
- LLM Compressor notes the original Kimi K2.6 checkpoint ships with 4-bit integer weights and can be dequantized by `CompressedTensorsDequantizer`: https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/kimi-k26/fp8-block-example/
- Blok's current routed expert contract assumes packed `I32` weights with BF16 scales and group size 32.

## FIEMAP/uGDS Constraint

- Linux FIEMAP docs: https://docs.kernel.org/filesystems/fiemap.html
- FIEMAP returns logical, physical, length, and flags in bytes.
- Extents with unknown, delayed, encoded, inline, tail, unwritten, merged, or unaligned flags are not valid for direct raw uGDS reads.
- Generate the map while the filesystem is mounted, then unmount and bind the NVMe namespace to uGDS before raw reads. Raw block access through FIEMAP extents while the filesystem remains mounted is explicitly unsafe.

## Actual Target Storage

- User-reported primary NVMe: Samsung 990 EVO Plus 1TB.
- Samsung 990 EVO Plus specs: https://www.samsung.com.cn/memory-storage/nvme-ssd/990-evo-plus-1tb-nvme-pcie-gen-4-mz-v9S1T0BW/
- Relevant Samsung specs: PCIe 4.0 x4 / 5.0 x2 NVMe 2.0, M.2 2280, 1TB, up to 7,150 MB/s sequential read and 6,300 MB/s sequential write, TLC, HMB.
- User-reported secondary SSD: Kingston SA400S37240G 240GB.
- Kingston SA400S37240G reference: https://smarthdd.com/database/KINGSTON-SA400S37240G/03200001/
- Relevant Kingston constraint: SATA SSD, not an NVMe/uGDS target.
- User-reported HDD: Seagate ST2000DM008-2FR102 2TB.
- Seagate desktop HDD family reference: https://www.seagate.com/gb/en/support/internal-hard-drives/desktop-hard-drives/desktop-hdd/
- Relevant Seagate constraint: SATA HDD, not an NVMe/uGDS target.
