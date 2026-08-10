#!/usr/bin/env python3
"""Recompute the integer claims in the Blok architecture notes.

This script produces derived evidence.  It does not run the CUDA executor or
measure the SSD.  Keeping that distinction in the output is intentional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def align(n: int, a: int) -> int:
    return ceil_div(n, a) * a


def decimal_ratio(n: int, d: int, places: int = 6) -> str:
    return f"{n / d:.{places}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s4(rows: int, cols: int) -> int:
    """Packed signed INT4 plus one BF16 scale per group of 32 weights."""
    assert cols % 32 == 0
    return rows * cols // 2 + 2 * rows * (cols // 32)


def s8(rows: int, cols: int) -> int:
    """E4M3 bytes plus one F32 inverse scale per 128 by 128 block."""
    return rows * cols + 4 * ceil_div(rows, 128) * ceil_div(cols, 128)


def kimi_claims() -> dict[str, object]:
    layers, moe_layers = 61, 60
    d, heads = 7168, 64
    q_nope, q_rope, v_head = 128, 64, 128
    q_rank, kv_rank = 1536, 512
    dense, expert_width = 18432, 2048
    experts, selected, vocab = 384, 8, 163840
    checkpoint = 595_148_192_736

    projection = s4(expert_width, d)
    assert projection == s4(d, expert_width) == 8_257_536
    expert_record = 3 * projection
    expert_bf16 = 3 * d * expert_width * 2
    routed_bank = moe_layers * experts * expert_record
    routed_selected = moe_layers * selected * expert_record

    attention_layer = 2 * (
        q_rank * d
        + q_rank
        + heads * (q_nope + q_rope) * q_rank
        + (kv_rank + q_rope) * d
        + kv_rank
        + heads * (q_nope + v_head) * kv_rank
        + d * heads * v_head
        + 2 * d
    )
    attention = layers * attention_layer
    dense_bytes = 3 * d * dense * 2
    router_layer = (experts * d + experts) * 2
    shared_layer = 3 * d * expert_width * 2
    fixed = attention + dense_bytes + moe_layers * (router_layer + shared_layer)
    head = vocab * d * 2
    embedding_row = d * 2
    cold_token = fixed + routed_selected + head + embedding_row

    weight_elements_attention = (
        q_rank * d
        + heads * (q_nope + q_rope) * q_rank
        + (kv_rank + q_rope) * d
        + heads * (q_nope + v_head) * kv_rank
        + d * heads * v_head
    )
    weight_elements = (
        layers * weight_elements_attention
        + 3 * d * dense
        + moe_layers * (experts * d + 3 * d * expert_width + selected * 3 * d * expert_width)
        + vocab * d
    )
    weight_flops = 2 * weight_elements

    expanded_kv_token = layers * (
        heads * (q_nope + q_rope) + heads * v_head
    ) * 4
    latent_kv_token = layers * (kv_rank + q_rope) * 2

    attention_calls_layer = (
        ceil_div(q_rank, 64)
        + ceil_div(heads * (q_nope + q_rope), 64)
        + ceil_div(kv_rank + q_rope, 64)
        + ceil_div(heads * (q_nope + v_head), 64)
        + ceil_div(d, 64)
        + 4
    )
    dense_calls = 2 * ceil_div(dense, 64) + ceil_div(d, 64)
    expert_calls = (
        4 * ceil_div(expert_width, 64)
        + 2 * ceil_div(d, 64)
    )
    shared_calls = 2 * ceil_div(expert_width, 64) + ceil_div(d, 64)
    moe_calls_layer = ceil_div(experts, 64) + 1 + selected * expert_calls + shared_calls
    transformer_calls = layers * attention_calls_layer + dense_calls + moe_layers * moe_calls_layer
    application_calls = transformer_calls + 1 + ceil_div(vocab, 256)

    result = {
        "architecture": {
            "layers": layers,
            "moe_layers": moe_layers,
            "hidden": d,
            "experts_per_layer": experts,
            "selected_per_token": selected,
        },
        "checkpoint_tensor_bytes": checkpoint,
        "expert_record_bytes": expert_record,
        "expert_bf16_equivalent_bytes": expert_bf16,
        "expert_compression_ratio": decimal_ratio(expert_bf16, expert_record),
        "routed_bank_bytes": routed_bank,
        "selected_routed_bytes_per_token": routed_selected,
        "routed_selection_reduction": str(Fraction(experts, selected)),
        "attention_and_norm_bytes_per_token": attention,
        "fixed_weight_bytes_per_token": fixed,
        "lm_head_bytes": head,
        "cold_weight_bytes_per_sampled_token": cold_token,
        "checkpoint_to_cold_step_ratio": decimal_ratio(checkpoint, cold_token),
        "weight_flops_per_token": weight_flops,
        "weight_flops_per_byte": decimal_ratio(weight_flops, cold_token),
        "expanded_kv_bytes_per_sequence_token": expanded_kv_token,
        "latent_kv_bytes_per_sequence_token": latent_kv_token,
        "kv_reduction_ratio": decimal_ratio(expanded_kv_token, latent_kv_token),
        "expanded_kv_bytes_at_max_context": expanded_kv_token * 262_144,
        "latent_kv_bytes_at_max_context": latent_kv_token * 262_144,
        "current_application_model_reads_per_sampled_token": application_calls,
        "current_model_allocations_per_sampled_token": transformer_calls + 2,
        "current_expert_application_reads_per_expert": expert_calls,
        "record_descriptors_per_cold_token": moe_layers * selected,
    }
    assert result["expert_record_bytes"] == 24_772_608
    assert result["routed_bank_bytes"] == 570_760_888_320
    assert result["selected_routed_bytes_per_token"] == 11_890_851_840
    assert result["attention_and_norm_bytes_per_token"] == 12_338_888_704
    assert result["fixed_weight_bytes_per_token"] == 18_746_782_720
    assert result["cold_weight_bytes_per_sampled_token"] == 32_986_459_136
    assert result["weight_flops_per_token"] == 63_372_132_352
    assert result["expanded_kv_bytes_per_sequence_token"] == 4_997_120
    assert result["latent_kv_bytes_per_sequence_token"] == 70_272
    assert result["current_application_model_reads_per_sampled_token"] == 217_686
    assert result["current_model_allocations_per_sampled_token"] == 217_047
    assert result["current_expert_application_reads_per_expert"] == 352
    return result


def glm_claims() -> dict[str, object]:
    layers, moe_layers, full_indexers = 78, 75, 21
    d, heads = 6144, 64
    q_nope, q_rope, v_head = 192, 64, 256
    q_rank, kv_rank = 2048, 512
    dense, expert_width = 12288, 2048
    experts, selected, vocab = 256, 8, 154880
    index_heads, index_dim = 32, 128
    checkpoint = 755_617_140_416

    projection = s8(expert_width, d)
    assert projection == s8(d, expert_width) == 12_585_984
    expert_record = 3 * projection
    expert_stride = align(expert_record, 4096)
    expert_bf16 = 3 * d * expert_width * 2
    routed_bank = moe_layers * experts * expert_record
    routed_selected = moe_layers * selected * expert_record

    attention_layer = (
        s8(q_rank, d)
        + s8(heads * (q_nope + q_rope), q_rank)
        + s8(kv_rank + q_rope, d)
        + s8(heads * (q_nope + v_head), kv_rank)
        + s8(d, heads * v_head)
        + 2 * (q_rank + kv_rank + 2 * d)
    )
    attention = layers * attention_layer
    indexer = (
        s8(index_heads * index_dim, q_rank)
        + s8(index_dim, d)
        + 2 * index_heads * d
        + 2 * 2 * index_dim
    )
    indexers = full_indexers * indexer
    dense_layer = 2 * s8(dense, d) + s8(d, dense)
    router_layer = (experts * d + experts) * 2
    shared_layer = expert_record
    fixed = attention + indexers + 3 * dense_layer + moe_layers * (router_layer + shared_layer)
    head = vocab * d * 2
    embedding_row = d * 2
    cold_token = fixed + routed_selected + head + embedding_row

    attention_weight_elements = (
        q_rank * d
        + heads * (q_nope + q_rope) * q_rank
        + (kv_rank + q_rope) * d
        + heads * (q_nope + v_head) * kv_rank
        + d * heads * v_head
    )
    index_weight_elements = index_heads * index_dim * q_rank + index_dim * d + index_heads * d
    weight_elements = (
        layers * attention_weight_elements
        + full_indexers * index_weight_elements
        + 3 * (3 * d * dense)
        + moe_layers * (experts * d + 3 * d * expert_width + selected * 3 * d * expert_width)
        + vocab * d
    )
    weight_flops = 2 * weight_elements

    expanded_kv_token = layers * heads * ((q_nope + q_rope) + v_head) * 2
    latent_kv_token = layers * (kv_rank + q_rope) * 2
    index_key_token = full_indexers * index_dim * 2
    cache_token = latent_kv_token + index_key_token
    max_context = 1_048_576

    result = {
        "architecture": {
            "layers": layers,
            "moe_layers": moe_layers,
            "hidden": d,
            "experts_per_layer": experts,
            "selected_per_token": selected,
            "full_indexers": full_indexers,
        },
        "checkpoint_tensor_bytes": checkpoint,
        "expert_record_bytes": expert_record,
        "expert_record_stride_4k": expert_stride,
        "expert_record_padding_bytes": expert_stride - expert_record,
        "all_expert_record_padding_bytes": (expert_stride - expert_record) * moe_layers * experts,
        "expert_bf16_equivalent_bytes": expert_bf16,
        "expert_compression_ratio": decimal_ratio(expert_bf16, expert_record),
        "routed_bank_bytes": routed_bank,
        "selected_routed_bytes_per_token": routed_selected,
        "routed_selection_reduction": str(Fraction(experts, selected)),
        "attention_bytes_per_token": attention,
        "indexer_weight_bytes": indexers,
        "fixed_weight_bytes_per_token": fixed,
        "lm_head_bytes": head,
        "cold_weight_bytes_per_sampled_token": cold_token,
        "checkpoint_to_cold_step_ratio": decimal_ratio(checkpoint, cold_token),
        "weight_flops_per_token": weight_flops,
        "weight_flops_per_byte": decimal_ratio(weight_flops, cold_token),
        "expanded_kv_bytes_per_sequence_token": expanded_kv_token,
        "latent_kv_bytes_per_sequence_token": latent_kv_token,
        "index_key_bytes_per_sequence_token": index_key_token,
        "latent_plus_index_bytes_per_sequence_token": cache_token,
        "kv_reduction_ratio": decimal_ratio(expanded_kv_token, latent_kv_token),
        "latent_plus_index_bytes_at_max_context": cache_token * max_context,
        "model_plus_cache_bytes_at_max_context": checkpoint + cache_token * max_context,
    }
    assert result["expert_record_bytes"] == 37_757_952
    assert result["expert_record_stride_4k"] == 37_761_024
    assert result["all_expert_record_padding_bytes"] == 58_982_400
    assert result["routed_bank_bytes"] == 724_952_678_400
    assert result["selected_routed_bytes_per_token"] == 22_654_771_200
    assert result["attention_bytes_per_token"] == 12_876_998_784
    assert result["indexer_weight_bytes"] == 200_991_168
    assert result["fixed_weight_bytes_per_token"] == 16_825_447_488
    assert result["cold_weight_bytes_per_sampled_token"] == 41_383_396_416
    assert result["weight_flops_per_token"] == 80_595_517_440
    assert result["expanded_kv_bytes_per_sequence_token"] == 5_111_808
    assert result["latent_kv_bytes_per_sequence_token"] == 89_856
    assert result["index_key_bytes_per_sequence_token"] == 5_376
    assert result["latent_plus_index_bytes_at_max_context"] == 99_857_989_632
    return result


def io_claims(kimi: dict[str, object], glm: dict[str, object]) -> dict[str, object]:
    page = 4096
    prp_command_cap = (page // 8 + 1) * page
    fallback = 128 * 1024
    target_bandwidth = 7_150_000_000
    queue = {}
    for size in (4096, 8192, 128 * 1024, 1024 * 1024, prp_command_cap):
        queue[str(size)] = {
            str(latency_us): ceil_div(target_bandwidth * latency_us, size * 1_000_000)
            for latency_us in (5, 10, 50, 100)
        }
    k_stride = int(kimi["expert_record_bytes"])
    g_stride = int(glm["expert_record_stride_4k"])
    return {
        "evidence_class": "derived from uGDS source constants; target MDTS is unmeasured",
        "controller_page_bytes_assumption": page,
        "single_prp_list_command_cap_bytes": prp_command_cap,
        "ugds_no_mdts_fallback_bytes": fallback,
        "ugds_batch_application_entry_capacity": 128,
        "ugds_batch_queue_depth_ceiling": 512,
        "ugds_prp_list_pages": 64,
        "commands_per_kimi_expert_at_prp_cap": ceil_div(k_stride, prp_command_cap),
        "commands_per_kimi_expert_at_fallback": ceil_div(k_stride, fallback),
        "commands_per_glm_expert_at_prp_cap": ceil_div(g_stride, prp_command_cap),
        "commands_per_glm_expert_at_fallback": ceil_div(g_stride, fallback),
        "commands_for_eight_kimi_experts_at_prp_cap": 8 * ceil_div(k_stride, prp_command_cap),
        "commands_for_eight_glm_experts_at_prp_cap": 8 * ceil_div(g_stride, prp_command_cap),
        "glm_selected_expert_commands_at_prp_cap": 600 * ceil_div(g_stride, prp_command_cap),
        "glm_selected_expert_commands_at_fallback": 600 * ceil_div(g_stride, fallback),
        "queue_depth_for_7_15GBps_by_size_and_latency_us": queue,
    }


def derive() -> dict[str, object]:
    kimi = kimi_claims()
    glm = glm_claims()
    sources = [
        ROOT / "src/kimi_exec.cu",
        ROOT / "blok/runtime.py",
        ROOT / "scripts/model_fetch.py",
        ROOT / "sub_dir/uGDS/src/ugds_batch.cpp",
        ROOT / "sub_dir/uGDS/src/ugds_internal.h",
    ]
    return {
        "schema": "blok-derived-architecture-evidence-v1",
        "evidence_class": "deterministic derivation and source audit; not a hardware benchmark",
        "reproduce": "python3 -m scripts.derive_architecture_claims --check",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "kimi_k2_6": kimi,
        "glm_5_2_fp8": glm,
        "io": io_claims(kimi, glm),
    }


def markdown(report: dict[str, object]) -> str:
    k = report["kimi_k2_6"]
    g = report["glm_5_2_fp8"]
    io = report["io"]
    rows = [
        ("Checkpoint tensor bytes", k["checkpoint_tensor_bytes"], g["checkpoint_tensor_bytes"]),
        ("Cold sampled-token weight bytes", k["cold_weight_bytes_per_sampled_token"], g["cold_weight_bytes_per_sampled_token"]),
        ("Routed bank bytes", k["routed_bank_bytes"], g["routed_bank_bytes"]),
        ("Selected routed bytes/token", k["selected_routed_bytes_per_token"], g["selected_routed_bytes_per_token"]),
        ("Expert record bytes", k["expert_record_bytes"], g["expert_record_bytes"]),
        ("Weight FLOPs/token", k["weight_flops_per_token"], g["weight_flops_per_token"]),
        ("Expanded KV bytes/sequence token", k["expanded_kv_bytes_per_sequence_token"], g["expanded_kv_bytes_per_sequence_token"]),
        ("Latent KV bytes/sequence token", k["latent_kv_bytes_per_sequence_token"], g["latent_kv_bytes_per_sequence_token"]),
    ]
    out = [
        "# Derived Architecture Evidence",
        "",
        "**Evidence class:** deterministic integer derivation and source audit. These values are not measured latency or bandwidth.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "python3 -m scripts.derive_architecture_claims --check",
        "```",
        "",
        "## Model results",
        "",
        "| Quantity | Kimi K2.6 | GLM-5.2 FP8 |",
        "|---|---:|---:|",
    ]
    out.extend(f"| {name} | {a:,} | {b:,} |" for name, a, b in rows)
    out += [
        "",
        "## Selection and representation ratios",
        "",
        f"- Kimi routed-bank reduction: {k['routed_selection_reduction']}; expert compression: {k['expert_compression_ratio']}×; KV reduction: {k['kv_reduction_ratio']}×.",
        f"- GLM routed-bank reduction: {g['routed_selection_reduction']}; expert compression: {g['expert_compression_ratio']}×; KV reduction: {g['kv_reduction_ratio']}×.",
        "",
        "## uGDS command split",
        "",
        f"With a 4 KiB controller page, the current one-PRP-list cap is {io['single_prp_list_command_cap_bytes']:,} bytes. This is a source-derived implementation limit, not a measured target MDTS.",
        "",
        "| Record | Commands at PRP cap | Commands at 128 KiB fallback |",
        "|---|---:|---:|",
        f"| Kimi expert | {io['commands_per_kimi_expert_at_prp_cap']:,} | {io['commands_per_kimi_expert_at_fallback']:,} |",
        f"| GLM expert | {io['commands_per_glm_expert_at_prp_cap']:,} | {io['commands_per_glm_expert_at_fallback']:,} |",
        "",
        "## Source identities",
        "",
        "| Source | SHA-256 |",
        "|---|---|",
    ]
    out.extend(f"| `{path}` | `{digest}` |" for path, digest in report["source_sha256"].items())
    out += [
        "",
        "## What this file does not prove",
        "",
        "It does not prove GPU correctness, PCIe peer routing, SSD bandwidth, request latency, expert reuse, thermal stability, or end-to-end token output. Those require the target run described in `target-measurement-protocol.md`.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=ROOT / "docs/architecture/evidence/derived-claims.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/architecture/evidence/derived-claims.md")
    parser.add_argument("--check", action="store_true", help="fail if committed generated evidence is stale")
    args = parser.parse_args()
    report = derive()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)

    if args.check:
        stale = []
        for path, expected in ((args.json, json_text), (args.markdown, markdown_text)):
            if not path.is_file() or path.read_text() != expected:
                stale.append(str(path))
        if stale:
            raise SystemExit("stale derived evidence: " + ", ".join(stale))
        print("ok: derived architecture evidence matches sources")
        return

    for path, value in ((args.json, json_text), (args.markdown, markdown_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
