#!/usr/bin/env python3
"""Replay exact route traces through candidate per-layer LRU capacities."""

import argparse
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

ROUTE = re.compile(r"\bpos=(\d+).*\blayer=(\d+).*\bexperts=([0-9,]+)")
TRACE = re.compile(
    r"trace router pos=(\d+) layer=(\d+) rank=(\d+) expert=(\d+)")
PREDICTOR = re.compile(
    r"predictor .*?(?:lookahead_layers=(\d+) .*?)?overlap=(\d+)/8 "
    r"predict_wall_us=(\d+) predict_gpu_us=(\d+) verify_lead_us=(\d+)")
PREDICTOR_DEADLINE = re.compile(
    r"predictor-deadline .*?(?:lookahead_layers=(\d+) )?expert_use_lead_us=(\d+)")
PREDICTOR_SETS = re.compile(
    r"predictor pos=(\d+) layer=(\d+).*?lookahead_layers=(\d+) .*?"
    r"predicted=([0-9,]+) exact=([0-9,]+)")
MODEL_GB = 4.035
NVME_GB_S = 6.388
GPU_S = 0.265
GB_PER_WAY = 2.018 / 4


Route = tuple[int, int, tuple[int, ...]]


def replay(routes: list[Route], ways: int) -> int:
    cache: dict[int, dict[int, int]] = {}
    clock = hits = 0
    for _, layer, selected in routes:
        entries = cache.setdefault(layer, {})
        pinned = {expert for expert in selected if expert in entries}
        hits += len(pinned)
        for expert in selected:
            clock += 1
            if expert in entries:
                entries[expert] = clock
                continue
            candidates = [item for item in entries if item not in pinned]
            if len(entries) >= ways and not candidates:
                continue
            if len(entries) >= ways:
                del entries[min(candidates, key=entries.get)]
            entries[expert] = clock
            pinned.add(expert)
    return hits


def oracle(routes: list[Route], ways: int) -> int:
    """Belady upper bound with current-token entries pinned until use."""
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for index, (_, layer, selected) in enumerate(routes):
        for expert in selected:
            future[layer, expert].append(index)
    cache: dict[int, set[int]] = defaultdict(set)
    hits = 0
    for index, (_, layer, selected) in enumerate(routes):
        entries = cache[layer]
        for expert in selected:
            assert future[layer, expert].popleft() == index
        pinned = entries.intersection(selected)
        hits += len(pinned)
        for expert in selected:
            if expert in entries:
                continue
            candidates = entries - pinned
            if len(entries) >= ways and not candidates:
                continue
            if len(entries) >= ways:
                def next_use(item: int) -> float:
                    queue = future[layer, item]
                    return queue[0] if queue else float("inf")
                entries.remove(max(candidates, key=next_use))
            entries.add(expert)
            pinned.add(expert)
    return hits


def joint_cache_predictor(
    routes: list[Route], predictions: dict[tuple[int, int], tuple[int, ...]],
    ways: int, topk: int,
) -> tuple[int, int, int, int, int]:
    """Replay the exact runtime LRU plus non-admitting speculative buffers.

    Returns cache hits, useful predicted misses, false predicted reads, late
    exact reads, and predictions skipped because the tensor was resident.
    Each count is in expert bundles (gate+up+down), whose mean byte cost is
    MODEL_GB / 8 per decode token.
    """
    cache: dict[int, dict[int, int]] = {}
    clock = cache_hits = useful = false = late = skipped = 0
    for pos, layer, selected in routes:
        entries = cache.setdefault(layer, {})
        resident = set(entries)
        candidates = tuple(expert for expert in predictions.get((pos, layer), ())[:topk]
                           if expert not in resident)
        selected_set = set(selected)
        cache_hits += len(selected_set & resident)
        useful += len(selected_set.intersection(candidates))
        false += len(set(candidates) - selected_set)
        late += len(selected_set - resident - set(candidates))
        skipped += topk - len(candidates)

        pinned = selected_set & resident
        for expert in selected:
            clock += 1
            if expert in entries:
                entries[expert] = clock
                continue
            victims = [item for item in entries if item not in pinned]
            if len(entries) >= ways and not victims:
                continue
            if len(entries) >= ways:
                del entries[min(victims, key=entries.get)]
            entries[expert] = clock
            pinned.add(expert)
    return cache_hits, useful, false, late, skipped


def predictor_scores(routes: list[Route]) -> list[tuple[str, int, int]]:
    previous: dict[int, tuple[int, ...]] = {}
    previous_hits = previous_total = 0
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    frequency_hits = frequency_total = 0
    all_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for _, layer, selected in routes:
        all_counts[layer].update(selected)
        if layer in previous:
            previous_hits += len(set(previous[layer]).intersection(selected))
            previous_total += 8
        if len(counts[layer]) >= 8:
            predicted = {item for item, _ in counts[layer].most_common(8)}
            frequency_hits += len(predicted.intersection(selected))
            frequency_total += 8
        counts[layer].update(selected)
        previous[layer] = selected
    static = {layer: {item for item, _ in count.most_common(8)}
              for layer, count in all_counts.items()}
    static_hits = sum(len(static[layer].intersection(selected))
                      for _, layer, selected in routes)
    results = [
        ("previous-token-8", previous_hits, previous_total),
        ("online-frequency-8", frequency_hits, frequency_total),
        ("offline-static-oracle-8", static_hits, len(routes) * 8),
    ]
    positions = sorted({pos for pos, _, _ in routes})
    if len(positions) >= 6:
        split = positions[max(1, int(len(positions) * .6)) - 1]

        def evaluate(decay: float, previous_bonus: float,
                     first: int, last: int) -> tuple[int, int]:
            scores: dict[int, dict[int, float]] = defaultdict(dict)
            prior: dict[int, tuple[int, ...]] = {}
            hits = total = 0
            for pos, layer, selected in routes:
                layer_scores = scores[layer]
                ranked = sorted(layer_scores,
                                key=lambda expert: (-(layer_scores[expert] +
                                      previous_bonus * (expert in prior.get(layer, ()))),
                                                    expert))[:8]
                if first <= pos <= last and len(ranked) == 8:
                    hits += len(set(ranked).intersection(selected))
                    total += 8
                for expert in tuple(layer_scores):
                    layer_scores[expert] *= decay
                for expert in selected:
                    layer_scores[expert] = layer_scores.get(expert, 0.0) + 1.0
                prior[layer] = selected
            return hits, total

        candidates = [(decay, bonus)
                      for decay in (0.0, .25, .5, .75, .9, .97, 1.0)
                      for bonus in (0.0, .5, 1.0, 2.0, 4.0, 8.0)]
        best = max(candidates,
                   key=lambda candidate: ((lambda score: score[0] / score[1]
                                            if score[1] else 0)
                                           (evaluate(*candidate, positions[0], split))))
        hits, total = evaluate(*best, split + 1, positions[-1])
        results.append((f"tuned-decay-{best[0]:g}-bonus-{best[1]:g}", hits, total))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    routes: list[Route] = []
    traced: dict[tuple[int, int], list[int | None]] = {}
    predictor_samples: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    predictor_deadlines: dict[int, list[int]] = defaultdict(list)
    predictor_sets: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(list)
    predictor_maps: dict[int, dict[tuple[int, int], tuple[int, ...]]] = defaultdict(dict)
    for line in args.log.read_text(errors="replace").splitlines():
        if match := PREDICTOR.search(line):
            horizon, *values = match.groups()
            predictor_samples[int(horizon or 0)].append(tuple(map(int, values)))
        if match := PREDICTOR_DEADLINE.search(line):
            predictor_deadlines[int(match.group(1) or 0)].append(int(match.group(2)))
        if match := PREDICTOR_SETS.search(line):
            pos, layer, horizon = map(int, match.group(1, 2, 3))
            predicted = tuple(map(int, match.group(4).split(",")))
            exact = tuple(map(int, match.group(5).split(",")))
            predictor_sets[horizon].append((predicted, exact))
            predictor_maps[horizon][pos, layer] = predicted
        if match := ROUTE.search(line):
            experts = tuple(map(int, match.group(3).split(",")))
            if len(experts) == 8:
                routes.append((int(match.group(1)), int(match.group(2)), experts))
        elif match := TRACE.search(line):
            key = int(match.group(1)), int(match.group(2))
            rank = int(match.group(3))
            slots = traced.setdefault(key, [None] * 8)
            if rank < 8:
                slots[rank] = int(match.group(4))
    if not routes:
        routes = [(pos, layer, tuple(experts))
                  for (pos, layer), experts in traced.items()
                  if all(expert is not None for expert in experts)]
    if not routes:
        raise SystemExit("no profiled experts= routes found")
    total = len(routes) * 8
    distinct = len({(layer, expert)
                    for _, layer, selected in routes for expert in selected})
    print(f"positions={len({pos for pos, _, _ in routes})} "
          f"layer_routes={len(routes)} selections={total} "
          f"compulsory={distinct} passive_max_hit={100*(1-distinct/total):.2f}%")
    print("ways cache_GB LRU_hit oracle_hit SSD_GB/token ideal_ms ideal_token/s")
    for ways in (0, 4, 8, 12, 16, 24, 32, 256):
        hits = replay(routes, ways) if ways else 0
        hit_rate = hits / total
        oracle_rate = oracle(routes, ways) / total if ways else 0
        ssd_gb = MODEL_GB * (1 - hit_rate)
        floor = max(GPU_S, ssd_gb / NVME_GB_S)
        print(f"{ways:>4} {ways * GB_PER_WAY:>8.3f} {100*hit_rate:>7.2f}% "
              f"{100*oracle_rate:>9.2f}% {ssd_gb:>12.3f} "
              f"{1000*floor:>8.1f} {1/floor:>13.3f}")

    print("\nhit_pct SSD_GB/token IO_floor_ms wall_floor_ms token/s")
    for hit_rate in (0, .581, .683, .808, .885, .95, 1):
        ssd_gb = MODEL_GB * (1 - hit_rate)
        io_floor = ssd_gb / NVME_GB_S
        floor = max(GPU_S, io_floor)
        print(f"{100*hit_rate:>7.1f} {ssd_gb:>12.3f} {1000*io_floor:>11.1f} "
              f"{1000*floor:>13.1f} {1/floor:>7.3f}")

    print("\npredictor evaluated recall precision false_GB late_miss_GB total_GB")
    for name, hits, evaluated in predictor_scores(routes):
        rate = hits / evaluated if evaluated else 0
        false_gb = MODEL_GB * (1 - rate)
        print(f"{name:<31} {evaluated:>9} {100*rate:>6.2f}% "
              f"{100*rate:>8.2f}% {false_gb:>8.3f} {false_gb:>12.3f} "
              f"{MODEL_GB + false_gb:>8.3f}")
    if predictor_samples:
        for horizon, samples in sorted(predictor_samples.items()):
            hits = sum(sample[0] for sample in samples)
            count = len(samples)
            wall = sum(sample[1] for sample in samples) / count
            gpu = sum(sample[2] for sample in samples) / count
            verify_lead = sum(sample[3] for sample in samples) / count
            deadlines = predictor_deadlines[horizon]
            use_lead = sum(deadlines) / len(deadlines) if deadlines else 0
            rate = hits / (count * 8)
            false_gb = MODEL_GB * (1 - rate)
            name = f"pre-attention-state-h{horizon}-8"
            print(f"{name:<31} {count * 8:>9} {100*rate:>6.2f}% "
                  f"{100*rate:>8.2f}% {false_gb:>8.3f} {false_gb:>12.3f} "
                  f"{MODEL_GB + false_gb:>8.3f} wall_us={wall:.1f} "
                  f"gpu_us={gpu:.1f} verify_lead_us={verify_lead:.1f} "
                  f"expert_use_lead_us={use_lead:.1f}")
    if predictor_sets:
        print("\nstate predictor horizon topk precision recall useful_early_GB "
              "false_GB late_miss_GB total_GB max_hidden_ms")
        for horizon, samples in sorted(predictor_sets.items()):
            for topk in range(1, 9):
                hits = sum(len(set(predicted[:topk]).intersection(exact))
                           for predicted, exact in samples)
                count = len(samples)
                precision = hits / (count * topk)
                recall = hits / (count * 8)
                predicted_gb = MODEL_GB * topk / 8
                useful_gb = MODEL_GB * hits / (count * 8)
                false_gb = MODEL_GB * (count * topk - hits) / (count * 8)
                late_gb = MODEL_GB * (count * 8 - hits) / (count * 8)
                total_gb = predicted_gb + late_gb
                print(f"{horizon:>23} {topk:>4} {100*precision:>8.2f}% "
                      f"{100*recall:>6.2f}% {useful_gb:>15.3f} "
                      f"{false_gb:>8.3f} {late_gb:>12.3f} {total_gb:>8.3f} "
                      f"{1000*useful_gb/NVME_GB_S:>13.1f}")

        print("\njoint exact-LRU + predictor (expert bundles; prediction does not "
              "change cache state)")
        print("horizon ways topk coverage cache_hit pred_hit pred_precision "
              "false_GB late_GB total_GB IO_floor_ms")
        unit_gb = MODEL_GB / 8
        for horizon, predictions in sorted(predictor_maps.items()):
            for ways in (4, 8, 16, 32):
                for topk in range(1, 9):
                    cache_hits, useful, false, late, skipped = joint_cache_predictor(
                        routes, predictions, ways, topk)
                    evaluated = len(routes) * topk - skipped
                    covered = cache_hits + useful
                    total_gb = (useful + false + late) * unit_gb / len(routes)
                    print(
                        f"{horizon:>7} {ways:>4} {topk:>4} "
                        f"{100*covered/total:>7.2f}% {100*cache_hits/total:>8.2f}% "
                        f"{100*useful/total:>7.2f}% "
                        f"{100*useful/evaluated if evaluated else 0:>13.2f}% "
                        f"{false*unit_gb/len(routes):>8.3f} "
                        f"{late*unit_gb/len(routes):>7.3f} {total_gb:>8.3f} "
                        f"{1000*total_gb/NVME_GB_S:>11.1f}"
                    )


if __name__ == "__main__":
    main()
