#!/usr/bin/env python3
"""Apply frozen clean-boundary count probes to natural anomalous traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    count_probe_predictions,
    count_probe_scores,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_targeted_explicit_count_scrub_source_and_blank,
)
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    _marker_char_spans,
    build_marker_scrubbed_list_registry,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _model,
)


SCHEMA = "realistic_niah_v5_natural_anomaly_geometry_v1"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _parser(row: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed = row.get("trace_parse")
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("parser"), Mapping):
        raise ValueError("Natural anomaly assay requires frozen trace_parse.parser")
    return parsed["parser"]


def _eligible(
    row: Mapping[str, Any],
    excluded_seeds: set[int],
    included_seeds: set[int],
    counts: set[int],
    cohort: str,
) -> bool:
    if str(row.get("model_label", row.get("model", ""))) != "Qwen3-8B":
        return False
    if int(row.get("seed", -1)) in excluded_seeds:
        return False
    if included_seeds and int(row.get("seed", -1)) not in included_seeds:
        return False
    parser = _parser(row)
    gold_count = len(row.get("gold_records", row.get("gold_pairs", [])))
    if counts and gold_count not in counts:
        return False
    item_count = int(parser.get("item_count", 0) or 0)
    if not bool(parser.get("detected")) or not 1 <= item_count <= 10:
        return False
    anomaly = bool(
        int(parser.get("duplicate_gold_city_items", 0) or 0) > 0
        or parser.get("missing_gold_cities")
        or str(parser.get("trace_order_class", "")) not in ("", "forward")
        or not bool(row.get("trace_parse", {}).get("exact_count", True))
    )
    clean = bool(
        parser.get("trace_one_to_one")
        and item_count == gold_count
        and bool(row.get("trace_parse", {}).get("exact_count", False))
    )
    if cohort == "anomaly":
        return anomaly
    if cohort == "clean":
        return clean
    return anomaly or clean


def _trace_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    parser = _parser(row)
    raw = str(row.get("raw_output_text", ""))
    starts = [int(value) for value in parser.get("item_start_chars", [])]
    ends = [int(value) for value in parser.get("item_end_chars", [])]
    cities = [str(value) for value in parser.get("item_gold_cities", [])]
    if not starts or len(starts) != len(ends) or len(starts) != len(cities):
        raise ValueError("Frozen parser has inconsistent natural item spans")
    marker_spans = [
        _marker_char_spans(raw[start:end], offset=start)
        for start, end in zip(starts, ends, strict=True)
    ]
    gold_records = list(row.get("gold_records", row.get("gold_pairs", [])))
    gold_index = {
        str(record.get("city", "")).casefold(): index
        for index, record in enumerate(gold_records, start=1)
    }
    return {
        "schema_version": "natural_anomaly_frozen_parser_bridge_v1",
        "status": "PASS",
        "eligible": True,
        "reasons": [],
        "gold_count": len(gold_records),
        "parsed_item_count": len(starts),
        "marker_kind": str(parser.get("marker_kind", "")),
        "original_parser_marker_kind": str(parser.get("marker_kind", "")),
        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
        "list_episode_contiguous": True,
        "item_char_spans": [[start, end] for start, end in zip(starts, ends, strict=True)],
        "item_marker_char_spans": marker_spans,
        "item_line_numbers": [int(value) for value in parser.get("item_line_numbers", [])],
        "item_gold_indices": [gold_index.get(city.casefold(), -1) for city in cities],
        "item_gold_cities": cities,
        "item_text_sha256": [],
    }


def _load_probes(path: Path) -> tuple[tuple[int, ...], dict[int, dict[str, np.ndarray]]]:
    archive = np.load(path)
    layers = tuple(int(value) for value in archive["frozen_layers"].tolist())
    probes = {}
    for layer in layers:
        probe = {
            "mean": np.asarray(archive[f"layer_{layer}_mean"]),
            "weights": np.asarray(archive[f"layer_{layer}_weights"]),
        }
        if f"layer_{layer}_ordinal_weights" in archive.files:
            probe.update(
                {
                    "ordinal_mean": np.asarray(archive[f"layer_{layer}_ordinal_mean"]),
                    "ordinal_weights": np.asarray(archive[f"layer_{layer}_ordinal_weights"]),
                    "ordinal_intercept": float(archive[f"layer_{layer}_ordinal_intercept"][0]),
                }
            )
        probes[layer] = probe
    return layers, probes


def _ordinal_prediction(probe: Mapping[str, Any], state: np.ndarray) -> float | None:
    if "ordinal_weights" not in probe:
        return None
    centered = np.asarray(state, dtype=np.float64) - np.asarray(
        probe["ordinal_mean"], dtype=np.float64
    )
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    value = float(probe["ordinal_intercept"]) + float(
        centered[0] @ np.asarray(probe["ordinal_weights"], dtype=np.float64)
    )
    return value


def _soft_count(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    shifted = values - float(np.max(values))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    return float(probabilities @ np.arange(1, 11, dtype=np.float64))


def _final_count(row: Mapping[str, Any]) -> int | None:
    parsed = row.get("trace_parse", {})
    value = parsed.get("parsed_count") if isinstance(parsed, Mapping) else None
    if value is None:
        return None
    result = int(value)
    return result if 1 <= result <= 10 else None


def _label(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if 1 <= result <= 10 else None


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = (
        "raw_item_ordinal",
        "unique_city_count",
        "explicit_marker",
        "gold_city_rank",
        "final_answer_count",
        "gold_total",
    )
    layer_summary = []
    for layer in sorted({int(row["layer"]) for row in rows}):
        frame = [row for row in rows if int(row["layer"]) == layer]
        entry: dict[str, Any] = {"layer": layer, "sites": len(frame)}
        for label in labels:
            eligible = [row for row in frame if row.get(label) is not None]
            entry[f"{label}_sites"] = len(eligible)
            entry[f"{label}_exact_rate"] = (
                float(np.mean([int(row["probe_prediction"]) == int(row[label]) for row in eligible]))
                if eligible
                else None
            )
            ordinal_eligible = [
                row for row in eligible if row.get("probe_ordinal_coordinate") is not None
            ]
            entry[f"{label}_ordinal_mae"] = (
                float(
                    np.mean(
                        [
                            abs(float(row["probe_ordinal_coordinate"]) - int(row[label]))
                            for row in ordinal_eligible
                        ]
                    )
                )
                if ordinal_eligible
                else None
            )
        layer_summary.append(entry)

    comparisons = []
    pairs = (
        ("raw_item_ordinal", "unique_city_count"),
        ("raw_item_ordinal", "gold_city_rank"),
        ("raw_item_ordinal", "final_answer_count"),
        ("raw_item_ordinal", "gold_total"),
        ("unique_city_count", "final_answer_count"),
        ("unique_city_count", "gold_total"),
    )
    for layer in sorted({int(row["layer"]) for row in rows}):
        frame = [row for row in rows if int(row["layer"]) == layer]
        for left, right in pairs:
            discriminating = [
                row
                for row in frame
                if row.get(left) is not None
                and row.get(right) is not None
                and int(row[left]) != int(row[right])
            ]
            if not discriminating:
                continue
            left_hits = sum(int(row["probe_prediction"]) == int(row[left]) for row in discriminating)
            right_hits = sum(int(row["probe_prediction"]) == int(row[right]) for row in discriminating)
            other = len(discriminating) - left_hits - right_hits
            comparisons.append(
                {
                    "layer": layer,
                    "left_label": left,
                    "right_label": right,
                    "discriminating_sites": len(discriminating),
                    "unique_seeds": len({int(row["seed"]) for row in discriminating}),
                    "left_hits": left_hits,
                    "right_hits": right_hits,
                    "other_predictions": other,
                    "ordinal_left_closer": sum(
                        abs(float(row["probe_ordinal_coordinate"]) - int(row[left]))
                        < abs(float(row["probe_ordinal_coordinate"]) - int(row[right]))
                        for row in discriminating
                        if row.get("probe_ordinal_coordinate") is not None
                    ),
                    "ordinal_right_closer": sum(
                        abs(float(row["probe_ordinal_coordinate"]) - int(row[right]))
                        < abs(float(row["probe_ordinal_coordinate"]) - int(row[left]))
                        for row in discriminating
                        if row.get("probe_ordinal_coordinate") is not None
                    ),
                    "ordinal_ties": sum(
                        abs(float(row["probe_ordinal_coordinate"]) - int(row[right]))
                        == abs(float(row["probe_ordinal_coordinate"]) - int(row[left]))
                        for row in discriminating
                        if row.get("probe_ordinal_coordinate") is not None
                    ),
                }
            )

    event_summary = []
    for layer in sorted({int(row["layer"]) for row in rows}):
        frame = [row for row in rows if int(row["layer"]) == layer]
        event_frames = {
            "duplicate_sites": [row for row in frame if bool(row["is_duplicate_city"])],
            "final_boundary_missing_trace": [
                row
                for row in frame
                if bool(row["is_final_item"])
                and int(row["unique_city_count"]) < int(row["gold_total"])
            ],
            "nonforward_rank_sites": [
                row
                for row in frame
                if row.get("gold_city_rank") is not None
                and int(row["gold_city_rank"]) != int(row["raw_item_ordinal"])
            ],
        }
        for event, event_rows in event_frames.items():
            if not event_rows:
                continue
            event_summary.append(
                {
                    "layer": layer,
                    "event": event,
                    "sites": len(event_rows),
                    "unique_seeds": len({int(row["seed"]) for row in event_rows}),
                    "raw_ordinal_exact": float(np.mean([row["probe_prediction"] == row["raw_item_ordinal"] for row in event_rows])),
                    "unique_city_exact": float(np.mean([row["probe_prediction"] == row["unique_city_count"] for row in event_rows])),
                    "final_answer_exact": float(np.mean([
                        row.get("final_answer_count") is not None
                        and row["probe_prediction"] == row["final_answer_count"]
                        for row in event_rows
                    ])),
                    "gold_total_exact": float(np.mean([row["probe_prediction"] == row["gold_total"] for row in event_rows])),
                    "predictions": [int(row["probe_prediction"]) for row in event_rows],
                    "mean_ordinal_coordinate": float(
                        np.mean(
                            [
                                float(row["probe_ordinal_coordinate"])
                                for row in event_rows
                                if row.get("probe_ordinal_coordinate") is not None
                            ]
                        )
                    ),
                    "raw_ordinal_mae": float(
                        np.mean(
                            [
                                abs(float(row["probe_ordinal_coordinate"]) - int(row["raw_item_ordinal"]))
                                for row in event_rows
                                if row.get("probe_ordinal_coordinate") is not None
                            ]
                        )
                    ),
                    "unique_city_mae": float(
                        np.mean(
                            [
                                abs(float(row["probe_ordinal_coordinate"]) - int(row["unique_city_count"]))
                                for row in event_rows
                                if row.get("probe_ordinal_coordinate") is not None
                            ]
                        )
                    ),
                }
            )
    return {
        "layers": layer_summary,
        "pairwise_discriminating_comparisons": comparisons,
        "event_summary": event_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), default="Qwen3-8B")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--include-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--counts", type=int, nargs="*", default=[])
    parser.add_argument("--cohort", choices=("anomaly", "clean", "all"), default="anomaly")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.command = "natural-anomaly-geometry"

    layers, probes = _load_probes(args.frozen_probes)
    excluded = {int(value) for value in args.exclude_seeds}
    included = {int(value) for value in args.include_seeds}
    counts = {int(value) for value in args.counts}
    selected = [
        row
        for row in read_jsonl(args.generations)
        if _eligible(row, excluded, included, counts, str(args.cohort))
    ]
    if not selected:
        raise ValueError("No eligible held-out natural anomaly traces")
    model, tokenizer, adapter = _model(args)
    output: list[dict[str, Any]] = []
    cohort_audit: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        parser_row = _parser(row)
        trace_audit = _trace_audit(row)
        clean, registry, registry_audit = build_marker_scrubbed_list_registry(
            row, tokenizer, trace_audit=trace_audit
        )
        source, _blank, scrub_audit = build_targeted_explicit_count_scrub_source_and_blank(
            row,
            clean,
            registry,
            tokenizer,
            random_seed=20260830 + int(row["seed"]),
            marker_kind=str(trace_audit["marker_kind"]),
            mask_index_punctuation=False,
            first_item_char_override=int(trace_audit["item_char_spans"][0][0]),
        )
        positions = []
        position_audits = []
        for occurrence in range(1, len(registry.trace_items) + 1):
            position, audit = select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )
            positions.append(int(position))
            position_audits.append(audit)
        captures = capture_decoder_block_input_states(
            model, adapter, source, positions, layers=layers
        )

        cities = [str(value) for value in parser_row.get("item_gold_cities", [])]
        markers = list(parser_row.get("item_markers", []))
        gold_records = list(row.get("gold_records", row.get("gold_pairs", [])))
        gold_rank = {
            str(record.get("city", "")).casefold(): order
            for order, record in enumerate(gold_records, start=1)
        }
        seen: set[str] = set()
        final_count = _final_count(row)
        for site_index, (position, city) in enumerate(zip(positions, cities, strict=True), start=1):
            folded = city.casefold()
            duplicate = folded in seen
            seen.add(folded)
            labels = {
                "raw_item_ordinal": site_index,
                "unique_city_count": len(seen),
                "explicit_marker": _label(markers[site_index - 1]) if site_index <= len(markers) else None,
                "gold_city_rank": _label(gold_rank.get(folded)),
                "final_answer_count": final_count,
                "gold_total": len(gold_records),
            }
            for layer in layers:
                state = captures[layer][site_index - 1 : site_index].numpy()
                scores = count_probe_scores(probes[layer], state)[0]
                prediction = int(count_probe_predictions(probes[layer], state)[0])
                output.append(
                    {
                        "schema_version": SCHEMA,
                        "request_id": str(row["request_id"]),
                        "model_label": "Qwen3-8B",
                        "seed": int(row["seed"]),
                        "layer": int(layer),
                        "boundary_position": int(position),
                        "site_index": site_index,
                        "city": city,
                        "is_duplicate_city": duplicate,
                        "is_final_item": site_index == len(cities),
                        "probe_prediction": prediction,
                        "probe_soft_count": _soft_count(scores),
                        "probe_ordinal_coordinate": _ordinal_prediction(
                            probes[layer], state
                        ),
                        "probe_scores": [float(value) for value in scores],
                        **labels,
                    }
                )
        cohort_audit.append(
            {
                "request_id": str(row["request_id"]),
                "seed": int(row["seed"]),
                "gold_total": len(gold_records),
                "parsed_item_count": len(cities),
                "duplicate_gold_city_items": int(parser_row.get("duplicate_gold_city_items", 0) or 0),
                "missing_gold_cities": list(parser_row.get("missing_gold_cities", [])),
                "trace_order_class": str(parser_row.get("trace_order_class", "")),
                "final_answer_count": final_count,
                "trace_audit": trace_audit,
                "registry_audit": registry_audit,
                "scrub_audit": scrub_audit,
                "boundary_audits": position_audits,
            }
        )
        print(f"[natural-anomaly-geometry] {index}/{len(selected)} {row['request_id']}", flush=True)

    write_jsonl(args.output, output)
    summary = {
        "schema_version": SCHEMA,
        "model_label": "Qwen3-8B",
        "generations": str(args.generations),
        "frozen_probes": str(args.frozen_probes),
        "excluded_probe_seeds": sorted(excluded),
        "included_seeds": sorted(included),
        "requested_counts": sorted(counts),
        "cohort": str(args.cohort),
        "selected_requests": len(selected),
        "selected_seeds": sorted({int(row["seed"]) for row in selected}),
        "frozen_layers": list(layers),
        "construction": "targeted_explicit_count_scrub with stored natural anomalous list spans",
        "cohort_audit": cohort_audit,
        **_summarize(output),
        "interpretation_limit": "Natural anomalies break selected label equivalences but are observational; probe was trained only on clean N=10 boundaries.",
    }
    _atomic_json(args.output.with_suffix(".summary.json"), summary)
    print(f"[natural-anomaly-geometry] wrote {args.output}")


if __name__ == "__main__":
    main()
