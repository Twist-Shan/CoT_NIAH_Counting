#!/usr/bin/env python3
"""Fit count probes on clean natural formats with all anomaly seeds held out."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    _prepare_features,
    count_prediction_metrics,
    count_probe_predictions,
    fit_dual_ridge_count_probe,
    leave_one_seed_out_probe_metrics,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_targeted_explicit_count_scrub_source_and_blank,
)
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    build_marker_scrubbed_list_registry,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from scripts.run_realistic_niah_v5_count_stream import _atomic_json, _model  # noqa: E402
from scripts.run_realistic_niah_v5_natural_anomaly_geometry import (  # noqa: E402
    _parser,
    _trace_audit,
    read_jsonl,
)


SCHEMA = "realistic_niah_v5_format_robust_boundary_probe_v1"


def _fit_ordinal_probe(
    states: np.ndarray, labels: np.ndarray, *, alpha: float
) -> dict[str, Any]:
    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    z = _prepare_features(x, mean=mean)
    intercept = float(np.mean(y))
    centered = y - intercept
    dual = np.linalg.solve(
        z @ z.T + float(alpha) * np.eye(len(z), dtype=np.float64), centered
    )
    return {
        "mean": mean.astype(np.float32),
        "weights": (z.T @ dual).astype(np.float32),
        "intercept": intercept,
    }


def _ordinal_predictions(probe: Mapping[str, Any], states: np.ndarray) -> np.ndarray:
    z = _prepare_features(
        np.asarray(states, dtype=np.float64),
        mean=np.asarray(probe["mean"], dtype=np.float64),
    )
    return float(probe["intercept"]) + z @ np.asarray(probe["weights"], dtype=np.float64)


def _ordinal_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    residual = y - pred
    denominator = float(np.sum((y - float(np.mean(y))) ** 2))
    rounded = np.clip(np.rint(pred), 1, 10).astype(np.int64)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": float(1.0 - np.sum(residual**2) / denominator),
        "pearson": float(np.corrcoef(y, pred)[0, 1]),
        "rounded_exact_accuracy": float(np.mean(rounded == y.astype(np.int64))),
    }


def _ordinal_loso(
    states: np.ndarray,
    labels: np.ndarray,
    seed_ids: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    predictions = np.zeros(len(labels), dtype=np.float64)
    folds = []
    for seed in sorted(set(int(value) for value in seed_ids)):
        test = seed_ids == seed
        train = ~test
        probe = _fit_ordinal_probe(states[train], labels[train], alpha=alpha)
        predictions[test] = _ordinal_predictions(probe, states[test])
        folds.append({"seed": seed, **_ordinal_metrics(labels[test], predictions[test])})
    return {**_ordinal_metrics(labels, predictions), "folds": folds}


def _is_anomaly(row: Mapping[str, Any]) -> bool:
    parser = _parser(row)
    item_count = int(parser.get("item_count", 0) or 0)
    gold_count = len(row.get("gold_records", row.get("gold_pairs", [])))
    return bool(
        not bool(parser.get("detected"))
        or not bool(parser.get("trace_one_to_one"))
        or item_count != gold_count
        or int(parser.get("duplicate_gold_city_items", 0) or 0) > 0
        or parser.get("missing_gold_cities")
        or str(parser.get("trace_order_class", "")) not in ("", "forward")
        or not bool(row.get("trace_parse", {}).get("exact_count", False))
    )


def _is_clean(row: Mapping[str, Any]) -> bool:
    parser = _parser(row)
    gold_count = len(row.get("gold_records", row.get("gold_pairs", [])))
    return bool(
        parser.get("detected")
        and parser.get("trace_one_to_one")
        and int(parser.get("item_count", 0) or 0) == gold_count
        and 1 <= gold_count <= 10
        and bool(row.get("trace_parse", {}).get("exact_count", False))
    )


def _balanced_panel(
    states: Mapping[tuple[int, int, int, int], np.ndarray],
    *,
    seeds: Iterable[int],
    layers: Iterable[int],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[dict[str, int]]]:
    selected: dict[int, list[np.ndarray]] = {int(layer): [] for layer in layers}
    labels = []
    seed_ids = []
    audit = []
    by_seed_count: dict[tuple[int, int], list[int]] = defaultdict(list)
    for seed, final_count, boundary, _layer in states:
        by_seed_count[(int(seed), int(boundary))].append(int(final_count))
    for seed in sorted(int(value) for value in seeds):
        for boundary in range(1, 11):
            candidates = sorted(set(by_seed_count[(seed, boundary)]))
            if not candidates:
                raise RuntimeError(f"Seed {seed} has no clean format for boundary {boundary}")
            # Rotate the selected final N across seeds/boundaries so no count
            # class is tied to one terminal list length or one trace format.
            chosen_n = candidates[(seed * 37 + boundary * 11) % len(candidates)]
            for layer in layers:
                selected[int(layer)].append(states[(seed, chosen_n, boundary, int(layer))])
            labels.append(boundary)
            seed_ids.append(seed)
            audit.append({"seed": seed, "boundary": boundary, "selected_final_count": chosen_n})
    return (
        {layer: np.stack(values).astype(np.float32) for layer, values in selected.items()},
        np.asarray(labels, dtype=np.int64),
        np.asarray(seed_ids, dtype=np.int64),
        audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), default="Qwen3-8B")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 16, 24])
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.command = "fit-format-robust-boundary-probe"

    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model", ""))) == "Qwen3-8B"
    ]
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    anomaly_seeds = sorted(
        seed for seed, frame in by_seed.items() if any(_is_anomaly(row) for row in frame)
    )
    training_seeds = sorted(set(by_seed) - set(anomaly_seeds))
    if set(training_seeds) & set(anomaly_seeds):
        raise RuntimeError("Anomaly seeds leaked into format-robust training")
    clean_rows = [
        row
        for seed in training_seeds
        for row in by_seed[seed]
        if _is_clean(row)
    ]
    clean_keys = {
        (int(row["seed"]), len(row.get("gold_records", row.get("gold_pairs", []))))
        for row in clean_rows
    }
    missing = [
        (seed, count)
        for seed in training_seeds
        for count in range(1, 11)
        if (seed, count) not in clean_keys
    ]
    if missing:
        raise RuntimeError(f"Clean-only training seeds lack complete N coverage: {missing}")

    layers = tuple(sorted({int(value) for value in args.layers}))
    model, tokenizer, adapter = _model(args)
    states: dict[tuple[int, int, int, int], np.ndarray] = {}
    capture_audit = []
    for index, row in enumerate(clean_rows, start=1):
        seed = int(row["seed"])
        final_count = len(row.get("gold_records", row.get("gold_pairs", [])))
        trace_audit = _trace_audit(row)
        clean, registry, registry_audit = build_marker_scrubbed_list_registry(
            row, tokenizer, trace_audit=trace_audit
        )
        source, _blank, scrub_audit = build_targeted_explicit_count_scrub_source_and_blank(
            row,
            clean,
            registry,
            tokenizer,
            random_seed=20260830 + seed,
            marker_kind=str(trace_audit["marker_kind"]),
            mask_index_punctuation=False,
            first_item_char_override=int(trace_audit["item_char_spans"][0][0]),
        )
        positions = []
        boundary_audits = []
        for boundary in range(1, final_count + 1):
            position, audit = select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=boundary
            )
            positions.append(int(position))
            boundary_audits.append(audit)
        captures = capture_decoder_block_input_states(
            model, adapter, source, positions, layers=layers
        )
        for layer in layers:
            for boundary in range(1, final_count + 1):
                states[(seed, final_count, boundary, layer)] = (
                    captures[layer][boundary - 1].numpy().astype(np.float32)
                )
        capture_audit.append(
            {
                "request_id": str(row["request_id"]),
                "seed": seed,
                "final_count": final_count,
                "marker_kind": str(trace_audit["marker_kind"]),
                "registry_audit": registry_audit,
                "scrub_audit": scrub_audit,
                "boundary_audits": boundary_audits,
            }
        )
        print(f"[format-robust-probe] {index}/{len(clean_rows)} {row['request_id']}", flush=True)

    panel, labels, seed_ids, balance_audit = _balanced_panel(
        states, seeds=training_seeds, layers=layers
    )
    probes: dict[int, dict[str, Any]] = {}
    ordinal_probes: dict[int, dict[str, Any]] = {}
    layer_summary = []
    for layer in layers:
        probe = fit_dual_ridge_count_probe(panel[layer], labels, alpha=float(args.alpha))
        probes[layer] = probe
        ordinal_probe = _fit_ordinal_probe(panel[layer], labels, alpha=float(args.alpha))
        ordinal_probes[layer] = ordinal_probe
        loso = leave_one_seed_out_probe_metrics(
            panel[layer], labels, seed_ids, alpha=float(args.alpha)
        )
        all_keys = sorted(key for key in states if key[3] == layer)
        all_states = np.stack([states[key] for key in all_keys])
        all_labels = np.asarray([key[2] for key in all_keys], dtype=np.int64)
        all_predictions = count_probe_predictions(probe, all_states)
        layer_summary.append(
            {
                "layer": layer,
                "balanced_training_sites": int(len(labels)),
                "balanced_loso": loso,
                "ordinal_loso": _ordinal_loso(
                    panel[layer], labels, seed_ids, alpha=float(args.alpha)
                ),
                "all_clean_training_support": count_prediction_metrics(all_labels, all_predictions),
                "all_clean_ordinal_support": _ordinal_metrics(
                    all_labels, _ordinal_predictions(ordinal_probe, all_states)
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frozen_layers": np.asarray(layers, dtype=np.int64),
        "alpha": np.asarray([float(args.alpha)], dtype=np.float64),
        "training_seeds": np.asarray(training_seeds, dtype=np.int64),
        "heldout_anomaly_seeds": np.asarray(anomaly_seeds, dtype=np.int64),
    }
    for layer in layers:
        payload[f"layer_{layer}_mean"] = np.asarray(probes[layer]["mean"])
        payload[f"layer_{layer}_weights"] = np.asarray(probes[layer]["weights"])
        payload[f"layer_{layer}_ordinal_mean"] = np.asarray(ordinal_probes[layer]["mean"])
        payload[f"layer_{layer}_ordinal_weights"] = np.asarray(ordinal_probes[layer]["weights"])
        payload[f"layer_{layer}_ordinal_intercept"] = np.asarray(
            [float(ordinal_probes[layer]["intercept"])], dtype=np.float64
        )
    np.savez(
        args.output_dir / "format_robust_training_panel_float16.npz",
        **{f"layer_{layer}": panel[layer].astype(np.float16) for layer in layers},
        labels=labels,
        seed_ids=seed_ids,
    )
    np.savez(args.output_dir / "format_robust_probes.npz", **payload)
    _atomic_json(
        args.output_dir / "format_robust_probe_summary.json",
        {
            "schema_version": SCHEMA,
            "model_label": "Qwen3-8B",
            "generations": str(args.generations),
            "layers": list(layers),
            "alpha": float(args.alpha),
            "training_seeds": training_seeds,
            "heldout_anomaly_seeds": anomaly_seeds,
            "training_requests": len(clean_rows),
            "balanced_panel_sites": int(len(labels)),
            "balance_audit": balance_audit,
            "layer_summary": layer_summary,
            "capture_audit": capture_audit,
            "selection_rule": "layers inherited from the previously frozen strict-N10 probe; no anomaly states or outcomes accessed",
        },
    )
    print(
        f"[format-robust-probe] training_seeds={training_seeds} anomaly_seeds={anomaly_seeds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
