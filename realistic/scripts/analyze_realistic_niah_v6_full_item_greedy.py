#!/usr/bin/env python3
"""Analyze the frozen-layer full-item-span greedy city-adoption extension."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (  # noqa: E402
    bootstrap_seed_mean_ci,
    sign_flip_pvalue,
)
from realistic_niah_v6.pipeline import sha256_file  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v6_full_item_greedy_analysis_v1"
MODES = ("enumeration_index", "enumeration_bullet")
MODE_LABELS = {
    "enumeration_index": "enumeration_index",
    "enumeration_bullet": "enumeration_bullet",
}
MODELS = ("Qwen3-8B", "Gemma4-E4B")
DIRECTIONS = ("forward_skip", "backward_rewind")
EXPECTED_CONDITIONS = {"receiver_self", "native_donor", "donor_to_receiver"}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _protocol_layer(protocol: Mapping[str, Any], mode: str, model: str) -> int:
    key = f"{model} x {MODE_LABELS[mode]}"
    layers = protocol["full_item_span_greedy_city_adoption"][
        "frozen_item_span_layers"
    ]
    return int(layers[key])


def _load(root: Path, protocol: Mapping[str, Any]) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for mode in MODES:
        for model in MODELS:
            expected_layer = _protocol_layer(protocol, mode, model)
            for direction in DIRECTIONS:
                run = root / mode / model / direction
                trials_path = run / "trials.jsonl"
                manifest_path = run / "manifest.json"
                if not trials_path.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing full-item greedy run under {run}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_receiver = 5 if direction == "forward_skip" else 7
                if (
                    manifest.get("status") != "PASS"
                    or str(manifest.get("patch_scope")) != "item_span"
                    or [int(value) for value in manifest.get("layers", ())]
                    != [expected_layer]
                    or int(manifest.get("receiver_occurrence_j", -1))
                    != expected_receiver
                    or int(manifest.get("donor_occurrence_k", -1)) != 6
                    or set(map(str, manifest.get("conditions", ())))
                    != EXPECTED_CONDITIONS
                    or set(map(str, manifest.get("generation_conditions", ())))
                    != EXPECTED_CONDITIONS
                    or int(manifest.get("seed_count", -1)) != 10
                ):
                    raise ValueError(f"Frozen full-item greedy contract changed: {run}")
                rows = _read_jsonl(trials_path)
                for row in rows:
                    all_rows.append(
                        {
                            **row,
                            "prompt_mode": mode,
                            "model_label": model,
                            "direction": direction,
                            "run_manifest_sha256": sha256_file(manifest_path),
                            "trials_sha256": sha256_file(trials_path),
                        }
                    )
    frame = pd.DataFrame(all_rows)
    required = {
        "prompt_mode",
        "model_label",
        "direction",
        "seed",
        "condition",
        "layer",
        "patch_scope",
        "greedy_donor_successor_adoption",
        "greedy_receiver_successor_retention",
        "first_generated_known_city_ordinal",
        "generation_truncated",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Full-item greedy rows lack fields: {missing}")
    return frame


def analyze(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["prompt_mode", "model_label", "direction", "seed"]
    if set(frame["condition"].astype(str)) != EXPECTED_CONDITIONS:
        raise ValueError("Full-item greedy factorial changed")
    if not frame["patch_scope"].astype(str).eq("item_span").all():
        raise ValueError("A non-item-span row entered the greedy analysis")
    if frame.duplicated(keys + ["condition"]).any():
        raise ValueError("Full-item greedy rows duplicate a condition")
    if frame.groupby(["prompt_mode", "model_label", "direction"])[
        "seed"
    ].nunique().ne(10).any():
        raise ValueError("Every full-item greedy direction must have ten seeds")

    active = frame.copy()
    for field in (
        "greedy_donor_successor_adoption",
        "greedy_receiver_successor_retention",
        "generation_truncated",
    ):
        active[field] = active[field].map(bool)
    donor = active.pivot(
        index=keys,
        columns="condition",
        values="greedy_donor_successor_adoption",
    ).astype(float)
    receiver = active.pivot(
        index=keys,
        columns="condition",
        values="greedy_receiver_successor_retention",
    ).astype(float)
    if donor.isna().any().any() or receiver.isna().any().any():
        raise ValueError("Full-item greedy factorial contains missing arms")
    seed_frame = donor.reset_index()[keys].copy()
    seed_frame["patched_donor_adoption"] = donor["donor_to_receiver"].to_numpy()
    seed_frame["self_donor_adoption"] = donor["receiver_self"].to_numpy()
    seed_frame["native_donor_adoption"] = donor["native_donor"].to_numpy()
    seed_frame["self_receiver_retention"] = receiver[
        "receiver_self"
    ].to_numpy()
    seed_frame["patched_receiver_retention"] = receiver[
        "donor_to_receiver"
    ].to_numpy()
    seed_frame["paired_adoption_effect"] = (
        seed_frame["patched_donor_adoption"]
        - seed_frame["self_donor_adoption"]
    )

    direction_summaries: list[dict[str, Any]] = []
    for group_key, group in seed_frame.groupby(
        ["prompt_mode", "model_label", "direction"], sort=True
    ):
        mode, model, direction = map(str, group_key)
        values = group["paired_adoption_effect"].to_numpy(float)
        interval = bootstrap_seed_mean_ci(
            values,
            samples=int(bootstrap_samples),
            seed=int(random_seed) + len(direction_summaries),
        )
        direction_summaries.append(
            {
                "prompt_mode": mode,
                "model_label": model,
                "direction": direction,
                "seed_count": len(group),
                "patched_donor_adoption_rate": float(
                    group["patched_donor_adoption"].mean()
                ),
                "receiver_self_donor_adoption_rate": float(
                    group["self_donor_adoption"].mean()
                ),
                "native_donor_adoption_rate": float(
                    group["native_donor_adoption"].mean()
                ),
                "receiver_self_retention_rate": float(
                    group["self_receiver_retention"].mean()
                ),
                "patched_receiver_retention_rate": float(
                    group["patched_receiver_retention"].mean()
                ),
                "paired_adoption_effect": interval,
                "paired_sign_flip_p_value": sign_flip_pvalue(values),
                "positive_95pct_ci": bool(interval["ci_low"] > 0.0),
            }
        )

    cell_summaries: list[dict[str, Any]] = []
    for group_key, group in seed_frame.groupby(
        ["prompt_mode", "model_label"], sort=True
    ):
        mode, model = map(str, group_key)
        per_seed = (
            group.groupby("seed", as_index=False)
            .agg(
                paired_adoption_effect=("paired_adoption_effect", "mean"),
                patched_donor_adoption=("patched_donor_adoption", "mean"),
                self_donor_adoption=("self_donor_adoption", "mean"),
                native_donor_adoption=("native_donor_adoption", "mean"),
                self_receiver_retention=("self_receiver_retention", "mean"),
            )
        )
        values = per_seed["paired_adoption_effect"].to_numpy(float)
        interval = bootstrap_seed_mean_ci(
            values,
            samples=int(bootstrap_samples),
            seed=int(random_seed) + 100 + len(cell_summaries),
        )
        cell_summaries.append(
            {
                "prompt_mode": mode,
                "model_label": model,
                "seed_count": len(per_seed),
                "direction_count_per_seed": 2,
                "patched_donor_adoption_rate": float(
                    per_seed["patched_donor_adoption"].mean()
                ),
                "receiver_self_donor_adoption_rate": float(
                    per_seed["self_donor_adoption"].mean()
                ),
                "native_donor_adoption_rate": float(
                    per_seed["native_donor_adoption"].mean()
                ),
                "receiver_self_retention_rate": float(
                    per_seed["self_receiver_retention"].mean()
                ),
                "paired_adoption_effect": interval,
                "paired_sign_flip_p_value": sign_flip_pvalue(values),
                "directional": bool(interval["mean_effect"] > 0.0),
                "strong_interval_gate_pass": bool(interval["ci_low"] > 0.0),
            }
        )

    truncated = active.groupby(
        ["prompt_mode", "model_label", "direction", "condition"], sort=True
    )["generation_truncated"].mean()
    claims = {
        "schema_version": SCHEMA_VERSION,
        "status": "POSTHOC_GREEDY_READOUT_EXTENSION_COMPLETE",
        "analysis_status": "POSTHOC_CONFIRMATION_SPLIT_REUSE",
        "primary_outcome": "greedy_donor_successor_adoption",
        "primary_estimand": "donor_to_receiver - receiver_self",
        "direction_summaries": direction_summaries,
        "cell_summaries": cell_summaries,
        "all_cells_directional": all(row["directional"] for row in cell_summaries),
        "all_cells_strong_interval_gate_pass": all(
            row["strong_interval_gate_pass"] for row in cell_summaries
        ),
        "maximum_generation_truncation_rate": float(truncated.max()),
        "frozen_layers_changed": False,
        "frozen_k_changed": False,
        "seed_selection_used_greedy_outcomes": False,
        "qualification": (
            "The readout was added after related V6 confirmation outcomes were "
            "inspected. It tests free-generation adoption at the already frozen "
            "full-item geometry and does not replace an original V6 gate."
        ),
    }
    return seed_frame, claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V2_INTERVENTION_OUTCOMES":
        raise ValueError("V2 protocol is not frozen")
    effects, claims = analyze(
        _load(args.root.resolve(), protocol),
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    claims.update(
        {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
        }
    )
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", claims)
    print(
        json.dumps(
            {
                "status": claims["status"],
                "all_cells_directional": claims["all_cells_directional"],
                "all_cells_strong_interval_gate_pass": claims[
                    "all_cells_strong_interval_gate_pass"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
