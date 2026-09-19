#!/usr/bin/env python3
"""Reparse frozen V2 full-item generations for content-bound multihop carryover."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
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


SCHEMA_VERSION = "realistic_niah_v6_full_item_multihop_analysis_v1"
PROTOCOL_STATUS = "FROZEN_BEFORE_V3_AGGREGATE_MULTIHOP_REPARSE"
ANALYSIS_STATUS = "POSTHOC_AGGREGATE_REPARSE_AFTER_ONE_SCHEMA_SMOKE_ROW"
MODES = ("enumeration_index", "enumeration_bullet")
MODELS = ("Qwen3-8B", "Gemma4-E4B")
DIRECTIONS = ("forward_skip", "backward_rewind")
CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")
DEPTHS = (1, 2, 4)


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


def _as_int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list, received {type(value).__name__}")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{field} contains a Boolean ordinal")
        result.append(int(item))
    return result


def exact_prefix_depth(observed: Iterable[int], expected: Iterable[int]) -> int:
    """Return the longest exact prefix; skipping or repair is intentionally forbidden."""

    depth = 0
    for actual, target in zip(observed, expected):
        if int(actual) != int(target):
            break
        depth += 1
    return depth


def _load(root: Path, protocol: Mapping[str, Any]) -> pd.DataFrame:
    contract = protocol["full_item_span_multihop_reparse"]
    frozen = contract["frozen_inputs"]
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for model in MODELS:
            for direction in DIRECTIONS:
                key = f"{mode}/{model}/{direction}"
                if key not in frozen:
                    raise ValueError(f"V3 protocol misses frozen input {key}")
                run = root / mode / model / direction
                manifest_path = run / "manifest.json"
                trials_path = run / "trials.jsonl"
                if not manifest_path.is_file() or not trials_path.is_file():
                    raise FileNotFoundError(f"Missing V2 full-item run under {run}")
                observed_manifest_hash = sha256_file(manifest_path)
                observed_trials_hash = sha256_file(trials_path)
                expected_hashes = frozen[key]
                if observed_manifest_hash != str(expected_hashes["manifest_sha256"]):
                    raise ValueError(f"Frozen manifest hash changed: {key}")
                if observed_trials_hash != str(expected_hashes["trials_sha256"]):
                    raise ValueError(f"Frozen trials hash changed: {key}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_receiver = int(contract["directions"][direction]["receiver_occurrence"])
                expected_donor = int(contract["directions"][direction]["donor_occurrence"])
                if (
                    manifest.get("status") != "PASS"
                    or str(manifest.get("patch_scope")) != "item_span"
                    or int(manifest.get("receiver_occurrence_j", -1)) != expected_receiver
                    or int(manifest.get("donor_occurrence_k", -1)) != expected_donor
                    or set(map(str, manifest.get("conditions", ()))) != set(CONDITIONS)
                    or set(map(str, manifest.get("generation_conditions", ()))) != set(CONDITIONS)
                    or int(manifest.get("seed_count", -1))
                    != int(contract["seed_count_per_direction"])
                ):
                    raise ValueError(f"Frozen V2 generation contract changed: {key}")
                for row in _read_jsonl(trials_path):
                    rows.append(
                        {
                            **row,
                            "prompt_mode": mode,
                            "model_label": model,
                            "direction": direction,
                            "run_manifest_sha256": observed_manifest_hash,
                            "trials_sha256": observed_trials_hash,
                        }
                    )
    frame = pd.DataFrame(rows)
    required = {
        "prompt_mode",
        "model_label",
        "direction",
        "seed",
        "request_id",
        "condition",
        "gold_count",
        "receiver_occurrence_j",
        "donor_occurrence_k",
        "donor_successor",
        "receiver_successor",
        "completion_text",
        "generated_known_city_ordinals_any_surface",
        "generation_truncated",
        "first_generated_known_city_ordinal",
        "patch_scope",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen generations lack V3 reparse fields: {missing}")
    return frame


def reparse_trials(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["prompt_mode", "model_label", "direction", "seed"]
    if set(frame["condition"].astype(str)) != set(CONDITIONS):
        raise ValueError("V3 full-item factorial changed")
    if frame.duplicated(keys + ["condition"]).any():
        raise ValueError("A seed/direction contains duplicate conditions")
    expected_rows = len(MODES) * len(MODELS) * len(DIRECTIONS) * len(CONDITIONS) * 10
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} V3 source rows, received {len(frame)}")
    if frame.groupby(["prompt_mode", "model_label", "direction"])["seed"].nunique().ne(10).any():
        raise ValueError("Every V3 cell/direction must retain all ten seeds")
    if not frame["patch_scope"].astype(str).eq("item_span").all():
        raise ValueError("A non-item-span row entered V3")

    reparsed: list[dict[str, Any]] = []
    for source in frame.to_dict(orient="records"):
        observed = _as_int_list(
            source["generated_known_city_ordinals_any_surface"],
            field="generated_known_city_ordinals_any_surface",
        )
        gold_count = int(source["gold_count"])
        donor_successor = int(source["donor_successor"])
        receiver_successor = int(source["receiver_successor"])
        donor_expected = list(range(donor_successor, gold_count + 1))
        receiver_expected = list(range(receiver_successor, gold_count + 1))
        if len(donor_expected) < max(DEPTHS):
            raise ValueError("Registered depth four is not eligible for a frozen row")
        donor_depth = exact_prefix_depth(observed, donor_expected)
        receiver_depth = exact_prefix_depth(observed, receiver_expected)
        first = source.get("first_generated_known_city_ordinal")
        first_value = None if first is None else int(first)
        first_list_value = observed[0] if observed else None
        if first_value != first_list_value:
            raise ValueError("Stored first ordinal disagrees with the registered ordinal list")
        monotone_consecutive = all(
            right == left + 1 for left, right in zip(observed, observed[1:])
        )
        ambiguous_value = source.get("ambiguous_known_city_bullet_lines", [])
        if isinstance(ambiguous_value, list):
            ambiguous_count = len(ambiguous_value)
        elif ambiguous_value is None:
            ambiguous_count = 0
        else:
            ambiguous_count = int(ambiguous_value)
        row = {
            "prompt_mode": str(source["prompt_mode"]),
            "model_label": str(source["model_label"]),
            "direction": str(source["direction"]),
            "seed": int(source["seed"]),
            "request_id": str(source["request_id"]),
            "condition": str(source["condition"]),
            "gold_count": gold_count,
            "receiver_occurrence_j": int(source["receiver_occurrence_j"]),
            "donor_occurrence_k": int(source["donor_occurrence_k"]),
            "donor_successor": donor_successor,
            "receiver_successor": receiver_successor,
            "generated_ordinals_json": json.dumps(observed, separators=(",", ":")),
            "generated_known_city_count": len(observed),
            "first_generated_known_city_ordinal": first_value,
            "donor_prefix_depth": donor_depth,
            "receiver_prefix_depth": receiver_depth,
            "donor_complete_path": int(donor_depth == len(donor_expected)),
            "receiver_complete_path": int(receiver_depth == len(receiver_expected)),
            "empty_known_city_generation": int(not observed),
            "duplicate_known_city_ordinal": int(len(observed) != len(set(observed))),
            "nonconsecutive_known_city_sequence": int(bool(observed) and not monotone_consecutive),
            "ambiguous_known_city_line_count": ambiguous_count,
            "generation_truncated": int(bool(source["generation_truncated"])),
            "generated_token_count": int(source.get("generated_token_count", 0)),
            "completion_text": str(source["completion_text"]),
            "run_manifest_sha256": str(source["run_manifest_sha256"]),
            "trials_sha256": str(source["trials_sha256"]),
        }
        for depth in DEPTHS:
            row[f"donor_depth_{depth}"] = int(donor_depth >= depth)
            row[f"receiver_depth_{depth}"] = int(receiver_depth >= depth)
        reparsed.append(row)
    return pd.DataFrame(reparsed)


def _interval(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    return bootstrap_seed_mean_ci(values, samples=int(samples), seed=int(seed))


def analyze(
    reparsed: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["prompt_mode", "model_label", "direction", "seed"]
    indexed = reparsed.set_index(keys + ["condition"])
    if indexed.index.duplicated().any():
        raise ValueError("V3 reparsed rows are not factorially unique")

    seed_rows: list[dict[str, Any]] = []
    for key, group in reparsed.groupby(keys, sort=True):
        arms = {str(row["condition"]): row for row in group.to_dict(orient="records")}
        if set(arms) != set(CONDITIONS):
            raise ValueError(f"Incomplete V3 arms for {key}")
        row: dict[str, Any] = dict(zip(keys, key))
        for depth in DEPTHS:
            patched = int(arms["donor_to_receiver"][f"donor_depth_{depth}"])
            receiver_self = int(arms["receiver_self"][f"donor_depth_{depth}"])
            native_donor = int(arms["native_donor"][f"donor_depth_{depth}"])
            row[f"patched_donor_depth_{depth}"] = patched
            row[f"receiver_self_donor_depth_{depth}"] = receiver_self
            row[f"native_donor_depth_{depth}"] = native_donor
            row[f"paired_donor_depth_{depth}_effect"] = patched - receiver_self
        row["receiver_self_receiver_depth_1"] = int(
            arms["receiver_self"]["receiver_depth_1"]
        )
        row["receiver_self_receiver_complete_path"] = int(
            arms["receiver_self"]["receiver_complete_path"]
        )
        row["native_donor_complete_path"] = int(
            arms["native_donor"]["donor_complete_path"]
        )
        row["patched_donor_complete_path"] = int(
            arms["donor_to_receiver"]["donor_complete_path"]
        )
        row["patched_donor_prefix_depth"] = int(
            arms["donor_to_receiver"]["donor_prefix_depth"]
        )
        seed_rows.append(row)
    effects = pd.DataFrame(seed_rows)

    direction_summaries: list[dict[str, Any]] = []
    counter = 0
    for group_key, group in effects.groupby(
        ["prompt_mode", "model_label", "direction"], sort=True
    ):
        mode, model, direction = map(str, group_key)
        summary: dict[str, Any] = {
            "prompt_mode": mode,
            "model_label": model,
            "direction": direction,
            "seed_count": len(group),
            "patched_mean_prefix_depth": float(group["patched_donor_prefix_depth"].mean()),
            "patched_complete_donor_path_rate": float(group["patched_donor_complete_path"].mean()),
            "native_donor_complete_path_rate": float(group["native_donor_complete_path"].mean()),
            "receiver_self_receiver_depth1_rate": float(group["receiver_self_receiver_depth_1"].mean()),
            "receiver_self_complete_receiver_path_rate": float(group["receiver_self_receiver_complete_path"].mean()),
        }
        for depth in DEPTHS:
            values = group[f"paired_donor_depth_{depth}_effect"].to_numpy(float)
            interval = _interval(
                values,
                samples=bootstrap_samples,
                seed=random_seed + counter,
            )
            counter += 1
            summary[f"depth_{depth}"] = {
                "patched_rate": float(group[f"patched_donor_depth_{depth}"].mean()),
                "receiver_self_donor_rate": float(
                    group[f"receiver_self_donor_depth_{depth}"].mean()
                ),
                "native_donor_rate": float(group[f"native_donor_depth_{depth}"].mean()),
                "paired_effect": interval,
                "paired_sign_flip_p_value": sign_flip_pvalue(values),
                "positive_95pct_ci": bool(interval["ci_low"] > 0.0),
            }
        direction_summaries.append(summary)

    cell_summaries: list[dict[str, Any]] = []
    for group_key, group in effects.groupby(["prompt_mode", "model_label"], sort=True):
        mode, model = map(str, group_key)
        aggregation: dict[str, tuple[str, str]] = {
            "patched_donor_prefix_depth": ("patched_donor_prefix_depth", "mean"),
            "patched_donor_complete_path": ("patched_donor_complete_path", "mean"),
            "native_donor_complete_path": ("native_donor_complete_path", "mean"),
            "receiver_self_receiver_depth_1": ("receiver_self_receiver_depth_1", "mean"),
            "receiver_self_receiver_complete_path": (
                "receiver_self_receiver_complete_path",
                "mean",
            ),
        }
        for depth in DEPTHS:
            aggregation[f"patched_donor_depth_{depth}"] = (
                f"patched_donor_depth_{depth}",
                "mean",
            )
            aggregation[f"receiver_self_donor_depth_{depth}"] = (
                f"receiver_self_donor_depth_{depth}",
                "mean",
            )
            aggregation[f"native_donor_depth_{depth}"] = (
                f"native_donor_depth_{depth}",
                "mean",
            )
            aggregation[f"paired_donor_depth_{depth}_effect"] = (
                f"paired_donor_depth_{depth}_effect",
                "mean",
            )
        per_seed = group.groupby("seed", as_index=False).agg(**aggregation)
        summary = {
            "prompt_mode": mode,
            "model_label": model,
            "seed_count": len(per_seed),
            "direction_count_per_seed": 2,
            "patched_mean_prefix_depth": float(per_seed["patched_donor_prefix_depth"].mean()),
            "patched_complete_donor_path_rate": float(per_seed["patched_donor_complete_path"].mean()),
            "native_donor_complete_path_rate": float(per_seed["native_donor_complete_path"].mean()),
            "receiver_self_receiver_depth1_rate": float(per_seed["receiver_self_receiver_depth_1"].mean()),
            "receiver_self_complete_receiver_path_rate": float(
                per_seed["receiver_self_receiver_complete_path"].mean()
            ),
        }
        for depth in DEPTHS:
            values = per_seed[f"paired_donor_depth_{depth}_effect"].to_numpy(float)
            interval = _interval(
                values,
                samples=bootstrap_samples,
                seed=random_seed + 100 + counter,
            )
            counter += 1
            summary[f"depth_{depth}"] = {
                "patched_rate": float(per_seed[f"patched_donor_depth_{depth}"].mean()),
                "receiver_self_donor_rate": float(
                    per_seed[f"receiver_self_donor_depth_{depth}"].mean()
                ),
                "native_donor_rate": float(per_seed[f"native_donor_depth_{depth}"].mean()),
                "paired_effect": interval,
                "paired_sign_flip_p_value": sign_flip_pvalue(values),
                "directional": bool(interval["mean_effect"] > 0.0),
                "strong_interval_gate_pass": bool(interval["ci_low"] > 0.0),
            }
        summary["primary_depth4_strong_gate_pass"] = bool(
            summary["depth_4"]["strong_interval_gate_pass"]
        )
        cell_summaries.append(summary)

    patched = reparsed[reparsed["condition"].eq("donor_to_receiver")].copy()
    depth1 = patched[patched["donor_depth_1"].eq(1)]
    conditional_persistence: list[dict[str, Any]] = []
    for (mode, model), group in patched.groupby(["prompt_mode", "model_label"], sort=True):
        eligible = depth1[
            depth1["prompt_mode"].eq(mode) & depth1["model_label"].eq(model)
        ]
        conditional_persistence.append(
            {
                "prompt_mode": str(mode),
                "model_label": str(model),
                "unconditional_row_count": len(group),
                "depth1_adopter_count": len(eligible),
                "depth2_given_depth1_count": int(eligible["donor_depth_2"].sum()),
                "depth2_given_depth1_rate": (
                    float(eligible["donor_depth_2"].mean()) if len(eligible) else None
                ),
                "depth4_given_depth1_count": int(eligible["donor_depth_4"].sum()),
                "depth4_given_depth1_rate": (
                    float(eligible["donor_depth_4"].mean()) if len(eligible) else None
                ),
                "qualification": "Descriptive only: conditioning uses the intervention outcome.",
            }
        )

    fixed_examples = []
    for (mode, model, direction), group in reparsed.groupby(
        ["prompt_mode", "model_label", "direction"], sort=True
    ):
        seed = int(group["seed"].min())
        for condition in CONDITIONS:
            row = group[group["seed"].eq(seed) & group["condition"].eq(condition)].iloc[0]
            fixed_examples.append(
                {
                    "prompt_mode": str(mode),
                    "model_label": str(model),
                    "direction": str(direction),
                    "seed": seed,
                    "condition": condition,
                    "generated_ordinals": json.loads(str(row["generated_ordinals_json"])),
                    "donor_prefix_depth": int(row["donor_prefix_depth"]),
                    "receiver_prefix_depth": int(row["receiver_prefix_depth"]),
                    "generation_truncated": bool(row["generation_truncated"]),
                    "completion_text": str(row["completion_text"]),
                }
            )

    failure_taxonomy = {
        "all_rows": int(len(reparsed)),
        "patched_rows": int(len(patched)),
        "generation_truncated_rows": int(reparsed["generation_truncated"].sum()),
        "empty_known_city_rows": int(reparsed["empty_known_city_generation"].sum()),
        "duplicate_ordinal_rows": int(reparsed["duplicate_known_city_ordinal"].sum()),
        "nonconsecutive_ordinal_rows": int(
            reparsed["nonconsecutive_known_city_sequence"].sum()
        ),
        "ambiguous_known_city_line_rows": int(
            reparsed["ambiguous_known_city_line_count"].gt(0).sum()
        ),
        "patched_depth1_failures": int(patched["donor_depth_1"].eq(0).sum()),
        "patched_depth2_failures": int(patched["donor_depth_2"].eq(0).sum()),
        "patched_depth4_failures": int(patched["donor_depth_4"].eq(0).sum()),
    }
    claims = {
        "schema_version": SCHEMA_VERSION,
        "status": "POSTHOC_MULTIHOP_REPARSE_COMPLETE",
        "analysis_status": ANALYSIS_STATUS,
        "primary_outcome": "donor_prefix_depth_at_least_4",
        "primary_estimand": (
            "donor_to_receiver minus receiver_self paired depth-4 adoption, "
            "direction averaged within true source seed"
        ),
        "registered_depths": list(DEPTHS),
        "row_count": int(len(reparsed)),
        "seed_effect_row_count": int(len(effects)),
        "direction_summaries": direction_summaries,
        "cell_summaries": cell_summaries,
        "conditional_persistence": conditional_persistence,
        "fixed_lowest_seed_examples": fixed_examples,
        "failure_taxonomy": failure_taxonomy,
        "all_cells_primary_depth4_directional": all(
            row["depth_4"]["directional"] for row in cell_summaries
        ),
        "all_cells_primary_depth4_strong_gate_pass": all(
            row["primary_depth4_strong_gate_pass"] for row in cell_summaries
        ),
        "all_frozen_inputs_verified": True,
        "new_model_forward_used": False,
        "all_ten_seeds_retained": True,
        "truncated_and_failed_rows_retained_in_denominators": True,
        "frozen_k_heads_layers_or_directions_changed": False,
        "qualification": (
            "Post-hoc aggregate reparse of frozen V2 free generations after one "
            "schema smoke row was seen. It can establish whether the recorded "
            "full-state intervention persisted across later generated cities, but "
            "it does not replace the original V6/V2 gates or identify a low-dimensional operator."
        ),
    }
    return effects, claims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=606831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError("V3 protocol is not frozen at the registered checkpoint")
    source = _load(args.root.resolve(), protocol)
    reparsed = reparse_trials(source)
    effects, claims = analyze(
        reparsed,
        bootstrap_samples=int(args.bootstrap_samples),
        random_seed=int(args.random_seed),
    )
    claims.update(
        {
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "input_root": str(args.root.resolve()),
        }
    )
    _atomic_csv(args.output / "trial_reparse.csv", reparsed)
    _atomic_csv(args.output / "seed_effects.csv", effects)
    _atomic_json(args.output / "claim_gates.json", claims)
    print(
        json.dumps(
            {
                "status": claims["status"],
                "row_count": claims["row_count"],
                "all_cells_primary_depth4_directional": claims[
                    "all_cells_primary_depth4_directional"
                ],
                "all_cells_primary_depth4_strong_gate_pass": claims[
                    "all_cells_primary_depth4_strong_gate_pass"
                ],
                "new_model_forward_used": claims["new_model_forward_used"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
