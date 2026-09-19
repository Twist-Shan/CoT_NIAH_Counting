#!/usr/bin/env python3
"""Analyze the frozen marker-component, layer-band, and exact-edge tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

PRE_READ_BANDS = ("L00_03", "L04_07", "L08_11", "L12_15", "L16_19", "L20_23")
VALID_VARIANT = "insert_valid_item"
PRIMARY_INVALID_VARIANT = "insert_markerless_valid_payload"
EDGE_CONTROLS = (
    "inserted_payload_first",
    "inserted_payload_mid",
    "inserted_closing",
    "pre_insertion_b5",
)


def sign_flip_pvalue(seed_effects: Sequence[float]) -> float:
    """Match the repository's exact two-sided sign-flip convention."""

    values = np.asarray(seed_effects, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count == 0:
        raise ValueError("Sign-flip test needs finite seed effects")
    observed = abs(float(values.mean()))
    if count <= 20:
        signs = 1 - 2 * ((np.arange(2**count)[:, None] >> np.arange(count)) & 1)
        permuted = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(544)
        signs = rng.choice((-1, 1), size=(200_000, count))
        permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (len(permuted) + 1))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in original order."""

    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    predicate: Callable[[Mapping[str, Any]], bool],
    *,
    expected: int,
) -> list[float]:
    values = [
        float(row["donor_axis_progress"])
        for row in rows
        if int(row["seed"]) == int(seed) and predicate(row)
    ]
    if len(values) != int(expected) or not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"Seed {seed} expected {expected} finite rows but found {len(values)}"
        )
    return values


def _describe(values: Sequence[float]) -> dict[str, Any]:
    active = [float(value) for value in values]
    if not active:
        raise ValueError("Cannot describe an empty estimand")
    return {
        "n_seeds": len(active),
        "mean": fmean(active),
        "median": float(np.median(active)),
        "minimum": min(active),
        "maximum": max(active),
        "positive_seed_count": sum(value > 0 for value in active),
        "positive_seed_rate": float(np.mean(np.asarray(active) > 0)),
        "exact_two_sided_sign_flip_pvalue": sign_flip_pvalue(active),
    }


def _component_seed_value(
    rows: Sequence[Mapping[str, Any]], seed: int, component: str
) -> float:
    values = _values(
        rows,
        seed,
        lambda row: str(row["condition"]) == "marker_component"
        and str(row["component_label"]) == component
        and str(row["layer_band"]) == "all_layers"
        and int(row["read_layer"]) == 24,
        expected=2,
    )
    return fmean(values)


def _band_seed_value(
    rows: Sequence[Mapping[str, Any]], seed: int, band: str
) -> float:
    values = _values(
        rows,
        seed,
        lambda row: str(row["condition"]) == "marker_layer_band"
        and str(row["component_label"]) == "KV"
        and str(row["layer_band"]) == band
        and int(row["read_layer"]) == 24,
        expected=2,
    )
    return fmean(values)


def _edge_seed_value(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    receiver: str,
    query: str,
    key: str,
) -> float:
    return _values(
        rows,
        seed,
        lambda row: str(row["condition"]) == "exact_attention_edge_mask"
        and str(row["receiver_variant"]) == receiver
        and str(row["edge_query_role"]) == query
        and str(row["edge_key_role"]) == key
        and int(row["read_layer"]) == 24,
        expected=1,
    )[0]


def _secondary_tables(
    rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    components = {
        label: {
            **_describe([_component_seed_value(rows, seed, label) for seed in seeds]),
            "per_seed": {
                str(seed): _component_seed_value(rows, seed, label) for seed in seeds
            },
        }
        for label in ("K", "V", "KV")
    }
    bands = {
        band: {
            **_describe([_band_seed_value(rows, seed, band) for seed in seeds]),
            "per_seed": {
                str(seed): _band_seed_value(rows, seed, band) for seed in seeds
            },
        }
        for band in (*PRE_READ_BANDS, "L24_35_postread_control")
    }
    edges: dict[str, Any] = {}
    for receiver in (VALID_VARIANT, PRIMARY_INVALID_VARIANT):
        for query in ("target_marker", "target_boundary"):
            for key in (
                "inserted_marker",
                "inserted_payload_first",
                "inserted_payload_mid",
                "inserted_closing",
                "pre_insertion_b5",
            ):
                values = [
                    _edge_seed_value(
                        rows,
                        seed,
                        receiver=receiver,
                        query=query,
                        key=key,
                    )
                    for seed in seeds
                ]
                edges[f"{receiver}|{query}|{key}"] = {
                    **_describe(values),
                    "per_seed": {
                        str(seed): value for seed, value in zip(seeds, values)
                    },
                }
    return {"components": components, "bands_KV": bands, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = _read_json(args.freeze)
    discovery = _read_jsonl(args.discovery)
    confirmation = _read_jsonl(args.confirmation)
    confirmation_summary = _read_json(args.confirmation_summary)
    seeds = tuple(int(value) for value in freeze["confirmation_seeds"])
    if sorted({int(row["seed"]) for row in confirmation}) != sorted(seeds):
        raise RuntimeError("Confirmation rows do not match the frozen seed cohort")
    if _sha256(args.discovery) != str(
        freeze["frozen_hashes"]["discovery_trials_sha256"]
    ):
        raise RuntimeError("Discovery trials do not match the pre-confirmation freeze")

    selected_component = str(
        freeze["frozen_discovery_selection"]["component"]["component_label"]
    )
    selected_band = str(
        freeze["frozen_discovery_selection"]["layer_band"]["band_label"]
    )
    selected_query = str(
        freeze["frozen_discovery_selection"]["exact_edge"]["query_role"]
    )
    if (selected_component, selected_band, selected_query) != (
        "V",
        "L20_23",
        "target_marker",
    ):
        raise RuntimeError("Unexpected frozen marker-circuit selection")

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        component = _component_seed_value(confirmation, seed, selected_component)
        selected_band_raw = _band_seed_value(confirmation, seed, selected_band)
        other_bands = [
            _band_seed_value(confirmation, seed, band)
            for band in PRE_READ_BANDS
            if band != selected_band
        ]
        band_contrast = selected_band_raw - fmean(other_bands)
        valid_marker = _edge_seed_value(
            confirmation,
            seed,
            receiver=VALID_VARIANT,
            query=selected_query,
            key="inserted_marker",
        )
        markerless_position = _edge_seed_value(
            confirmation,
            seed,
            receiver=PRIMARY_INVALID_VARIANT,
            query=selected_query,
            key="inserted_marker",
        )
        valid_controls = [
            _edge_seed_value(
                confirmation,
                seed,
                receiver=VALID_VARIANT,
                query=selected_query,
                key=key,
            )
            for key in EDGE_CONTROLS
        ]
        edge_composite = (
            valid_marker - 0.5 * markerless_position - 0.5 * fmean(valid_controls)
        )
        per_seed.append(
            {
                "seed": seed,
                "component_primary": component,
                "selected_band_raw": selected_band_raw,
                "other_pre_read_bands_mean": fmean(other_bands),
                "band_primary_contrast": band_contrast,
                "valid_marker_edge_raw": valid_marker,
                "markerless_same_position_edge": markerless_position,
                "valid_non_marker_key_controls_mean": fmean(valid_controls),
                "edge_primary_composite": edge_composite,
            }
        )

    definitions = (
        (
            "component",
            "component_primary",
            float(freeze["confirmation_primary_tests"][0]["minimum_mean_effect"]),
        ),
        (
            "layer_band",
            "band_primary_contrast",
            float(freeze["confirmation_primary_tests"][1]["minimum_mean_effect"]),
        ),
        (
            "exact_edge",
            "edge_primary_composite",
            float(freeze["confirmation_primary_tests"][2]["minimum_mean_effect"]),
        ),
    )
    primary = []
    raw_pvalues = []
    for family, field, floor in definitions:
        values = [float(row[field]) for row in per_seed]
        cell = {
            "family": family,
            "per_seed_field": field,
            **_describe(values),
            "minimum_mean_effect": floor,
            "effect_floor_pass": fmean(values) >= floor,
        }
        primary.append(cell)
        raw_pvalues.append(float(cell["exact_two_sided_sign_flip_pvalue"]))
    adjusted = holm_adjust(raw_pvalues)
    for cell, pvalue in zip(primary, adjusted):
        cell["holm_adjusted_pvalue"] = float(pvalue)
        cell["holm_alpha"] = 0.05
        cell["statistical_pass"] = float(pvalue) < 0.05
        cell["confirmation_pass"] = bool(cell["effect_floor_pass"]) and bool(
            cell["statistical_pass"]
        )

    band_raw = [float(row["selected_band_raw"]) for row in per_seed]
    edge_raw = [float(row["valid_marker_edge_raw"]) for row in per_seed]
    primary[1]["additional_raw_selected_band"] = _describe(band_raw)
    primary[1]["additional_raw_selected_band_minimum"] = float(
        freeze["confirmation_primary_tests"][1][
            "additional_raw_selected_band_minimum"
        ]
    )
    primary[1]["additional_raw_floor_pass"] = fmean(band_raw) >= float(
        primary[1]["additional_raw_selected_band_minimum"]
    )
    primary[1]["confirmation_pass"] = bool(primary[1]["confirmation_pass"]) and bool(
        primary[1]["additional_raw_floor_pass"]
    )
    primary[2]["additional_raw_valid_marker_edge"] = _describe(edge_raw)
    primary[2]["additional_raw_valid_marker_edge_minimum"] = float(
        freeze["confirmation_primary_tests"][2][
            "additional_raw_valid_marker_edge_minimum"
        ]
    )
    primary[2]["additional_raw_floor_pass"] = fmean(edge_raw) >= float(
        primary[2]["additional_raw_valid_marker_edge_minimum"]
    )
    primary[2]["confirmation_pass"] = bool(primary[2]["confirmation_pass"]) and bool(
        primary[2]["additional_raw_floor_pass"]
    )

    secondary_confirmation = _secondary_tables(confirmation, seeds)
    discovery_seeds = tuple(sorted({int(row["seed"]) for row in discovery}))
    secondary_discovery = _secondary_tables(discovery, discovery_seeds)
    component_additivity = {
        "confirmation_K_plus_V_mean": float(
            secondary_confirmation["components"]["K"]["mean"]
            + secondary_confirmation["components"]["V"]["mean"]
        ),
        "confirmation_joint_KV_mean": float(
            secondary_confirmation["components"]["KV"]["mean"]
        ),
    }
    component_additivity["joint_minus_sum"] = float(
        component_additivity["confirmation_joint_KV_mean"]
        - component_additivity["confirmation_K_plus_V_mean"]
    )

    postread_values = [
        _band_seed_value(confirmation, seed, "L24_35_postread_control")
        for seed in seeds
    ]
    audit = {
        "discovery_hash_matches_freeze": True,
        "confirmation_seeds_match_freeze": True,
        "custom_4d_clean_all_probe_predictions_match": bool(
            confirmation_summary["custom_4d_clean_all_probe_predictions_match"]
        ),
        "custom_4d_clean_min_cosine_similarity": float(
            confirmation_summary["custom_4d_clean_min_cosine_similarity"]
        ),
        "all_edge_masks_delete_exactly_one_allowed_edge": bool(
            confirmation_summary["all_edge_masks_delete_exactly_one_allowed_edge"]
        ),
        "postread_control_maximum_absolute_progress": max(
            abs(value) for value in postread_values
        ),
        "postread_control_exactly_zero": all(value == 0.0 for value in postread_values),
    }
    if not all(
        (
            audit["custom_4d_clean_all_probe_predictions_match"],
            audit["all_edge_masks_delete_exactly_one_allowed_edge"],
            audit["postread_control_exactly_zero"],
        )
    ):
        raise RuntimeError("A marker-circuit implementation audit failed")

    output = {
        "schema_version": "realistic_niah_v5_marker_circuit_analysis_v1",
        "freeze_record": str(args.freeze),
        "confirmation_seeds": list(seeds),
        "frozen_selection": {
            "component": selected_component,
            "layer_band": selected_band,
            "edge_query": selected_query,
            "edge_key": "inserted_marker",
        },
        "per_seed_primary_estimands": per_seed,
        "primary_confirmation_tests": primary,
        "all_three_primary_families_pass": all(
            bool(cell["confirmation_pass"]) for cell in primary
        ),
        "passed_primary_families": [
            str(cell["family"]) for cell in primary if bool(cell["confirmation_pass"])
        ],
        "failed_primary_families": [
            str(cell["family"]) for cell in primary if not bool(cell["confirmation_pass"])
        ],
        "component_additivity": component_additivity,
        "secondary_confirmation": secondary_confirmation,
        "secondary_discovery": secondary_discovery,
        "implementation_audit": audit,
        "interpretation": {
            "component": (
                "A passing V-only arm shows that marker values are a sufficient "
                "carrier under cache transplantation; K-only remains a secondary "
                "contributor and natural-use necessity is not established."
            ),
            "layer_band": (
                "A passing L20-23 contrast localizes most marker-cache transport "
                "immediately before the L24 readout."
            ),
            "exact_edge": (
                "The preregistered marker-key edge claim passes only if its "
                "specificity-adjusted composite survives confirmation. A failed "
                "edge test is evidence against that unique route, but not against "
                "redundant marker-mediated routing."
            ),
        },
    }
    _atomic_json(args.output, output)
    print(
        json.dumps(
            {
                "primary_confirmation_tests": primary,
                "passed_primary_families": output["passed_primary_families"],
                "failed_primary_families": output["failed_primary_families"],
                "component_additivity": component_additivity,
                "implementation_audit": audit,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
