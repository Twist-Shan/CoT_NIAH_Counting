#!/usr/bin/env python3
"""Audit and summarize the native count-stream full-state follow-up.

This is deliberately descriptive development analysis.  It preserves the
outcome-blind pair plan, pairs every intervention to its own self-patch arm,
and seed-equalizes before bootstrapping.  It does not label development CIs as
confirmation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FULL_STATE_DIRS = {
    "middle": "middle_full_state_clamp_rank5",
    "terminal_last": "terminal_last_full_state_clamp_rank10",
}
SOURCE_MASK_DIR = "source_mask_query_only_all_heads"
GEOMETRIES = ("endpoint", "suffix4", "suffix8", "full_span")


def _read_shards(path: Path) -> pd.DataFrame:
    files = sorted((path / "shards").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL shards under {path}")
    rows = []
    for file in files:
        rows.extend(
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return pd.DataFrame(rows)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _bootstrap_seed_mean(
    seed_values: np.ndarray, *, samples: int, random_seed: int
) -> tuple[float, float, float]:
    values = np.asarray(seed_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(random_seed))
    draws = rng.choice(values, size=(int(samples), len(values)), replace=True).mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _seed_equal_summary(
    frame: pd.DataFrame,
    *,
    grouping: dict[str, Any],
    metrics: Iterable[str],
    bootstrap_samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        values = pd.to_numeric(frame[metric], errors="coerce")
        finite = frame.loc[np.isfinite(values)].copy()
        finite[metric] = values.loc[finite.index]
        if finite.empty:
            continue
        seed_effects = finite.groupby("seed", as_index=False)[metric].mean()
        mean, low, high = _bootstrap_seed_mean(
            seed_effects[metric].to_numpy(),
            samples=bootstrap_samples,
            random_seed=int(random_seed) + metric_index,
        )
        rows.append(
            {
                **grouping,
                "metric": metric,
                "mean_seed_equal": mean,
                "ci_low": low,
                "ci_high": high,
                "median_pair": float(finite[metric].median()),
                "mean_pair": float(finite[metric].mean()),
                "n_pairs": int(len(finite)),
                "n_seeds": int(seed_effects["seed"].nunique()),
                "development_only": True,
            }
        )
    return rows


def _paired_full_state(panel: str, trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials.loc[trials["status"].eq("ok")].copy()
    identity = [
        "model_label",
        "seed",
        "request_id",
        "pair_sha256",
        "selection_cell_id",
        "selection_rank",
        "gold_count",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "patch_geometry",
        "patch_layer_mode",
    ]
    outcomes = [
        "expected_count",
        "expected_count_absolute_error",
        "correct_count_margin",
        "correct_count_probability",
        "correct_count_log_score",
        "predicted_count_among_candidates",
        "answer_query_full_state_displacement_rms",
        "downstream_item_progress_subspace_displacement_rms",
        "patch_realized_aggregate_fro_norm",
    ]
    for optional in (
        "prediction",
        "predicted_count",
        "signed_error",
        "absolute_error",
        "exact_count",
        "invalid_count_output",
    ):
        if optional in ok.columns:
            outcomes.append(optional)
    duplicate = ok.duplicated(identity + ["condition"], keep=False)
    if duplicate.any():
        raise ValueError(f"{panel} has duplicate condition rows")
    wide = ok.pivot(index=identity, columns="condition", values=outcomes)
    required = {"clean", "self_patch", "full_donor_patch"}
    observed = set(wide.columns.get_level_values(1))
    if not required <= observed:
        raise ValueError(f"{panel} lacks full-state conditions {required - observed}")
    wide.columns = [f"{outcome}__{condition}" for outcome, condition in wide.columns]
    paired = wide.reset_index()

    for outcome in (
        "expected_count",
        "correct_count_margin",
        "correct_count_probability",
        "correct_count_log_score",
    ):
        clean = pd.to_numeric(paired[f"{outcome}__clean"], errors="coerce")
        self_patch = pd.to_numeric(
            paired[f"{outcome}__self_patch"], errors="coerce"
        )
        difference = np.abs(clean - self_patch)
        if np.nanmax(difference.to_numpy()) > 1e-6:
            raise ValueError(f"{panel} self-patch is not a no-op for {outcome}")

    paired["expected_count_shift"] = (
        paired["expected_count__full_donor_patch"]
        - paired["expected_count__self_patch"]
    )
    paired["donor_adoption_fraction"] = (
        paired["expected_count_shift"] / paired["donor_offset"]
    )
    paired["correct_margin_damage"] = (
        paired["correct_count_margin__self_patch"]
        - paired["correct_count_margin__full_donor_patch"]
    )
    paired["correct_probability_damage"] = (
        paired["correct_count_probability__self_patch"]
        - paired["correct_count_probability__full_donor_patch"]
    )
    paired["correct_log_score_damage"] = (
        paired["correct_count_log_score__self_patch"]
        - paired["correct_count_log_score__full_donor_patch"]
    )
    paired["candidate_prediction_shift"] = (
        paired["predicted_count_among_candidates__full_donor_patch"]
        - paired["predicted_count_among_candidates__self_patch"]
    )
    paired["candidate_added_error"] = (
        paired["predicted_count_among_candidates__self_patch"].eq(
            paired["gold_count"]
        )
        & paired["predicted_count_among_candidates__full_donor_patch"].ne(
            paired["gold_count"]
        )
    )
    paired["candidate_corrected_error"] = (
        paired["predicted_count_among_candidates__self_patch"].ne(
            paired["gold_count"]
        )
        & paired["predicted_count_among_candidates__full_donor_patch"].eq(
            paired["gold_count"]
        )
    )
    paired["answer_query_displacement"] = paired[
        "answer_query_full_state_displacement_rms__full_donor_patch"
    ]
    paired["downstream_progress_displacement"] = paired[
        "downstream_item_progress_subspace_displacement_rms__full_donor_patch"
    ]
    paired["patch_fro_norm"] = paired[
        "patch_realized_aggregate_fro_norm__full_donor_patch"
    ]
    greedy_prediction_field = next(
        (
            field
            for field in ("prediction", "predicted_count")
            if f"{field}__full_donor_patch" in paired.columns
        ),
        None,
    )
    if greedy_prediction_field is not None:
        paired["greedy_count_shift"] = (
            pd.to_numeric(
                paired[f"{greedy_prediction_field}__full_donor_patch"],
                errors="coerce",
            )
            - pd.to_numeric(
                paired[f"{greedy_prediction_field}__self_patch"],
                errors="coerce",
            )
        )
        paired["greedy_donor_adoption_fraction"] = (
            paired["greedy_count_shift"] / paired["donor_offset"]
        )
    if "absolute_error__full_donor_patch" in paired.columns:
        paired["greedy_absolute_error_increase"] = (
            pd.to_numeric(
                paired["absolute_error__full_donor_patch"], errors="coerce"
            )
            - pd.to_numeric(paired["absolute_error__self_patch"], errors="coerce")
        )
    if "exact_count__full_donor_patch" in paired.columns:
        paired["greedy_exact_loss"] = (
            pd.to_numeric(
                paired["exact_count__self_patch"], errors="coerce"
            ).astype(float)
            - pd.to_numeric(
                paired["exact_count__full_donor_patch"], errors="coerce"
            ).astype(float)
        )
    if "invalid_count_output__full_donor_patch" in paired.columns:
        paired["greedy_invalid_increase"] = (
            pd.to_numeric(
                paired["invalid_count_output__full_donor_patch"], errors="coerce"
            ).astype(float)
            - pd.to_numeric(
                paired["invalid_count_output__self_patch"], errors="coerce"
            ).astype(float)
        )
    paired["panel"] = panel
    return paired


def _geometry_eligibility(panel: str, trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()
    if "exclusion_reason" not in trials.columns:
        trials["exclusion_reason"] = ""
    identity = ["model_label", "pair_sha256", "patch_geometry"]
    grouped = (
        trials.groupby(identity, as_index=False)
        .agg(
            status=("status", lambda values: "ok" if set(values) == {"ok"} else "not_applicable"),
            exclusion_reason=(
                "exclusion_reason",
                lambda values: next(
                    (str(value) for value in values if pd.notna(value)), ""
                ),
            ),
        )
    )
    result = (
        grouped.groupby(["model_label", "patch_geometry", "status"], as_index=False)
        .agg(pair_count=("pair_sha256", "nunique"))
    )
    result["panel"] = panel
    return result


def _full_state_summaries(
    paired_by_panel: dict[str, pd.DataFrame],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    metrics = [
        "expected_count_shift",
        "donor_adoption_fraction",
        "correct_margin_damage",
        "correct_probability_damage",
        "correct_log_score_damage",
        "candidate_prediction_shift",
        "answer_query_displacement",
        "downstream_progress_displacement",
    ]
    rows: list[dict[str, Any]] = []
    for panel, paired in paired_by_panel.items():
        intersection_ids = set.intersection(
            *(
                set(paired.loc[paired["patch_geometry"].eq(geometry), "pair_sha256"])
                for geometry in GEOMETRIES
            )
        )
        scopes = {
            "all_eligible": paired,
            "four_geometry_intersection": paired.loc[
                paired["pair_sha256"].isin(intersection_ids)
            ],
        }
        for scope, active in scopes.items():
            if active.empty:
                continue
            for geometry in GEOMETRIES:
                geometry_frame = active.loc[active["patch_geometry"].eq(geometry)]
                if geometry_frame.empty:
                    continue
                offsets: list[int | str] = ["all", *sorted(geometry_frame["donor_offset"].unique())]
                for offset in offsets:
                    frame = (
                        geometry_frame
                        if offset == "all"
                        else geometry_frame.loc[geometry_frame["donor_offset"].eq(offset)]
                    )
                    rows.extend(
                        _seed_equal_summary(
                            frame,
                            grouping={
                                "panel": panel,
                                "sample_scope": scope,
                                "patch_geometry": geometry,
                                "donor_offset": offset,
                            },
                            metrics=metrics
                            + [
                                metric
                                for metric in (
                                    "greedy_count_shift",
                                    "greedy_donor_adoption_fraction",
                                    "greedy_absolute_error_increase",
                                    "greedy_exact_loss",
                                    "greedy_invalid_increase",
                                )
                                if metric in frame
                            ],
                            bootstrap_samples=bootstrap_samples,
                            random_seed=random_seed + len(rows),
                        )
                    )
    return pd.DataFrame(rows)


def _source_mask_effects(trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials.loc[trials["status"].eq("ok")].copy()
    identity = ["model_label", "seed", "request_id", "gold_count"]
    outcomes = [
        "correct_count_margin",
        "correct_count_probability",
        "correct_count_log_score",
        "expected_count_absolute_error",
        "predicted_count_among_candidates",
    ]
    wide = ok.pivot(index=identity, columns="condition", values=outcomes)
    wide.columns = [f"{outcome}__{condition}" for outcome, condition in wide.columns]
    frame = wide.reset_index()
    definitions = {
        "trace": ("block_trace_items", "block_trace_items_matched_control"),
        "prompt": ("block_prompt_records", "block_prompt_records_matched_control"),
    }
    for source, (treatment, control) in definitions.items():
        frame[f"{source}_margin_damage"] = (
            frame["correct_count_margin__clean"]
            - frame[f"correct_count_margin__{treatment}"]
        )
        frame[f"{source}_margin_specificity"] = (
            frame[f"correct_count_margin__{control}"]
            - frame[f"correct_count_margin__{treatment}"]
        )
        frame[f"{source}_error_increase"] = (
            frame[f"expected_count_absolute_error__{treatment}"]
            - frame["expected_count_absolute_error__clean"]
        )
        frame[f"{source}_error_specificity"] = (
            frame[f"expected_count_absolute_error__{treatment}"]
            - frame[f"expected_count_absolute_error__{control}"]
        )
        frame[f"{source}_added_candidate_error"] = (
            frame["predicted_count_among_candidates__clean"].eq(frame["gold_count"])
            & frame[f"predicted_count_among_candidates__{treatment}"].ne(
                frame["gold_count"]
            )
        )
    return frame


def _source_mask_summary(
    effects: pd.DataFrame, *, bootstrap_samples: int, random_seed: int
) -> pd.DataFrame:
    metrics = [
        f"{source}_{suffix}"
        for source in ("trace", "prompt")
        for suffix in (
            "margin_damage",
            "margin_specificity",
            "error_increase",
            "error_specificity",
        )
    ]
    return pd.DataFrame(
        _seed_equal_summary(
            effects,
            grouping={"mask_application": "answer_query_only_all_heads"},
            metrics=metrics,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
    )


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    completion = json.loads(
        (root / "followup_complete.json").read_text(encoding="utf-8")
    )
    if completion.get("status") != "PASS":
        raise ValueError("Follow-up supervisor audit is not PASS")
    paired_by_panel: dict[str, pd.DataFrame] = {}
    eligibility = []
    for panel, directory in FULL_STATE_DIRS.items():
        trials = _read_shards(root / directory)
        paired_by_panel[panel] = _paired_full_state(panel, trials)
        eligibility.append(_geometry_eligibility(panel, trials))
    source_trials = _read_shards(root / SOURCE_MASK_DIR)
    if set(source_trials["mask_scope"]) != {"answer_query_only"}:
        raise ValueError("Source-mask analysis received a non-query-only run")
    source_effects = _source_mask_effects(source_trials)

    output = args.output.resolve()
    full_pair_effects = pd.concat(paired_by_panel.values(), ignore_index=True)
    full_summary = _full_state_summaries(
        paired_by_panel,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    source_summary = _source_mask_summary(
        source_effects,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed + 10_000,
    )
    eligibility_frame = pd.concat(eligibility, ignore_index=True)
    _atomic_csv(output / "full_state_pair_effects.csv", full_pair_effects)
    _atomic_csv(output / "full_state_seed_equal_summary.csv", full_summary)
    _atomic_csv(output / "geometry_eligibility.csv", eligibility_frame)
    _atomic_csv(output / "source_mask_request_effects.csv", source_effects)
    _atomic_csv(output / "source_mask_seed_equal_summary.csv", source_summary)
    _atomic_json(
        output / "manifest.json",
        {
            "schema_version": "realistic_niah_v5_count_stream_followup_analysis_v1",
            "status": "PASS",
            "design_status": "development_only",
            "formal_inference_eligible": False,
            "model_label": completion["model_label"],
            "run_root": str(root),
            "bootstrap_samples": int(args.bootstrap_samples),
            "random_seed": int(args.random_seed),
            "middle_pairs_analyzed": int(
                paired_by_panel["middle"]["pair_sha256"].nunique()
            ),
            "terminal_pairs_analyzed": int(
                paired_by_panel["terminal_last"]["pair_sha256"].nunique()
            ),
            "source_mask_requests_analyzed": int(source_effects["request_id"].nunique()),
            "claim_scope": (
                "Full-state transfer is a sufficiency assay; source masks test "
                "all-head source-edge necessity at the answer query."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
