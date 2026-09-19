from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODELS = ("Qwen3-8B", "Gemma4-E4B")
VARIANTS = ("v4.1", "v4.2", "v4.3", "v4.4")
CONFIRMATION_SEEDS = tuple(range(1254, 1264))
DISCOVERY_SEEDS = tuple(range(1234, 1254))
STRICT_GREEDY_METRIC = "strict_greedy_complete_numeric_generation"


@dataclass(frozen=True)
class ScreenDesign:
    model: str
    stage: str
    root: Path
    design: dict[str, Any]


def _io_path(path: Path) -> Path:
    """Return an extended-length path for deep Windows run trees."""
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return Path(resolved)
    if resolved.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + resolved[2:])
    return Path("\\\\?\\" + resolved)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_layers(model: str) -> list[int]:
    if model == "Qwen3-8B":
        return [9, 18, 26]
    if model == "Gemma4-E4B":
        return [10, 20, 31]
    raise KeyError(model)


def _matches_screen_design(stage: str, model: str, design: dict[str, Any]) -> bool:
    common = (
        design.get("model_label") == model
        and design.get("behavior_metric") == STRICT_GREEDY_METRIC
        and design.get("confirmation_variants") == list(VARIANTS)
        and design.get("confirmation_seeds") == list(CONFIRMATION_SEEDS)
    )
    if not common:
        return False
    if stage == "ablation":
        return (
            design.get("family") == "generation_head_ablation_v1"
            and design.get("confirmation_counts") == [7, 8, 9, 10]
            and design.get("poolings") == ["span_end"]
            and design.get("scopes") == ["answer_query"]
            and design.get("top_ns") == [4, 8]
            and int(design.get("random_replicates", -1)) == 1
        )
    if stage == "patching":
        return (
            design.get("family") == "generation_residual_patching_v1"
            and design.get("confirmation_counts") == list(range(1, 11))
            and design.get("layers") == _expected_layers(model)
            and design.get("sites") == ["toggled_needle_end"]
            and design.get("needle_protocols") == ["cumulative_from_layer"]
            and design.get("directed_count_pairs")
            == [[5, 6], [6, 5], [7, 8], [8, 7], [9, 10], [10, 9]]
        )
    if stage == "steering":
        return (
            design.get("family") == "geometric_steering_v1"
            and design.get("confirmation_counts") == list(range(1, 11))
            and design.get("discovery_seeds") == list(DISCOVERY_SEEDS)
            and design.get("layers") == _expected_layers(model)
            and design.get("methods") == ["centroid_delta"]
            and design.get("alphas") == [1.0]
            and int(design.get("orthogonal_random_replicates", -1)) == 1
            and design.get("directed_count_pairs")
            == [[7, 8], [8, 7], [9, 10], [10, 9], [5, 10], [10, 5]]
        )
    raise KeyError(stage)


def find_screen_designs(run_root: str | Path) -> dict[str, dict[str, ScreenDesign]]:
    run_root = Path(run_root).resolve()
    families = {
        "ablation": "generation_head_ablation_v1",
        "patching": "generation_residual_patching_v1",
        "steering": "geometric_steering_v1",
    }
    selected: dict[str, dict[str, ScreenDesign]] = {}
    for model in MODELS:
        selected[model] = {}
        causal_root = run_root / model / "numeric" / "causal"
        for stage, family in families.items():
            matches: list[ScreenDesign] = []
            for root in sorted((causal_root / family).glob("design_*")):
                design_path = root / "design.json"
                if not design_path.is_file():
                    continue
                design = _read_json(design_path)
                if _matches_screen_design(stage, model, design):
                    matches.append(ScreenDesign(model, stage, root, design))
            _require(
                len(matches) == 1,
                f"Expected one {model}/{stage} screen design, found {len(matches)}",
            )
            selected[model][stage] = matches[0]
    return selected


def _validate_behavior_alignment(
    detail: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    context: str,
) -> None:
    baseline_columns = [
        "stimulus_id",
        "design_variant",
        "seed",
        "gold_count",
        "baseline_outcome",
        "baseline_is_correct",
        "baseline_format_valid",
        "baseline_predicted_count",
        "baseline_count_error",
    ]
    missing = sorted(set(baseline_columns) - set(detail.columns))
    _require(not missing, f"{context}: missing baseline columns {missing}")
    baseline = detail[baseline_columns].drop_duplicates()
    _require(
        not baseline.duplicated("stimulus_id").any(),
        f"{context}: inconsistent baseline fields for a stimulus",
    )
    expected = labels[
        [
            "stimulus_id",
            "design_variant",
            "seed",
            "gold_count",
            "outcome_group",
            "is_correct",
            "format_valid",
            "parsed_count",
            "count_error",
        ]
    ].copy()
    merged = baseline.merge(
        expected,
        on="stimulus_id",
        how="left",
        validate="one_to_one",
        suffixes=("_causal", "_behavior"),
        indicator=True,
    )
    _require(
        (merged["_merge"] == "both").all(),
        f"{context}: at least one causal baseline lacks a behavior label",
    )
    comparisons = {
        "design_variant": ("design_variant_causal", "design_variant_behavior"),
        "seed": ("seed_causal", "seed_behavior"),
        "gold_count": ("gold_count_causal", "gold_count_behavior"),
        "outcome": ("baseline_outcome", "outcome_group"),
        "is_correct": ("baseline_is_correct", "is_correct"),
        "format_valid": ("baseline_format_valid", "format_valid"),
        "predicted_count": ("baseline_predicted_count", "parsed_count"),
        "count_error": ("baseline_count_error", "count_error"),
    }
    for label, (left, right) in comparisons.items():
        left_values = merged[left]
        right_values = merged[right]
        equal = left_values.eq(right_values) | (
            left_values.isna() & right_values.isna()
        )
        _require(
            bool(equal.all()), f"{context}: baseline {label} disagrees with behavior"
        )


def _validate_generation_rows(detail: pd.DataFrame, *, context: str) -> None:
    _require(len(detail) > 0, f"{context}: detail table is empty")
    _require(
        set(detail["behavior_metric"].dropna().unique()) == {STRICT_GREEDY_METRIC},
        f"{context}: behavior metric is not strict greedy numeric generation",
    )
    valid = detail["patched_format_valid"].astype(bool)
    correct = detail["patched_is_correct"].astype(bool)
    expected_outcome = np.where(correct, "correct", np.where(valid, "wrong", "invalid"))
    _require(
        np.array_equal(
            detail["patched_outcome"].astype(str).to_numpy(), expected_outcome
        ),
        f"{context}: patched outcome labels are inconsistent",
    )
    valid_rows = detail[valid].copy()
    predicted = pd.to_numeric(valid_rows["patched_predicted_count"], errors="coerce")
    gold = pd.to_numeric(valid_rows["gold_count"], errors="coerce")
    error = pd.to_numeric(valid_rows["patched_count_error"], errors="coerce")
    _require(
        predicted.notna().all(), f"{context}: valid rows contain nonnumeric predictions"
    )
    _require(
        np.allclose((predicted - gold).to_numpy(), error.to_numpy()),
        f"{context}: patched count errors do not match final predictions",
    )


def _validate_csv_capture(
    design: ScreenDesign,
    *,
    capture_subdir: str,
    expected_index_rows: int,
    expected_rows_per_shard: int,
    expected_detail_rows: int,
    unique_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    capture_root = design.root / capture_subdir
    index_path = capture_root / "capture_index.jsonl"
    index = _read_jsonl(index_path)
    context = f"{design.model}/{design.stage}"
    _require(
        len(index) == expected_index_rows,
        f"{context}: expected {expected_index_rows} index rows, found {len(index)}",
    )
    shard_paths = [capture_root / str(row["shard_path"]) for row in index]
    _require(
        len(set(shard_paths)) == len(shard_paths), f"{context}: duplicate shard path"
    )
    shard_rows = 0
    for row, shard in zip(index, shard_paths):
        io_shard = _io_path(shard)
        _require(io_shard.is_file(), f"{context}: missing shard {shard}")
        frame = pd.read_csv(io_shard, compression="gzip")
        _require(
            len(frame) == expected_rows_per_shard,
            f"{context}: {shard.name} has {len(frame)} rows, expected {expected_rows_per_shard}",
        )
        _require(
            int(row["rows"]) == len(frame),
            f"{context}: index row count disagrees for {shard.name}",
        )
        shard_rows += len(frame)
    detail_path = design.root / "detail.csv.gz"
    detail = pd.read_csv(detail_path, compression="gzip")
    _require(
        len(detail) == expected_detail_rows == shard_rows,
        f"{context}: expected {expected_detail_rows} detail rows, found {len(detail)}",
    )
    missing_unique = sorted(set(unique_columns) - set(detail.columns))
    _require(
        not missing_unique, f"{context}: missing identity columns {missing_unique}"
    )
    _require(
        not detail.duplicated(unique_columns).any(),
        f"{context}: duplicate intervention identities",
    )
    return detail, {
        "capture_index_rows": len(index),
        "shards": len(shard_paths),
        "rows_per_shard": expected_rows_per_shard,
        "detail_rows": len(detail),
    }


def _validate_discovery_npz(design: ScreenDesign) -> dict[str, Any]:
    capture_root = design.root / "discovery_capture"
    index = _read_jsonl(capture_root / "capture_index.jsonl")
    context = f"{design.model}/steering-discovery"
    _require(
        len(index) == 800, f"{context}: expected 800 index rows, found {len(index)}"
    )
    expected_layers = np.asarray(_expected_layers(design.model), dtype=np.int64)
    shard_paths = [capture_root / str(row["shard_path"]) for row in index]
    _require(len(set(shard_paths)) == 800, f"{context}: duplicate shard path")
    hidden_size: int | None = None
    for shard in shard_paths:
        io_shard = _io_path(shard)
        _require(io_shard.is_file(), f"{context}: missing shard {shard}")
        with np.load(io_shard, allow_pickle=False) as payload:
            _require(
                set(payload.files)
                == {
                    "layer_indices",
                    "query_states",
                    "query_position",
                    "sequence_length",
                },
                f"{context}: unexpected NPZ schema in {shard.name}",
            )
            layers = np.asarray(payload["layer_indices"], dtype=np.int64)
            states = np.asarray(payload["query_states"])
            _require(
                np.array_equal(layers, expected_layers), f"{context}: layer mismatch"
            )
            _require(
                states.ndim == 2 and states.shape[0] == 3, f"{context}: bad state shape"
            )
            _require(np.isfinite(states).all(), f"{context}: nonfinite query state")
            if hidden_size is None:
                hidden_size = int(states.shape[1])
            _require(
                states.shape[1] == hidden_size, f"{context}: inconsistent hidden size"
            )
    return {
        "capture_index_rows": len(index),
        "npz_shards": len(shard_paths),
        "layers": expected_layers.tolist(),
        "hidden_size": hidden_size,
    }


def _table_row_count(path: Path, *, required_columns: set[str]) -> int:
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    _require(not missing, f"{path}: missing required columns {missing}")
    _require(len(frame) > 0, f"{path}: empty table")
    return len(frame)


def _mean_bool(series: pd.Series) -> float:
    return float(series.astype(bool).mean())


def _stage_effects(stage: str, detail: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if stage == "ablation":
        for (top_n, condition), frame in detail.groupby(
            ["top_n", "condition"], sort=True
        ):
            rows.append(
                {
                    "top_n": int(top_n),
                    "condition": str(condition),
                    "examples": int(len(frame)),
                    "patched_accuracy": _mean_bool(frame["patched_is_correct"]),
                    "prediction_changed_rate": _mean_bool(frame["prediction_changed"]),
                    "mean_generated_count_shift": float(
                        frame["generated_count_shift"].mean()
                    ),
                    "mean_absolute_error_delta": float(
                        frame["absolute_error_delta"].mean()
                    ),
                }
            )
    elif stage == "patching":
        detail = detail.copy()
        direction_sign = np.sign(pd.to_numeric(detail["gold_count_offset"]))
        detail["signed_generated_shift"] = (
            pd.to_numeric(detail["generated_count_shift"]) * direction_sign
        )
        for (layer, direction), frame in detail.groupby(
            ["start_layer", "direction"], sort=True
        ):
            rows.append(
                {
                    "layer": int(layer),
                    "direction": str(direction),
                    "examples": int(len(frame)),
                    "prediction_changed_rate": _mean_bool(frame["prediction_changed"]),
                    "moved_toward_donor_gold_rate": _mean_bool(
                        frame["moved_toward_donor_gold"]
                    ),
                    "follows_donor_gold_rate": _mean_bool(frame["follows_donor_gold"]),
                    "mean_generated_count_shift": float(
                        frame["generated_count_shift"].mean()
                    ),
                    "mean_direction_aligned_shift": float(
                        frame["signed_generated_shift"].mean()
                    ),
                }
            )
    elif stage == "steering":
        detail = detail.copy()
        intended_sign = np.sign(pd.to_numeric(detail["intended_count_shift"]))
        detail["signed_generated_shift"] = (
            pd.to_numeric(detail["generated_count_shift"]) * intended_sign
        )
        for (layer, condition), frame in detail.groupby(
            ["layer", "condition"], sort=True
        ):
            rows.append(
                {
                    "layer": int(layer),
                    "condition": str(condition),
                    "examples": int(len(frame)),
                    "prediction_changed_rate": _mean_bool(frame["prediction_changed"]),
                    "target_hit_rate": _mean_bool(frame["follows_donor_gold"]),
                    "moved_toward_target_rate": _mean_bool(
                        frame["moved_toward_donor_gold"]
                    ),
                    "mean_generated_count_shift": float(
                        frame["generated_count_shift"].mean()
                    ),
                    "mean_direction_aligned_shift": float(
                        frame["signed_generated_shift"].mean()
                    ),
                }
            )
    else:
        raise KeyError(stage)
    return rows


def audit_screen_8h(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    status = _read_json(run_root / "causal_campaign.status")
    marker = _read_json(run_root / "causal_screen_8h.complete")
    _require(status.get("state") == "COMPLETE", "Campaign status is not COMPLETE")
    _require(marker.get("state") == "COMPLETE", "Completion marker is not COMPLETE")
    designs = find_screen_designs(run_root)
    report: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_causal_screen_audit_v1",
        "run_root": str(run_root),
        "profile": status.get("profile"),
        "campaign_updated_utc": status.get("updated_utc"),
        "models": {},
    }
    for model in MODELS:
        labels = pd.read_csv(
            run_root
            / model
            / "numeric"
            / "behavior"
            / "capture"
            / "generation_labels.csv"
        )
        model_report: dict[str, Any] = {}

        ablation = designs[model]["ablation"]
        ablation_detail, ablation_counts = _validate_csv_capture(
            ablation,
            capture_subdir="capture",
            expected_index_rows=160,
            expected_rows_per_shard=4,
            expected_detail_rows=640,
            unique_columns=[
                "stimulus_id",
                "pooling",
                "scope",
                "condition",
                "top_n",
                "random_replicate",
            ],
        )
        _validate_generation_rows(ablation_detail, context=f"{model}/ablation")
        _validate_behavior_alignment(
            ablation_detail, labels, context=f"{model}/ablation"
        )
        _require(
            set(ablation_detail["pooling"]) == {"span_end"}
            and set(ablation_detail["scope"]) == {"answer_query"},
            f"{model}/ablation: unexpected pooling or scope",
        )
        ablation_counts.update(
            {
                "design": str(ablation.root.relative_to(run_root)),
                "summary_rows": _table_row_count(
                    ablation.root / "summary.csv",
                    required_columns={"condition", "top_n", "prediction_changed_rate"},
                ),
                "control_rows": _table_row_count(
                    ablation.root / "broad_vs_layer_matched_random.csv",
                    required_columns={
                        "metric",
                        "ranked_minus_random_mean",
                        "ci95_low",
                        "ci95_high",
                    },
                ),
                "effects": _stage_effects("ablation", ablation_detail),
            }
        )
        model_report["ablation"] = ablation_counts

        patching = designs[model]["patching"]
        patch_detail, patch_counts = _validate_csv_capture(
            patching,
            capture_subdir="capture",
            expected_index_rows=40,
            expected_rows_per_shard=18,
            expected_detail_rows=720,
            unique_columns=[
                "receiver_stimulus_id",
                "donor_stimulus_id",
                "site",
                "patch_protocol",
                "start_layer",
            ],
        )
        _validate_generation_rows(patch_detail, context=f"{model}/patching")
        _validate_behavior_alignment(patch_detail, labels, context=f"{model}/patching")
        _require(
            set(patch_detail["status"]) == {"ok"}, f"{model}/patching: skipped rows"
        )
        _require(
            set(patch_detail["site"]) == {"toggled_needle_end"}
            and set(patch_detail["patch_protocol"]) == {"cumulative_from_layer"},
            f"{model}/patching: unexpected intervention site or protocol",
        )
        patch_counts.update(
            {
                "design": str(patching.root.relative_to(run_root)),
                "successful_rows": int((patch_detail["status"] == "ok").sum()),
                "skipped_rows": int((patch_detail["status"] != "ok").sum()),
                "summary_rows": _table_row_count(
                    patching.root / "summary.csv",
                    required_columns={
                        "start_layer",
                        "direction",
                        "moved_toward_donor_gold_rate",
                    },
                ),
                "effects": _stage_effects("patching", patch_detail),
            }
        )
        model_report["patching"] = patch_counts

        steering = designs[model]["steering"]
        discovery_counts = _validate_discovery_npz(steering)
        steering_detail, steering_counts = _validate_csv_capture(
            steering,
            capture_subdir="confirmation_capture",
            expected_index_rows=40,
            expected_rows_per_shard=36,
            expected_detail_rows=1440,
            unique_columns=[
                "receiver_stimulus_id",
                "target_stimulus_id",
                "layer",
                "steering_method",
                "condition",
                "random_replicate",
                "alpha",
            ],
        )
        _validate_generation_rows(steering_detail, context=f"{model}/steering")
        _validate_behavior_alignment(
            steering_detail, labels, context=f"{model}/steering"
        )
        _require(
            set(steering_detail["site"]) == {"answer_query"}
            and set(steering_detail["steering_method"]) == {"centroid_delta"}
            and set(steering_detail["condition"])
            == {"geometric", "orthogonal_norm_matched_random"},
            f"{model}/steering: unexpected intervention design",
        )
        geometry = pd.read_csv(steering.root / "centroid_geometry_summary.csv")
        _require(len(geometry) == 12, f"{model}/steering: expected 12 geometry rows")
        _require(
            np.isfinite(geometry.select_dtypes(include=[np.number]).to_numpy()).all(),
            f"{model}/steering: nonfinite centroid geometry",
        )
        steering_counts.update(
            {
                "design": str(steering.root.relative_to(run_root)),
                "discovery": discovery_counts,
                "summary_rows": _table_row_count(
                    steering.root / "summary.csv",
                    required_columns={"condition", "layer", "moved_toward_target_rate"},
                ),
                "control_rows": _table_row_count(
                    steering.root / "geometric_vs_random.csv",
                    required_columns={
                        "metric",
                        "geometric_minus_random_mean",
                        "ci95_low",
                        "ci95_high",
                    },
                ),
                "geometry_rows": len(geometry),
                "effects": _stage_effects("steering", steering_detail),
            }
        )
        model_report["steering"] = steering_counts
        report["models"][model] = model_report

    log_root = run_root / "logs"
    error_terms = ("Traceback", "CUDA out of memory", "OOM", "FAILED")
    error_hits: list[str] = []
    for path in sorted(log_root.glob("causal_screen_8h*.log")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if any(term in line for term in error_terms):
                error_hits.append(f"{path.name}:{line_number}:{line}")
    _require(not error_hits, f"Causal logs contain errors: {error_hits[:3]}")
    report["log_error_hits"] = error_hits
    report["validated"] = True
    return report


def write_audit(report: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
