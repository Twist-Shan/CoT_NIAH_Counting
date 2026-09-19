from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .causal_v2 import (
    confirmation_statistics,
    head_ablation_confirmation_seed_effects,
    head_ablation_confirmation_statistics,
)
from .causal_v2_analysis import (
    paired_control_adjusted_transport,
    summarize_ablation_sweep,
    summarize_layer_k_transport,
)


SCHEMA_VERSION = "realistic_niah_v4_4_correct_only_slices_v1"
MODEL_ORDER = ("Qwen3-8B", "Gemma4-E4B")
PATCH_FAMILIES = ("prompt_patching", "answer_patching")
PATCH_CONDITION_COLUMNS = (
    "model_label",
    "site",
    "patch_protocol",
    "start_layer",
    "k",
)
ABLATION_SEED_EFFECT_COLUMNS = (
    "model_label",
    "head_bank",
    "top_n",
    "seed",
    "examples",
    "ranked_valid_rate",
    "random_valid_rate",
    "ranked_minus_random_accuracy_delta",
    "ranked_minus_random_absolute_error_delta",
    "ranked_minus_random_prediction_changed",
)
ABLATION_STATISTIC_COLUMNS = (
    "model_label",
    "head_bank",
    "top_n",
    "metric",
    "is_primary_endpoint",
    "harmful_direction",
    "seeds",
    "ranked_minus_random_mean",
    "ci95_low",
    "ci95_high",
    "exact_sign_flip_p",
    "harmful_seed_fraction",
    "nonopposing_seed_fraction",
    "bootstrap_repetitions",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def clean_correct_patching_rows(detail: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose receiver, semantic donor, and actual state source are correct.

    The semantic donor is the target-count example for the whole matched pair.  The
    actual state source differs only for controls such as ``same_count_seed``.  A
    self-patch therefore remains eligible whenever the receiver and semantic donor
    are both correct, while a same-count control is retained only if its state source
    is also correct.
    """

    required = {
        "baseline_is_correct",
        "baseline_format_valid",
        "donor_baseline_outcome",
        "state_donor_baseline_outcome",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Patching detail is missing correctness columns: {missing}")
    eligible = (
        _as_bool(detail["baseline_is_correct"])
        & _as_bool(detail["baseline_format_valid"])
        & detail["donor_baseline_outcome"].astype(str).eq("correct")
        & detail["state_donor_baseline_outcome"].astype(str).eq("correct")
    )
    result = detail.loc[eligible].copy()
    if "status" in result.columns:
        result = result[result["status"].astype(str).eq("ok")].copy()
    return result.reset_index(drop=True)


def clean_correct_ablation_rows(detail: pd.DataFrame) -> pd.DataFrame:
    """Return ablation rows whose unmodified complete generation was correct."""

    required = {"baseline_is_correct", "baseline_format_valid"}
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Ablation detail is missing correctness columns: {missing}")
    eligible = _as_bool(detail["baseline_is_correct"]) & _as_bool(
        detail["baseline_format_valid"]
    )
    return detail.loc[eligible].copy().reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip:
        frame.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one path matching {pattern!r} under {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


@dataclass
class SourceLedger:
    paths: dict[str, Path]

    def __init__(self) -> None:
        self.paths = {}

    def add(self, label: str, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.paths[label] = resolved
        return resolved

    def csv(self, label: str, path: Path) -> pd.DataFrame:
        return pd.read_csv(self.add(label, path), low_memory=False)

    def json(self, label: str, path: Path) -> dict[str, Any]:
        return json.loads(self.add(label, path).read_text(encoding="utf-8"))

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "source_label": label,
                "source_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for label, path in sorted(self.paths.items())
        ]


def _copy_overall(
    ledger: SourceLedger,
    *,
    label: str,
    source: Path,
    destination: Path,
) -> None:
    checked = ledger.add(label, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checked, destination)
    if _sha256(checked) != _sha256(destination):
        raise RuntimeError(f"Overall summary copy failed checksum validation: {source}")


def _patch_availability(
    detail: pd.DataFrame,
    correct: pd.DataFrame,
    *,
    model: str,
    family: str,
    phase: str,
) -> list[dict[str, Any]]:
    treatment = detail[detail["condition"].astype(str).eq("donor_transport")]
    eligible = correct[correct["condition"].astype(str).eq("donor_transport")]
    identity = ["seed", "receiver_count", "donor_count", "k", "target_direction"]
    all_pairs = treatment[identity].drop_duplicates()
    correct_pairs = eligible[identity].drop_duplicates()
    keys = sorted(
        set(
            zip(
                pd.to_numeric(all_pairs["k"], errors="raise").astype(int),
                all_pairs["target_direction"].astype(str),
            )
        )
    )
    rows: list[dict[str, Any]] = []
    for k, direction in keys:
        all_group = all_pairs[
            pd.to_numeric(all_pairs["k"], errors="raise").astype(int).eq(k)
            & all_pairs["target_direction"].astype(str).eq(direction)
        ]
        correct_group = correct_pairs[
            pd.to_numeric(correct_pairs["k"], errors="raise").astype(int).eq(k)
            & correct_pairs["target_direction"].astype(str).eq(direction)
        ]
        rows.append(
            {
                "model_label": model,
                "family": family,
                "phase": phase,
                "k": int(k),
                "target_direction": direction,
                "overall_pairs": int(len(all_group)),
                "clean_correct_pairs": int(len(correct_group)),
                "clean_correct_pair_fraction": (
                    float(len(correct_group) / len(all_group)) if len(all_group) else math.nan
                ),
                "clean_correct_seed_clusters": int(correct_group["seed"].nunique()),
            }
        )
    return rows


def _ablation_availability(
    detail: pd.DataFrame,
    correct: pd.DataFrame,
    *,
    model: str,
    phase: str,
) -> dict[str, Any]:
    identity = ["stimulus_id", "seed", "gold_count"]
    overall = detail[identity].drop_duplicates()
    selected = correct[identity].drop_duplicates()
    return {
        "model_label": model,
        "phase": phase,
        "overall_stimuli": int(len(overall)),
        "clean_correct_stimuli": int(len(selected)),
        "clean_correct_fraction": (
            float(len(selected) / len(overall)) if len(overall) else math.nan
        ),
        "clean_correct_seed_clusters": int(selected["seed"].nunique()),
        "clean_correct_counts": ";".join(
            str(value)
            for value in sorted(
                pd.to_numeric(selected["gold_count"], errors="raise")
                .astype(int)
                .unique()
                .tolist()
            )
        ),
    }


def _empty_ablation_seed_effects() -> pd.DataFrame:
    return pd.DataFrame(columns=ABLATION_SEED_EFFECT_COLUMNS)


def _empty_ablation_statistics() -> pd.DataFrame:
    return pd.DataFrame(columns=ABLATION_STATISTIC_COLUMNS)


def _process_patching(
    *,
    model: str,
    causal_root: Path,
    output: Path,
    ledger: SourceLedger,
    availability_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    bootstrap_repetitions: int,
) -> None:
    for family in PATCH_FAMILIES:
        overall_tables = causal_root / "analysis" / "tables"
        for source_name in (
            f"{family}_layer_k_summary.csv",
            f"{family}_confirmation_statistics.csv",
        ):
            _copy_overall(
                ledger,
                label=f"{model}.{family}.overall.{source_name}",
                source=overall_tables / source_name,
                destination=output
                / "overall"
                / model
                / "patching"
                / source_name,
            )

        for phase in ("screen", "confirmation"):
            stage = _one(causal_root / family, f"{phase}_*")
            detail_path = stage / "detail.csv.gz"
            detail = ledger.csv(f"{model}.{family}.{phase}.detail", detail_path)
            correct = clean_correct_patching_rows(detail)
            availability_rows.extend(
                _patch_availability(
                    detail,
                    correct,
                    model=model,
                    family=family,
                    phase=phase,
                )
            )
            relative = Path("clean_correct") / model / "patching" / family / phase
            detail_output = output / relative / "detail.clean_correct.csv.gz"
            _write_csv(correct, detail_output, gzip=True)
            paired = paired_control_adjusted_transport(correct, family=family)
            paired_output = output / relative / "paired_effects.clean_correct.csv.gz"
            _write_csv(paired, paired_output, gzip=True)
            summary = summarize_layer_k_transport(
                paired,
                family=family,
                bootstrap_repetitions=bootstrap_repetitions,
            )
            summary_output = output / relative / "layer_k_summary.clean_correct.csv"
            _write_csv(summary, summary_output)
            inventory_rows.extend(
                [
                    {
                        "scope": "clean_correct",
                        "model_label": model,
                        "family": family,
                        "phase": phase,
                        "artifact": "detail",
                        "path": str(detail_output.relative_to(output)),
                        "rows": int(len(correct)),
                    },
                    {
                        "scope": "clean_correct",
                        "model_label": model,
                        "family": family,
                        "phase": phase,
                        "artifact": "paired_effects",
                        "path": str(paired_output.relative_to(output)),
                        "rows": int(len(paired)),
                    },
                    {
                        "scope": "clean_correct",
                        "model_label": model,
                        "family": family,
                        "phase": phase,
                        "artifact": "layer_k_summary",
                        "path": str(summary_output.relative_to(output)),
                        "rows": int(len(summary)),
                    },
                ]
            )
            if phase == "confirmation":
                statistics = confirmation_statistics(
                    correct,
                    family=family,
                    bootstrap_repetitions=bootstrap_repetitions,
                )
                if not statistics.empty:
                    statistics.insert(0, "analysis_scope", "posthoc_clean_correct_pair_subset")
                    statistics.insert(1, "selection_basis", "frozen_overall_screen_selection")
                statistics_output = (
                    output / relative / "confirmation_statistics.clean_correct.csv"
                )
                _write_csv(statistics, statistics_output)
                inventory_rows.append(
                    {
                        "scope": "clean_correct",
                        "model_label": model,
                        "family": family,
                        "phase": phase,
                        "artifact": "confirmation_statistics",
                        "path": str(statistics_output.relative_to(output)),
                        "rows": int(len(statistics)),
                    }
                )


def _process_ablation(
    *,
    model: str,
    causal_root: Path,
    ablation_confirmation_root: Path,
    output: Path,
    ledger: SourceLedger,
    availability_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    overall_tables = causal_root / "analysis" / "tables"
    _copy_overall(
        ledger,
        label=f"{model}.ablation.overall.top_k_sweep",
        source=overall_tables / "ablation_top_k_sweep.csv",
        destination=output
        / "overall"
        / model
        / "ablation"
        / "ablation_top_k_sweep.csv",
    )
    discovery_stage = _one(causal_root / "answer_query_head_ablation", "screen_*")
    discovery = ledger.csv(
        f"{model}.ablation.discovery.detail", discovery_stage / "detail.csv.gz"
    )
    discovery_correct = clean_correct_ablation_rows(discovery)
    availability_rows.append(
        _ablation_availability(
            discovery,
            discovery_correct,
            model=model,
            phase="discovery_top_n_sweep",
        )
    )
    discovery_relative = Path("clean_correct") / model / "ablation" / "discovery"
    discovery_detail_output = output / discovery_relative / "detail.clean_correct.csv.gz"
    _write_csv(discovery_correct, discovery_detail_output, gzip=True)
    if discovery_correct.empty:
        discovery_summary = pd.DataFrame(
            columns=[
                "model_label",
                "head_bank",
                "top_n",
                "examples",
                "seeds",
                "accuracy_effect",
                "absolute_error_effect",
                "prediction_change_effect",
                "ranked_valid_rate",
                "random_overlap_mean",
            ]
        )
    else:
        discovery_summary = summarize_ablation_sweep(discovery_correct)
    discovery_summary_output = (
        output / discovery_relative / "top_k_sweep.clean_correct.csv"
    )
    _write_csv(discovery_summary, discovery_summary_output)

    confirm_causal = (
        ablation_confirmation_root / model / "numeric" / "causal_v2"
    )
    confirm_stage = _one(
        confirm_causal / "answer_query_head_ablation", "confirmation_*"
    )
    for name in (
        "head_ablation_confirmation_statistics.csv",
        "head_ablation_seed_effects.csv",
    ):
        _copy_overall(
            ledger,
            label=f"{model}.ablation.frozen.overall.{name}",
            source=confirm_stage / "analysis" / name,
            destination=output / "overall" / model / "ablation" / name,
        )
    confirm_detail = ledger.csv(
        f"{model}.ablation.frozen.detail", confirm_stage / "detail.csv.gz"
    )
    confirm_correct = clean_correct_ablation_rows(confirm_detail)
    confirm_availability = _ablation_availability(
        confirm_detail,
        confirm_correct,
        model=model,
        phase="frozen_model_specific_confirmation",
    )
    availability_rows.append(confirm_availability)
    confirm_relative = Path("clean_correct") / model / "ablation" / "confirmation"
    confirm_detail_output = output / confirm_relative / "detail.clean_correct.csv.gz"
    _write_csv(confirm_correct, confirm_detail_output, gzip=True)
    if confirm_correct.empty:
        seed_effects = _empty_ablation_seed_effects()
        statistics = _empty_ablation_statistics()
    else:
        seed_effects = head_ablation_confirmation_seed_effects(confirm_correct)
        statistics = head_ablation_confirmation_statistics(
            seed_effects,
            bootstrap_repetitions=bootstrap_repetitions,
        )
    if not statistics.empty:
        statistics.insert(0, "analysis_scope", "posthoc_clean_correct_baseline_subset")
        statistics.insert(1, "selection_basis", "frozen_model_specific_bank")
        statistics.insert(
            2,
            "sufficient_seed_clusters",
            pd.to_numeric(statistics["seeds"], errors="raise").ge(7),
        )
    seed_output = output / confirm_relative / "seed_effects.clean_correct.csv"
    statistics_output = (
        output / confirm_relative / "confirmation_statistics.clean_correct.csv"
    )
    _write_csv(seed_effects, seed_output)
    _write_csv(statistics, statistics_output)
    inventory_rows.extend(
        [
            {
                "scope": "clean_correct",
                "model_label": model,
                "family": "head_ablation",
                "phase": "discovery_top_n_sweep",
                "artifact": "detail",
                "path": str(discovery_detail_output.relative_to(output)),
                "rows": int(len(discovery_correct)),
            },
            {
                "scope": "clean_correct",
                "model_label": model,
                "family": "head_ablation",
                "phase": "discovery_top_n_sweep",
                "artifact": "top_k_sweep",
                "path": str(discovery_summary_output.relative_to(output)),
                "rows": int(len(discovery_summary)),
            },
            {
                "scope": "clean_correct",
                "model_label": model,
                "family": "head_ablation",
                "phase": "frozen_model_specific_confirmation",
                "artifact": "detail",
                "path": str(confirm_detail_output.relative_to(output)),
                "rows": int(len(confirm_correct)),
            },
            {
                "scope": "clean_correct",
                "model_label": model,
                "family": "head_ablation",
                "phase": "frozen_model_specific_confirmation",
                "artifact": "seed_effects",
                "path": str(seed_output.relative_to(output)),
                "rows": int(len(seed_effects)),
            },
            {
                "scope": "clean_correct",
                "model_label": model,
                "family": "head_ablation",
                "phase": "frozen_model_specific_confirmation",
                "artifact": "confirmation_statistics",
                "path": str(statistics_output.relative_to(output)),
                "rows": int(len(statistics)),
            },
        ]
    )
    design = ledger.json(f"{model}.ablation.frozen.design", confirm_stage / "design.json")
    return {
        "model_label": model,
        "frozen_top_n": int(design["frozen_model_specific_top_n"]),
        "correct_confirmation_stimuli": int(
            confirm_availability["clean_correct_stimuli"]
        ),
        "correct_confirmation_seed_clusters": int(
            confirm_availability["clean_correct_seed_clusters"]
        ),
        "correct_only_confirmation_is_sufficient": bool(
            int(confirm_availability["clean_correct_seed_clusters"]) >= 7
        ),
    }


def _audit_output(output: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for model in MODEL_ORDER:
        for family in PATCH_FAMILIES:
            for phase in ("screen", "confirmation"):
                path = (
                    output
                    / "clean_correct"
                    / model
                    / "patching"
                    / family
                    / phase
                    / "detail.clean_correct.csv.gz"
                )
                frame = pd.read_csv(path, low_memory=False)
                valid = (
                    _as_bool(frame["baseline_is_correct"]).all()
                    and _as_bool(frame["baseline_format_valid"]).all()
                    and frame["donor_baseline_outcome"].astype(str).eq("correct").all()
                    and frame["state_donor_baseline_outcome"]
                    .astype(str)
                    .eq("correct")
                    .all()
                )
                check(
                    f"{model}.{family}.{phase}.all_rows_clean_correct",
                    bool(valid),
                    f"rows={len(frame)}",
                )
        for phase in ("discovery", "confirmation"):
            path = (
                output
                / "clean_correct"
                / model
                / "ablation"
                / phase
                / "detail.clean_correct.csv.gz"
            )
            frame = pd.read_csv(path, low_memory=False)
            valid = frame.empty or (
                _as_bool(frame["baseline_is_correct"]).all()
                and _as_bool(frame["baseline_format_valid"]).all()
            )
            check(
                f"{model}.ablation.{phase}.all_rows_clean_correct",
                bool(valid),
                f"rows={len(frame)}",
            )
    failures = [row for row in checks if not row["passed"]]
    return {
        "schema_version": f"{SCHEMA_VERSION}_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "error_count": len(failures),
        "checks": checks,
    }


def _write_sha256sums(output: Path) -> None:
    checksum_path = output / "SHA256SUMS"
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_correct_only_slices(
    *,
    qwen_run_root: Path,
    gemma_run_root: Path,
    ablation_confirmation_run_root: Path,
    output_root: Path,
    bootstrap_repetitions: int = 10_000,
) -> dict[str, Any]:
    """Build an immutable derived package without changing any source run."""

    if output_root.exists():
        raise FileExistsError(
            f"Derived output already exists and will not be overwritten: {output_root}"
        )
    if int(bootstrap_repetitions) < 100:
        raise ValueError("bootstrap_repetitions must be at least 100")
    output_root.mkdir(parents=True)
    ledger = SourceLedger()
    availability_rows: list[dict[str, Any]] = []
    ablation_availability_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    ablation_status: list[dict[str, Any]] = []
    run_roots = {
        "Qwen3-8B": qwen_run_root,
        "Gemma4-E4B": gemma_run_root,
    }
    for model in MODEL_ORDER:
        causal_root = run_roots[model] / model / "numeric" / "causal_v2"
        _process_patching(
            model=model,
            causal_root=causal_root,
            output=output_root,
            ledger=ledger,
            availability_rows=availability_rows,
            inventory_rows=inventory_rows,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        ablation_status.append(
            _process_ablation(
                model=model,
                causal_root=causal_root,
                ablation_confirmation_root=ablation_confirmation_run_root,
                output=output_root,
                ledger=ledger,
                availability_rows=ablation_availability_rows,
                inventory_rows=inventory_rows,
                bootstrap_repetitions=bootstrap_repetitions,
            )
        )
    _write_csv(pd.DataFrame(availability_rows), output_root / "patching_pair_availability.csv")
    _write_csv(
        pd.DataFrame(ablation_availability_rows),
        output_root / "ablation_baseline_availability.csv",
    )
    _write_csv(pd.DataFrame(inventory_rows), output_root / "artifact_inventory.csv")
    _write_csv(pd.DataFrame(ledger.rows()), output_root / "source_ledger.csv")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "derivation_only": True,
        "new_model_generations": 0,
        "overall_scope": "unfiltered audited source summaries copied under overall/",
        "clean_correct_patching_filter": {
            "receiver": "baseline_is_correct and baseline_format_valid",
            "semantic_donor": "donor_baseline_outcome == correct",
            "actual_state_source": "state_donor_baseline_outcome == correct",
        },
        "clean_correct_ablation_filter": (
            "baseline_is_correct and baseline_format_valid before intervention"
        ),
        "patching_inference_scope": (
            "post-hoc clean-correct conditional sensitivity analysis; frozen overall "
            "screen selections are not reselected"
        ),
        "ablation_inference_scope": (
            "post-hoc clean-correct conditional sensitivity analysis; frozen model-specific "
            "banks are not reselected"
        ),
        "ablation_status": {row["model_label"]: row for row in ablation_status},
        "bootstrap_repetitions": int(bootstrap_repetitions),
    }
    _write_json(output_root / "summary.json", summary)
    readme = """# V4.4 causal-v2 overall and clean-correct slices

This package is a deterministic, zero-generation derivation from the audited
Qwen and Gemma causal-v2 runs.  It does not replace or mutate the source runs.

## Directory scopes

- `overall/` contains copies of the audited unfiltered summary/statistics tables.
- `clean_correct/` contains filtered raw detail, paired effects, and recomputed
  summaries.  Patching requires a correct receiver, correct semantic donor, and
  correct actual state source.  Ablation requires a correct unmodified baseline.

The clean-correct analysis is explicitly post hoc.  Existing patching condition
selections and model-specific frozen head banks are reused without reselection.
Empty ablation tables are intentional when a model has no clean-correct source
stimuli; an empty table is not evidence of a zero effect.

See `patching_pair_availability.csv`, `ablation_baseline_availability.csv`,
`summary.json`, `audit/audit.json`, `source_ledger.csv`, and `SHA256SUMS` before
using any effect table.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    audit = _audit_output(output_root)
    _write_json(output_root / "audit" / "audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Correct-only slice audit failed: {audit['error_count']} errors")
    _write_sha256sums(output_root)
    return summary

