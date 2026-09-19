from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realistic_niah_v3_1.analysis import (
    accuracy_condition_table,
    behavior_tables,
    load_request_table,
    paired_mode_comparisons,
)
from realistic_niah_v3_1.cot_style import (
    build_blinded_annotation_samples,
    classify_request_table,
)
from realistic_niah_v3_1.laws import (
    CANDIDATE_BY_NAME,
    _analysis_frame,
    bootstrap_reselection_stability,
    bootstrap_selected_coefficients,
    configure_fit_backend,
    fit_candidate_grid,
    fit_backend_metadata,
    fit_law,
    leave_one_model_out_structure,
    nested_held_axis_validation,
    nested_seed_validation,
    probability_distribution_diagnostics,
)
from realistic_niah_v3_1.spec import EXPECTED_REQUESTS, PROTOCOL_VERSION


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_state(path: Path, stage: str, **details: Any) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "realistic_niah_v3_1_analysis_state_v1",
            "protocol_version": PROTOCOL_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **details,
        },
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression="gzip" if path.suffix == ".gz" else None,
    )
    temporary.replace(path)


def _distribution_outputs(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    details: list[pd.DataFrame] = []
    calibrations: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        outcome_model = str(row.outcome_model)
        if outcome_model not in {"binomial", "beta_binomial"}:
            continue
        mode = str(row.prompt_mode)
        mode_rows = requests.loc[requests["prompt_mode"] == mode]
        cells = accuracy_condition_table(mode_rows)
        fit = fit_law(
            cells,
            CANDIDATE_BY_NAME[str(row.candidate)],
            outcome_model,
        )
        detail, calibration, summary = probability_distribution_diagnostics(
            fit,
            cells,
        )
        detail.insert(0, "outcome_model", outcome_model)
        calibration.insert(0, "prompt_mode", mode)
        calibration.insert(1, "outcome_model", outcome_model)
        summary["prompt_mode"] = mode
        details.append(detail)
        calibrations.append(calibration)
        summaries.append(summary)
    return (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        pd.concat(calibrations, ignore_index=True) if calibrations else pd.DataFrame(),
        summaries,
    )


def _backend_parity_audit(
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    torch_device: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selection in selected.itertuples(index=False):
        mode = str(selection.prompt_mode)
        outcome_model = str(selection.outcome_model)
        candidate = CANDIDATE_BY_NAME[str(selection.candidate)]
        mode_rows = requests.loc[requests["prompt_mode"] == mode]
        frame = _analysis_frame(mode_rows, outcome_model, full=True)
        scipy_fit = fit_law(
            frame,
            candidate,
            outcome_model,
            backend="scipy",
            device="cpu",
        )
        torch_fit = fit_law(
            frame,
            candidate,
            outcome_model,
            levels=scipy_fit.levels,
            backend="torch",
            device=torch_device,
        )
        if outcome_model == "bias":
            scipy_prediction = scipy_fit.predict_bias(frame)
            torch_prediction = torch_fit.predict_bias(frame)
        else:
            scipy_prediction = scipy_fit.predict_probability(frame)
            torch_prediction = torch_fit.predict_probability(frame)
        prediction_difference = float(
            np.max(np.abs(torch_prediction - scipy_prediction))
        )
        coefficient_difference = float(
            np.max(np.abs(scipy_fit.beta - torch_fit.beta))
        )
        rows.append(
            {
                "prompt_mode": mode,
                "outcome_model": outcome_model,
                "candidate": candidate.name,
                "scipy_converged": bool(scipy_fit.converged),
                "torch_converged": bool(torch_fit.converged),
                "max_absolute_prediction_difference": prediction_difference,
                "max_absolute_coefficient_difference": coefficient_difference,
                "passed": bool(
                    scipy_fit.converged
                    and torch_fit.converged
                    and prediction_difference <= 5e-4
                    and coefficient_difference <= 5e-3
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered V3.1 behavior and empirical-law analysis."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--interaction-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--coefficient-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--validation-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--run-lomo", action="store_true")
    parser.add_argument("--lomo-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--run-bootstrap-reselection", action="store_true")
    parser.add_argument("--reselection-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--annotation-random-size", type=int, default=600)
    parser.add_argument("--annotation-challenge-size", type=int, default=200)
    parser.add_argument("--fit-backend", choices=("scipy", "torch"), default="scipy")
    parser.add_argument("--analysis-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    configure_fit_backend(backend=args.fit_backend, device=args.analysis_device)

    run_root = Path(args.run_root).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "analysis" / "v3_1_behavior_empirical_law"
    )
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    state_path = output / "analysis_state.json"
    _write_state(state_path, "loading", fit_backend=fit_backend_metadata())

    requests, sources = load_request_table(run_root)
    if len(requests) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected {EXPECTED_REQUESTS:,} requests, found {len(requests):,}"
        )
    _write_csv(tables / "request_level.csv.gz", requests)
    summary, accuracy_cells, bias_cells, outcomes = behavior_tables(requests)
    paired = paired_mode_comparisons(requests, bootstrap_replicates=2_000)
    _write_csv(tables / "model_mode_summary.csv", summary)
    _write_csv(tables / "accuracy_cells.csv", accuracy_cells)
    _write_csv(tables / "bias_cells.csv", bias_cells)
    _write_csv(tables / "outcome_composition.csv", outcomes)
    _write_csv(tables / "paired_mode_comparisons.csv", paired)
    _write_state(state_path, "behavior_complete", requests=len(requests))

    styles = classify_request_table(requests)
    style_requests = requests.merge(
        styles,
        on="request_id",
        validate="one_to_one",
        suffixes=("", "_style"),
    )
    _write_csv(tables / "cot_style_request_level.csv.gz", style_requests)
    style_summary = (
        style_requests.groupby(
            ["comparison_slot", "prompt_mode", "dominant_style"],
            sort=True,
            dropna=False,
        )
        .agg(
            requests=("request_id", "size"),
            parsed_exact_accuracy=("exact_count", "mean"),
            parse_rate=("parse_success", "mean"),
            format_compliance_rate=("format_compliant", "mean"),
            mean_signed_deviation=("signed_deviation", "mean"),
        )
        .reset_index()
    )
    _write_csv(tables / "cot_style_summary.csv", style_summary)
    random_sample, challenge_sample = build_blinded_annotation_samples(
        requests,
        styles,
        random_size=args.annotation_random_size,
        challenge_size=args.annotation_challenge_size,
    )
    _write_csv(tables / "cot_style_annotation_random_blinded.csv", random_sample)
    _write_csv(tables / "cot_style_annotation_challenge_blinded.csv", challenge_sample)
    _write_state(state_path, "cot_style_complete", classified=len(styles))

    comparison, selected, coefficients, interactions = fit_candidate_grid(
        requests,
        interaction_bootstrap_replicates=args.interaction_bootstrap_replicates,
    )
    _write_csv(tables / "candidate_comparison.csv", comparison)
    _write_csv(tables / "selected_laws.csv", selected)
    _write_csv(tables / "coefficients.csv", coefficients)
    _write_csv(tables / "interaction_tests.csv", interactions)
    if args.fit_backend == "torch":
        backend_parity = _backend_parity_audit(
            requests,
            selected,
            torch_device=args.analysis_device,
        )
        _write_csv(tables / "fit_backend_parity.csv", backend_parity)
        if not backend_parity["passed"].all():
            raise RuntimeError(
                "Torch fit backend failed the preregistered SciPy parity gate"
            )
    _write_state(
        state_path,
        "candidate_selection_complete",
        attempted_laws=len(comparison),
        selected_laws=len(selected),
    )

    nested_seed = nested_seed_validation(
        requests,
        interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
    )
    held_n = nested_held_axis_validation(
        requests,
        axis="N",
        interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
    )
    held_l = nested_held_axis_validation(
        requests,
        axis="L",
        interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
    )
    _write_csv(tables / "nested_held_seed_validation.csv", nested_seed)
    _write_csv(tables / "held_N_validation.csv", held_n)
    _write_csv(tables / "held_L_validation.csv", held_l)
    coefficient_bootstrap = bootstrap_selected_coefficients(
        requests,
        selected,
        replicates=args.coefficient_bootstrap_replicates,
    )
    _write_csv(tables / "selected_coefficient_bootstrap.csv", coefficient_bootstrap)

    if args.run_bootstrap_reselection:
        eligible_rows = (
            interactions.loc[interactions["interaction_eligible"].astype(bool)]
            if "interaction_eligible" in interactions
            else pd.DataFrame()
        )
        preapproved = {
            (str(row.outcome_model), str(row.prompt_mode), str(row.candidate))
            for row in eligible_rows.itertuples(index=False)
        }
        reselection_frequency, reselection_draws = bootstrap_reselection_stability(
            requests,
            preapproved_interactions=preapproved,
            replicates=args.reselection_bootstrap_replicates,
        )
        _write_csv(
            tables / "bootstrap_reselection_frequency.csv",
            reselection_frequency,
        )
        _write_csv(tables / "bootstrap_reselection_draws.csv.gz", reselection_draws)

    if args.run_lomo:
        lomo = leave_one_model_out_structure(
            requests,
            interaction_bootstrap_replicates=args.lomo_bootstrap_replicates,
        )
        _write_csv(tables / "leave_one_model_out_structure.csv", lomo)
    else:
        lomo = pd.DataFrame()

    distribution_detail, distribution_calibration, distribution_summaries = (
        _distribution_outputs(requests, selected)
    )
    _write_csv(tables / "accuracy_distribution_predictions.csv", distribution_detail)
    _write_csv(
        tables / "accuracy_distribution_calibration.csv", distribution_calibration
    )
    _atomic_json(
        tables / "accuracy_distribution_diagnostics.json",
        distribution_summaries,
    )

    _write_state(
        state_path,
        "complete",
        lomo_run=bool(args.run_lomo),
        bootstrap_reselection_run=bool(args.run_bootstrap_reselection),
        output_tables=len(list(tables.glob("*"))),
    )
    output_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "realistic_niah_v3_1_analysis_manifest_v1",
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "request_rows": len(requests),
        "sources": sources,
        "settings": vars(args),
        "fit_backend": fit_backend_metadata(),
        "lomo_run": bool(args.run_lomo),
        "bootstrap_reselection_run": bool(args.run_bootstrap_reselection),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in output_files
            if path.name != "analysis_manifest.json"
        ],
    }
    _atomic_json(output / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "requests": len(requests),
                "selected_laws": len(selected),
                "output": str(output),
                "lomo_run": bool(args.run_lomo),
                "bootstrap_reselection_run": bool(args.run_bootstrap_reselection),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
