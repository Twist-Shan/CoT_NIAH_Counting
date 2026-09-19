from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realistic_niah_v3.analysis import (
    CANDIDATES,
    CONTINUOUS_TARGETS,
    accuracy_distribution_diagnostics,
    analysis_manifest,
    behavior_tables,
    fit_candidate_grid,
    load_request_table,
    native_thinking_style_tables,
    paired_mode_comparisons,
    write_request_table_gzip,
)
from realistic_niah_v3.reporting import (
    build_all_plots,
    write_behavior_report,
    write_empirical_law_report,
)
from realistic_niah_v3.spec import EXPECTED_REQUESTS


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_state(path: Path, stage: str, **details: Any) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "realistic_niah_v3_analysis_state_v2",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **details,
        },
    )


def _analysis_plan() -> str:
    candidate_lines = "\n".join(
        f"- `{candidate.name}`: "
        + (
            "model intercept only"
            if not candidate.features
            else " + ".join(candidate.features)
        )
        for candidate in CANDIDATES
    )
    return f"""# Realistic NIAH V3 analysis plan

## Scope

This analysis covers only behavior comparison and empirical laws. It does not
analyze hidden states, attention, probes, interventions, or causal mediation.

## Primary behavioral outcome

`parseable_exact_accuracy = 1` exactly when an integer is parsed and equals the
true count N. Parse failure, wrong count, format failure, and truncation remain
in the denominator. Parse rate and strict registered accuracy are separate
diagnostics.

## Native-thinking visible counting style

Every native-thinking trace is classified into exactly one frozen observable
category: indexed list, bullet list, mixed structured list, ordinal-word
enumeration, inline tally/arithmetic, prose reasoning, or no visible
reasoning. The analysis reports category frequency and exact accuracy overall
and by N/L, with HTML excerpts and complete examples in JSONL. This is a
description of visible text, not a claim about a latent
mechanism.

## Numeric-error estimands

For successfully parsed responses, signed deviation is `predicted_count - N`.
No numeric deviation is imputed for a parse failure. The registered condition
summaries are:

- signed mean deviation;
- absolute mean deviation;
- signed median deviation;
- 10% trimmed signed mean deviation;
- sample variance of signed deviation with denominator `n - 1`.

## Candidate grid

All non-intercept coefficients are shared across behavior-comparison slots
within one prompt mode. Each slot receives one fixed intercept. `L_k = L/1000`
and all logarithms are natural.

{candidate_lines}

Both interaction candidates are hierarchical and contain only a first-order
term multiplied by another first-order term.

## Validation and selection

Five-fold GroupKFold holds out complete seeds across every N, L, model, and
mode. Accuracy jointly searches each response surface with Binomial-logit,
Binomial-probit, Binomial-complementary-log-log, and Beta-Binomial-logit.
Candidates are evaluated with held-out predictive negative log density,
Bernoulli log loss, Brier score, and deviance explained relative to a per-model
training prevalence. Dunn--Smyth randomized quantile residual Q--Q plots,
Shapiro--Wilk, and Cramer--von Mises diagnose the selected distribution but do
not select it. Continuous targets use held-out condition-level R-squared, MAE,
and RMSE. Candidate selection applies a one-standard-error-style tolerance and
chooses the simplest near-best form. Interaction candidates are not eligible
when the interaction coefficient has p >= 0.05.

All attempted formulas, convergence flags, validation metrics, and
coefficients are retained.
"""


def _readme(run_root: Path, output: Path) -> str:
    return f"""# Realistic NIAH V3 behavior and empirical-law analysis

Source run: `{run_root}`

Generated outputs:

- `behavior_report.html`
- `empirical_law_report.html`
- `tables/request_level.csv.gz`
- `tables/model_mode_summary.csv`
- `tables/condition_summary.csv`
- `tables/outcome_composition.csv`
- `tables/paired_mode_comparisons.csv`
- `tables/native_thinking_style_summary.csv`
- `tables/native_thinking_style_by_condition.csv`
- `tables/native_thinking_style_examples.jsonl`
- `tables/candidate_comparison.csv`
- `tables/accuracy_probability_distribution_comparison.csv`
- `tables/deviation_regression_comparison.csv`
- `tables/selected_laws.csv`
- `tables/coefficients.csv`
- `tables/accuracy_randomized_quantile_residuals.csv`
- `tables/accuracy_distribution_diagnostics.csv`
- `figures/behavior/`
- `figures/empirical_law/`
- `analysis_plan.md`
- `analysis_manifest.json`
- `analysis_state.json`

Re-run from the repository root:

```bash
PYTHONPATH=src python scripts/analyze_realistic_niah_v3.py \\
  --run-root {run_root} \\
  --output-dir {output}
```

The script refuses a source run whose final V3 shard audit has not passed.
Raw request, manifest, QC, prompt, and stimulus files are read-only inputs.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible V3 behavior and empirical-law tables, plots, "
            "and HTML reports."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "analysis" / "v3_behavior_empirical_law"
    )
    tables = output / "tables"
    figures = output / "figures"
    reports = output / "reports"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    state_path = output / "analysis_state.json"
    (output / "analysis_plan.md").write_text(
        _analysis_plan(),
        encoding="utf-8",
    )
    _write_state(state_path, "loading")

    requests, sources = load_request_table(run_root)
    if len(requests) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected {EXPECTED_REQUESTS:,} requests, found {len(requests):,}"
        )
    write_request_table_gzip(
        tables / "request_level.csv.gz",
        requests,
    )
    summary, by_condition, outcomes = behavior_tables(requests)
    paired_comparisons = paired_mode_comparisons(requests)
    (
        native_style_summary,
        native_style_by_condition,
        native_style_examples,
    ) = native_thinking_style_tables(requests)
    summary.to_csv(tables / "model_mode_summary.csv", index=False)
    by_condition.to_csv(tables / "condition_summary.csv", index=False)
    outcomes.to_csv(tables / "outcome_composition.csv", index=False)
    paired_comparisons.to_csv(
        tables / "paired_mode_comparisons.csv",
        index=False,
    )
    native_style_summary.to_csv(
        tables / "native_thinking_style_summary.csv",
        index=False,
    )
    native_style_by_condition.to_csv(
        tables / "native_thinking_style_by_condition.csv",
        index=False,
    )
    native_style_examples.to_json(
        tables / "native_thinking_style_examples.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    _write_state(
        state_path,
        "behavior_tables_complete",
        requests=len(requests),
    )

    comparison, selected, coefficients = fit_candidate_grid(
        requests,
        targets=(
            "parseable_exact_accuracy",
            *CONTINUOUS_TARGETS,
        ),
        n_splits=args.cv_folds,
    )
    comparison.to_csv(tables / "candidate_comparison.csv", index=False)
    comparison.loc[
        comparison["target"] == "parseable_exact_accuracy"
    ].to_csv(
        tables / "accuracy_probability_distribution_comparison.csv",
        index=False,
    )
    comparison.loc[
        comparison["target"] != "parseable_exact_accuracy"
    ].to_csv(
        tables / "deviation_regression_comparison.csv",
        index=False,
    )
    selected.to_csv(tables / "selected_laws.csv", index=False)
    coefficients.to_csv(tables / "coefficients.csv", index=False)
    accuracy_residuals, accuracy_diagnostics = (
        accuracy_distribution_diagnostics(requests, selected)
    )
    accuracy_residuals.to_csv(
        tables / "accuracy_randomized_quantile_residuals.csv",
        index=False,
    )
    accuracy_diagnostics.to_csv(
        tables / "accuracy_distribution_diagnostics.csv",
        index=False,
    )
    _write_state(
        state_path,
        "candidate_grid_complete",
        attempted_fits=len(comparison),
        selected_laws=len(selected),
    )

    behavior_plots, empirical_plots = build_all_plots(
        requests=requests,
        selected=selected,
        output_dir=figures,
        accuracy_quantile_residuals=accuracy_residuals,
        accuracy_distribution_diagnostics=accuracy_diagnostics,
    )
    behavior_report = write_behavior_report(
        output_path=reports / "behavior_report.html",
        summary=summary,
        by_condition=by_condition,
        outcomes=outcomes,
        paired_comparisons=paired_comparisons,
        plot_paths=behavior_plots,
        native_style_summary=native_style_summary,
        native_style_by_condition=native_style_by_condition,
        native_style_examples=native_style_examples,
    )
    empirical_report = write_empirical_law_report(
        output_path=reports / "empirical_law_report.html",
        selected=selected,
        comparisons=comparison,
        coefficients=coefficients,
        plot_paths=empirical_plots,
        accuracy_distribution_diagnostics=accuracy_diagnostics,
    )
    (output / "README.md").write_text(
        _readme(run_root, output),
        encoding="utf-8",
    )
    _write_state(
        state_path,
        "reports_complete",
        behavior_report=str(behavior_report),
        empirical_law_report=str(empirical_report),
    )

    pre_manifest_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    _write_state(
        state_path,
        "complete",
        output_files=len(pre_manifest_files) + 1,
    )
    output_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = analysis_manifest(
        run_root=run_root,
        output_root=output,
        sources=sources,
        requests=requests,
        output_files=output_files,
    )
    _atomic_json(output / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "requests": len(requests),
                "candidate_fits": len(comparison),
                "selected_laws": len(selected),
                "behavior_report": str(behavior_report),
                "empirical_law_report": str(empirical_report),
                "analysis_manifest": str(
                    output / "analysis_manifest.json"
                ),
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
