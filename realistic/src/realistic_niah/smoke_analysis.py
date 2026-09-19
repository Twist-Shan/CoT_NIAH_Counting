from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SMOKE_PROMPT_MODE = "native_thinking"


def load_request_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def summarize_guarded_smoke(
    rows: Iterable[dict[str, Any]],
    *,
    expected_models: Iterable[str] | None = None,
    expected_requests_per_model: int | None = None,
) -> dict[str, Any]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if str(row.get("prompt_mode")) != SMOKE_PROMPT_MODE:
            continue
        model = str(row["model_label"])
        stimulus = str(row["stimulus_id"])
        key = (model, stimulus)
        if key in indexed:
            raise ValueError(f"Duplicate guarded smoke result for {key}")
        indexed[key] = row
        by_model[model].append(row)

    if not indexed:
        raise ValueError("No guarded native-thinking smoke rows were found")

    summaries: dict[str, Any] = {}
    for model in sorted(by_model):
        model_rows = by_model[model]
        evaluations = [row["evaluation"] for row in model_rows]
        output_tokens = [int(row["output_tokens"]) for row in model_rows]
        truncations = sum(
            bool(evaluation["truncated"]) for evaluation in evaluations
        )
        summaries[model] = {
            "requests": len(model_rows),
            "truncations": truncations,
            "zero_truncations": truncations == 0,
            "parse_failures": sum(
                evaluation["parse_status"] == "parse_fail"
                for evaluation in evaluations
            ),
            "format_failures": sum(
                not bool(evaluation["response_format_compliant"])
                for evaluation in evaluations
            ),
            "registered_accuracy": sum(
                bool(evaluation["registered_success"])
                for evaluation in evaluations
            )
            / len(evaluations),
            "exact_count_accuracy": sum(
                bool(evaluation["exact_count"])
                for evaluation in evaluations
            )
            / len(evaluations),
            "overthinking_flags": sum(
                bool(evaluation["overthinking_flag"])
                for evaluation in evaluations
            ),
            "mean_output_tokens": sum(output_tokens) / len(output_tokens),
            "max_output_tokens": max(output_tokens),
        }

    observed_models = set(by_model)
    expected = set(expected_models or observed_models)
    missing_models = sorted(expected - observed_models)
    unexpected_models = sorted(observed_models - expected)
    request_count_mismatches = {
        model: summary["requests"]
        for model, summary in summaries.items()
        if expected_requests_per_model is not None
        and summary["requests"] != expected_requests_per_model
    }
    total_truncations = sum(
        summary["truncations"] for summary in summaries.values()
    )
    gate_passed = (
        total_truncations == 0
        and not missing_models
        and not unexpected_models
        and not request_count_mismatches
    )
    return {
        "schema_version": "realistic_niah_guarded_smoke_summary_v2",
        "prompt_mode": SMOKE_PROMPT_MODE,
        "gate": {
            "requirement": "zero_truncations",
            "passed": gate_passed,
            "total_truncations": total_truncations,
            "missing_models": missing_models,
            "unexpected_models": unexpected_models,
            "request_count_mismatches": request_count_mismatches,
        },
        "models": summaries,
    }
