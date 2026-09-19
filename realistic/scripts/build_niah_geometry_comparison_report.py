#!/usr/bin/env python3
"""Build a self-contained three-column NIAH geometry comparison report.

The three displayed populations are deliberately distinct:

1. non-thinking prompt needle endpoints on the full registered seed panel;
2. native-thinking response item endpoints after one-to-one trace filtering;
3. native-thinking response item endpoints on the full registered seed panel,
   retaining every parser-observed ordinal from partial traces.

Position (1--10) is the geometry class.  Final-answer correctness is carried as
an independent trajectory-level display attribute and never changes the
position label or the primary aligned cohort.  Native panels also expose a
registered marker-kind-aware anchor sensitivity; uniform item_end remains the
primary, cross-trace-comparable estimand.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    ModeDataset,
    TRACE_AWARE_SITE_BY_MARKER_KIND,
    load_native_thinking_capture,
    load_non_thinking_capture,
)
from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    PCA_WHITEN,
    SCHEMA_VERSION as DUAL_ENDPOINT_SCHEMA_VERSION,
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    PARSER_UPSTREAM_COMMIT,
    PARSER_UPSTREAM_REPOSITORY,
)
from realistic_niah_v5.trace_stratified_geometry import (  # noqa: E402
    LAYER_COMPARISON_POLICY as TRACE_STRATIFIED_LAYER_COMPARISON_POLICY,
    SCHEMA_VERSION as TRACE_STRATIFIED_SCHEMA_VERSION,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
PCA_DIMS = (32,)
TRACE_STRATIFIED_PCA_DIM = 16
DUAL_ENDPOINT_DIRECTORY = "pca16_whiten"
EXPECTED_FULL_PANEL = {
    "discovery": list(range(1234, 1254)),
    "confirmation": list(range(1254, 1264)),
}
EXPECTED_COUNTS = tuple(range(1, 11))
EXPECTED_TRAJECTORIES = 300


def expected_trajectory_keys() -> set[tuple[str, int, int]]:
    return {
        (split, int(seed), int(gold_count))
        for split, seeds in EXPECTED_FULL_PANEL.items()
        for seed in seeds
        for gold_count in EXPECTED_COUNTS
    }


def require_full_trajectory_panel(
    keys: Iterable[tuple[str, int, int]], context: str
) -> None:
    observed = set(keys)
    expected = expected_trajectory_keys()
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    require(
        observed == expected,
        f"{context} is not the registered 10-count x 30-seed panel: "
        f"missing={missing[:8]}, extra={extra[:8]}",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def html_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _best(rows: list[dict[str, str]], field: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row[field]))


def _support_range(audit: Mapping[str, Any], mode: str) -> tuple[int, int]:
    values = [
        int(value)
        for value in audit["position_support"][mode]["confirmation"].values()
    ]
    return min(values), max(values)


def load_trace_stratified_results(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Load the marker-kind-stratified discovery-selected site sweep."""

    results: dict[str, dict[str, Any]] = {}
    inputs: list[Path] = []
    for model in MODELS:
        directory = root / model / f"pca{TRACE_STRATIFIED_PCA_DIM}"
        eligibility_path = directory / "trace_stratum_eligibility.csv"
        selection_path = directory / "trace_stratum_discovery_selected_sites.csv"
        metrics_path = directory / "trace_stratum_site_layer_metrics.csv"
        audit_path = directory / "trace_stratum_site_sweep_audit.json"
        audit = read_json(audit_path)
        eligibility = read_csv(eligibility_path)
        selection = read_csv(selection_path)
        require(
            audit.get("schema_version") == TRACE_STRATIFIED_SCHEMA_VERSION,
            f"trace-stratified schema mismatch for {model}",
        )
        require(audit.get("model_label") == model, f"trace-stratified model mismatch for {model}")
        require(
            int(audit.get("pca_dim_requested", -1)) == TRACE_STRATIFIED_PCA_DIM,
            f"trace-stratified PCA mismatch for {model}",
        )
        require(
            audit.get("stratification_variable") == "parser marker_kind",
            f"trace-stratified grouping mismatch for {model}",
        )
        require(
            audit.get("layer_comparison_policy")
            == TRACE_STRATIFIED_LAYER_COMPARISON_POLICY,
            f"trace-stratified layer policy mismatch for {model}",
        )
        require(
            all(row.get("model_label") == model for row in eligibility + selection),
            f"trace-stratified row model mismatch for {model}",
        )
        require(
            {row["selector"] for row in selection}
            <= {
                "fixed_item_end",
                "post_marker_site_search",
                "pre_label_site_search",
                "low_leakage_site_search",
                "all_site_search",
            },
            f"unregistered trace-stratified selector for {model}",
        )
        results[model] = {
            "audit": audit,
            "eligibility": eligibility,
            "selection": selection,
        }
        inputs.extend([eligibility_path, selection_path, metrics_path, audit_path])
    return results, inputs


def load_dual_endpoint_results(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Load independently selected running-index and final-count results."""

    results: dict[str, dict[str, Any]] = {}
    inputs: list[Path] = []
    for model in MODELS:
        directory = root / model / DUAL_ENDPOINT_DIRECTORY
        paths = {
            "running_candidates": directory / "running_index_candidate_metrics.csv",
            "running_selected": directory / "running_index_selected.csv",
            "eligibility": directory / "running_index_group_eligibility.csv",
            "final_candidates": directory / "final_count_candidate_metrics.csv",
            "final_selected": directory / "final_count_selected.csv",
            "audit": directory / "dual_endpoint_geometry_audit.json",
            "runtime": directory / "runtime_log.json",
        }
        audit = read_json(paths["audit"])
        require(
            audit.get("schema_version") == DUAL_ENDPOINT_SCHEMA_VERSION,
            f"dual-endpoint schema mismatch for {model}",
        )
        require(audit.get("model_label") == model, f"dual-endpoint model mismatch for {model}")
        require(bool(audit.get("pca_whiten")) == PCA_WHITEN, "dual PCA whitening mismatch")
        payload = {
            "audit": audit,
            "runtime": read_json(paths["runtime"]),
            "running_candidates": read_csv(paths["running_candidates"]),
            "running_selected": read_csv(paths["running_selected"]),
            "eligibility": read_csv(paths["eligibility"]),
            "final_candidates": read_csv(paths["final_candidates"]),
            "final_selected": read_csv(paths["final_selected"]),
        }
        for key in ("running_selected", "final_selected"):
            require(
                all(row.get("model_label") == model for row in payload[key]),
                f"dual-endpoint row model mismatch for {model}/{key}",
            )
        results[model] = payload
        inputs.extend(paths.values())
    return results, inputs


def _one_row(rows: Iterable[Mapping[str, Any]], **criteria: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    require(len(matches) == 1, f"expected one row for {criteria}; found {len(matches)}")
    return matches[0]


def load_metric_comparison(
    aligned_root: Path,
    one_to_one_root: Path,
    trace_aware_aligned_root: Path,
    trace_aware_one_to_one_root: Path,
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, int],
    list[Path],
]:
    """Read audited metrics for the three report columns."""

    comparison: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    aligned_peak_layer: dict[str, int] = {}
    inputs: list[Path] = []
    for model in MODELS:
        for pca_dim in PCA_DIMS:
            aligned_dir = aligned_root / model / f"pca{pca_dim}"
            complete_dir = one_to_one_root / model / f"pca{pca_dim}"
            aligned_audit_path = aligned_dir / "cross_mode_geometry_audit.json"
            complete_audit_path = complete_dir / "cross_mode_geometry_audit.json"
            aligned_global_path = aligned_dir / "global_covariance_geometry.csv"
            complete_global_path = complete_dir / "global_covariance_geometry.csv"
            trace_aligned_dir = trace_aware_aligned_root / model / f"pca{pca_dim}"
            trace_complete_dir = (
                trace_aware_one_to_one_root / model / f"pca{pca_dim}"
            )
            trace_aligned_audit_path = (
                trace_aligned_dir / "cross_mode_geometry_audit.json"
            )
            trace_complete_audit_path = (
                trace_complete_dir / "cross_mode_geometry_audit.json"
            )
            trace_aligned_global_path = (
                trace_aligned_dir / "global_covariance_geometry.csv"
            )
            trace_complete_global_path = (
                trace_complete_dir / "global_covariance_geometry.csv"
            )
            aligned_audit = read_json(aligned_audit_path)
            complete_audit = read_json(complete_audit_path)
            trace_aligned_audit = read_json(trace_aligned_audit_path)
            trace_complete_audit = read_json(trace_complete_audit_path)
            aligned_rows = read_csv(aligned_global_path)
            complete_rows = read_csv(complete_global_path)
            trace_aligned_rows = read_csv(trace_aligned_global_path)
            trace_complete_rows = read_csv(trace_complete_global_path)

            require(aligned_audit["model_label"] == model, "aligned model mismatch")
            require(complete_audit["model_label"] == model, "one-to-one model mismatch")
            require(aligned_audit["native_cohort"] == "parser_hit", "aligned cohort mismatch")
            require(complete_audit["native_cohort"] == "one_to_one", "one-to-one cohort mismatch")
            require(
                aligned_audit.get("native_site_policy", "uniform") == "uniform",
                "aligned primary site policy mismatch",
            )
            require(
                complete_audit.get("native_site_policy", "uniform") == "uniform",
                "one-to-one primary site policy mismatch",
            )
            for trace_audit, cohort in (
                (trace_aligned_audit, "parser_hit"),
                (trace_complete_audit, "one_to_one"),
            ):
                require(trace_audit["model_label"] == model, "trace-aware model mismatch")
                require(trace_audit["native_cohort"] == cohort, "trace-aware cohort mismatch")
                require(
                    trace_audit["native_site_policy"]
                    == "trace_aware_count_boundary",
                    "trace-aware site policy mismatch",
                )
                require(
                    trace_audit["native_site_policy_mapping"]
                    == TRACE_AWARE_SITE_BY_MARKER_KIND,
                    "trace-aware mapping mismatch",
                )
            require(
                aligned_audit["analysis_design"]
                == "fixed_registered_seed_panel_observed_positions",
                "aligned design mismatch",
            )
            require(
                complete_audit["analysis_design"]
                == "complete_trajectory_paired_sensitivity",
                "one-to-one design mismatch",
            )
            require(
                trace_aligned_audit["analysis_design"]
                == "fixed_registered_seed_panel_observed_positions",
                "trace-aware aligned design mismatch",
            )
            require(
                trace_complete_audit["analysis_design"]
                == "complete_trajectory_paired_sensitivity",
                "trace-aware one-to-one design mismatch",
            )
            require(
                aligned_audit["registered_seed_panel"] == EXPECTED_FULL_PANEL,
                "aligned seed panel is not the registered 30-seed panel",
            )
            expected_split_trajectories = {
                "discovery": 200,
                "confirmation": 100,
            }
            require(
                aligned_audit.get("registered_trajectory_panel")
                == {
                    "non_thinking": expected_split_trajectories,
                    "native_thinking": expected_split_trajectories,
                },
                "aligned analysis is not the registered 300-trajectory panel",
            )
            require(
                trace_aligned_audit["registered_seed_panel"]
                == aligned_audit["registered_seed_panel"],
                "trace-aware aligned seed panel changed",
            )
            require(
                trace_aligned_audit.get("registered_trajectory_panel")
                == aligned_audit.get("registered_trajectory_panel"),
                "trace-aware aligned trajectory panel changed",
            )
            require(
                trace_complete_audit["registered_seed_panel"]
                == complete_audit["registered_seed_panel"],
                "trace-aware one-to-one seed panel changed",
            )
            require(
                trace_complete_audit.get("registered_trajectory_panel")
                == complete_audit.get("registered_trajectory_panel"),
                "trace-aware one-to-one trajectory panel changed",
            )
            for audit in (
                aligned_audit,
                complete_audit,
                trace_aligned_audit,
                trace_complete_audit,
            ):
                require(audit["evaluation_split"] == "confirmation only", "split leakage")
                require(audit["preprocessing_fit_split"] == "discovery only", "PCA leakage")
                require(audit["probe_fit_split"] == "discovery only", "probe leakage")
                require(audit["cluster_labels"] == list(range(1, 11)), "label mismatch")

            non_rows = [row for row in aligned_rows if row["mode"] == "non_thinking"]
            complete_non_rows = [
                row for row in complete_rows if row["mode"] == "non_thinking"
            ]
            trace_aligned_non_rows = [
                row
                for row in trace_aligned_rows
                if row["mode"] == "non_thinking"
            ]
            trace_complete_non_rows = [
                row
                for row in trace_complete_rows
                if row["mode"] == "non_thinking"
            ]
            aligned_native_rows = [
                row for row in aligned_rows if row["mode"] == "native_thinking"
            ]
            complete_native_rows = [
                row for row in complete_rows if row["mode"] == "native_thinking"
            ]
            trace_aligned_native_rows = [
                row
                for row in trace_aligned_rows
                if row["mode"] == "native_thinking"
            ]
            trace_complete_native_rows = [
                row
                for row in trace_complete_rows
                if row["mode"] == "native_thinking"
            ]
            require(
                non_rows
                and aligned_native_rows
                and complete_native_rows
                and trace_aligned_native_rows
                and trace_complete_native_rows,
                "missing mode rows",
            )
            require(
                non_rows == trace_aligned_non_rows,
                "trace-aware policy changed aligned non-thinking metrics",
            )
            require(
                complete_non_rows == trace_complete_non_rows,
                "trace-aware policy changed one-to-one non-thinking metrics",
            )

            def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
                logistic = _best(rows, "logistic_balanced_accuracy")
                ncc = _best(rows, "ncc_balanced_accuracy")
                snr = _best(rows, "class_balanced_snr")
                return {
                    "n_confirmation": int(logistic["n_confirmation"]),
                    "logistic": float(logistic["logistic_balanced_accuracy"]),
                    "logistic_layer": int(logistic["layer"]),
                    "ncc": float(ncc["ncc_balanced_accuracy"]),
                    "ncc_layer": int(ncc["layer"]),
                    "snr": float(snr["class_balanced_snr"]),
                    "snr_db": float(snr["class_balanced_snr_db"]),
                    "snr_layer": int(snr["layer"]),
                }

            def layerwise(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
                return {
                    str(int(row["layer"])): {
                        "logistic": float(row["logistic_balanced_accuracy"]),
                        "ncc": float(row["ncc_balanced_accuracy"]),
                        "snr": float(row["class_balanced_snr"]),
                        "snr_db": float(row["class_balanced_snr_db"]),
                        "n_confirmation": int(row["n_confirmation"]),
                    }
                    for row in rows
                }

            comparison[model][pca_dim] = {
                "non_thinking": summarize(non_rows),
                "native_one_to_one": summarize(complete_native_rows),
                "native_aligned": summarize(aligned_native_rows),
                "native_one_to_one_trace_aware": summarize(
                    trace_complete_native_rows
                ),
                "native_aligned_trace_aware": summarize(
                    trace_aligned_native_rows
                ),
                "non_support": _support_range(aligned_audit, "non_thinking"),
                "one_support": _support_range(complete_audit, "native_thinking"),
                "aligned_support": _support_range(aligned_audit, "native_thinking"),
                "one_seed_panel": complete_audit["registered_seed_panel"],
                "layerwise": {
                    "non_thinking": layerwise(non_rows),
                    "native_one_to_one": layerwise(complete_native_rows),
                    "native_aligned": layerwise(aligned_native_rows),
                    "native_one_to_one_trace_aware": layerwise(
                        trace_complete_native_rows
                    ),
                    "native_aligned_trace_aware": layerwise(
                        trace_aligned_native_rows
                    ),
                },
            }
            if pca_dim == 32:
                aligned_peak_layer[model] = comparison[model][pca_dim][
                    "native_aligned"
                ]["logistic_layer"]
            inputs.extend(
                [
                    aligned_audit_path,
                    aligned_global_path,
                    complete_audit_path,
                    complete_global_path,
                    trace_aligned_audit_path,
                    trace_aligned_global_path,
                    trace_complete_audit_path,
                    trace_complete_global_path,
                ]
            )
    return comparison, aligned_peak_layer, inputs


def non_thinking_outcomes(
    export_root: Path, model: str
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], Path]:
    path = (
        export_root
        / model
        / "numeric"
        / "behavior"
        / "capture"
        / "generation_label_index.jsonl"
    )
    mapping: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row["design_variant"] != "v4.4":
            continue
        key = (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
        value = {
            "exact_count": bool(row["is_correct"]),
            "parsed_count": (
                int(row["parsed_count"])
                if row.get("parsed_count") is not None
                else None
            ),
            "count_error": (
                int(row["count_error"])
                if row.get("count_error") is not None
                else None
            ),
        }
        if key in mapping:
            require(mapping[key] == value, f"inconsistent non-thinking outcome {model}/{key}")
        else:
            mapping[key] = value
    require_full_trajectory_panel(mapping, f"non-thinking outcomes for {model}")
    return mapping, path


def native_outcomes(
    capture_index: Path,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(capture_index)
    mapping: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        gold_count = int(row["gold_count"])
        key = (str(row["split"]), int(row["seed"]), gold_count)
        require(key not in mapping, f"duplicate native outcome {key} in {capture_index}")
        mapping[key] = {
            "exact_count": bool(row.get("exact_count")),
            "parsed_count": row.get("parsed_count"),
            "count_error": (
                int(row["parsed_count"]) - gold_count
                if row.get("parsed_count") is not None
                else None
            ),
        }
    require_full_trajectory_panel(mapping, f"native outcomes in {capture_index}")
    return mapping, rows


def partial_trace_rows(
    capture_index: Path,
    index_rows: list[dict[str, Any]],
    trace_by_request: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in index_rows:
        if bool(row.get("trace_one_to_one")):
            continue
        manifest_path = capture_index.parent / str(row["manifest_path"])
        manifest = read_json(manifest_path)
        item_sites = [
            site for site in manifest["site_rows"] if site.get("site_kind") == "item_end"
        ]
        parser = manifest["parser"]
        value = {
                "model": row["model_label"],
                "request_id": row["request_id"],
                "split": row["split"],
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "observed": len(item_sites),
                "occurrences": [int(site["occurrence"]) for site in item_sites],
                "cities": [str(site.get("city")) for site in item_sites],
                "missing": [str(city) for city in parser.get("missing_gold_cities", [])],
                "parsed_count": row.get("parsed_count"),
                "exact_count": bool(row.get("exact_count")),
                "trace_category": row.get("trace_category"),
                "endpoint_sites": [
                    {
                        "occurrence": int(site["occurrence"]),
                        "city": str(site.get("city")),
                        "char_start": int(site["char_start"]),
                        "char_end": int(site["char_end"]),
                        "endpoint_token": int(site["endpoint_token"]),
                        "sequence_query_position": (
                            int(manifest["prompt_token_count"])
                            + int(site["endpoint_token"])
                        ),
                        "alignment_strategy": str(site["alignment_strategy"]),
                    }
                    for site in item_sites
                ],
            }
        if trace_by_request is not None:
            trace = trace_by_request.get(str(row["request_id"]))
            require(trace is not None, f"trace archive lacks {row['request_id']}")
            require(
                int(trace["prompt_token_count"]) == int(manifest["prompt_token_count"]),
                f"prompt token mismatch for {row['request_id']}",
            )
            require(
                len(trace["output_token_ids"]) == int(manifest["output_token_count"]),
                f"output token mismatch for {row['request_id']}",
            )
            raw = str(trace["raw_output_text"])
            start = int(parser.get("list_start_char") or 0)
            end = int(parser.get("cut_char") or start)
            value["raw_excerpt"] = raw[start:end].strip()
            for endpoint in value["endpoint_sites"]:
                endpoint["item_text"] = raw[
                    endpoint["char_start"] : endpoint["char_end"]
                ].strip()
        result.append(value)
    return sorted(
        result,
        key=lambda row: (row["split"], row["seed"], row["gold_count"]),
    )


def display_layers(dataset: ModeDataset, aligned_peak: int) -> list[int]:
    available = sorted(dataset.states_by_layer)
    require(int(aligned_peak) in available, "aligned peak layer is unavailable")
    return available


def load_trace_archive(
    trace_root: Path | None, model: str
) -> tuple[dict[str, dict[str, Any]] | None, Path | None]:
    if trace_root is None:
        return None, None
    path = trace_root / model / "generations.jsonl"
    rows = read_jsonl(path)
    mapping = {str(row["request_id"]): row for row in rows}
    require(len(mapping) == len(rows), f"duplicate request IDs for {model}")
    require_full_trajectory_panel(
        (
            (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
            for row in rows
        ),
        f"native trace archive for {model}",
    )
    return mapping, path


def native_alignment_summary(
    capture_index: Path,
    index_rows: list[dict[str, Any]],
    trace_by_request: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    ordered_rows = sorted(
        index_rows,
        key=lambda row: (
            str(row["split"]) != "confirmation",
            int(row["seed"]),
            int(row["gold_count"]),
        ),
    )
    for row in ordered_rows:
        manifest = read_json(capture_index.parent / str(row["manifest_path"]))
        trace = (
            None
            if trace_by_request is None
            else trace_by_request.get(str(row["request_id"]))
        )
        if trace_by_request is not None:
            require(trace is not None, f"trace archive lacks {row['request_id']}")
            require(
                int(trace["prompt_token_count"]) == int(manifest["prompt_token_count"]),
                f"prompt token mismatch for {row['request_id']}",
            )
            require(
                len(trace["output_token_ids"]) == int(manifest["output_token_count"]),
                f"output token mismatch for {row['request_id']}",
            )
        for site in manifest["site_rows"]:
            if site.get("site_kind") != "item_end":
                continue
            strategy = str(site["alignment_strategy"])
            counts[strategy] += 1
            if strategy in examples:
                continue
            example = {
                "model": str(row["model_label"]),
                "split": str(row["split"]),
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "occurrence": int(site["occurrence"]),
                "city": str(site.get("city")),
                "char_start": int(site["char_start"]),
                "char_end": int(site["char_end"]),
                "endpoint_token": int(site["endpoint_token"]),
                "prompt_token_count": int(manifest["prompt_token_count"]),
                "sequence_query_position": (
                    int(manifest["prompt_token_count"]) + int(site["endpoint_token"])
                ),
                "alignment_strategy": strategy,
            }
            if trace is not None:
                raw = str(trace["raw_output_text"])
                example["item_text"] = raw[
                    int(site["char_start"]) : int(site["char_end"])
                ].strip()
            examples[strategy] = example
    return {
        "item_end_sites": int(sum(counts.values())),
        "strategy_counts": dict(sorted(counts.items())),
        "examples": list(examples.values()),
        "trace_archive_verified": trace_by_request is not None,
    }


def native_trace_policy_summary(
    capture_index: Path,
    index_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audit the registered trace-aware anchor without changing eligibility."""

    grouped: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        marker_kind = str(row.get("marker_kind"))
        require(
            marker_kind in TRACE_AWARE_SITE_BY_MARKER_KIND,
            f"unregistered marker_kind {marker_kind!r}",
        )
        selected_site_kind = TRACE_AWARE_SITE_BY_MARKER_KIND[marker_kind]
        bucket = grouped.setdefault(
            marker_kind,
            {
                "marker_kind": marker_kind,
                "selected_site_kind": selected_site_kind,
                "trace_count": 0,
                "one_to_one_count": 0,
                "split_trace_counts": Counter(),
                "one_to_one_split_trace_counts": Counter(),
                "trace_categories": Counter(),
                "selected_state_count": 0,
                "alignment_strategies": Counter(),
            },
        )
        require(
            bucket["selected_site_kind"] == selected_site_kind,
            f"non-deterministic policy for {marker_kind}",
        )
        bucket["trace_count"] += 1
        bucket["one_to_one_count"] += int(bool(row.get("trace_one_to_one")))
        bucket["split_trace_counts"][str(row["split"])] += 1
        if bool(row.get("trace_one_to_one")):
            bucket["one_to_one_split_trace_counts"][str(row["split"])] += 1
        bucket["trace_categories"][str(row.get("trace_category"))] += 1
        manifest = read_json(capture_index.parent / str(row["manifest_path"]))
        manifest_marker = str(manifest.get("parser", {}).get("marker_kind"))
        require(
            manifest_marker == marker_kind,
            f"marker_kind mismatch for {row.get('request_id')}",
        )
        selected = [
            site
            for site in manifest["site_rows"]
            if site.get("site_kind") == selected_site_kind
            and site.get("occurrence") is not None
        ]
        item_end = [
            site
            for site in manifest["site_rows"]
            if site.get("site_kind") == "item_end"
            and site.get("occurrence") is not None
        ]
        require(
            [int(site["occurrence"]) for site in selected]
            == [int(site["occurrence"]) for site in item_end],
            f"trace-aware site changes ordinal support for {row.get('request_id')}",
        )
        bucket["selected_state_count"] += len(selected)
        for site in selected:
            bucket["alignment_strategies"][str(site["alignment_strategy"])] += 1
    result = []
    for marker_kind in TRACE_AWARE_SITE_BY_MARKER_KIND:
        if marker_kind not in grouped:
            continue
        bucket = grouped[marker_kind]
        result.append(
            {
                "marker_kind": marker_kind,
                "selected_site_kind": bucket["selected_site_kind"],
                "trace_count": int(bucket["trace_count"]),
                "one_to_one_count": int(bucket["one_to_one_count"]),
                "split_trace_counts": {
                    split: int(bucket["split_trace_counts"].get(split, 0))
                    for split in ("discovery", "confirmation")
                },
                "one_to_one_split_trace_counts": {
                    split: int(
                        bucket["one_to_one_split_trace_counts"].get(split, 0)
                    )
                    for split in ("discovery", "confirmation")
                },
                "trace_categories": dict(sorted(bucket["trace_categories"].items())),
                "selected_state_count": int(bucket["selected_state_count"]),
                "alignment_strategies": dict(
                    sorted(bucket["alignment_strategies"].items())
                ),
            }
        )
    require(
        sum(row["trace_count"] for row in result) == EXPECTED_TRAJECTORIES,
        "trace policy trajectory loss",
    )
    return result


def fit_display_coordinates(
    dataset: ModeDataset,
    layers: Iterable[int],
    outcomes: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = dataset.metadata.reset_index(drop=True)
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    require(discovery.sum() >= 3, f"{dataset.mode}: too few discovery rows")
    result: dict[str, Any] = {}
    for layer in layers:
        states = np.asarray(dataset.states_by_layer[int(layer)], dtype=np.float32)
        scaler = StandardScaler().fit(states[discovery])
        scaled_discovery = scaler.transform(states[discovery])
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(
            scaled_discovery
        )
        coordinates = pca.transform(scaler.transform(states))
        points = []
        for index, row in metadata.iterrows():
            split = str(row["split"])
            seed = int(row["seed"])
            gold_count = int(row["gold_count"])
            outcome = outcomes[(split, seed, gold_count)]
            points.append(
                [
                    split,
                    seed,
                    int(row["occurrence"]),
                    1 if outcome["exact_count"] else 0,
                    outcome.get("parsed_count"),
                    round(float(coordinates[index, 0]), 5),
                    round(float(coordinates[index, 1]), 5),
                    round(float(coordinates[index, 2]), 5),
                    gold_count,
                ]
            )
        result[str(layer)] = {
            "evr": [round(float(value), 6) for value in pca.explained_variance_ratio_],
            "points": points,
        }
    return result


def fit_dual_display_coordinates(dataset: ModeDataset) -> dict[str, Any]:
    """Fit a separate discovery-only PCA3 display for every available layer."""

    metadata = dataset.metadata.reset_index(drop=True)
    discovery = metadata["split"].astype(str).eq("discovery").to_numpy()
    require(discovery.sum() >= 3, f"{dataset.mode}: too few dual-endpoint discovery rows")
    result: dict[str, Any] = {}
    for layer, values in sorted(dataset.states_by_layer.items()):
        states = np.asarray(values, dtype=np.float32)
        scaler = StandardScaler().fit(states[discovery])
        scaled_discovery = scaler.transform(states[discovery])
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(
            scaled_discovery
        )
        coordinates = pca.transform(scaler.transform(states))
        points = [
            [
                str(row.split),
                int(row.seed),
                int(row.occurrence),
                round(float(coordinates[index, 0]), 5),
                round(float(coordinates[index, 1]), 5),
                round(float(coordinates[index, 2]), 5),
                int(row.gold_count),
            ]
            for index, row in enumerate(metadata.itertuples(index=False))
        ]
        result[str(layer)] = {
            "evr": [round(float(value), 6) for value in pca.explained_variance_ratio_],
            "points": points,
        }
    return result


def _dual_metric_curve(
    candidates: Iterable[Mapping[str, Any]], selected: Mapping[str, Any]
) -> dict[str, Any]:
    criteria = {
        key: str(selected[key])
        for key in ("mode", "analysis_group", "selector", "token_site")
    }
    rows = [
        row
        for row in candidates
        if all(str(row.get(key)) == value for key, value in criteria.items())
    ]
    require(rows, f"no dual candidate curve for {criteria}")
    result = {}
    for row in rows:
        result[str(int(row["layer"]))] = {
            "discovery_logistic": float(
                row["discovery_oof_logistic_balanced_accuracy"]
            ),
            "discovery_ncc": float(row["discovery_oof_ncc_balanced_accuracy"]),
            "discovery_score": float(row["discovery_selection_score"]),
            "confirmation_logistic": float(
                row["confirmation_logistic_balanced_accuracy"]
            ),
            "confirmation_ncc": float(row["confirmation_ncc_balanced_accuracy"]),
            "confirmation_snr_db": float(
                row["confirmation_class_balanced_snr_db"]
            ),
        }
    return result


def build_dual_visual_data(
    export_root: Path,
    native_running_root: Path,
    native_final_root: Path,
    dual_results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    visual: dict[str, Any] = {}
    inputs: list[Path] = []
    for model in MODELS:
        payload = dual_results[model]
        running_selected = payload["running_selected"]
        final_selected = payload["final_selected"]
        running_non_row = _one_row(
            running_selected, mode="non_thinking", analysis_group="all_traces"
        )
        running_native_row = _one_row(
            running_selected,
            mode="native_thinking",
            analysis_group="all_traces",
        )
        final_non_row = _one_row(final_selected, mode="non_thinking")
        final_native_row = _one_row(final_selected, mode="native_thinking")

        non_running_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "capture"
            / "capture_index.jsonl"
        )
        native_running_index = native_running_root / model / "capture_index.jsonl"
        non_final_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "answer_query_all_layers_v1"
            / "capture_index.jsonl"
        )
        native_final_candidates = (
            native_final_root / model / "capture_index.jsonl",
            native_final_root
            / model
            / "representation"
            / "capture_primary"
            / "capture_index.jsonl",
        )
        native_final_index = next(
            (path for path in native_final_candidates if path.is_file()),
            native_final_candidates[0],
        )
        datasets = {
            "running_non": load_non_thinking_capture(
                non_running_index,
                design_variant="v4.4",
                pooling=str(running_non_row["token_site"]),
            ),
            "running_native": load_native_thinking_capture(
                native_running_index,
                site_kind=str(running_native_row["token_site"]),
                cohort="parser_hit",
            ),
            "final_non": load_non_thinking_final_count(non_final_index),
            "final_native": load_native_thinking_final_count(native_final_index),
        }
        selected_rows = {
            "running_non": running_non_row,
            "running_native": running_native_row,
            "final_non": final_non_row,
            "final_native": final_native_row,
        }
        candidate_rows = {
            "running_non": payload["running_candidates"],
            "running_native": payload["running_candidates"],
            "final_non": payload["final_candidates"],
            "final_native": payload["final_candidates"],
        }
        panels: dict[str, Any] = {}
        for key, dataset in datasets.items():
            selected = selected_rows[key]
            panels[key] = {
                "endpoint": str(selected["endpoint"]),
                "mode": str(selected["mode"]),
                "token_site": str(selected["token_site"]),
                "default_layer": int(selected["layer"]),
                "layers": sorted(dataset.states_by_layer),
                "coordinates": fit_dual_display_coordinates(dataset),
                "metrics": _dual_metric_curve(candidate_rows[key], selected),
            }
        visual[model] = {"panels": panels}
        inputs.extend(
            [
                non_running_index,
                native_running_index,
                non_final_index,
                native_final_index,
            ]
        )
        del datasets
        gc.collect()
    return visual, inputs


def build_visual_data(
    export_root: Path,
    outcome_root: Path,
    native_capture_root: Path,
    aligned_peak_layer: Mapping[str, int],
    comparison: Mapping[str, Mapping[int, Mapping[str, Any]]],
    trace_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    visual: dict[str, Any] = {}
    partials: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for model in MODELS:
        non_index = (
            export_root
            / model
            / "numeric"
            / "representation"
            / "capture"
            / "capture_index.jsonl"
        )
        native_index = native_capture_root / model / "capture_index.jsonl"
        non_outcome, outcome_path = non_thinking_outcomes(outcome_root, model)
        native_outcome, native_rows = native_outcomes(native_index)
        trace_by_request, trace_path = load_trace_archive(trace_root, model)
        partials.extend(
            partial_trace_rows(native_index, native_rows, trace_by_request)
        )

        non = load_non_thinking_capture(non_index, design_variant="v4.4", pooling="span_end")
        aligned = load_native_thinking_capture(
            native_index, site_kind="item_end", cohort="parser_hit"
        )
        one = load_native_thinking_capture(
            native_index, site_kind="item_end", cohort="one_to_one"
        )
        aligned_trace_aware = load_native_thinking_capture(
            native_index,
            site_policy="trace_aware_count_boundary",
            cohort="parser_hit",
        )
        one_trace_aware = load_native_thinking_capture(
            native_index,
            site_policy="trace_aware_count_boundary",
            cohort="one_to_one",
        )
        common_layers = sorted(
            set(non.states_by_layer)
            & set(aligned.states_by_layer)
            & set(one.states_by_layer)
            & set(aligned_trace_aware.states_by_layer)
            & set(one_trace_aware.states_by_layer)
        )
        probe = ModeDataset(
            mode="common",
            model_label=model,
            metadata=non.metadata,
            states_by_layer={layer: non.states_by_layer[layer] for layer in common_layers},
        )
        layers = display_layers(probe, aligned_peak_layer[model])
        visual[model] = {
            "layers": layers,
            "default_layer": int(aligned_peak_layer[model]),
            "metrics": comparison[model][32]["layerwise"],
            "alignment": native_alignment_summary(
                native_index, native_rows, trace_by_request
            ),
            "trace_policy": native_trace_policy_summary(native_index, native_rows),
            "panels": {
                "non_thinking": fit_display_coordinates(non, layers, non_outcome),
                "native_one_to_one": fit_display_coordinates(one, layers, native_outcome),
                "native_aligned": fit_display_coordinates(aligned, layers, native_outcome),
                "native_one_to_one_trace_aware": fit_display_coordinates(
                    one_trace_aware, layers, native_outcome
                ),
                "native_aligned_trace_aware": fit_display_coordinates(
                    aligned_trace_aware, layers, native_outcome
                ),
            },
        }
        inputs.extend([non_index, native_index, outcome_path])
        if trace_path is not None:
            inputs.append(trace_path)
        for provenance_path in (
            non_index.parent / "capture_manifest.json",
            native_index.parent / "export_audit.json",
        ):
            if provenance_path.is_file():
                inputs.append(provenance_path)
        del non, aligned, one, aligned_trace_aware, one_trace_aware, probe
        gc.collect()
    return visual, partials, inputs


def metric_table(comparison: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> str:
    rows = []
    for model in MODELS:
        for pca_dim in PCA_DIMS:
            item = comparison[model][pca_dim]
            non = item["non_thinking"]
            one = item["native_one_to_one"]
            aligned = item["native_aligned"]
            one_trace = item["native_one_to_one_trace_aware"]
            aligned_trace = item["native_aligned_trace_aware"]
            one_panel = item["one_seed_panel"]

            def cell(value: Mapping[str, Any], support: tuple[int, int], seeds: str) -> str:
                support_text = (
                    str(support[0])
                    if support[0] == support[1]
                    else f"{support[0]}–{support[1]}"
                )
                return (
                    f"<strong>logistic {pct(value['logistic'])} @ L{value['logistic_layer']}</strong>"
                    f"<br>NCC {pct(value['ncc'])} @ L{value['ncc_layer']}"
                    f"<br>SNR {value['snr']:.3f} / {value['snr_db']:.2f} dB @ L{value['snr_layer']}"
                    f"<br><span class=\"muted\">D/C seeds {esc(seeds)} · confirmation nₖ {support_text}</span>"
                )

            one_seeds = (
                f"{len(one_panel['discovery'])}/{len(one_panel['confirmation'])}"
            )
            rows.append(
                (
                    esc(model),
                    str(pca_dim),
                    cell(non, item["non_support"], "20/10"),
                    cell(one, item["one_support"], one_seeds),
                    cell(one_trace, item["one_support"], one_seeds),
                    cell(aligned, item["aligned_support"], "20/10"),
                    cell(aligned_trace, item["aligned_support"], "20/10"),
                )
            )
    return html_table(
        [
            "模型",
            "PCA",
            "Non-thinking · full panel",
            "Native 1:1 · item_end",
            "Native 1:1 · trace-aware",
            "Native aligned · item_end",
            "Native aligned · trace-aware",
        ],
        rows,
    )


def partial_table(partials: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for row in partials:
        rows.append(
            (
                esc(row["model"]),
                esc(row["split"]),
                str(row["seed"]),
                str(row["gold_count"]),
                str(row["observed"]),
                esc(", ".join(map(str, row["occurrences"]))),
                esc(", ".join(row["cities"])),
                esc(str(row["parsed_count"])),
                "correct" if row["exact_count"] else "wrong",
                esc(row["trace_category"]),
            )
        )
    return html_table(
        [
            "模型",
            "split",
            "seed",
            "gold N",
            "observed items",
            "ordinal labels",
            "parser-observed cities",
            "final count",
            "final outcome",
            "trace category",
        ],
        rows,
    )


def endpoint_table(endpoints: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for endpoint in endpoints:
        strategy = str(endpoint["alignment_strategy"])
        forward = (
            "full baseline sequence"
            if strategy == "literal_baseline_token_prefix"
            else "text-exact retokenized prefix"
        )
        rows.append(
            (
                str(endpoint["occurrence"]),
                esc(endpoint["city"]),
                esc(endpoint.get("item_text", "archive text unavailable")),
                f"[{endpoint['char_start']}, {endpoint['char_end']})",
                str(endpoint["endpoint_token"]),
                str(endpoint["sequence_query_position"]),
                esc(forward),
            )
        )
    return html_table(
        [
            "k",
            "city",
            "parser item text",
            "raw char span",
            "output endpoint (0-based)",
            "model sequence position (0-based)",
            "forward path",
        ],
        rows,
    )


def partial_trace_details(partials: Iterable[Mapping[str, Any]]) -> str:
    blocks = []
    for row in partials:
        if row.get("raw_excerpt") is None:
            continue
        blocks.append(
            "<details>"
            f"<summary>{esc(row['model'])} · {esc(row['split'])} · seed {row['seed']} · "
            f"gold N={row['gold_count']} · "
            f"{row['observed']} observed item_end states · final {esc(row['parsed_count'])}</summary>"
            f"<pre class=\"trace\">{esc(row['raw_excerpt'])}</pre>"
            f"{endpoint_table(row['endpoint_sites'])}</details>"
        )
    if not blocks:
        return '<p class="small">Raw generation archive was not supplied.</p>'
    return "".join(blocks)


def token_extraction_section(visual: Mapping[str, Any]) -> str:
    count_rows = []
    example_rows = []
    policy_rows = []
    for model in MODELS:
        alignment = visual[model]["alignment"]
        strategies = alignment["strategy_counts"]
        count_rows.append(
            (
                esc(model),
                str(alignment["item_end_sites"]),
                str(strategies.get("literal_baseline_token_prefix", 0)),
                str(strategies.get("text_exact_boundary_retokenization", 0)),
                "yes" if alignment["trace_archive_verified"] else "no",
            )
        )
        for example in alignment["examples"]:
            example_rows.append(
                (
                    esc(model),
                    f"{esc(example['split'])} / {example['seed']} / "
                    f"N={example['gold_count']} / k={example['occurrence']}",
                    esc(example["city"]),
                    esc(example.get("item_text", "archive text unavailable")),
                    f"[{example['char_start']}, {example['char_end']})",
                    str(example["endpoint_token"]),
                    str(example["sequence_query_position"]),
                    esc(example["alignment_strategy"]),
                )
            )
        for row in visual[model]["trace_policy"]:
            categories = ", ".join(
                f"{key}={value}"
                for key, value in row["trace_categories"].items()
            )
            strategies = ", ".join(
                f"{key}={value}"
                for key, value in row["alignment_strategies"].items()
            )
            policy_rows.append(
                (
                    esc(model),
                    esc(row["marker_kind"]),
                    esc(row["selected_site_kind"]),
                    str(row["trace_count"]),
                    str(row["one_to_one_count"]),
                    (
                        f"{row['split_trace_counts']['discovery']}/"
                        f"{row['split_trace_counts']['confirmation']}"
                    ),
                    (
                        f"{row['one_to_one_split_trace_counts']['discovery']}/"
                        f"{row['one_to_one_split_trace_counts']['confirmation']}"
                    ),
                    str(row["selected_state_count"]),
                    esc(categories),
                    esc(strategies),
                )
            )
    counts = html_table(
        [
            "模型",
            "all-count item_end sites",
            "literal full-sequence",
            "retokenized prefix",
            "raw archive verified",
        ],
        count_rows,
    )
    examples = html_table(
        [
            "模型",
            "example",
            "city",
            "parser item text",
            "raw char span",
            "output endpoint",
            "sequence position",
            "alignment strategy",
        ],
        example_rows,
    )
    policy = html_table(
        [
            "模型",
            "marker_kind",
            "trace-aware site",
            "all-count trajectories",
            "one-to-one",
            "all D/C traces",
            "1:1 D/C traces",
            "states",
            "trace categories",
            "alignment strategies",
        ],
        policy_rows,
    )
    return f"""
<section id="tokens"><h2>Trace → token → hidden state</h2>
<p>Native trace 先由 hybrid parser 构造 span-supported 的 <code>1..M</code> episode：优先配对局部 city span 与显式 rank evidence，在每次 rank-1 restart 处分段；若 structural parser 的 city sequence 以该 ranked episode 为精确前缀并继续加入新城市，则保留这段无标号续接。相同 <code>(rank, city)</code> 的重复 evidence 会合并，但 rank 前进时重复 city 不会被去重。注册 gold N 与最终 <code>Total</code> 都不参与 episode 构造或选择；synthetic evidence fallback 单独标记且不能进入 one-to-one cohort。第 k 个 parser item 的字符区间是 <code>[char_start, char_end)</code>，主站点 <code>item_end:k</code> 取该区间结束边界。随后用原始生成时保存的 <code>output_token_ids</code> 做 text-exact 对齐；只有 <code>alignment_eligible=true</code> 的站点进入 geometry。</p>
<div class="callout"><strong>实际 capture 公式：</strong>相对 output 的 0-based endpoint 是 <code>endpoint_token = prefix_token_count - 1</code>；模型输入中的 0-based query position 是 <code>prompt_token_count + endpoint_token</code>。forward hook 读取每个 decoder block 的输出在该位置的 residual-stream vector，所以 L0 表示第 0 个 decoder block 的输出，而不是 embedding layer。</div>
<div class="definitions two"><div><h3>Literal baseline boundary</h3><p>若 <code>raw_text[:char_end]</code> 与原始 output token prefix 完全同边界，则在一次完整 <code>prompt + output</code> forward 中直接取上述位置。</p></div><div><h3>Text-exact boundary retokenization</h3><p>若字符边界切开 tokenizer 的合并，不能错误地借用跨边界 token。此时将 <code>raw_text[:char_end]</code> 精确重分词，单独 forward <code>prompt + retokenized prefix</code>，取最后一个 token；该 state 与原 trace 共享边界前的 baseline prefix，但不是完整 trace forward 中某个 token 的冒名替代。</p></div></div>
<div class="callout warning"><strong>Non-thinking 对照：</strong>站点不来自 response trace，而来自 prompt 中第 k 个 exact needle token span <code>[start,end)</code>；<code>span_end</code> 明确定义为每层 block 输出的 <code>hidden[0, end-1]</code>。因此三列共享 ordinal k，但 native 与 non-thinking 的 token 语义不同。</div>
{counts}<details><summary>Representative character/token alignments</summary>{examples}</details>
<h3>Parser-aware anchor sensitivity</h3>
<p>parser 同时给出完整性类别 <code>trace_category</code> 和表面格式 <code>marker_kind</code>。实现固定到 <a href="{esc(PARSER_UPSTREAM_REPOSITORY)}/commit/{esc(PARSER_UPSTREAM_COMMIT)}"><code>{esc(PARSER_UPSTREAM_COMMIT[:12])}</code></a>。这里<strong>不让 trace_category 决定 token</strong>：one-to-one/partial/duplicate 是整条轨迹的覆盖结果，用它反向挑 token 会把 completion selection 写进 measurement。敏感性策略只读取 marker_kind：<code>indexed/ordinal/inline_count → marker_end</code>；<code>bullet/audit_sentence/completion_recap/evidence_sequence → item_end</code>。无显式序号的格式没有可读取的 k marker。</p>
<div class="callout warning"><strong>解释边界：</strong>trace-aware 是异质站点的诊断，不替代统一 <code>item_end</code> 主分析。尤其 indexed/ordinal 的 marker 本身可能直接暴露 k，分类变好可能只是显式文本 cue，而不是更紧密的内部 count state。报告因此允许切换两套 anchor，并分别重算每层 PCA、probe 与 SNR。</div>
{policy}
<div class="callout warning"><strong>split/site composition audit：</strong>上表分别列出 discovery/confirmation 的 marker-kind、one-to-one 与 selected-site 支持。若两侧格式构成不同，异质 trace-aware 分数会同时包含位置可读性与 site-composition shift；这种结果只能作敏感性诊断，不能单独作为内部 counter 更紧的证据。</div>
</section>"""


def _float_or_nan(row: Mapping[str, Any], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _pct_or_dash(row: Mapping[str, Any], field: str) -> str:
    value = _float_or_nan(row, field)
    return "—" if not np.isfinite(value) else pct(value)


def _db_or_dash(row: Mapping[str, Any], field: str) -> str:
    value = _float_or_nan(row, field)
    return "—" if not np.isfinite(value) else f"{value:+.2f} dB"


def trace_stratified_section(
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    cue_labels = {
        "indexed": "逐项唯一数字（显式 k cue）",
        "ordinal": "逐项 ordinal word（显式/半显式 k cue）",
        "bullet": "各项相同 bullet（不唯一标识 k）",
        "inline_count": "city 后 Count/Record k（显式 k cue）",
        "audit_sentence": "句式 fallback（无逐项 marker）",
        "completion_recap": "recap fallback（无逐项 marker）",
        "evidence_sequence": "score-supported evidence chain（无逐项 marker）",
    }
    marker_order = {
        name: index
        for index, name in enumerate(
            [
                "indexed",
                "ordinal",
                "inline_count",
                "bullet",
                "audit_sentence",
                "completion_recap",
                "evidence_sequence",
            ]
        )
    }
    selector_labels = {
        "fixed_item_end": "fixed item_end",
        "post_marker_site_search": "post-marker search",
        "pre_label_site_search": "pre-marker / pre-label",
        "low_leakage_site_search": "low-leakage site search",
        "all_site_search": "all-site search",
    }
    selector_order = {
        "fixed_item_end": 0,
        "post_marker_site_search": 1,
        "pre_label_site_search": 2,
        "low_leakage_site_search": 3,
        "all_site_search": 4,
    }
    grade_labels = {
        "claim_grade": "bounded-claim eligible",
        "exploratory_only": "exploratory only",
        "not_evaluable": "not evaluable",
    }

    eligibility_rows = []
    selected_rows: list[dict[str, Any]] = []
    for model in MODELS:
        payload = results[model]
        for row in sorted(
            payload["eligibility"],
            key=lambda item: marker_order.get(item["marker_kind"], 999),
        ):
            labels = [int(value) for value in row["retained_labels"].split()]
            discovery_support = {
                int(key): int(value)
                for key, value in json.loads(row["discovery_support"]).items()
            }
            confirmation_support = {
                int(key): int(value)
                for key, value in json.loads(row["confirmation_support"]).items()
            }
            if labels:
                support = (
                    f"D {min(discovery_support[k] for k in labels)}–"
                    f"{max(discovery_support[k] for k in labels)} / "
                    f"C {min(confirmation_support[k] for k in labels)}–"
                    f"{max(confirmation_support[k] for k in labels)}"
                )
                retained = f"{min(labels)}–{max(labels)} ({len(labels)} classes)"
            else:
                support = "—"
                retained = "—"
            eligibility_rows.append(
                (
                    esc(model),
                    f"<code>{esc(row['marker_kind'])}</code>",
                    esc(cue_labels.get(row["marker_kind"], "unknown")),
                    f"{row['discovery_seed_count']} / {row['confirmation_seed_count']}",
                    esc(retained),
                    esc(support),
                    esc(grade_labels.get(row["eligibility"], row["eligibility"])),
                )
            )

        selections = sorted(
            payload["selection"],
            key=lambda item: (
                marker_order.get(item["marker_kind"], 999),
                selector_order.get(item["selector"], 999),
            ),
        )
        baselines = {
            row["marker_kind"]: row
            for row in selections
            if row["selector"] == "fixed_item_end"
        }
        for row in selections:
            baseline = baselines[row["marker_kind"]]
            logistic_delta = 100 * (
                _float_or_nan(row, "confirmation_logistic_balanced_accuracy")
                - _float_or_nan(
                    baseline, "confirmation_logistic_balanced_accuracy"
                )
            )
            ncc_delta = 100 * (
                _float_or_nan(row, "confirmation_ncc_balanced_accuracy")
                - _float_or_nan(baseline, "confirmation_ncc_balanced_accuracy")
            )
            selected_rows.append(
                {
                    "model": model,
                    "marker_kind": row["marker_kind"],
                    "eligibility": row["eligibility"],
                    "selector": row["selector"],
                    "logistic_delta": logistic_delta,
                    "ncc_delta": ncc_delta,
                    "baseline_layer": int(baseline["layer"]),
                    "html": (
                        esc(model),
                        f"<code>{esc(row['marker_kind'])}</code>",
                        esc(grade_labels[row["eligibility"]]),
                        esc(selector_labels[row["selector"]]),
                        f"<code>{esc(row['site_kind'])}</code> @ L{row['layer']}",
                        (
                            f"{_pct_or_dash(row, 'discovery_oof_logistic_balanced_accuracy')} / "
                            f"{_pct_or_dash(row, 'discovery_oof_ncc_balanced_accuracy')}"
                        ),
                        (
                            f"{_pct_or_dash(row, 'confirmation_logistic_balanced_accuracy')} / "
                            f"{_pct_or_dash(row, 'confirmation_ncc_balanced_accuracy')}"
                        ),
                        (
                            f"vs <code>item_end</code> @ L{baseline['layer']}: "
                            f"{logistic_delta:+.1f} / {ncc_delta:+.1f} pp"
                        ),
                        _db_or_dash(row, "confirmation_class_balanced_snr_db"),
                        (
                            f"{row['confirmation_seed_count']} seeds; "
                            f"nₖ {row['confirmation_support_min']}–"
                            f"{row['confirmation_support_max']}"
                        ),
                    ),
                    "row": row,
                }
            )

    implicit_claim_rows = [
        item
        for item in selected_rows
        if item["eligibility"] == "claim_grade"
        and item["marker_kind"]
        in {"bullet", "audit_sentence", "completion_recap", "evidence_sequence"}
        and item["selector"] == "low_leakage_site_search"
    ]
    implicit_sentences = []
    for item in implicit_claim_rows:
        row = item["row"]
        implicit_sentences.append(
            f"{esc(item['model'])} <code>{esc(item['marker_kind'])}</code>: "
            f"discovery 选出 <code>{esc(row['site_kind'])}</code> @ L{row['layer']}；"
            f"confirmation Logistic/NCC = "
            f"{_pct_or_dash(row, 'confirmation_logistic_balanced_accuracy')} / "
            f"{_pct_or_dash(row, 'confirmation_ncc_balanced_accuracy')}，"
            f"SNR = {_db_or_dash(row, 'confirmation_class_balanced_snr_db')} "
            f"(chance = {_pct_or_dash(row, 'chance_balanced_accuracy')})"
        )
    implicit_summary = (
        "；".join(implicit_sentences)
        if implicit_sentences
        else "没有通过 support gate 的无唯一序号 strata"
    )
    post_marker_deltas = [
        (
            f"{esc(item['model'])} <code>{esc(item['marker_kind'])}</code> "
            f"<code>{esc(item['row']['site_kind'])}</code> @ L{item['row']['layer']}: "
            f"相对独立选择的 <code>item_end</code> "
            f"@ L{item['baseline_layer']}, "
            f"ΔLog/NCC = {item['logistic_delta']:+.1f} / "
            f"{item['ncc_delta']:+.1f} pp"
        )
        for item in selected_rows
        if item["eligibility"] == "claim_grade"
        and item["selector"] == "post_marker_site_search"
    ]
    post_marker_summary = "；".join(post_marker_deltas)

    return f"""
<section id="strata"><h2>按 trace 格式分层：token-site × layer sweep</h2>
<p>这里按 parser 的表面格式 <code>marker_kind</code> 分层，不按 one-to-one/partial 等 <code>trace_category</code> 分层。每个 stratum 在语义存在时尝试 <code>pre_marker</code>、<code>marker_end</code>、<code>pre_city</code>、<code>city_end</code>、<code>city_unit_end</code>、<code>item_end</code> 与 <code>post_boundary</code>。其中 <code>city_unit_end</code> 是包含该城市的 sentence/physical-line 末端，用来保留 senior parser 的语义单元边界优势。<strong>每个 selector 都在自己的候选 token sites × 全部层中独立选择最佳组合，不要求不同 selector 或两个模式使用同一层。</strong>选择只使用 discovery seeds 的 seed-grouped Logistic/NCC 平均 balanced accuracy；同一 seed 的十个 count 条件不会跨 train/test。每个 fold 内重新拟合 StandardScaler 与 PCA16，再在 confirmation 上评价冻结组合。</p>
<div class="callout warning"><strong>分析地位：</strong>这是看到 pooled geometry 后新增的 post-hoc robustness analysis，不是预注册的独立复现。程序化选择不读取 confirmation，但这批 confirmation seeds 已在更早的总体分析中出现过。因此它能降低直接的 site/layer overfitting，不能把结果升级为全新的 confirmatory evidence。</div>
<h3>哪些类别有足够支持？</h3>
{html_table(['模型', 'marker_kind', '表面 cue', 'D/C seeds', '保留 k', '逐类支持', '证据等级'], eligibility_rows)}
<h3>Discovery-frozen 位置与 confirmation 结果</h3>
<p class="small"><code>fixed item_end</code> 只固定 token-site，层仍由自己的 discovery sweep 选择；<code>pre-label</code> 取显式 marker 之前；<code>low-leakage</code> 排除 <code>marker_end</code>；<code>all-site</code> 允许显式 marker，作为 lexical-cue positive control。Δ 比较同一 stratum 下两个<strong>各自在 discovery 上选出的最佳组合</strong>，混合 site 与 layer 变化，不能解释为同层控制后的纯 token-site 效应。</p>
{html_table(['模型', 'stratum', '等级', 'selector', '各自独立 D-selected site/layer', 'D OOF Log/NCC', 'C Log/NCC', 'ΔC vs 各自最佳 item_end', 'C SNR', 'C support'], [item['html'] for item in selected_rows])}
<div class="callout"><strong>位置搜索结果：</strong>对通过 support gate 的 strata，post-marker selector 相对各自独立选择的 fixed item_end 的 confirmation 变化为：{post_marker_summary or '没有可比较的 claim-grade stratum'}。是否改善必须逐行判断；报告不会预设某个 token site 跨格式占优。</div>
<div class="callout"><strong>可支持的表述：</strong>“在 native-thinking response 中，ordinal position 在按表面格式分层后仍可由 hidden states held-out 解码；而且这一现象至少在没有逐项唯一序号 token 的格式中存在，因此 pooled decodability 不能完全归结为读取显式编号。”当前无唯一编号、通过 support gate 的 strata 为：{implicit_summary}。</div>
<div class="callout warning"><strong>不可支持的表述：</strong>这些结果仍不能单独证明离散计数器、count chord 或递增更新机制。即使取 <code>post_boundary</code>，自回归上下文仍包含先前项目；probe 可能读取序列长度、句式进度、重复次数或其他位置相关 cue。indexed 的近满分尤其应解释为显式序号 positive control，而不是内部 counter 的主证据。SNR 与 classification 若方向不一致，应写成“更可解码但不一定更紧密”。</div>
</section>"""


def dual_endpoint_section(
    dual_results: Mapping[str, Mapping[str, Any]],
    dual_visual: Mapping[str, Any],
) -> str:
    summary_rows = []
    category_rows = []
    for model in MODELS:
        payload = dual_results[model]
        selected = payload["running_selected"] + payload["final_selected"]
        for row in selected:
            if str(row["analysis_group"]) not in {"all_traces", "all_counts"}:
                continue
            summary_rows.append(
                (
                    esc(model),
                    "running index" if row["endpoint"] == "running_index" else "final count",
                    "non-thinking" if row["mode"] == "non_thinking" else "native-thinking",
                    f"<code>{esc(row['token_site'])}</code> @ L{int(row['layer'])}",
                    (
                        f"Log {pct(row['discovery_oof_logistic_balanced_accuracy'])} / "
                        f"NCC {pct(row['discovery_oof_ncc_balanced_accuracy'])}"
                    ),
                    (
                        f"Log {pct(row['confirmation_logistic_balanced_accuracy'])} / "
                        f"NCC {pct(row['confirmation_ncc_balanced_accuracy'])}"
                    ),
                    f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                    (
                        f"{int(float(row['confirmation_rows']))} states / "
                        f"{int(float(row['confirmation_seed_count']))} seeds / "
                        f"nₖ {int(float(row['confirmation_support_min']))}–"
                        f"{int(float(row['confirmation_support_max']))}"
                    ),
                )
            )
        for row in payload["running_selected"]:
            if str(row["mode"]) != "native_thinking" or str(
                row["analysis_group"]
            ) == "all_traces":
                continue
            group = str(row["analysis_group"])
            role = (
                "lexical positive control"
                if str(row.get("selector")) == "lexical_marker_positive_control"
                else (
                    "format-aware pre-label search"
                    if str(row.get("selector")) == "format_aware_pre_label"
                    else "low-leakage site/layer search"
                )
            )
            category_rows.append(
                (
                    esc(model),
                    f"<code>{esc(group)}</code>",
                    esc(role),
                    esc(row["retained_labels"]),
                    f"<code>{esc(row['token_site'])}</code> @ L{int(row['layer'])}",
                    (
                        f"Log {pct(row['discovery_oof_logistic_balanced_accuracy'])} / "
                        f"NCC {pct(row['discovery_oof_ncc_balanced_accuracy'])}"
                    ),
                    (
                        f"Log {pct(row['confirmation_logistic_balanced_accuracy'])} / "
                        f"NCC {pct(row['confirmation_ncc_balanced_accuracy'])}"
                    ),
                    f"{float(row['confirmation_class_balanced_snr_db']):.2f} dB",
                )
            )

    panel_labels = {
        "running_non": (
            "Running index · non-thinking",
            "Prompt evidence span 的最后一个 token；站点固定为 span_end，只由 discovery 选择 layer。",
        ),
        "running_native": (
            "Running index · native-thinking",
            "Thinking trace 完整 item 的最后一个 token；站点固定为 item_end，只由 discovery 选择 layer。",
        ),
        "final_non": (
            "Final count · non-thinking",
            "Prompt 末尾 Total: query state；类别是 gold N=1…10。",
        ),
        "final_native": (
            "Final count · native-thinking",
            "Thinking trace 中紧邻 numeric final answer 之前的最后一个 literal token。",
        ),
    }
    model_blocks = []
    for model in MODELS:
        slug = "qwen" if model.startswith("Qwen") else "gemma"
        cards = []
        for panel, (title, description) in panel_labels.items():
            payload = dual_visual[model]["panels"][panel]
            options = "".join(
                f'<option value="{layer}"{(" selected" if layer == payload["default_layer"] else "")}>L{layer}</option>'
                for layer in payload["layers"]
            )
            full_label = "full panel · 300 trajectories · N=1…10"
            evaluation_label = "confirmation · 100 trajectories · N=1…10"
            cards.append(
                f"""<article class="geometry-card dual-card"><h3>{esc(title)}</h3>
<p>{esc(description)}</p><div class="controls"><label>Layer<select id="dual-{slug}-{panel}-layer">{options}</select></label>
<label>Rows<select id="dual-{slug}-{panel}-split"><option value="all">{esc(full_label)}</option><option value="confirmation">{esc(evaluation_label)}</option></select></label></div>
<canvas id="dual-{slug}-{panel}" data-model="{esc(model)}" data-panel="{panel}"></canvas>
<div class="rotate-hint">drag to rotate · every layer available · discovery-fitted PCA3</div>
<div class="panel-stats" id="dual-{slug}-{panel}-stats"></div></article>"""
            )
        model_blocks.append(
            f"<h3>{esc(model)}</h3><div class=\"dual-grid\">{''.join(cards)}</div>"
        )

    category_html = ""
    if category_rows:
        category_html = f"""
<details><summary>Appendix diagnostic · pooled trace-format groups</summary>
<p class="small">这些是格式异质性诊断，不参与主结果的 site/layer 选择。</p>
{html_table(['模型', 'pooled group', '角色', '保留 k', 'D-selected site/layer', 'D OOF', '冻结层 held-out', 'SNR'], category_rows)}
</details>"""

    return f"""
<section id="dual"><h2>两个 endpoint，各自在自己的最佳表征上比较</h2>
<p>Running-index 的测量算子固定为单-token 完成边界：non-thinking 使用 <code>span_end</code>，native-thinking 使用 <code>item_end</code>，主分析不再搜索其他站点。两个模式仍各自在 discovery 中选择自己的 decoder layer；程序按 5-fold seed-grouped OOF Logistic/NCC balanced accuracy 的平均值选层，confirmation 不进入 selector。定量空间是每 fold 内重拟合的 StandardScaler + whitened PCA16；下方 3D 仅作显示，每层独立用 discovery 拟合 PCA3。</p>
<div class="definitions two"><div><h3>Running index</h3><p><strong>non-thinking：</strong>prompt 中第 k 个 evidence span。<strong>native-thinking：</strong>thinking trace 中 parser-observed 的第 k 项。两边类别都是 k=1…10，但 token 语义不同。</p></div><div><h3>Final count</h3><p><strong>non-thinking：</strong>prompt-final <code>Total:</code> query。<strong>native-thinking：</strong>numeric final answer 前的最后一个 thinking token。两边类别都是 gold N=1…10。</p></div></div>
<div class="callout"><strong>判读规则：</strong>running-index 使用固定的 end site 和各自 discovery-frozen 最佳 layer；final-count 也使用预先定义的 query site 和各自最佳 layer。只有 held-out Logistic 与 NCC 同向提高时才写“更可解码”；只有 SNR 也提高时才进一步写“类间/类内比更高”。三项不一致时分别报告，不能笼统写成“几何更紧”。</div>
<div class="callout"><strong>统一 split：</strong>running-index 与 final-count 都使用注册的 20-seed discovery / 10-seed confirmation。每个 final-count 模式的 confirmation 恰好是 10 counts × 10 seeds = 100 个 states；同一 seed 的十个 count 条件始终在同一侧。</div>
{html_table(['模型', 'endpoint', '模式', '固定 site / D-selected layer', 'D OOF', '冻结层 held-out', 'held-out SNR', 'support'], summary_rows)}
{category_html}
<h3>每层 3D：四个 panel 各自切 layer</h3>
<p class="small">每张图固定使用该 panel 由 discovery 选中的 token site，但 layer 可独立浏览全部 decoder blocks；因此不会把两个模式锁到同一层。点色是 running k 或 gold final count N。统计栏同时显示该层的 discovery OOF 与 held-out Logistic/NCC/SNR；all-layer held-out 曲线仅作透明诊断，程序化赢家仍只读取 discovery 列。</p>
{''.join(model_blocks)}
</section>"""


def model_section(model: str, payload: Mapping[str, Any]) -> str:
    slug = "qwen" if model.startswith("Qwen") else "gemma"
    options = "".join(
        f'<option value="{layer}"{(" selected" if layer == payload["default_layer"] else "")}>L{layer}</option>'
        for layer in payload["layers"]
    )
    cards = []
    definitions = (
        (
            "non_thinking",
            "1 · Non-thinking",
            "10 个 gold count × 30 seeds；prompt 中第 k 个真实 needle 的 span-end。",
        ),
        (
            "native_one_to_one",
            "2 · Native-thinking · one-to-one",
            "parser-observed city multiset 与 gold 严格相等；不按最终答案正确性筛选。",
        ),
        (
            "native_aligned",
            "3 · Native-thinking · ordinal-aligned",
            "完整 300 条轨迹；实际写出的第 k 项就是位置 k，允许后段缺失。",
        ),
    )
    for key, title, description in definitions:
        cards.append(
            f'<article class="geometry-card"><h3>{esc(title)}</h3>'
            f'<p>{esc(description)}</p>'
            f'<canvas id="{slug}-{key}" data-model="{esc(model)}" data-panel="{key}"></canvas>'
            f'<div class="rotate-hint">drag to rotate · discovery-fitted PC1/PC2/PC3</div>'
            f'<div class="panel-stats" id="{slug}-{key}-stats"></div></article>'
        )
    return f"""
<section id="{slug}">
  <div class="section-title"><div><div class="eyebrow">MODEL COMPARISON</div><h2>{esc(model)}</h2></div>
  <div class="controls"><label>Layer <select id="{slug}-layer">{options}</select></label>
  <label>Displayed panel <select id="{slug}-split"><option value="confirmation">confirmation · 100 trajectories</option><option value="all">all · 300 trajectories</option></select></label>
  <label>Native anchor <select id="{slug}-anchor"><option value="uniform">uniform item_end · primary</option><option value="trace_aware">trace-aware count boundary · sensitivity</option></select></label>
  <label>Final outcome <select id="{slug}-outcome"><option value="all">all</option><option value="correct">correct</option><option value="wrong">wrong</option></select></label></div></div>
  <div class="geometry-grid">{''.join(cards)}</div>
  <div class="metric-block"><h3>Every-layer held-out tightness</h3><p class="small">PCA32；standardization/PCA/probes 只在 discovery 拟合。三张曲线都只评价 confirmation；Native anchor 切换会同步替换两条 native 曲线，点击曲线附近可切换上面的 3D layer。</p>
  <div class="metric-grid"><article><h4>Logistic balanced accuracy</h4><canvas id="{slug}-metric-logistic"></canvas></article><article><h4>Nearest-centroid balanced accuracy</h4><canvas id="{slug}-metric-ncc"></canvas></article><article><h4>Class-balanced SNR (dB)</h4><canvas id="{slug}-metric-snr_db"></canvas></article></div></div>
</section>
"""


def build_html(
    comparison: Mapping[str, Mapping[int, Mapping[str, Any]]],
    visual: Mapping[str, Any],
    partials: list[dict[str, Any]],
    trace_stratified: Mapping[str, Mapping[str, Any]],
    dual_results: Mapping[str, Mapping[str, Any]],
    dual_visual: Mapping[str, Any],
) -> str:
    visual_json = json.dumps(visual, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    dual_visual_json = json.dumps(
        dual_visual, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    css = """
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#626A74;--line:#C9C2B6;--indigo:#23165C;--violet:#6750E8;--teal:#00A88F;--yellow:#D6B52C}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}nav{position:sticky;top:0;z-index:5;display:flex;gap:18px;padding:10px 22px;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line)}nav a{color:var(--indigo);font-size:13px;font-weight:750;text-decoration:none}main{max-width:1480px;margin:auto;padding:38px 28px 80px}header{max-width:1080px;border-bottom:2px solid var(--ink);padding-bottom:28px}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;color:var(--teal)}h1{font-size:44px;line-height:1.08;margin:10px 0 16px;letter-spacing:-.035em}h2{font-size:29px;margin:0}.lead{font-size:18px;color:#404852;max-width:92ch}section{padding:46px 0;border-bottom:1px solid var(--line)}.callout{max-width:1080px;background:var(--surface);border-left:4px solid var(--teal);padding:15px 19px;margin:20px 0}.warning{border-left-color:var(--yellow)}.definitions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.definitions.two{grid-template-columns:repeat(2,minmax(0,1fr))}.definitions>div,.geometry-card{background:var(--surface);border:1px solid var(--line);padding:17px}.definitions h3,.geometry-card h3{color:var(--indigo);margin:0 0 8px;font-size:17px}.definitions p,.geometry-card p{font-size:13px;color:var(--muted);margin:0 0 12px}.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:18px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label{font-size:12px;font-weight:700;color:var(--muted)}select{display:block;margin-top:4px;border:1px solid var(--line);background:var(--surface);padding:7px 28px 7px 9px;color:var(--ink)}.geometry-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.geometry-card canvas{display:block;width:100%;height:360px;background:#F8F4EC;border:1px solid #DDD5C9;touch-action:none;cursor:grab}.geometry-card canvas:active{cursor:grabbing}.rotate-hint{margin-top:5px;color:#7A7270;font:10px/1.4 Consolas,monospace}.panel-stats{min-height:70px;margin-top:7px;color:var(--muted);font:12px/1.5 Consolas,monospace}.metric-block{margin-top:24px}.metric-block h3{color:var(--indigo);margin-bottom:2px}.metric-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:12px}.metric-grid article{background:var(--surface);border:1px solid var(--line);padding:12px}.metric-grid h4{font-size:13px;color:var(--indigo);margin:0 0 7px}.metric-grid canvas{width:100%;height:190px;display:block;background:#F8F4EC;border:1px solid #DDD5C9;cursor:crosshair}.table-scroll{overflow:auto;background:var(--surface);border:1px solid var(--line);margin:16px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #DED8CE}th{position:sticky;top:0;background:#ECE6DA;color:#303744}.muted{color:var(--muted);font-size:11px}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;font-size:13px}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px;background:var(--violet)}.dot.correct{border:3px solid white;box-shadow:0 0 0 1px #49515B}.dot.wrong{border:2px solid #20242D}.small{font-size:13px;color:var(--muted);max-width:110ch}details{background:var(--surface);border:1px solid var(--line);margin:18px 0}summary{cursor:pointer;padding:12px 15px;font-weight:750;color:var(--indigo)}details .table-scroll{border:0;border-top:1px solid var(--line);margin:0}.trace{white-space:pre-wrap;overflow-wrap:anywhere;background:#F8F4EC;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0;padding:16px;font:12px/1.55 Consolas,monospace}.provenance{font:11px/1.6 Consolas,monospace;color:var(--muted)}
.dual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:28px}.dual-card .controls{margin:10px 0}.dual-card canvas{height:390px}
@media(max-width:1050px){.geometry-grid,.dual-grid,.definitions,.definitions.two,.metric-grid{grid-template-columns:1fr}.geometry-card canvas{height:390px}.section-title{align-items:flex-start;flex-direction:column}}@media(max-width:650px){main{padding:25px 13px 60px}h1{font-size:34px}.geometry-card canvas{height:330px}}
"""
    script = """
const DATA=__VISUAL_DATA__;
const DUAL=__DUAL_VISUAL_DATA__;
const COLORS=['#6750E8','#00A9D8','#00A88F','#2DBE77','#A7C957','#D6B52C','#F29E4C','#E76F51','#D94B86','#8E5DB7'];
const PANEL_COLORS={non_thinking:'#20242D',native_one_to_one:'#D6A900',native_aligned:'#00A88F'};
const PANELS=['non_thinking','native_one_to_one','native_aligned'];
const VIEWS={};
function slug(model){return model.startsWith('Qwen')?'qwen':'gemma'}
function controls(model){const s=slug(model);return {layer:+document.getElementById(s+'-layer').value,split:document.getElementById(s+'-split').value,anchor:document.getElementById(s+'-anchor').value,outcome:document.getElementById(s+'-outcome').value}}
function panelKey(panel,anchor){return panel==='non_thinking'||anchor==='uniform'?panel:panel+'_trace_aware'}
function filtered(model,panel,layer,split,outcome,anchor){
  const key=panelKey(panel,anchor),block=DATA[model].panels[key][String(layer)];if(!block)return {points:[],evr:[],key};
  return {evr:block.evr,key,points:block.points.filter(p=>(split==='all'||p[0]===split)&&(outcome==='all'||(outcome==='correct'?p[3]===1:p[3]===0)))};
}
function rotate3(x,y,z,view){
  const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
  const x1=cy*x+sy*z,z1=-sy*x+cy*z;
  return [x1,cp*y-sp*z1,sp*y+cp*z1];
}
function draw3D(canvas,model,panel,layer,split,outcome,anchor){
  const {points,evr,key}=filtered(model,panel,layer,split,outcome,anchor),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,stat=document.getElementById(canvas.id+'-stats');c.clearRect(0,0,w,h);
  if(!points.length){c.fillStyle='#6A727D';c.font='14px Segoe UI';c.fillText('No states match this filter.',20,30);stat.textContent=`L${layer} · no states match this display filter`;return}
  const view=VIEWS[canvas.id]||(VIEWS[canvas.id]={yaw:-.72,pitch:.46});
  const groups=new Map();for(const p of points){if(!groups.has(p[2]))groups.set(p[2],[]);groups.get(p[2]).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>[k,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.reduce((s,p)=>s+p[6],0)/ps.length,ps.reduce((s,p)=>s+p[7],0)/ps.length,ps.length]);
  const rotated=points.map(p=>({p,r:rotate3(p[5],p[6],p[7],view)}));
  const rcent=cent.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)}));
  const maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[5]),Math.abs(p[6]),Math.abs(p[7])]),1e-6),axisLen=maxAbs*.72;
  const axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));
  const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:23,r:23,t:18,b:22},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.25;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#2C3440';c.globalAlpha=.8;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();
  const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const p=o.p,depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.42+.45*depth;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[3]===1?'#FFFDF8':'#20242D';c.lineWidth=p[3]===1?2.25:1.05;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.7+1.25*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){const p=o.p;c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}
  const seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[8])).size,counts=cent.map(p=>p[4]),registered=split==='confirmation'?100:300,metric=DATA[model].metrics[key][String(layer)],anchorText=panel==='non_thinking'?'prompt span_end':(anchor==='uniform'?'uniform item_end':'trace-aware boundary');
  const metricText=metric?`held-out C: logistic BA ${(100*metric.logistic).toFixed(1)}% · NCC BA ${(100*metric.ncc).toFixed(1)}% · SNR ${metric.snr.toFixed(3)} (${metric.snr_db.toFixed(2)} dB)`:'';
  stat.textContent=`${anchorText} · L${layer} · registered ${registered} trajectories · displayed ${trajectories} trajectories / ${points.length} states · ${seeds} seeds · nₖ ${Math.min(...counts)}–${Math.max(...counts)} · EVR(PC1–3) ${(100*evr.reduce((a,b)=>a+b,0)).toFixed(1)}% · ${metricText}`;
}
function drawMetric(canvas,model,field){
  const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,pad={l:39,r:10,t:13,b:27},layers=DATA[model].layers,ctl=controls(model),current=ctl.layer;
  const series=PANELS.map(panel=>({panel,values:layers.map(layer=>DATA[model].metrics[panelKey(panel,ctl.anchor)][String(layer)][field])}));let ymin,ymax;if(field==='logistic'||field==='ncc'){ymin=0;ymax=1}else{const all=series.flatMap(s=>s.values),span=Math.max(Math.max(...all)-Math.min(...all),1);ymin=Math.min(...all)-.08*span;ymax=Math.max(...all)+.08*span}
  const sx=layer=>pad.l+(layer-layers[0])/Math.max(layers[layers.length-1]-layers[0],1)*(w-pad.l-pad.r),sy=value=>h-pad.b-(value-ymin)/Math.max(ymax-ymin,1e-9)*(h-pad.t-pad.b);
  c.clearRect(0,0,w,h);c.strokeStyle='#D7D0C5';c.lineWidth=1;for(let i=0;i<=4;i++){const y=pad.t+i*(h-pad.t-pad.b)/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke()}
  c.strokeStyle='#8B8490';c.setLineDash([4,3]);c.beginPath();c.moveTo(sx(current),pad.t);c.lineTo(sx(current),h-pad.b);c.stroke();c.setLineDash([]);
  for(const s of series){c.strokeStyle=PANEL_COLORS[s.panel];c.lineWidth=s.panel==='native_aligned'?2.5:1.75;c.beginPath();layers.forEach((layer,i)=>i?c.lineTo(sx(layer),sy(s.values[i])):c.moveTo(sx(layer),sy(s.values[i])));c.stroke();const idx=layers.indexOf(current);c.fillStyle=PANEL_COLORS[s.panel];c.beginPath();c.arc(sx(current),sy(s.values[idx]),3.5,0,Math.PI*2);c.fill()}
  c.fillStyle='#4F5863';c.font='10px Consolas';c.fillText(field==='snr_db'?ymax.toFixed(1):Math.round(100*ymax)+'%',3,pad.t+3);c.fillText(field==='snr_db'?ymin.toFixed(1):Math.round(100*ymin)+'%',3,h-pad.b+3);c.fillText('L'+layers[0],pad.l-5,h-8);c.fillText('L'+layers[layers.length-1],w-pad.r-23,h-8);c.fillText('selected L'+current,Math.max(pad.l,Math.min(w-86,sx(current)-30)),12);
  let lx=pad.l;for(const [panel,label] of [['non_thinking','NT'],['native_one_to_one','1:1'],['native_aligned','aligned']]){c.fillStyle=PANEL_COLORS[panel];c.fillRect(lx,h-20,12,2);c.fillStyle='#4F5863';c.fillText(label,lx+15,h-16);lx+=label==='aligned'?70:48}
  canvas.onclick=e=>{const box=canvas.getBoundingClientRect(),raw=(e.clientX-box.left-pad.l)/Math.max(box.width-pad.l-pad.r,1),target=layers[0]+Math.max(0,Math.min(1,raw))*(layers[layers.length-1]-layers[0]);const nearest=layers.reduce((a,b)=>Math.abs(b-target)<Math.abs(a-target)?b:a);document.getElementById(slug(model)+'-layer').value=String(nearest);redraw(model)};
}
function redraw(model){const ctl=controls(model),s=slug(model);for(const panel of PANELS)draw3D(document.getElementById(s+'-'+panel),model,panel,ctl.layer,ctl.split,ctl.outcome,ctl.anchor);for(const field of ['logistic','ncc','snr_db'])drawMetric(document.getElementById(s+'-metric-'+field),model,field)}
function setup3D(canvas,model,panel){let active=false,lastX=0,lastY=0;canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!active)return;const view=VIEWS[canvas.id]||(VIEWS[canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;const ctl=controls(model);draw3D(canvas,model,panel,ctl.layer,ctl.split,ctl.outcome,ctl.anchor)});const stop=()=>{active=false};canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop)}
for(const model of Object.keys(DATA)){const s=slug(model);for(const key of ['layer','split','anchor','outcome'])document.getElementById(s+'-'+key).addEventListener('change',()=>redraw(model));for(const panel of PANELS)setup3D(document.getElementById(s+'-'+panel),model,panel);redraw(model)}
function dualIds(model,panel){const base='dual-'+slug(model)+'-'+panel;return {base,layer:document.getElementById(base+'-layer'),split:document.getElementById(base+'-split'),canvas:document.getElementById(base)}}
function drawDual3D(model,panel){
  const ids=dualIds(model,panel),payload=DUAL[model].panels[panel],layer=+ids.layer.value,split=ids.split.value,block=payload.coordinates[String(layer)],points=block.points.filter(p=>split==='all'||p[0]==='confirmation'),canvas=ids.canvas,rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);const w=rect.width,h=rect.height,stat=document.getElementById(ids.base+'-stats');c.clearRect(0,0,w,h);
  if(!points.length){c.fillStyle='#6A727D';c.font='14px Segoe UI';c.fillText('No states match this split.',20,30);stat.textContent=`L${layer} · no states`;return}
  const view=VIEWS[canvas.id]||(VIEWS[canvas.id]={yaw:-.72,pitch:.46}),groups=new Map();for(const p of points){if(!groups.has(p[2]))groups.set(p[2],[]);groups.get(p[2]).push(p)}
  const cent=[...groups.entries()].sort((a,b)=>a[0]-b[0]).map(([k,ps])=>[k,ps.reduce((s,p)=>s+p[3],0)/ps.length,ps.reduce((s,p)=>s+p[4],0)/ps.length,ps.reduce((s,p)=>s+p[5],0)/ps.length,ps.length]);
  const rotated=points.map(p=>({p,r:rotate3(p[3],p[4],p[5],view)})),rcent=cent.map(p=>({p,r:rotate3(p[1],p[2],p[3],view)})),maxAbs=Math.max(...points.flatMap(p=>[Math.abs(p[3]),Math.abs(p[4]),Math.abs(p[5])]),1e-6),axisLen=maxAbs*.72,axes=[['PC1','#D14B4B',rotate3(axisLen,0,0,view)],['PC2','#008E7B',rotate3(0,axisLen,0,view)],['PC3','#6750E8',rotate3(0,0,axisLen,view)]];
  const xy=rotated.map(o=>o.r).concat(axes.map(a=>a[2]),[[0,0,0]]);let xmin=Math.min(...xy.map(v=>v[0])),xmax=Math.max(...xy.map(v=>v[0])),ymin=Math.min(...xy.map(v=>v[1])),ymax=Math.max(...xy.map(v=>v[1]));const dx=Math.max(xmax-xmin,1e-6),dy=Math.max(ymax-ymin,1e-6);xmin-=dx*.11;xmax+=dx*.11;ymin-=dy*.11;ymax+=dy*.11;const pad={l:23,r:23,t:18,b:22},sx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),sy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  for(const [label,color,end] of axes){c.strokeStyle=color;c.lineWidth=1.25;c.beginPath();c.moveTo(sx(0),sy(0));c.lineTo(sx(end[0]),sy(end[1]));c.stroke();c.fillStyle=color;c.font='10px Consolas';c.fillText(label,sx(end[0])+3,sy(end[1])-3)}
  c.strokeStyle='#2C3440';c.globalAlpha=.8;c.lineWidth=2;c.beginPath();rcent.forEach((o,i)=>i?c.lineTo(sx(o.r[0]),sy(o.r[1])):c.moveTo(sx(o.r[0]),sy(o.r[1])));c.stroke();const depths=rotated.map(o=>o.r[2]),zmin=Math.min(...depths),zmax=Math.max(...depths),zspan=Math.max(zmax-zmin,1e-6);rotated.sort((a,b)=>a.r[2]-b.r[2]);
  for(const o of rotated){const p=o.p,depth=(o.r[2]-zmin)/zspan;c.globalAlpha=.42+.45*depth;c.fillStyle=COLORS[p[2]-1];c.strokeStyle=p[0]==='confirmation'?'#FFFDF8':'#20242D';c.lineWidth=p[0]==='confirmation'?2.2:1;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),2.7+1.25*depth,0,Math.PI*2);c.fill();c.stroke()}
  c.globalAlpha=1;for(const o of rcent){const p=o.p;c.fillStyle=COLORS[p[0]-1];c.strokeStyle='#20242D';c.lineWidth=1.3;c.beginPath();c.arc(sx(o.r[0]),sy(o.r[1]),5.8,0,Math.PI*2);c.fill();c.stroke();c.fillStyle='#20242D';c.font='10px Consolas';c.fillText(String(p[0]),sx(o.r[0])+7,sy(o.r[1])-6)}
  const seeds=new Set(points.map(p=>p[0]+':'+p[1])).size,trajectories=new Set(points.map(p=>p[0]+':'+p[1]+':'+p[6])).size,counts=cent.map(p=>p[4]),metric=payload.metrics[String(layer)],evr=block.evr.reduce((a,b)=>a+b,0);stat.textContent=`${payload.token_site} · L${layer} · ${trajectories} trajectories / ${points.length} states / ${seeds} seeds · nₖ ${Math.min(...counts)}–${Math.max(...counts)} · EVR3 ${(100*evr).toFixed(1)}% · D OOF Log/NCC ${(100*metric.discovery_logistic).toFixed(1)}%/${(100*metric.discovery_ncc).toFixed(1)}% · held-out Log/NCC ${(100*metric.confirmation_logistic).toFixed(1)}%/${(100*metric.confirmation_ncc).toFixed(1)}% · SNR ${metric.confirmation_snr_db.toFixed(2)} dB`;
}
function setupDual3D(model,panel){const ids=dualIds(model,panel);ids.layer.addEventListener('change',()=>drawDual3D(model,panel));ids.split.addEventListener('change',()=>drawDual3D(model,panel));let active=false,lastX=0,lastY=0;ids.canvas.addEventListener('pointerdown',e=>{active=true;lastX=e.clientX;lastY=e.clientY;ids.canvas.setPointerCapture(e.pointerId)});ids.canvas.addEventListener('pointermove',e=>{if(!active)return;const view=VIEWS[ids.canvas.id]||(VIEWS[ids.canvas.id]={yaw:-.72,pitch:.46});view.yaw+=(e.clientX-lastX)*.009;view.pitch=Math.max(-1.45,Math.min(1.45,view.pitch+(e.clientY-lastY)*.009));lastX=e.clientX;lastY=e.clientY;drawDual3D(model,panel)});const stop=()=>{active=false};ids.canvas.addEventListener('pointerup',stop);ids.canvas.addEventListener('pointercancel',stop);drawDual3D(model,panel)}
for(const model of Object.keys(DUAL))for(const panel of Object.keys(DUAL[model].panels))setupDual3D(model,panel);
let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(()=>{Object.keys(DATA).forEach(redraw);for(const model of Object.keys(DUAL))for(const panel of Object.keys(DUAL[model].panels))drawDual3D(model,panel)},100)});
""".replace("__VISUAL_DATA__", visual_json).replace("__DUAL_VISUAL_DATA__", dual_visual_json)

    partial_confirmation = [row for row in partials if row["split"] == "confirmation"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Geometry Comparison</title><style>{css}</style></head>
<body><nav><a href="#design">口径</a><a href="#tokens">Token 提取</a><a href="#dual">独立最佳层</a><a href="#strata">分类选点</a><a href="#qwen">Qwen</a><a href="#gemma">Gemma</a><a href="#metrics">指标</a><a href="#partial">部分轨迹</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · THREE-COHORT GEOMETRY</div><h1>NiaH Geometry Comparison</h1>
<p class="lead">同一份报告并列展示 non-thinking、经过 one-to-one 结构清洗的 native-thinking，以及按实际出现 ordinal 对齐的 native-thinking；每个模型都从完整的 10 counts × 30 seeds = 300 条注册轨迹出发。</p></header>
<section id="design"><h2>比较口径</h2><div class="definitions"><div><h3>1 · Non-thinking</h3><p>V4.4 的 gold N=1…10 各配 30 seeds。第 k 类是 prompt 中第 k 个真实 needle 的 span-end state；一条 gold-N 轨迹贡献 k=1…N。</p></div><div><h3>2 · Native · one-to-one</h3><p>从同一批 300 条注册轨迹出发，只保留 parser-observed city multiset 与 gold 严格相等、无重复或遗漏者；不筛最终答案正确性。这是 completion-conditioned sensitivity。</p></div><div><h3>3 · Native · ordinal-aligned</h3><p>300 条 parser-hit 全部保留。模型实际写出的第 k 个计数事件标为 k；若一条轨迹只写到 M，就只贡献 1…M，不插值、不补到 gold N。</p></div></div>
<div class="legend"><span><i class="dot"></i>填充颜色 = 位置 k</span><span><i class="dot correct"></i>白色粗边 = 最终答对</span><span><i class="dot wrong"></i>深色边 = 最终答错</span></div>
<div class="callout"><strong>标签分离：</strong>位置标签始终是 <code>occurrence=k</code>；<code>final exact_count</code> 只控制点的轮廓，不参与 PCA、probe class 或 aligned cohort 入选。</div>
<div class="callout"><strong>trajectory 与 state 不混用：</strong>confirmation 是 10 counts × 10 seeds = 100 条轨迹，discovery 是 10 counts × 20 seeds = 200 条轨迹。non-thinking 的 300 条轨迹固定贡献 30×(1+…+10)=1650 个 running-index states；native-thinking 每条按实际 trace 长度 Mᵢ 贡献 Mᵢ 个 states，因此总 state 数可不同。</div>
<div class="callout"><strong>“少数了”的处理：</strong>gold N=10 的 native trace 即使只数到 8，也仍作为一条样本进入，只贡献标签 1…8；最终输出是否正确是独立的轨迹属性，不决定 running-index 标签，也不会虚构 9、10 两个 state。</div>
<div class="callout warning"><strong>站点语义边界：</strong>non-thinking 是 prompt needle endpoint，native-thinking 是 response count-event endpoint。三列比较的是“运行位置几何是否形成”，不是声称它们是同一个 token-level random variable。图可在 confirmation 100 条轨迹与全注册 300 条轨迹之间切换，并始终另报实际可观测 state 数。</div></section>
{token_extraction_section(visual)}
{dual_endpoint_section(dual_results, dual_visual)}
{trace_stratified_section(trace_stratified)}
{model_section('Qwen3-8B', visual['Qwen3-8B'])}
{model_section('Gemma4-E4B', visual['Gemma4-E4B'])}
<section id="metrics"><h2>Held-out 定量比较</h2><p class="small">所有标准化、PCA32、logistic 与 nearest-centroid prototype 只在 discovery 拟合，数值只在 confirmation 评价。Logistic/NCC 报 balanced accuracy，以免 aligned panel 的 late-position 支持较少而改变类权重。SNR 是 confirmation 上的 class-balanced trace ratio：十个 centroid 围绕其等权 grand centroid 的平均平方距离，除以各类内部平均平方残差；同时报告 ratio 与 <code>10 log10(ratio)</code> dB，越高表示单位类内噪声对应的类间信号越强。表中跨层最大值是描述性 layer scan；one-to-one 与 full-panel 的 seed population 不同，不能把差值直接归因于清洗操作。trace-aware 与 item_end 使用相同轨迹和 ordinal support，但 selected token 不同；其差值是 anchor sensitivity，也不能直接解释为内部 counter 增强。</p>
<div class="callout warning"><strong>不要误读跨层最大值：</strong>表中每个指标的峰值可能位于不同层，且 one-to-one 与 full-panel 的轨迹构成不同。<code>marker_end</code> 还直接包含显式 k cue；它只能作为 lexical positive control。机制主张应优先依据 discovery-frozen、低泄漏站点上的 held-out 结果及其逐类支持。</div>{metric_table(comparison)}</section>
<section id="partial"><h2>部分轨迹如何进入 aligned 列</h2><p>下面列出 confirmation 中所有非 one-to-one 轨迹。<code>ordinal labels</code> 正是进入第三列的 class；例如一条 gold N=10 轨迹只出现 <code>1…8</code>，就只贡献八个 state。final Total 只用于审计，不会补出第 9、10 个 endpoint。</p><details open><summary>Confirmation partial trajectories · {len(partial_confirmation)} rows</summary>{partial_table(partial_confirmation)}</details>
<h3>原始 trace 与实际 endpoint</h3><p class="small">下列片段来自服务器 generation 存档，并已按 request ID、prompt token count、output token count 与本地 capture manifest 对账。表中 sequence position 是实际送入对应 forward 的 0-based query index。</p>{partial_trace_details(partial_confirmation)}</section>
<section><h2>解释优先级</h2><div class="callout"><strong>主结果：</strong>第三列（ordinal-aligned full 300-trajectory panel）回答总体问题。第二列只作为敏感性分析，回答“条件于 trace 对其各自 gold N 完整且一一对应时，几何怎样”。若两列不同，首先解释为 trajectory-completion selection，而不是几何被“修复”。</div>
<p class="provenance">Report schema: niah_geometry_comparison_v6_all_counts · 10 counts × 30 seeds per model · display PCA3: discovery-fitted independently per panel and layer · dual endpoint quantitative PCA16-whiten with seed-grouped-CV site/layer selection · pooled quantitative PCA32 · probes: discovery selection / frozen held-out evaluation</p></section>
</main><script>{script}</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--non-thinking-export-root",
        type=Path,
        required=True,
        help=(
            "Root containing the all-count V4.4 representation/capture and "
            "answer_query_all_layers_v1 trees."
        ),
    )
    parser.add_argument(
        "--non-thinking-outcome-root",
        type=Path,
        help=(
            "Optional separate V4 export root containing behavior/capture/"
            "generation_label_index.jsonl. Defaults to --non-thinking-export-root."
        ),
    )
    parser.add_argument("--native-capture-root", type=Path, required=True)
    parser.add_argument("--aligned-geometry-root", type=Path, required=True)
    parser.add_argument("--one-to-one-geometry-root", type=Path, required=True)
    parser.add_argument(
        "--trace-aware-aligned-geometry-root", type=Path, required=True
    )
    parser.add_argument(
        "--trace-aware-one-to-one-geometry-root", type=Path, required=True
    )
    parser.add_argument("--trace-stratified-geometry-root", type=Path, required=True)
    parser.add_argument("--dual-endpoint-root", type=Path, required=True)
    parser.add_argument(
        "--native-final-count-root",
        type=Path,
        required=True,
        help=(
            "Root containing <model>/capture_index.jsonl with answer_query_v3; "
            "the same unified all-count V5 capture root may be used."
        ),
    )
    parser.add_argument(
        "--native-trace-root",
        type=Path,
        help=(
            "Optional root containing <model>/generations.jsonl. When supplied, "
            "raw partial-trace excerpts and character/token endpoint audits are embedded."
        ),
    )
    parser.add_argument(
        "--parser-audit",
        type=Path,
        help=(
            "Optional 600-row hybrid parser audit used for marker and trace-category "
            "proportion sections. Must be supplied together with the "
            "two Qwen band-diagnostic inputs."
        ),
    )
    parser.add_argument(
        "--qwen-band-audit",
        type=Path,
        help="Optional JSON from analyze_native_geometry_bands.py.",
    )
    parser.add_argument(
        "--qwen-band-points",
        type=Path,
        help="Optional confirmation_points.csv from the Qwen band diagnostic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    diagnostic_inputs = (
        args.parser_audit,
        args.qwen_band_audit,
        args.qwen_band_points,
    )
    if any(value is not None for value in diagnostic_inputs) and not all(
        value is not None for value in diagnostic_inputs
    ):
        parser.error(
            "--parser-audit, --qwen-band-audit, and --qwen-band-points must "
            "be supplied together"
        )

    comparison, aligned_peak, metric_inputs = load_metric_comparison(
        args.aligned_geometry_root.resolve(),
        args.one_to_one_geometry_root.resolve(),
        args.trace_aware_aligned_geometry_root.resolve(),
        args.trace_aware_one_to_one_geometry_root.resolve(),
    )
    visual, partials, visual_inputs = build_visual_data(
        args.non_thinking_export_root.resolve(),
        (
            args.non_thinking_export_root.resolve()
            if args.non_thinking_outcome_root is None
            else args.non_thinking_outcome_root.resolve()
        ),
        args.native_capture_root.resolve(),
        aligned_peak,
        comparison,
        None if args.native_trace_root is None else args.native_trace_root.resolve(),
    )
    trace_stratified, trace_stratified_inputs = load_trace_stratified_results(
        args.trace_stratified_geometry_root.resolve()
    )
    dual_results, dual_result_inputs = load_dual_endpoint_results(
        args.dual_endpoint_root.resolve()
    )
    dual_visual, dual_visual_inputs = build_dual_visual_data(
        args.non_thinking_export_root.resolve(),
        args.native_capture_root.resolve(),
        args.native_final_count_root.resolve(),
        dual_results,
    )
    document = build_html(
        comparison,
        visual,
        partials,
        trace_stratified,
        dual_results,
        dual_visual,
    )
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    all_inputs = sorted(
        set(
            metric_inputs
            + visual_inputs
            + trace_stratified_inputs
            + dual_result_inputs
            + dual_visual_inputs
        ),
        key=str,
    )
    manifest = {
        "schema_version": "niah_geometry_comparison_v6_all_counts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "three_columns": [
            "non_thinking_full_panel",
            "native_thinking_one_to_one",
            "native_thinking_ordinal_aligned_full_panel",
        ],
        "position_label": "ordinal occurrence 1-10",
        "registered_trajectory_panel": {
            "counts": list(EXPECTED_COUNTS),
            "seeds": EXPECTED_FULL_PANEL,
            "discovery_trajectories": 200,
            "confirmation_trajectories": 100,
            "total_trajectories_per_model": EXPECTED_TRAJECTORIES,
            "running_state_rule": (
                "trajectory i contributes actual parser-observed occurrences 1..M_i; "
                "never pad to gold N or final Total"
            ),
        },
        "native_anchor_options": {
            "primary": "uniform item_end",
            "sensitivity": {
                "policy": "trace_aware_count_boundary",
                "mapping": TRACE_AWARE_SITE_BY_MARKER_KIND,
            },
        },
        "parser_upstream": {
            "repository": PARSER_UPSTREAM_REPOSITORY,
            "commit": PARSER_UPSTREAM_COMMIT,
        },
        "display_geometry": "all decoder layers; discovery-fitted PCA3; interactive orthographic rotation",
        "quantitative_geometry": "discovery-fitted PCA32; confirmation-only balanced accuracy and class-balanced SNR",
        "trace_stratified_site_sweep": {
            "schema_version": TRACE_STRATIFIED_SCHEMA_VERSION,
            "pca_dim": TRACE_STRATIFIED_PCA_DIM,
            "stratification": "parser marker_kind",
            "selection": "leave-one-discovery-seed-out Logistic/NCC; confirmation excluded from programmed selection",
            "layer_policy": TRACE_STRATIFIED_LAYER_COMPARISON_POLICY,
            "status": "post-hoc robustness analysis, not an independent preregistered confirmation",
        },
        "dual_endpoint_analysis": {
            "schema_version": DUAL_ENDPOINT_SCHEMA_VERSION,
            "pca_dim": 16,
            "pca_whiten": PCA_WHITEN,
            "comparisons": [
                "non-thinking prompt running index vs native thinking-trace running index",
                "non-thinking prompt-final count vs native thinking-trace final count",
            ],
            "layer_policy": "independently selected within mode from discovery grouped CV",
            "display": "all decoder layers at each panel's discovery-selected token site",
            "registered_split": EXPECTED_FULL_PANEL,
            "final_count_confirmation_trajectories_per_mode": 100,
        },
        "native_primary_site": "parser item_end:k aligned to endpoint_token=prefix_token_count-1",
        "final_correctness_role": "display attribute only; never a geometry class or primary cohort filter",
        "inputs": {str(path): sha256(path) for path in all_inputs},
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if all(value is not None for value in diagnostic_inputs):
        from augment_niah_geometry_comparison_report import update_report

        update_report(
            report=output,
            manifest=manifest_path,
            parser_audit=args.parser_audit.resolve(),
            native_capture_root=args.native_capture_root.resolve(),
            native_trace_root=(
                None
                if args.native_trace_root is None
                else args.native_trace_root.resolve()
            ),
            dual_root=args.dual_endpoint_root.resolve(),
            band_audit=args.qwen_band_audit.resolve(),
            band_points=args.qwen_band_points.resolve(),
        )
        manifest = read_json(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
