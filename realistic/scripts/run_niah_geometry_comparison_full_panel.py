#!/usr/bin/env python3
"""CPU post-processing pipeline for the strict 300/100 geometry report."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    analyze_dual_endpoint_geometry,
)
from scripts.analyze_native_geometry_bands import analyze as analyze_bands  # noqa: E402
from scripts.audit_realistic_niah_v5_geometry_capture import (  # noqa: E402
    audit_capture,
)
from scripts.build_niah_geometry_comparison_report import (  # noqa: E402
    expected_trajectory_keys,
    read_csv,
    read_json,
    read_jsonl,
)
from scripts.build_niah_geometry_comparison_report_v7 import (  # noqa: E402
    build_report,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _one_native_running_winner(path: Path) -> dict[str, str]:
    matches = [
        row
        for row in read_csv(path)
        if str(row.get("mode")) == "native_thinking"
        and str(row.get("analysis_group")) == "all_traces"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one pooled native running winner in {path}; got {len(matches)}"
        )
    return matches[0]


def _audit_final_capture(path: Path, model: str) -> None:
    rows = read_jsonl(path)
    keys = {
        (str(row["split"]), int(row["seed"]), int(row["gold_count"]))
        for row in rows
    }
    if len(rows) != 300 or keys != expected_trajectory_keys():
        raise ValueError(
            f"{model} final capture is not the registered 10 x 30 panel: "
            f"rows={len(rows)}, unique_keys={len(keys)}"
        )
    for row in rows:
        if str(row.get("model_label")) != model:
            raise ValueError(f"{model} final capture contains another model")
        manifest_path = path.parent / str(row["manifest_path"])
        states_path = path.parent / str(row["states_path"])
        if not manifest_path.is_file() or not states_path.is_file():
            raise FileNotFoundError(
                f"{model} final capture is missing files for {row.get('request_id')}"
            )
        manifest = read_json(manifest_path)
        sites = list(manifest.get("site_rows", []))
        if len(sites) != 1 or str(sites[0].get("site_kind")) != "answer_query_v3":
            raise ValueError(
                f"{model}/{row.get('request_id')}: final shard must contain "
                "exactly one answer_query_v3"
            )


def run_pipeline(
    *,
    non_thinking_root: Path,
    native_running_root: Path,
    native_final_root: Path,
    native_trace_root: Path,
    parser_audit: Path,
    grammar_registry: Path,
    analysis_root: Path,
    band_root: Path,
    grammar_filter_root: Path | None = None,
    pure_trace_n10_root: Path | None = None,
    indexed_numeric_n10_root: Path | None = None,
    output: Path,
    manifest: Path,
    domain_transfer_root: Path | None = None,
    covariance_root: Path | None = None,
    pca_dim: int = 16,
    cv_folds: int = 5,
    random_state: int = 0,
) -> dict[str, object]:
    command = " ".join(shlex.quote(argument) for argument in sys.argv)
    for model in MODELS:
        non_running = (
            non_thinking_root
            / model
            / "numeric"
            / "representation"
            / "capture"
            / "capture_index.jsonl"
        )
        non_final = (
            non_thinking_root
            / model
            / "numeric"
            / "representation"
            / "answer_query_all_layers_v1"
            / "capture_index.jsonl"
        )
        native_running = native_running_root / model / "capture_index.jsonl"
        native_final = native_final_root / model / "capture_index.jsonl"
        model_analysis = analysis_root / model / "pca16_whiten"
        running_audit = audit_capture(native_running)
        if str(running_audit["model_label"]) != model:
            raise ValueError(
                f"Native running audit model mismatch: "
                f"{running_audit['model_label']} versus {model}"
            )
        _audit_final_capture(native_final, model)
        print(f"[geometry CPU] dual endpoint start: {model}", flush=True)
        paths = analyze_dual_endpoint_geometry(
            non_thinking_running_index=non_running,
            native_thinking_running_index=native_running,
            non_thinking_final_count=non_final,
            native_thinking_final_count=native_final,
            output_dir=model_analysis,
            pca_dim=pca_dim,
            cv_folds=cv_folds,
            random_state=random_state,
            command=command,
        )
        winner = _one_native_running_winner(paths["running_selected"])
        print(
            f"[geometry CPU] band appendix start: {model} "
            f"{winner['token_site']} @ L{int(float(winner['layer']))}",
            flush=True,
        )
        analyze_bands(
            capture_index=native_running,
            trace_archive=native_trace_root / model / "generations.jsonl",
            layer=int(float(winner["layer"])),
            site_kind=str(winner["token_site"]),
            output_dir=band_root / model,
            random_state=random_state,
        )

    print("[geometry CPU] report build start", flush=True)
    return build_report(
        non_thinking_export_root=non_thinking_root,
        native_running_root=native_running_root,
        native_final_root=native_final_root,
        dual_endpoint_root=analysis_root,
        parser_audit=parser_audit,
        band_root=band_root,
        grammar_registry=grammar_registry,
        grammar_filter_root=grammar_filter_root,
        pure_trace_n10_root=pure_trace_n10_root,
        indexed_numeric_n10_root=indexed_numeric_n10_root,
        output=output,
        manifest_path=manifest,
        domain_transfer_root=domain_transfer_root,
        covariance_root=covariance_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-thinking-root", type=Path, required=True)
    parser.add_argument("--native-running-root", type=Path, required=True)
    parser.add_argument("--native-final-root", type=Path, required=True)
    parser.add_argument("--native-trace-root", type=Path, required=True)
    parser.add_argument("--parser-audit", type=Path, required=True)
    parser.add_argument("--grammar-registry", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--band-root", type=Path, required=True)
    parser.add_argument("--grammar-filter-root", type=Path)
    parser.add_argument("--pure-trace-n10-root", type=Path)
    parser.add_argument("--indexed-numeric-n10-root", type=Path)
    parser.add_argument("--domain-transfer-root", type=Path)
    parser.add_argument("--covariance-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    result = run_pipeline(
        non_thinking_root=args.non_thinking_root.resolve(),
        native_running_root=args.native_running_root.resolve(),
        native_final_root=args.native_final_root.resolve(),
        native_trace_root=args.native_trace_root.resolve(),
        parser_audit=args.parser_audit.resolve(),
        grammar_registry=args.grammar_registry.resolve(),
        analysis_root=args.analysis_root.resolve(),
        band_root=args.band_root.resolve(),
        grammar_filter_root=(
            args.grammar_filter_root.resolve()
            if args.grammar_filter_root is not None
            else None
        ),
        pure_trace_n10_root=(
            args.pure_trace_n10_root.resolve()
            if args.pure_trace_n10_root is not None
            else None
        ),
        indexed_numeric_n10_root=(
            args.indexed_numeric_n10_root.resolve()
            if args.indexed_numeric_n10_root is not None
            else None
        ),
        domain_transfer_root=(
            args.domain_transfer_root.resolve()
            if args.domain_transfer_root is not None
            else None
        ),
        covariance_root=(
            args.covariance_root.resolve()
            if args.covariance_root is not None
            else None
        ),
        output=args.output.resolve(),
        manifest=args.manifest.resolve(),
        pca_dim=args.pca_dim,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
