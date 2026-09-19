from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "realistic_niah_v4_4_report_reviewer_manifest_v1"


# This is the review surface for the integrated report.  It intentionally
# excludes unrelated experiments that happen to live beside these files.
FILES: tuple[tuple[str, str, str], ...] = (
    # Final report and its two-stage rendering chain.
    ("report", "final HTML", "reports/realistic_niah_v4_4_non_thinking_integrated_mechanism_report.html"),
    ("report_builder", "integrated report renderer", "scripts/build_realistic_niah_v4_4_integrated_report.py"),
    ("report_builder", "V4.4 representation base renderer", "scripts/build_realistic_niah_v4_4_report.py"),
    ("report_builder", "shared V4 representation analysis/renderer", "scripts/build_realistic_niah_v4_representation_report.py"),
    ("report_input", "self-contained representation/attention base HTML", "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html"),
    # Authoritative configurations.
    ("config", "base V4 stimulus/model/representation design", "configs/realistic_niah_v4.json"),
    ("config", "causal-v2 patching/steering design", "configs/realistic_niah_v4_causal_v2.json"),
    ("config", "full-span top-k fresh-seed design", "configs/realistic_niah_v4_4_full_span_topk_confirmation.json"),
    ("config", "frozen full-span rankings and K grid", "configs/realistic_niah_v4_4_full_span_topk_confirmation_selection.json"),
    ("config", "correct-only patching/ablation supplement", "configs/realistic_niah_v4_4_correct_interventions.json"),
    ("config", "Qwen natural pre-O OV design", "configs/realistic_niah_v4_4_4.json"),
    ("config", "Qwen alpha/value read-write design", "configs/realistic_niah_v4_4_4_readwrite.json"),
    ("config", "Qwen full-span early-to-L28 K sweep", "configs/realistic_niah_v4_4_4_qwen_full_span_upstream_k_sweep.json"),
    ("config", "Qwen fresh early-to-L28 confirmation", "configs/realistic_niah_v4_4_4_upstream_confirmation.json"),
    ("config", "Gemma full-span K2 residual path", "configs/realistic_niah_v4_4_4_gemma_full_span_residual_k2.json"),
    # Machine-readable report inputs actually consumed by build_report_clear.
    ("report_input", "macro patching/steering summary", "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json"),
    ("report_input", "full-span top-k audited summary", "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/seed_extrapolation_summary_v2.json"),
    ("report_input", "full-span top-k primary statistics", "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/full_span_topk_primary_statistics.csv"),
    ("report_input", "Qwen natural OV analysis", "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json"),
    ("report_input", "Qwen alpha/value read-write analysis", "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json"),
    ("report_input", "Qwen full-span upstream analysis", "reports/v4_non-thinking_causal/v4_4_4/qwen/full_span_upstream/realistic_niah_v4_4_4_upstream_path_analysis.json"),
    ("report_input", "Qwen independent upstream confirmation", "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json"),
    ("report_input", "Gemma K2 residual analysis", "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json"),
    ("report_input", "fresh correct-only route analysis", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json"),
    ("report_input", "fresh correct-only effect rows", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/effects.csv.gz"),
    ("report_input", "fresh correct-only route table", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/route_summary.csv"),
    ("report_input", "fresh correct-only geometry table", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/geometry_summary.csv"),
    ("provenance", "causal-v2 source ledger with remote paths and hashes", "reports/v4_non-thinking_causal/v4_4_causal_v2/source_ledger.csv"),
    ("provenance", "Qwen V4.4.4 FileStream manifest", "reports/v4_non-thinking_causal/v4_4_4/filestream_manifest.json"),
    # Experiment entry points and audits.
    ("entrypoint", "base V4 runner", "scripts/run_realistic_niah_v4.py"),
    ("entrypoint", "freeze causal-v2 stimuli", "scripts/freeze_realistic_niah_v4_causal_v2.py"),
    ("entrypoint", "causal-v2 runner", "scripts/run_realistic_niah_v4_causal_v2.py"),
    ("entrypoint", "causal-v2 analysis", "scripts/analyze_realistic_niah_v4_causal_v2.py"),
    ("entrypoint", "causal-v2 audit", "scripts/audit_realistic_niah_v4_causal_v2.py"),
    ("entrypoint", "correct-only parallel launcher", "scripts/launch_realistic_niah_v4_4_correct_interventions_4x4.sh"),
    ("entrypoint", "correct-only worker/merge runner", "scripts/run_realistic_niah_v4_4_correct_interventions_parallel.py"),
    ("entrypoint", "correct-only audit", "scripts/audit_realistic_niah_v4_4_correct_interventions.py"),
    ("entrypoint", "full-span top-k launcher", "scripts/launch_realistic_niah_v4_4_full_span_topk_confirmation.sh"),
    ("entrypoint", "full-span top-k audit", "scripts/audit_realistic_niah_v4_4_ablation_seed_extrapolation.py"),
    ("entrypoint", "full-span top-k statistics", "scripts/analyze_realistic_niah_v4_4_full_span_topk.py"),
    ("entrypoint", "Qwen natural OV runner", "scripts/run_realistic_niah_v4_4_4.py"),
    ("entrypoint", "Qwen read-write runner", "scripts/run_realistic_niah_v4_4_4_readwrite.py"),
    ("entrypoint", "Qwen upstream K-sweep runner", "scripts/run_realistic_niah_v4_4_4_upstream_path.py"),
    ("entrypoint", "Qwen independent upstream runner", "scripts/run_realistic_niah_v4_4_4_upstream_confirmation.py"),
    ("entrypoint", "Gemma residual-path runner", "scripts/run_realistic_niah_v4_4_4_gemma_full_span_residual.py"),
    # Core V4 implementation used by the entry points.
    ("core", "model and revision registry", "src/realistic_niah_v4/spec.py"),
    ("core", "controlled stimuli", "src/realistic_niah_v4/stimuli.py"),
    ("core", "prompt rendering and token alignment", "src/realistic_niah_v4/prompts.py"),
    ("core", "model adapters and hooks", "src/realistic_niah_v4/modeling.py"),
    ("core", "base V4 pipeline", "src/realistic_niah_v4/pipeline.py"),
    ("core", "hidden-state geometry", "src/realistic_niah_v4/representation.py"),
    ("core", "attention extraction", "src/realistic_niah_v4/attention.py"),
    ("core", "attention/outcome analysis", "src/realistic_niah_v4/attention_outcomes.py"),
    ("core", "causal-v2 interventions", "src/realistic_niah_v4/causal_v2.py"),
    ("core", "causal generation and scoring", "src/realistic_niah_v4/causal_generation.py"),
    ("core", "centroid steering", "src/realistic_niah_v4/geometric_steering.py"),
    ("core", "causal-v2 statistics", "src/realistic_niah_v4/causal_v2_analysis.py"),
    ("core", "correct-only selection and aggregation", "src/realistic_niah_v4/correct_interventions.py"),
    ("core", "correct-only row filters", "src/realistic_niah_v4/correct_only_slices.py"),
    # Shared pre-O and set-space implementation inherited by V4.4.4.
    ("core", "atomic IO and stage layout", "src/realistic_niah_v4_4_3/io.py"),
    ("core", "head/value geometry", "src/realistic_niah_v4_4_3/geometry.py"),
    ("core", "single-head pre-O hooks", "src/realistic_niah_v4_4_3/interventions.py"),
    ("core", "set-level geometry", "src/realistic_niah_v4_4_3/set_geometry.py"),
    ("core", "set-level pre-O hooks", "src/realistic_niah_v4_4_3/set_interventions.py"),
    ("core", "shared V4.4.3 pipeline", "src/realistic_niah_v4_4_3/pipeline.py"),
    ("core", "shared V4.4.3 specification", "src/realistic_niah_v4_4_3/spec.py"),
    # Qwen/Gemma V4.4.4 mechanism implementation.
    ("core", "V4.4.4 natural-OV specification", "src/realistic_niah_v4_4_4/spec.py"),
    ("core", "V4.4.4 dataset/model pipeline", "src/realistic_niah_v4_4_4/pipeline.py"),
    ("core", "natural pre-O steering/removal", "src/realistic_niah_v4_4_4/interventions.py"),
    ("core", "natural-OV statistics and audit", "src/realistic_niah_v4_4_4/analysis.py"),
    ("core", "alpha/value factorization", "src/realistic_niah_v4_4_4/readwrite.py"),
    ("core", "read-write pipeline", "src/realistic_niah_v4_4_4/readwrite_pipeline.py"),
    ("core", "read-write statistics", "src/realistic_niah_v4_4_4/readwrite_analysis.py"),
    ("core", "relay utilities used by read-write", "src/realistic_niah_v4_4_4/relay.py"),
    ("core", "relay data loader used by read-write", "src/realistic_niah_v4_4_4/relay_pipeline.py"),
    ("core", "early-to-late path intervention", "src/realistic_niah_v4_4_4/upstream_path.py"),
    ("core", "early-to-late path pipeline", "src/realistic_niah_v4_4_4/upstream_path_pipeline.py"),
    ("core", "early-to-late path statistics", "src/realistic_niah_v4_4_4/upstream_path_analysis.py"),
    ("core", "independent upstream specification", "src/realistic_niah_v4_4_4/upstream_confirmation_spec.py"),
    ("core", "independent upstream statistics", "src/realistic_niah_v4_4_4/upstream_confirmation_analysis.py"),
    ("core", "Gemma K-sweep specification", "src/realistic_niah_v4_4_4/gemma_full_span_residual_spec.py"),
    ("core", "Gemma source-set operations", "src/realistic_niah_v4_4_4/gemma_cross_layer.py"),
    ("core", "Gemma residual interventions", "src/realistic_niah_v4_4_4/gemma_residual.py"),
    ("core", "Gemma residual pipeline", "src/realistic_niah_v4_4_4/gemma_residual_pipeline.py"),
    ("core", "Gemma residual statistics", "src/realistic_niah_v4_4_4/gemma_residual_analysis.py"),
    ("core", "tokenizer adapter used by V4.4.4", "src/dataset_generation/dynamic_niah.py"),
    # Environments and this checker.
    ("environment", "GPU/runtime pins", "requirements-mechanistic-v4.txt"),
    ("environment", "CPU analysis pins", "requirements-analysis.txt"),
    ("review", "bundle hash/audit checker", "scripts/audit_realistic_niah_v4_4_report_bundle.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(document: Any, *keys: str) -> Any:
    value = document
    for key in keys:
        value = value[key]
    return value


def load_json(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def scientific_audit(root: Path) -> list[dict[str, Any]]:
    checks: list[tuple[str, str, tuple[str, ...], Any]] = [
        ("causal-v2 Qwen", "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json", ("audits", "Qwen3-8B", "status"), "PASS"),
        ("causal-v2 Gemma", "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json", ("audits", "Gemma4-E4B", "status"), "PASS"),
        ("full-span top-k", "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/seed_extrapolation_summary_v2.json", ("audit", "status"), "passed"),
        ("Qwen natural OV", "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json", ("audit", "all_checks_pass"), True),
        ("Qwen read-write", "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json", ("audit", "all_checks_pass"), True),
        ("Qwen full-span upstream", "reports/v4_non-thinking_causal/v4_4_4/qwen/full_span_upstream/realistic_niah_v4_4_4_upstream_path_analysis.json", ("audit", "all_checks_pass"), True),
        ("Qwen independent upstream", "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json", ("audit", "all_checks_pass"), True),
        ("Gemma K2 residual", "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json", ("audit", "all_checks_pass"), True),
        ("fresh correct-only routes", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json", ("audits", "all_checks_pass"), True),
    ]
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for name, relative, keys, expected in checks:
        document = cache.setdefault(relative, load_json(root, relative))
        observed = nested(document, *keys)
        rows.append(
            {
                "name": name,
                "path": relative,
                "field": ".".join(keys),
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    return rows


def report_structure_audit(root: Path) -> dict[str, Any]:
    relative = "reports/realistic_niah_v4_4_non_thinking_integrated_mechanism_report.html"
    text = (root / relative).read_text(encoding="utf-8")
    section_ids = (
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "synthesis",
        "limits",
    )
    checks = {
        "ten_unique_required_sections": all(text.count(f'id="{name}"') == 1 for name in section_ids),
        "balanced_sections": text.count("<section") == text.count("</section>"),
        "balanced_figcaptions": text.count("<figcaption") == text.count("</figcaption>"),
        "source_ledger_present": "Source ledger" in text,
        "no_unicode_replacement_character": "\ufffd" not in text,
    }
    return {"path": relative, "checks": checks, "passed": all(checks.values())}


def build_manifest(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for category, role, relative in FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        entries.append(
            {
                "category": category,
                "role": role,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing reviewer-bundle files: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_report": "reports/realistic_niah_v4_4_non_thinking_integrated_mechanism_report.html",
        "scope_note": "Hashes cover the code, configs, local summaries, provenance and environment required to inspect this report; raw tensors remain on FileStream.",
        "files": entries,
        "scientific_audits": scientific_audit(root),
        "report_structure": report_structure_audit(root),
        "known_gap": {
            "component": "fresh correct-only route campaign",
            "run_root": "runs/v4_4_4_correct_state_routes/run_20260805_dual_model_lowcount_correct",
            "status": "local aggregate artifacts are present, but a dedicated runner/analysis source file was not found in the local repository on 2026-08-06; the stopped SSH host could not be queried",
        },
    }


def verify(root: Path, manifest_path: Path) -> tuple[bool, dict[str, Any]]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in expected["files"]:
        path = root / item["path"]
        exists = path.is_file()
        observed_hash = sha256(path) if exists else None
        observed_bytes = path.stat().st_size if exists else None
        passed = (
            exists
            and observed_hash == item["sha256"]
            and observed_bytes == item["bytes"]
        )
        rows.append(
            {
                "path": item["path"],
                "exists": exists,
                "hash_matches": observed_hash == item["sha256"],
                "bytes_match": observed_bytes == item["bytes"],
                "passed": passed,
            }
        )
    science = scientific_audit(root)
    structure = report_structure_audit(root)
    passed = all(row["passed"] for row in rows + science) and structure["passed"]
    return passed, {
        "schema_version": "realistic_niah_v4_4_report_bundle_verification_v1",
        "passed": passed,
        "file_checks": rows,
        "scientific_audits": science,
        "report_structure": structure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or verify the V4.4 report reviewer manifest")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-manifest", type=Path)
    mode.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.write_manifest:
        output = args.write_manifest.resolve()
        document = build_manifest(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "files": len(document["files"])}, indent=2))
        return 0
    passed, document = verify(root, args.manifest.resolve())
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
