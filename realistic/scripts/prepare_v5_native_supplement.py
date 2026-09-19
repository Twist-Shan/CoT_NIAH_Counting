#!/usr/bin/env python3
"""Freeze a portable, outcome-independent Native supplement bundle (CPU only)."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v4.spec import resolve_model_spec
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.native_supplement import (
    ANSWER_SEEDS, LAYERS, MODELS, OLD_LAYERS, PROGRESS_SEEDS, SCHEMA,
    canonical_digest, digest, read_rows, validate_answer, validate_pairs,
    validate_progress, verify_bundle, write_json, write_rows,
)


def script_closure(entries: list[str]) -> list[Path]:
    """Include repository-local script imports, without unrelated run outputs."""
    pending = [ROOT / "scripts" / name for name in entries]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + a.name for a in node.names]
            for name in names:
                parts = name.split(".")
                candidate = (ROOT / Path(*parts)).with_suffix(".py") if parts[0] == "scripts" else ROOT / "scripts" / (parts[0] + ".py")
                if candidate.is_file() and candidate not in found:
                    pending.append(candidate)
    return sorted(found)


def prepare(output: Path) -> dict:
    started = time.monotonic()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Use a new bundle directory; refusing to replace {output}")
    aligned = ROOT / "work/v5_native_sample_aligned_20260829"
    old_alignment = json.loads((aligned / "answer_query_layer_sweep_plan/alignment_manifest.json").read_text())
    input_audit = json.loads((aligned / "alignment_audit.json").read_text())
    output_audit = json.loads((aligned / "output_alignment_audit.json").read_text())
    if input_audit["status"] != "PASS" or output_audit["status"] != "PASS":
        raise ValueError("Historical sample alignment did not pass")

    # Validate ALL historical sources before creating the portable bundle.
    models = {}
    payloads = {}
    previous_keys = None
    for model in MODELS:
        pair_path = aligned / f"answer_query_layer_sweep_plan/{model}_pairs.jsonl"
        old_root = aligned / f"runs/{model}/answer_query_layer_sweep"
        pairs = read_rows(pair_path)
        keys = validate_pairs(pairs, model)
        if previous_keys is not None and keys != previous_keys:
            raise ValueError("Cross-model directed sample keys differ")
        previous_keys = keys
        old_rows = read_rows(old_root / "trials.jsonl")
        old_audit = json.loads((old_root / "analysis/audit.json").read_text())
        for p, field in [(pair_path, "pairs_sha256"), (old_root / "trials.jsonl", "trials_sha256")]:
            if digest(p) != old_audit[field]:
                raise ValueError(f"Historical checksum mismatch: {p}")
        registry_field = "qwen_pairs_sha256" if model == "Qwen3-8B" else "gemma_pairs_sha256"
        if canonical_digest(pairs) != old_alignment[registry_field]:
            raise ValueError("Frozen canonical pair registry changed")
        validate_answer(old_rows, pairs, OLD_LAYERS[model])
        generations = ROOT / f"work/v5_trace_parser_v2/{model}_generations_reparsed.jsonl"
        old_plan = ROOT / f"reports/v5_native_answer_query_layer_sweep_20260828/{model}/plan/{model}__answer_query_layer_plan_audit.json"
        if digest(generations) != json.loads(old_plan.read_text())["generations_sha256"]:
            raise ValueError("Archived source generations changed")
        all_rows = read_rows(generations)
        wanted = {p[f] for p in pairs for f in ("receiver_request_id", "donor_request_id")}
        selected = [r for r in all_rows if r["request_id"] in wanted]
        if len(selected) != len(wanted) or len({r["request_id"] for r in selected}) != len(wanted):
            raise ValueError("Missing or duplicate archived generation")
        by_id = {r["request_id"]: r for r in selected}
        for pair in pairs:
            for role in ("receiver", "donor"):
                row = by_id[pair[f"{role}_request_id"]]
                parsed = parse_trace_record(row)
                if not parsed["parser"]["trace_one_to_one"] or not parsed["exact_count"]:
                    raise ValueError("Archived eligibility no longer passes the parser")
                if int(parsed["gold_count"]) != pair[f"{role}_count"] or int(row["seed"]) != pair["seed"]:
                    raise ValueError("Archived generation key differs from pair")
        missing_layers = sorted(set(range(LAYERS[model])) - set(OLD_LAYERS[model]))
        models[model] = {
            "model_spec": asdict(resolve_model_spec(model)),
            "old_layers": OLD_LAYERS[model], "missing_layers": missing_layers,
            "all_layers": list(range(LAYERS[model])), "pairs": 40, "seed_clusters": 10,
            "replay_trials": 640, "new_layer_trials": 80 * len(missing_layers),
            "dense_trials": 80 * LAYERS[model], "source_generations_sha256": digest(generations),
            "source_generations": str(generations.relative_to(ROOT).as_posix()),
            "packed_generation_rows": len(selected),
        }
        payloads[model] = (pair_path, old_root / "trials.jsonl", old_root / "analysis/audit.json", selected)

    progress_root = ROOT / "work/gemma_prompt_conditioned_noindex_20260827"
    cohort_path = progress_root / "cohort_full_20_10/frozen_cohort.jsonl"
    cohort = [r for r in read_rows(cohort_path) if int(r["seed"]) in PROGRESS_SEEDS]
    if sorted(int(r["seed"]) for r in cohort) != PROGRESS_SEEDS:
        raise ValueError("Gemma existing confirmation cohort changed")
    forward_path = progress_root / "forward_l16_item_span/confirmation/forward_k6/trials.jsonl"
    validate_progress(read_rows(forward_path), [6], direction="forward")

    output.mkdir(parents=True)
    for model, (pair_path, trials_path, audit_path, selected) in payloads.items():
        destination = output / "inputs" / model
        destination.mkdir(parents=True)
        shutil.copy2(pair_path, destination / "pairs.jsonl")
        shutil.copy2(trials_path, destination / "historical_answer_trials.jsonl")
        shutil.copy2(audit_path, destination / "historical_answer_audit.json")
        write_rows(destination / "generations.jsonl", selected)
    write_rows(output / "inputs/Gemma4-E4B/prompted_cohort.jsonl", cohort)
    shutil.copy2(forward_path, output / "inputs/Gemma4-E4B/historical_forward_k6.jsonl")
    shutil.copy2(cohort_path.parent / "manifest.json", output / "inputs/Gemma4-E4B/historical_cohort_manifest.json")
    shutil.copy2(aligned / "answer_query_layer_sweep_plan/alignment_manifest.json", output / "inputs/historical_pair_manifest.json")

    code = output / "code"
    for source in sorted((ROOT / "src").rglob("*.py")):
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    entries = [
        "run_v5_native_supplement.py", "run_realistic_niah_v5.py",
        "analyze_v5_answer_query_layer_sweep.py", "run_realistic_niah_v5_natural_aligned_progress_transplant.py",
    ]
    for source in script_closure(entries):
        target = code / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ["realistic_niah_v5.json", "realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json", "realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"]:
        target = code / "configs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "configs" / name, target)
    shutil.copy2(ROOT / "requirements.txt", code / "requirements.txt")
    shutil.copy2(ROOT / "plans/native-thinking-supplement-20260911.md", output / "RUNBOOK.md")
    core = {
        "schema_version": SCHEMA, "status": "PREPARED_NOT_GPU_RUN", "models": models,
        "answer_seeds": ANSWER_SEEDS,
        "answer_estimand": "seed-equal exact Target-count adoption after single post-block answer_query_v3 full-state patch",
        "answer_max_new_tokens": 16, "attention_backend": "sdpa", "torch_dtype": "bfloat16",
        "dense_policy": "replay all old layers, check exact behavior/positions, then measure missing layers; combine NEW run rows only",
        "cohort_is_fresh_confirmation": False,
        "gemma_backward": {
            "seeds": PROGRESS_SEEDS, "target_k": [4, 6, 8], "receiver_k": [5, 7, 9],
            "layer": 16, "max_new_tokens": 96, "pairs": 30, "condition_rows": 90,
            "forward_k6_replay_rows": 30, "source_cohort_sha256": digest(cohort_path),
            "claim_scope": "prompt-conditioned no-index capability; same-cohort direction extension",
        },
        "source_alignment_audit_sha256": digest(aligned / "alignment_audit.json"),
        "source_output_alignment_audit_sha256": digest(aligned / "output_alignment_audit.json"),
        "files_sha256": {str(p.relative_to(output).as_posix()): digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    plan = {**core, "plan_sha256": canonical_digest(core)}
    write_json(output / "plan.json", plan)
    verify_bundle(output)
    write_json(output.parent / f"{output.name}_prepare_audit.json", {
        "status": "PASS", "gpu_executed": False, "elapsed_seconds": time.monotonic() - started,
        "plan_sha256": plan["plan_sha256"], "answer_new_trials": 4960, "answer_replay_trials": 1280,
        "answer_total_trials": 6240, "gemma_backward_new_rows": 90, "gemma_forward_replay_rows": 30,
        "file_count": len(core["files_sha256"]),
    })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.output)
    if args.archive:
        path = args.output.with_suffix(".tar.gz")
        if path.exists():
            raise ValueError(f"Archive already exists: {path}")
        with tarfile.open(path, "w:gz") as archive:
            archive.add(args.output, arcname=args.output.name)
        print(json.dumps({"archive": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}))
    print(json.dumps({"status": plan["status"], "bundle": str(args.output.resolve()), "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    main()
