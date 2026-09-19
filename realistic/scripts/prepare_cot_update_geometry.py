"""Freeze portable no-index update jobs at reproducibly selected layers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--noindex-selection-dir", type=Path,
                   help="Completed no-index scan directory containing results/, folds and saved-state verification")
    p.add_argument("--layer-eligibility", type=Path,
                   help="Architecture audit listing each model's K/V-writing layers; requires no-index selection")
    args = p.parse_args()
    start = time.monotonic()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    choices = ROOT.parent / "Figures/paper_ncc_unification_20260913/selection.json"
    selection = json.loads(choices.read_text(encoding="utf-8"))
    scan = args.noindex_selection_dir.resolve() if args.noindex_selection_dir else None
    eligibility = None
    if args.layer_eligibility:
        if not scan:
            raise ValueError("Architecture eligibility requires no-index discovery scores")
        eligibility = json.loads(args.layer_eligibility.read_text(encoding="utf-8"))
        assert eligibility["status"] == "VERIFIED_ARCHITECTURE"
    if scan:
        verified = json.loads((scan / "saved_state_replay.json").read_text(encoding="utf-8"))
        assert verified["status"] == "PASS" and sum(r["layers_replayed"] for r in verified["results"]) == 78
    inputs = {
        "Qwen3-8B": {
            "confirmation": "work/audit_n10_first_pass_noindex_20260827/confirmation_rows_first_pass_noindex_v5.jsonl",
            "geometry": "work/audit_n10_first_pass_noindex_20260827/selected_rows_first_pass_noindex_v5.jsonl",
            "manifest": "work/counting_mechanism_diagnostics_20260827/pca_n10/manifest.json",
            "mode": "natural_noindex", "bank": "qwen_shared_k128",
        },
        "Gemma4-E4B": {
            "confirmation": "work/cot_completion_20260912/inputs/Gemma4-E4B_prompted_cohort.jsonl",
            "geometry": "work/gemma_prompt_conditioned_noindex_20260827/cohort_full_20_10/frozen_cohort.jsonl",
            "manifest": "work/gemma_prompt_conditioned_noindex_20260827/cohort_full_20_10/manifest.json",
            "mode": "prompt_conditioned_noindex", "bank": "gemma_shared_k6",
        },
    }
    hashes = {}
    provenance = {}

    def copy(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative).replace("\\", "/")] = sha(target)

    for section in ("src", "scripts"):
        for f in sorted((ROOT / section).rglob("*.py")):
            if "__pycache__" not in f.parts:
                copy(f, Path("code") / f.relative_to(ROOT))
    for name in ("test_realistic_niah_v5_natural_aligned_progress.py",
                 "test_realistic_niah_v5_same_site_progress_transplant.py",
                 "test_v4_attention_backend_restore.py"):
        copy(ROOT / "tests" / name, Path("code/tests") / name)
    if scan:
        for name in ("fold_assignments.json", "saved_state_replay.json", "REPORT.md"):
            copy(scan / name, Path("noindex_selection") / name)
        if eligibility:
            copy(args.layer_eligibility, "layer_eligibility.json")
            copy(ROOT / "plans/cot-update-kv-eligible-20260915.md", "PROTOCOL.md")
        else:
            copy(ROOT / "plans/cot-update-noindex-ncc-20260915.md", "PROTOCOL.md")
    else:
        copy(choices, "geometry_selection.json")
        copy(ROOT / "plans/cot-update-geometry-20260914.md", "PROTOCOL.md")
    for model, info in inputs.items():
        source = (scan / "results" / model / "discovery_layer_metrics.csv") if scan else (
            ROOT / f"reports/v5_dual_endpoint_geometry_full300/{model}/pca16_whiten/running_index_candidate_metrics.csv")
        with source.open(encoding="utf-8-sig") as handle:
            candidates = list(csv.DictReader(handle))
        if not scan:
            candidates = [r for r in candidates if r["mode"] == "native_thinking" and r["analysis_group"] == "all_traces"]
        layer_key = "layer_zero_based" if scan else "layer"
        selection_key = lambda r: (
            round(float(r["discovery_oof_ncc_balanced_accuracy"]), 12),
            round(float(r["discovery_oof_logistic_balanced_accuracy"]), 12), -int(r[layer_key]))
        unconstrained = max(candidates, key=selection_key)
        layer = int(unconstrained[layer_key])
        if scan:
            selected_path = scan / "results" / model / "selection.json"
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            assert selected["status"] == "PASS" and layer == selected["selected"]["layer_zero_based"]
            assert len(candidates) == selected["layers_evaluated"]
            assert selected["protocol"]["confirmation_used"] is False
            assert sha(ROOT / info["geometry"]) == selected["protocol"]["input_sha256"]
            copy(selected_path, f"noindex_selection/{model}/selection.json")
            copy(scan / "results" / model / "capture_audit.json", f"noindex_selection/{model}/capture_audit.json")
        else:
            assert layer == selection["selected"][model]["running_index"]
        eligible_layers = sorted(int(r[layer_key]) + 1 for r in candidates)
        if eligibility:
            architecture = eligibility["models"][model]
            assert architecture["num_layers"] == len(candidates)
            writers = architecture["kv_writing_layers_one_based"]
            assert writers and all(1 <= v <= len(candidates) for v in writers)
            eligible_layers = [v for v in eligible_layers if any(w > v for w in writers)]
            assert eligible_layers == architecture["eligible_post_block_layers_one_based"]
            candidates = [r for r in candidates if int(r[layer_key]) + 1 in eligible_layers]
            if not candidates:
                raise ValueError(f"No cross-token K/V path remains for {model}")
        chosen = max(candidates, key=selection_key)
        layer = int(chosen[layer_key])
        split = json.loads((ROOT / info["manifest"]).read_text(encoding="utf-8"))
        confirmation = sorted(split["confirmation_seeds"])
        discovery = sorted(split["discovery_seeds"])
        assert len(discovery) == 20 and len(confirmation) == 10 and not set(discovery) & set(confirmation)
        if scan:
            assert discovery == sorted(selected["protocol"]["discovery_seeds"])
        cohort = read_rows(ROOT / info["confirmation"])
        geometry = read_rows(ROOT / info["geometry"])
        assert sorted(r["seed"] for r in cohort) == confirmation
        assert sorted(r["seed"] for r in geometry) == sorted(discovery + confirmation)
        full = {r["seed"]: r for r in geometry}
        for row in cohort:
            # Freeze identical prompts and continuations despite older audit metadata.
            for key in ("input_ids", "output_token_ids", "request_id"):
                assert row[key] == full[row["seed"]][key], (model, row["seed"], key)
        for kind in ("confirmation", "geometry"):
            copy(ROOT / info[kind], f"inputs/{model}_{kind}.jsonl")
        copy(source, f"inputs/{model}_layer_discovery.csv")
        for role in ("targeted_selection", "causal_routes"):
            name = f"realistic_niah_v5_{info['bank']}_{role}_frozen.json"
            copy(ROOT / "configs" / name, Path("code/configs") / name)
        provenance[model] = dict(layer_zero_based=layer, layer_one_based=layer + 1,
                                 cohort_mode=info["mode"], bank=info["bank"],
                                 discovery_seeds=discovery, confirmation_seeds=confirmation,
                                 eligible_layers_one_based=eligible_layers,
                                 selected_discovery_ncc=float(chosen["discovery_oof_ncc_balanced_accuracy"]),
                                 unconstrained_ncc_layer_one_based=int(unconstrained[layer_key]) + 1,
                                 original_inputs={k: info[k] for k in ("confirmation", "geometry")})
    manifest = dict(schema="cot_update_noindex_ncc_20260915_v1" if scan else "cot_update_geometry_20260914_v1", models=provenance,
                    layer_selection_basis="No-index discovery NCC; selected and frozen before these causal jobs" if scan else "Full-cohort discovery running-index geometry",
                    fixed_layers=True, scopes=["item_span", "endpoint", "four_token_tail"],
                    donor_k=[4, 6, 8], directions=["forward", "backward"],
                    max_new_tokens=96, backend="sdpa", dtype="bfloat16", chunk_size=512,
                    expected_directed_cells=360, expected_condition_rows=1080,
                    input_code_sha256=hashes, preparation_seconds=time.monotonic() - start)
    if eligibility:
        manifest.update(schema="cot_update_kv_eligible_20260915_v1",
                        layer_selection_basis="No-index discovery NCC among post-block layers with a later K/V writer",
                        selection_revision="Architecture restriction added after Gemma L32 diagnostic; frozen before L23 outcomes",
                        architecture_audit_sha256=sha(args.layer_eligibility))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(out)).replace("\\", "/"))
    print(json.dumps(dict(bundle=str(out), archive=str(archive), files=len(hashes),
                         archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)))


if __name__ == "__main__":
    main()
