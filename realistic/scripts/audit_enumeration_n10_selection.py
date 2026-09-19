"""Recompute N10 discovery probes and check frozen selection without GPU access."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.read_audit import read, require, sha, json_sha, jsonl
from analyze_enumeration_fresh_relay import verify_inventory
from analyze_enumeration_fresh_answer_patch import load_module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--contracts", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    inv = verify_inventory(a.source)
    contract_inv = verify_inventory(a.contracts)
    stage = a.source / "fresh_n10_update_v1"
    frozen_code = a.contracts / "fresh_n10_update_v1/code"
    manifest, selection = read(stage / "manifest.json"), read(stage / "selection_manifest.json")
    require(sha(stage / "manifest.json") == inv["n10_manifest_sha256"], "N10 manifest mismatch")
    require(sha(stage / "selection_manifest.json") == inv["selection_manifest_sha256"], "Selection manifest mismatch")
    require(not selection["confirmation_used_for_selection"] and selection["confirmation_is_reused"], "Selection chronology mismatch")
    contracts_path = frozen_code / "src/realistic_niah_v6/update_n10.py"
    require(sha(contracts_path) == manifest["code_sha256"]["src/realistic_niah_v6/update_n10.py"], "Frozen selection helper mismatch")
    contracts = load_module(contracts_path, "_n10_contracts_frozen")
    native_root = a.inputs / "fresh_causal_v1/code/src"
    sys.path.insert(0, str(native_root.resolve()))
    from realistic_niah_v5 import trace_stratified_geometry as native
    require(Path(native.__file__).resolve() == (native_root / "realistic_niah_v5/trace_stratified_geometry.py").resolve(), "Wrong Native implementation imported")
    require(sha(native.__file__) == manifest["code_sha256"]["src/realistic_niah_v5/trace_stratified_geometry.py"], "Native probe source mismatch")
    cells, metric_rows = [], []
    with threadpool_limits(limits=1):
        for cell in selection["cells"]:
            tick = time.monotonic()
            model, mode = cell["model"], cell["mode"]
            folder = stage / "cells" / model / mode
            cohort, chosen = read(folder / "cohort.json"), read(folder / "selection.json")
            require(sha(folder / "cohort.json") == cell["cohort_sha256"] and sha(folder / "selection.json") == cell["selection_sha256"], "Frozen cell selection mismatch")
            require(not cohort["selection_used_final_correctness"] and not cohort["intervention_outcomes_accessed"], "Outcome selection gate")
            for name, key in [("candidate_ledger.json", "ledger_sha256"), ("selected_generations.jsonl", "inputs_sha256"), ("geometry.json", "geometry_sha256")]:
                require(sha(folder / name) == cohort[key], "Cohort evidence mismatch")
            candidates = manifest["initial_discovery_seeds"] + (manifest["reserve_seeds"] if model in manifest["reserve_models"] else [])
            ledger = read(folder / "candidate_ledger.json")
            expected = contracts.select_discovery(ledger, candidates, manifest["all_original_confirmation_candidates"])
            require(expected == cohort["discovery_seeds"] == cell["discovery_seeds"], "Discovery first-eligible rule mismatch")
            original = next(c for c in manifest["cells"] if c["model"] == model and c["mode"] == mode)
            require(original["confirmation_seeds"] == cohort["confirmation_seeds"], "Confirmation seeds changed")
            layer = contracts.validate_selection(selection, model, mode, cohort["confirmation_seeds"])
            geometry = read(folder / "geometry.json")
            generated = {r["seed"]: r for r in jsonl(folder / "selected_generations.jsonl")}
            arrays, metadata = {}, {}
            for split, seeds, nstates in [("discovery", cohort["discovery_seeds"], 200), ("confirmation", cohort["confirmation_seeds"], 100)]:
                audit = read(folder / f"{split}_capture_audit.json")
                require(audit["status"] == "PASS" and audit["first_trace_repeat_exact"], "Capture technical audit failed")
                require(sha(folder / f"{split}_states.npz") == audit["states_sha256"] and sha(folder / f"{split}_metadata.csv") == audit["metadata_sha256"], "Capture evidence changed")
                with np.load(folder / f"{split}_states.npz") as archive:
                    states, layers = archive["states"], archive["layer_indices"]
                expected_layers = np.arange(36 if model == "Qwen3-8B" else 42) if split == "discovery" else np.asarray([layer])
                require(np.array_equal(layers, expected_layers) and states.shape[:2] == (nstates, len(expected_layers)) and np.isfinite(states).all(), "Captured layer/state coverage mismatch")
                frame = pd.read_csv(folder / f"{split}_metadata.csv")
                require(len(frame) == nstates and set(frame["seed"]) == set(seeds) and set(frame["split"]) == {split} and set(frame["gold_count"]) == {10}, "Captured metadata population mismatch")
                for seed, group in frame.groupby("seed", sort=False):
                    require(group["occurrence"].tolist() == list(range(1, 11)), "Item labels missing or repeated")
                    require(group["position"].tolist() == geometry[str(seed)]["positions"] and group["token_id"].tolist() == geometry[str(seed)]["token_ids"], "Endpoint metadata/geometry mismatch")
                    require(set(group["request_id"]) == {generated[seed]["request_id"]}, "Metadata request mismatch")
                arrays[split], metadata[split] = states, frame
            frame, states = metadata["discovery"], arrays["discovery"]
            folds = read(folder / "discovery_folds.json")
            actual_folds = []
            for train, test in GroupKFold(5).split(states[:, 0], frame["occurrence"], groups=frame["seed"]):
                actual_folds.append({"train_rows": train.tolist(), "test_rows": test.tolist(),
                    "train_seeds": sorted(set(map(int, frame.iloc[train]["seed"]))), "test_seeds": sorted(set(map(int, frame.iloc[test]["seed"])))})
            require(folds == actual_folds and sha(folder / "discovery_folds.json") == chosen["folds_sha256"], "Seed folds changed")
            require(sha(folder / "discovery_layer_metrics.csv") == chosen["metrics_sha256"], "Discovery metrics changed")
            saved = pd.read_csv(folder / "discovery_layer_metrics.csv").to_dict("records")
            recomputed = []
            for index, original_metric in enumerate(saved):
                value = native.grouped_discovery_cv_metrics(states[:, index], frame, np.arange(1, 11), pca_dim=16, random_state=0, folds=5, pca_whiten=True)
                require(all(abs(value[k] - original_metric[k]) < 1e-12 for k in value), f"Probe recomputation differs: {model}/{mode} L{index+1}")
                recomputed.append({"layer_zero_based": index, "layer_one_based": index + 1, **value})
                metric_rows.append({"model": model, "mode": mode, **recomputed[-1]})
            winner = contracts.rank_layers(recomputed, model)[0]
            require(winner["layer_one_based"] == cell["layer_one_based"] == chosen["selected"]["layer_one_based"], "NCC/tie-break winner differs")
            confirm = read(folder / "confirmation_readout.json")
            logistic, ncc, _, components = native._fit_projection_and_predict(states[:, layer], frame["occurrence"].to_numpy(dtype=int),
                arrays["confirmation"][:, 0], np.arange(1, 11), pca_dim=16, random_state=0, pca_whiten=True)
            require(components == 16 and confirm["selection_manifest_sha256"] == sha(stage / "selection_manifest.json"), "Confirmation projection/selection mismatch")
            predictions = confirm["predictions"]
            require(np.array_equal(logistic, [r["logistic"] for r in predictions]) and np.array_equal(ncc, [r["ncc"] for r in predictions]), "Fixed-layer confirmation predictions differ")
            truth = metadata["confirmation"]["occurrence"].to_numpy()
            require(abs(float((ncc == truth).mean()) - confirm["ncc_balanced_accuracy"]) < 1e-12 and abs(float((logistic == truth).mean()) - confirm["logistic_balanced_accuracy"]) < 1e-12, "Confirmation accuracy mismatch")
            result = {"model": model, "mode": mode, "audit": "PASS", "selected_layer_one_based": layer + 1,
                "discovery_ncc": winner["discovery_oof_ncc_balanced_accuracy"], "confirmation_ncc": confirm["ncc_balanced_accuracy"],
                "discovery_logistic": winner["discovery_oof_logistic_balanced_accuracy"], "confirmation_logistic": confirm["logistic_balanced_accuracy"],
                "layers_recomputed": len(recomputed), "confirmation_predictions_recomputed": 100,
                "discovery_seeds": cohort["discovery_seeds"], "confirmation_seeds": cohort["confirmation_seeds"], "seconds": time.monotonic() - tick}
            cells.append(result)
            print(json.dumps({k: v for k, v in result.items() if not k.endswith("seeds")}), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "enumeration_n10_selection_audit_v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells, "source_files_verified": len(inv["files"]), "contract_files_verified": len(contract_inv["files"]),
        "selection_manifest_sha256": sha(stage / "selection_manifest.json"), "native_probe_sha256": sha(native.__file__),
        "script_sha256": sha(__file__), "runtime": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "confirmation_is_reused": True, "limits": ["CPU replay verifies saved representations and all probe fits; it does not repeat GPU activation capture.",
            "The N10 amendment follows observation of the old mixed-N intervention results; original confirmation inputs are reused."],
        "command": sys.argv, "seconds": time.monotonic() - start}
    (a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(a.output / "recomputed_discovery_metrics.csv", index=False)


if __name__ == "__main__":
    main()
