"""Verify the complete frozen rerun, execution settings and saved readout data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analyze_cot_update_geometry import interval, read_rows


def options(arguments):
    result, key = {}, None
    for value in arguments:
        if value.startswith("--"):
            key = value
            result[key] = []
        else:
            assert key is not None
            result[key].append(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    base = args.bundle
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = read(base / "manifest.json")
    status = read(base / "status_all.json")
    assert status["state"] == "COMPLETE" and status["jobs_complete"] == status["jobs_total"] == 38
    analysis = read(args.analysis / "summary.json")
    assert analysis["status"] == "COMPLETE_PAIRED_AUDIT" and analysis["condition_rows"] == 1080
    assert not analysis["primary_only"] and len(analysis["models"]) == 2
    cpu_validation = read(base / "cpu_validation.json")
    assert cpu_validation["exit_code"] == 0
    smoke_validation = read(base / "smoke_validation.json")
    assert len(smoke_validation) == 2 and all(r["status"] == "PASS" for r in smoke_validation)
    for relative, expected in frozen["input_code_sha256"].items():
        assert sha(base / relative) == expected, relative
    common_packages, loaded_models, jobs = None, {}, {}
    for model in frozen["models"]:
        for scope in frozen["scopes"]:
            for direction in frozen["directions"]:
                for k in frozen["donor_k"]:
                    folder = base / "runs" / model / scope / f"{direction}_k{k}"
                    command = read(folder / "command.json")
                    runtime = read(folder / "runtime.json")
                    assert runtime["job_sha256"] == sha(folder / "command.json")
                    if common_packages is None:
                        common_packages = runtime["packages"]
                    assert runtime["packages"] == common_packages
                    event, = [r for r in read_rows(folder / "backend_events.jsonl") if r["event"] == "model_loaded"]
                    assert event["training"] is False and event["dtype"] == "torch.bfloat16"
                    assert set(event["backend"].values()) == {"sdpa"}
                    identity = {key: event[key] for key in ("revision", "model_class", "model_source_sha256", "layers")}
                    if model in loaded_models:
                        assert identity == loaded_models[model]
                    loaded_models[model] = identity
                    jobs[(model, scope, direction, k)] = options(command["args"])
    allowed = {"--model", "--generations", "--cohort-mode", "--layers", "--targeted-selection",
               "--targeted-routing", "--seeds", "--output"}
    comparisons = []
    for scope in frozen["scopes"]:
        for direction in frozen["directions"]:
            for k in frozen["donor_k"]:
                left, right = [jobs[(model, scope, direction, k)] for model in frozen["models"]]
                assert left.keys() == right.keys()
                differing = [key for key in left if left[key] != right[key]]
                assert set(differing) <= allowed, differing
                comparisons.append(dict(scope=scope, direction=direction, k=k, differing_options=differing))
    geometry = {}
    for model, info in frozen["models"].items():
        folder = base / "geometry" / model
        assert read(folder / "process_status.json")["status"] == "COMPLETE"
        metrics = read(folder / "metrics.json")
        assert metrics["layer_one_based"] == info["layer_one_based"]
        assert metrics["input_sha256"] == frozen["input_code_sha256"][f"inputs/{model}_geometry.jsonl"]
        meta = pd.read_csv(folder / "state_metadata.csv")
        predictions = pd.read_csv(folder / "confirmation_predictions.csv")
        states = np.load(folder / "selected_states.npz")["states"]
        assert states.ndim == 2 and states.shape[0] == len(meta) == 300
        assert np.isfinite(states).all()
        for split, seeds in (("discovery", info["discovery_seeds"]), ("confirmation", info["confirmation_seeds"])):
            subset = meta[meta.split.eq(split)]
            assert set(subset.seed) == set(seeds) and len(subset) == len(seeds) * 10
            assert all(set(group.occurrence) == set(range(1, 11)) for _, group in subset.groupby("seed"))
        assert not set(info["discovery_seeds"]) & set(info["confirmation_seeds"])
        expected = meta[meta.split.eq("confirmation")].reset_index(drop=True)
        assert predictions[expected.columns].equals(expected)
        result = {}
        for method in ("ncc", "logistic"):
            correct = predictions[f"{method}_prediction"].eq(predictions.occurrence)
            assert abs(float(correct.mean()) - metrics[f"confirmation_{method}_balanced_accuracy"]) < 1e-12
            seed_values = [float(correct[predictions.seed.eq(seed)].mean()) for seed in info["confirmation_seeds"]]
            result[method] = interval(seed_values)
        geometry[model] = dict(layer=info["layer_one_based"], states_shape=list(states.shape),
                               readout=result, files_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
    report = dict(status="PASS", seconds=time.monotonic() - started,
                  frozen_files=len(frozen["input_code_sha256"]), patch_jobs=len(jobs),
                  cpu_validation=cpu_validation, smoke_validation=smoke_validation,
                  model_matched_comparisons=comparisons, packages=common_packages,
                  models=loaded_models, geometry=geometry)
    (args.analysis / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", patch_jobs=len(jobs), geometry=geometry)))


if __name__ == "__main__":
    main()
