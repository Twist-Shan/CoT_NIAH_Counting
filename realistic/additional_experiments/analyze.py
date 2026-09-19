"""Seed-level exploratory summaries; models and reasoning modes stay separate."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics import ridge_predict
from protocol import read_jsonl, sha256, write_json


def interval(values: dict, draws: int, seed=20260905):
    units = np.asarray([np.mean(v) for _, v in sorted(values.items())], dtype=float)
    if not len(units):
        return {"mean": None, "lower": None, "upper": None, "seeds": 0}
    mean = float(units.mean())
    if len(units) < 2:
        return {"mean": mean, "lower": None, "upper": None, "seeds": len(units)}
    samples = np.random.default_rng(seed).choice(units, size=(draws, len(units)), replace=True).mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {"mean": mean, "lower": float(lo), "upper": float(hi), "seeds": len(units)}


def write_csv(path, rows):
    if not rows:
        path.write_text("status\nNO_ELIGIBLE_ROWS\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def source_basis(x, y, rank=3):
    center = x.mean(0)
    centroids = np.stack([x[y == v].mean(0) - center for v in np.unique(y)])
    _, s, vh = np.linalg.svd(centroids, full_matrices=False)
    effective = min(rank, int(np.count_nonzero(s > max(s[0], 1.) * 1e-6)))
    if effective < 1:
        raise ValueError("Source centroids have no usable rank")
    return center, vh[:effective].T


def analyze(frozen: Path, outputs: list[Path], output: Path, allow_partial=False):
    config = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    cases = read_jsonl(frozen / "cases.jsonl")
    lookup = {r["case_id"]: r for r in cases}
    behavior, causal, observations, inputs, availability = [], [], [], {}, []
    seen_models = set()
    for root in outputs:
        contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
        model = contract["config"]["current_model"]
        if model in seen_models or contract["frozen_audit_sha256"] != sha256(frozen / "freeze_audit.json"):
            raise ValueError("Duplicate model output or mismatched frozen input")
        seen_models.add(model)
        inputs[str(root)] = sha256(root / "run_contract.json")
        captures = list((root / "captures").glob("*/*/complete.json"))
        expected_keys = {(mode, case["case_id"]) for mode in config["modes"] for case in cases}
        actual_keys = {(p.parent.parent.name, p.parent.name) for p in captures}
        if actual_keys != expected_keys and not allow_partial:
            raise ValueError("Incomplete/extra capture coverage; use --allow-partial for explicitly partial analysis")
        if not actual_keys <= expected_keys:
            raise ValueError("Unregistered capture keys")
        for complete in captures:
            folder, mode = complete.parent, complete.parent.parent.name
            completed = json.loads(complete.read_text(encoding="utf-8"))
            for rel, expected in completed["files"].items():
                if sha256(folder / rel) != expected:
                    raise ValueError(f"Capture audit failed: {folder / rel}")
            case = lookup[folder.name]
            generated = json.loads((folder / "generation.json").read_text(encoding="utf-8"))
            meta = json.loads((folder / "sites.json").read_text(encoding="utf-8"))
            behavior.append({"model": model, "mode": mode, "case_id": case["case_id"],
                             "task": case["task"], "seed": case["seed"], "split": case["split"],
                             "correct": int(generated["correct"]), "parse_ok": int(generated["parse_ok"]),
                             "truncated": int(generated["generation_truncated"]),
                             "tokens": generated["generated_token_count"],
                             "endpoint_available": int(meta["endpoint_input_ids"] is not None),
                             "prompt_tokens": meta["prompt_token_count"]})
            with np.load(folder / "states.npz") as z:
                for layer in config["capture_layers"][model]:
                    for index, site in enumerate(meta["sites"]):
                        if site["kind"] == "answer_query":
                            continue
                        observations.append({"model": model, "mode": mode, "case_id": case["case_id"],
                            "seed": case["seed"], "split": case["split"], "task": case["task"],
                            "layer": layer, **site, "x": z[f"layer_{layer}"][index]})
        expected_causal = {(m, c["case_id"]) for m in config["modes"] for c in cases if c["split"] == "confirmation"}
        causal_paths = list((root / "causal").glob("*/*.json"))
        actual_causal = {(p.parent.name, p.stem) for p in causal_paths}
        if actual_causal != expected_causal and not allow_partial:
            raise ValueError("Incomplete causal coverage")
        if not actual_causal <= expected_causal:
            raise ValueError("Unregistered causal case")
        for p in causal_paths:
            row = json.loads(p.read_text(encoding="utf-8"))
            availability.append({"model": model, "mode": row["mode"], "case_id": row["case_id"],
                                 "status": row["status"], "reason": row.get("reason", "")})
            if row["status"] != "PASS":
                continue
            arms = {a["arm"]: a for a in row["arms"]}
            randoms = [arms[f"random_bank_{i}"] for i in range(config["random_head_replicates"])]
            common = {"model": model, "mode": row["mode"], "task": row["task"], "seed": row["seed"], "case_id": row["case_id"]}
            causal.append({**common, "contrast": "legacy_bank_accuracy_damage_minus_random",
                           "value": float(np.mean([r["correct"] for r in randoms])) - arms["legacy_broad_bank"]["correct"]})
            donor = row["donor_model_prediction"]
            for layer in config["patch_layers"][model]:
                if donor is not None and arms["clean"]["prediction"] is not None and donor != arms["clean"]["prediction"]:
                    causal.append({**common, "contrast": f"donor_prediction_transport_minus_orthogonal_L{layer}",
                        "value": int(arms[f"donor_patch_L{layer}"]["prediction"] == donor) -
                                 int(arms[f"orthogonal_patch_L{layer}"]["prediction"] == donor)})
    if not behavior:
        raise ValueError("No real completed model outputs; will not create an empty experiment report")
    if seen_models != set(config["models"]) and not allow_partial:
        raise ValueError("Both registered models are required for a complete report")
    output.mkdir(parents=True, exist_ok=False)
    behavior_summary, causal_summary, probe_summary, probe_details = [], [], [], []
    groups = defaultdict(list)
    for row in behavior:
        groups[row["model"], row["mode"], row["task"], row["split"]].append(row)
    for (model, mode, task, split), rows in sorted(groups.items()):
        for metric in ("correct", "parse_ok", "truncated", "tokens", "endpoint_available", "prompt_tokens"):
            by_seed = defaultdict(list)
            for r in rows:
                by_seed[r["seed"]].append(r[metric])
            behavior_summary.append({"model": model, "mode": mode, "task": task, "split": split,
                                     "metric": metric, "cases": len(rows), **interval(by_seed, config["bootstrap_replicates"])})
    groups = defaultdict(list)
    for row in causal:
        groups[row["model"], row["mode"], row["task"], row["contrast"]].append(row)
    for (model, mode, task, contrast), rows in sorted(groups.items()):
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row["value"])
        causal_summary.append({"model": model, "mode": mode, "task": task, "contrast": contrast,
                               "cases": len(rows), **interval(by_seed, config["bootstrap_replicates"])})
    obs_groups = defaultdict(list)
    for o in observations:
        obs_groups[o["model"], o["mode"], o["kind"], o["layer"]].append(o)
    for (model, mode, kind, layer), rows in sorted(obs_groups.items()):
        source = [r for r in rows if r["split"] == "discovery" and r["task"] == "count_all"]
        if len(source) < 2:
            continue
        sx = np.stack([r["x"] for r in source])
        sy = np.asarray([r["total_prefix"] for r in source])
        center, basis = source_basis(sx, sy)
        rng = np.random.default_rng(config["random_seed"] + layer)
        rand = rng.standard_normal(basis.shape)
        rand -= basis @ (basis.T @ rand)
        random_basis = np.linalg.qr(rand)[0][:, :basis.shape[1]]
        for task in config["tasks"]:
            train_task = [r for r in rows if r["split"] == "discovery" and r["task"] == task]
            test = [r for r in rows if r["split"] == "confirmation" and r["task"] == task]
            if not test or len(train_task) < 2:
                continue
            target = "target_prefix" if task == "topic_count" else "total_prefix"
            tx = np.stack([r["x"] for r in test])
            truth = np.asarray([r[target] for r in test])
            predictors = {
                "reference_full_state_transfer": ridge_predict(sx, sy, tx, config["ridge_alpha"]),
                "reference_rank3_transfer": ridge_predict((sx-center) @ basis, sy, (tx-center) @ basis, config["ridge_alpha"]),
                "reference_random_subspace_transfer": ridge_predict((sx-center) @ random_basis, sy, (tx-center) @ random_basis, config["ridge_alpha"]),
                "within_task_full_state": ridge_predict(np.stack([r["x"] for r in train_task]),
                    [r[target] for r in train_task], tx, config["ridge_alpha"]),
                "within_task_position_ordinal_baseline": ridge_predict(
                    [[r["position"] / 10000, r["ordinal"] / 10] for r in train_task],
                    [r[target] for r in train_task],
                    [[r["position"] / 10000, r["ordinal"] / 10] for r in test], config["ridge_alpha"]),
            }
            legacy = frozen / "legacy_native_bases" / model / "item_end_discovery_basis.npz"
            with np.load(legacy) as z:
                if kind == "native_item_end" and f"basis_L{layer}" in z.files:
                    b, c = z[f"basis_L{layer}"], z[f"center_L{layer}"]
                    predictors["legacy_native_basis_reference_calibration"] = ridge_predict(
                        (sx-c) @ b, sy, (tx-c) @ b, config["ridge_alpha"])
            for estimator, prediction in predictors.items():
                for metric, values in (("mae", np.abs(prediction-truth)),
                                       ("rounded_accuracy", np.rint(prediction) == truth)):
                    by_seed = defaultdict(list)
                    for r, v in zip(test, values):
                        by_seed[r["seed"]].append(float(v))
                    probe_summary.append({"model": model, "mode": mode, "site": kind, "layer": layer,
                        "task": task, "estimator": estimator, "metric": metric, "observations": len(test),
                        "train_seeds": len({r["seed"] for r in (train_task if estimator.startswith("within_task") else source)}),
                        "source_rank": basis.shape[1], **interval(by_seed, config["bootstrap_replicates"])})
                for r, truth_i, pred in zip(test, truth, prediction):
                    probe_details.append({"model": model, "mode": mode, "site": kind, "layer": layer,
                        "task": task, "case_id": r["case_id"], "seed": r["seed"], "position": r["position"],
                        "estimator": estimator, "truth": float(truth_i), "prediction": float(pred),
                        "explicit_index": r.get("explicit_index", False)})
    for name, rows in (("behavior_cases", behavior), ("behavior_summary", behavior_summary),
                       ("causal_case_contrasts", causal), ("causal_summary", causal_summary),
                       ("causal_availability", availability), ("probe_summary", probe_summary), ("probe_predictions", probe_details)):
        write_csv(output / (name + ".csv"), rows)
    write_json(output / "analysis_audit.json", {"status": "PASS", "partial_analysis_requested": allow_partial,
        "models_present": sorted(seen_models), "inputs": inputs, "bootstrap_unit": "seed",
        "confirmatory_inference": False, "source_probe_fit_split": "discovery", "probe_test_split": "confirmation",
        "limitations": ["Four confirmation seeds in pilot; bootstrap intervals are exploratory.",
            "Topic descriptions change token lengths; kth passages are exactly shared across k.",
            "Native line-end grammar may differ from historical item_end sites; inspect explicit_index strata.",
            "Native terminal interventions fix the existing trace, which can already contain the answer.",
            "Legacy broad-head reuse and full-state transport do not establish a shared unique counter.",
            "No native intermediate item-to-item causal rollout is implemented in this pilot."]})
    lines = ["# Additional task transfer: exploratory results", "",
             "Paired results are grouped by model, mode and task. Intervals resample seeds. This pilot does not establish a shared full causal chain.", "",
             "| Model | Mode | Task | Confirmation accuracy | Seeds |", "|---|---|---|---:|---:|"]
    for r in behavior_summary:
        if r["split"] == "confirmation" and r["metric"] == "correct":
            lines.append(f"| {r['model']} | {r['mode']} | {r['task']} | {r['mean']:.3f} | {r['seeds']} |")
    lines += ["", "Causal effects: see `causal_summary.csv`; positive bank damage means more harm than matched random heads. Positive donor transport means more following of the donor model prediction than the orthogonal displacement control.",
              "", "Representation transfer: see `probe_summary.csv` and the prediction-level table. Information readout alone does not demonstrate causal use. No intermediate native causal-chain result is claimed."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analyze(args.frozen, args.outputs, args.output, args.allow_partial)
