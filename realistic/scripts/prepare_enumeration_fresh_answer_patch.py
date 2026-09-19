"""Freeze Enumeration answer-state pairs using the current Native pair rule."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_edges(counts):
    """Exact low/high anchor and tie rule of the frozen Native selector."""
    if counts != sorted(set(counts)):
        raise ValueError("Common counts must be sorted and unique")
    edges = list(zip(counts[:-1], counts[1:]))
    if len(edges) < 2:
        raise ValueError("Need at least two common edges")
    chosen = []
    for target in (1.5, 9.5):
        chosen.append(min((edge for edge in edges if edge not in chosen),
            key=lambda edge: (abs((edge[0] + edge[1]) / 2 - target), edge[1] - edge[0], edge[0])))
    return sorted(chosen)


def eligibility(entry, row, *, clean=None, require_clean=False):
    """Answer-state transfer has its own correct-baseline population."""
    parsed = row["trace_parse"]
    if not parsed["parser"].get("trace_one_to_one"):
        return "not_one_to_one"
    if not parsed.get("exact_count"):
        return "baseline_count_incorrect"
    if not entry["read_eligible"]:
        return "no_legal_answer_query"
    if int(parsed["gold_count"]) != int(row["gold_count"]):
        raise ValueError("Baseline parser count disagrees with source count")
    if require_clean:
        if clean is None or clean["status"] != "ok" or clean["condition"] != "clean":
            raise ValueError("Missing unmodified answer-query replay")
        if clean["request_id"] != row["request_id"] or clean["gold_count"] != row["gold_count"]:
            raise ValueError("Clean replay belongs to a different input")
        if clean["exact_count"] != (clean["prediction"] == row["gold_count"]):
            raise ValueError("Inconsistent clean replay accuracy")
        if not clean["exact_count"]:
            return "answer_query_clean_count_incorrect"
        generated = clean["generated"]
        if generated["generation_truncated"] or not generated["stopped_on_eos"] or len(generated["generated_token_ids"]) >= 16:
            raise ValueError("32-token clean result cannot certify the 16-token query baseline")
    return None


def select_pairs(indexed, seeds, models):
    panels, counts_by_seed, shortfalls = {model: [] for model in models}, {}, []
    for seed in seeds:
        common = sorted(set.intersection(*(set(indexed[model].get(seed, {})) for model in models)))
        counts_by_seed[str(seed)] = common
        try:
            edges = selected_edges(common)
        except ValueError as error:
            shortfalls.append({"seed": seed, "common_counts": common, "reason": str(error)})
            continue
        for lower, upper in edges:
            for receiver, donor in ((lower, upper), (upper, lower)):
                alignment_id = f"seed{seed}_R{receiver}_D{donor}"
                for model in models:
                    panels[model].append({"pair_id": f"{model}__{alignment_id}",
                        "alignment_pair_id": alignment_id, "alignment_key": [seed, receiver, donor],
                        "model_label": model, "seed": seed, "split": "confirmation",
                        "receiver_count": receiver, "donor_count": donor,
                        "receiver_request_id": indexed[model][seed][receiver]["request_id"],
                        "donor_request_id": indexed[model][seed][donor]["request_id"],
                        "receiver_exact_count": True, "donor_exact_count": True,
                        "receiver_site_id": "answer_query_v3", "donor_site_id": "answer_query_v3",
                        "pair_direction": "higher_to_lower" if donor > receiver else "lower_to_higher",
                        "pair_selection_uses_patch_outcomes": False})
    keys = [[pair["alignment_key"] for pair in panels[model]] for model in models]
    if any(value != keys[0] for value in keys[1:]):
        raise RuntimeError("Cross-model pair keys differ")
    return panels, counts_by_seed, shortfalls


def prepare(args):
    tick = time.monotonic()
    cfg = read(args.protocol)
    bundle, causal = args.root / "fresh_v1", args.root / "fresh_causal_v1"
    baseline = read(bundle / "manifest.json")
    code_manifest = read(causal / "code_manifest.json")
    for name, expected in {**code_manifest["original_code_sha256"], **code_manifest["additive_code_sha256"]}.items():
        if sha(causal / "code" / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")
    seeds = baseline["read_seeds"]
    assert seeds == sorted(set(seeds)) and len(seeds) == cfg["expected_source_seeds"]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", cfg)
    result = {"status": "FROZEN_BEFORE_ANSWER_PATCH_GPU", "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_bundle": str(bundle), "causal_stage": str(causal), "source_seeds": seeds,
        "baseline_manifest_sha256": sha(bundle / "manifest.json"),
        "causal_code_manifest_sha256": sha(causal / "code_manifest.json"),
        "protocol_sha256": sha(args.output / "protocol.json"), "compiler_sha256": sha(__file__),
        "modes": {}, "cells": [], "command": sys.argv, "intervention_outcomes_accessed": False,
        "clean_regeneration_required": bool(cfg.get("clean_regeneration_required", False)),
        "selection_amendment": cfg.get("selection_amendment"), "clean_regeneration_sources": []}
    for mode in cfg["modes"]:
        indexed = {}
        for model in cfg["models"]:
            source = causal / "registries" / model / mode
            manifest = read(source / "manifest.json")
            assert manifest["baseline_manifest_sha256"] == result["baseline_manifest_sha256"]
            assert sha(source / "ledger.json") == manifest["ledger_sha256"]
            assert sha(source / "adapted_generations.jsonl") == manifest["adapted_generations_sha256"]
            with (source / "adapted_generations.jsonl").open(encoding="utf-8") as handle:
                rows = {row["request_id"]: row for row in map(json.loads, handle)}
            entries = [entry for entry in read(source / "ledger.json") if "unfiltered_read" in entry["roles"]]
            assert len(entries) == 100 and {entry["seed"] for entry in entries} == set(seeds)
            clean = {}
            if cfg.get("clean_regeneration_required", False):
                clean_root = causal / "gpu_jobs/formal/read" / model / mode
                clean_status, clean_runtime = read(clean_root / "status.json"), read(clean_root / "runtime.json")
                assert clean_status["status"] == "COMPLETE" and clean_status["completed"] == 500
                assert sha(clean_root / "trials.jsonl") == clean_status["trials_sha256"]
                assert clean_runtime["registry_sha256"] == sha(source / "manifest.json")
                assert clean_runtime["runner_sha256"] == sha(causal / "code/scripts/run_enumeration_fresh_read.py")
                assert clean_runtime["max_new_tokens"] == 32 and clean_runtime["backend"] == cfg["attention_backend"]
                with (clean_root / "trials.jsonl").open(encoding="utf-8") as handle:
                    for trial in map(json.loads, handle):
                        if trial["condition"] != "clean":
                            continue
                        assert trial["model"] == model and trial["mode"] == mode
                        assert trial["request_id"] not in clean
                        clean[trial["request_id"]] = trial
                assert set(clean) == {entry["request_id"] for entry in entries}
                result["clean_regeneration_sources"].append({"model": model, "mode": mode,
                    "path": str(clean_root), "selected_condition": "clean",
                    "files_sha256": {name: sha(clean_root / name) for name in ("trials.jsonl", "status.json", "runtime.json")},
                    "input_count": len(clean), "selection_uses_other_conditions": False})
            ledger, indexed[model] = [], {}
            for entry in entries:
                row = rows[entry["request_id"]]
                path = source / entry["geometry_file"]
                assert sha(path) == entry["geometry_sha256"]
                geometry = read(path)
                assert json_sha(row) == geometry["adapted_row_sha256"]
                error = eligibility(entry, row, clean=clean.get(entry["request_id"]),
                                    require_clean=bool(cfg.get("clean_regeneration_required", False)))
                record = {"seed": entry["seed"], "gold_count": entry["gold_count"],
                    "request_id": entry["request_id"], "eligible": error is None, "exclusion": error,
                    "geometry_file": str(path), "geometry_sha256": entry["geometry_sha256"],
                    "adapted_row_sha256": geometry["adapted_row_sha256"],
                    "baseline_count": row["trace_parse"].get("parsed_count")}
                if clean:
                    observation = clean[entry["request_id"]]
                    assert observation["geometry_sha256"] == entry["geometry_sha256"]
                    record.update(clean_regeneration_prediction=observation["prediction"],
                                  clean_regeneration_row_sha256=json_sha(observation),
                                  clean_regeneration_generated=observation["generated"])
                if error is None:
                    by_count = indexed[model].setdefault(entry["seed"], {})
                    if entry["gold_count"] in by_count:
                        raise ValueError("Duplicate eligible source seed/count")
                    by_count[entry["gold_count"]] = record
                    record["read_geometry"] = geometry["read"]
                ledger.append(record)
            folder = args.output / model / mode
            write(folder / "eligibility.json", ledger)
            result["cells"].append({"model": model, "mode": mode, "source_registry": str(source),
                "source_registry_sha256": sha(source / "manifest.json"),
                "source_generations_sha256": sha(source / "adapted_generations.jsonl"),
                "eligibility_sha256": sha(folder / "eligibility.json"), "eligible_inputs": sum(row["eligible"] for row in ledger)})
        panels, counts, shortfalls = select_pairs(indexed, seeds, cfg["models"])
        for model, pairs in panels.items():
            for pair in pairs:
                pair.update(mode=mode, layers=list(range(cfg["num_layers"][model])))
            folder = args.output / model / mode
            write(folder / "pairs.json", pairs)
            cell = next(value for value in result["cells"] if value["model"] == model and value["mode"] == mode)
            cell.update(pairs=len(pairs), pairs_sha256=sha(folder / "pairs.json"),
                layers=cfg["num_layers"][model], expected_rows=len(pairs) * cfg["num_layers"][model] * 2)
        result["modes"][mode] = {"common_counts_by_seed": counts, "shortfalls": shortfalls,
            "pairs_per_model": len(panels[cfg["models"][0]]), "cross_model_exact_pair_alignment": True}
        if shortfalls:
            result["status"] = "INSUFFICIENT_BASELINE_ELIGIBILITY_GPU_NOT_AUTHORIZED_BY_REGISTRY"
    result["seconds"] = time.monotonic() - tick
    write(args.output / "manifest.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(parser.parse_args())
