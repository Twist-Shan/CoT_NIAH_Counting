"""Compile outcome-independent Retrieve queries and Native-style head banks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
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


def rank_heads(observations):
    """Equal-seed aggregation; repeated queries cannot reweight a seed."""
    by_seed, seen, universe = defaultdict(list), set(), None
    for record in observations:
        key = (record["seed"], record["query_id"])
        if key in seen:
            raise ValueError("Duplicate source query")
        seen.add(key)
        values = {(int(v["layer"]), int(v["head"])): float(v["mass"]) for v in record["heads"]}
        if len(values) != len(record["heads"]) or not values:
            raise ValueError("Empty or duplicated head universe")
        if universe is None:
            universe = set(values)
        if set(values) != universe or any(not math.isfinite(x) or not 0 <= x <= 1.01 for x in values.values()):
            raise ValueError("Inconsistent or nonfinite attention measurements")
        by_seed[int(record["seed"])].append(values)
    if len(by_seed) < 2:
        raise ValueError("Discovery requires at least two independent source seeds")
    scores = {h: sum(sum(q[h] for q in queries) / len(queries) for queries in by_seed.values()) / len(by_seed) for h in universe}
    return [{"layer": h[0], "head": h[1], "score": scores[h]} for h in sorted(scores, key=lambda h: (-scores[h], *h))]


def freeze_banks(ranking, grid, seeds, *, salt, repeats=3):
    heads = [(int(r["layer"]), int(r["head"])) for r in ranking]
    if not heads or len(heads) != len(set(heads)) or grid != sorted(set(grid)) or grid[0] < 1 or grid[-1] > len(heads):
        raise ValueError("Invalid ranking or dose grid")
    if seeds != sorted(set(seeds)) or not seeds:
        raise ValueError("Expected ordered unique confirmation seeds")
    universe, capacity = set(heads), Counter(h[0] for h in heads)
    kmax = grid[-1]
    selected, selected_set = heads[:kmax], set(heads[:kmax])
    remaining = universe - selected_set
    available = Counter(h[0] for h in remaining)
    if len(remaining) < kmax:
        raise ValueError("Insufficient unselected heads even for global random")
    feasible, infeasible = [], []
    for k in grid:
        counts = Counter(h[0] for h in heads[:k])
        bad = {str(layer): {"selected_at_dose": n, "outside_full_selected_bank": available[layer]} for layer, n in counts.items() if n > available[layer]}
        if bad:
            infeasible.append({"k": k, "insufficient_layers": bad})
        else:
            feasible.append(k)
    matched_max = max(feasible, default=0)
    controls = {str(k): "layer_matched_random" if k in feasible else "global_random_capacity_fallback" for k in grid}
    random_banks = {}
    for seed in seeds:
        random_banks[str(seed)] = {str(k): [] for k in grid}
        for repeat in range(repeats):
            pools = {}
            for layer in sorted(capacity):
                pool = sorted(h for h in remaining if h[0] == layer)
                key = f"{salt}/{seed}/{repeat}/{layer}"
                random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16)).shuffle(pool)
                pools[layer] = iter(pool)
            matched = [next(pools[layer]) for layer, _head in selected[:matched_max]]
            global_pool = sorted(remaining)
            key = f"{salt}/{seed}/{repeat}/global_capacity_fallback"
            random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16)).shuffle(global_pool)
            for k in grid:
                bank = (matched if k in feasible else global_pool)[:k]
                assert not set(bank) & selected_set and len(set(bank)) == k
                if k in feasible:
                    assert Counter(h[0] for h in bank) == Counter(h[0] for h in selected[:k])
                random_banks[str(seed)][str(k)].append([list(h) for h in bank])
    return {"ks": grid, "selected_heads": [list(h) for h in selected], "random_banks": random_banks,
            "random_control_by_k": controls, "layer_matching_infeasible": infeasible, "uses_behavioral_outcomes": False}


def choose_confirmation(cells, seeds):
    """Use Native highest-common-count rule within one Enumeration format."""
    indexed = {}
    for model, queries in cells.items():
        table = {}
        for q in queries:
            if q["split"] != "confirmation" or q["seed"] not in seeds:
                continue
            n = q["gold_count"]
            if [q["from_occurrence"], q["to_occurrence"]] != [n - 1, n]:
                continue
            key = (q["seed"], n)
            if key in table:
                raise ValueError("More than one final query per seed/count")
            table[key] = q
        indexed[model] = table
    common = set.intersection(*(set(table) for table in indexed.values()))
    panels, missing = {model: [] for model in cells}, []
    for seed in seeds:
        eligible = sorted(n for s, n in common if s == seed)
        if not eligible:
            missing.append(seed)
            continue
        n = eligible[-1]
        for model in cells:
            panels[model].append({**indexed[model][seed, n], "alignment_key": [seed, n, n - 1, n]})
    return panels, missing


def compile_cell(args):
    tick = time.monotonic()
    cfg, baseline = read(args.output / "protocol.json"), read(args.root / "fresh_v1/manifest.json")
    causal = args.root / "fresh_causal_v1"
    sys.path[:0] = [str(causal / "code/src"), str(causal / "code")]
    from realistic_niah_v4.modeling import load_registered_tokenizer
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
    install_v6_kernel_adapters()
    install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5 import causal as kernel
    tokenizer = load_registered_tokenizer(resolve_model_spec(args.model), cache_dir=args.cache_dir)
    source = causal / "registries" / args.model / args.mode
    source_manifest = read(source / "manifest.json")
    assert sha(source / "adapted_generations.jsonl") == source_manifest["adapted_generations_sha256"]
    assert sha(source / "ledger.json") == source_manifest["ledger_sha256"]
    entries = {r["request_id"]: r for r in read(source / "ledger.json")}
    role = cfg["anchor_roles"][args.model][args.mode]
    queries, ledger = [], []
    with (source / "adapted_generations.jsonl").open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            seed, n = int(row["seed"]), int(row["gold_count"])
            if seed not in baseline["discovery_seeds"] + baseline["read_seeds"]:
                continue
            entry = entries[row["request_id"]]
            path = source / entry["geometry_file"]
            assert sha(path) == entry["geometry_sha256"]
            assert json_sha(row) == read(path)["adapted_row_sha256"]
            eligible, excluded = kernel.mechanism_continuations(row, tokenizer, mechanism="retrieval_anchor_localization")
            chosen = [q for q in eligible if role in q["anchor_roles"]]
            if args.mode == "enumeration_index":
                chosen = [q for q in chosen if "rank_before_city" in q["grammar_pair"]]
            ids = []
            for q in chosen:
                query = int(q["query_output_token_index"])
                assert query < int(q["target_output_token_start"])
                enc = kernel.build_native_causal_encoding(row, tokenizer, query_output_token_index=query,
                    sequence_output_token_end=query + 1, selected_site=q)
                matches = [s for s in enc.prompt_record_spans if s.city.casefold() == q["target_city"].casefold()]
                assert len(matches) == 1
                span = matches[0]
                assert 0 <= span.start < span.end <= enc.prompt_token_count <= enc.query_position == enc.sequence_length - 1
                query_id = row["request_id"] + "/" + q["anchor_equivalence_id"]
                assert query_id not in ids
                ids.append(query_id)
                queries.append({**q, "query_id": query_id, "request_id": row["request_id"],
                    "seed": seed, "gold_count": n, "split": row["split"], "model": args.model, "mode": args.mode,
                    "frozen_anchor_role": role, "row_sha256": json_sha(row),
                    "query_position": enc.query_position, "prompt_token_count": enc.prompt_token_count,
                    "input_ids_sha256": json_sha(list(enc.input_ids)), "attention_mask_sha256": json_sha(list(enc.attention_mask)),
                    "source_record_span": [span.start, span.end]})
            ledger.append({"seed": seed, "gold_count": n, "request_id": row["request_id"], "split": row["split"],
                "baseline_exact_count": row["trace_parse"].get("exact_count"),
                "query_ids": ids, "eligible_queries": len(ids),
                "excluded_requested_role": [q for q in excluded if role == q.get("anchor_role") or role in q.get("anchor_roles", [])]})
    folder = args.output / args.model / args.mode
    assert len(ledger) == 120 and len({q["query_id"] for q in queries}) == len(queries)
    write(folder / "queries.json", queries)
    write(folder / "eligibility.json", ledger)
    result = {"model": args.model, "mode": args.mode, "source_registry": str(source),
        "source_registry_sha256": sha(source / "manifest.json"), "source_generations_sha256": sha(source / "adapted_generations.jsonl"),
        "queries_sha256": sha(folder / "queries.json"), "eligibility_sha256": sha(folder / "eligibility.json"),
        "discovery_queries": sum(q["split"] == "discovery" for q in queries),
        "discovery_eligible_seeds": sorted({q["seed"] for q in queries if q["split"] == "discovery"}),
        "seconds": time.monotonic() - tick}
    write(folder / "cell_manifest.json", result)
    print(json.dumps(result), flush=True)


def prepare(args):
    tick = time.monotonic()
    cfg = read(args.protocol)
    causal = args.root / "fresh_causal_v1"
    code = read(causal / "code_manifest.json")
    for name, expected in {**code["original_code_sha256"], **code["additive_code_sha256"]}.items():
        assert sha(causal / "code" / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", cfg)
    write(args.output / "status.json", {"status": "RUNNING", "phase": "CPU_QUERY_COMPILATION"})
    baseline = read(args.root / "fresh_v1/manifest.json")
    cells, panels, missing = [], {}, {}
    try:
        for mode in cfg["modes"]:
            queries = {}
            for model in cfg["models"]:
                command = [sys.executable, str(Path(__file__).resolve()), "cell", "--root", str(args.root),
                    "--output", str(args.output), "--cache-dir", str(args.cache_dir), "--model", model, "--mode", mode]
                with (args.output / f"{model}_{mode}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                folder = args.output / model / mode
                cells.append(read(folder / "cell_manifest.json"))
                queries[model] = read(folder / "queries.json")
            panels[mode], missing[mode] = choose_confirmation(queries, baseline["read_seeds"])
            for model in cfg["models"]:
                path = args.output / model / mode / "confirmation.json"
                write(path, panels[mode][model])
                cell = next(c for c in cells if c["model"] == model and c["mode"] == mode)
                cell["confirmation_sha256"] = sha(path)
                cell["confirmation_pairs"] = len(panels[mode][model])
                cell["confirmation_counts"] = [q["gold_count"] for q in panels[mode][model]]
        if any(missing.values()):
            write(args.output / "missing_confirmation.json", missing)
            raise ValueError(f"Native highest-common-count rule has missing source seeds: {missing}")
        if any(len(c["discovery_eligible_seeds"]) < 2 for c in cells):
            raise ValueError("Insufficient independent discovery seeds")
        manifest = {"status": "QUERY_GEOMETRY_FROZEN_BEFORE_ATTENTION_AND_INTERVENTION",
            "created_utc": datetime.now(timezone.utc).isoformat(), "causal_stage": str(causal),
            "causal_code_manifest_sha256": sha(causal / "code_manifest.json"),
            "baseline_manifest_sha256": sha(args.root / "fresh_v1/manifest.json"),
            "protocol_sha256": sha(args.output / "protocol.json"), "compiler_sha256": sha(__file__),
            "discovery_seeds": baseline["discovery_seeds"], "confirmation_seeds": baseline["read_seeds"],
            "cells": cells, "uses_intervention_outcomes": False, "uses_final_correctness_for_selection": False,
            "command": sys.argv, "seconds": time.monotonic() - tick}
        write(args.output / "manifest.json", manifest)
        write(args.output / "status.json", {"status": "COMPLETE", "phase": "QUERY_REGISTRY_FROZEN_GPU_PENDING", "seconds": time.monotonic() - tick})
        print(json.dumps(manifest), flush=True)
    except BaseException as error:
        write(args.output / "status.json", {"status": "FAILED", "error": repr(error), "seconds": time.monotonic() - tick})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cell"))
    for name in ("root", "output", "cache-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--mode")
    args = parser.parse_args()
    (prepare if args.action == "prepare" else compile_cell)(args)
