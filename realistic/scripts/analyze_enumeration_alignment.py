#!/usr/bin/env python3
"""Recompute comparable definitions on saved, still-unmatched experimental cohorts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.aligned_reporting import continuation_summary, relay_point, relay_summary

CONDITIONS = {"receiver_self", "native_donor", "donor_to_receiver"}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; archived results are immutable")
    started = time.monotonic()
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")

    def load_run(path, *, mode, model, scope, direction, k, layer, seeds=None):
        rows = [json.loads(s) for s in read(path).splitlines() if s.strip()]
        generated_conditions = set(json.loads(read(path.parent / "manifest.json"))["generation_conditions"])
        if not {"receiver_self", "donor_to_receiver"} <= generated_conditions <= CONDITIONS:
            raise ValueError(f"Unexpected generation contract: {path}")
        identities = {(r["seed"], r["condition"]) for r in rows}
        observed_seeds = {r["seed"] for r in rows}
        if len(rows) != 30 or len(identities) != 30 or len(observed_seeds) != 10:
            raise ValueError(f"Incomplete / duplicated ten-seed grid: {path}")
        if seeds is not None and observed_seeds != set(seeds):
            raise ValueError(f"Unexpected source seeds: {path}")
        if identities != {(s, c) for s in observed_seeds for c in CONDITIONS}:
            raise ValueError(f"Unpaired conditions: {path}")
        for r in rows:
            receiver = k - 1 if direction == "forward" else k + 1
            if (r["gold_count"], r["donor_occurrence_k"], r["receiver_occurrence_j"], r["layer"]) != (10, k, receiver, layer - 1):
                raise ValueError(f"Changed count, pair or layer: {path}")
            if r["donor_successor"] != k + 1:
                raise ValueError(f"Inconsistent saved parser fields: {path}")
            generated = r["condition"] in generated_conditions
            if generated != ("generated_known_city_ordinals_any_surface" in r):
                raise ValueError(f"Missing or unregistered generation: {path}")
            if generated and r["first_generated_known_city_ordinal"] != next(iter(r["generated_known_city_ordinals_any_surface"]), None):
                raise ValueError(f"Inconsistent first generated ordinal: {path}")
            expected_scope = "item_span" if scope == "item_span" else "fixed_suffix"
            if r["patch_scope"] != expected_scope:
                raise ValueError(f"Unexpected scope: {path}")
            r.update(mode=mode, model=model, scope=scope, direction=direction,
                     layer_display=layer, generation_available=generated)
        return rows

    config = json.loads(read(args.config))
    if config["schema_version"] != "enumeration_alignment_reanalysis_v1":
        raise ValueError("Unsupported configuration")
    read(Path(__file__))
    read(ROOT / "src/realistic_niah_v6/aligned_reporting.py")
    bundle = ROOT / config["thinking_bundle"]
    manifest = json.loads(read(bundle / "manifest.json"))
    rows = []
    for model, info in manifest["models"].items():
        layer = config["expected_thinking_layers_one_based"][model]
        if info["layer_one_based"] != layer:
            raise ValueError("The specified Thinking bundle is not the current L19/L21 experiment")
        for scope in manifest["scopes"]:
            for direction in manifest["directions"]:
                for k in manifest["donor_k"]:
                    run = bundle / "runs" / model / scope / f"{direction}_k{k}"
                    if json.loads(read(run / "process_status.json"))["status"] != "COMPLETE":
                        raise ValueError(f"Incomplete job: {run}")
                    rows += load_run(run / "results/trials.jsonl", mode="thinking", model=model,
                                     scope=scope, direction=direction, k=k, layer=layer,
                                     seeds=info["confirmation_seeds"])
    thinking_count = len(rows)
    if thinking_count != config["expected_thinking_rows"]:
        raise ValueError("Incomplete Thinking grid")
    attention_flags = []
    for mode in ("enumeration_index", "enumeration_bullet"):
        for model, layer in config["expected_enumeration_layers_one_based"].items():
            for direction, old_name in (("forward", "forward_skip"), ("backward", "backward_rewind")):
                run = ROOT / config["enumeration_continuations"] / mode / model / old_name
                meta = json.loads(read(run / "manifest.json"))
                adapter = json.loads(read(run / "v6_adapter_manifest.json"))
                if meta["status"] != "PASS" or adapter["count_stream_seed_membership_adapter"]["seed_aliasing"]:
                    raise ValueError(f"Unaudited seed identity or failed run: {run}")
                attention_flags.append({"run": str(run), "run_attention": meta["run_attention"]})
                rows += load_run(run / "trials.jsonl", mode=mode, model=model, scope="item_span",
                                 direction=direction, k=6, layer=layer)
    if len(rows) - thinking_count != config["expected_enumeration_rows"]:
        raise ValueError("Incomplete Enumeration grid")
    load_seconds = time.monotonic() - started
    groups = defaultdict(list)
    scoring_only = []
    for r in rows:
        if not r["generation_available"]:
            scoring_only.append({k: r[k] for k in ("mode", "model", "scope", "direction", "seed", "condition", "donor_occurrence_k", "request_id")})
            continue
        key = tuple(r[k] for k in ("mode", "model", "scope", "direction", "condition", "layer_display"))
        groups[(*key, "all")].append(r)
        groups[(*key, str(r["donor_occurrence_k"]))].append(r)
    if (len(rows)-len(scoring_only), len(scoring_only)) != (config["expected_generated_rows"], config["expected_scoring_only_rows"]):
        raise ValueError("Changed generation/scoring-only support")
    summaries = []
    options = dict(draws=config["bootstrap_draws"], random_seed=config["bootstrap_random_seed"])
    for key, group in sorted(groups.items()):
        identity = dict(zip(("mode", "model", "scope", "direction", "condition", "layer", "donor_k"), key))
        for hop in config["hops"]:
            summaries.append({**identity, **continuation_summary(group, hop, **options)})
    continuation_seconds = time.monotonic() - started - load_seconds

    pair_rows = list(csv.DictReader(read(ROOT / config["thinking_relay_pairs"]).splitlines()))
    keys = {(r["seed"], r["gold_count"], r["donor_offset"]) for r in pair_rows}
    if len(keys) != len(pair_rows) or len(pair_rows) != config["expected_thinking_relay_pairs"]:
        raise ValueError("Changed / duplicated Thinking relay pair registry")
    relays = []
    for model, suffix in (("Qwen3-8B", "qwen"), ("Gemma4-E4B", "gemma")):
        effects = []
        for r in pair_rows:
            natural = float(r[f"patch_damage_natural__{suffix}"])
            remaining = float(r[f"patch_damage__post_terminal_suffix__{suffix}"])
            if not np.isclose(natural - remaining, float(r[f"specific_mediation__post_terminal_suffix__{suffix}"]), atol=1e-10, rtol=0):
                raise ValueError("Thinking relay factorial identity failed")
            effects.append(dict(seed=int(r["seed"]), natural_damage=natural, remaining_signed_damage=remaining))
        relays.append(dict(mode="thinking", model=model, item_patch_tokens=8,
                           evidence="saved_pair_effects_rebootstrap", **relay_summary(effects, **options)))
    html = read(ROOT / config["enumeration_report"])
    start = html.index('id="report-manifest"')
    report = json.loads(html[html.index(">", start) + 1:html.index("</script>", start)])
    cells = report["answer_trace_extension"]["cells"]
    if len(cells) != 4 or len({(c["prompt_mode"], c["model_label"]) for c in cells}) != 4:
        raise ValueError("Enumeration relay summary misses a mode/model cell")
    for cell in cells:
        natural = cell["terminal_patch"]["estimate"]
        point = relay_point(natural, natural - cell["suffix_mediation"]["estimate"])
        ratio = cell["suffix_residual_ratio"]
        if not np.isclose(point["absolute_reduction"], 1 - ratio["estimate"], atol=1e-12, rtol=0):
            raise ValueError("Archived relay estimands do not obey the assumed identity")
        relays.append(dict(mode=cell["prompt_mode"], model=cell["model_label"],
                           item_patch_tokens=int(cell["relay_geometry"].removeprefix("suffix")),
                           source_seed_count=cell["relay_eligible_seed_count"], **point,
                           evidence="archived_aggregate_algebra_only",
                           ci95={"absolute_reduction": [1-ratio["high"], 1-ratio["low"]]},
                           signed_ci_status="requires_original_paired_effects; marginal_CIs_cannot_be_combined"))
    relay_seconds = time.monotonic() - started - load_seconds - continuation_seconds
    # Recheck the read-only evidence before publishing the result of this run.
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input changed while analyzing: {path}")
    args.output.mkdir(parents=True)
    def dump(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    dump("config.json", config)
    dump("registered_scoring_only.json", scoring_only)
    dump("continuation.json", summaries)
    dump("relay.json", relays)
    flat = [{k:v for k,v in r.items() if not isinstance(v, (dict, list))} | {
        "unconditional_rate": r["unconditional"]["estimate"], "conditional_rate": r["conditional"]["estimate"]
    } for r in summaries]
    write_csv(args.output / "continuation.csv", flat)
    lines = ["# Enumeration alignment: saved-output reanalysis", "",
             "Status: CPU reanalysis completed; GPU alignment reruns have not started.", "",
             "These tables harmonize definitions. Cohorts, prompts, layers, spans and token budgets remain historical.",
             "Continuation uses saved city ordinals; it does not newly verify item scores, syntax or final answers.", "",
             "## Item-span Target continuations (all historical k values pooled)", "",
             "| Mode | Model | Direction | Hop | Exact prefix / horizon | Next success / prior success |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in summaries:
        if r["scope"] == "item_span" and r["condition"] == "donor_to_receiver" and r["donor_k"] == "all":
            lines.append(f"| {r['mode']} | {r['model']} | {r['direction']} | {r['hop']} | {r['successes']}/{r['horizon_eligible']} | {r['successes']}/{r['conditional_eligible']} |")
    lines += ["", "No remaining donor record is a horizon exclusion, not a failed generation. Earlier failures and truncations remain in the unconditional horizon denominator.",
              "The full exports retain self controls, each k, and the Thinking endpoint/tail controls. Native-donor continuations exist only for Enumeration; Thinking registered 360 native-donor scoring-only rows. These have no generation outcome and are separately inventoried, never counted as failures or successes.", "",
              "## Terminal relay (post-terminal reset)", "",
              "| Mode | Model | Item tail | Seeds | Natural damage | Remaining signed damage | Signed reduction | Absolute reduction |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in relays:
        lines.append(f"| {r['mode']} | {r['model']} | {r['item_patch_tokens']} | {r['source_seed_count']} | {r['natural_damage']:.6f} | {r['remaining_signed_damage']:.6f} | {r['signed_reduction']:.1%} | {r['absolute_reduction']:.1%} |")
    lines += ["", "Thinking intervals resample paired seed means. Enumeration points are algebraic re-expressions of archived aggregates; only the archived absolute-ratio interval is transformed. Its signed-ratio interval still needs original paired effects or new runs.",
              "Item-tail width is the source patch support. The post-terminal reset covers all positions after the terminal item through the answer query.",
              "A signed reduction above 100% indicates a sign reversal of remaining damage; it is not a bounded mediation percentage.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    dump("audit.json", dict(status="PASS_CPU_REANALYSIS", gpu_rerun=False,
                            thinking_condition_rows=thinking_count, enumeration_condition_rows=len(rows)-thinking_count,
                            generated_rows=len(rows)-len(scoring_only), registered_scoring_only_rows=len(scoring_only),
                            input_sha256=hashes, enumeration_attention_flags=attention_flags,
                            command=sys.argv, python=sys.version, numpy=np.__version__, platform=platform.platform(),
                            timings_seconds=dict(load=load_seconds, continuation=continuation_seconds,
                                                 relay=relay_seconds, total=time.monotonic()-started)))
    print(json.dumps(dict(status="PASS_CPU_REANALYSIS", condition_rows=len(rows), output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
