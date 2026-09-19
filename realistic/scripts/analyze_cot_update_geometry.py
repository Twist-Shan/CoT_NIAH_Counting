"""Audit a complete geometry-layer update grid and summarize paired results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

CONDITIONS = ("receiver_self", "native_donor", "donor_to_receiver")
BOOTSTRAP_SEED = 20260914
DRAWS = 10000


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(s) for s in handle if s.strip()]


def interval(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (10,) and np.isfinite(values).all()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = values[rng.integers(0, 10, size=(DRAWS, 10))].mean(axis=1)
    return dict(estimate=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                seed_values=values.tolist(), bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=DRAWS)


def steps(rows):
    result = []
    for hop in range(1, 5):
        eligible, successes = [], []
        for r in rows:
            k = int(r["donor_occurrence_k"])
            cities = r["generated_known_city_ordinals_any_surface"]
            if k + hop <= 10 and cities[:hop - 1] == list(range(k + 1, k + hop)):
                eligible.append(r)
                if len(cities) >= hop and cities[hop - 1] == k + hop:
                    successes.append(r)
        result.append(dict(hop=hop, success=len(successes), eligible=len(eligible),
                           eligible_seed_count=len({r['seed'] for r in eligible})))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=("Qwen3-8B", "Gemma4-E4B"))
    args = parser.parse_args()
    start = time.monotonic()
    base = args.bundle
    manifest = json.loads((base / "manifest.json").read_text())
    if args.models:
        manifest["models"] = {m: info for m, info in manifest["models"].items() if m in args.models}
    scopes = ["item_span"] if args.primary_only else manifest["scopes"]
    allrows, input_hashes = [], {}
    for model, info in manifest["models"].items():
        expected = {(seed, condition) for seed in info["confirmation_seeds"] for condition in CONDITIONS}
        for scope in scopes:
            for direction in manifest["directions"]:
                for k in manifest["donor_k"]:
                    folder = base / "runs" / model / scope / f"{direction}_k{k}"
                    assert json.loads((folder / "process_status.json").read_text())["status"] == "COMPLETE"
                    path = folder / "results/trials.jsonl"
                    input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    rows = read_rows(path)
                    assert len(rows) == len(expected) == 30
                    assert {(r["seed"], r["condition"]) for r in rows} == expected
                    assert Counter(r["seed"] for r in rows) == {s: 3 for s in info["confirmation_seeds"]}
                    geometries = read_rows(folder / "results/geometry_audit.jsonl")
                    assert len(geometries) == 10 and {r["seed"] for r in geometries} == set(info["confirmation_seeds"])
                    for r in rows:
                        assert r["layer"] == info["layer_zero_based"] and r["gold_count"] == 10
                        assert r["receiver_occurrence_j"] == (k - 1 if direction == "forward" else k + 1)
                        assert r["donor_occurrence_k"] == k and r["patch_applications"] == 1
                        assert r["patch_scope"] == ("item_span" if scope == "item_span" else "fixed_suffix")
                        if scope != "item_span":
                            assert r["patch_width"] == (4 if scope == "four_token_tail" else 1)
                        assert all(np.isfinite(r[f]) for f in ("donor_vs_receiver_sum_logodds",
                                                               "donor_vs_receiver_attention_log_ratio",
                                                               "realized_patch_delta_norm"))
                        if r["condition"] in ("receiver_self", "native_donor"):
                            assert r["realized_patch_delta_norm"] < 1e-5
                        if r["condition"] != "native_donor":
                            first = r["generated_known_city_ordinals_any_surface"][:1]
                            assert r["greedy_donor_successor_adoption"] == (first == [k + 1])
                        r.update(model=model, scope=scope, direction=direction, layer_display=info["layer_one_based"])
                    allrows.extend(rows)
    index = {(r["model"], r["scope"], r["direction"], r["donor_occurrence_k"], r["seed"], r["condition"]): r for r in allrows}
    assert len(index) == len(allrows)
    # Scope comparisons must preserve the directed pair and absolute endpoint.
    for key, r in index.items():
        model, scope, direction, k, seed, condition = key
        reference = index[(model, "item_span", direction, k, seed, condition)]
        for field in ("request_id", "shared_commit_position", "receiver_occurrence_j", "donor_occurrence_k", "layer"):
            assert r[field] == reference[field], (key, field)
    summaries = []
    for model, info in manifest["models"].items():
        seeds = info["confirmation_seeds"]
        for scope in scopes:
            for direction in ("forward", "backward", "both"):
                rows = [r for r in allrows if r["model"] == model and r["scope"] == scope
                        and (direction == "both" or r["direction"] == direction)]
                target = [r for r in rows if r["condition"] == "donor_to_receiver"]
                controls = [r for r in rows if r["condition"] == "receiver_self"]
                assert len(target) == len(controls) == (60 if direction == "both" else 30)
                targets = [np.mean([r["greedy_donor_successor_adoption"] for r in target if r["seed"] == seed]) for seed in seeds]
                selves = [np.mean([r["greedy_donor_successor_adoption"] for r in controls if r["seed"] == seed]) for seed in seeds]
                differences = []
                for seed in seeds:
                    pairs = [(r, index[(model, scope, r["direction"], r["donor_occurrence_k"], seed, "receiver_self")])
                             for r in target if r["seed"] == seed]
                    differences.append(np.mean([r["donor_vs_receiver_sum_logodds"] - c["donor_vs_receiver_sum_logodds"] for r, c in pairs]))
                summaries.append(dict(model=model, layer=info["layer_one_based"], scope=scope, direction=direction,
                                      n=len(target), target_adoption=sum(r["greedy_donor_successor_adoption"] for r in target),
                                      self_adoption=sum(r["greedy_donor_successor_adoption"] for r in controls),
                                      self_receiver_retention=sum(r["greedy_receiver_successor_retention"] for r in controls),
                                      target_truncated=sum(r["generation_truncated"] for r in target),
                                      self_truncated=sum(r["generation_truncated"] for r in controls),
                                      adoption=interval(targets), self=interval(selves),
                                      target_minus_self=interval(np.asarray(targets) - selves),
                                      logodds_target_minus_self=interval(differences),
                                      width_min=min(r["patch_width"] for r in target), width_max=max(r["patch_width"] for r in target),
                                      equal_item_width_cells=sum(r["equal_length_complete_item"] for r in target),
                                      continuation_steps=steps(target), self_continuation_steps=steps(controls)))
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(status="COMPLETE_PAIRED_AUDIT", primary_only=args.primary_only,
                  models=list(manifest["models"]), condition_rows=len(allrows),
                  summaries=summaries, input_sha256=input_hashes, seconds=time.monotonic() - start)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    flat = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in summaries]
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    layer_text = ", ".join(f"{model} L{info['layer_one_based']}" for model, info in manifest["models"].items())
    basis = manifest.get("layer_selection_basis", "Full-cohort discovery running-index geometry")
    report = ["# Update patching at geometry-selected layers", "",
              f"Layer-selection basis: {basis}. Frozen layers: {layer_text}. "
              "This rerun uses the original ten confirmation seeds per model; it is not fresh independent confirmation.", "",
              "## Item-span continuation", "",
              "| Model / layer | Direction | Target successor | Self control | Target minus self (95% seed CI) | Target truncated |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in summaries:
        if r["scope"] == "item_span" and r["direction"] != "both":
            effect = r["target_minus_self"]
            lo, hi = effect["ci95"]
            report.append(f"| {r['model']} L{r['layer']} | {r['direction']} | {r['target_adoption']}/{r['n']} | "
                          f"{r['self_adoption']}/{r['n']} | {100*effect['estimate']:.1f} pp [{100*lo:.1f}, {100*hi:.1f}] | "
                          f"{r['target_truncated']}/{r['n']} |")
    report += ["", "### Conditional subsequent steps", "",
               "Each denominator requires all earlier steps to succeed and another item to remain. "
               "These are follow-ups of the same generated continuations.", "",
               "| Model | Direction | k+1 | k+2 | k+3 | k+4 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in summaries:
        if r["scope"] == "item_span" and r["direction"] != "both":
            cells = " | ".join(f"{s['success']}/{s['eligible']}" for s in r["continuation_steps"])
            report.append(f"| {r['model']} | {r['direction']} | {cells} |")
    if not args.primary_only:
        report += ["", "## Scope controls at the same layer", "",
                   "The table pools both directions and the three k values (60 directed cells per scope and model). "
                   "All scopes use the same directed pairs, absolute endpoints and intervention layer.", "",
                   "| Model / layer | Scope | Actual width | Target successor | Self control |", "| --- | --- | --- | --- | --- |"]
        for r in summaries:
            if r["direction"] == "both":
                report.append(f"| {r['model']} L{r['layer']} | {r['scope']} | {r['width_min']}–{r['width_max']} | "
                              f"{r['target_adoption']}/{r['n']} | {r['self_adoption']}/{r['n']} |")
    if all((base / "geometry" / model / "metrics.json").exists() for model in manifest["models"]):
        report += ["", "## Readout on the no-index cohorts", "",
                   "The layer stays fixed. Standardization, whitened PCA16 and both classifiers fit 200 discovery states "
                   "from 20 seeds and evaluate 100 confirmation states from ten seeds; labels k=1–10 are balanced.", "",
                   "| Model / layer | NCC | Logistic |", "| --- | --- | --- |"]
        for model in manifest["models"]:
            m = json.loads((base / "geometry" / model / "metrics.json").read_text())
            report.append(f"| {model} L{m['layer_one_based']} | {100*m['confirmation_ncc_balanced_accuracy']:.1f}% | "
                          f"{100*m['confirmation_logistic_balanced_accuracy']:.1f}% |")
    report += ["", "## Interpretation and limits", "",
               "- The layer choice is independent of the new patching outcomes. Original confirmation seeds are reused.",
               "- Qwen uses natural no-index traces; Gemma uses its separately prompted FOUND format. Populations are model-specific.",
               "- Each item-span intervention uses the largest endpoint-aligned suffix fitting inside both items. "
               "It transfers progress, city identity and syntax together, and does not isolate an arithmetic update operator.",
               "- Scope comparisons now hold layer fixed. Width and perturbation norm still differ by construction.",
               "- Geometry uses the original traces. Patching aligns absolute positions by deleting ordinary prompt filler; "
               "this is not an identical hidden-state context.",
               "- All failures and truncated generations remain in the reported denominators. "
               "Intervals resample the ten seed clusters with 10,000 draws and random seed 20260914.", ""]
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], condition_rows=len(allrows), summaries=flat)))


if __name__ == "__main__":
    main()
