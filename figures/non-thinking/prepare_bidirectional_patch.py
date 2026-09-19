"""Prepare full-layer restore/corrupt effects on a common true-count scale.

Inputs must be the audited historical restoration_effects.csv and the completed
reverse run's data directory (model/prompts/seed_*_count_*.jsonl). This script
does not run models. It refuses sparse validation curves and incomplete grids.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json

import numpy as np

MODELS = {"Qwen3-8B": 36, "Gemma4-E4B": 42}
SEEDS = range(1254, 1264)
COUNTS = range(1, 11)


def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restoration-effects", required=True, type=Path)
    parser.add_argument("--reverse-data", type=Path)
    parser.add_argument("--reverse-effects", type=Path,
                        help="Compact per-input reverse CSV extracted from verified journals")
    parser.add_argument("--reverse-source-audit", type=Path)
    parser.add_argument("--restoration-only", action="store_true",
                        help="Prepare the historical restore arm while reverse records are unavailable")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    if not args.restoration_only and args.reverse_data is None and args.reverse_effects is None:
        parser.error("--reverse-data or --reverse-effects is required unless --restoration-only is specified")
    if args.reverse_effects and (args.reverse_data or not args.reverse_source_audit):
        parser.error("Compact reverse input requires --reverse-source-audit and excludes --reverse-data")
    directions = ("restore",) if args.restoration_only else ("restore", "corrupt")
    prefix = "restoration" if args.restoration_only else "bidirectional_patch"
    values = {}
    source_hashes = {}

    def record(path):
        source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def add(model, seed, count, layer, direction, value):
        key = (model, seed, count, layer, direction)
        if key in values:
            raise ValueError(f"Duplicate measurement: {key}")
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite measurement: {key}")
        values[key] = float(value)

    record(args.restoration_effects)
    for row in read_csv(args.restoration_effects):
        model, seed = row["model_label"], int(row["seed"])
        if model not in MODELS or seed not in SEEDS or row["patch_kind"] != "needle_full":
            continue
        count, layer = int(row["gold_count"]), int(row["patch_layer"])
        effect = abs(float(row["corrupt_expected_count"]) - count) - abs(float(row["expected_count"]) - count)
        if abs(effect - float(row["expected_absolute_error_reduction"])) > 1e-7:
            raise ValueError("Historical restoration metric disagrees with its baseline")
        add(model, seed, count, layer, "restore", effect)

    if args.reverse_effects and not args.restoration_only:
        record(args.reverse_effects)
        record(args.reverse_source_audit)
        audit = json.loads(args.reverse_source_audit.read_text())
        if (audit['status'] != 'PASS' or audit['source_run_audit']['total_rows'] != 76800
                or audit['source_run_audit']['status'] != 'PASS'
                or audit['csv_sha256'] != source_hashes[str(args.reverse_effects.resolve())]
                or audit['source_journal_count'] != 200 or audit['subset_rows'] != 7800):
            raise ValueError('Compact reverse source audit failed')
        for row in read_csv(args.reverse_effects):
            model, seed = row['model_label'], int(row['seed'])
            count, layer = int(row['gold_count']), int(row['patch_layer'])
            effect = abs(float(row['expected_count']) - count) - abs(float(row['clean_expected_count']) - count)
            if abs(effect - float(row['expected_error_increase'])) > 1e-7:
                raise ValueError('Compact reverse metric disagrees with its baseline')
            add(model, seed, count, layer, 'corrupt', effect)

    for model, n_layers in ([] if args.restoration_only or args.reverse_effects else MODELS.items()):
        for seed in SEEDS:
            for count in COUNTS:
                path = args.reverse_data / model / "prompts" / f"seed_{seed}_count_{count}.jsonl"
                record(path)
                marker_path = path.with_suffix(".complete.json")
                marker = json.loads(marker_path.read_text())
                if marker.get("status") != "PASS" or marker.get("journal_sha256") != source_hashes[str(path.resolve())]:
                    raise ValueError(f"Missing or mismatched completion audit: {path}")
                record(marker_path)
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if any(r.get("cell_audit") != "PASS" for r in rows):
                    raise ValueError(f"Unvalidated reverse cell: {path}")
                clean = [r for r in rows if r["condition"] == "clean" and int(r["patch_layer"]) == -1]
                if len(clean) != 1:
                    raise ValueError(f"Missing or duplicate clean baseline: {path}")
                for row in rows:
                    if row["condition"] != "reverse_needle_full":
                        continue
                    if int(row["seed"]) != seed or int(row["gold_count"]) != count:
                        raise ValueError(f"Journal identity mismatch: {path}")
                    effect = abs(float(row["expected_count"]) - count) - abs(float(clean[0]["expected_count"]) - count)
                    if abs(effect - float(row["expected_error_increase"])) > 1e-7:
                        raise ValueError(f"Reverse metric disagrees with its baseline: {path}")
                    add(model, seed, count, int(row["patch_layer"]), "corrupt", effect)

    expected = {(m, s, n, layer, direction)
                for m, layers in MODELS.items() for s in SEEDS for n in COUNTS
                for layer in range(layers) for direction in directions}
    if set(values) != expected:
        raise ValueError(f"Full matched grid required: {len(expected - set(values))} missing, "
                         f"{len(set(values) - expected)} extra cells")

    # Raw reverse means must reproduce the exact full-run source already delivered.
    reference = [] if args.restoration_only else read_csv(Path(__file__).resolve().parent / "data/reverse_layer_summary.csv")
    for row in reference:
        if row["population"] != "confirmation" or row["condition"] != "reverse_needle_full" or row["metric"] != "expected_error_increase":
            continue
        mean = np.mean([values[row["model"], s, n, int(row["layer"]), "corrupt"]
                        for s in SEEDS for n in COUNTS])
        if abs(mean - float(row["mean"])) > 1e-10:
            raise ValueError("Reverse records do not reproduce the verified full-run mean")

    summary, seed_rows, paired_rows = [], [], []
    rng = np.random.default_rng(20260909)
    draws = rng.integers(0, len(SEEDS), (10000, len(SEEDS)))
    for model, n_layers in MODELS.items():
        for layer in range(n_layers):
            for direction in directions:
                per_seed = []
                for seed in SEEDS:
                    per_count = []
                    for count in COUNTS:
                        raw = values[model, seed, count, layer, direction]
                        effect = raw / count
                        per_count.append(effect)
                        paired_rows.append(dict(model=model, seed=seed, gold_count=count,
                                                layer=layer, direction=direction,
                                                error_change=raw, relative_effect=effect))
                    effect = float(np.mean(per_count))
                    per_seed.append(effect)
                    seed_rows.append(dict(model=model, seed=seed, layer=layer,
                                          direction=direction, effect=effect, counts=10))
                per_seed = np.array(per_seed)
                low, high = np.quantile(per_seed[draws].mean(axis=1), [.025, .975])
                summary.append(dict(model=model, layer=layer, direction=direction,
                                    mean=float(per_seed.mean()), ci95_low=float(low),
                                    ci95_high=float(high), seed_count=10))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / f"{prefix}_plot.csv", summary)
    write_csv(args.output_dir / f"{prefix}_seed_effects.csv", seed_rows)
    write_csv(args.output_dir / f"{prefix}_count_effects.csv", paired_rows)
    audit = dict(status="PASS", measurement_cells=len(values), seeds=list(SEEDS),
                 counts=list(COUNTS), num_layers=MODELS, bootstrap_repetitions=10000,
                 bootstrap_seed=20260909, normalization="Per-input error change / gold_count, then equal mean within seed and across seeds",
                 restore="(|E_corrupt-N| - |E_restored-N|) / N",
                 corrupt="(|E_patched-N| - |E_clean-N|) / N",
                 negative_effects="retained; no clipping",
                 backend_caveat="Historical Gemma restoration contains mixed SDPA-donor/eager-recipient cases; reverse uses eager/eager. Matching cohorts does not imply identical numerical backends.",
                 source_sha256=source_hashes)
    audit['directions'] = list(directions)
    (args.output_dir / f"{prefix}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cells": len(values), "curve_rows": len(summary)}))


if __name__ == "__main__":
    main()
