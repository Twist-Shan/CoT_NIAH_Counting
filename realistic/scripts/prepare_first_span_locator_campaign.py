#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("Qwen3-8B", "Gemma4-E4B")
KS = (1, 2, 4, 8, 16, 32)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capture_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("design_variant") == "v4.4"
            and row.get("split") == "discovery"
            and int(row.get("count", -1)) == 10
        ):
            rows.append(row)
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(rows) != 20 or len(seeds) != 20:
        raise RuntimeError(
            f"Expected 20 V4.4 discovery N=10 shards in {index_path}, "
            f"found rows={len(rows)} seeds={seeds}"
        )
    return rows


def score_model(index_path: Path) -> list[dict[str, Any]]:
    root = index_path.parent
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    model_labels: set[str] = set()
    for index_row in capture_rows(index_path):
        shard = root / str(index_row["shard_path"])
        with gzip.open(shard, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["design_variant"] != "v4.4" or int(row["count"]) != 10:
                    continue
                model_labels.add(row["model_label"])
                masses = np.asarray(json.loads(row["needle_span_masses"]), dtype=float)
                if masses.shape != (10,) or not np.isfinite(masses).all():
                    raise RuntimeError(
                        f"Invalid ten-span profile {row['stimulus_id']} "
                        f"L{row['layer']}H{row['head']}: {masses.shape}"
                    )
                grouped[(int(row["layer"]), int(row["head"]))].append(masses)
    if len(model_labels) != 1:
        raise RuntimeError(f"Expected one model in {index_path}, found {model_labels}")
    model = next(iter(model_labels))
    scored: list[dict[str, Any]] = []
    for (layer, head), profiles in grouped.items():
        matrix = np.stack(profiles)
        if matrix.shape[0] != 20:
            raise RuntimeError(f"{model} L{layer}H{head} has {matrix.shape[0]} rows")
        totals = matrix.sum(axis=1)
        shares = np.divide(
            matrix[:, 0], totals, out=np.full_like(totals, np.nan), where=totals > 0
        )
        profile = matrix.mean(axis=0)
        m1 = float(profile[0])
        total = float(profile.sum())
        finite_shares = shares[np.isfinite(shares)]
        scored.append(
            {
                "model": model,
                "layer": layer,
                "head": head,
                "discovery_seeds": 20,
                "first_span_mass_mean": m1,
                "ten_span_total_mass_mean": total,
                "first_share_ratio_of_means": m1 / total if total > 0 else math.nan,
                "first_share_mean_of_rows": (
                    float(finite_shares.mean()) if len(finite_shares) else math.nan
                ),
                "other_nine_span_mass_mean": float(profile[1:].mean()),
                "first_minus_other_mean": m1 - float(profile[1:].mean()),
                "span_1_mass": float(profile[0]),
                "span_2_mass": float(profile[1]),
                "span_3_mass": float(profile[2]),
                "span_4_mass": float(profile[3]),
                "span_5_mass": float(profile[4]),
                "span_6_mass": float(profile[5]),
                "span_7_mass": float(profile[6]),
                "span_8_mass": float(profile[7]),
                "span_9_mass": float(profile[8]),
                "span_10_mass": float(profile[9]),
                "ten_span_mass_profile": json.dumps(profile.tolist()),
            }
        )
    scored.sort(
        key=lambda row: (
            -float(row["first_span_mass_mean"]),
            -float(row["ten_span_total_mass_mean"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    for rank, row in enumerate(scored, start=1):
        row["first_span_absolute_mass_rank"] = rank
    return scored


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def m10_matched_controls(
    scored: list[dict[str, Any]], ranked: list[dict[str, Any]]
) -> dict[str, Any]:
    lookup = {(int(row["layer"]), int(row["head"])): row for row in scored}
    ranked_heads = [(int(row["layer"]), int(row["head"])) for row in ranked]
    plan: dict[str, Any] = {}
    for k in KS:
        selected = ranked_heads[:k]
        selected_set = set(selected)
        replicates = []
        previously_used: set[tuple[int, int]] = set()
        for replicate in range(3):
            used: set[tuple[int, int]] = set()
            matches = []
            for source in selected:
                layer = source[0]
                source_total = float(lookup[source]["ten_span_total_mass_mean"])
                candidates = [
                    key
                    for key in lookup
                    if key[0] == layer and key not in used
                ]
                if not candidates:
                    raise RuntimeError(f"No M10-matched candidate for {source}, K={k}")
                candidates.sort(
                    key=lambda key: (
                        key in selected_set,
                        key in previously_used,
                        abs(
                            math.log(max(float(lookup[key]["ten_span_total_mass_mean"]), 1e-30))
                            - math.log(max(source_total, 1e-30))
                        ),
                        key[1],
                    )
                )
                target = candidates[0]
                used.add(target)
                matches.append(
                    {
                        "ranked": {"layer": source[0], "head": source[1]},
                        "control": {"layer": target[0], "head": target[1]},
                        "ranked_M10": source_total,
                        "control_M10": float(lookup[target]["ten_span_total_mass_mean"]),
                    }
                )
            previously_used.update(used)
            replicates.append(
                {
                    "replicate": replicate,
                    "heads": [item["control"] for item in matches],
                    "matches": matches,
                    "ranked_control_overlap": len(
                        set(selected) & {
                            (int(item["control"]["layer"]), int(item["control"]["head"]))
                            for item in matches
                        }
                    ),
                }
            )
        plan[str(k)] = replicates
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--qwen-index", type=Path, required=True)
    parser.add_argument("--gemma-index", type=Path, required=True)
    parser.add_argument("--old-qwen-ranking", type=Path, required=True)
    parser.add_argument("--old-gemma-ranking", type=Path, required=True)
    parser.add_argument("--selection-template", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.campaign_root / "inputs"
    analysis = args.campaign_root / "discovery_analysis"
    inputs.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for model, index_path, old_path in (
        ("Qwen3-8B", args.qwen_index, args.old_qwen_ranking),
        ("Gemma4-E4B", args.gemma_index, args.old_gemma_ranking),
    ):
        scored = score_model(index_path)
        if scored[0]["model"] != model:
            raise RuntimeError(f"Model mismatch: expected {model}, got {scored[0]['model']}")
        all_rows.extend(scored)
        ranked = scored[:32]
        old = read_json(old_path)
        old["schema_version"] = "realistic_niah_v4_first_span_absolute_mass_v1"
        old["selection_split"] = "discovery"
        old["design_variant"] = "v4.4"
        old["mass_definition"] = (
            "answer-query attention summed over every token in each needle literal span"
        )
        old["first_locator_definition"] = (
            "rank descending by mean absolute attention mass on the complete first needle span, "
            "computed only on N=10 discovery examples"
        )
        old["rankings"]["first_locator"] = [
            {"rank": i, "layer": int(row["layer"]), "head": int(row["head"])}
            for i, row in enumerate(ranked, start=1)
        ]
        ranking_path = inputs / f"{model}.first_span_absolute_mass_rankings.json"
        write_json(ranking_path, old)
        control_path = inputs / f"{model}.M10_matched_controls.json"
        write_json(
            control_path,
            {
                "model": model,
                "matching_metric": "ten_span_total_mass_mean",
                "ranking_metric": "first_span_mass_mean",
                "sets": m10_matched_controls(scored, ranked),
            },
        )
        write_csv(analysis / f"{model}.all_head_first_span_scores.csv", scored)
        write_csv(analysis / f"{model}.top32_first_span_representation.csv", ranked)
        manifests[model] = {
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "ranking": str(ranking_path),
            "ranking_sha256": sha256(ranking_path),
            "controls": str(control_path),
            "controls_sha256": sha256(control_path),
            "heads_scored": len(scored),
            "top_head": old["rankings"]["first_locator"][0],
            "top_head_metrics": ranked[0],
        }
    write_csv(analysis / "all_models_first_span_scores.csv", all_rows)
    manifest_path = inputs / "first_span_absolute_mass_ranking_manifest.json"
    write_json(
        manifest_path,
        {
            "campaign": "v4.4_first_span_absolute_mass_answer_query_ablation",
            "query": "final Total: answer-query token",
            "discovery_population": "V4.4 N=10, seeds 1234..1253",
            "ranking_metric": "mean absolute complete-first-needle-span attention mass",
            "reported_metrics": [
                "first_span_mass_mean",
                "ten_span_total_mass_mean",
                "first_share_ratio_of_means",
                "ten_span_mass_profile",
            ],
            "models": manifests,
        },
    )
    selection = read_json(args.selection_template)
    # The causal-v2 runner intentionally whitelists this container schema.
    # The changed estimand is recorded in selection_rule/discovery_source.
    selection["schema_version"] = (
        "realistic_niah_v4_4_ablation_seed_extrapolation_selection_v1"
    )
    selection["selection_rule"].update(
        {
            "candidate_head_bank": "first_locator",
            "frozen_values_source": "N=10 full-span attention mass on discovery seeds 1234..1253",
            "interpretation": "causal necessity sweep ranked by absolute mass on the complete first needle span",
        }
    )
    selection["discovery_source"].update(
        {
            "table": str(manifest_path),
            "table_sha256": sha256(manifest_path),
            "ablation_discovery_seeds": list(range(1234, 1254)),
            "prior_seed_end_inclusive": 1253,
            "prior_seed_scope_note": "ranking uses only V4.4 N=10 discovery attention captures",
        }
    )
    for model in MODELS:
        selection["models"][model]["freeze_rationale"] = (
            "nested first-complete-span absolute-mass K sweep"
        )
    write_json(inputs / "selection_first_span_absolute_mass.json", selection)
    print(json.dumps({"manifest": str(manifest_path), "models": manifests}, indent=2))


if __name__ == "__main__":
    main()
