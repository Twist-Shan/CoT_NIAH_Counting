from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = "realistic_niah_v4_causal_v2_ablation_confirmation_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_normalized_text(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(
    *,
    run_root: Path,
    repo_root: Path,
    stimuli: Path,
    selection_path: Path,
    ranking_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(
        model: str,
        category: str,
        name: str,
        passed: bool,
        observed: Any,
        expected: Any,
    ) -> None:
        checks.append(
            {
                "model_label": model,
                "category": category,
                "check": name,
                "passed": bool(passed),
                "observed": json.dumps(observed, sort_keys=True, default=str),
                "expected": json.dumps(expected, sort_keys=True, default=str),
            }
        )

    selection = _json(selection_path)
    selection_sha = _sha256(selection_path)
    expected_seeds = tuple(
        int(value) for value in selection["confirmation_design"]["seeds"]
    )
    expected_counts = tuple(
        int(value) for value in selection["confirmation_design"]["counts"]
    )
    discovery_seeds = tuple(
        int(value) for value in selection["discovery_source"]["screen_seeds"]
    )
    source_table = repo_root / selection["discovery_source"]["table"]
    record(
        "campaign",
        "selection",
        "selection_frozen_before_confirmation",
        selection.get("selection_status") == "frozen_before_confirmation",
        selection.get("selection_status"),
        "frozen_before_confirmation",
    )
    record(
        "campaign",
        "selection",
        "confirmation_seeds_are_new",
        not set(expected_seeds).intersection(discovery_seeds),
        sorted(set(expected_seeds).intersection(discovery_seeds)),
        [],
    )
    record(
        "campaign",
        "selection",
        "discovery_table_hash",
        source_table.is_file()
        and _sha256_normalized_text(source_table)
        == selection["discovery_source"]["table_sha256"],
        _sha256_normalized_text(source_table) if source_table.is_file() else "missing",
        selection["discovery_source"]["table_sha256"],
    )
    stimulus_rows = _jsonl(stimuli)
    cells = {
        (int(row["seed"]), int(row["gold_count"]))
        for row in stimulus_rows
        if row.get("design_variant") == "v4.4"
    }
    expected_cells = {
        (seed, count) for seed in expected_seeds for count in expected_counts
    }
    record(
        "campaign",
        "dataset",
        "independent_confirmation_grid",
        expected_cells.issubset(cells),
        len(cells.intersection(expected_cells)),
        len(expected_cells),
    )

    for model, ranking_path in ranking_paths.items():
        expected_top_n = int(selection["models"][model]["top_n"])
        family = (
            run_root
            / model
            / "numeric"
            / "causal_v2"
            / "answer_query_head_ablation"
        )
        candidates = sorted(family.glob("confirmation_*/complete.json"))
        record(
            model,
            "stage",
            "single_complete_confirmation_stage",
            len(candidates) == 1,
            len(candidates),
            1,
        )
        if len(candidates) != 1:
            continue
        stage_root = candidates[0].parent
        complete = _json(candidates[0])
        design = _json(stage_root / "design.json")
        record(
            model,
            "design",
            "model_specific_frozen_top_n",
            design.get("top_ns") == [expected_top_n]
            and complete.get("top_ns") == [expected_top_n],
            {"design": design.get("top_ns"), "complete": complete.get("top_ns")},
            [expected_top_n],
        )
        record(
            model,
            "design",
            "frozen_broad_bank_only",
            design.get("head_banks") == ["broad_aggregation"]
            and complete.get("head_banks") == ["broad_aggregation"],
            {
                "design": design.get("head_banks"),
                "complete": complete.get("head_banks"),
            },
            ["broad_aggregation"],
        )
        record(
            model,
            "design",
            "selection_hash",
            design.get("selection_sha256") == selection_sha
            and complete.get("selection_sha256") == selection_sha,
            {
                "design": design.get("selection_sha256"),
                "complete": complete.get("selection_sha256"),
            },
            selection_sha,
        )
        record(
            model,
            "design",
            "ranking_hash",
            design.get("rankings_sha256") == _sha256(ranking_path),
            design.get("rankings_sha256"),
            _sha256(ranking_path),
        )
        record(
            model,
            "design",
            "seeds_and_counts",
            tuple(complete.get("seeds", ())) == expected_seeds
            and tuple(complete.get("counts", ())) == expected_counts,
            {"seeds": complete.get("seeds"), "counts": complete.get("counts")},
            {"seeds": expected_seeds, "counts": expected_counts},
        )

        baseline_path = stage_root / "clean_baseline" / "generation_labels.csv"
        baseline = pd.read_csv(baseline_path)
        baseline_cells = set(
            zip(
                pd.to_numeric(baseline["seed"]).astype(int),
                pd.to_numeric(baseline["gold_count"]).astype(int),
            )
        )
        record(
            model,
            "baseline",
            "fresh_clean_baseline_grid",
            len(baseline) == 40
            and baseline["stimulus_id"].is_unique
            and baseline_cells == expected_cells,
            {"rows": len(baseline), "cells": len(baseline_cells)},
            {"rows": 40, "cells": 40},
        )
        record(
            model,
            "baseline",
            "baseline_hash_recorded",
            complete.get("baseline_labels_sha256") == _sha256(baseline_path),
            complete.get("baseline_labels_sha256"),
            _sha256(baseline_path),
        )

        capture_root = stage_root / "capture"
        index_rows = _jsonl(capture_root / "capture_index.jsonl")
        shard_hashes_ok = all(
            (capture_root / row["shard_path"]).is_file()
            and _sha256(capture_root / row["shard_path"]) == row["sha256"]
            for row in index_rows
        )
        record(
            model,
            "capture",
            "forty_complete_shards",
            len(index_rows) == 40
            and all(int(row["rows"]) == 4 for row in index_rows)
            and shard_hashes_ok,
            {
                "shards": len(index_rows),
                "rows": sorted({int(row["rows"]) for row in index_rows}),
                "hashes_ok": shard_hashes_ok,
            },
            {"shards": 40, "rows": [4], "hashes_ok": True},
        )

        detail = pd.read_csv(stage_root / "detail.csv.gz", compression="gzip")
        ranked = detail[detail["condition"].astype(str).eq("ranked")]
        random = detail[
            detail["condition"].astype(str).eq("layer_matched_random")
        ]
        identity = ["stimulus_id", "head_bank", "top_n"]
        ranked_unique = not ranked.duplicated(identity).any()
        random_reps = (
            random.groupby(identity)["random_replicate"]
            .apply(lambda series: tuple(sorted(pd.to_numeric(series).astype(int))))
            .tolist()
        )
        detail_ok = (
            len(detail) == 160
            and len(ranked) == 40
            and len(random) == 120
            and ranked_unique
            and all(reps == (0, 1, 2) for reps in random_reps)
            and set(pd.to_numeric(detail["top_n"]).astype(int)) == {expected_top_n}
            and set(detail["head_bank"].astype(str)) == {"broad_aggregation"}
            and set(detail["evidence_split"].astype(str))
            == {"independent_ablation_confirmation"}
        )
        record(
            model,
            "detail",
            "paired_ranked_and_three_random_controls",
            detail_ok,
            {
                "rows": len(detail),
                "ranked": len(ranked),
                "random": len(random),
                "top_n": sorted(set(pd.to_numeric(detail["top_n"]).astype(int))),
            },
            {
                "rows": 160,
                "ranked": 40,
                "random": 120,
                "top_n": [expected_top_n],
            },
        )
        head_counts = detail["heads"].astype(str).map(
            lambda value: len([item for item in value.split(",") if item])
        )
        record(
            model,
            "detail",
            "every_intervention_has_frozen_bank_size",
            set(head_counts) == {expected_top_n},
            sorted(set(head_counts)),
            [expected_top_n],
        )

        seed_effects = pd.read_csv(
            stage_root / "analysis" / "head_ablation_seed_effects.csv"
        )
        statistics = pd.read_csv(
            stage_root / "analysis" / "head_ablation_confirmation_statistics.csv"
        )
        stats_ok = (
            len(seed_effects) == 10
            and set(pd.to_numeric(seed_effects["seed"]).astype(int))
            == set(expected_seeds)
            and set(pd.to_numeric(seed_effects["examples"]).astype(int)) == {4}
            and len(statistics) == 3
            and set(statistics["metric"].astype(str))
            == {"accuracy_delta", "absolute_error_delta", "prediction_changed"}
            and set(pd.to_numeric(statistics["seeds"]).astype(int)) == {10}
            and statistics["ranked_minus_random_mean"].map(math.isfinite).all()
            and statistics["exact_sign_flip_p"].between(0.0, 1.0).all()
        )
        record(
            model,
            "statistics",
            "ten_seed_cluster_statistics",
            stats_ok,
            {
                "seed_rows": len(seed_effects),
                "statistic_rows": len(statistics),
                "metrics": sorted(set(statistics["metric"].astype(str))),
            },
            {
                "seed_rows": 10,
                "statistic_rows": 3,
                "metrics": [
                    "absolute_error_delta",
                    "accuracy_delta",
                    "prediction_changed",
                ],
            },
        )

    table = pd.DataFrame(checks)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ablation_confirmation_audit_checks.csv", index=False)
    errors = table[~table["passed"]]
    summary = {
        "schema_version": SCHEMA,
        "status": "passed" if errors.empty else "failed",
        "checks": int(len(table)),
        "passed": int(table["passed"].sum()),
        "errors": int(len(errors)),
        "models": sorted(ranking_paths),
        "selection_sha256": selection_sha,
        "run_root": str(run_root),
    }
    (output_dir / "ablation_confirmation_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not errors.empty:
        raise RuntimeError(
            "Ablation confirmation audit failed: "
            + ", ".join(errors["check"].astype(str).tolist())
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict audit for frozen model-specific head-ablation confirmation."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--qwen-rankings", required=True)
    parser.add_argument("--gemma-rankings", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "audit" / "ablation_confirmation"
    )
    summary = audit(
        run_root=run_root,
        repo_root=Path(args.repo_root).resolve(),
        stimuli=Path(args.stimuli).resolve(),
        selection_path=Path(args.selection_json).resolve(),
        ranking_paths={
            "Qwen3-8B": Path(args.qwen_rankings).resolve(),
            "Gemma4-E4B": Path(args.gemma_rankings).resolve(),
        },
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
