from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = "realistic_niah_v4_4_ablation_seed_extrapolation_audit_v2"
SELECTION_SCHEMAS = {
    "realistic_niah_v4_4_ablation_seed_extrapolation_selection_v1",
    "realistic_niah_v4_4_full_span_topk_confirmation_selection_v2",
}
EXPECTED_RANDOM_BASELINE = (
    "all_heads_in_matched_layers_without_replacement_overlap_allowed"
)


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


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _implementation_fingerprint(repo_root: Path) -> str:
    relative_paths = (
        "scripts/run_realistic_niah_v4_causal_v2.py",
        "src/realistic_niah_v4/attention.py",
        "src/realistic_niah_v4/behavior.py",
        "src/realistic_niah_v4/causal_generation.py",
        "src/realistic_niah_v4/causal_v2.py",
        "src/realistic_niah_v4/correct_interventions.py",
        "src/realistic_niah_v4/correct_only_slices.py",
        "src/realistic_niah_v4/geometric_steering.py",
        "src/realistic_niah_v4/modeling.py",
        "src/realistic_niah_v4/prompts.py",
        "src/realistic_niah_v4/stimuli.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _head_count(value: Any) -> int:
    return len([item for item in str(value).split(",") if item.strip()])


def _hook_application_count(value: Any) -> int:
    payload = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty intervention_hook_applications value")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
    if isinstance(payload, dict):
        return sum(_hook_application_count(count) for count in payload.values())
    numeric = float(payload)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError(f"invalid intervention_hook_applications count: {payload!r}")
    return int(numeric)


def _row_keys(frame: pd.DataFrame) -> set[tuple[str, int, str, int]]:
    replicate = pd.to_numeric(frame["random_replicate"], errors="coerce").fillna(-1)
    return set(
        zip(
            frame["stimulus_id"].astype(str),
            pd.to_numeric(frame["top_n"], errors="raise").astype(int),
            frame["condition"].astype(str),
            replicate.astype(int),
        )
    )


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
    confirmation = selection.get("confirmation_design", {})
    expected_top_ns_by_model = {
        str(model): tuple(int(value) for value in plan.get("top_ns", ()))
        for model, plan in selection.get("models", {}).items()
    }
    expected_seeds = tuple(int(value) for value in confirmation.get("seeds", ()))
    expected_counts = tuple(int(value) for value in confirmation.get("counts", ()))
    prior_seed_end = int(
        selection.get("discovery_source", {}).get("prior_seed_end_inclusive", -1)
    )
    source_table = repo_root / str(selection["discovery_source"]["table"])
    candidate_table = repo_root / str(
        selection["discovery_source"]["candidate_summary_table"]
    )
    record(
        "campaign",
        "selection",
        "selection_schema_and_frozen_status",
        selection.get("schema_version") in SELECTION_SCHEMAS
        and selection.get("selection_status") == "frozen_before_seed_extrapolation",
        {
            "schema": selection.get("schema_version"),
            "status": selection.get("selection_status"),
        },
        {
            "schema": sorted(SELECTION_SCHEMAS),
            "status": "frozen_before_seed_extrapolation",
        },
    )
    seed_plan_ok = (
        len(expected_seeds) == 20
        and len(set(expected_seeds)) == 20
        and expected_seeds == tuple(range(expected_seeds[0], expected_seeds[0] + 20))
        and min(expected_seeds) > prior_seed_end
        and expected_counts == (1, 2, 3, 4, 5)
    )
    record(
        "campaign",
        "selection",
        "twenty_new_contiguous_seeds_and_registered_counts",
        seed_plan_ok,
        {
            "seeds": expected_seeds,
            "counts": expected_counts,
            "prior_seed_end": prior_seed_end,
        },
        {
            "seeds": "one frozen contiguous twenty-seed suffix",
            "counts": (1, 2, 3, 4, 5),
            "minimum_seed": prior_seed_end + 1,
        },
    )
    record(
        "campaign",
        "selection",
        "frozen_model_specific_doses",
        set(expected_top_ns_by_model) == {"Qwen3-8B", "Gemma4-E4B"}
        and all(
            len(values) >= 2
            and values == tuple(sorted(set(values)))
            and all(1 <= value <= 32 for value in values)
            and selection["models"][model].get("head_bank") == "broad_aggregation"
            for model, values in expected_top_ns_by_model.items()
        ),
        {
            model: selection.get("models", {}).get(model)
            for model in expected_top_ns_by_model
        },
        {model: list(values) for model, values in expected_top_ns_by_model.items()},
    )
    record(
        "campaign",
        "selection",
        "discovery_tables_are_hash_locked",
        source_table.is_file()
        and candidate_table.is_file()
        and _sha256_normalized_text(source_table)
        == selection["discovery_source"]["table_sha256"]
        and _sha256_normalized_text(candidate_table)
        == selection["discovery_source"]["candidate_summary_table_sha256"],
        {
            "source": (
                _sha256_normalized_text(source_table)
                if source_table.is_file()
                else "missing"
            ),
            "candidate": (
                _sha256_normalized_text(candidate_table)
                if candidate_table.is_file()
                else "missing"
            ),
        },
        {
            "source": selection["discovery_source"]["table_sha256"],
            "candidate": selection["discovery_source"][
                "candidate_summary_table_sha256"
            ],
        },
    )
    record(
        "campaign",
        "selection",
        "registered_random_control",
        confirmation.get("random_replicates") == 3
        and confirmation.get("random_baseline") == EXPECTED_RANDOM_BASELINE
        and confirmation.get("scope") == "answer_query",
        {
            "replicates": confirmation.get("random_replicates"),
            "baseline": confirmation.get("random_baseline"),
            "scope": confirmation.get("scope"),
        },
        {
            "replicates": 3,
            "baseline": EXPECTED_RANDOM_BASELINE,
            "scope": "answer_query",
        },
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
    expected_frozen_cells = {
        (seed, count) for seed in expected_seeds for count in range(0, 11)
    }
    record(
        "campaign",
        "dataset",
        "full_frozen_grid_contains_exact_registered_analysis_subset",
        cells == expected_frozen_cells and expected_cells.issubset(cells),
        {
            "rows": len(stimulus_rows),
            "frozen_v4_4_cells": len(cells),
            "registered_analysis_cells": len(cells.intersection(expected_cells)),
        },
        {
            "frozen_v4_4_cells": len(expected_frozen_cells),
            "registered_analysis_cells": len(expected_cells),
        },
    )

    expected_implementation = _implementation_fingerprint(repo_root)
    expected_base_config = repo_root / str(confirmation.get("base_config", ""))
    if not expected_base_config.is_file():
        raise FileNotFoundError(expected_base_config)
    expected_causal_config = repo_root / "configs/realistic_niah_v4_causal_v2.json"

    for model, ranking_path in ranking_paths.items():
        expected_top_ns = expected_top_ns_by_model[model]
        doses = len(expected_top_ns)
        rows_per_stimulus = doses * 4
        ranking = _json(ranking_path)
        broad_rows = ranking.get("rankings", {}).get("broad_aggregation", [])
        model_plan = selection["models"][model]
        record(
            model,
            "ranking",
            "hash_locked_full_span_literal_mass_order",
            ranking.get("model_label") == model
            and ranking.get("selection_split") == "discovery"
            and ranking.get("mass_definition")
            == "sum of answer-query attention over every token in each active needle span"
            and ranking.get("broad_aggregation_definition")
            == "mean(broad_mass * exp(entropy(per-needle mass))/needle_count)"
            and len(broad_rows) >= max(expected_top_ns)
            and _sha256(ranking_path)
            == str(model_plan.get("ranking_registry_sha256", "")),
            {
                "mass_definition": ranking.get("mass_definition"),
                "score": ranking.get("broad_aggregation_definition"),
                "rows": len(broad_rows),
                "sha256": _sha256(ranking_path),
            },
            "hash-locked full-span registry",
        )
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
            "single_complete_seed_extrapolation_stage",
            len(candidates) == 1,
            len(candidates),
            1,
        )
        if len(candidates) != 1:
            continue
        stage_root = candidates[0].parent
        complete = _json(candidates[0])
        design_path = stage_root / "design.json"
        design = _json(design_path)
        design_hash = _json_hash(design)
        record(
            model,
            "design",
            "design_hash_and_stage_name",
            complete.get("design_hash") == design_hash
            and stage_root.name == f"confirmation_{design_hash}",
            {
                "complete": complete.get("design_hash"),
                "stage": stage_root.name,
            },
            {"hash": design_hash, "stage": f"confirmation_{design_hash}"},
        )
        record(
            model,
            "design",
            "model_specific_frozen_top_ns",
            tuple(design.get("top_ns", ())) == expected_top_ns
            and tuple(design.get("frozen_model_specific_top_ns", ()))
            == expected_top_ns
            and tuple(complete.get("top_ns", ())) == expected_top_ns
            and "frozen_model_specific_top_n" not in design,
            {
                "design": design.get("top_ns"),
                "frozen": design.get("frozen_model_specific_top_ns"),
                "complete": complete.get("top_ns"),
            },
            list(expected_top_ns),
        )
        record(
            model,
            "design",
            "frozen_bank_selection_and_control",
            design.get("head_banks") == ["broad_aggregation"]
            and complete.get("head_banks") == ["broad_aggregation"]
            and design.get("selection_status")
            == "frozen_before_seed_extrapolation"
            and design.get("random_baseline") == EXPECTED_RANDOM_BASELINE
            and int(design.get("random_replicates", 0)) == 3,
            {
                "bank": design.get("head_banks"),
                "status": design.get("selection_status"),
                "control": design.get("random_baseline"),
                "replicates": design.get("random_replicates"),
            },
            {
                "bank": ["broad_aggregation"],
                "status": "frozen_before_seed_extrapolation",
                "control": EXPECTED_RANDOM_BASELINE,
                "replicates": 3,
            },
        )
        provenance_ok = (
            design.get("selection_sha256") == selection_sha
            and complete.get("selection_sha256") == selection_sha
            and design.get("rankings_sha256") == _sha256(ranking_path)
            and design.get("stimuli_sha256") == _sha256(stimuli)
            and design.get("base_config_sha256") == _sha256(expected_base_config)
            and design.get("causal_config_sha256") == _sha256(expected_causal_config)
            and design.get("implementation_sha256") == expected_implementation
        )
        record(
            model,
            "design",
            "hash_locked_provenance",
            provenance_ok,
            {
                "selection": design.get("selection_sha256"),
                "rankings": design.get("rankings_sha256"),
                "stimuli": design.get("stimuli_sha256"),
                "base_config": design.get("base_config_sha256"),
                "causal_config": design.get("causal_config_sha256"),
                "implementation": design.get("implementation_sha256"),
            },
            "all hashes match the supplied immutable inputs and current implementation",
        )
        record(
            model,
            "design",
            "exact_seeds_and_counts",
            tuple(design.get("seeds", ())) == expected_seeds
            and tuple(design.get("counts", ())) == expected_counts
            and tuple(complete.get("seeds", ())) == expected_seeds
            and tuple(complete.get("counts", ())) == expected_counts,
            {
                "design_seeds": design.get("seeds"),
                "design_counts": design.get("counts"),
                "complete_seeds": complete.get("seeds"),
                "complete_counts": complete.get("counts"),
            },
            {"seeds": expected_seeds, "counts": expected_counts},
        )

        baseline_path = stage_root / "clean_baseline" / "generation_labels.csv"
        baseline = pd.read_csv(baseline_path, low_memory=False)
        baseline_cells = set(
            zip(
                pd.to_numeric(baseline["seed"], errors="raise").astype(int),
                pd.to_numeric(baseline["gold_count"], errors="raise").astype(int),
            )
        )
        baseline_ok = (
            len(baseline) == 100
            and baseline["stimulus_id"].is_unique
            and baseline_cells == expected_cells
        )
        record(
            model,
            "baseline",
            "fresh_clean_baseline_grid",
            baseline_ok,
            {"rows": len(baseline), "cells": len(baseline_cells)},
            {"rows": 100, "cells": 100},
        )
        record(
            model,
            "baseline",
            "baseline_hash_and_count_recorded",
            complete.get("baseline_labels_sha256") == _sha256(baseline_path)
            and int(complete.get("baseline_rows", -1)) == 100,
            {
                "hash": complete.get("baseline_labels_sha256"),
                "rows": complete.get("baseline_rows"),
            },
            {"hash": _sha256(baseline_path), "rows": 100},
        )

        capture_root = stage_root / "capture"
        index_rows = _jsonl(capture_root / "capture_index.jsonl")
        shard_cells = {
            (int(row["seed"]), int(row["count"])) for row in index_rows
        }
        shard_hashes_ok = all(
            (capture_root / row["shard_path"]).is_file()
            and _sha256(capture_root / row["shard_path"]) == row["sha256"]
            for row in index_rows
        )
        capture_ok = (
            len(index_rows) == 100
            and shard_cells == expected_cells
            and all(int(row["rows"]) == rows_per_stimulus for row in index_rows)
            and shard_hashes_ok
        )
        record(
            model,
            "capture",
            "one_hundred_complete_eight_row_shards",
            capture_ok,
            {
                "shards": len(index_rows),
                "cells": len(shard_cells),
                "row_counts": sorted({int(row["rows"]) for row in index_rows}),
                "hashes_ok": shard_hashes_ok,
            },
            {"shards": 100, "cells": 100, "row_counts": [rows_per_stimulus], "hashes_ok": True},
        )

        detail_path = stage_root / "detail.csv.gz"
        detail = pd.read_csv(detail_path, compression="gzip", low_memory=False)
        required_columns = {
            "stimulus_id",
            "seed",
            "gold_count",
            "model_label",
            "head_bank",
            "top_n",
            "condition",
            "random_replicate",
            "heads",
            "ranked_random_head_overlap",
            "baseline_is_correct",
            "baseline_format_valid",
            "patched_is_correct",
            "patched_format_valid",
            "accuracy_delta",
            "absolute_error_delta",
            "generated_count_shift",
            "prediction_changed",
            "intervention_hook_applications",
            "evidence_split",
        }
        missing_columns = sorted(required_columns - set(detail.columns))
        ranked = detail[detail.get("condition", pd.Series(dtype=str)).astype(str).eq("ranked")]
        random = detail[
            detail.get("condition", pd.Series(dtype=str))
            .astype(str)
            .eq("layer_matched_random")
        ]
        identity = ["stimulus_id", "head_bank", "top_n"]
        random_reps = (
            random.groupby(identity)["random_replicate"]
            .apply(lambda values: tuple(sorted(pd.to_numeric(values).astype(int))))
            .tolist()
            if not missing_columns
            else []
        )
        detail_ok = (
            not missing_columns
            and len(detail) == 100 * rows_per_stimulus
            and len(ranked) == 100 * doses
            and len(random) == 300 * doses
            and not ranked.duplicated(identity).any()
            and not random.duplicated([*identity, "random_replicate"]).any()
            and len(random_reps) == 100 * doses
            and all(replicates == (0, 1, 2) for replicates in random_reps)
            and tuple(sorted(set(pd.to_numeric(detail["top_n"]).astype(int))))
            == expected_top_ns
            and set(detail["head_bank"].astype(str)) == {"broad_aggregation"}
            and set(detail["model_label"].astype(str)) == {model}
            and set(detail["evidence_split"].astype(str))
            == {"independent_seed_extrapolation"}
            and set(detail["stimulus_id"].astype(str))
            == set(baseline["stimulus_id"].astype(str))
        )
        record(
            model,
            "detail",
            "paired_ranked_and_three_random_controls_for_both_doses",
            detail_ok,
            {
                "missing_columns": missing_columns,
                "rows": len(detail),
                "ranked": len(ranked),
                "random": len(random),
                "top_ns": sorted(set(pd.to_numeric(detail["top_n"]).astype(int))),
                "random_identity_groups": len(random_reps),
            },
            {
                "missing_columns": [],
                "rows": 100 * rows_per_stimulus,
                "ranked": 100 * doses,
                "random": 300 * doses,
                "top_ns": list(expected_top_ns),
                "random_identity_groups": 100 * doses,
            },
        )
        top_n_values = pd.to_numeric(detail["top_n"], errors="raise").astype(int)
        head_counts = detail["heads"].map(_head_count)
        overlap = pd.to_numeric(
            detail["ranked_random_head_overlap"], errors="raise"
        ).astype(int)
        intervention_ok = (
            (head_counts == top_n_values).all()
            and (overlap[ranked.index] == top_n_values[ranked.index]).all()
            and (overlap[random.index] >= 0).all()
            and (overlap[random.index] <= top_n_values[random.index]).all()
            and (
                detail["intervention_hook_applications"].map(_hook_application_count)
                > 0
            ).all()
        )
        record(
            model,
            "detail",
            "head_counts_overlap_and_hook_applications",
            intervention_ok,
            {
                "head_counts": sorted(set(head_counts)),
                "random_overlap_min": int(overlap[random.index].min()),
                "random_overlap_max": int(overlap[random.index].max()),
                "hook_min": int(
                    detail["intervention_hook_applications"]
                    .map(_hook_application_count)
                    .min()
                ),
            },
            {
                "head_count_equals_row_top_n": True,
                "random_overlap_within_0_and_top_n": True,
                "hook_minimum": 1,
            },
        )

        detail_baseline_correct = _as_bool(detail["baseline_is_correct"])
        detail_baseline_valid = _as_bool(detail["baseline_format_valid"])
        expected_correct = detail["stimulus_id"].map(
            _as_bool(baseline.set_index("stimulus_id")["is_correct"])
            if "is_correct" in baseline.columns
            else _as_bool(baseline.set_index("stimulus_id")["baseline_is_correct"])
        )
        expected_valid = detail["stimulus_id"].map(
            _as_bool(baseline.set_index("stimulus_id")["format_valid"])
            if "format_valid" in baseline.columns
            else _as_bool(baseline.set_index("stimulus_id")["baseline_format_valid"])
        )
        baseline_alignment_ok = (
            not expected_correct.isna().any()
            and not expected_valid.isna().any()
            and (detail_baseline_correct.to_numpy() == expected_correct.to_numpy()).all()
            and (detail_baseline_valid.to_numpy() == expected_valid.to_numpy()).all()
        )
        record(
            model,
            "detail",
            "baseline_labels_propagate_exactly_to_interventions",
            baseline_alignment_ok,
            {
                "detail_correct_rows": int(detail_baseline_correct.sum()),
                "detail_valid_rows": int(detail_baseline_valid.sum()),
            },
            "each detail row matches its fresh clean baseline label",
        )

        clean_path = (
            stage_root
            / "analysis"
            / "detail.clean_correct.seed_extrapolation.csv.gz"
        )
        clean = pd.read_csv(clean_path, compression="gzip", low_memory=False)
        expected_clean = detail.loc[
            detail_baseline_correct & detail_baseline_valid
        ].reset_index(drop=True)
        clean_group_sizes = clean.groupby("stimulus_id").size()
        clean_ok = (
            len(clean) == len(expected_clean)
            and _row_keys(clean) == _row_keys(expected_clean)
            and _as_bool(clean["baseline_is_correct"]).all()
            and _as_bool(clean["baseline_format_valid"]).all()
            and not clean_group_sizes.empty
            and set(clean_group_sizes.astype(int)) == {rows_per_stimulus}
            and int(complete.get("clean_correct_rows", -1)) == len(clean)
        )
        record(
            model,
            "clean_correct",
            "clean_correct_is_exact_baseline_correct_subset",
            clean_ok,
            {
                "rows": len(clean),
                "expected_rows": len(expected_clean),
                "stimuli": int(clean["stimulus_id"].nunique()),
                "rows_per_stimulus": sorted(set(clean_group_sizes.astype(int))),
                "complete_rows": complete.get("clean_correct_rows"),
            },
            {
                "same_row_keys": True,
                "rows_per_stimulus": [rows_per_stimulus],
                "complete_rows_matches": True,
            },
        )

        dual_path = (
            stage_root
            / "analysis"
            / "dual_population_seed_extrapolation_summary.csv"
        )
        dual = pd.read_csv(dual_path, low_memory=False)
        expected_pairs = {
            (population, top_n)
            for population in ("all_examples_signed", "clean_correct_only")
            for top_n in expected_top_ns
        }
        observed_pairs = set(
            zip(
                dual["analysis_population"].astype(str),
                pd.to_numeric(dual["top_n"], errors="raise").astype(int),
            )
        )
        correct_examples = int(clean["stimulus_id"].nunique())
        correct_seed_clusters = int(clean["seed"].nunique())
        summary_counts_ok = True
        for row in dual.to_dict(orient="records"):
            if row["analysis_population"] == "all_examples_signed":
                summary_counts_ok &= int(row["examples"]) == 100
                summary_counts_ok &= int(row["seed_clusters"]) == 20
                summary_counts_ok &= (
                    row["primary_metric"]
                    == "ranked_minus_random_absolute_count_shift"
                )
            else:
                summary_counts_ok &= int(row["examples"]) == correct_examples
                summary_counts_ok &= (
                    int(row["seed_clusters"]) == correct_seed_clusters
                )
                summary_counts_ok &= (
                    row["primary_metric"]
                    == "ranked_minus_random_correct_to_wrong"
                )
        finite_columns = [
            "primary_effect",
            "primary_effect_ci95_low",
            "primary_effect_ci95_high",
        ]
        dual_ok = (
            len(dual) == 2 * doses
            and observed_pairs == expected_pairs
            and set(dual["model_label"].astype(str)) == {model}
            and set(dual["head_bank"].astype(str)) == {"broad_aggregation"}
            and set(dual["selection_status"].astype(str))
            == {"frozen_before_seed_extrapolation"}
            and set(dual["evidence_split"].astype(str))
            == {"independent_seed_extrapolation"}
            and set(pd.to_numeric(dual["random_replicates_min"]).astype(int)) == {3}
            and set(pd.to_numeric(dual["bootstrap_repetitions"]).astype(int))
            == {10_000}
            and dual[finite_columns]
            .apply(pd.to_numeric, errors="coerce")
            .applymap(math.isfinite)
            .all()
            .all()
            and (
                pd.to_numeric(dual["primary_effect_ci95_low"])
                <= pd.to_numeric(dual["primary_effect_ci95_high"])
            ).all()
            and summary_counts_ok
        )
        record(
            model,
            "statistics",
            "frozen_dual_population_estimates_for_every_dose",
            dual_ok,
            {
                "rows": len(dual),
                "pairs": sorted(observed_pairs),
                "all_examples": 100,
                "clean_correct_examples": correct_examples,
                "clean_correct_seed_clusters": correct_seed_clusters,
            },
            {
                "rows": 2 * doses,
                "pairs": sorted(expected_pairs),
                "all_examples": 100,
                "clean_correct_examples": correct_examples,
                "clean_correct_seed_clusters": correct_seed_clusters,
            },
        )

        seed_effects = pd.read_csv(
            stage_root / "analysis" / "head_ablation_seed_effects.csv",
            low_memory=False,
        )
        statistics = pd.read_csv(
            stage_root
            / "analysis"
            / "head_ablation_confirmation_statistics.csv",
            low_memory=False,
        )
        seed_group_sizes = seed_effects.groupby("top_n").size()
        stats_ok = (
            len(seed_effects) == 20 * doses
            and set(pd.to_numeric(seed_effects["seed"]).astype(int))
            == set(expected_seeds)
            and set(pd.to_numeric(seed_effects["examples"]).astype(int)) == {5}
            and set(pd.to_numeric(seed_group_sizes).astype(int)) == {20}
            and len(statistics) == 3 * doses
            and set(statistics["metric"].astype(str))
            == {"accuracy_delta", "absolute_error_delta", "prediction_changed"}
            and tuple(
                sorted(set(pd.to_numeric(statistics["top_n"]).astype(int)))
            )
            == expected_top_ns
            and set(pd.to_numeric(statistics["seeds"]).astype(int)) == {20}
            and statistics["ranked_minus_random_mean"].map(math.isfinite).all()
            and statistics["ci95_low"].map(math.isfinite).all()
            and statistics["ci95_high"].map(math.isfinite).all()
            and statistics["exact_sign_flip_p"].between(0.0, 1.0).all()
        )
        record(
            model,
            "statistics",
            "twenty_seed_companion_statistics_for_each_dose",
            stats_ok,
            {
                "seed_rows": len(seed_effects),
                "seed_groups": seed_group_sizes.to_dict(),
                "statistic_rows": len(statistics),
                "metrics": sorted(set(statistics["metric"].astype(str))),
            },
            {
                "seed_rows": 20 * doses,
                "seed_groups": {str(value): 20 for value in expected_top_ns},
                "statistic_rows": 3 * doses,
                "metrics": [
                    "absolute_error_delta",
                    "accuracy_delta",
                    "prediction_changed",
                ],
            },
        )

    table = pd.DataFrame(checks)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks_path = output_dir / "ablation_seed_extrapolation_audit_checks.csv"
    summary_path = output_dir / "ablation_seed_extrapolation_audit.json"
    table.to_csv(checks_path, index=False)
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
        "checks_path": str(checks_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not errors.empty:
        raise RuntimeError(
            "Ablation seed-extrapolation audit failed: "
            + ", ".join(errors["check"].astype(str).tolist())
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict audit for frozen model-specific V4.4 ablation seed "
            "extrapolation."
        )
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
        else run_root / "audit" / "ablation_seed_extrapolation"
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
