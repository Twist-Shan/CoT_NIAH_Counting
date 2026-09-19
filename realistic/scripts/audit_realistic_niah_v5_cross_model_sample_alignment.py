#!/usr/bin/env python3
"""Fail-closed audit for the frozen Native-thinking cross-model sample panels."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty JSONL input: {path}")
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty CSV input: {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multiset(
    rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], tuple[Any, ...]]
) -> collections.Counter[tuple[Any, ...]]:
    return collections.Counter(key(row) for row in rows)


def assert_equal_multiset(
    label: str,
    left: collections.Counter[tuple[Any, ...]],
    right: collections.Counter[tuple[Any, ...]],
) -> dict[str, Any]:
    if left != right:
        missing = list((left - right).elements())[:10]
        extra = list((right - left).elements())[:10]
        raise ValueError(
            f"{label} is not exactly sample aligned; "
            f"missing_from_gemma={missing} extra_in_gemma={extra}"
        )
    digest = hashlib.sha256(
        json.dumps(
            sorted((list(key), count) for key, count in left.items()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "PASS",
        "aligned_cells": int(sum(left.values())),
        "unique_keys": int(len(left)),
        "key_multiset_sha256": digest,
    }


def phase(seed: int) -> str:
    if 1234 <= seed <= 1253:
        return "discovery"
    if 1254 <= seed <= 1263:
        return "confirmation"
    raise ValueError(f"Seed is outside the frozen 20+10 contract: {seed}")


def audit(root: Path) -> dict[str, Any]:
    aligned = root / "work/v5_native_sample_aligned_20260829"
    evidence: dict[str, Any] = {}

    routed_dir = aligned / "shared_routed_transition_panel"
    routed_rows = {
        model: read_jsonl(routed_dir / f"{model}_anchor_panel.jsonl")
        for model in MODELS
    }
    transition_key = lambda row: (
        phase(int(row["seed"])),
        int(row["seed"]),
        int(row["gold_count"]),
        int(row["from_occurrence"]),
        int(row["to_occurrence"]),
    )
    evidence["shared_routed_transition"] = assert_equal_multiset(
        "shared routed transition",
        multiset(routed_rows[MODELS[0]], transition_key),
        multiset(routed_rows[MODELS[1]], transition_key),
    )
    expected_phase_counts = {"discovery": 20, "confirmation": 10}
    observed_phase_counts = collections.Counter(
        phase(int(row["seed"])) for row in routed_rows[MODELS[0]]
    )
    if dict(observed_phase_counts) != expected_phase_counts:
        raise ValueError(
            "Routed panel must contain exactly 20 discovery and 10 confirmation "
            f"rows, observed={dict(observed_phase_counts)}"
        )

    grammar_dir = aligned / "shared_grammar_panel_v2"
    for timing in ("rank_after_city", "rank_before_city"):
        rows = {
            model: read_jsonl(grammar_dir / f"{model}_{timing}_panel.jsonl")
            for model in MODELS
        }
        key = lambda row, timing=timing: (
            phase(int(row["seed"])),
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["from_occurrence"]),
            int(row["to_occurrence"]),
            timing,
        )
        evidence[f"grammar_{timing}"] = assert_equal_multiset(
            f"grammar {timing}",
            multiset(rows[MODELS[0]], key),
            multiset(rows[MODELS[1]], key),
        )

    view_dir = aligned / "generation_views_routed"
    views = {
        model: read_jsonl(view_dir / f"{model}_generations_aligned.jsonl")
        for model in MODELS
    }
    view_key = lambda row: (
        phase(int(row["seed"])),
        int(row["seed"]),
        int(row["gold_count"]),
    )
    evidence["aligned_generation_view"] = assert_equal_multiset(
        "aligned generation view",
        multiset(views[MODELS[0]], view_key),
        multiset(views[MODELS[1]], view_key),
    )

    indexed_dir = aligned / "indexed_progress_control_panel"
    indexed_rows = {
        model: read_jsonl(indexed_dir / f"{model}.jsonl") for model in MODELS
    }
    indexed_key = lambda row: (
        str(row["split"]),
        int(row["seed"]),
        int(row["gold_count"]),
    )
    evidence["controlled_explicit_index"] = assert_equal_multiset(
        "surface-matched explicit-index positive control",
        multiset(indexed_rows[MODELS[0]], indexed_key),
        multiset(indexed_rows[MODELS[1]], indexed_key),
    )
    qwen_indexed_surface = {
        indexed_key(row): str(row["controlled_indexed_surface_text"])
        for row in indexed_rows[MODELS[0]]
    }
    gemma_indexed_surface = {
        indexed_key(row): str(row["controlled_indexed_surface_text"])
        for row in indexed_rows[MODELS[1]]
    }
    if qwen_indexed_surface != gemma_indexed_surface:
        raise ValueError("Explicit-index inner surfaces are not exactly matched")
    if evidence["controlled_explicit_index"]["aligned_cells"] != 30:
        raise ValueError("Explicit-index control must contain exactly 30 samples/model")
    evidence["controlled_explicit_index"].update(
        {
            "inner_surface_exactly_equal": True,
            "gold_count": 10,
            "controlled_teacher_forced_trace": True,
            "claim_role": "explicit-index positive control only",
        }
    )

    pair_dir = aligned / "answer_query_layer_sweep_plan"
    pairs = {
        model: read_jsonl(pair_dir / f"{model}_pairs.jsonl") for model in MODELS
    }
    pair_key = lambda row: (
        "confirmation",
        int(row["seed"]),
        int(row["receiver_count"]),
        int(row["donor_count"]),
        str(row["pair_direction"]),
    )
    evidence["answer_query_pairs"] = assert_equal_multiset(
        "answer-query donor/receiver plan",
        multiset(pairs[MODELS[0]], pair_key),
        multiset(pairs[MODELS[1]], pair_key),
    )

    plan_dir = aligned / "native_loop_plans"
    for split in ("discovery", "confirmation"):
        plans: dict[str, list[dict[str, Any]]] = {}
        for model in MODELS:
            frame = pd.read_csv(
                plan_dir / model / split / "native_loop_plan.csv",
                keep_default_na=False,
            )
            plans[model] = frame.to_dict(orient="records")
        plan_key = lambda row, split=split: (
            split,
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["receiver_occurrence"]),
            int(row["donor_occurrence"]),
            int(row["donor_offset"]),
            str(row["panel_kind"]),
        )
        evidence[f"native_loop_{split}"] = assert_equal_multiset(
            f"native-loop {split}",
            multiset(plans[MODELS[0]], plan_key),
            multiset(plans[MODELS[1]], plan_key),
        )

    # Running item-end representation is analyzed only on exact model-common
    # cells.  The prior independent parser-eligible panels are historical and
    # are not admissible for a model contrast.
    running_geometry_root = root / "reports/v5_native_cross_model_aligned_geometry"
    running_geometry_audit = json.loads(
        (running_geometry_root / "audit.json").read_text(encoding="utf-8")
    )
    if not bool(running_geometry_audit.get("exact_cross_model_sample_alignment")):
        raise ValueError("Running representation common-support audit failed")
    evidence["running_representation_common_support"] = {
        "status": "PASS",
        "aligned_cells": int(running_geometry_audit["common_state_rows"]),
        "unique_keys": int(running_geometry_audit["common_state_rows"]),
        "common_trajectory_cells": int(
            running_geometry_audit["common_trajectory_cells"]
        ),
        "key_multiset_sha256": str(running_geometry_audit["common_key_sha256"]),
    }

    # Final answer-query geometry uses the full registered 30 seeds x 10 N
    # panel in both models.  Verify the registered support dictionaries rather
    # than relying on the equal row count alone.
    dual_root = root / "reports/v5_dual_endpoint_geometry_full300"
    final_audits = {
        model: json.loads(
            (
                dual_root
                / model
                / "pca16_whiten"
                / "dual_endpoint_geometry_audit.json"
            ).read_text(encoding="utf-8")
        )["final_count_audit"]
        for model in MODELS
    }
    if final_audits[MODELS[0]]["registered_seed_panel"] != final_audits[MODELS[1]][
        "registered_seed_panel"
    ]:
        raise ValueError("Final answer-query representation seed panels differ")
    if final_audits[MODELS[0]]["support"]["native_thinking"] != final_audits[
        MODELS[1]
    ]["support"]["native_thinking"]:
        raise ValueError("Final answer-query representation support differs")
    evidence["final_answer_query_representation"] = {
        "status": "PASS",
        "aligned_cells": 300,
        "unique_keys": 300,
        "key_multiset_sha256": hashlib.sha256(
            json.dumps(
                {
                    "seeds": final_audits[MODELS[0]]["registered_seed_panel"],
                    "support": final_audits[MODELS[0]]["support"][
                        "native_thinking"
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }

    # The controlled unnumbered counter restoration uses the same seed x
    # target-occurrence cells.  Decoder-layer grids are architecture-specific
    # and therefore audited after projecting away the layer axis.
    unnumbered_root = root / "work_remote_snapshots/ncc_unnumbered_supplement_20260823"
    for split in ("discovery", "confirmation"):
        unnumbered = {
            model: read_csv_rows(
                unnumbered_root
                / model
                / f"unnumbered_analysis_{split}"
                / "patch_rows_derived.csv"
            )
            for model in MODELS
        }
        projected = {
            model: {
                (split, int(row["seed"]), int(row["target_occurrence"]))
                for row in unnumbered[model]
            }
            for model in MODELS
        }
        if projected[MODELS[0]] != projected[MODELS[1]]:
            raise ValueError(f"Controlled unnumbered {split} support differs")
        layer_counts = {
            model: len({int(row["source_layer"]) for row in unnumbered[model]})
            for model in MODELS
        }
        evidence[f"controlled_unnumbered_{split}"] = {
            "status": "PASS",
            "aligned_cells": len(projected[MODELS[0]]),
            "unique_keys": len(projected[MODELS[0]]),
            "key_multiset_sha256": hashlib.sha256(
                json.dumps(
                    sorted(list(key) for key in projected[MODELS[0]]),
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "architecture_specific_layer_counts": layer_counts,
        }

    # Whole-source answer blanking is already fully paired on the same
    # confirmation seed, gold-N, condition, and repeat grid.
    answer_layouts = {
        "Qwen3-8B": {
            "trace": "answer_tracebank_top32_confirmation_all20_v2",
            "prompt": "answer_promptbank_top32_confirmation_all20_v2",
        },
        "Gemma4-E4B": {
            "trace": "answer_tracebank_top32_confirmation_all20_v1",
            "prompt": "answer_promptbank_top32_confirmation_all20_v1",
        },
    }
    token_root = root / "reports/v5_native_token_level_ablation"
    answer_detail_paths: list[Path] = []
    for bank in ("trace", "prompt"):
        answer_rows: dict[str, list[dict[str, Any]]] = {}
        for model in MODELS:
            detail = (
                token_root
                / model
                / answer_layouts[model][bank]
                / "analysis_registered_v1"
                / "token_level_detail.csv"
            )
            answer_detail_paths.append(detail)
            answer_rows[model] = read_csv_rows(detail)
        answer_key = lambda row, bank=bank: (
            "confirmation",
            int(row["seed"]),
            int(row["gold_count"]),
            bank,
            str(row["condition"]),
            int(row["control_repeat"]),
            str(row.get("matched_control_for", "")),
        )
        evidence[f"answer_source_blank_{bank}"] = assert_equal_multiset(
            f"answer source blank {bank}",
            multiset(answer_rows[MODELS[0]], answer_key),
            multiset(answer_rows[MODELS[1]], answer_key),
        )

    # Same-trial terminal relay results are reported only on the exact
    # cross-model intersection, never on the larger model-specific plans.
    relay_root = (
        root
        / "work/relay_bridge_export_20260829/common_support_confirmation_rerun"
    )
    relay_summary = json.loads((relay_root / "summary.json").read_text(encoding="utf-8"))
    relay_pairs = read_csv_rows(relay_root / "common_support_pairs.csv")
    if relay_summary.get("status") != "PASS" or int(
        relay_summary.get("common_pair_count_per_model", -1)
    ) != len(relay_pairs):
        raise ValueError("Terminal relay common-support audit failed")
    relay_keys = [
        (
            "confirmation",
            int(row["seed"]),
            int(row["gold_count"]),
            int(row["donor_offset"]),
        )
        for row in relay_pairs
    ]
    evidence["terminal_relay_common_support"] = {
        "status": "PASS",
        "aligned_cells": len(relay_keys),
        "unique_keys": len(set(relay_keys)),
        "key_multiset_sha256": hashlib.sha256(
            json.dumps(sorted(relay_keys), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "common_seed_count": int(relay_summary["common_seed_count"]),
        "selection_uses_outcomes": bool(relay_summary["selection_uses_outcomes"]),
    }

    source_paths = sorted(
        path
        for directory in (
            routed_dir,
            grammar_dir,
            view_dir,
            pair_dir,
            plan_dir,
            indexed_dir,
            running_geometry_root,
            relay_root,
        )
        for path in directory.rglob("*")
        if path.is_file()
    )
    source_paths.extend(answer_detail_paths)
    source_paths.extend(
        dual_root / model / "pca16_whiten" / "dual_endpoint_geometry_audit.json"
        for model in MODELS
    )
    source_paths.extend(
        unnumbered_root
        / model
        / f"unnumbered_analysis_{split}"
        / "patch_rows_derived.csv"
        for model in MODELS
        for split in ("discovery", "confirmation")
    )
    source_paths = sorted(set(source_paths))
    return {
        "schema_version": "realistic_niah_v5_cross_model_sample_alignment_audit_v1",
        "status": "PASS",
        "contract": (
            "identical multisets of phase, seed, gold count, and every "
            "experiment-specific donor/receiver or transition key"
        ),
        "architecture_specific_parameters_allowed": [
            "layer index/grid",
            "head identity",
            "frozen bank width",
            "tokenizer span",
        ],
        "evidence": evidence,
        "source_sha256": {
            path.relative_to(root).as_posix(): sha256(path) for path in source_paths
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "evidence": result["evidence"]}))


if __name__ == "__main__":
    main()
