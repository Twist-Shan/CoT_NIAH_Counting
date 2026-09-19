#!/usr/bin/env python3
"""Freeze one outcome-blind N=10 final-transition request per V6 seed slot.

The targeted-retrieval behavior kernel historically evaluated only N=10.
That silently drops seeds when a structured-enumeration response is not in the
strict causal cohort. V6 now resolves every failed N=10 cell through the
frozen-amendment replacement registry before any intervention outcome is read.
The selected transition is therefore always the original report's fixed 9->10
transition rather than a post-hoc lower-count fallback.

Three registries are emitted because the audited kernels use two distinct
anchor identities:

* ``behavior_anchor_registry.jsonl`` preserves the exact localizer produced by
  causal-source-writes and is replayed by free-generation behavior assays;
* ``targeted_registry.jsonl`` converts the same single query to the routed
  ``...@route-qK`` identity expected by the teacher-forced bridge kernels;
* ``mode_panel.jsonl`` adds only the immutable V6 mode/timing audit fields used
  by the stratified NCC and direct logit-margin runners.

No source-write magnitude, head ranking, behavior result, or intervention
outcome participates in row selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    registered_records,
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.replacement import (  # noqa: E402
    resolved_generation_records,
)


SCHEMA_VERSION = "realistic_niah_v6_final_transition_panel_v1"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Source-write shard starts with a non-object: {path}")
                return value
    raise ValueError(f"Source-write shard is empty: {path}")


def _mode_contract(config: V6Config) -> tuple[str, str]:
    if config.prompt_mode == "enumeration_index":
        return "post_marker", "rank_before_city"
    if config.prompt_mode == "enumeration_bullet":
        return "p0_item_end", "structural_item_end"
    raise ValueError(f"Unsupported V6 prompt mode: {config.prompt_mode}")


def build_panel(
    *,
    config_path: Path,
    model_label: str,
    generations_path: Path,
    cohort_registry: Path,
    source_writes: Path,
    seed_role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return behavior, targeted, panel, and manifest payloads."""

    config = V6Config.load(config_path)
    if model_label not in config.model_labels:
        raise ValueError(f"Model {model_label!r} is outside the V6 config")
    if seed_role not in {"discovery", "confirmation"}:
        raise ValueError("seed_role must be discovery or confirmation")
    anchor_role, timing = _mode_contract(config)
    expected_slot_seeds = tuple(
        config.discovery_seeds
        if seed_role == "discovery"
        else config.confirmation_seeds
    )
    expected_slot_seed_set = set(map(int, expected_slot_seeds))

    generation_rows = read_jsonl(generations_path)
    validate_generation_contracts(
        generation_rows,
        config,
        model_label=model_label,
        config_sha256=sha256_file(config_path),
    )
    formal = resolved_generation_records(
        generation_rows,
        config,
        registry_path=cohort_registry,
        model_label=model_label,
    )
    if {str(row["split"]) for row in formal} != {seed_role}:
        raise ValueError("Resolved final-transition registry has the wrong seed role")
    formal_by_id = {
        str(row["request_id"]): row for row in formal
    }
    if len(formal_by_id) != len(formal):
        raise ValueError("Formal V6 generation requests are not unique")
    terminal_rows = [row for row in formal if int(row["gold_count"]) == 10]
    if len(terminal_rows) != len(expected_slot_seeds):
        raise ValueError("Resolved V6 cohort does not contain one N=10 row per slot")
    if {
        int(row["v6_analysis_slot_seed"]) for row in terminal_rows
    } != expected_slot_seed_set:
        raise ValueError("Resolved N=10 rows changed the original analysis slots")
    terminal_by_id = {str(row["request_id"]): row for row in terminal_rows}

    shard_dir = source_writes / "shards"
    shard_paths = sorted(shard_dir.glob("*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No source-write shards under {shard_dir}")
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in shard_paths:
        row = _first_jsonl_row(path)
        request_id = str(row.get("request_id", ""))
        if request_id not in terminal_by_id:
            continue
        if str(row.get("model_label")) != model_label:
            continue
        if str(row.get("split")) != seed_role:
            continue
        roles = {
            str(value)
            for value in row.get("anchor_roles", [row.get("anchor_role")])
            if value is not None
        }
        if anchor_role not in roles:
            continue
        count = int(row["gold_count"])
        if count != 10:
            continue
        if int(row["from_occurrence"]) != count - 1:
            continue
        if int(row["to_occurrence"]) != count:
            continue
        if not bool(row.get("capture_complete", True)):
            raise ValueError(f"Selected source-write shard is incomplete: {path}")
        candidates.append((row, path))

    by_request: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for row, path in candidates:
        request_id = str(row["request_id"])
        by_request.setdefault(request_id, []).append((row, path))
    missing = sorted(set(terminal_by_id) - set(by_request))
    if missing:
        raise ValueError(
            "No strictly eligible N=10 final transition for resolved requests: "
            f"{missing}"
        )

    selected: list[tuple[dict[str, Any], Path]] = []
    for terminal in sorted(
        terminal_rows, key=lambda row: int(row["v6_analysis_slot_seed"])
    ):
        request_id = str(terminal["request_id"])
        matches = by_request[request_id]
        identities = {
            (
                str(value[0]["request_id"]),
                int(value[0]["from_occurrence"]),
                int(value[0]["to_occurrence"]),
                int(value[0]["query_output_token_index"]),
            )
            for value in matches
        }
        if len(identities) != 1:
            raise ValueError(
                f"N=10 final transition is not unique for {request_id}: "
                f"{sorted(identities)}"
            )
        selected.append(matches[-1])

    behavior_rows: list[dict[str, Any]] = []
    targeted_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    selected_shards: dict[str, dict[str, str]] = {}
    for source, shard_path in selected:
        request_id = str(source["request_id"])
        seed = int(source["seed"])
        generation = terminal_by_id[request_id]
        analysis_slot_seed = int(generation["v6_analysis_slot_seed"])
        count = int(source["gold_count"])
        from_occurrence = int(source["from_occurrence"])
        to_occurrence = int(source["to_occurrence"])
        query = int(source["query_output_token_index"])
        grammar_pair = str(source.get("grammar_pair", ""))
        target_grammar = grammar_pair.rsplit(" -> ", 1)[-1]
        raw_anchor = str(source["anchor_equivalence_id"])
        expected_raw = f"{from_occurrence}->{to_occurrence}@q{query}"
        if raw_anchor != expected_raw:
            raise ValueError(
                f"Unexpected local final-transition identity {raw_anchor!r}; "
                f"expected {expected_raw!r}"
            )
        roles = [str(value) for value in source.get("anchor_roles", [anchor_role])]
        if anchor_role not in roles:
            raise ValueError("Selected behavior anchor lost its registered role")
        common = {
            "request_id": request_id,
            "seed": seed,
            "analysis_slot_seed": analysis_slot_seed,
            "source_seed": seed,
            "replacement_applied": bool(generation["v6_replacement_applied"]),
            "gold_count": count,
            "from_occurrence": from_occurrence,
            "to_occurrence": to_occurrence,
            "target_grammar_class": target_grammar,
            "target_retrieval_surface_variant": str(
                source.get("target_retrieval_surface_variant", "")
            ),
        }
        behavior_rows.append(
            {
                **common,
                "anchor_equivalence_id": raw_anchor,
                "anchor_roles": roles,
            }
        )
        routed = {
            **common,
            "anchor_equivalence_id": (
                f"{from_occurrence}->{to_occurrence}@route-q{query}"
            ),
            "anchor_roles": ["grammar_routed_retrieval_window"],
            "query_output_token_index": query,
            "source_anchor_equivalence_id": raw_anchor,
            "source_anchor_role": anchor_role,
            "mode_timing_stratum": timing,
            "outcome_blind": True,
            "selection_rank_used": False,
        }
        targeted_rows.append(routed)
        panel_rows.append(
            {
                **routed,
                "grammar_span_timing_stratum": timing,
                "stratified_ncc_seed_role": seed_role,
                "stratified_ncc_outcome_blind": True,
                "stratified_ncc_selection_rank_used": False,
            }
        )
        selected_shards[request_id] = {
            "path": str(shard_path.resolve()),
            "sha256": sha256_file(shard_path),
        }

    if len(behavior_rows) != len(expected_slot_seeds):
        raise RuntimeError("Final-transition panel changed the one-row-per-seed contract")
    if {
        int(row["analysis_slot_seed"]) for row in behavior_rows
    } != expected_slot_seed_set:
        raise RuntimeError("Final-transition panel analysis-slot set changed")
    if any("selection_rank" in row for row in behavior_rows + targeted_rows + panel_rows):
        raise RuntimeError("Final-transition panel must not contain selection_rank")

    core = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_OUTCOME_BLIND",
        "prompt_mode": config.prompt_mode,
        "model_label": model_label,
        "seed_role": seed_role,
        "analysis_slot_seeds": list(map(int, expected_slot_seeds)),
        "source_seeds": [int(row["seed"]) for row in behavior_rows],
        "seed_count": len(expected_slot_seeds),
        "anchor_role": anchor_role,
        "mode_timing_stratum": timing,
        "selection_rule": (
            "fixed_N10_final_transition_from_outcome_blind_resolved_registry; "
            "replacement_only_for_generation_or_fresh_strict_parser_failure"
        ),
        "selected_count_by_analysis_slot": {
            str(row["analysis_slot_seed"]): int(row["gold_count"])
            for row in behavior_rows
        },
        "selected_request_by_analysis_slot": {
            str(row["analysis_slot_seed"]): str(row["request_id"])
            for row in behavior_rows
        },
        "candidate_count": len(candidates),
        "formal_generation_count": len(formal_by_id),
        "generations": str(generations_path.resolve()),
        "generations_sha256": sha256_file(generations_path),
        "cohort_registry": str(cohort_registry.resolve()),
        "cohort_registry_sha256": sha256_file(cohort_registry),
        "v6_config": str(config_path.resolve()),
        "v6_config_sha256": sha256_file(config_path),
        "source_writes": str(source_writes.resolve()),
        "source_manifest_sha256": sha256_file(source_writes / "manifest.json"),
        "selected_source_shards": selected_shards,
        "strict_format_membership_used": True,
        "fixed_gold_count": 10,
        "intervention_outcomes_read": False,
        "source_write_values_used_for_selection": False,
        "head_ranks_used_for_selection": False,
        "selection_rank_used": False,
    }
    manifest = {**core, "panel_sha256": _sha256_json(core)}
    return behavior_rows, targeted_rows, panel_rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--source-writes", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=("discovery", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    behavior, targeted, panel, manifest = build_panel(
        config_path=args.v6_config.resolve(),
        model_label=str(args.model),
        generations_path=args.generations.resolve(),
        cohort_registry=args.cohort_registry.resolve(),
        source_writes=args.source_writes.resolve(),
        seed_role=str(args.seed_role),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "behavior_anchor_registry.jsonl", behavior)
    _atomic_jsonl(args.output / "targeted_registry.jsonl", targeted)
    _atomic_jsonl(args.output / "mode_panel.jsonl", panel)
    output_hashes = {
        name: sha256_file(args.output / name)
        for name in (
            "behavior_anchor_registry.jsonl",
            "targeted_registry.jsonl",
            "mode_panel.jsonl",
        )
    }
    _atomic_json(args.output / "manifest.json", {**manifest, "outputs": output_hashes})
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": args.model,
                "prompt_mode": manifest["prompt_mode"],
                "seed_count": manifest["seed_count"],
                "selected_count_by_analysis_slot": manifest[
                    "selected_count_by_analysis_slot"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
