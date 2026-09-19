#!/usr/bin/env python3
"""Audit *realized* Qwen/Gemma trial support for the aligned Native suite.

The input-panel audit proves that the registered cohorts are aligned.  This
script is deliberately stricter: it reads the completed trial artifacts and
checks that the realized semantic sample/condition multisets are identical
after projecting out only preregistered architecture-specific layer axes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODELS = ("Qwen3-8B", "Gemma4-E4B")
PHASES = ("discovery", "confirmation")
ANSWER_QUERY_LAYERS = {
    "Qwen3-8B": (0, 5, 10, 15, 20, 25, 30, 35),
    "Gemma4-E4B": (0, 6, 12, 18, 23, 29, 35, 41),
}
INDEXED_DISCOVERY_LAYERS = {
    "Qwen3-8B": tuple(range(36)),
    "Gemma4-E4B": tuple(range(42)),
}
INDEXED_CONFIRMATION_LAYER = {"Qwen3-8B": 19, "Gemma4-E4B": 16}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    files = sorted(set(Path(path) for path in paths))
    if not files:
        raise FileNotFoundError("No JSONL artifacts matched")
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL {path}:{line_number}") from error
    return rows


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite key component: {value}")
        return value
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    raise TypeError(f"Unsupported key component {type(value)!r}: {value!r}")


def _key(*values: Any) -> tuple[Any, ...]:
    return tuple(_json_scalar(value) for value in values)


def _digest(counter: collections.Counter[tuple[Any, ...]]) -> str:
    payload = [
        {"key": list(key), "multiplicity": int(multiplicity)}
        for key, multiplicity in sorted(
            counter.items(),
            key=lambda item: json.dumps(
                list(item[0]), ensure_ascii=False, separators=(",", ":")
            ),
        )
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_ok_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    failed = [
        row
        for row in rows
        if "status" in row and str(row.get("status")) != "ok"
    ]
    if failed:
        statuses = collections.Counter(str(row.get("status")) for row in failed)
        raise ValueError(f"{label}: non-ok realized rows: {dict(statuses)}")
    incomplete = [row for row in rows if row.get("trial_complete") is False]
    if incomplete:
        raise ValueError(f"{label}: {len(incomplete)} incomplete trial rows")


def _compare(
    name: str,
    keys: Mapping[str, Sequence[tuple[Any, ...]]],
    *,
    projection: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counters = {model: collections.Counter(keys[model]) for model in MODELS}
    if counters[MODELS[0]] != counters[MODELS[1]]:
        only_q = counters[MODELS[0]] - counters[MODELS[1]]
        only_g = counters[MODELS[1]] - counters[MODELS[0]]
        raise ValueError(
            f"{name}: realized cross-model multiset mismatch; "
            f"Q-only={list(only_q.items())[:5]} G-only={list(only_g.items())[:5]}"
        )
    digest = _digest(counters[MODELS[0]])
    value: dict[str, Any] = {
        "status": "PASS",
        "projection": projection,
        "realized_cells_per_model": int(sum(counters[MODELS[0]].values())),
        "unique_semantic_keys": int(len(counters[MODELS[0]])),
        "key_multiset_sha256": digest,
        "model_digests": {
            model: _digest(counters[model]) for model in MODELS
        },
    }
    if extra:
        value.update(dict(extra))
    return value


def _targeted_retrieval(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    for model in MODELS:
        for phase in PHASES:
            rows = _read_jsonl_files(
                (root / model / "targeted_retrieval" / phase / "shards").glob(
                    "*.jsonl"
                )
            )
            _assert_ok_rows(rows, label=f"{model} targeted retrieval {phase}")
            for row in rows:
                keys[model].append(
                    _key(
                        phase,
                        int(row["seed"]),
                        int(row["gold_count"]),
                        int(row["from_occurrence"]),
                        int(row["to_occurrence"]),
                        str(row["condition"]),
                        int(row.get("repeat", 0)),
                    )
                )
    return _compare(
        "targeted_retrieval",
        keys,
        projection="phase, seed, gold_count, from/to occurrence, condition, repeat",
    )


def _targeted_counter_write(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    timing_counts: dict[str, dict[str, dict[str, int]]] = {
        model: {} for model in MODELS
    }
    for model in MODELS:
        for phase in PHASES:
            rows = _read_jsonl_files(
                (root / model / "targeted_counter_write" / phase / "shards").glob(
                    "*.jsonl"
                )
            )
            _assert_ok_rows(rows, label=f"{model} counter write {phase}")
            timing_counts[model][phase] = dict(
                sorted(
                    collections.Counter(
                        str(row["grammar_timing_stratum"]) for row in rows
                    ).items()
                )
            )
            for row in rows:
                keys[model].append(
                    _key(
                        phase,
                        int(row["seed"]),
                        int(row["gold_count"]),
                        int(row["targeted_from_occurrence"]),
                        int(row["targeted_to_occurrence"]),
                        str(row["condition"]),
                    )
                )
    return _compare(
        "targeted_counter_write",
        keys,
        projection="phase, seed, gold_count, targeted transition, condition",
        extra={
            "model_specific_grammar_timing_counts": timing_counts,
            "grammar_timing_projection_reason": (
                "The routed panel aligns semantic transition keys. Natural trace "
                "grammar timing is an observed model-specific surface property, not "
                "a cross-model pairing key; it remains reported per model."
            ),
        },
    )


def _grammar_span(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    phase_counts: dict[str, dict[str, int]] = {model: {} for model in MODELS}
    for model in MODELS:
        for phase in PHASES:
            rows = _read_jsonl_files(
                (root / model / "grammar_span" / phase / "shards").glob("*.jsonl")
            )
            _assert_ok_rows(rows, label=f"{model} grammar span {phase}")
            phase_counts[model][phase] = len({int(row["seed"]) for row in rows})
            for row in rows:
                keys[model].append(
                    _key(
                        phase,
                        int(row["seed"]),
                        int(row["gold_count"]),
                        str(row["grammar_timing_stratum"]),
                        int(row["target_occurrence"]),
                        str(row["condition"]),
                    )
                )
    return _compare(
        "grammar_span",
        keys,
        projection="phase, seed, gold_count, grammar timing, target occurrence, condition",
        extra={"effective_seed_counts": phase_counts},
    )


def _npz_factorial(
    root: Path, *, directory: str, experiment_id: str
) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    for model in MODELS:
        for timing in ("rank_after_city", "rank_before_city"):
            for phase in PHASES:
                files = sorted(
                    (
                        root
                        / model
                        / directory
                        / timing
                        / phase
                        / "shards"
                    ).glob("*.npz")
                )
                if not files:
                    raise FileNotFoundError(
                        f"No {directory} NPZ shards for {model}/{timing}/{phase}"
                    )
                for path in files:
                    with np.load(path, allow_pickle=False) as archive:
                        metadata = json.loads(str(archive["metadata_json"].item()))
                        conditions = tuple(str(x) for x in archive["condition_names"])
                    if str(metadata.get("experiment_id")) != experiment_id:
                        raise ValueError(f"{path}: experiment id changed")
                    if not bool(metadata.get("outcome_blind")) or bool(
                        metadata.get("selection_rank_used")
                    ):
                        raise ValueError(f"{path}: outcome-blind contract failed")
                    for condition in conditions:
                        keys[model].append(
                            _key(
                                phase,
                                timing,
                                int(metadata["seed"]),
                                int(metadata["gold_count"]),
                                condition,
                            )
                        )
    return _compare(
        directory,
        keys,
        projection=(
            "phase, timing, seed, gold_count, condition; layer/endpoint tensor axes "
            "audited within-model and projected out"
        ),
    )


def _direct_count_logit_margin(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    for model in MODELS:
        for timing in ("rank_after_city", "rank_before_city"):
            for phase in PHASES:
                directory = (
                    root
                    / model
                    / "direct_count_logit_margin"
                    / timing
                    / phase
                    / "shards"
                )
                files = sorted(directory.glob("*.json"))
                if not files:
                    raise FileNotFoundError(
                        f"No direct-count margin shards for {model}/{timing}/{phase}"
                    )
                for path in files:
                    row = _read_json(path)
                    if (
                        row.get("experiment_id")
                        != "targeted_retrieval_query_to_direct_count_logit_margin"
                        or str(row.get("timing_branch")) != timing
                        or not bool(row.get("outcome_blind_panel"))
                        or bool(row.get("selection_rank_used"))
                    ):
                        raise ValueError(f"{path}: direct-count margin contract failed")
                    conditions = tuple(
                        str(item["condition"]) for item in row["conditions"]
                    )
                    for condition in conditions:
                        keys[model].append(
                            _key(
                                phase,
                                timing,
                                int(row["seed"]),
                                int(row["gold_count"]),
                                condition,
                            )
                        )
    return _compare(
        "direct_count_logit_margin",
        keys,
        projection="phase, timing, seed, gold_count, factorial condition",
    )


def _next_city_token_blank(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    for model in MODELS:
        rows = _read_jsonl_files(
            (root / model / "next_city_token_blank" / "shards").glob("*.jsonl")
        )
        _assert_ok_rows(rows, label=f"{model} next-city token blank")
        for row in rows:
            keys[model].append(
                _key(
                    int(row["seed"]),
                    int(row["gold_count"]),
                    int(row["from_occurrence"]),
                    int(row["to_occurrence"]),
                    str(row["condition"]),
                    int(row.get("control_repeat", 0)),
                )
            )
    return _compare(
        "next_city_token_blank",
        keys,
        projection=(
            "seed, gold_count, from/to occurrence, blank condition, matched-control repeat"
        ),
    )


def _answer_query(root: Path, plan_root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    grids: dict[str, list[int]] = {}
    integrity: dict[str, dict[str, str]] = {}
    manifest = _read_json(plan_root / "alignment_manifest.json")
    if (
        manifest.get("status") != "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT"
        or manifest.get("exact_cross_model_key_equality") is not True
        or manifest.get("pair_selection_uses_patch_outcome") is not False
    ):
        raise ValueError("Answer-query plan manifest is not outcome-blind and frozen")
    for model in MODELS:
        pair_path = plan_root / f"{model}_pairs.jsonl"
        pairs = _read_jsonl_files([pair_path])
        registry_digest = _sha256_json(pairs)
        file_digest = _sha256_file(pair_path)
        manifest_key = (
            "qwen_pairs_sha256" if model == "Qwen3-8B" else "gemma_pairs_sha256"
        )
        if manifest.get(manifest_key) != registry_digest:
            raise ValueError(f"{model}: answer-query canonical registry hash changed")
        analysis_path = (
            root / model / "answer_query_layer_sweep" / "analysis" / "audit.json"
        )
        analysis = _read_json(analysis_path)
        if (
            analysis.get("status") != "passed"
            or analysis.get("pairs_sha256") != file_digest
            or int(analysis.get("registered_pairs", -1)) != 40
        ):
            raise ValueError(f"{model}: answer-query file/analysis provenance mismatch")
        integrity[model] = {
            "canonical_registry_sha256": registry_digest,
            "jsonl_file_sha256": file_digest,
            "analysis_pairs_sha256": str(analysis["pairs_sha256"]),
        }
        alignment_id = {
            str(row["pair_id"]): str(row["alignment_pair_id"]) for row in pairs
        }
        rows = _read_jsonl_files(
            [root / model / "answer_query_layer_sweep" / "trials.jsonl"]
        )
        observed_layers = sorted({int(row["layer"]) for row in rows})
        if tuple(observed_layers) != ANSWER_QUERY_LAYERS[model]:
            raise ValueError(f"{model}: answer-query layer grid changed")
        grids[model] = observed_layers
        layer_ordinal = {layer: index for index, layer in enumerate(observed_layers)}
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id not in alignment_id:
                raise ValueError(f"{model}: unregistered answer pair {pair_id}")
            keys[model].append(
                _key(
                    alignment_id[pair_id],
                    layer_ordinal[int(row["layer"])],
                    str(row["condition"]),
                )
            )
    return _compare(
        "answer_query_layer_sweep",
        keys,
        projection=(
            "cross-model alignment_pair_id, preregistered layer-grid ordinal, condition"
        ),
        extra={
            "architecture_specific_layer_grids": grids,
            "pair_registry_integrity": integrity,
        },
    )


def _commit_to_query(root: Path) -> dict[str, Any]:
    keys: dict[str, list[tuple[Any, ...]]] = {model: [] for model in MODELS}
    for model in MODELS:
        for phase in PHASES:
            rows = _read_jsonl_files(
                (
                    root
                    / model
                    / "commit_state_to_targeted_query"
                    / phase
                    / "shards"
                ).glob("*.jsonl")
            )
            rows = [
                row
                for row in rows
                if str(row.get("experiment_id"))
                == "p0_count_state_to_targeted_retrieval"
            ]
            if not rows:
                raise ValueError(f"{model} {phase}: no commit-to-query rows")
            _assert_ok_rows(rows, label=f"{model} commit-to-query {phase}")
            for row in rows:
                keys[model].append(
                    _key(
                        phase,
                        int(row["seed"]),
                        int(row["gold_count"]),
                        int(row["receiver_occurrence"]),
                        int(row["donor_occurrence"]),
                        int(row["donor_offset"]),
                        str(row["condition"]),
                    )
                )
    return _compare(
        "commit_state_to_targeted_query",
        keys,
        projection=(
            "phase, seed, gold_count, receiver/donor occurrence, donor offset, condition"
        ),
    )


def _indexed_progress(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery_keys: dict[str, list[tuple[Any, ...]]] = {
        model: [] for model in MODELS
    }
    confirmation_keys: dict[str, list[tuple[Any, ...]]] = {
        model: [] for model in MODELS
    }
    discovery_grids: dict[str, list[int]] = {}
    confirmation_layers: dict[str, list[int]] = {}
    for model in MODELS:
        discovery = root / model / "indexed_progress_control" / "discovery_layer_sweep"
        baseline_rows = _read_jsonl_files(discovery.glob("baseline/*/trials.jsonl"))
        span_rows = _read_jsonl_files(discovery.glob("item_span/*/trials.jsonl"))
        _assert_ok_rows(baseline_rows + span_rows, label=f"{model} indexed discovery")
        observed_grid = sorted({int(row["layer"]) for row in span_rows})
        if tuple(observed_grid) != INDEXED_DISCOVERY_LAYERS[model]:
            raise ValueError(f"{model}: indexed discovery layer grid changed")
        discovery_grids[model] = observed_grid
        for row in baseline_rows + span_rows:
            receiver = int(row["receiver_occurrence_j"])
            donor = int(row["donor_occurrence_k"])
            direction = "forward_skip" if receiver < donor else "backward_rewind"
            # Compare the unique semantic support.  The span layer multiplicity is
            # architecture-specific and is checked separately above.
            discovery_keys[model].append(
                _key(
                    int(row["seed"]),
                    int(row["gold_count"]),
                    receiver,
                    donor,
                    direction,
                    str(row["patch_scope"]),
                    str(row["condition"]),
                )
            )
        discovery_keys[model] = list(set(discovery_keys[model]))

        confirmation_root = (
            root
            / model
            / "indexed_progress_control"
            / "confirmation_runs"
            / "confirmation"
            / "item_span"
        )
        rows = _read_jsonl_files(confirmation_root.glob("*/trials.jsonl"))
        _assert_ok_rows(rows, label=f"{model} indexed confirmation")
        observed_layers = sorted({int(row["layer"]) for row in rows})
        if observed_layers != [INDEXED_CONFIRMATION_LAYER[model]]:
            raise ValueError(f"{model}: indexed confirmation layer changed")
        confirmation_layers[model] = observed_layers
        for row in rows:
            receiver = int(row["receiver_occurrence_j"])
            donor = int(row["donor_occurrence_k"])
            confirmation_keys[model].append(
                _key(
                    int(row["seed"]),
                    int(row["gold_count"]),
                    receiver,
                    donor,
                    "forward_skip" if receiver < donor else "backward_rewind",
                    str(row["condition"]),
                )
            )
    discovery_result = _compare(
        "indexed_progress_discovery_support",
        discovery_keys,
        projection=(
            "unique seed/count/receiver/donor/direction/scope/condition support; "
            "architecture-specific layer multiplicity projected out"
        ),
        extra={"architecture_specific_layer_grids": discovery_grids},
    )
    confirmation_result = _compare(
        "indexed_progress_confirmation",
        confirmation_keys,
        projection=(
            "seed, gold_count, receiver/donor occurrence, direction, condition; "
            "representation-frozen model-specific layer projected out"
        ),
        extra={"architecture_specific_frozen_layers": confirmation_layers},
    )
    return discovery_result, confirmation_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("work/v5_native_sample_aligned_20260829/runs"),
    )
    parser.add_argument(
        "--input-alignment-audit",
        type=Path,
        default=Path("work/v5_native_sample_aligned_20260829/alignment_audit.json"),
    )
    parser.add_argument(
        "--answer-query-plan-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/answer_query_layer_sweep_plan"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/output_alignment_audit.json"
        ),
    )
    args = parser.parse_args()

    input_audit = _read_json(args.input_alignment_audit)
    if input_audit.get("status") != "PASS" or not all(
        row.get("status") == "PASS" for row in input_audit.get("evidence", {}).values()
    ):
        raise ValueError("Input sample-alignment audit is not PASS")
    suite_complete: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        value = _read_json(args.run_root / model / "suite_complete.json")
        if value.get("status") != "PASS" or value.get("model_label") != model:
            raise ValueError(f"{model}: suite completion sentinel failed")
        suite_complete[model] = value

    evidence: dict[str, Any] = {}
    evidence["targeted_retrieval"] = _targeted_retrieval(args.run_root)
    evidence["targeted_counter_write"] = _targeted_counter_write(args.run_root)
    evidence["grammar_span"] = _grammar_span(args.run_root)
    evidence["stratified_ncc"] = _npz_factorial(
        args.run_root,
        directory="stratified_ncc",
        experiment_id="teacher_forced_stratified_targeted_counter_ncc",
    )
    evidence["direct_count_logit_margin"] = _direct_count_logit_margin(
        args.run_root
    )
    evidence["next_city_token_blank"] = _next_city_token_blank(args.run_root)
    evidence["answer_query_layer_sweep"] = _answer_query(
        args.run_root, args.answer_query_plan_root
    )
    evidence["commit_state_to_targeted_query"] = _commit_to_query(args.run_root)
    indexed_discovery, indexed_confirmation = _indexed_progress(args.run_root)
    evidence["indexed_progress_discovery_support"] = indexed_discovery
    evidence["indexed_progress_confirmation"] = indexed_confirmation

    value = {
        "schema_version": (
            "realistic_niah_v5_cross_model_sample_aligned_output_audit_v1"
        ),
        "status": "PASS",
        "contract": (
            "realized Qwen/Gemma semantic trial multisets must be identical; only "
            "predeclared architecture-specific layer axes may be projected out"
        ),
        "input_alignment_audit_sha256": hashlib.sha256(
            args.input_alignment_audit.read_bytes()
        ).hexdigest(),
        "suite_complete_sha256": {
            model: hashlib.sha256(
                (args.run_root / model / "suite_complete.json").read_bytes()
            ).hexdigest()
            for model in MODELS
        },
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
