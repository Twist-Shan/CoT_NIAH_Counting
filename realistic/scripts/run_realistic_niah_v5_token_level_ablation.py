#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import mechanism_continuations
from realistic_niah_v5.pipeline import read_jsonl, registered_records
from realistic_niah_v5.spec import V5Config
from realistic_niah_v5.token_level_ablation import (
    ANSWER_TOKEN_BLANK_CONDITIONS,
    TARGETING_TRACE_BLANK_CONDITIONS,
    TOKEN_LEVEL_ABLATION_SCHEMA_VERSION,
    run_answer_token_blank_trials,
    run_targeting_trace_token_trials,
)


DEFAULT_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _safe_stem(*values: Any) -> str:
    readable = "__".join(str(value) for value in values)
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:16]
    return f"trial_{digest}"


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    spec = resolve_model_spec(args.model)
    return load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, received {type(parsed).__name__}")
    return parsed


def _parse_heads(value: Any) -> tuple[tuple[int, int], ...]:
    heads = tuple((int(layer), int(head)) for layer, head in _parse_json_list(value))
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("Frozen heads must be nonempty and unique")
    return heads


def _causal_plan_bank_sha256(heads: Sequence[tuple[int, int]]) -> str:
    """Match the frozen causal-plan serializer, including its JSON spaces."""

    return hashlib.sha256(
        json.dumps([list(value) for value in heads]).encode("utf-8")
    ).hexdigest()


def _load_plan(args: argparse.Namespace) -> tuple[pd.DataFrame | None, str | None]:
    if args.heads_json is not None:
        return None, None
    if args.bank_plan is None:
        raise ValueError("Supply either --heads-json or --bank-plan")
    frame = pd.read_csv(
        args.bank_plan, engine="python", dtype=str, keep_default_na=False
    )
    needed = {"model_label", "condition", "heads", "bank_sha256"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen bank plan is missing {missing}")
    selected = frame.loc[
        frame["model_label"].eq(args.model)
        & frame["condition"].eq(args.bank_condition)
    ].copy()
    if getattr(args, "bank_size", None) is not None:
        if "bank_size" not in selected.columns:
            raise ValueError("--bank-size was supplied but the plan has no bank_size")
        selected = selected.loc[
            pd.to_numeric(selected["bank_size"], errors="raise").eq(
                int(args.bank_size)
            )
        ]
    if args.target_grammar_class is not None:
        grammar_column = next(
            (
                name
                for name in (
                    "selection_target_grammar_class",
                    "target_grammar_class",
                )
                if name in selected.columns
            ),
            None,
        )
        if grammar_column is not None:
            selected = selected.loc[
                selected[grammar_column].eq(str(args.target_grammar_class))
            ]
    if selected.empty:
        raise ValueError("No frozen bank-plan row matches the requested scope")
    plan_sha = hashlib.sha256(args.bank_plan.read_bytes()).hexdigest()
    return selected.reset_index(drop=True), plan_sha


def _bank_for_task(
    args: argparse.Namespace,
    plan: pd.DataFrame | None,
    *,
    seed: int,
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
    if args.heads_json is not None:
        heads = _parse_heads(args.heads_json)
        return heads, {
            "bank_source": "literal_heads_json",
            "bank_sha256": _sha256_json([list(value) for value in heads]),
            "bank_plan_row": None,
        }
    assert plan is not None
    candidates = plan.copy()
    if "validation_seeds" in candidates.columns:
        candidates = candidates.loc[
            candidates["validation_seeds"].map(
                lambda value: int(seed) in {int(item) for item in _parse_json_list(value)}
            )
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one frozen bank for seed {seed}, found {len(candidates)}"
        )
    index = int(candidates.index[0])
    row = candidates.iloc[0]
    heads = _parse_heads(row["heads"])
    causal_sha = _causal_plan_bank_sha256(heads)
    canonical_sha = _sha256_json([list(value) for value in heads])
    registered_sha = str(row["bank_sha256"])
    if registered_sha not in {causal_sha, canonical_sha}:
        raise ValueError("Frozen bank SHA disagrees with its head list")
    return heads, {
        "bank_source": "frozen_plan",
        "bank_sha256": registered_sha,
        "bank_sha256_serializer": (
            "json_default_separators"
            if registered_sha == causal_sha
            else "json_canonical_compact"
        ),
        "bank_plan_row": index,
        "bank_plan_fold": int(row.get("fold", 0) or 0),
        "selection_anchor_role": str(
            row.get("selection_anchor_role", args.anchor_role or "")
        ),
        "selection_metric": str(row.get("selection_metric", "")),
        "selection_target_grammar_class": str(
            row.get("selection_target_grammar_class", "")
        ),
    }


def _registry_rows(
    path: Path | None,
) -> dict[str, tuple[dict[str, Any], ...]] | None:
    if path is None:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        request_id = str(row.get("request_id", row.get("stimulus_id", "")))
        if not request_id:
            raise ValueError("Anchor registry row lacks request_id")
        grouped.setdefault(request_id, []).append(dict(row))
    if not grouped:
        raise ValueError("Anchor registry is empty")
    return {
        request_id: tuple(
            sorted(
                values,
                key=lambda row: (
                    int(row.get("from_occurrence", -1)),
                    int(row.get("to_occurrence", -1)),
                    str(row.get("anchor_equivalence_id", "")),
                ),
            )
        )
        for request_id, values in grouped.items()
    }


def _registry_event_matches(
    specification: Mapping[str, Any],
    registry_row: Mapping[str, Any],
    *,
    match_mode: str,
) -> bool:
    """Join a routed registry event to a continuation specification.

    Formal causal registries store the persistent-intervention route anchor
    (for example ``1->2@route-q192``), whereas a frozen head bank may have
    been ranked at a later exact localizer such as ``city_pre_d1``.  The
    In legacy ``transition`` mode, the stable identity is request + transition
    k->k+1 and every compatible localizer is retained.  ``exact`` mode is the
    fail-closed option for aligned panels: it additionally requires the frozen
    route-anchor string, yielding one preregistered query site per transition.
    """

    transition_matches = (
        int(specification.get("from_occurrence", -1))
        == int(registry_row.get("from_occurrence", -2))
        and int(specification.get("to_occurrence", -1))
        == int(registry_row.get("to_occurrence", -2))
    )
    if not transition_matches:
        return False
    if match_mode == "transition":
        return True
    if match_mode == "exact":
        specification_anchor = str(
            specification.get("anchor_equivalence_id", "")
        ).replace("@route-q", "@q")
        registry_anchor = str(
            registry_row.get("anchor_equivalence_id", "")
        ).replace("@route-q", "@q")
        return bool(specification_anchor) and specification_anchor == registry_anchor
    raise ValueError(f"Unsupported registry anchor match mode: {match_mode}")


def _build_tasks(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    plan: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    registry = _registry_rows(args.anchor_registry)
    tasks: list[dict[str, Any]] = []
    if args.mode == "answer":
        for row in rows:
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            if registry is not None and request_id not in registry:
                continue
            heads, bank_audit = _bank_for_task(args, plan, seed=int(row["seed"]))
            tasks.append(
                {
                    "request_id": request_id,
                    "row": row,
                    "heads": heads,
                    "bank_audit": bank_audit,
                    "specification": None,
                }
            )
    else:
        for row in rows:
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            registry_events = None if registry is None else registry.get(request_id, ())
            if registry is not None and not registry_events:
                continue
            specifications, _excluded = mechanism_continuations(
                row, tokenizer, mechanism="retrieval_anchor_localization"
            )
            for specification in specifications:
                roles = {str(value) for value in specification.get("anchor_roles", [])}
                if args.anchor_role is not None and str(args.anchor_role) not in roles:
                    continue
                matched_events: tuple[Mapping[str, Any] | None, ...]
                if registry_events is None:
                    matched_events = (None,)
                else:
                    matched_events = tuple(
                        event
                        for event in registry_events
                        if _registry_event_matches(
                            specification,
                            event,
                            match_mode=str(args.registry_anchor_match),
                        )
                    )
                for registry_event in matched_events:
                    grammar = str(
                        (
                            specification.get("target_grammar_class")
                            if registry_event is None
                            else registry_event.get("target_grammar_class")
                        )
                        or ""
                    )
                    if (
                        args.target_grammar_class is not None
                        and grammar != str(args.target_grammar_class)
                    ):
                        continue
                    routed = dict(specification)
                    routed["target_grammar_class"] = grammar
                    if registry_event is not None:
                        routed["registry_anchor_equivalence_id"] = str(
                            registry_event.get("anchor_equivalence_id", "")
                        )
                        routed["registry_anchor_roles"] = list(
                            registry_event.get("anchor_roles", [])
                        )
                        routed["target_retrieval_surface_variant"] = str(
                            registry_event.get(
                                "target_retrieval_surface_variant",
                                routed.get("target_retrieval_surface_variant", ""),
                            )
                        )
                    heads, bank_audit = _bank_for_task(
                        args, plan, seed=int(row["seed"])
                    )
                    tasks.append(
                        {
                            "request_id": request_id,
                            "row": row,
                            "heads": heads,
                            "bank_audit": bank_audit,
                            "specification": routed,
                        }
                    )
    tasks.sort(
        key=lambda task: (
            int(task["row"]["seed"]),
            str(task["request_id"]),
            str(
                ""
                if task["specification"] is None
                else task["specification"]["anchor_equivalence_id"]
            ),
        )
    )
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]
    return tasks


def _filter_rows(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = list(rows)
    if args.split != "all":
        selected = [row for row in selected if str(row.get("split")) == args.split]
    if args.seeds is not None:
        seeds = {int(value) for value in args.seeds}
        selected = [row for row in selected if int(row["seed"]) in seeds]
    if args.counts is not None:
        counts = {int(value) for value in args.counts}
        selected = [
            row
            for row in selected
            if len(row.get("gold_records", row.get("gold_pairs", []))) in counts
        ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Position-preserving token-state blanks for Native-thinking answer "
            "readout and targeted-retrieval trigger attribution"
        )
    )
    parser.add_argument("--mode", required=True, choices=["answer", "targeting"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    bank = parser.add_mutually_exclusive_group(required=True)
    bank.add_argument("--heads-json")
    bank.add_argument("--bank-plan", type=Path)
    parser.add_argument("--bank-condition", default="selected_bank")
    parser.add_argument(
        "--bank-size",
        type=int,
        help="Select one K when a frozen plan contains multiple bank sizes.",
    )
    parser.add_argument("--anchor-role")
    parser.add_argument("--target-grammar-class")
    parser.add_argument("--anchor-registry", type=Path)
    parser.add_argument(
        "--registry-anchor-match",
        choices=("transition", "exact"),
        default="transition",
        help=(
            "When an anchor registry is supplied, join either every localizer "
            "for the registered transition (legacy) or only the exact registered "
            "anchor_equivalence_id."
        ),
    )
    parser.add_argument("--split", choices=["discovery", "confirmation", "all"], default="confirmation")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--matched-control-repeats", type=int, default=3)
    parser.add_argument("--run-greedy", action="store_true")
    parser.add_argument(
        "--skip-target-score",
        action="store_true",
        help=(
            "Targeting mode only: capture selected-bank attention/OV metrics "
            "without a second teacher-forced city-span scoring forward."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if int(args.worker_count) < 1 or not 0 <= int(args.worker_index) < int(args.worker_count):
        raise ValueError("worker index/count must satisfy 0 <= index < count")
    if int(args.matched_control_repeats) < 1:
        raise ValueError("matched-control repeats must be positive")

    config = V5Config.load(args.config)
    plan, plan_sha = _load_plan(args)
    model, tokenizer, adapter = _load_model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _filter_rows(args, rows)
    tasks = _build_tasks(args, rows, tokenizer, plan)
    assigned = [
        task
        for index, task in enumerate(tasks)
        if index % int(args.worker_count) == int(args.worker_index)
    ]
    if not assigned:
        raise ValueError("No token-level tasks were assigned to this worker")
    conditions = tuple(
        args.conditions
        or (
            ANSWER_TOKEN_BLANK_CONDITIONS
            if args.mode == "answer"
            else TARGETING_TRACE_BLANK_CONDITIONS
        )
    )
    allowed = set(
        ANSWER_TOKEN_BLANK_CONDITIONS
        if args.mode == "answer"
        else TARGETING_TRACE_BLANK_CONDITIONS
    )
    unknown = sorted(set(conditions) - allowed)
    if unknown:
        raise ValueError(f"Unknown {args.mode} conditions: {unknown}")

    shard_dir = args.output / "shards"
    completed = 0
    skipped = 0
    output_rows = 0
    for task_index, task in enumerate(assigned, start=1):
        specification = task["specification"]
        anchor_id = None if specification is None else specification["anchor_equivalence_id"]
        shard = shard_dir / (
            _safe_stem(
                args.mode,
                task["request_id"],
                anchor_id,
                task["bank_audit"]["bank_sha256"],
            )
            + ".jsonl"
        )
        if args.resume and shard.exists():
            skipped += 1
            with shard.open("r", encoding="utf-8") as handle:
                output_rows += sum(1 for line in handle if line.strip())
            continue
        if shard.exists() and not args.resume:
            raise FileExistsError(f"Shard already exists: {shard}")
        if args.mode == "answer":
            results = run_answer_token_blank_trials(
                model,
                tokenizer,
                adapter,
                task["row"],
                heads=task["heads"],
                conditions=conditions,
                max_new_tokens=int(args.max_new_tokens),
            )
        else:
            results = []
            for condition in conditions:
                repeats = (
                    range(1, int(args.matched_control_repeats) + 1)
                    if condition.endswith("_matched_control")
                    else (1,)
                )
                for repeat in repeats:
                    results.extend(
                        run_targeting_trace_token_trials(
                            model,
                            tokenizer,
                            adapter,
                            task["row"],
                            specification,
                            heads=task["heads"],
                            conditions=[condition],
                            control_repeat=int(repeat),
                            score_target=not bool(args.skip_target_score),
                            run_greedy=bool(args.run_greedy),
                            max_new_tokens=int(args.max_new_tokens),
                        )
                    )
        for result in results:
            result.update(
                {
                    **task["bank_audit"],
                    "bank_plan_sha256": plan_sha,
                    "worker_index": int(args.worker_index),
                    "worker_count": int(args.worker_count),
                }
            )
        _atomic_jsonl(shard, results)
        completed += 1
        output_rows += len(results)
        print(
            f"[token-level {args.mode}] {task_index}/{len(assigned)} "
            f"request={task['request_id']} anchor={anchor_id} rows={len(results)}",
            flush=True,
        )

    code_paths = [
        Path(__file__).resolve(),
        SRC / "realistic_niah_v5" / "token_level_ablation.py",
        SRC / "realistic_niah_v5" / "causal.py",
        SRC / "realistic_niah_v5" / "count_stream.py",
        SRC / "realistic_niah_v5" / "encoding.py",
    ]
    manifest = {
        "schema_version": TOKEN_LEVEL_ABLATION_SCHEMA_VERSION,
        "mode": args.mode,
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "generations_sha256": hashlib.sha256(args.generations.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "bank_plan": None if args.bank_plan is None else str(args.bank_plan.resolve()),
        "bank_plan_sha256": plan_sha,
        "conditions": list(conditions),
        "matched_control_repeats": int(args.matched_control_repeats),
        "run_greedy": bool(args.run_greedy),
        "target_span_scoring": not bool(args.skip_target_score),
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "global_task_count": len(tasks),
        "assigned_task_count": len(assigned),
        "newly_completed_shards": completed,
        "resume_skipped_shards": skipped,
        "output_rows": output_rows,
        "code_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths
        },
        "intervention_contract": {
            "token_deletion_used": False,
            "sequence_length_preserved": True,
            "absolute_query_position_preserved": True,
            "blank_operation": "zero embedding and every post-block residual at registered source positions",
            "targeting_factorial": {
                "clean": [1, 1],
                "cumulative_trace_blank": [0, 1],
                "recent_transition_blank": [1, 0],
                "full_trace_blank": [0, 0],
            },
        },
    }
    _atomic_json(
        args.output / f"worker_{int(args.worker_index):02d}_manifest.json", manifest
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
