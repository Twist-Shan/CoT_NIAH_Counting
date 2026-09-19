#!/usr/bin/env python3
"""Fill V6 strict-format quotas from an outcome-blind reserve seed pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.generation import (  # noqa: E402
    GENERATION_SCHEMA_VERSION,
    generate_structured_enumeration,
    render_structured_prompt,
)
from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    sha256_file,
    validate_generation_contracts,
    write_jsonl,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    POOL_SCHEMA_VERSION,
    load_replacement_policy,
    resolve_replacement_panel,
    resolved_generation_records,
    role_contract,
)
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


RUN_SCHEMA_VERSION = "realistic_niah_v6_replacement_generation_run_v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _shard_name(row: Mapping[str, Any]) -> str:
    identity = (
        f"{row.get('stimulus_id')}|{row.get('seed')}|{row.get('gold_count')}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return (
        f"seed_{int(row['seed'])}__count_{int(row['gold_count'])}__"
        f"{digest}.json"
    )


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    spec = resolve_model_spec(args.model)
    model, tokenizer, _adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    model.eval()
    return model, tokenizer, spec


def _runtime_failures(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _merge_generations(
    generation_root: Path,
    *,
    config: V6Config,
    config_sha256: str,
    model_label: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in sorted((generation_root / "shards").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") == GENERATION_SCHEMA_VERSION
            and value.get("prompt_mode") == config.prompt_mode
            and value.get("model_label") == model_label
            and value.get("v6_config_sha256") == config_sha256
        ):
            merged.append(value)
    merged.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    if len({str(row["request_id"]) for row in merged}) != len(merged):
        raise RuntimeError("Merged V6 generation requests are not unique")
    write_jsonl(generation_root / "generations.jsonl", merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    parser.add_argument("--replacement-stimuli", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--seed-role", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--confirmation-freeze", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()

    config = V6Config.load(args.v6_config)
    policy = load_replacement_policy(args.replacement_policy, config)
    if args.seed_role == "confirmation":
        if args.confirmation_freeze is None:
            raise ValueError("Confirmation replacements require a discovery freeze")
        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=args.model,
        )
    pool_manifest_path = args.replacement_stimuli.parent / "manifest.json"
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    if pool_manifest.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Replacement stimulus pool manifest has the wrong schema")
    if pool_manifest.get("status") != "PASS_AMENDMENT_RESERVE_POOL":
        raise ValueError("Replacement stimulus pool did not pass construction")
    if pool_manifest.get("stimuli_sha256") != sha256_file(args.replacement_stimuli):
        raise ValueError("Replacement stimulus pool hash changed")
    if pool_manifest.get("replacement_policy_sha256") != sha256_file(
        args.replacement_policy
    ):
        raise ValueError("Replacement stimulus pool used a different policy")
    expected_stimulus_contract = {
        "design_variant": config.design_variant,
        "counts": list(map(int, config.counts)),
        "original_discovery_seeds": list(map(int, config.discovery_seeds)),
        "original_confirmation_seeds": list(map(int, config.confirmation_seeds)),
    }
    if pool_manifest.get("v6_stimulus_contract") != expected_stimulus_contract:
        raise ValueError("Replacement stimulus pool changed the V6 stimulus contract")

    original_seeds, replacement_pool, quota = role_contract(
        policy, config, args.seed_role
    )
    pool_rows = [
        row
        for row in read_jsonl(args.replacement_stimuli)
        if int(row.get("seed", -1)) in set(replacement_pool)
        and str(row.get("split")) == args.seed_role
    ]
    pool_by_key = {
        (int(row["seed"]), int(row["gold_count"])): row for row in pool_rows
    }
    expected_pool_keys = {
        (seed, count) for seed in replacement_pool for count in config.counts
    }
    if set(pool_by_key) != expected_pool_keys or len(pool_rows) != len(
        expected_pool_keys
    ):
        raise ValueError("Replacement stimulus role pool is incomplete or duplicated")

    args.output.mkdir(parents=True, exist_ok=True)
    args.generation_root.mkdir(parents=True, exist_ok=True)
    (args.generation_root / "shards").mkdir(parents=True, exist_ok=True)
    failure_path = args.output / "runtime_failures.jsonl"
    config_hash = sha256_file(args.v6_config)
    generations_path = args.generation_root / "generations.jsonl"
    if not generations_path.is_file():
        raise FileNotFoundError(
            "Original V6 generation panel must exist before supplementation"
        )
    merged = read_jsonl(generations_path)
    validate_generation_contracts(
        merged,
        config,
        model_label=args.model,
        config_sha256=config_hash,
    )
    original_keys = {
        (int(row["seed"]), int(row["gold_count"]))
        for row in merged
        if int(row.get("seed", -1)) in set(original_seeds)
    }
    expected_original_keys = {
        (seed, count) for seed in original_seeds for count in config.counts
    }
    if original_keys != expected_original_keys:
        raise ValueError("Original V6 panel is incomplete; do not hide missing originals")

    failures = _runtime_failures(failure_path)
    resolution = resolve_replacement_panel(
        merged,
        config=config,
        model_label=args.model,
        seed_role=args.seed_role,
        policy=policy,
        runtime_failures=failures,
    )
    model = tokenizer = spec = None
    consecutive_runtime_failures = 0
    while not bool(resolution["complete"]):
        candidates = list(resolution["next_candidates"])
        if not candidates:
            raise RuntimeError(
                "The frozen-amendment replacement pool is exhausted before quota; "
                "fail closed and amend the protocol explicitly"
            )
        if model is None:
            model, tokenizer, spec = _load_model(args)
        for candidate in candidates:
            seed = int(candidate["seed"])
            count = int(candidate["gold_count"])
            stimulus = pool_by_key[(seed, count)]
            shard_path = args.generation_root / "shards" / _shard_name(stimulus)
            if shard_path.exists():
                raise RuntimeError(
                    f"Replacement shard exists but was absent from merge: {shard_path}"
                )
            try:
                prompt = render_structured_prompt(
                    stimulus,
                    tokenizer=tokenizer,
                    model_spec=spec,
                    prompt_mode=config.prompt_mode,
                )
                value = generate_structured_enumeration(
                    model,
                    tokenizer,
                    prompt,
                    decoding=config.decoding,
                    sampling_seed=seed,
                )
                value["v6_config_sha256"] = config_hash
                value["source_stimulus_sha256"] = hashlib.sha256(
                    json.dumps(
                        stimulus,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                value["v6_replacement_candidate"] = True
                value["v6_replacement_seed_role"] = args.seed_role
                value["v6_replacement_policy_sha256"] = sha256_file(
                    args.replacement_policy
                )
                value["v6_replacement_intervention_outcomes_available"] = False
                _atomic_json(shard_path, value)
                consecutive_runtime_failures = 0
                strict = bool(value["trace_parse"]["strict_causal_eligible"])
                print(
                    f"[v6 replacement generate] role={args.seed_role} "
                    f"seed={seed} count={count} strict={strict}",
                    flush=True,
                )
            except Exception as error:
                consecutive_runtime_failures += 1
                failures.append(
                    {
                        "schema_version": (
                            "realistic_niah_v6_replacement_runtime_failure_v1"
                        ),
                        "model_label": args.model,
                        "prompt_mode": config.prompt_mode,
                        "split": args.seed_role,
                        "seed": seed,
                        "gold_count": count,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "intervention_outcomes_available": False,
                    }
                )
                _atomic_jsonl(failure_path, failures)
                print(
                    f"[v6 replacement runtime failure] seed={seed} count={count} "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if consecutive_runtime_failures >= 3:
                    raise RuntimeError(
                        "Three consecutive replacement generations failed at runtime; "
                        "abort instead of consuming the reserve pool"
                    ) from error
        merged = _merge_generations(
            args.generation_root,
            config=config,
            config_sha256=config_hash,
            model_label=args.model,
        )
        validate_generation_contracts(
            merged,
            config,
            model_label=args.model,
            config_sha256=config_hash,
        )
        resolution = resolve_replacement_panel(
            merged,
            config=config,
            model_label=args.model,
            seed_role=args.seed_role,
            policy=policy,
            runtime_failures=failures,
        )
        print(
            json.dumps(
                {
                    "selected_per_count": resolution["selected_per_count"],
                    "shortfalls": resolution["shortfalls"],
                    "replacement_count": resolution["replacement_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_path = args.output / "selected_cells.jsonl"
    mapping_path = args.output / "replacement_mapping.jsonl"
    attempts_path = args.output / "attempt_ledger.jsonl"
    _atomic_jsonl(selected_path, resolution["selected_cells"])
    _atomic_jsonl(mapping_path, resolution["replacement_mapping"])
    _atomic_jsonl(attempts_path, resolution["attempt_ledger"])
    resolved = resolved_generation_records(
        merged,
        config,
        registry_path=selected_path,
        model_label=args.model,
    )
    if len(resolved) != quota * len(config.counts):
        raise RuntimeError("Final resolved V6 replacement panel has the wrong size")
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "PASS_STRICT_QUOTA_FILLED",
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "seed_role": args.seed_role,
        "original_seeds": list(original_seeds),
        "replacement_seed_pool": list(replacement_pool),
        "counts": list(map(int, config.counts)),
        "quota_per_count": quota,
        "resolved_rows": len(resolved),
        "replacement_count": int(resolution["replacement_count"]),
        "selected_per_count": resolution["selected_per_count"],
        "strict_parser_recomputed": True,
        "protocol_change_class": "prospective_user_authorized_amendment",
        "protocol_timing_context": {
            "original_generation_outputs_existed_before_amendment": True,
            "some_original_foundation_or_source_write_artifacts_may_have_existed": True,
            "reserve_seed_order_frozen_before_any_reserve_model_output": True,
            "replacement_mapping_frozen_before_downstream_behavior_interventions": True,
            "original_failure_rate_may_have_informed_reserve_pool_capacity_only": True,
        },
        "selection_inputs": [
            "generation_presence_or_runtime_status",
            "fresh_v6_parse.strict_causal_eligible",
            "ascending_frozen_amendment_seed_order",
        ],
        "intervention_outcomes_read": False,
        "hidden_states_read": False,
        "attention_scores_read": False,
        "head_ranks_read": False,
        "all_sample_panel_replaced": False,
        "replacement_policy": str(args.replacement_policy.resolve()),
        "replacement_policy_sha256": sha256_file(args.replacement_policy),
        "replacement_stimuli": str(args.replacement_stimuli.resolve()),
        "replacement_stimuli_sha256": sha256_file(args.replacement_stimuli),
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
        "v6_config": str(args.v6_config.resolve()),
        "v6_config_sha256": config_hash,
        "generations": str(generations_path.resolve()),
        "generations_sha256": sha256_file(generations_path),
        "outputs": {
            selected_path.name: sha256_file(selected_path),
            mapping_path.name: sha256_file(mapping_path),
            attempts_path.name: sha256_file(attempts_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if failure_path.is_file():
        manifest["outputs"][failure_path.name] = sha256_file(failure_path)
    _atomic_json(args.output / "manifest.json", manifest)
    (args.output / f"{args.seed_role}.COMPLETE").write_text(
        "PASS\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
