#!/usr/bin/env python3
"""Fill a true-source-coherent broad or native-loop seed panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.generation import (  # noqa: E402
    generate_structured_enumeration,
    render_structured_prompt,
)
from realistic_niah_v6.kernel import install_v6_kernel_adapters  # noqa: E402
from realistic_niah_v6.pipeline import (  # noqa: E402
    read_jsonl,
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    POOL_SCHEMA_VERSION,
    load_coherent_broad_policy,
    load_replacement_policy,
    resolve_coherent_broad_panel,
    resolve_coherent_native_loop_panel,
    resolved_generation_records,
    role_contract,
)
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_contract as load_answer_trace_extension_contract,
    resolve_coherent_answer_trace_panel,
)
from scripts.run_realistic_niah_v6_replacement_generation import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _load_model,
    _merge_generations,
    _runtime_failures,
    _shard_name,
)


RUN_SCHEMA_VERSION = "realistic_niah_v6_coherent_broad_generation_run_v1"
NATIVE_LOOP_RUN_SCHEMA_VERSION = (
    "realistic_niah_v6_coherent_native_loop_generation_run_v1"
)
ANSWER_TRACE_RUN_SCHEMA_VERSION = (
    "realistic_niah_v6_coherent_answer_trace_generation_run_v1"
)


def _replacement_policy_lineage_names(
    policy_path: Path, policy: dict[str, Any]
) -> set[str]:
    """Return the current policy and every hash-validated ancestor filename."""

    names = {policy_path.name}
    for key in (
        "pool_exhaustion_amendment",
        "confirmation_pool_exhaustion_amendment",
    ):
        amendment = policy.get(key)
        if not isinstance(amendment, dict):
            continue
        base = Path(str(amendment.get("base_policy", ""))).name
        if base:
            names.add(base)
    return names


def _load_native_loop_policy(
    path: Path,
    *,
    replacement_policy: Path,
    replacement_policy_value: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version")
        != "realistic_niah_v6_coherent_native_loop_replacement_policy_v1"
        or value.get("status") != "FROZEN_USER_AUTHORIZED_FAILURE_TOP_UP"
    ):
        raise ValueError("Coherent native-loop policy is not frozen")
    declared_base = Path(str(value.get("base_replacement_policy", ""))).name
    if declared_base not in _replacement_policy_lineage_names(
        replacement_policy, replacement_policy_value
    ):
        raise ValueError("Native-loop policy names a different base policy")
    if list(map(int, value.get("required_counts_per_slot", ()))) != list(range(2, 11)):
        raise ValueError("Native-loop policy changed the required count trajectory")
    if value.get("statistical_identity") != "true_source_seed" or value.get(
        "seed_aliasing"
    ) is not False:
        raise ValueError("Native-loop policy permits seed aliasing")
    return value


def _validate_pool(
    *,
    replacement_stimuli: Path,
    replacement_policy: Path,
    config: V6Config,
) -> dict[str, Any]:
    manifest_path = replacement_stimuli.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Replacement stimulus pool manifest has the wrong schema")
    if manifest.get("status") != "PASS_AMENDMENT_RESERVE_POOL":
        raise ValueError("Replacement stimulus pool did not pass construction")
    if manifest.get("stimuli_sha256") != sha256_file(replacement_stimuli):
        raise ValueError("Replacement stimulus pool hash changed")
    if manifest.get("replacement_policy_sha256") != sha256_file(
        replacement_policy
    ):
        raise ValueError("Replacement stimulus pool used a different base policy")
    expected_contract = {
        "design_variant": config.design_variant,
        "counts": list(map(int, config.counts)),
        "original_discovery_seeds": list(map(int, config.discovery_seeds)),
        "original_confirmation_seeds": list(map(int, config.confirmation_seeds)),
    }
    if manifest.get("v6_stimulus_contract") != expected_contract:
        raise ValueError("Replacement stimulus pool changed the V6 contract")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    parser.add_argument(
        "--panel-kind",
        choices=("broad", "native_loop", "answer_trace"),
        default="broad",
    )
    parser.add_argument("--coherent-broad-policy", type=Path)
    parser.add_argument("--coherent-native-loop-policy", type=Path)
    parser.add_argument("--answer-trace-extension-contract", type=Path)
    parser.add_argument("--replacement-stimuli", type=Path, required=True)
    parser.add_argument("--base-cohort-registry", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "k_selection_discovery",
            "confirmation",
            "native_loop_discovery",
            "native_loop_confirmation",
            "answer_trace_confirmation",
        ),
        required=True,
    )
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
    if args.panel_kind == "broad":
        if (
            args.coherent_broad_policy is None
            or args.coherent_native_loop_policy is not None
            or args.answer_trace_extension_contract is not None
        ):
            raise ValueError("Broad panels require only --coherent-broad-policy")
        if args.phase not in {"k_selection_discovery", "confirmation"}:
            raise ValueError("Broad panel received a native-loop phase")
        coherent_policy = load_coherent_broad_policy(args.coherent_broad_policy)
        coherent_policy_path = args.coherent_broad_policy
    elif args.panel_kind == "native_loop":
        if (
            args.coherent_native_loop_policy is None
            or args.coherent_broad_policy is not None
            or args.answer_trace_extension_contract is not None
        ):
            raise ValueError(
                "Native-loop panels require only --coherent-native-loop-policy"
            )
        if args.phase not in {"native_loop_discovery", "native_loop_confirmation"}:
            raise ValueError("Native-loop panel received a broad phase")
        coherent_policy = _load_native_loop_policy(
            args.coherent_native_loop_policy,
            replacement_policy=args.replacement_policy,
            replacement_policy_value=policy,
        )
        coherent_policy_path = args.coherent_native_loop_policy
    else:
        if (
            args.answer_trace_extension_contract is None
            or args.coherent_broad_policy is not None
            or args.coherent_native_loop_policy is not None
        ):
            raise ValueError(
                "Answer/trace panels require only "
                "--answer-trace-extension-contract"
            )
        if args.phase != "answer_trace_confirmation":
            raise ValueError("Answer/trace panel requires its confirmation phase")
        coherent_policy = load_answer_trace_extension_contract(
            args.answer_trace_extension_contract
        )
        coherent_policy_path = args.answer_trace_extension_contract
    declared_base = Path(
        str(
            coherent_policy.get(
                "base_replacement_policy",
                coherent_policy.get("cohort", {}).get("base_replacement_policy", ""),
            )
        )
    )
    if declared_base.name not in _replacement_policy_lineage_names(
        args.replacement_policy, policy
    ):
        raise ValueError("Coherent policy names a different base replacement policy")
    seed_role = (
        "discovery"
        if args.phase in {"k_selection_discovery", "native_loop_discovery"}
        else "confirmation"
    )
    if seed_role == "confirmation":
        if args.confirmation_freeze is None:
            raise ValueError("Confirmation coherent replacement requires a freeze")
        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=args.model,
        )
    elif args.confirmation_freeze is not None:
        raise ValueError("Discovery coherent replacement must not open confirmation")

    install_v6_kernel_adapters()
    from realistic_niah_v5.count_stream import NativeCountMechanismSpec

    mechanism = NativeCountMechanismSpec.load(args.mechanism_config)
    if tuple(map(int, mechanism.development_seeds)) != tuple(
        map(int, config.discovery_seeds)
    ) or tuple(map(int, mechanism.confirmation_seeds)) != tuple(
        map(int, config.confirmation_seeds)
    ):
        raise ValueError("Mechanism and V6 seed registries differ")
    if config.prompt_mode not in str(mechanism.experiment_id):
        raise ValueError("Mechanism config has the wrong enumeration mode")

    pool_manifest = _validate_pool(
        replacement_stimuli=args.replacement_stimuli,
        replacement_policy=args.replacement_policy,
        config=config,
    )
    original_seeds, replacement_pool, _quota = role_contract(
        policy, config, seed_role
    )
    pool_rows = [
        row
        for row in read_jsonl(args.replacement_stimuli)
        if int(row.get("seed", -1)) in set(replacement_pool)
        and str(row.get("split")) == seed_role
    ]
    pool_by_key = {
        (int(row["seed"]), int(row["gold_count"])): row for row in pool_rows
    }
    expected_pool_keys = {
        (seed, count) for seed in replacement_pool for count in config.counts
    }
    if set(pool_by_key) != expected_pool_keys:
        raise ValueError("Replacement stimulus role pool is incomplete or duplicated")

    args.output.mkdir(parents=True, exist_ok=True)
    args.generation_root.mkdir(parents=True, exist_ok=True)
    (args.generation_root / "shards").mkdir(parents=True, exist_ok=True)
    generation_path = args.generation_root / "generations.jsonl"
    if not generation_path.is_file():
        raise FileNotFoundError("Original/cell-resolved V6 generations are missing")
    config_hash = sha256_file(args.v6_config)
    merged = read_jsonl(generation_path)
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
    if original_keys != {
        (seed, count) for seed in original_seeds for count in config.counts
    }:
        raise ValueError("Original V6 panel is incomplete")
    base_registry = read_jsonl(args.base_cohort_registry)
    failure_path = args.output / "runtime_failures.jsonl"
    failures = _runtime_failures(failure_path)

    def resolve_panel(active_rows: list[dict[str, Any]]) -> dict[str, Any]:
        common = {
            "config": config,
            "model_label": args.model,
            "seed_role": seed_role,
            "policy": policy,
            "base_registry": base_registry,
            "runtime_failures": failures,
        }
        if args.panel_kind == "native_loop":
            return resolve_coherent_native_loop_panel(active_rows, **common)
        if args.panel_kind == "answer_trace":
            common.pop("seed_role")
            return resolve_coherent_answer_trace_panel(active_rows, **common)
        return resolve_coherent_broad_panel(
            active_rows,
            mechanism=mechanism,
            phase=args.phase,
            **common,
        )

    resolution = resolve_panel(merged)
    model = tokenizer = spec = None
    consecutive_runtime_failures = 0
    while not bool(resolution["complete"]):
        if bool(resolution["pool_exhausted"]):
            raise RuntimeError(
                "Coherent panel reserve pool exhausted; fail closed and amend "
                "the protocol explicitly"
            )
        candidates = list(resolution["next_candidates"])
        if not candidates:
            raise RuntimeError("Incomplete coherent panel has no next candidate")
        if model is None:
            model, tokenizer, spec = _load_model(args)
        for candidate in candidates:
            seed = int(candidate["seed"])
            count = int(candidate["gold_count"])
            stimulus = pool_by_key[(seed, count)]
            shard_path = args.generation_root / "shards" / _shard_name(stimulus)
            if shard_path.exists():
                raise RuntimeError(
                    f"Reserve shard exists but was absent from merge: {shard_path}"
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
                value["v6_replacement_seed_role"] = seed_role
                value["v6_replacement_policy_sha256"] = sha256_file(
                    args.replacement_policy
                )
                value["v6_coherent_panel_candidate"] = True
                value["v6_coherent_panel_kind"] = args.panel_kind
                value["v6_coherent_panel_phase"] = args.phase
                value["v6_coherent_panel_policy_sha256"] = sha256_file(
                    coherent_policy_path
                )
                value["v6_replacement_intervention_outcomes_available"] = False
                _atomic_json(shard_path, value)
                consecutive_runtime_failures = 0
                print(
                    f"[v6 coherent {args.panel_kind} generate] phase={args.phase} "
                    f"seed={seed} count={count} "
                    f"strict={bool(value['trace_parse']['strict_causal_eligible'])}",
                    flush=True,
                )
            except Exception as error:
                consecutive_runtime_failures += 1
                failures.append(
                    {
                        "schema_version": (
                            "realistic_niah_v6_coherent_panel_runtime_failure_v1"
                        ),
                        "model_label": args.model,
                        "prompt_mode": config.prompt_mode,
                        "split": seed_role,
                        "phase": args.phase,
                        "seed": seed,
                        "gold_count": count,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                        "intervention_outcomes_available": False,
                    }
                )
                _atomic_jsonl(failure_path, failures)
                print(
                    f"[v6 coherent {args.panel_kind} runtime failure] seed={seed} "
                    f"count={count} {type(error).__name__}: {error}",
                    flush=True,
                )
                if consecutive_runtime_failures >= 3:
                    raise RuntimeError(
                        "Three consecutive coherent panel generations failed; abort"
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
        resolution = resolve_panel(merged)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "affected_slots": resolution["affected_slots"],
                    "accepted_replacement_seed_by_slot": resolution[
                        "accepted_replacement_seed_by_slot"
                    ],
                    "next_candidates": resolution["next_candidates"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_path = args.output / "selected_cells.jsonl"
    mapping_path = args.output / "coherent_mapping.jsonl"
    attempts_path = args.output / "attempt_ledger.jsonl"
    _atomic_jsonl(selected_path, resolution["selected_cells"])
    _atomic_jsonl(mapping_path, resolution["coherent_mapping"])
    _atomic_jsonl(attempts_path, resolution["attempt_ledger"])
    materialized = resolved_generation_records(
        merged,
        config,
        registry_path=selected_path,
        model_label=args.model,
    )
    expected_rows = len(original_seeds) * len(config.counts)
    if len(materialized) != expected_rows:
        raise RuntimeError("Coherent panel registry changed the formal panel size")

    manifest = {
        "schema_version": (
            RUN_SCHEMA_VERSION
            if args.panel_kind == "broad"
            else NATIVE_LOOP_RUN_SCHEMA_VERSION
            if args.panel_kind == "native_loop"
            else ANSWER_TRACE_RUN_SCHEMA_VERSION
        ),
        "status": (
            "PASS_TRUE_SOURCE_COHERENT_BROAD_PANEL"
            if args.panel_kind == "broad"
            else "PASS_TRUE_SOURCE_COHERENT_NATIVE_LOOP_PANEL"
            if args.panel_kind == "native_loop"
            else "PASS_TRUE_SOURCE_COHERENT_ANSWER_TRACE_PANEL"
        ),
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "panel_kind": args.panel_kind,
        "seed_role": seed_role,
        "phase": args.phase,
        "affected_slots": resolution["affected_slots"],
        "replacement_seed_count": resolution["replacement_seed_count"],
        "accepted_replacement_seed_by_slot": resolution[
            "accepted_replacement_seed_by_slot"
        ],
        "required_counts_by_slot": resolution["required_counts_by_slot"],
        "resolved_rows": len(materialized),
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "successful_original_cells_may_be_replaced_for_seed_coherence": True,
        "strict_parser_recomputed": True,
        "selection_inputs": [
            "generation_presence_or_runtime_status",
            "fresh_v6_parse.strict_causal_eligible",
            "ascending_role_specific_reserve_seed_order",
            "source_request_uniqueness",
        ],
        "intervention_outcomes_read": False,
        "hidden_states_read": False,
        "attention_scores_read": False,
        "head_ranks_read": False,
        "base_cohort_registry": str(args.base_cohort_registry.resolve()),
        "base_cohort_registry_sha256": sha256_file(args.base_cohort_registry),
        "replacement_policy": str(args.replacement_policy.resolve()),
        "replacement_policy_sha256": sha256_file(args.replacement_policy),
        "coherent_policy": str(coherent_policy_path.resolve()),
        "coherent_policy_sha256": sha256_file(coherent_policy_path),
        "replacement_stimuli": str(args.replacement_stimuli.resolve()),
        "replacement_stimuli_sha256": sha256_file(args.replacement_stimuli),
        "pool_manifest_sha256": sha256_file(
            args.replacement_stimuli.parent / "manifest.json"
        ),
        "pool_status": pool_manifest["status"],
        "v6_config": str(args.v6_config.resolve()),
        "v6_config_sha256": config_hash,
        "mechanism_config": str(args.mechanism_config.resolve()),
        "mechanism_config_sha256": sha256_file(args.mechanism_config),
        "generations": str(generation_path.resolve()),
        "generations_sha256": sha256_file(generation_path),
        "outputs": {
            selected_path.name: sha256_file(selected_path),
            mapping_path.name: sha256_file(mapping_path),
            attempts_path.name: sha256_file(attempts_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.panel_kind == "broad":
        manifest.update(
            {
                "coherent_broad_policy": str(coherent_policy_path.resolve()),
                "coherent_broad_policy_sha256": sha256_file(
                    coherent_policy_path
                ),
            }
        )
    elif args.panel_kind == "native_loop":
        manifest.update(
            {
                "coherent_native_loop_policy": str(
                    coherent_policy_path.resolve()
                ),
                "coherent_native_loop_policy_sha256": sha256_file(
                    coherent_policy_path
                ),
                "required_native_loop_counts": list(range(2, 11)),
            }
        )
    else:
        manifest.update(
            {
                "answer_trace_extension_contract": str(
                    coherent_policy_path.resolve()
                ),
                "answer_trace_extension_contract_sha256": sha256_file(
                    coherent_policy_path
                ),
                "required_answer_trace_counts": list(range(1, 11)),
                "protocol_relation": coherent_policy["protocol_relation"],
            }
        )
    if failure_path.is_file():
        manifest["outputs"][failure_path.name] = sha256_file(failure_path)
    _atomic_json(args.output / "manifest.json", manifest)
    (args.output / f"{args.phase}.COMPLETE").write_text(
        "PASS\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
