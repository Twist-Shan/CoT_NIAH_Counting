#!/usr/bin/env python3
"""Run the isolated V6 index/bullet replication pipeline.

The V6 orchestration layer changes prompt/container/parser contracts only.  It
installs process-local adapters around the audited V5 numerical kernels; it
never rewrites V5 source files or V5 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v6.pipeline import (  # noqa: E402
    EXPECTED_SOURCE_STIMULI_SHA256,
    audit_generation_panel,
    audit_stimulus_panel,
    parse_records,
    read_jsonl,
    registered_records,
    sha256_file,
    validate_generation_contracts,
    write_jsonl,
)
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.replacement import (  # noqa: E402
    resolved_generation_records,
)
from realistic_niah_v6.suite import (  # noqa: E402
    EXPERIMENTS,
    discovery_ledger_template,
    freeze_confirmation,
    suite_document,
    validate_confirmation_freeze,
)
from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_contract as load_answer_trace_extension_contract,
    model_contract as answer_trace_model_contract,
    validate_pair_registry as validate_answer_query_pair_registry,
)


DEFAULT_STIMULI = (
    ROOT / "work" / "nonthinking_report_filestream_stage3" / "stimuli.jsonl"
)
DEFAULT_INDEX_CONFIG = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_paths": (run("status", "--short") or "").splitlines(),
    }


def _runtime_base(started: float) -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "elapsed_seconds": time.perf_counter() - started,
        "git": _git_state(),
    }


def _config(path: str | Path) -> V6Config:
    return V6Config.load(path)


def _seed_role_rows(
    rows: Iterable[Mapping[str, Any]], *, config: V6Config, seed_role: str
) -> list[dict[str, Any]]:
    selected = registered_records(rows, config)
    if seed_role == "discovery":
        seeds = set(config.discovery_seeds)
    elif seed_role == "confirmation":
        seeds = set(config.confirmation_seeds)
    elif seed_role == "all":
        seeds = set(config.all_seeds)
    else:
        raise ValueError(f"Unknown seed role: {seed_role}")
    return [row for row in selected if int(row["seed"]) in seeds]


def _validate_confirmation_freeze(
    path: Path | None, *, config: V6Config, model_label: str
) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "Confirmation is locked. Supply --confirmation-freeze after all "
            "discovery choices have been hash-frozen."
        )
    return validate_confirmation_freeze(
        path,
        prompt_mode=config.prompt_mode,
        model_label=model_label,
    )


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    model.eval()
    return model, tokenizer, adapter, spec


def _generation_shard_name(row: Mapping[str, Any]) -> str:
    identity = (
        f"{row.get('stimulus_id')}|{row.get('seed')}|"
        f"{row.get('gold_count', len(row.get('gold_pairs', ())))}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"seed_{int(row['seed'])}__count_{int(row['gold_count'])}__{digest}.json"


def command_preflight(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config = _config(args.config)
    source_hash = sha256_file(args.stimuli)
    rows = read_jsonl(args.stimuli)
    audit = audit_stimulus_panel(rows, config)
    audit.update(
        {
            "source_path": str(args.stimuli.resolve()),
            "source_rows": len(rows),
            "source_sha256": source_hash,
            "expected_source_sha256": EXPECTED_SOURCE_STIMULI_SHA256,
            "source_hash_matches": source_hash == EXPECTED_SOURCE_STIMULI_SHA256,
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "resolved_config": config.to_dict(),
            "runtime": _runtime_base(started),
        }
    )
    if not audit["source_hash_matches"]:
        audit["status"] = "FAIL"
        audit.setdefault("errors", []).append("frozen source stimuli SHA256 changed")
    _atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


def command_generate(args: argparse.Namespace) -> None:
    from realistic_niah_v6.generation import (
        GENERATION_SCHEMA_VERSION,
        generate_structured_enumeration,
        render_structured_prompt,
    )

    started = time.perf_counter()
    config = _config(args.config)
    if args.model not in config.model_labels:
        raise ValueError(f"Model {args.model!r} is outside the frozen V6 registry")
    stimuli_hash = sha256_file(args.stimuli)
    if stimuli_hash != EXPECTED_SOURCE_STIMULI_SHA256:
        raise ValueError(
            "V6 generation refuses a stimuli file whose SHA256 differs from "
            "the frozen V4.4 source"
        )
    if args.seed_role == "confirmation":
        _validate_confirmation_freeze(
            args.confirmation_freeze, config=config, model_label=args.model
        )
    rows = _seed_role_rows(
        read_jsonl(args.stimuli), config=config, seed_role=args.seed_role
    )
    if args.counts:
        requested = tuple(sorted(set(int(value) for value in args.counts)))
        unsupported = sorted(set(requested) - set(config.counts))
        if unsupported:
            raise ValueError(f"Counts outside frozen grid: {unsupported}")
        rows = [row for row in rows if int(row["gold_count"]) in set(requested)]
    rows.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No registered V6 stimuli remain after filtering")

    output = args.output
    shards = output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _adapter, model_spec = _load_model(args)
    config_hash = sha256_file(args.config)
    captured: list[dict[str, Any]] = []
    for row_index, stimulus in enumerate(rows, start=1):
        shard_path = shards / _generation_shard_name(stimulus)
        if shard_path.exists() and not args.overwrite:
            value = json.loads(shard_path.read_text(encoding="utf-8"))
            if (
                value.get("schema_version") != GENERATION_SCHEMA_VERSION
                or value.get("prompt_mode") != config.prompt_mode
                or value.get("model_label") != args.model
                or value.get("stimulus_id") != stimulus.get("stimulus_id")
                or value.get("v6_config_sha256") != config_hash
            ):
                raise RuntimeError(
                    f"Incompatible generation shard {shard_path}; use --overwrite "
                    "or a new output directory"
                )
            action = "reused"
        else:
            prompt = render_structured_prompt(
                stimulus,
                tokenizer=tokenizer,
                model_spec=model_spec,
                prompt_mode=config.prompt_mode,
            )
            value = generate_structured_enumeration(
                model,
                tokenizer,
                prompt,
                decoding=config.decoding,
                sampling_seed=int(stimulus["seed"]),
            )
            value["v6_config_sha256"] = config_hash
            value["source_stimulus_sha256"] = hashlib.sha256(
                json.dumps(
                    stimulus, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()
            _atomic_json(shard_path, value)
            action = "generated"
        captured.append(value)
        parsed = value["trace_parse"]
        print(
            f"[v6 generate] {row_index}/{len(rows)} {action} "
            f"seed={stimulus['seed']} count={stimulus['gold_count']} "
            f"strict={bool(parsed['strict_causal_eligible'])} "
            f"elapsed={float(value.get('elapsed_seconds', 0.0)):.2f}s",
            flush=True,
        )

    # Merge every compatible shard already present, allowing discovery and a
    # later confirmation invocation to accumulate without overwriting either.
    merged: list[dict[str, Any]] = []
    for shard_path in sorted(shards.glob("*.json")):
        value = json.loads(shard_path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") == GENERATION_SCHEMA_VERSION
            and value.get("prompt_mode") == config.prompt_mode
            and value.get("model_label") == args.model
            and value.get("v6_config_sha256") == config_hash
        ):
            merged.append(value)
    merged.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    merged_path = output / "generations.jsonl"
    write_jsonl(merged_path, merged)
    audit = audit_generation_panel(
        captured,
        config,
        model_label=args.model,
        require_complete=args.limit is None and not args.counts,
        seed_role=args.seed_role,
        config_sha256=config_hash,
    )
    manifest = {
        "schema_version": "realistic_niah_v6_generation_run_v1",
        "status": audit["status"],
        "prompt_mode": config.prompt_mode,
        "model_label": args.model,
        "model_spec": {
            "model_id": model_spec.model_id,
            "revision": model_spec.revision,
            "family": model_spec.family,
            "loader_class": model_spec.loader_class,
            "chat_template_control": model_spec.chat_template_control,
        },
        "load_settings": {
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "attention_backend": args.attention_backend,
            "cache_dir": str(args.cache_dir.resolve()),
        },
        "seed_role": args.seed_role,
        "selected_rows": len(rows),
        "merged_rows": len(merged),
        "restartable_shards": True,
        "generation_audit": audit,
        "stimuli_path": str(args.stimuli.resolve()),
        "stimuli_sha256": stimuli_hash,
        "config_path": str(args.config.resolve()),
        "config_sha256": config_hash,
        "generations_path": str(merged_path.resolve()),
        "runtime": _runtime_base(started),
    }
    _atomic_json(output / f"manifest_{args.seed_role}.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


def command_parse(args: argparse.Namespace) -> None:
    rows = parse_records(read_jsonl(args.input), include_input=not args.compact)
    count = write_jsonl(args.output, rows)
    print(f"[v6 parse] wrote {count} rows to {args.output}")


def command_audit_generations(args: argparse.Namespace) -> None:
    config = _config(args.config)
    audit = audit_generation_panel(
        read_jsonl(args.input),
        config,
        model_label=args.model,
        require_complete=not args.allow_partial,
        seed_role=args.seed_role,
        config_sha256=sha256_file(args.config),
    )
    _atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


def _analysis_rows(args: argparse.Namespace, config: V6Config) -> list[dict[str, Any]]:
    source_rows = read_jsonl(args.generations)
    validate_generation_contracts(
        source_rows,
        config,
        model_label=args.model,
        config_sha256=sha256_file(args.config),
    )
    registry_paths: list[Path] = []
    for raw in (
        ([args.cohort_registry] if args.cohort_registry is not None else [])
        + list(getattr(args, "additional_cohort_registry", ()))
    ):
        resolved = raw.resolve()
        if resolved not in registry_paths:
            registry_paths.append(resolved)
    if registry_paths:
        if args.include_nonstrict:
            raise ValueError(
                "A resolved replacement registry is strict-only; all-sample "
                "analysis must retain the original preregistered panel"
            )
        rows = []
        registry_roles: list[str] = []
        for registry_path in registry_paths:
            resolved_rows = resolved_generation_records(
                source_rows,
                config,
                registry_path=registry_path,
                model_label=args.model,
            )
            roles = {str(row["split"]) for row in resolved_rows}
            if len(roles) != 1:
                raise ValueError("Each resolved cohort registry must have one role")
            registry_roles.append(next(iter(roles)))
            rows.extend(resolved_rows)
        if len(registry_roles) != len(set(registry_roles)):
            raise ValueError("At most one resolved cohort registry is allowed per role")
        request_ids = [str(row["request_id"]) for row in rows]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Resolved cohort registries reuse one generation request")
        roles = {str(row["split"]) for row in rows}
        if args.seed_role == "all" and roles != {"discovery", "confirmation"}:
            raise ValueError(
                "A resolved all-seed analysis requires discovery and confirmation "
                "cohort registries"
            )
        if args.seed_role != "all" and roles != {args.seed_role}:
            raise ValueError("Resolved cohort seed role differs from --seed-role")
        rows.sort(
            key=lambda row: (
                0 if str(row["split"]) == "discovery" else 1,
                int(row["gold_count"]),
                int(row.get("v6_analysis_slot_seed", row["seed"])),
            )
        )
    else:
        rows = _seed_role_rows(source_rows, config=config, seed_role=args.seed_role)
        rows = [
            row
            for row in rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
        if not args.include_nonstrict:
            rows = registered_records(
                rows, config, model_label=args.model, formal_only=True
            )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No V6 generation rows match the requested analysis cohort")
    return rows


def _cohort_registry_manifest(args: argparse.Namespace) -> dict[str, Any]:
    paths: list[Path] = []
    for raw in (
        ([args.cohort_registry] if args.cohort_registry is not None else [])
        + list(getattr(args, "additional_cohort_registry", ()))
    ):
        resolved = raw.resolve()
        if resolved not in paths:
            paths.append(resolved)
    return {
        "cohort_registry": str(paths[0]) if paths else None,
        "cohort_registry_sha256": sha256_file(paths[0]) if paths else None,
        "cohort_registries": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
    }


def command_capture(args: argparse.Namespace) -> None:
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    started = time.perf_counter()
    adapter_audit = install_v6_kernel_adapters()
    from realistic_niah_v5.capture import capture_trace_shards

    config = _config(args.config)
    rows = _analysis_rows(args, config)
    model, tokenizer, adapter, model_spec = _load_model(args)
    index = capture_trace_shards(
        model,
        adapter,
        tokenizer,
        rows,
        config=config,
        output_dir=args.output,
        layers=args.layers,
        site_kinds=args.site_kinds,
        capture_span_pooling=not args.skip_span_pooling,
        overwrite=args.overwrite,
        site_ids=args.site_ids,
    )
    adapter_audit.update(
        {
            "run_status": "COMPLETE",
            "prompt_mode": config.prompt_mode,
            "model_label": args.model,
            "model_id": model_spec.model_id,
            "model_revision": model_spec.revision,
            "v6_config": str(args.config.resolve()),
            "v6_config_sha256": sha256_file(args.config),
            "generations": str(args.generations.resolve()),
            "generations_sha256": sha256_file(args.generations),
            "formal_cohort": not args.include_nonstrict,
            **_cohort_registry_manifest(args),
            "runtime": _runtime_base(started),
        }
    )
    _atomic_json(args.output / "v6_adapter_manifest.json", adapter_audit)
    print(f"[v6 capture] index={index} rows={len(rows)}")


def command_attention(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    install_v6_kernel_adapters()
    from realistic_niah_v5.capture import capture_trace_attention_metrics

    config = _config(args.config)
    rows = _analysis_rows(args, config)
    model, tokenizer, adapter, model_spec = _load_model(args)
    frames = []
    started = time.perf_counter()
    for row_index, row in enumerate(rows, start=1):
        frames.append(
            capture_trace_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                config=config,
                mechanisms=tuple(args.mechanisms),
            )
        )
        print(f"[v6 attention] {row_index}/{len(rows)}", flush=True)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    _atomic_json(
        args.output.with_suffix(".manifest.json"),
        {
            "schema_version": "realistic_niah_v6_attention_run_v1",
            "rows": len(output),
            "requests": len(rows),
            "prompt_mode": config.prompt_mode,
            "model_label": args.model,
            "model_id": model_spec.model_id,
            "model_revision": model_spec.revision,
            "v6_config_sha256": sha256_file(args.config),
            "generations_sha256": sha256_file(args.generations),
            "seed_role": args.seed_role,
            "formal_cohort": not args.include_nonstrict,
            **_cohort_registry_manifest(args),
            "runtime": _runtime_base(started),
        },
    )


def command_attention_answer_query(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    install_v6_kernel_adapters()
    from realistic_niah_v5.capture import capture_answer_query_attention_metrics

    config = _config(args.config)
    rows = _analysis_rows(args, config)
    model, tokenizer, adapter, model_spec = _load_model(args)
    frames = []
    started = time.perf_counter()
    for row_index, row in enumerate(rows, start=1):
        frames.append(
            capture_answer_query_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                site_id=args.site_id,
            )
        )
        print(f"[v6 attention-answer-query] {row_index}/{len(rows)}", flush=True)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    _atomic_json(
        args.output.with_suffix(".manifest.json"),
        {
            "schema_version": "realistic_niah_v6_answer_query_attention_run_v1",
            "rows": len(output),
            "requests": len(rows),
            "site_id": args.site_id,
            "prompt_mode": config.prompt_mode,
            "model_label": args.model,
            "model_id": model_spec.model_id,
            "model_revision": model_spec.revision,
            "v6_config_sha256": sha256_file(args.config),
            "generations_sha256": sha256_file(args.generations),
            "seed_role": args.seed_role,
            "formal_cohort": not args.include_nonstrict,
            **_cohort_registry_manifest(args),
            "runtime": _runtime_base(started),
        },
    )


def command_representation(args: argparse.Namespace) -> None:
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    started = time.perf_counter()
    adapter_audit = install_v6_kernel_adapters()
    from realistic_niah_v5.representation import analyze_representation

    config = _config(args.config)
    paths = analyze_representation(
        args.capture_index,
        args.output,
        config=config,
        cohorts=tuple(args.cohorts),
    )
    adapter_audit.update(
        {
            "run_status": "COMPLETE",
            "prompt_mode": config.prompt_mode,
            "v6_config": str(args.config.resolve()),
            "v6_config_sha256": sha256_file(args.config),
            "capture_index": str(args.capture_index.resolve()),
            "capture_index_sha256": sha256_file(args.capture_index),
            "outputs": {key: str(path) for key, path in paths.items()},
            "runtime": _runtime_base(started),
        }
    )
    _atomic_json(args.output / "v6_adapter_manifest.json", adapter_audit)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_answer_query_patch(args: argparse.Namespace) -> None:
    """Run the Native-thinking full answer-query state patch on a V6 cohort."""

    if args.seed_role != "confirmation":
        raise ValueError("The V6 answer-query extension is confirmation-only")
    if args.limit is not None:
        raise ValueError("Formal answer-query patching does not permit --limit")
    if args.cohort_registry is None or args.additional_cohort_registry:
        raise ValueError(
            "Answer-query patching requires exactly one frozen coherent "
            "confirmation cohort registry"
        )
    if tuple(args.conditions) != ("self_patch", "full_donor_patch"):
        raise ValueError("Answer-query patch conditions are frozen to self/full")
    if args.basis is not None:
        raise ValueError("The frozen full-state answer-query extension has no basis")

    started = time.perf_counter()
    config = _config(args.config)
    _validate_confirmation_freeze(
        args.confirmation_freeze,
        config=config,
        model_label=args.model,
    )
    contract = load_answer_trace_extension_contract(args.extension_contract)
    frozen = answer_trace_model_contract(
        contract, prompt_mode=config.prompt_mode, model_label=args.model
    )
    observed_layers = sorted({int(value) for value in args.layers})
    if observed_layers != list(frozen["answer_layers"]):
        raise ValueError(
            "Answer-query patch layers differ from the frozen extension contract"
        )
    if args.receiver_site_id != frozen["answer_site_id"] or (
        args.donor_site_id != frozen["answer_site_id"]
    ):
        raise ValueError("Answer-query patch sites differ from the frozen contract")

    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    adapter_audit = install_v6_kernel_adapters()
    rows = _analysis_rows(args, config)
    row_by_id = {str(row["request_id"]): row for row in rows}
    pairs = read_jsonl(args.pairs)
    pair_audit = validate_answer_query_pair_registry(
        pairs,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        expected_layers=frozen["answer_layers"],
        expected_slots=config.confirmation_seeds,
    )
    config_hash = sha256_file(args.config)
    contract_hash = sha256_file(args.extension_contract)
    freeze_hash = sha256_file(args.confirmation_freeze)
    cohort_hash = sha256_file(args.cohort_registry)
    generation_hash = sha256_file(args.generations)
    for pair in pairs:
        if pair["receiver_request_id"] not in row_by_id or (
            pair["donor_request_id"] not in row_by_id
        ):
            raise ValueError("Frozen answer-query pair is absent from the V6 cohort")
        expected_hashes = {
            "v6_config_sha256": config_hash,
            "extension_contract_sha256": contract_hash,
            "confirmation_freeze_sha256": freeze_hash,
            "cohort_registry_sha256": cohort_hash,
            "generations_sha256": generation_hash,
        }
        for key, expected in expected_hashes.items():
            if pair.get(key) != expected:
                raise ValueError(f"Frozen answer-query pair {key} changed")

    # The process-local adapter supplies the resolved V6 rows to the unchanged
    # Native-thinking numerical orchestrator.  No V5 source or artifact changes.
    import run_realistic_niah_v5 as legacy
    from realistic_niah_v6.spec import V6Config as ActiveV6Config

    legacy.V5Config = ActiveV6Config
    legacy.registered_records = lambda *_args, **_kwargs: list(rows)
    legacy.command_causal_patch(args)

    trial_rows = read_jsonl(args.output)
    pair_by_id = {str(pair["pair_id"]): pair for pair in pairs}
    expected_trials = len(pairs) * len(observed_layers) * len(args.conditions)
    if len(trial_rows) != expected_trials:
        raise ValueError(
            f"Answer-query patch grid incomplete: {len(trial_rows)} != {expected_trials}"
        )
    for trial in trial_rows:
        pair = pair_by_id[str(trial["pair_id"])]
        true_source = int(pair["v6_source_seed"])
        observed_seed = int(trial.get("seed", true_source))
        if observed_seed != true_source:
            raise ValueError("Answer-query trial aliases the true source seed")
        trial.update(
            {
                "schema_version": "realistic_niah_v6_answer_query_patch_trial_v1",
                "prompt_mode": config.prompt_mode,
                "seed": true_source,
                "v6_source_seed": true_source,
                "v6_analysis_slot_seed": int(pair["v6_analysis_slot_seed"]),
                "seed_aliasing": False,
                "extension_contract_sha256": contract_hash,
            }
        )
    write_jsonl(args.output, trial_rows)
    manifest = {
        "schema_version": "realistic_niah_v6_answer_query_patch_run_v1",
        "status": "PASS_COMPLETE",
        "prompt_mode": config.prompt_mode,
        "model_label": args.model,
        "protocol_relation": contract["protocol_relation"],
        "conditions": list(args.conditions),
        "layers": observed_layers,
        "site_id": frozen["answer_site_id"],
        "registered_pairs": len(pairs),
        "completed_trials": len(trial_rows),
        "pair_registry_audit": pair_audit,
        "seed_aliasing": False,
        "patch_outcomes_used_for_pair_selection": False,
        "v6_config": str(args.config.resolve()),
        "v6_config_sha256": config_hash,
        "extension_contract": str(args.extension_contract.resolve()),
        "extension_contract_sha256": contract_hash,
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_sha256": freeze_hash,
        "cohort_registry": str(args.cohort_registry.resolve()),
        "cohort_registry_sha256": cohort_hash,
        "generations": str(args.generations.resolve()),
        "generations_sha256": generation_hash,
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "trials": str(args.output.resolve()),
        "trials_sha256": sha256_file(args.output),
        "kernel_adapter": adapter_audit,
        "runtime": _runtime_base(started),
    }
    _atomic_json(args.output.with_suffix(args.output.suffix + ".v6_adapter.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


def command_validate_confirmation_freeze(args: argparse.Namespace) -> None:
    """Recheck the immutable discovery handoff without opening any outcomes."""

    config = _config(args.config)
    value = validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        verify_artifacts=True,
    )
    audit = {
        "schema_version": "realistic_niah_v6_confirmation_freeze_validation_v1",
        "status": "PASS",
        "prompt_mode": config.prompt_mode,
        "model_label": args.model,
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_file_sha256": sha256_file(args.confirmation_freeze),
        "content_freeze_sha256": value["freeze_sha256"],
        "confirmation_outcomes_read": False,
    }
    if args.output is not None:
        _atomic_json(args.output, audit)
    print(json.dumps(audit, sort_keys=True))


def command_suite_audit(args: argparse.Namespace) -> None:
    from realistic_niah_v6.generation import build_v6_user_text

    started = time.perf_counter()
    config = _config(args.config)
    document = suite_document()
    missing_entrypoints = []
    direct_v5_entrypoints = []
    for experiment in EXPERIMENTS:
        text = experiment.kernel_entrypoint
        relative_paths = re.findall(
            r"(?:scripts|src)/[A-Za-z0-9_./-]+\.py", text
        )
        if not relative_paths:
            missing_entrypoints.append(f"unparsed:{text}")
            continue
        for relative in relative_paths:
            if not (ROOT / relative).exists():
                missing_entrypoints.append(relative)
            if "realistic_niah_v5" in relative:
                direct_v5_entrypoints.append(relative)
    probe = "V6_PROMPT_PROBE"
    prompt = build_v6_user_text(probe, prompt_mode=config.prompt_mode)
    v5_diff = subprocess.run(
        ["git", "diff", "--", "src/realistic_niah_v5"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    audit = {
        "schema_version": "realistic_niah_v6_suite_audit_v1",
        "status": (
            "PASS"
            if not missing_entrypoints and not direct_v5_entrypoints
            else "FAIL"
        ),
        "prompt_mode": config.prompt_mode,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "prompt_probe_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "missing_entrypoints": sorted(set(missing_entrypoints)),
        "direct_v5_entrypoints": sorted(set(direct_v5_entrypoints)),
        "v5_source_worktree_diff_empty": not bool(v5_diff.stdout),
        "v5_source_diff": v5_diff.stdout.splitlines(),
        "suite": document,
        "runtime": _runtime_base(started),
    }
    _atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


def command_freeze_confirmation(args: argparse.Namespace) -> None:
    config = _config(args.config)
    if (args.mechanism_config is None) != (args.frozen_mechanism_output is None):
        raise ValueError(
            "--mechanism-config and --frozen-mechanism-output must be supplied together"
        )
    ledger = json.loads(args.discovery_ledger.read_text(encoding="utf-8"))
    value = freeze_confirmation(
        prompt_mode=config.prompt_mode,
        model_label=args.model,
        discovery_ledger=ledger,
        artifact_paths=args.artifacts,
    )
    _atomic_json(args.output, value)
    if args.mechanism_config is not None:
        mechanism = json.loads(args.mechanism_config.read_text(encoding="utf-8"))
        if mechanism.get("schema_version") != "realistic_niah_v6_count_stream_v1":
            raise ValueError("Mechanism config is not a V6 count-stream config")
        if config.prompt_mode not in str(mechanism.get("experiment_id", "")):
            raise ValueError("Mechanism config prompt mode disagrees with V6 config")
        if mechanism.get("status") != "development_only":
            raise ValueError("Only a development_only mechanism config can be frozen")
        mechanism["status"] = "frozen_confirmation"
        _atomic_json(args.frozen_mechanism_output, mechanism)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def command_print_suite(args: argparse.Namespace) -> None:
    value = suite_document()
    if args.output is not None:
        _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def command_init_discovery_ledger(args: argparse.Namespace) -> None:
    config = _config(args.config)
    value = discovery_ledger_template(
        prompt_mode=config.prompt_mode, model_label=args.model
    )
    _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_INDEX_CONFIG)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=MODEL_LABELS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_analysis_rows(parser: argparse.ArgumentParser) -> None:
    _add_config(parser)
    _add_model(parser)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument(
        "--seed-role", choices=["discovery", "confirmation", "all"], default="discovery"
    )
    parser.add_argument("--include-nonstrict", action="store_true")
    parser.add_argument(
        "--cohort-registry",
        type=Path,
        help=(
            "Outcome-blind selected_cells.jsonl that fills strict parser "
            "failures from the frozen-amendment reserve pool."
        ),
    )
    parser.add_argument(
        "--additional-cohort-registry",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional disjoint-role selected_cells.jsonl. Use this with "
            "--seed-role all to materialize resolved discovery+confirmation."
        ),
    )
    parser.add_argument("--limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V6 index/bullet full-suite replication pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_config(preflight)
    preflight.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)

    generate = subparsers.add_parser("generate")
    _add_config(generate)
    _add_model(generate)
    generate.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--seed-role", choices=["discovery", "confirmation"], default="discovery"
    )
    generate.add_argument("--confirmation-freeze", type=Path)
    generate.add_argument("--counts", type=int, nargs="+")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(func=command_generate)

    parse = subparsers.add_parser("parse")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--compact", action="store_true")
    parse.set_defaults(func=command_parse)

    audit = subparsers.add_parser("audit-generations")
    _add_config(audit)
    audit.add_argument("--model", required=True, choices=MODEL_LABELS)
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--seed-role", choices=["discovery", "confirmation", "all"], default="all"
    )
    audit.add_argument("--allow-partial", action="store_true")
    audit.set_defaults(func=command_audit_generations)

    capture = subparsers.add_parser("capture")
    _add_analysis_rows(capture)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--layers", type=int, nargs="+")
    capture.add_argument("--site-kinds", nargs="+")
    capture.add_argument("--site-ids", nargs="+")
    capture.add_argument("--skip-span-pooling", action="store_true")
    capture.add_argument("--overwrite", action="store_true")
    capture.set_defaults(func=command_capture)

    attention = subparsers.add_parser("attention")
    _add_analysis_rows(attention)
    attention.add_argument("--output", type=Path, required=True)
    attention.add_argument(
        "--mechanisms",
        nargs="+",
        choices=["targeted_retrieval", "progress_transition"],
        default=["targeted_retrieval", "progress_transition"],
    )
    attention.set_defaults(func=command_attention)

    answer = subparsers.add_parser("attention-answer-query")
    _add_analysis_rows(answer)
    answer.add_argument("--output", type=Path, required=True)
    answer.add_argument("--site-id", default="answer_query_v3")
    answer.set_defaults(func=command_attention_answer_query)

    representation = subparsers.add_parser("representation")
    _add_config(representation)
    representation.add_argument("--capture-index", type=Path, required=True)
    representation.add_argument("--output", type=Path, required=True)
    representation.add_argument(
        "--cohorts",
        nargs="+",
        default=["parser_hit", "one_to_one", "one_to_one_correct"],
    )
    representation.set_defaults(func=command_representation)

    answer_patch = subparsers.add_parser("answer-query-patch")
    _add_analysis_rows(answer_patch)
    answer_patch.add_argument("--extension-contract", type=Path, required=True)
    answer_patch.add_argument("--confirmation-freeze", type=Path, required=True)
    answer_patch.add_argument("--pairs", type=Path, required=True)
    answer_patch.add_argument("--basis", type=Path)
    answer_patch.add_argument("--output", type=Path, required=True)
    answer_patch.add_argument("--layer", type=int)
    answer_patch.add_argument("--layers", type=int, nargs="+", required=True)
    answer_patch.add_argument(
        "--conditions",
        nargs="+",
        choices=["self_patch", "full_donor_patch"],
        default=["self_patch", "full_donor_patch"],
    )
    answer_patch.add_argument("--receiver-site-id", default="answer_query_v3")
    answer_patch.add_argument("--donor-site-id", default="answer_query_v3")
    answer_patch.add_argument("--max-new-tokens", type=int, default=16)
    answer_patch.add_argument("--restartable", action="store_true")
    answer_patch.add_argument("--overwrite", action="store_true")
    answer_patch.set_defaults(func=command_answer_query_patch)

    validate_freeze = subparsers.add_parser("validate-confirmation-freeze")
    _add_config(validate_freeze)
    validate_freeze.add_argument("--model", required=True, choices=MODEL_LABELS)
    validate_freeze.add_argument("--confirmation-freeze", type=Path, required=True)
    validate_freeze.add_argument("--output", type=Path)
    validate_freeze.set_defaults(func=command_validate_confirmation_freeze)

    suite_audit = subparsers.add_parser("suite-audit")
    _add_config(suite_audit)
    suite_audit.add_argument("--output", type=Path, required=True)
    suite_audit.set_defaults(func=command_suite_audit)

    freeze = subparsers.add_parser("freeze-confirmation")
    _add_config(freeze)
    freeze.add_argument("--model", required=True, choices=MODEL_LABELS)
    freeze.add_argument("--discovery-ledger", type=Path, required=True)
    freeze.add_argument("--artifacts", type=Path, nargs="+", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--mechanism-config", type=Path)
    freeze.add_argument("--frozen-mechanism-output", type=Path)
    freeze.set_defaults(func=command_freeze_confirmation)

    show = subparsers.add_parser("print-suite")
    show.add_argument("--output", type=Path)
    show.set_defaults(func=command_print_suite)

    ledger = subparsers.add_parser("init-discovery-ledger")
    _add_config(ledger)
    ledger.add_argument("--model", required=True, choices=MODEL_LABELS)
    ledger.add_argument("--output", type=Path, required=True)
    ledger.set_defaults(func=command_init_discovery_ledger)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
