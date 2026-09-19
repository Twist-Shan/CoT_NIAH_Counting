#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.pipeline import (
    parse_records,
    read_jsonl,
    registered_records,
    write_jsonl,
)
from realistic_niah_v5.spec import V5Config


DEFAULT_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"


def _config(args: argparse.Namespace) -> V5Config:
    return V5Config.load(args.config)


def _model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
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


def command_parse(args: argparse.Namespace) -> None:
    rows = parse_records(read_jsonl(args.input), include_input=not args.compact)
    count = write_jsonl(args.output, rows)
    print(f"[v5 parse] wrote {count} rows to {args.output}")


def _row_gold_count(row: dict[str, Any]) -> int:
    return int(
        row.get(
            "gold_count",
            len(row.get("gold_records", row.get("gold_pairs", []))),
        )
    )


def _evaluation_counts(
    config: V5Config, requested: list[int] | tuple[int, ...] | None
) -> tuple[int, ...]:
    if requested is None:
        return tuple(config.counts)
    values = tuple(int(value) for value in requested)
    if not values:
        raise ValueError("At least one count is required")
    if len(set(values)) != len(values):
        raise ValueError(f"Evaluation counts must be unique: {values}")
    unsupported = sorted(set(values) - set(config.counts))
    if unsupported:
        raise ValueError(
            "Evaluation counts fall outside the registered config: "
            f"{unsupported}"
        )
    return tuple(sorted(values))


def _filter_rows_by_count(
    rows: list[dict[str, Any]],
    *,
    config: V5Config,
    requested: list[int] | tuple[int, ...] | None,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    counts = _evaluation_counts(config, requested)
    allowed = set(counts)
    return [row for row in rows if _row_gold_count(row) in allowed], counts


def command_generate(args: argparse.Namespace) -> None:
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.generation import generate_native_trace, render_native_prompt

    config = _config(args)
    spec = resolve_model_spec(args.model)
    model, tokenizer, _adapter = _model(args)
    stimuli, evaluation_counts = _filter_rows_by_count(
        registered_records(
            read_jsonl(args.stimuli), config, model_label=args.model
        ),
        config=config,
        requested=args.counts,
    )
    if not stimuli:
        raise ValueError(
            f"No registered stimuli remain for counts {evaluation_counts}"
        )
    print(
        f"[v5 generate] evaluation_counts={list(evaluation_counts)} "
        f"stimuli={len(stimuli)}",
        flush=True,
    )
    output_rows = []
    for index, stimulus in enumerate(stimuli, start=1):
        prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=spec)
        output_rows.append(
            generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=config.decoding,
                sampling_seed=int(stimulus["seed"]),
            )
        )
        print(
            f"[v5 generate] {index}/{len(stimuli)} {prompt.stimulus_id} "
            f"parser={output_rows[-1]['trace_parse']['parser']['trace_category']}",
            flush=True,
        )
    count = write_jsonl(args.output, output_rows)
    print(f"[v5 generate] wrote {count} rows to {args.output}")


def command_capture(args: argparse.Namespace) -> None:
    from realistic_niah_v5.capture import capture_trace_shards

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    rows = (
        [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
        if args.allow_unregistered
        else registered_records(raw_rows, config, model_label=args.model)
    )
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
    print(f"[v5 capture] index: {index}")


def command_attention(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.capture import capture_trace_attention_metrics

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    frames = []
    for index, row in enumerate(rows, start=1):
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
        print(f"[v5 attention] {index}/{len(rows)}", flush=True)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"[v5 attention] wrote {len(output)} rows to {args.output}")


def command_attention_pre_city(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import (
        capture_pre_city_attention_metrics,
        write_pre_city_audit,
    )

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _parser_cohort_rows(rows, "one_to_one")
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    exclusion_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.csv.gz"
        exclusion_path = shard_root / f"{index:04d}__{safe_id}.exclusions.jsonl"
        if args.overwrite or not shard_path.exists():
            frame, row_exclusions = capture_pre_city_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                depths=tuple(args.depths),
                include_anchor=not args.no_anchor,
            )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(shard_path)
            write_jsonl(exclusion_path, row_exclusions)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        exclusion_paths.append(exclusion_path)
        print(
            f"[v5 attention-pre-city] {index}/{len(rows)} {action} {request_id}",
            flush=True,
        )
    frames = [pd.read_csv(path) for path in shard_paths]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    exclusions = []
    for path in exclusion_paths:
        if path.exists():
            exclusions.extend(read_jsonl(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    exclusion_path = args.output.with_suffix(".exclusions.jsonl")
    write_jsonl(exclusion_path, exclusions)
    audit_path = args.output.with_suffix(".audit.json")
    write_pre_city_audit(output, exclusions, audit_path)
    print(
        f"[v5 attention-pre-city] wrote rows={len(output)} "
        f"exclusions={len(exclusions)} output={args.output}",
        flush=True,
    )


def command_attention_answer_query(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.capture import (
        capture_answer_query_attention_metrics,
    )

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.csv.gz"
        if args.overwrite or not shard_path.exists():
            frame = capture_answer_query_attention_metrics(
                model,
                adapter,
                tokenizer,
                row,
                site_id=args.site_id,
            )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(shard_path)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 attention-answer-query] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    frames = [pd.read_csv(path) for path in shard_paths]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit = {
        "schema_version": "realistic_niah_v5_answer_query_extension_v1",
        "site_id": args.site_id,
        "query_definition": (
            "last literal Total: prefix before the first numeric answer token"
        ),
        "cohort": args.cohort,
        "split": args.split,
        "requests": len(rows),
        "rows": len(output),
        "restartable_shards": True,
        "confirmation_used_for_selection": False,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[v5 attention-answer-query] wrote requests={len(rows)} "
        f"rows={len(output)} output={args.output}",
        flush=True,
    )


def command_representation(args: argparse.Namespace) -> None:
    from realistic_niah_v5.representation import analyze_representation

    paths = analyze_representation(
        args.capture_index,
        args.output,
        config=_config(args),
        cohorts=args.cohorts,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import build_causal_plan

    paths = build_causal_plan(
        args.source_writes,
        args.output,
        config=_config(args),
        bank_size=args.bank_size,
        anchor_role=args.anchor_role,
        minimum_layer=args.minimum_layer,
        maximum_layer=args.maximum_layer,
        selection_metric=args.selection_metric,
        target_grammar_class=args.target_grammar_class,
        target_retrieval_surface_variant=(
            args.target_retrieval_surface_variant
        ),
        selection_eligibility_scope=args.selection_eligibility_scope,
        selection_aggregation=args.selection_aggregation,
        allow_incomplete_development_smoke=args.development_smoke,
        include_random_controls=not args.selected_only_smoke,
        random_control_matching=args.random_control_matching,
        confirmation_plan=args.confirmation_plan,
        full_panel_plan=args.full_panel_plan,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_answer_query_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import build_answer_query_causal_plan

    paths = build_answer_query_causal_plan(
        args.attention,
        args.output,
        config=_config(args),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_pre_city_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.pre_city import build_pre_city_causal_plan

    paths = build_pre_city_causal_plan(
        args.attention,
        args.output,
        config=_config(args),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def command_response_reference_attention(args: argparse.Namespace) -> None:
    """Attach model-specific response types to exact reused pre-city d1 maps."""

    import pandas as pd

    from realistic_niah_v5.response_reference import (
        RESPONSE_REFERENCE_SCHEMA_VERSION,
        attach_response_reference_types,
    )

    generations = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    attention = pd.read_csv(args.pre_city_attention)
    attention = attention.loc[
        attention["model_label"].astype(str).eq(args.model)
    ].copy()
    enriched, exclusions = attach_response_reference_types(attention, generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    write_jsonl(args.output.with_suffix(".exclusions.jsonl"), exclusions)
    counts = (
        enriched[["request_id", "occurrence", "split", "response_type"]]
        .drop_duplicates()
        .groupby(["split", "response_type"])
        .size()
        .to_dict()
    )
    audit = {
        "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
        "model_label": args.model,
        "source_attention": str(args.pre_city_attention.resolve()),
        "source_query_variants": [
            "pre_city_d1",
            "pre_city_d2",
            "pre_city_anchor",
        ],
        "query_position_reuse_audit": "PASS_IDENTICAL_SOURCE_PRE_CITY",
        "requests_in_generation_source": len(generations),
        "attention_rows": len(enriched),
        "registered_query_counts": {
            f"{split}:{response_type}": int(value)
            for (split, response_type), value in sorted(counts.items())
        },
        "exclusions": len(exclusions),
        "confirmation_used_for_selection": False,
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_attention_targeted_reference(args: argparse.Namespace) -> None:
    """Capture dedicated k-to-k attention at parser-registered citations."""

    import pandas as pd

    from realistic_niah_v5.response_reference import (
        RESPONSE_REFERENCE_SCHEMA_VERSION,
        capture_targeted_retrieval_attention_metrics,
    )

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _parser_cohort_rows(rows, "one_to_one")
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    exclusion_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.csv.gz"
        exclusion_path = shard_root / f"{index:04d}__{safe_id}.exclusions.jsonl"
        if args.overwrite or not shard_path.exists():
            frame, row_exclusions = capture_targeted_retrieval_attention_metrics(
                model, adapter, tokenizer, row
            )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(shard_path)
            write_jsonl(exclusion_path, row_exclusions)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        exclusion_paths.append(exclusion_path)
        print(
            f"[v5 attention-targeted-reference] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    frames = [pd.read_csv(path) for path in shard_paths]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    exclusions = []
    for path in exclusion_paths:
        if path.exists():
            exclusions.extend(read_jsonl(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    write_jsonl(args.output.with_suffix(".exclusions.jsonl"), exclusions)
    registry = (
        output[
            [
                "request_id",
                "occurrence",
                "target_city",
                "response_type",
                "citation_start_kind",
                "query_output_token_count",
                "target_after_token",
            ]
        ].drop_duplicates()
        if not output.empty
        else pd.DataFrame()
    )
    audit = {
        "schema_version": RESPONSE_REFERENCE_SCHEMA_VERSION,
        "experiment_id": "trace_pre_reference_k_to_k_targeted_retrieval_v1",
        "model_label": args.model,
        "query_variant": "pre_reference_d1",
        "query_definition": (
            "one real baseline token immediately before each deterministic "
            "parser-registered citation start"
        ),
        "k_to_k_definition": (
            "response occurrence k targets the unique exact-city prompt needle "
            "span registered for that occurrence"
        ),
        "response_types": sorted(registry["response_type"].unique().tolist())
        if not registry.empty
        else [],
        "citation_start_counts": (
            registry["citation_start_kind"].value_counts().sort_index().to_dict()
            if not registry.empty
            else {}
        ),
        "requests": len(rows),
        "registered_occurrences": len(registry),
        "attention_rows": len(output),
        "exclusions": len(exclusions),
        "restartable_shards": True,
        "confirmation_used_for_selection": False,
        "reported_attention_metrics": [
            "target_needle_raw_mass",
            "target_needle_relative_mass",
        ],
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_response_reference_causal_plan(args: argparse.Namespace) -> None:
    from realistic_niah_v5.response_reference import (
        build_response_reference_causal_plan,
    )

    paths = build_response_reference_causal_plan(
        args.attention, args.output, config=_config(args)
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


def _parse_heads(value: str) -> list[tuple[int, int]]:
    parsed = json.loads(value)
    return [(int(layer), int(head)) for layer, head in parsed]


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    return [row for row in rows if str(row.get("split")) == split]


def _parser_cohort_rows(
    rows: list[dict[str, Any]], cohort: str
) -> list[dict[str, Any]]:
    from realistic_niah_v5.parsing import parse_trace_record

    selected = []
    for row in rows:
        parsed = parse_trace_record(row)
        parser = parsed["parser"]
        keep = bool(parser.get("detected"))
        if cohort in {"one_to_one", "one_to_one_correct"}:
            keep &= bool(parser.get("trace_one_to_one"))
        if cohort == "one_to_one_correct":
            keep &= bool(parsed.get("exact_count"))
        if keep:
            selected.append(row)
    return selected


def _row_site_ids(
    row: dict[str, Any], *, site_kind: str | None, site_id: str | None
) -> list[str]:
    if site_id is not None:
        return [site_id]
    if site_kind is None:
        raise ValueError("Either site_kind or site_id is required")
    from realistic_niah_v5.parsing import parse_trace_record

    parsed = parse_trace_record(row)
    return [
        str(site["site_id"])
        for site in parsed["char_sites"]
        if str(site["site_kind"]) == site_kind
    ]


def _safe_stem(*values: Any) -> str:
    readable = "__".join(str(value) for value in values)
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:12]
    return f"trial_{digest}"


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _claim_restartable_task(
    shard_path: Path,
    *,
    worker_id: str,
    stale_seconds: float,
) -> Path | None:
    if shard_path.exists():
        return None
    claim = shard_path.with_suffix(shard_path.suffix + ".claim")
    if claim.exists():
        try:
            age = time.time() - claim.stat().st_mtime
            owner_dead = False
            try:
                owner = json.loads(claim.read_text(encoding="utf-8"))
                if str(owner.get("hostname")) == socket.gethostname():
                    owner_pid = int(owner["pid"])
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        owner_dead = True
                    except PermissionError:
                        owner_dead = False
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                owner_dead = False
            if (
                (owner_dead or age > float(stale_seconds))
                and not shard_path.exists()
            ):
                claim.unlink()
        except FileNotFoundError:
            pass
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "worker_id": worker_id,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "claimed_unix": time.time(),
                    "target": str(shard_path),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError:
        return None
    if shard_path.exists():
        claim.unlink(missing_ok=True)
        return None
    return claim
def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Commit a complete JSONL shard with one atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        handle.flush()
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_completed_behavior_shard(
    path: Path, *, expected_trial_id: str
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Validate a resumable behavioral shard before treating it as complete."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            saved = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if len(saved) != 1:
        return None, f"expected one result row, found {len(saved)}"
    result = saved[0]
    if str(result.get("trial_id")) != expected_trial_id:
        return None, (
            "trial_id mismatch: "
            f"expected {expected_trial_id!r}, found {result.get('trial_id')!r}"
        )
    if result.get("trial_complete") is not True:
        return None, "trial_complete is not true"
    return saved, None


def _archive_invalid_behavior_shard(path: Path) -> Path:
    """Preserve an invalid shard by content hash before exact recomputation."""

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    archive = path.parent.parent / "corrupt_shards"
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / f"{path.stem}.{digest}.jsonl"
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def _frame_records(frame: Any) -> list[dict[str, Any]]:
    """Use pandas' strict null conversion before JSON shard serialization."""

    return json.loads(frame.to_json(orient="records"))


def _anchor_in_registered_scope(
    specification: dict[str, Any],
    *,
    include_secondary: bool,
    include_block_pre: bool,
) -> bool:
    if not include_block_pre and not bool(specification.get("event_specific")):
        return False
    if include_secondary:
        return bool(specification.get("local_anchor_eligible"))
    return bool(specification.get("primary_anchor_eligible"))


def _diverse_anchor_subset(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]], limit: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministically maximize role/grammar/cohort/seed smoke coverage."""

    if limit is None or int(limit) >= len(tasks):
        return tasks
    if int(limit) < 1:
        raise ValueError("--limit must be positive")
    remaining = sorted(
        tasks,
        key=lambda value: (
            str(value[0].get("request_id", value[0].get("stimulus_id"))),
            int(value[1]["from_occurrence"]),
            int(value[1]["query_output_token_index"]),
        ),
    )
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_roles: set[str] = set()
    seen_grammars: set[str] = set()
    seen_cohorts: set[str] = set()
    seen_seeds: set[int] = set()
    role_counts: dict[str, int] = {}
    grammar_counts: dict[str, int] = {}
    transition_counts: dict[tuple[int, int], int] = {}
    seed_counts: dict[int, int] = {}

    def score(index: int) -> tuple[int, int, int, int, int, int, int, int, int]:
        row, specification = remaining[index]
        roles = {
            str(value)
            for value in specification.get(
                "anchor_roles", [specification.get("anchor_role")]
            )
            if value is not None
        }
        grammar = str(specification.get("grammar_pair"))
        cohort = str(specification.get("causal_cohort"))
        transition = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
        )
        seed = int(row["seed"])
        least_role_count = min((role_counts.get(role, 0) for role in roles), default=0)
        return (
            len(roles - seen_roles),
            int(grammar not in seen_grammars),
            int(cohort not in seen_cohorts),
            -least_role_count,
            -grammar_counts.get(grammar, 0),
            -transition_counts.get(transition, 0),
            int(seed not in seen_seeds),
            -seed_counts.get(seed, 0),
            -index,
        )

    while remaining and len(chosen) < int(limit):
        best_index = max(range(len(remaining)), key=score)
        row, specification = remaining.pop(best_index)
        chosen.append((row, specification))
        roles = {
            str(value)
            for value in specification.get(
                "anchor_roles", [specification.get("anchor_role")]
            )
            if value is not None
        }
        seen_roles.update(roles)
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        grammar = str(specification.get("grammar_pair"))
        seen_grammars.add(grammar)
        grammar_counts[grammar] = grammar_counts.get(grammar, 0) + 1
        seen_cohorts.add(str(specification.get("causal_cohort")))
        transition = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
        )
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        seed = int(row["seed"])
        seen_seeds.add(seed)
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
    return chosen


def _seed_first_anchor_subset(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]], limit: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose at most one anchor per seed before taking any second anchor."""

    if limit is None:
        limit = len(tasks)
    if int(limit) < 1:
        raise ValueError("--limit must be positive")
    remaining = sorted(
        tasks,
        key=lambda value: (
            str(value[0].get("request_id", value[0].get("stimulus_id"))),
            int(value[1]["from_occurrence"]),
            int(value[1]["query_output_token_index"]),
        ),
    )
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_seeds: set[int] = set()
    seen_grammars: set[str] = set()
    grammar_counts: dict[str, int] = {}
    transition_counts: dict[tuple[int, int], int] = {}

    def score(index: int) -> tuple[int, int, int, int, int]:
        row, specification = remaining[index]
        seed = int(row["seed"])
        grammar = str(specification.get("grammar_pair"))
        transition = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
        )
        return (
            int(seed not in seen_seeds),
            int(grammar not in seen_grammars),
            -grammar_counts.get(grammar, 0),
            -transition_counts.get(transition, 0),
            -index,
        )

    while remaining and len(chosen) < min(int(limit), len(tasks)):
        best_index = max(range(len(remaining)), key=score)
        row, specification = remaining.pop(best_index)
        chosen.append((row, specification))
        seen_seeds.add(int(row["seed"]))
        grammar = str(specification.get("grammar_pair"))
        seen_grammars.add(grammar)
        grammar_counts[grammar] = grammar_counts.get(grammar, 0) + 1
        transition = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
        )
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    return chosen


def _candidate_target_grammar(specification: dict[str, Any]) -> str:
    routed = specification.get("routed_target_grammar_class")
    if routed is not None and str(routed).strip():
        return str(routed)
    direct = specification.get("target_grammar_class")
    if direct is not None and str(direct).strip():
        return str(direct)
    return str(specification.get("grammar_pair", "")).rsplit(" -> ", 1)[-1]


def _prompt_balanced_anchor_subset(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]], limit: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose at most one transition per prompt while balancing grammars."""

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, specification in tasks:
        request_id = str(row.get("request_id", row.get("stimulus_id")))
        grouped.setdefault(request_id, []).append((row, specification))
    if limit is None:
        limit = len(grouped)
    if int(limit) < 1:
        raise ValueError("--limit must be positive")

    prompt_groups = sorted(
        grouped.items(),
        key=lambda value: (
            len(
                {
                    _candidate_target_grammar(specification)
                    for _row, specification in value[1]
                }
            ),
            int(value[1][0][0]["seed"]),
            _row_gold_count(value[1][0][0]),
            value[0],
        ),
    )
    grammar_counts: dict[str, int] = {}
    surface_counts: dict[str, int] = {}
    transition_counts: dict[tuple[int, int], int] = {}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _request_id, candidates in prompt_groups[: int(limit)]:
        ordered = sorted(
            candidates,
            key=lambda value: (
                _candidate_target_grammar(value[1]),
                str(value[1].get("target_retrieval_surface_variant", "")),
                int(value[1]["from_occurrence"]),
                int(value[1]["query_output_token_index"]),
            ),
        )

        def score(
            value: tuple[dict[str, Any], dict[str, Any]]
        ) -> tuple[int, int, int, int, int]:
            _row, specification = value
            grammar = _candidate_target_grammar(specification)
            surface = str(
                specification.get("target_retrieval_surface_variant", "")
            )
            transition = (
                int(specification["from_occurrence"]),
                int(specification["to_occurrence"]),
            )
            return (
                -grammar_counts.get(grammar, 0),
                -surface_counts.get(surface, 0),
                -transition_counts.get(transition, 0),
                -int(specification["from_occurrence"]),
                -int(specification["query_output_token_index"]),
            )

        chosen = max(ordered, key=score)
        selected.append(chosen)
        _row, specification = chosen
        grammar = _candidate_target_grammar(specification)
        surface = str(
            specification.get("target_retrieval_surface_variant", "")
        )
        transition = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
        )
        grammar_counts[grammar] = grammar_counts.get(grammar, 0) + 1
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
    return selected


def _prompt_final_transition_anchor_subset(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]], limit: int | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Choose the registered N-1 -> N retrieval transition per prompt.

    The deterministic rule is outcome-blind and leaves exactly one remaining
    trace item on every selected branch, making it suitable for propagation
    tests that terminate at the final count.
    """

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, specification in tasks:
        count = _row_gold_count(row)
        if (
            int(specification["from_occurrence"]) == count - 1
            and int(specification["to_occurrence"]) == count
        ):
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            grouped.setdefault(request_id, []).append((row, specification))
    if limit is None:
        limit = len(grouped)
    if int(limit) < 1:
        raise ValueError("--limit must be positive")
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for request_id in sorted(
        grouped,
        key=lambda value: (
            int(grouped[value][0][0]["seed"]),
            _row_gold_count(grouped[value][0][0]),
            value,
        ),
    )[: int(limit)]:
        candidates = grouped[request_id]
        if len(candidates) != 1:
            raise ValueError(
                "Final-transition routing must yield one candidate per prompt: "
                f"{request_id} has {len(candidates)}"
            )
        selected.append(candidates[0])
    return selected


def _causal_manifest(
    *,
    command: str,
    config: V5Config,
    args: argparse.Namespace,
    completed_shards: int,
) -> dict[str, Any]:
    payload = config.to_dict()
    code_paths = (
        Path(__file__).resolve(),
        SRC / "realistic_niah_v4" / "modeling.py",
        SRC / "realistic_niah_v4" / "spec.py",
        SRC / "realistic_niah_v5" / "causal.py",
        SRC / "realistic_niah_v5" / "causal_sites.py",
        SRC / "realistic_niah_v5" / "encoding.py",
        SRC / "realistic_niah_v5" / "pipeline.py",
        SRC / "realistic_niah_v5" / "spec.py",
    )
    return {
        "schema_version": "realistic_niah_v5_causal_run_manifest_v3",
        "command": command,
        "model_label": args.model,
        "generations": str(args.generations.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "code_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in code_paths
        },
        "development_seeds": list(config.causal_development_seeds),
        "include_secondary": bool(args.include_secondary),
        "include_block_pre": bool(args.include_block_pre),
        "anchor_role": getattr(args, "anchor_role", None),
        "target_grammar_class": getattr(args, "target_grammar_class", None),
        "target_retrieval_surface_variant": getattr(
            args, "target_retrieval_surface_variant", None
        ),
        "requested_behavior_target_grammar_class": getattr(
            args, "behavior_target_grammar_class", None
        ),
        "allow_cross_grammar_bank_transfer": bool(
            getattr(args, "allow_cross_grammar_bank_transfer", False)
        ),
        "allow_selection_scope_bank_transfer": bool(
            getattr(args, "allow_selection_scope_bank_transfer", False)
        ),
        "evaluation_split": getattr(args, "evaluation_split", None),
        "evaluation_counts": list(
            _evaluation_counts(config, getattr(args, "counts", None))
        ),
        "anchor_sampling": getattr(args, "anchor_sampling", None),
        "behavior_all_routed_grammars": bool(
            getattr(args, "behavior_all_routed_grammars", False)
        ),
        "selection_metric": getattr(args, "selection_metric", None),
        "head_ablation_scope": getattr(args, "head_ablation_scope", None),
        "final_transition_only": bool(
            getattr(args, "final_transition_only", False)
        ),
        "minimum_layer": getattr(args, "minimum_layer", None),
        "maximum_layer": getattr(args, "maximum_layer", None),
        "layers": (
            None
            if getattr(args, "layers", None) is None
            else sorted({int(value) for value in args.layers})
        ),
        "completed_shards": int(completed_shards),
        "resume_policy": "atomic_anchor_or_anchor_condition_shards",
    }


def _assert_resume_manifest_compatible(
    output_dir: Path, expected: dict[str, Any]
) -> None:
    path = output_dir / "manifest.json"
    if not path.exists():
        return
    observed = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "schema_version",
        "command",
        "model_label",
        "generations",
        "config_sha256",
        "include_secondary",
        "include_block_pre",
        "layers",
        "plan_sha256",
        "code_sha256",
        "anchor_role",
        "target_grammar_class",
        "target_retrieval_surface_variant",
        "requested_behavior_target_grammar_class",
        "allow_cross_grammar_bank_transfer",
        "allow_selection_scope_bank_transfer",
        "evaluation_split",
        "evaluation_counts",
        "anchor_sampling",
        "behavior_all_routed_grammars",
        "selection_metric",
        "head_ablation_scope",
        "final_transition_only",
        "anchor_routing_sha256",
        "anchor_routing_policy_id",
        "minimum_layer",
        "maximum_layer",
        "branch_policy",
        "conditions",
        "max_new_tokens",
        "decode_head_ablation_steps",
        "reference_transition_sha256",
        "anchor_registry_input_sha256",
    )
    mismatches = {
        key: {"existing": observed.get(key), "requested": expected.get(key)}
        for key in keys
        if observed.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            "Refusing to mix incompatible causal shards in one output "
            f"directory: {mismatches}"
        )


def _load_layer_bases(path: Path) -> dict[int, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        result = {
            int(key.removeprefix("basis_L")): np.asarray(archive[key], dtype=np.float32)
            for key in archive.files
            if key.startswith("basis_L")
        }
    if not result:
        raise ValueError(f"No basis_L<layer> arrays found in {path}")
    return result


def causal_patch_conditions_need_basis(conditions: Sequence[str]) -> bool:
    """Only projected/orthogonal patch arms consume a learned basis archive."""

    return bool(
        {"projected_donor_patch", "orthogonal_norm_matched"}
        & {str(value) for value in conditions}
    )


def _load_layer_directions(path: Path) -> dict[int, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        result: dict[int, Any] = {}
        for key in archive.files:
            if key.startswith("direction_L"):
                result[int(key.removeprefix("direction_L"))] = np.asarray(
                    archive[key], dtype=np.float32
                )
            elif key.startswith("basis_L"):
                basis = np.asarray(archive[key], dtype=np.float32)
                if basis.ndim != 2 or basis.shape[1] < 1:
                    raise ValueError(f"Invalid direction basis {key} in {path}")
                result[int(key.removeprefix("basis_L"))] = basis[:, 0]
    if not result:
        raise ValueError(f"No direction_L<layer> or basis_L<layer> arrays in {path}")
    return result


def command_causal_source_writes(args: argparse.Namespace) -> None:
    """Capture same-query target attention and OV writes at exact anchors."""

    from realistic_niah_v5.causal import (
        capture_source_specific_head_writes,
        mechanism_continuations,
    )
    from realistic_niah_v5.encoding import build_native_causal_encoding

    config = _config(args)
    output_dir = Path(args.output)
    expected_manifest = _causal_manifest(
        command="causal-source-writes",
        config=config,
        args=args,
        completed_shards=0,
    )
    _assert_resume_manifest_compatible(output_dir, expected_manifest)
    if not (output_dir / "manifest.json").exists():
        _atomic_json(output_dir / "manifest.json", expected_manifest)
    model, tokenizer, adapter = _model(args)
    development = set(config.causal_development_seeds)
    rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in development
    ]
    requested_counts = {
        int(value) for value in (args.counts or tuple(config.counts))
    }
    rows = [row for row in rows if int(row["gold_count"]) in requested_counts]
    if not rows:
        raise ValueError(
            f"No source-write rows remain for counts {sorted(requested_counts)}"
        )
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        specifications, excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        excluded_rows.extend(
            {
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                **value,
            }
            for value in excluded
        )
        tasks.extend(
            (row, specification)
            for specification in specifications
            if (
                args.anchor_role is None
                or str(args.anchor_role)
                in {
                    str(value)
                    for value in specification.get("anchor_roles", [])
                }
            )
            if (
                args.target_grammar_class is None
                or str(specification.get("grammar_pair", "")).rsplit(
                    " -> ", 1
                )[-1]
                == str(args.target_grammar_class)
            )
            if (
                args.target_retrieval_surface_variant is None
                or str(
                    specification.get(
                        "target_retrieval_surface_variant", ""
                    )
                )
                == str(args.target_retrieval_surface_variant)
            )
            if _anchor_in_registered_scope(
                specification,
                include_secondary=args.include_secondary,
                include_block_pre=args.include_block_pre,
            )
            if (
                not args.final_transition_only
                or (
                    int(specification["from_occurrence"])
                    == int(row["gold_count"]) - 1
                    and int(specification["to_occurrence"])
                    == int(row["gold_count"])
                )
            )
        )
    tasks = (
        _seed_first_anchor_subset(tasks, args.limit)
        if args.anchor_role is not None
        else _diverse_anchor_subset(tasks, args.limit)
    )
    shard_dir = output_dir / "shards"
    existing = list(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Causal source-write shards already exist in {shard_dir}; "
            "resume or choose a new output directory"
        )
    completed = 0
    skipped = 0
    for task_index, (row, specification) in enumerate(tasks, start=1):
        capture_id = _safe_stem(
            row.get("request_id", row.get("stimulus_id")),
            specification["from_occurrence"],
            specification["to_occurrence"],
            specification["anchor_equivalence_id"],
            "source_specific_ov",
        )
        shard_path = shard_dir / f"{capture_id}.jsonl"
        if args.resume and shard_path.exists():
            skipped += 1
            continue
        encoding = build_native_causal_encoding(
            row,
            tokenizer,
            query_output_token_index=int(
                specification["query_output_token_index"]
            ),
            sequence_output_token_end=int(
                specification["target_output_token_end"]
            ),
            selected_site=specification,
        )
        frame, _vectors = capture_source_specific_head_writes(
            model,
            adapter,
            encoding,
            target_city=str(specification["target_city"]),
            attention_audit_cities=[
                span.city for span in encoding.prompt_record_spans
            ],
            layers=args.layers,
        )
        records = _frame_records(frame)
        if not records:
            raise RuntimeError("Source-specific write capture returned no heads")
        for record in records:
            record["capture_id"] = capture_id
            record["capture_complete"] = True
        _atomic_jsonl(shard_path, records)
        completed += 1
        print(
            f"[v5 causal-source-writes] {task_index}/{len(tasks)} "
            f"{specification['anchor_equivalence_id']} rows={len(records)}",
            flush=True,
        )
        _atomic_json(
            output_dir / "manifest.json",
            _causal_manifest(
                command="causal-source-writes",
                config=config,
                args=args,
                completed_shards=len(list(shard_dir.glob("*.jsonl"))),
            ),
        )
    _atomic_jsonl(output_dir / "excluded_anchors.jsonl", excluded_rows)
    manifest = _causal_manifest(
        command="causal-source-writes",
        config=config,
        args=args,
        completed_shards=len(list(shard_dir.glob("*.jsonl"))),
    )
    manifest.update(
        {
            "eligible_anchor_tasks": len(tasks),
            "newly_completed": completed,
            "resume_skipped": skipped,
            "source_metrics": [
                "target_source_relative_attention_mass",
                "source_attention_mass",
                "target_source_attention_top1",
                "target_minus_max_wrong_source_attention_mass",
                "source_specific_ov_write_norm",
            ],
            "attention_query_policy": (
                "same_exact_query_token_as_downstream_head_ablation"
            ),
            "attention_audit_scope": "all_registered_gold_prompt_records",
            "target_policy": "fixed_next_city_token_span",
            "requested_counts": sorted(requested_counts),
            "final_transition_only": bool(args.final_transition_only),
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[v5 causal-source-writes] complete={completed} skipped={skipped} "
        f"directory={output_dir}",
        flush=True,
    )


def _validation_seeds(value: Any) -> set[int]:
    parsed = json.loads(str(value))
    return {int(seed) for seed in parsed}


def _causal_result_rows(path: Path) -> list[dict[str, Any]]:
    """Read atomic causal result shards from a file or output directory."""

    source = Path(path)
    if source.is_file():
        files = [source]
    elif (source / "shards").is_dir():
        files = sorted((source / "shards").glob("*.jsonl"))
    elif source.is_dir():
        files = sorted(source.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"Causal result path does not exist: {source}")
    if not files:
        raise ValueError(f"No causal JSONL result shards found at {source}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def command_causal_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.causal import (
        mechanism_continuations,
        run_mechanism_head_ablation_trials,
    )

    config = _config(args)
    output_dir = Path(args.output)
    expected_manifest = _causal_manifest(
        command="causal-heads",
        config=config,
        args=args,
        completed_shards=0,
    )
    expected_manifest.update(
        {
            "plan": str(args.plan.resolve()),
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        }
    )
    _assert_resume_manifest_compatible(output_dir, expected_manifest)
    if not (output_dir / "manifest.json").exists():
        _atomic_json(output_dir / "manifest.json", expected_manifest)
    model, tokenizer, adapter = _model(args)
    development = set(config.causal_development_seeds)
    rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in development
    ]
    plan = pd.read_csv(args.plan).reset_index().rename(
        columns={"index": "plan_file_row"}
    )
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan["plan_file_row"].isin(selected)]
    if plan.empty:
        raise ValueError(f"No causal plan rows remain for {args.model}")
    if set(plan["mechanism"].astype(str)) != {"retrieval_anchor_localization"}:
        raise ValueError("Causal-head runner requires the rebooted anchor plan")
    plan_validation_seeds = sorted(
        {
            seed
            for value in plan["validation_seeds"]
            for seed in _validation_seeds(value)
        }
    )
    rows = [row for row in rows if int(row["seed"]) in plan_validation_seeds]
    if not rows:
        raise ValueError("No generation rows match the plan validation seeds")

    tasks: list[tuple[dict[str, Any], dict[str, Any], Any | None]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        specifications, excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        excluded_rows.extend(
            {
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                **value,
            }
            for value in excluded
        )
        specifications = [
            value
            for value in specifications
            if _anchor_in_registered_scope(
                value,
                include_secondary=args.include_secondary,
                include_block_pre=args.include_block_pre,
            )
        ]
        if not specifications:
            continue
        seed = int(row["seed"])
        row_plan = plan.loc[
            plan["validation_seeds"].map(lambda value: seed in _validation_seeds(value))
        ]
        if row_plan.empty:
            raise ValueError(f"No cross-fit validation bank for seed {seed}")
        for specification in specifications:
            tasks.append((row, specification, None))
            tasks.extend(
                (row, specification, plan_row)
                for plan_row in row_plan.itertuples(index=False)
            )
    if args.limit is not None:
        distinct: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        for row, specification, _plan_row in tasks:
            key = (
                str(row.get("request_id", row.get("stimulus_id"))),
                str(specification["anchor_equivalence_id"]),
            )
            distinct.setdefault(key, (row, specification))
        selected_anchors = _diverse_anchor_subset(
            list(distinct.values()), args.limit
        )
        allowed = {
            (
                str(row.get("request_id", row.get("stimulus_id"))),
                str(specification["anchor_equivalence_id"]),
            )
            for row, specification in selected_anchors
        }
        tasks = [
            task
            for task in tasks
            if (
                str(task[0].get("request_id", task[0].get("stimulus_id"))),
                str(task[1]["anchor_equivalence_id"]),
            )
            in allowed
        ]

    shard_dir = output_dir / "shards"
    existing = list(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Causal-head shards already exist in {shard_dir}; "
            "resume or choose a new output directory"
        )
    completed = 0
    skipped = 0
    for task_index, (row, specification, plan_row) in enumerate(tasks, start=1):
        if plan_row is None:
            condition = "clean"
            repeat = 0
            heads: list[tuple[int, int]] = []
            plan_file_row = None
            fold = next(
                int(value)
                for value in plan.loc[
                    plan["validation_seeds"].map(
                        lambda raw: int(row["seed"]) in _validation_seeds(raw)
                    ),
                    "fold",
                ].unique()
            )
            bank_sha256 = "clean"
            planned_bank_size = int(
                plan.loc[
                    plan["validation_seeds"].map(
                        lambda raw: int(row["seed"]) in _validation_seeds(raw)
                    ),
                    "bank_size",
                ].iloc[0]
            )
        else:
            condition = str(plan_row.condition)
            repeat = int(plan_row.repeat)
            heads = _parse_heads(plan_row.heads)
            plan_file_row = int(plan_row.plan_file_row)
            fold = int(plan_row.fold)
            bank_sha256 = str(plan_row.bank_sha256)
            planned_bank_size = int(plan_row.bank_size)
        trial_id = _safe_stem(
            row.get("request_id", row.get("stimulus_id")),
            specification["anchor_equivalence_id"],
            condition,
            repeat,
            bank_sha256,
            args.head_ablation_scope,
        )
        shard_path = shard_dir / f"{trial_id}.jsonl"
        if args.resume and shard_path.exists():
            skipped += 1
            continue
        results = run_mechanism_head_ablation_trials(
            model,
            tokenizer,
            adapter,
            row,
            mechanism="retrieval_anchor_localization",
            heads=heads,
            condition=condition,
            anchor_equivalence_ids=[specification["anchor_equivalence_id"]],
            head_ablation_scope=str(args.head_ablation_scope),
        )
        if len(results) != 1 or results[0].get("status") != "ok":
            raise RuntimeError(
                f"Expected one completed anchor trial, received {results}"
            )
        result = results[0]
        result.update(
            {
                "trial_id": trial_id,
                "trial_complete": True,
                "plan_file_row": plan_file_row,
                "repeat": repeat,
                "crossfit_fold": fold,
                "bank_sha256": bank_sha256,
                "planned_bank_size": planned_bank_size,
            }
        )
        _atomic_jsonl(shard_path, [result])
        completed += 1
        print(
            f"[v5 causal-heads] {task_index}/{len(tasks)} "
            f"{specification['anchor_equivalence_id']} {condition} r{repeat}",
            flush=True,
        )
        progress_manifest = _causal_manifest(
            command="causal-heads",
            config=config,
            args=args,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
        )
        progress_manifest.update(
            {
                "plan": str(args.plan.resolve()),
                "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            }
        )
        _atomic_json(output_dir / "manifest.json", progress_manifest)
    _atomic_jsonl(output_dir / "excluded_anchors.jsonl", excluded_rows)
    manifest = _causal_manifest(
        command="causal-heads",
        config=config,
        args=args,
        completed_shards=len(list(shard_dir.glob("*.jsonl"))),
    )
    manifest.update(
        {
            "scheduled_anchor_condition_trials": len(tasks),
            "newly_completed": completed,
            "resume_skipped": skipped,
            "plan": str(args.plan.resolve()),
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "target_policy": "fixed_next_city_token_span",
            "head_ablation_scope": str(args.head_ablation_scope),
            "plan_validation_seeds": plan_validation_seeds,
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[v5 causal-heads] complete={completed} skipped={skipped} "
        f"directory={output_dir}",
        flush=True,
    )


def command_causal_source_edge(args: argparse.Namespace) -> None:
    """Run record-specific removal/restoration at localized retrieval sites."""

    import pandas as pd

    from realistic_niah_v5.causal import (
        mechanism_continuations,
        run_retrieval_source_edge_trials,
    )

    config = _config(args)
    output_dir = Path(args.output)
    plan_sha256 = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    manifest_extra = {
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "anchor_role": str(args.anchor_role),
        "source_control_policy": (
            "per_layer_closest_natural_write_norm_wrong_gold_record_with_"
            "natural_and_exact_residual_norm_matched_arms"
        ),
        "intervention_scope": "one_selected_layer_group_at_a_time",
        "restoration_layer_policy": "first_head_layer_in_crossfit_ranked_bank",
    }
    expected_manifest = _causal_manifest(
        command="causal-source-edge",
        config=config,
        args=args,
        completed_shards=0,
    )
    expected_manifest.update(manifest_extra)
    _assert_resume_manifest_compatible(output_dir, expected_manifest)
    if not (output_dir / "manifest.json").exists():
        _atomic_json(output_dir / "manifest.json", expected_manifest)

    model, tokenizer, adapter = _model(args)
    development = set(config.causal_development_seeds)
    rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in development
    ]
    plan = pd.read_csv(args.plan).reset_index().rename(
        columns={"index": "plan_file_row"}
    )
    plan = plan.loc[
        plan["model_label"].eq(args.model)
        & plan["mechanism"].eq("retrieval_anchor_localization")
        & plan["condition"].eq("selected_bank")
    ].reset_index(drop=True)
    if plan.empty:
        raise ValueError(f"No selected retrieval bank remains for {args.model}")
    plan_validation_seeds = sorted(
        {
            seed
            for value in plan["validation_seeds"]
            for seed in _validation_seeds(value)
        }
    )
    rows = [row for row in rows if int(row["seed"]) in plan_validation_seeds]
    if not rows:
        raise ValueError("No generation rows match the plan validation seeds")

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        specifications, excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        excluded_rows.extend(
            {
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                **value,
            }
            for value in excluded
        )
        for specification in specifications:
            roles = {str(value) for value in specification.get("anchor_roles", [])}
            if str(args.anchor_role) not in roles:
                continue
            if not _anchor_in_registered_scope(
                specification,
                include_secondary=args.include_secondary,
                include_block_pre=args.include_block_pre,
            ):
                continue
            candidates.append((row, specification))
    if not candidates:
        raise ValueError(f"No registered {args.anchor_role} source-edge anchors")
    distinct: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row, specification in candidates:
        key = (
            str(row.get("request_id", row.get("stimulus_id"))),
            str(specification["anchor_equivalence_id"]),
        )
        distinct.setdefault(key, (row, specification))
    tasks = _seed_first_anchor_subset(list(distinct.values()), args.limit)
    selected_seeds = sorted({int(row["seed"]) for row, _specification in tasks})

    shard_dir = output_dir / "shards"
    existing = list(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Source-edge shards already exist in {shard_dir}; "
            "resume or choose a new output directory"
        )
    completed = 0
    skipped = 0
    scheduled_condition_rows = 0
    for task_index, (row, specification) in enumerate(tasks, start=1):
        seed = int(row["seed"])
        row_plan = plan.loc[
            plan["validation_seeds"].map(
                lambda value: seed in _validation_seeds(value)
            )
        ]
        if len(row_plan) != 1:
            raise ValueError(
                f"Expected one selected cross-fit bank for seed {seed}, "
                f"found {len(row_plan)}"
            )
        plan_row = row_plan.iloc[0]
        heads = _parse_heads(plan_row["heads"])
        bank_sha256 = str(plan_row["bank_sha256"])
        anchor_trial_id = _safe_stem(
            row.get("request_id", row.get("stimulus_id")),
            specification["anchor_equivalence_id"],
            "source_edge",
            bank_sha256,
        )
        shard_path = shard_dir / f"{anchor_trial_id}.jsonl"
        if args.resume and shard_path.exists():
            skipped += 1
            with shard_path.open("r", encoding="utf-8") as handle:
                scheduled_condition_rows += sum(1 for line in handle if line.strip())
            continue
        results = run_retrieval_source_edge_trials(
            model,
            tokenizer,
            adapter,
            row,
            heads=heads,
            anchor_equivalence_id=str(specification["anchor_equivalence_id"]),
        )
        if not results or any(result.get("status") != "ok" for result in results):
            raise RuntimeError(
                f"Expected completed source-edge rows, received {results}"
            )
        for condition_index, result in enumerate(results):
            result.update(
                {
                    "anchor_trial_id": anchor_trial_id,
                    "trial_id": _safe_stem(
                        anchor_trial_id,
                        result["intervention_layer"],
                        result["condition"],
                    ),
                    "condition_index": int(condition_index),
                    "crossfit_fold": int(plan_row["fold"]),
                    "plan_file_row": int(plan_row["plan_file_row"]),
                    "bank_sha256": bank_sha256,
                    "planned_bank_size": int(plan_row["bank_size"]),
                }
            )
        _atomic_jsonl(shard_path, results)
        completed += 1
        scheduled_condition_rows += len(results)
        print(
            f"[v5 causal-source-edge] {task_index}/{len(tasks)} "
            f"{specification['anchor_equivalence_id']} rows={len(results)}",
            flush=True,
        )
        progress_manifest = _causal_manifest(
            command="causal-source-edge",
            config=config,
            args=args,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
        )
        progress_manifest.update(manifest_extra)
        _atomic_json(output_dir / "manifest.json", progress_manifest)
    _atomic_jsonl(output_dir / "excluded_anchors.jsonl", excluded_rows)
    manifest = _causal_manifest(
        command="causal-source-edge",
        config=config,
        args=args,
        completed_shards=len(list(shard_dir.glob("*.jsonl"))),
    )
    manifest.update(
        {
            **manifest_extra,
            "scheduled_anchor_trials": len(tasks),
            "scheduled_condition_rows": int(scheduled_condition_rows),
            "newly_completed": completed,
            "resume_skipped": skipped,
            "selected_seeds": selected_seeds,
            "missing_development_seeds": sorted(development - set(selected_seeds)),
            "target_policy": "fixed_next_city_token_span",
            "plan_validation_seeds": plan_validation_seeds,
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[v5 causal-source-edge] complete={completed} skipped={skipped} "
        f"rows={scheduled_condition_rows} directory={output_dir}",
        flush=True,
    )


def _load_behavior_anchor_routing(path: Path) -> dict[str, Any]:
    """Load a frozen target-grammar to semantic-anchor routing policy."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Anchor routing must be a JSON object")
    if str(raw.get("axis")) != "target_grammar_class":
        raise ValueError(
            "Behavior anchor routing axis must be target_grammar_class"
        )
    policy_id = str(raw.get("policy_id", "")).strip()
    if not policy_id:
        raise ValueError("Anchor routing requires a non-empty policy_id")
    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError("Anchor routing requires a non-empty routes object")

    def normalize_rule(label: str, value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            raise ValueError(f"Anchor route {label!r} must be an object")
        required = [str(role) for role in value.get("required", [])]
        optional = [str(role) for role in value.get("optional", [])]
        if not required:
            raise ValueError(
                f"Anchor route {label!r} requires at least one required role"
            )
        if any(not role for role in (*required, *optional)):
            raise ValueError(f"Anchor route {label!r} has an empty role")
        if len(set((*required, *optional))) != len((*required, *optional)):
            raise ValueError(f"Anchor route {label!r} repeats a role")
        return {"required": required, "optional": optional}

    routes = {
        str(grammar): normalize_rule(str(grammar), rule)
        for grammar, rule in raw_routes.items()
    }
    fallback = raw.get("fallback")
    return {
        "schema_version": str(raw.get("schema_version", "")),
        "policy_id": policy_id,
        "axis": "target_grammar_class",
        "routes": routes,
        "fallback": (
            None
            if fallback is None
            else normalize_rule("fallback", fallback)
        ),
    }


def _validate_behavior_selection_window(
    routing: dict[str, Any],
    *,
    selection_anchor_role: str | None,
    target_grammar_class: str | None,
    require_selection_anchor: bool = True,
) -> list[str]:
    """Validate one grammar route and optionally require the selection site."""

    if require_selection_anchor and selection_anchor_role is None:
        raise ValueError(
            "A multi-site head-bank experiment requires a recorded "
            "selection_anchor_role"
        )
    if target_grammar_class is None:
        declared_roles = list(
            dict.fromkeys(
                role
                for rule in routing["routes"].values()
                for role in [*rule["required"], *rule["optional"]]
            )
        )
        if (
            require_selection_anchor
            and str(selection_anchor_role) not in declared_roles
        ):
            raise ValueError(
                "The all-grammar routing policy does not contain the exact "
                f"selection site {selection_anchor_role!r}"
            )
        return declared_roles
    rule = routing["routes"].get(
        str(target_grammar_class), routing.get("fallback")
    )
    if rule is None:
        raise ValueError(
            "The target grammar is absent from the routed window: "
            f"{target_grammar_class!r}"
        )
    declared_roles = list(dict.fromkeys([*rule["required"], *rule["optional"]]))
    if require_selection_anchor and str(selection_anchor_role) not in declared_roles:
        raise ValueError(
            "The multi-site ablation window must contain the exact site used "
            "to select the head bank: "
            f"selection={selection_anchor_role!r}, "
            f"window={declared_roles!r}"
        )
    return declared_roles


def _selection_intervention_site_decoupled(
    selection_anchor_role: str | None,
    intervention_anchor_roles: list[str] | tuple[str, ...],
) -> bool:
    """Return whether bank localization and intervention use disjoint sites."""

    if selection_anchor_role is None:
        return False
    roles = {str(value) for value in intervention_anchor_roles}
    if not roles:
        raise ValueError("At least one intervention anchor role is required")
    return str(selection_anchor_role) not in roles


def _route_transition_anchors(
    specifications: list[dict[str, Any]],
    routing: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile one fixed multi-site prefill window per routed transition."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for specification in specifications:
        key = (
            int(specification["from_occurrence"]),
            int(specification["to_occurrence"]),
            int(specification["target_output_token_start"]),
            int(specification["target_output_token_end"]),
            str(specification["target_city"]).casefold(),
        )
        grouped.setdefault(key, []).append(specification)
    routed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for transition_key, transition_specs in sorted(grouped.items()):
        grammar_pairs = {
            str(value["grammar_pair"]) for value in transition_specs
        }
        if len(grammar_pairs) != 1:
            raise ValueError("One transition resolved to multiple grammar pairs")
        grammar_pair = next(iter(grammar_pairs))
        target_grammar = grammar_pair.rsplit(" -> ", 1)[-1]
        rule = routing["routes"].get(target_grammar, routing.get("fallback"))
        if rule is None:
            excluded.append(
                {
                    "status": "not_applicable",
                    "exclusion_reason": "target_grammar_not_routed",
                    "target_grammar_class": target_grammar,
                    "grammar_pair": grammar_pair,
                    "from_occurrence": transition_key[0],
                    "to_occurrence": transition_key[1],
                }
            )
            continue
        by_role: dict[str, dict[str, Any]] = {}
        for specification in transition_specs:
            for role in specification.get("anchor_roles", []):
                role = str(role)
                existing = by_role.get(role)
                if (
                    existing is not None
                    and str(existing["anchor_equivalence_id"])
                    != str(specification["anchor_equivalence_id"])
                ):
                    raise ValueError(
                        f"Transition has multiple anchors for role {role!r}"
                    )
                by_role[role] = specification
        missing_required = [
            role for role in rule["required"] if role not in by_role
        ]
        if missing_required:
            excluded.append(
                {
                    "status": "not_applicable",
                    "exclusion_reason": "required_routing_role_unavailable",
                    "missing_required_roles": missing_required,
                    "target_grammar_class": target_grammar,
                    "grammar_pair": grammar_pair,
                    "from_occurrence": transition_key[0],
                    "to_occurrence": transition_key[1],
                }
            )
            continue
        requested_roles = [*rule["required"], *rule["optional"]]
        applied_roles = [role for role in requested_roles if role in by_role]
        missing_optional = [
            role for role in rule["optional"] if role not in by_role
        ]
        selected_by_id: dict[str, dict[str, Any]] = {}
        for role in applied_roles:
            specification = by_role[role]
            selected_by_id.setdefault(
                str(specification["anchor_equivalence_id"]), specification
            )
        selected = sorted(
            selected_by_id.values(),
            key=lambda value: int(value["query_output_token_index"]),
        )
        if not selected:
            raise RuntimeError("A valid route produced no anchor sites")
        latest = dict(selected[-1])
        anchor_ids = [
            str(value["anchor_equivalence_id"]) for value in selected
        ]
        query_indices = [
            int(value["query_output_token_index"]) for value in selected
        ]
        route_id = (
            f"{transition_key[0]}->{transition_key[1]}@route-q"
            + "-".join(str(value) for value in query_indices)
        )
        latest.update(
            {
                "anchor_equivalence_id": route_id,
                "query_site_id": route_id,
                "anchor_role": "grammar_routed_retrieval_window",
                "anchor_roles": ["grammar_routed_retrieval_window"],
                "timing_stage": "grammar_routed_retrieval_window",
                "routed_target_grammar_class": target_grammar,
                "routed_anchor_roles_required": list(rule["required"]),
                "routed_anchor_roles_optional": list(rule["optional"]),
                "routed_anchor_roles_applied": applied_roles,
                "routed_missing_optional_roles": missing_optional,
                "routed_anchor_equivalence_ids": anchor_ids,
                "routed_query_output_token_indices": query_indices,
            }
        )
        routed.append(latest)
    return routed, excluded


def command_causal_heads_behavior(args: argparse.Namespace) -> None:
    """Free-generate from city-pre after selected/random head ablations."""

    import pandas as pd

    from realistic_niah_v5.causal import (
        mechanism_continuations,
        run_retrieval_head_behavior_trial,
    )

    config = _config(args)
    output_dir = Path(args.output)
    plan_sha256 = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    requested_conditions = tuple(dict.fromkeys(args.conditions))
    anchor_routing: dict[str, Any] | None = None
    anchor_routing_sha256: str | None = None
    if args.anchor_routing is not None:
        anchor_routing = _load_behavior_anchor_routing(args.anchor_routing)
        anchor_routing_sha256 = hashlib.sha256(
            args.anchor_routing.read_bytes()
        ).hexdigest()
    reference_keys: set[tuple[str, int, int]] | None = None
    reference_sha256: str | None = None
    if args.reference_results is not None:
        reference_rows = [
            row
            for row in _causal_result_rows(args.reference_results)
            if str(row.get("model_label")) == str(args.model)
            and str(row.get("condition")) == str(args.reference_condition)
            and str(row.get("behavior_outcome"))
            == str(args.reference_behavior_outcome)
        ]
        reference_keys = {
            (
                str(row.get("request_id", row.get("stimulus_id"))),
                int(row["from_occurrence"]),
                int(row["to_occurrence"]),
            )
            for row in reference_rows
        }
        if not reference_keys:
            raise ValueError("No reference transitions match the requested filter")
        reference_sha256 = hashlib.sha256(
            json.dumps(sorted(reference_keys)).encode("utf-8")
        ).hexdigest()
    frozen_anchor_registry_rows: list[dict[str, Any]] | None = None
    frozen_anchor_registry_sha256: str | None = None
    if args.anchor_registry_input is not None:
        with args.anchor_registry_input.open("r", encoding="utf-8") as handle:
            frozen_anchor_registry_rows = [
                json.loads(line) for line in handle if line.strip()
            ]
        if not frozen_anchor_registry_rows:
            raise ValueError("--anchor-registry-input is empty")
        registry_keys = [
            (
                str(row["request_id"]),
                int(row["from_occurrence"]),
                int(row["to_occurrence"]),
            )
            for row in frozen_anchor_registry_rows
        ]
        if len(registry_keys) != len(set(registry_keys)):
            raise ValueError("--anchor-registry-input contains duplicate transitions")
        if int(args.limit) < len(frozen_anchor_registry_rows):
            raise ValueError(
                "--limit cannot truncate a supplied frozen anchor registry: "
                f"limit={args.limit}, registry={len(frozen_anchor_registry_rows)}"
            )
        frozen_anchor_registry_sha256 = hashlib.sha256(
            args.anchor_registry_input.read_bytes()
        ).hexdigest()
    manifest_extra = {
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_sha256,
        "anchor_role": (
            "grammar_routed_retrieval_window"
            if anchor_routing is not None
            else str(args.anchor_role)
        ),
        "anchor_routing": (
            str(args.anchor_routing.resolve())
            if args.anchor_routing is not None
            else None
        ),
        "anchor_routing_sha256": anchor_routing_sha256,
        "anchor_routing_policy_id": (
            str(anchor_routing["policy_id"])
            if anchor_routing is not None
            else None
        ),
        "branch_policy": (
            "teacher_force_through_"
            + (
                "latest_registered_anchor"
                if anchor_routing is not None
                else "registered_anchor"
            )
            + (
                "_then_persistent_decode_head_ablation"
                if int(args.decode_head_ablation_steps) == -1
                else
                "_then_fixed_decode_head_ablation_window_then_free_generate"
                if int(args.decode_head_ablation_steps) > 0
                else "_then_free_generate"
            )
        ),
        "conditions": list(requested_conditions),
        "max_new_tokens": int(args.max_new_tokens),
        "decode_head_ablation_steps": int(
            args.decode_head_ablation_steps
        ),
        "reference_results": (
            str(args.reference_results.resolve())
            if args.reference_results is not None
            else None
        ),
        "reference_condition": (
            str(args.reference_condition)
            if args.reference_results is not None
            else None
        ),
        "reference_behavior_outcome": (
            str(args.reference_behavior_outcome)
            if args.reference_results is not None
            else None
        ),
        "reference_transition_sha256": reference_sha256,
        "reference_transition_count": (
            len(reference_keys) if reference_keys is not None else None
        ),
        "anchor_registry_input": (
            str(args.anchor_registry_input.resolve())
            if args.anchor_registry_input is not None
            else None
        ),
        "anchor_registry_input_sha256": frozen_anchor_registry_sha256,
    }
    expected_manifest = _causal_manifest(
        command="causal-heads-behavior",
        config=config,
        args=args,
        completed_shards=0,
    )
    expected_manifest.update(manifest_extra)
    _assert_resume_manifest_compatible(output_dir, expected_manifest)
    if not (output_dir / "manifest.json").exists():
        _atomic_json(output_dir / "manifest.json", expected_manifest)

    model, tokenizer, adapter = _model(args)
    development = set(config.causal_development_seeds)
    causal_confirmation = set(config.causal_confirmation_seeds)
    causal_evaluation_seeds = development | causal_confirmation
    rows = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) in causal_evaluation_seeds
        and (
            str(args.evaluation_split) == "all"
            or str(row.get("split")) == str(args.evaluation_split)
        )
    ]
    rows, evaluation_counts = _filter_rows_by_count(
        rows,
        config=config,
        requested=args.counts,
    )
    if not rows:
        raise ValueError(
            "No registered behavioral generations remain for "
            f"counts {evaluation_counts} and split {args.evaluation_split!r}"
        )
    plan = pd.read_csv(args.plan).reset_index().rename(
        columns={"index": "plan_file_row"}
    )
    plan = plan.loc[
        plan["model_label"].eq(args.model)
        & plan["mechanism"].eq("retrieval_anchor_localization")
    ].reset_index(drop=True)
    if plan.empty:
        raise ValueError(f"No retrieval-head plan remains for {args.model}")
    planned_anchor_role: str | None = None
    if "selection_anchor_role" in plan:
        planned_roles = {
            str(value)
            for value in plan["selection_anchor_role"].dropna().unique()
            if str(value).strip()
        }
        if len(planned_roles) > 1:
            raise ValueError(
                f"Behavior plan mixes selection anchor roles: {planned_roles}"
            )
        if planned_roles:
            planned_anchor_role = next(iter(planned_roles))
    planned_target_grammar: str | None = None
    if "selection_target_grammar_class" in plan:
        planned_grammars = {
            str(value)
            for value in plan[
                "selection_target_grammar_class"
            ].dropna().unique()
            if str(value).strip()
        }
        if len(planned_grammars) > 1:
            raise ValueError(
                "Behavior plan mixes target grammar classes: "
                f"{planned_grammars}"
            )
        if planned_grammars:
            planned_target_grammar = next(iter(planned_grammars))
    if args.behavior_all_routed_grammars and anchor_routing is None:
        raise ValueError(
            "--behavior-all-routed-grammars requires --anchor-routing"
        )
    if (
        args.behavior_all_routed_grammars
        and args.behavior_target_grammar_class is not None
    ):
        raise ValueError(
            "Choose either --behavior-all-routed-grammars or one explicit "
            "--behavior-target-grammar-class"
        )
    behavior_target_grammar = (
        None
        if args.behavior_all_routed_grammars
        else (
            planned_target_grammar
            if args.behavior_target_grammar_class is None
            else str(args.behavior_target_grammar_class)
        )
    )
    cross_grammar_bank_transfer = bool(
        args.behavior_all_routed_grammars
        or (
            planned_target_grammar is not None
            and behavior_target_grammar is not None
            and behavior_target_grammar != planned_target_grammar
        )
    )
    allow_selection_scope_bank_transfer = bool(
        args.allow_cross_grammar_bank_transfer
        or args.allow_selection_scope_bank_transfer
    )
    planned_target_surface: str | None = None
    if "selection_target_retrieval_surface_variant" in plan:
        planned_surfaces = {
            str(value)
            for value in plan[
                "selection_target_retrieval_surface_variant"
            ].dropna().unique()
            if str(value).strip()
        }
        if len(planned_surfaces) > 1:
            raise ValueError(
                "Behavior plan mixes target retrieval surface variants: "
                f"{planned_surfaces}"
            )
        if planned_surfaces:
            planned_target_surface = next(iter(planned_surfaces))
    requested_target_surface = args.target_retrieval_surface_variant
    site_bank_transfer = bool(
        planned_anchor_role is not None
        and anchor_routing is None
        and str(args.anchor_role) != planned_anchor_role
    )
    surface_bank_transfer = bool(
        planned_target_surface is not None
        and requested_target_surface is not None
        and planned_target_surface != str(requested_target_surface)
    )
    selection_scope_bank_transfer = bool(
        cross_grammar_bank_transfer
        or site_bank_transfer
        or surface_bank_transfer
    )
    if selection_scope_bank_transfer and not allow_selection_scope_bank_transfer:
        raise ValueError(
            "Selection grammar/surface/site differs from the behavior scope. "
            "Pass --allow-selection-scope-bank-transfer for an explicit "
            "development transfer audit: "
            f"selection=({planned_target_grammar!r}, "
            f"{planned_target_surface!r}, {planned_anchor_role!r}), "
            f"behavior=({behavior_target_grammar!r}, "
            f"{requested_target_surface!r}, {args.anchor_role!r})"
        )
    target_surface_filter = (
        requested_target_surface
        if selection_scope_bank_transfer
        else (
            planned_target_surface
            if planned_target_surface is not None
            else requested_target_surface
        )
    )
    selection_window_roles: list[str] = []
    if anchor_routing is not None:
        selection_window_roles = _validate_behavior_selection_window(
            anchor_routing,
            selection_anchor_role=planned_anchor_role,
            target_grammar_class=behavior_target_grammar,
            require_selection_anchor=not selection_scope_bank_transfer,
        )
    manifest_extra.update(
        {
            "plan_selection_anchor_role": planned_anchor_role,
            "plan_selection_target_grammar_class": planned_target_grammar,
            "plan_selection_target_retrieval_surface_variant": (
                planned_target_surface
            ),
            "behavior_target_grammar_class": behavior_target_grammar,
            "behavior_target_retrieval_surface_variant": (
                target_surface_filter
            ),
            "cross_grammar_bank_transfer": cross_grammar_bank_transfer,
            "selection_scope_bank_transfer": selection_scope_bank_transfer,
            "selection_scope_bank_transfer_explicitly_allowed": bool(
                allow_selection_scope_bank_transfer
            ),
            "selection_ablation_site_identity_enforced": bool(
                planned_anchor_role is not None
                and anchor_routing is None
                and not selection_scope_bank_transfer
            ),
            "selection_anchor_in_intervention_window_enforced": bool(
                planned_anchor_role is not None
                and anchor_routing is not None
                and not selection_scope_bank_transfer
            ),
            "selection_window_declared_roles": selection_window_roles,
            "same_selected_bank_reused_across_intervention_sites": bool(
                anchor_routing is not None
            ),
        }
    )
    plan_validation_seeds = sorted(
        {
            seed
            for value in plan["validation_seeds"]
            for seed in _validation_seeds(value)
        }
    )
    rows = [row for row in rows if int(row["seed"]) in plan_validation_seeds]
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        specifications, excluded = mechanism_continuations(
            row,
            tokenizer,
            mechanism="retrieval_anchor_localization",
        )
        excluded_rows.extend(
            {
                "request_id": row.get("request_id", row.get("stimulus_id")),
                "model_label": row.get("model_label"),
                "seed": row.get("seed"),
                **value,
            }
            for value in excluded
        )
        if anchor_routing is None:
            row_candidates = [
                specification
                for specification in specifications
                if str(args.anchor_role)
                in {
                    str(value)
                    for value in specification.get("anchor_roles", [])
                }
                if (
                    behavior_target_grammar is None
                    or str(specification.get("grammar_pair", "")).rsplit(
                        " -> ", 1
                    )[-1]
                    == behavior_target_grammar
                )
                if (
                    target_surface_filter is None
                    or str(
                        specification.get(
                            "target_retrieval_surface_variant", ""
                        )
                    )
                    == str(target_surface_filter)
                )
            ]
        else:
            row_candidates, route_exclusions = _route_transition_anchors(
                specifications,
                anchor_routing,
            )
            if behavior_target_grammar is not None:
                row_candidates = [
                    specification
                    for specification in row_candidates
                    if str(
                        specification.get("routed_target_grammar_class")
                    )
                    == behavior_target_grammar
                ]
            if target_surface_filter is not None:
                row_candidates = [
                    specification
                    for specification in row_candidates
                    if str(
                        specification.get(
                            "target_retrieval_surface_variant", ""
                        )
                    )
                    == str(target_surface_filter)
                ]
            excluded_rows.extend(
                {
                    "request_id": row.get(
                        "request_id", row.get("stimulus_id")
                    ),
                    "model_label": row.get("model_label"),
                    "seed": row.get("seed"),
                    **value,
                }
                for value in route_exclusions
            )
        for specification in row_candidates:
            transition_key = (
                str(row.get("request_id", row.get("stimulus_id"))),
                int(specification["from_occurrence"]),
                int(specification["to_occurrence"]),
            )
            if reference_keys is not None and transition_key not in reference_keys:
                continue
            if not _anchor_in_registered_scope(
                specification,
                include_secondary=args.include_secondary,
                include_block_pre=args.include_block_pre,
            ):
                continue
            candidates.append((row, specification))
    distinct: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row, specification in candidates:
        key = (
            str(row.get("request_id", row.get("stimulus_id"))),
            str(specification["anchor_equivalence_id"]),
        )
        distinct.setdefault(key, (row, specification))
    if frozen_anchor_registry_rows is not None:
        candidates_by_transition: dict[
            tuple[str, int, int], tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        for row, specification in distinct.values():
            transition_key = (
                str(row.get("request_id", row.get("stimulus_id"))),
                int(specification["from_occurrence"]),
                int(specification["to_occurrence"]),
            )
            if transition_key in candidates_by_transition:
                raise ValueError(
                    "Multiple routed candidates map to frozen registry key "
                    f"{transition_key}"
                )
            candidates_by_transition[transition_key] = (row, specification)
        scheduled_registry_rows = [
            registry_row
            for registry_row in frozen_anchor_registry_rows
            if int(registry_row["seed"])
            in {int(row["seed"]) for row in rows}
            and int(registry_row["gold_count"]) in set(evaluation_counts)
        ]
        anchors = []
        for registry_row in scheduled_registry_rows:
            transition_key = (
                str(registry_row["request_id"]),
                int(registry_row["from_occurrence"]),
                int(registry_row["to_occurrence"]),
            )
            if transition_key not in candidates_by_transition:
                raise ValueError(
                    "Frozen anchor registry row is not reproducible under the "
                    f"current parser/routing/split: {transition_key}"
                )
            row, specification = candidates_by_transition[transition_key]
            observed = {
                "anchor_equivalence_id": str(
                    specification["anchor_equivalence_id"]
                ),
                "target_grammar_class": _candidate_target_grammar(
                    specification
                ),
                "target_retrieval_surface_variant": str(
                    specification.get("target_retrieval_surface_variant", "")
                ),
                "anchor_roles": [
                    str(value)
                    for value in specification.get("anchor_roles", [])
                ],
            }
            mismatches = {
                key: {"frozen": registry_row.get(key), "observed": value}
                for key, value in observed.items()
                if registry_row.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "Frozen anchor registry metadata changed for "
                    f"{transition_key}: {mismatches}"
                )
            anchors.append((row, specification))
    elif args.anchor_sampling == "prompt_balanced":
        anchors = _prompt_balanced_anchor_subset(list(distinct.values()), args.limit)
    elif args.anchor_sampling == "prompt_final_transition":
        anchors = _prompt_final_transition_anchor_subset(
            list(distinct.values()), args.limit
        )
    else:
        anchors = _seed_first_anchor_subset(
            list(distinct.values()), args.limit
        )
    if not anchors:
        requested_site = (
            "grammar-routed retrieval-window"
            if anchor_routing is not None
            else str(args.anchor_role)
        )
        raise ValueError(f"No registered {requested_site} behavioral anchors")
    scheduled_anchor_registry_rows = [
        {
            "request_id": str(
                row.get("request_id", row.get("stimulus_id"))
            ),
            "seed": int(row["seed"]),
            "gold_count": _row_gold_count(row),
            "anchor_equivalence_id": str(
                specification["anchor_equivalence_id"]
            ),
            "from_occurrence": int(specification["from_occurrence"]),
            "to_occurrence": int(specification["to_occurrence"]),
            "target_grammar_class": _candidate_target_grammar(
                specification
            ),
            "target_retrieval_surface_variant": str(
                specification.get("target_retrieval_surface_variant", "")
            ),
            "anchor_roles": [
                str(value)
                for value in specification.get("anchor_roles", [])
            ],
        }
        for row, specification in anchors
    ]
    anchor_registry_rows = (
        frozen_anchor_registry_rows
        if frozen_anchor_registry_rows is not None
        else scheduled_anchor_registry_rows
    )
    anchor_registry_path = output_dir / "selected_anchor_registry.jsonl"
    _atomic_jsonl(anchor_registry_path, anchor_registry_rows)
    anchor_registry_sha256 = hashlib.sha256(
        anchor_registry_path.read_bytes()
    ).hexdigest()
    manifest_extra.update(
        {
            "selected_anchor_registry": str(anchor_registry_path.resolve()),
            "selected_anchor_registry_sha256": anchor_registry_sha256,
            "selected_anchor_count": len(anchor_registry_rows),
            "selected_prompt_count": len(
                {row["request_id"] for row in anchor_registry_rows}
            ),
            "scheduled_anchor_count": len(scheduled_anchor_registry_rows),
            "scheduled_prompt_count": len(
                {row["request_id"] for row in scheduled_anchor_registry_rows}
            ),
        }
    )
    if args.freeze_anchor_registry_only:
        _atomic_jsonl(output_dir / "excluded_anchors.jsonl", excluded_rows)
        manifest = _causal_manifest(
            command="causal-heads-behavior",
            config=config,
            args=args,
            completed_shards=0,
        )
        manifest.update(
            {
                **manifest_extra,
                "registry_only": True,
                "selected_seeds": sorted(
                    {int(row["seed"]) for row in anchor_registry_rows}
                ),
                "selected_anchor_grammar_counts": {
                    grammar: sum(
                        row["target_grammar_class"] == grammar
                        for row in anchor_registry_rows
                    )
                    for grammar in sorted(
                        {
                            row["target_grammar_class"]
                            for row in anchor_registry_rows
                        }
                    )
                },
                "selected_anchor_count_counts": {
                    str(count): sum(
                        int(row["gold_count"]) == count
                        for row in anchor_registry_rows
                    )
                    for count in sorted(
                        {
                            int(row["gold_count"])
                            for row in anchor_registry_rows
                        }
                    )
                },
                "scheduled_anchor_grammar_counts": {
                    grammar: sum(
                        row["target_grammar_class"] == grammar
                        for row in scheduled_anchor_registry_rows
                    )
                    for grammar in sorted(
                        {
                            row["target_grammar_class"]
                            for row in scheduled_anchor_registry_rows
                        }
                    )
                },
                "scheduled_anchor_count_counts": {
                    str(count): sum(
                        int(row["gold_count"]) == count
                        for row in scheduled_anchor_registry_rows
                    )
                    for count in sorted(
                        {
                            int(row["gold_count"])
                            for row in scheduled_anchor_registry_rows
                        }
                    )
                },
                "plan_validation_seeds": plan_validation_seeds,
            }
        )
        _atomic_json(output_dir / "manifest.json", manifest)
        print(
            "[v5 causal-heads-behavior] froze anchor registry only: "
            f"anchors={len(anchor_registry_rows)} "
            f"sha256={anchor_registry_sha256} directory={output_dir}",
            flush=True,
        )
        return
    tasks: list[tuple[dict[str, Any], dict[str, Any], Any | None]] = []
    for row, specification in anchors:
        seed = int(row["seed"])
        row_plan = plan.loc[
            plan["validation_seeds"].map(
                lambda value: seed in _validation_seeds(value)
            )
        ]
        selected_rows = row_plan.loc[
            row_plan["condition"].eq("selected_bank")
        ]
        random_conditions = {
            "layer_matched_random",
            "global_random",
        } & set(requested_conditions)
        if len(selected_rows) != 1 or row_plan["fold"].nunique() != 1:
            raise ValueError(
                f"Seed {seed} does not map to exactly one selected bank "
                "from one cross-fit fold"
            )
        for random_condition in sorted(random_conditions):
            random_rows = row_plan.loc[
                row_plan["condition"].eq(random_condition)
            ]
            if len(random_rows) != int(config.causal_random_controls):
                raise ValueError(
                    f"Seed {seed} does not have the requested "
                    f"{config.causal_random_controls} {random_condition} banks"
                )
        if "clean" in requested_conditions:
            tasks.append((row, specification, None))
        tasks.extend(
            (row, specification, plan_row)
            for plan_row in row_plan.itertuples(index=False)
            if str(plan_row.condition) in requested_conditions
        )

    shard_dir = output_dir / "shards"
    existing = list(shard_dir.glob("*.jsonl")) if shard_dir.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Behavioral shards already exist in {shard_dir}; "
            "resume or choose a new output directory"
        )
    completed = 0
    skipped = 0
    outcome_counts: dict[str, int] = {}
    for task_index, (row, specification, plan_row) in enumerate(tasks, start=1):
        seed = int(row["seed"])
        if plan_row is None:
            condition = "clean"
            repeat = 0
            heads: list[tuple[int, int]] = []
            bank_sha256 = "clean"
            plan_file_row = None
            fold = next(
                int(value)
                for value in plan.loc[
                    plan["validation_seeds"].map(
                        lambda raw: seed in _validation_seeds(raw)
                    ),
                    "fold",
                ].unique()
            )
            seed_bank_sizes = {
                int(value)
                for value in plan.loc[
                    plan["validation_seeds"].map(
                        lambda raw: seed in _validation_seeds(raw)
                    ),
                    "bank_size",
                ].unique()
            }
            if len(seed_bank_sizes) != 1:
                raise ValueError(
                    f"Seed {seed} has ambiguous planned bank sizes: "
                    f"{sorted(seed_bank_sizes)}"
                )
            planned_bank_size = next(iter(seed_bank_sizes))
        else:
            condition = str(plan_row.condition)
            repeat = int(plan_row.repeat)
            heads = _parse_heads(plan_row.heads)
            bank_sha256 = str(plan_row.bank_sha256)
            plan_file_row = int(plan_row.plan_file_row)
            fold = int(plan_row.fold)
            planned_bank_size = int(plan_row.bank_size)
        trial_id = _safe_stem(
            row.get("request_id", row.get("stimulus_id")),
            specification["anchor_equivalence_id"],
            "free_continuation",
            f"decode_window_{int(args.decode_head_ablation_steps)}",
            condition,
            repeat,
            bank_sha256,
        )
        shard_path = shard_dir / f"{trial_id}.jsonl"
        if args.resume and shard_path.exists():
            saved, resume_error = _load_completed_behavior_shard(
                shard_path, expected_trial_id=trial_id
            )
            if resume_error is None:
                assert saved is not None
                skipped += 1
                for result in saved:
                    outcome = str(result.get("behavior_outcome"))
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                continue
            archived = _archive_invalid_behavior_shard(shard_path)
            print(
                "[v5 causal-heads-behavior] invalid resume shard; "
                f"recomputing exact trial path={shard_path} "
                f"archive={archived} error={resume_error}",
                file=sys.stderr,
                flush=True,
            )
        intervention_anchor_ids = specification.get(
            "routed_anchor_equivalence_ids",
            [str(specification["anchor_equivalence_id"])],
        )
        result = run_retrieval_head_behavior_trial(
            model,
            tokenizer,
            adapter,
            row,
            heads=heads,
            condition=condition,
            anchor_equivalence_id=intervention_anchor_ids,
            max_new_tokens=args.max_new_tokens,
            decode_head_ablation_steps=args.decode_head_ablation_steps,
        )
        if "routed_anchor_equivalence_ids" in specification:
            result.update(
                {
                    "branch_anchor_equivalence_id": result[
                        "anchor_equivalence_id"
                    ],
                    "anchor_equivalence_id": str(
                        specification["anchor_equivalence_id"]
                    ),
                    "query_site_id": str(specification["query_site_id"]),
                    "anchor_role": "grammar_routed_retrieval_window",
                    "anchor_roles": ["grammar_routed_retrieval_window"],
                    "timing_stage": "grammar_routed_retrieval_window",
                    "anchor_routing_policy_id": str(
                        anchor_routing["policy_id"]
                    ),
                    "routed_target_grammar_class": str(
                        specification["routed_target_grammar_class"]
                    ),
                    "routed_anchor_roles_required": list(
                        specification["routed_anchor_roles_required"]
                    ),
                    "routed_anchor_roles_optional": list(
                        specification["routed_anchor_roles_optional"]
                    ),
                    "routed_anchor_roles_applied": list(
                        specification["routed_anchor_roles_applied"]
                    ),
                    "routed_missing_optional_roles": list(
                        specification["routed_missing_optional_roles"]
                    ),
                }
            )
        result.update(
            {
                "trial_id": trial_id,
                "repeat": repeat,
                "crossfit_fold": fold,
                "plan_file_row": plan_file_row,
                "bank_sha256": bank_sha256,
                "planned_bank_size": planned_bank_size,
                "head_selection_anchor_role": planned_anchor_role,
                "head_selection_target_grammar_class": (
                    planned_target_grammar
                ),
                "behavior_target_grammar_class": behavior_target_grammar,
                "cross_grammar_bank_transfer": cross_grammar_bank_transfer,
                "selection_scope_bank_transfer": selection_scope_bank_transfer,
                "head_selection_target_retrieval_surface_variant": (
                    planned_target_surface
                ),
                "selection_anchor_in_intervention_window": bool(
                    planned_anchor_role is not None
                    and planned_anchor_role
                    in {
                        str(value)
                        for value in result.get(
                            "intervention_anchor_roles", []
                        )
                    }
                ),
                "same_selected_bank_reused_across_intervention_sites": bool(
                    int(result.get("intervention_site_count", 1)) > 1
                ),
            }
        )
        _atomic_jsonl(shard_path, [result])
        completed += 1
        outcome = str(result["behavior_outcome"])
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        print(
            f"[v5 causal-heads-behavior] {task_index}/{len(tasks)} "
            f"{specification['anchor_equivalence_id']} {condition} r{repeat} "
            f"outcome={outcome} "
            f"city={result.get('first_generated_city_record')}",
            flush=True,
        )
        progress_manifest = _causal_manifest(
            command="causal-heads-behavior",
            config=config,
            args=args,
            completed_shards=len(list(shard_dir.glob("*.jsonl"))),
        )
        progress_manifest.update(manifest_extra)
        _atomic_json(output_dir / "manifest.json", progress_manifest)
    _atomic_jsonl(output_dir / "excluded_anchors.jsonl", excluded_rows)
    selected_seeds = sorted({int(row["seed"]) for row, _specification in anchors})
    manifest = _causal_manifest(
        command="causal-heads-behavior",
        config=config,
        args=args,
        completed_shards=len(list(shard_dir.glob("*.jsonl"))),
    )
    manifest.update(
        {
            **manifest_extra,
            "scheduled_anchor_condition_trials": len(tasks),
            "newly_completed": completed,
            "resume_skipped": skipped,
            "selected_seeds": selected_seeds,
            "missing_development_seeds": sorted(development - set(selected_seeds)),
            "causal_confirmation_seeds": sorted(causal_confirmation),
            "selected_causal_confirmation_seeds": sorted(
                causal_confirmation & set(selected_seeds)
            ),
            "outcome_counts": outcome_counts,
            "plan_validation_seeds": plan_validation_seeds,
            "selected_anchor_grammar_counts": {
                grammar: sum(
                    row["target_grammar_class"] == grammar
                    for row in anchor_registry_rows
                )
                for grammar in sorted(
                    {
                        row["target_grammar_class"]
                        for row in anchor_registry_rows
                    }
                )
            },
            "selected_anchor_count_counts": {
                str(count): sum(
                    int(row["gold_count"]) == count
                    for row in anchor_registry_rows
                )
                for count in sorted(
                    {int(row["gold_count"]) for row in anchor_registry_rows}
                )
            },
            "scheduled_anchor_grammar_counts": {
                grammar: sum(
                    row["target_grammar_class"] == grammar
                    for row in scheduled_anchor_registry_rows
                )
                for grammar in sorted(
                    {
                        row["target_grammar_class"]
                        for row in scheduled_anchor_registry_rows
                    }
                )
            },
            "scheduled_anchor_count_counts": {
                str(count): sum(
                    int(row["gold_count"]) == count
                    for row in scheduled_anchor_registry_rows
                )
                for count in sorted(
                    {
                        int(row["gold_count"])
                        for row in scheduled_anchor_registry_rows
                    }
                )
            },
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    print(
        f"[v5 causal-heads-behavior] complete={completed} skipped={skipped} "
        f"outcomes={outcome_counts} directory={output_dir}",
        flush=True,
    )


def command_causal_pre_city_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import run_pre_city_head_ablation_trials

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    rows = (
        [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
        if args.allow_unregistered
        else registered_records(raw_rows, config, model_label=args.model)
    )
    split = "all" if args.include_discovery else args.split
    rows = _split_rows(rows, split)
    rows = _parser_cohort_rows(rows, args.cohort)
    if args.gold_count is not None:
        from realistic_niah_v5.parsing import parse_trace_record

        rows = [
            row
            for row in rows
            if int(parse_trace_record(row)["gold_count"]) == int(args.gold_count)
        ]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    variants = set(plan["query_variant"].astype(str))
    if variants != {args.query_variant}:
        raise ValueError(
            f"Selected plan rows have variants={sorted(variants)}; "
            f"expected only {args.query_variant}"
        )
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for row_index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", row_index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{row_index:05d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows = run_pre_city_head_ablation_trials(
                model,
                tokenizer,
                adapter,
                row,
                query_variant=args.query_variant,
                heads=[],
                condition="clean",
            )
            for plan_row in plan.itertuples(index=True):
                results = run_pre_city_head_ablation_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    query_variant=args.query_variant,
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                )
                for result in results:
                    result["plan_row"] = int(plan_row.Index)
                    result["repeat"] = int(plan_row.repeat)
                    result["mechanism"] = str(plan_row.mechanism)
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        result[field] = float(getattr(plan_row, field))
                    for field in (
                        "relative_mass_defined_heads",
                        "raw_mass_defined_heads",
                        "confirmation_raw_mass_defined_heads",
                        "confirmation_relative_mass_defined_heads",
                    ):
                        result[field] = int(getattr(plan_row, field))
                    result["attention_mass_split"] = str(
                        plan_row.attention_mass_split
                    )
                    result["confirmation_mass_split"] = str(
                        plan_row.confirmation_mass_split
                    )
                    result["attention_mass_aggregation"] = str(
                        plan_row.attention_mass_aggregation
                    )
                trial_rows.extend(results)
            _atomic_write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-pre-city-heads] {row_index}/{len(rows)} "
            f"{action} request={request_id} variant={args.query_variant}",
            flush=True,
        )
    output_rows = []
    for shard_path in shard_paths:
        output_rows.extend(read_jsonl(shard_path))
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-pre-city-heads] requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_pre_city_all_sites(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.pre_city import (
        run_all_site_pre_city_damage_trial,
    )

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    raw_rows = read_jsonl(args.generations)
    if args.allow_unregistered:
        rows = [
            row
            for row in raw_rows
            if str(row.get("model_label", row.get("model"))) == args.model
        ]
    else:
        rows = registered_records(raw_rows, config, model_label=args.model)
    if not args.include_discovery:
        rows = [row for row in rows if row["split"] == "confirmation"]
    rows = _parser_cohort_rows(rows, args.cohort)
    if args.gold_count is not None:
        from realistic_niah_v5.parsing import gold_records

        rows = [
            row
            for row in rows
            if len(gold_records(row)) == int(args.gold_count)
        ]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].eq(args.model)].reset_index(drop=True)
    if args.plan_rows:
        selected_rows = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected_rows)]
    else:
        plan = plan.loc[
            plan["query_variant"].astype(str).eq(args.query_variant)
        ]
    variants = set(plan["query_variant"].astype(str))
    if variants != {args.query_variant}:
        raise ValueError(
            f"Selected plan rows have variants={sorted(variants)}; "
            f"expected only {args.query_variant}"
        )
    worker_id = str(args.worker_id or f"{socket.gethostname()}-{os.getpid()}")
    shard_root = args.output.parent / f"{args.output.stem}_shards_v2"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        request_id = str(
            row.get("request_id", row.get("stimulus_id", row_index))
        )
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        row_root = shard_root / f"{row_index:04d}__{safe_id}"
        tasks.append(
            {
                "task_id": f"{row_index:04d}:clean",
                "row_index": row_index,
                "request_id": request_id,
                "row": row,
                "plan": None,
                "shard": row_root / "clean.jsonl",
            }
        )
        for plan_index, plan_row in plan.iterrows():
            tasks.append(
                {
                    "task_id": f"{row_index:04d}:plan:{int(plan_index):04d}",
                    "row_index": row_index,
                    "request_id": request_id,
                    "row": row,
                    "plan": {**plan_row.to_dict(), "plan_row": int(plan_index)},
                    "shard": row_root / f"plan_{int(plan_index):04d}.jsonl",
                }
            )
    if args.overwrite:
        if worker_id != "primary":
            raise ValueError("Only worker_id=primary may initialize --overwrite")
        for task in tasks:
            Path(task["shard"]).unlink(missing_ok=True)
            Path(task["shard"]).with_suffix(".jsonl.claim").unlink(
                missing_ok=True
            )

    deadline = time.time() + float(args.worker_wait_timeout)
    completed_here = 0
    while True:
        missing = [task for task in tasks if not Path(task["shard"]).exists()]
        if not missing:
            break
        claimed_any = False
        for task in missing:
            shard_path = Path(task["shard"])
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            claim = _claim_restartable_task(
                shard_path,
                worker_id=worker_id,
                stale_seconds=float(args.claim_stale_seconds),
            )
            if claim is None:
                continue
            claimed_any = True
            try:
                plan_row = task["plan"]
                if plan_row is None:
                    result = run_all_site_pre_city_damage_trial(
                        model,
                        tokenizer,
                        adapter,
                        task["row"],
                        query_variant=args.query_variant,
                        heads=[],
                        condition="clean",
                    )
                else:
                    result = run_all_site_pre_city_damage_trial(
                        model,
                        tokenizer,
                        adapter,
                        task["row"],
                        query_variant=args.query_variant,
                        heads=_parse_heads(str(plan_row["heads"])),
                        condition=str(plan_row["condition"]),
                    )
                    result["plan_row"] = int(plan_row["plan_row"])
                    result["repeat"] = int(plan_row["repeat"])
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        result[field] = float(plan_row[field])
                    result["attention_mass_split"] = str(
                        plan_row["attention_mass_split"]
                    )
                    result["confirmation_mass_split"] = str(
                        plan_row["confirmation_mass_split"]
                    )
                    result["attention_mass_aggregation"] = str(
                        plan_row["attention_mass_aggregation"]
                    )
                result["restartable_task_id"] = str(task["task_id"])
                result["execution_worker_id"] = worker_id
                _atomic_write_jsonl(shard_path, [result])
                completed_here += 1
                print(
                    "[v5 causal-pre-city-all-sites] "
                    f"worker={worker_id} completed={completed_here} "
                    f"task={task['task_id']} request={task['request_id']} "
                    f"variant={args.query_variant}",
                    flush=True,
                )
            finally:
                claim.unlink(missing_ok=True)
        if all(Path(task["shard"]).exists() for task in tasks):
            break
        if time.time() > deadline:
            remaining = sum(
                not Path(task["shard"]).exists() for task in tasks
            )
            raise TimeoutError(
                f"Timed out waiting for {remaining}/{len(tasks)} shared tasks"
            )
        if not claimed_any:
            time.sleep(float(args.worker_poll_seconds))

    output_rows = []
    for task in tasks:
        shard_rows = read_jsonl(Path(task["shard"]))
        if len(shard_rows) != 1:
            raise ValueError(f"Task shard must contain one row: {task['shard']}")
        output_rows.extend(shard_rows)
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-pre-city-all-sites] wrote requests={len(rows)} "
        f"rows={len(output_rows)} worker={worker_id} "
        f"completed_here={completed_here} output={args.output}",
        flush=True,
    )


def command_causal_marker_needle_patch(args: argparse.Namespace) -> None:
    from realistic_niah_v5.pre_city import (
        pre_city_token_queries,
        run_marker_needle_patch_trials,
    )

    model, tokenizer, adapter = _model(args)
    generations: dict[str, dict[str, Any]] = {}
    for path in args.generations:
        for row in read_jsonl(path):
            if str(row.get("model_label", row.get("model"))) != args.model:
                continue
            request_id = str(row.get("request_id", row.get("stimulus_id")))
            if request_id in generations:
                prior = generations[request_id]
                if int(prior["seed"]) != int(row["seed"]):
                    raise ValueError(f"Conflicting generation row: {request_id}")
                continue
            generations[request_id] = row
    pairs = [
        row
        for row in read_jsonl(args.pairs)
        if str(row["model_label"]) == args.model
        and (args.split == "all" or str(row["split"]) == args.split)
    ]
    if args.max_pairs is not None:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise ValueError("No marker-needle patch pairs matched the request")
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for pair_index, pair in enumerate(pairs, start=1):
        full_id = str(pair["full_request_id"])
        counterfactual_id = str(pair["counterfactual_request_id"])
        if full_id not in generations or counterfactual_id not in generations:
            raise ValueError(
                "Marker pair rows missing from generations: "
                f"{full_id}/{counterfactual_id}"
            )
        for variant in args.query_variants:
            for layer in args.layers:
                safe_pair = str(pair["pair_id"]).replace("/", "__").replace("\\", "__")
                shard = (
                    shard_root
                    / f"{pair_index:05d}__{safe_pair}__{variant}__L{int(layer):03d}.jsonl"
                )
                tasks.append((pair, variant, int(layer), shard))
    if args.overwrite:
        for _pair, _variant, _layer, shard in tasks:
            shard.unlink(missing_ok=True)
    completed = 0
    alias_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for task_index, (pair, variant, layer, shard) in enumerate(tasks, start=1):
        full_id = str(pair["full_request_id"])
        query_candidates, _query_exclusions = pre_city_token_queries(
            generations[full_id], tokenizer
        )
        query_matches = [
            query
            for query in query_candidates
            if query.query_variant == variant
            and int(query.occurrence) == int(pair["occurrence"])
        ]
        if len(query_matches) != 1:
            raise ValueError(
                f"Cannot resolve marker alias for {pair['pair_id']}/{variant}"
            )
        alias_key = (
            str(pair["pair_id"]),
            int(layer),
            int(query_matches[0].query_output_token_count),
        )
        if shard.exists():
            rows = read_jsonl(shard)
            if len(rows) != 6:
                raise ValueError(f"Incomplete marker patch shard: {shard}")
            alias_cache.setdefault(alias_key, rows)
            action = "reused"
        elif alias_key in alias_cache:
            source_rows = alias_cache[alias_key]
            source_variant = str(source_rows[0]["query_variant"])
            rows = []
            for source in source_rows:
                row = dict(source)
                row["query_variant"] = variant
                row["query_alias_forward_reused"] = True
                row["query_alias_reused_from_variant"] = source_variant
                rows.append(row)
            _atomic_write_jsonl(shard, rows)
            completed += 1
            action = f"alias_reused_from_{source_variant}"
        else:
            counterfactual_id = str(pair["counterfactual_request_id"])
            rows = run_marker_needle_patch_trials(
                model,
                tokenizer,
                adapter,
                generations[full_id],
                generations[counterfactual_id],
                query_variant=variant,
                occurrence=int(pair["occurrence"]),
                layer=layer,
            )
            for row in rows:
                row["pair_id"] = str(pair["pair_id"])
                row["pair_eligibility"] = str(pair["pair_eligibility"])
                row["full_exact_count"] = bool(pair["full_exact_count"])
                row["counterfactual_exact_count"] = bool(
                    pair["counterfactual_exact_count"]
                )
                row["query_alias_forward_reused"] = False
                row["query_alias_reused_from_variant"] = None
            _atomic_write_jsonl(shard, rows)
            alias_cache[alias_key] = rows
            completed += 1
            action = "captured"
        print(
            f"[v5 causal-marker-needle-patch] {task_index}/{len(tasks)} "
            f"{action} variant={variant} L{layer} pair={pair['pair_id']}",
            flush=True,
        )
    output_rows = []
    for _pair, _variant, _layer, shard in tasks:
        shard_rows = read_jsonl(shard)
        if len(shard_rows) != 6:
            raise ValueError(f"Marker patch shard must contain six rows: {shard}")
        output_rows.extend(shard_rows)
    _atomic_write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-marker-needle-patch] pairs={len(pairs)} "
        f"tasks={len(tasks)} rows={len(output_rows)} completed_here={completed} "
        f"output={args.output}",
        flush=True,
    )


def command_causal_answer_query_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.causal import run_answer_query_head_ablation_trial

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].astype(str).eq(args.model)].reset_index(
        drop=True
    )
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows = [
                run_answer_query_head_ablation_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    heads=[],
                    condition="clean",
                    site_id=args.site_id,
                )
            ]
            for plan_row in plan.itertuples(index=True):
                result = run_answer_query_head_ablation_trial(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                    site_id=args.site_id,
                )
                result["plan_row"] = int(plan_row.Index)
                result["mechanism"] = str(plan_row.mechanism)
                result["repeat"] = int(plan_row.repeat)
                for field in (
                    "target_needle_raw_mass",
                    "target_needle_relative_mass",
                    "selected_aggregation_raw_mass",
                    "selected_aggregation_relative_mass",
                    "confirmation_target_needle_raw_mass",
                    "confirmation_target_needle_relative_mass",
                    "confirmation_selected_aggregation_raw_mass",
                    "confirmation_selected_aggregation_relative_mass",
                    "selected_aggregation_broad_score",
                    "selected_aggregation_broad_coverage",
                    "confirmation_selected_aggregation_broad_score",
                    "confirmation_selected_aggregation_broad_coverage",
                ):
                    value = getattr(plan_row, field)
                    if value is None or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        continue
                    result[field] = float(value)
                for field in (
                    "selected_head_count",
                    "prompt_bank_size",
                    "trace_bank_size",
                    "prompt_trace_head_overlap",
                    "prompt_aggregation_raw_mass",
                    "prompt_aggregation_relative_mass",
                    "trace_aggregation_raw_mass",
                    "trace_aggregation_relative_mass",
                    "prompt_aggregation_broad_score",
                    "prompt_aggregation_broad_coverage",
                    "trace_aggregation_broad_score",
                    "trace_aggregation_broad_coverage",
                    "confirmation_prompt_aggregation_raw_mass",
                    "confirmation_prompt_aggregation_relative_mass",
                    "confirmation_trace_aggregation_raw_mass",
                    "confirmation_trace_aggregation_relative_mass",
                    "confirmation_prompt_aggregation_broad_score",
                    "confirmation_prompt_aggregation_broad_coverage",
                    "confirmation_trace_aggregation_broad_score",
                    "confirmation_trace_aggregation_broad_coverage",
                ):
                    if hasattr(plan_row, field):
                        value = getattr(plan_row, field)
                        # Prompt-only and trace-only plan rows intentionally
                        # leave the joint-bank audit columns empty.  Pandas
                        # represents those optional CSV cells as NaN; omit
                        # them from the heterogeneous JSONL row instead of
                        # attempting int(NaN) or emitting non-standard NaN.
                        if value is None or (
                            isinstance(value, float) and math.isnan(value)
                        ):
                            continue
                        result[field] = (
                            int(value)
                            if field in {
                                "selected_head_count",
                                "prompt_bank_size",
                                "trace_bank_size",
                                "prompt_trace_head_overlap",
                            }
                            else float(value)
                        )
                result["attention_mass_split"] = str(
                    plan_row.attention_mass_split
                )
                result["confirmation_mass_split"] = str(
                    plan_row.confirmation_mass_split
                )
                result["attention_mass_aggregation"] = str(
                    plan_row.attention_mass_aggregation
                )
                result["selected_aggregation_metric"] = str(
                    plan_row.selected_aggregation_metric
                )
                trial_rows.append(result)
            write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-answer-query-heads] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    output_rows = []
    for path in shard_paths:
        output_rows.extend(read_jsonl(path))
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-answer-query-heads] wrote requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_response_reference_heads(args: argparse.Namespace) -> None:
    import pandas as pd

    from realistic_niah_v5.response_reference import (
        run_response_reference_head_ablation_trials,
    )

    model, tokenizer, adapter = _model(args)
    rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label", row.get("model"))) == args.model
    ]
    rows = _parser_cohort_rows(rows, args.cohort)
    rows = _split_rows(rows, args.split)
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan = pd.read_csv(args.plan)
    plan = plan.loc[plan["model_label"].astype(str).eq(args.model)].reset_index(
        drop=True
    )
    if args.response_types:
        allowed = {str(value) for value in args.response_types}
        plan = plan.loc[plan["response_type"].astype(str).isin(allowed)]
    if args.position_variants:
        allowed_positions = {str(value) for value in args.position_variants}
        plan = plan.loc[
            plan["position_variant"].astype(str).isin(allowed_positions)
        ]
    if args.bank_scopes:
        allowed_scopes = {str(value) for value in args.bank_scopes}
        plan = plan.loc[plan["bank_scope"].astype(str).isin(allowed_scopes)]
    if args.plan_rows:
        selected = {int(value) for value in args.plan_rows}
        plan = plan.loc[plan.index.isin(selected)]
    if plan.empty:
        raise ValueError("No response-reference plan rows were selected")

    shard_root = args.output.parent / f"{args.output.stem}_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for index, row in enumerate(rows, start=1):
        request_id = str(row.get("request_id", row.get("stimulus_id", index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        shard_path = shard_root / f"{index:04d}__{safe_id}.jsonl"
        if args.overwrite or not shard_path.exists():
            trial_rows: list[dict[str, Any]] = []
            clean_strata = sorted(
                {
                    (str(plan_row.response_type), str(plan_row.position_variant))
                    for plan_row in plan.itertuples(index=False)
                }
            )
            for response_type, position_variant in clean_strata:
                trial_rows.extend(
                    run_response_reference_head_ablation_trials(
                        model,
                        tokenizer,
                        adapter,
                        row,
                        response_type=response_type,
                        position_variant=position_variant,
                        heads=[],
                        condition="clean",
                    )
                )
            for plan_row in plan.itertuples(index=True):
                result_rows = run_response_reference_head_ablation_trials(
                    model,
                    tokenizer,
                    adapter,
                    row,
                    response_type=str(plan_row.response_type),
                    position_variant=str(plan_row.position_variant),
                    heads=_parse_heads(plan_row.heads),
                    condition=str(plan_row.condition),
                )
                for result in result_rows:
                    result["plan_row"] = int(plan_row.Index)
                    result["repeat"] = int(plan_row.repeat)
                    result["bank_scope"] = str(plan_row.bank_scope)
                    for field in (
                        "target_needle_raw_mass",
                        "target_needle_relative_mass",
                        "discovery_target_needle_raw_mass",
                        "discovery_target_needle_relative_mass",
                        "confirmation_target_needle_raw_mass",
                        "confirmation_target_needle_relative_mass",
                    ):
                        if hasattr(plan_row, field):
                            result[field] = float(getattr(plan_row, field))
                trial_rows.extend(result_rows)
            write_jsonl(shard_path, trial_rows)
            action = "captured"
        else:
            action = "reused"
        shard_paths.append(shard_path)
        print(
            f"[v5 causal-response-reference] {index}/{len(rows)} "
            f"{action} {request_id}",
            flush=True,
        )
    output_rows = []
    for path in shard_paths:
        output_rows.extend(read_jsonl(path))
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-response-reference] requests={len(rows)} "
        f"rows={len(output_rows)} output={args.output}",
        flush=True,
    )


def command_causal_tokens(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import run_token_corruption_trial

    config = _config(args)
    model, tokenizer, _adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    if not args.include_discovery:
        rows = [row for row in rows if row["split"] == "confirmation"]
    output_rows = []
    for index, row in enumerate(rows, start=1):
        output_rows.extend(
            run_token_corruption_trial(
                model,
                tokenizer,
                row,
                config=config,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(f"[v5 causal-tokens] {index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)


def command_causal_context(args: argparse.Namespace) -> None:
    import numpy as np

    from realistic_niah_v5.causal import run_query_context_mask_trial

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    output_dir = Path(args.output)
    state_dir = output_dir / "states"
    index_rows = []
    for row_index, row in enumerate(rows, start=1):
        site_ids = _row_site_ids(
            row, site_kind=args.site_kind, site_id=args.site_id
        )
        for site_id in site_ids:
            for condition in args.conditions:
                result = run_query_context_mask_trial(
                    model,
                    adapter,
                    tokenizer,
                    row,
                    site_id=site_id,
                    condition=condition,
                    layers=args.layers,
                )
                states = np.asarray(result.pop("states"), dtype=np.float32)
                layers = np.asarray(result["layers"], dtype=np.int64)
                stem = _safe_stem(result["request_id"], site_id, condition)
                state_path = state_dir / f"{stem}.npz"
                _atomic_npz(state_path, states=states, layers=layers)
                result["states_path"] = str(state_path.relative_to(output_dir))
                index_rows.append(result)
        print(f"[v5 causal-context] {row_index}/{len(rows)}", flush=True)
    index_path = output_dir / "context_index.jsonl"
    write_jsonl(index_path, index_rows)
    print(f"[v5 causal-context] wrote {len(index_rows)} trials to {index_path}")


def command_causal_writes(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    from realistic_niah_v5.causal import capture_natural_head_writes
    from realistic_niah_v5.encoding import build_native_trace_encoding

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    directions = (
        _load_layer_directions(args.directions) if args.directions else None
    )
    frames = []
    vector_index = []
    output = Path(args.output)
    for row_index, row in enumerate(rows, start=1):
        encoding = build_native_trace_encoding(
            row,
            tokenizer,
            site_id=args.site_id,
            candidate_counts=config.candidate_counts,
        )
        frame, writes = capture_natural_head_writes(
            model,
            adapter,
            encoding,
            layers=args.layers,
            count_direction=directions,
        )
        frames.append(frame)
        if args.save_vectors:
            ordered = sorted(writes)
            stem = _safe_stem(encoding.request_id, args.site_id, "natural_writes")
            vector_path = output.parent / f"{output.stem}_vectors" / f"{stem}.npz"
            _atomic_npz(
                vector_path,
                layers=np.asarray([layer for layer, _head in ordered], dtype=np.int64),
                heads=np.asarray([head for _layer, head in ordered], dtype=np.int64),
                writes=np.stack([writes[key].numpy() for key in ordered]),
            )
            vector_index.append(
                {
                    "request_id": encoding.request_id,
                    "site_id": args.site_id,
                    "vectors_path": str(vector_path.resolve()),
                }
            )
        print(f"[v5 causal-writes] {row_index}/{len(rows)}", flush=True)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if args.save_vectors:
        write_jsonl(output.with_suffix(".vectors.jsonl"), vector_index)
    print(f"[v5 causal-writes] wrote {len(result)} rows to {output}")


def command_causal_subspace_fit(args: argparse.Namespace) -> None:
    import numpy as np

    from realistic_niah_v5.causal import fit_centroid_subspace
    from realistic_niah_v5.representation import cohort_mask, load_capture_dataset

    config = _config(args)
    dataset = load_capture_dataset(
        args.capture_index, site_kinds=[args.site_kind]
    )
    metadata = dataset.metadata
    mask = cohort_mask(metadata, args.cohort)
    mask &= metadata["split"].astype(str).eq("discovery").to_numpy()
    if args.site_id:
        mask &= metadata["site_id"].astype(str).eq(args.site_id).to_numpy()
    if (
        args.site_kind not in {"answer_query", "answer_query_v2", "answer_query_v3"}
        and config.representation_n10_only
    ):
        mask &= metadata["gold_count"].astype(int).eq(10).to_numpy()
    available_layers = sorted(
        int(value) for value in metadata.loc[mask, "layer"].unique()
    )
    layers = available_layers if args.layers is None else sorted(set(args.layers))
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"Subspace fit layers are unavailable: {missing}")
    arrays: dict[str, Any] = {}
    fit_rows = []
    for layer in layers:
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        states = dataset.states[layer_mask]
        label_column = (
            "gold_count"
            if args.site_kind in {"answer_query", "answer_query_v2", "answer_query_v3"}
            else "occurrence"
        )
        labels = metadata.loc[layer_mask, label_column].to_numpy(dtype=int)
        center, basis = fit_centroid_subspace(states, labels, rank=args.rank)
        arrays[f"center_L{layer}"] = center
        arrays[f"basis_L{layer}"] = basis
        fit_rows.append(
            {
                "layer": int(layer),
                "observations": int(len(states)),
                "labels": sorted(int(value) for value in np.unique(labels)),
                "effective_rank": int(basis.shape[1]),
            }
        )
    _atomic_npz(args.output, **arrays)
    audit = {
        "schema_version": "realistic_niah_v5_subspace_fit_v1",
        "capture_index": str(args.capture_index.resolve()),
        "selection_split": "discovery",
        "confirmation_used_for_fit": False,
        "cohort": args.cohort,
        "site_kind": args.site_kind,
        "site_id": args.site_id,
        "label": (
            "gold_count"
            if args.site_kind in {"answer_query", "answer_query_v2", "answer_query_v3"}
            else "occurrence"
        ),
        "basis_orientation": (
            "each component is signed toward increasing label when its linear "
            "label association is nonzero"
        ),
        "requested_rank": int(args.rank),
        "fits": fit_rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[v5 causal-subspace-fit] wrote {len(fit_rows)} bases to {args.output}")


def command_causal_subspace(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import run_subspace_ablation

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    rows = _split_rows(rows, args.split)
    bases = _load_layer_bases(args.basis)
    import numpy as np

    with np.load(args.basis, allow_pickle=False) as archive:
        centers = {
            int(key.removeprefix("center_L")): np.asarray(
                archive[key], dtype=np.float32
            )
            for key in archive.files
            if key.startswith("center_L")
        }
    layers = sorted(bases) if args.layers is None else sorted(set(args.layers))
    if sorted(set(layers) - (set(bases) & set(centers))):
        raise ValueError("Every requested layer needs both center and basis arrays")
    output_rows = []
    for row_index, row in enumerate(rows, start=1):
        site_ids = _row_site_ids(
            row, site_kind=args.site_kind, site_id=args.site_id
        )
        for site_id in site_ids:
            for layer in layers:
                for dose in (0.0, *args.doses):
                    result = run_subspace_ablation(
                        model,
                        tokenizer,
                        adapter,
                        row,
                        site_id=site_id,
                        layer=layer,
                        center=centers[layer],
                        basis=bases[layer],
                        dose=dose,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if dose == 0.0:
                        result["condition"] = "clean_dose_zero"
                    output_rows.append(result)
        print(f"[v5 causal-subspace] {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, output_rows)
    print(f"[v5 causal-subspace] wrote {len(output_rows)} trials to {args.output}")


def command_causal_patch(args: argparse.Namespace) -> None:
    from realistic_niah_v4.modeling import generate_with_residual_interventions
    from realistic_niah_v5.causal import (
        capture_site_state,
        run_projected_patch_trials_from_states,
    )
    from realistic_niah_v5.parsing import parse_trace_record

    config = _config(args)
    model, tokenizer, adapter = _model(args)
    rows = registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    )
    row_by_id = {
        str(row.get("request_id", row.get("stimulus_id"))): row for row in rows
    }
    requested_conditions = tuple(args.conditions)
    needs_basis = causal_patch_conditions_need_basis(requested_conditions)
    if needs_basis and args.basis is None:
        raise ValueError(
            "--basis is required when projected or orthogonal conditions are requested"
        )
    bases = _load_layer_bases(args.basis) if args.basis is not None else {}
    explicit_layers = (
        tuple(sorted({int(value) for value in args.layers}))
        if args.layers is not None
        else None
    )
    if explicit_layers is not None and args.layer is not None:
        raise ValueError("Use either --layer or --layers, not both")
    if explicit_layers is not None and not explicit_layers:
        raise ValueError("--layers must contain at least one layer")
    output_rows = []
    pairs = read_jsonl(args.pairs)
    shard_root = args.output.parent / f"{args.output.stem}_shards"
    if args.restartable:
        shard_root.mkdir(parents=True, exist_ok=True)
    state_cache: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    self_patch_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def cached_state(
        request_id: str,
        site_id: str,
        layer: int,
    ) -> tuple[Any, Any]:
        key = (request_id, site_id, int(layer))
        if key not in state_cache:
            state_cache[key] = capture_site_state(
                model,
                adapter,
                tokenizer,
                row_by_id[request_id],
                site_id=site_id,
                layer=int(layer),
            )
        return state_cache[key]

    total_cells = sum(
        len(explicit_layers)
        if explicit_layers is not None
        else 1
        for _pair in pairs
    )
    completed_cells = 0
    for pair_index, pair in enumerate(pairs, start=1):
        receiver_id = str(pair["receiver_request_id"])
        donor_id = str(pair["donor_request_id"])
        if not bool(pair.get("receiver_exact_count")) or not bool(
            pair.get("donor_exact_count")
        ):
            raise ValueError(
                "Answer execution is correct-only: both receiver and donor "
                f"must be clean-correct ({receiver_id}/{donor_id})"
            )
        if receiver_id not in row_by_id or donor_id not in row_by_id:
            raise KeyError(f"Unknown receiver/donor pair: {receiver_id}/{donor_id}")
        receiver_parsed = parse_trace_record(row_by_id[receiver_id])
        donor_parsed = parse_trace_record(row_by_id[donor_id])
        if (
            not bool(receiver_parsed["parser"].get("trace_one_to_one"))
            or not bool(donor_parsed["parser"].get("trace_one_to_one"))
            or not bool(receiver_parsed.get("exact_count"))
            or not bool(donor_parsed.get("exact_count"))
        ):
            raise ValueError(
                "Answer execution runtime revalidation failed: receiver and "
                "donor must both be strict one-to-one and baseline-final-answer "
                f"correct ({receiver_id}/{donor_id})"
            )
        if int(receiver_parsed["gold_count"]) != int(pair["receiver_count"]):
            raise ValueError(f"Receiver count mismatch for {receiver_id}")
        if int(donor_parsed["gold_count"]) != int(pair["donor_count"]):
            raise ValueError(f"Donor count mismatch for {donor_id}")
        receiver_site = str(pair.get("receiver_site_id", args.receiver_site_id))
        donor_site = str(pair.get("donor_site_id", args.donor_site_id))
        pair_layers = (
            explicit_layers
            if explicit_layers is not None
            else (int(pair.get("layer", args.layer)),)
            if pair.get("layer", args.layer) is not None
            else ()
        )
        if not pair_layers:
            raise ValueError(
                "Every patch pair needs a layer unless --layer/--layers is supplied"
            )
        for layer in pair_layers:
            if needs_basis and layer not in bases:
                raise ValueError(f"Patch basis has no layer {layer}")
            shard_path = (
                shard_root / f"L{layer:03d}" / f"{pair_index:05d}.jsonl"
                if explicit_layers is not None
                else shard_root / f"{pair_index:05d}.jsonl"
            )
            if args.restartable and shard_path.exists() and not args.overwrite:
                output_rows.extend(read_jsonl(shard_path))
                completed_cells += 1
                print(
                    f"[v5 causal-patch] {completed_cells}/{total_cells} "
                    f"L{layer} pair={pair_index} reused",
                    flush=True,
                )
                continue
            receiver, receiver_state = cached_state(
                receiver_id,
                receiver_site,
                layer,
            )
            donor, donor_state = cached_state(
                donor_id,
                donor_site,
                layer,
            )
            self_key = (receiver_id, receiver_site, int(layer))
            if (
                "self_patch" in requested_conditions
                and self_key not in self_patch_cache
            ):
                self_patch_cache[self_key] = generate_with_residual_interventions(
                    model,
                    tokenizer,
                    adapter,
                    receiver,
                    {int(layer): ([receiver.query_position], receiver_state)},
                    max_new_tokens=args.max_new_tokens,
                )
            trial_rows = run_projected_patch_trials_from_states(
                model,
                tokenizer,
                adapter,
                receiver,
                receiver_state,
                donor,
                donor_state,
                receiver_site_id=receiver_site,
                donor_site_id=donor_site,
                layer=layer,
                basis=bases.get(layer),
                max_new_tokens=args.max_new_tokens,
                self_patch_result=self_patch_cache.get(self_key),
                requested_conditions=requested_conditions,
            )
            for result in trial_rows:
                result["pair_id"] = pair.get("pair_id", pair_index)
                result["donor_role"] = pair.get("donor_role", "registered")
                for field in (
                    "pair_direction",
                    "receiver_exact_count",
                    "donor_exact_count",
                    "pair_eligibility",
                ):
                    if field in pair:
                        result[field] = pair[field]
            if args.restartable:
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(shard_path, trial_rows)
            output_rows.extend(trial_rows)
            completed_cells += 1
            print(
                f"[v5 causal-patch] {completed_cells}/{total_cells} "
                f"L{layer} pair={pair_index}",
                flush=True,
            )
    write_jsonl(args.output, output_rows)
    print(
        f"[v5 causal-patch] wrote {len(output_rows)} trials to {args.output}; "
        f"cached_states={len(state_cache)} cached_self_patches={len(self_patch_cache)}"
    )


def command_causal_analyze(args: argparse.Namespace) -> None:
    from realistic_niah_v5.causal import analyze_paired_causal_results

    result = analyze_paired_causal_results(
        args.trials,
        args.output,
        treatment=args.treatment,
        control=args.control,
        outcome=args.outcome,
        config=_config(args),
        mechanism=args.mechanism,
        bank_size=args.bank_size,
        transition_phase=args.transition_phase,
        anchor_role=args.anchor_role,
    )
    print(result.to_string(index=False))


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_split(parser: argparse.ArgumentParser, *, default: str = "confirmation") -> None:
    parser.add_argument(
        "--split",
        choices=["discovery", "confirmation", "all"],
        default=default,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V5 native-thinking representation and causal pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse = subparsers.add_parser("parse")
    parse.add_argument("--input", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--compact", action="store_true")
    parse.set_defaults(func=command_parse)

    generate = subparsers.add_parser("generate")
    _add_config(generate)
    _add_model(generate)
    generate.add_argument("--stimuli", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--counts",
        type=int,
        nargs="+",
        help=(
            "Optional registered count subset. The formal native/non-thinking "
            "paired panel uses --counts 1 2 3 4 5."
        ),
    )
    generate.set_defaults(func=command_generate)

    capture = subparsers.add_parser("capture")
    _add_config(capture)
    _add_model(capture)
    capture.add_argument("--generations", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--layers", type=int, nargs="+")
    capture.add_argument(
        "--site-kinds",
        nargs="+",
        help=(
            "Optional registered site subset. Use answer_query_v3 for a "
            "small final-count-only shard, or list running sites separately."
        ),
    )
    capture.add_argument(
        "--skip-span-pooling",
        action="store_true",
        help="Do not save auxiliary item-span mean/end arrays.",
    )
    capture.add_argument("--overwrite", action="store_true")
    capture.add_argument(
        "--site-ids",
        nargs="+",
        help=(
            "Explicit registered site IDs for an isolated extension capture; "
            "omitting this preserves the frozen main-site configuration."
        ),
    )
    capture.add_argument(
        "--allow-unregistered",
        action="store_true",
        help=(
            "Allow isolated supplement rows outside the primary 300-seed "
            "registry; rows are still model-filtered and never merged in place."
        ),
    )
    capture.set_defaults(func=command_capture)

    attention = subparsers.add_parser("attention")
    _add_config(attention)
    _add_model(attention)
    attention.add_argument("--generations", type=Path, required=True)
    attention.add_argument("--output", type=Path, required=True)
    attention.add_argument(
        "--mechanisms",
        nargs="+",
        choices=["targeted_retrieval", "progress_transition"],
        default=["targeted_retrieval", "progress_transition"],
    )
    attention.set_defaults(func=command_attention)

    attention_pre_city = subparsers.add_parser("attention-pre-city")
    _add_config(attention_pre_city)
    _add_model(attention_pre_city)
    attention_pre_city.add_argument("--generations", type=Path, required=True)
    attention_pre_city.add_argument("--output", type=Path, required=True)
    attention_pre_city.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    attention_pre_city.add_argument("--no-anchor", action="store_true")
    attention_pre_city.add_argument(
        "--split", choices=["all", "discovery", "confirmation"], default="all"
    )
    attention_pre_city.add_argument("--max-rows", type=int)
    attention_pre_city.add_argument("--overwrite", action="store_true")
    attention_pre_city.set_defaults(func=command_attention_pre_city)

    response_reference_attention = subparsers.add_parser(
        "response-reference-attention"
    )
    response_reference_attention.add_argument("--model", required=True)
    response_reference_attention.add_argument(
        "--generations", type=Path, required=True
    )
    response_reference_attention.add_argument(
        "--pre-city-attention", type=Path, required=True
    )
    response_reference_attention.add_argument("--output", type=Path, required=True)
    response_reference_attention.set_defaults(
        func=command_response_reference_attention
    )

    targeted_reference_attention = subparsers.add_parser(
        "attention-targeted-reference"
    )
    _add_config(targeted_reference_attention)
    _add_model(targeted_reference_attention)
    targeted_reference_attention.add_argument(
        "--generations", type=Path, required=True
    )
    targeted_reference_attention.add_argument("--output", type=Path, required=True)
    targeted_reference_attention.add_argument(
        "--split", choices=["all", "discovery", "confirmation"], default="all"
    )
    targeted_reference_attention.add_argument("--max-rows", type=int)
    targeted_reference_attention.add_argument("--overwrite", action="store_true")
    targeted_reference_attention.set_defaults(
        func=command_attention_targeted_reference
    )

    attention_answer_query = subparsers.add_parser("attention-answer-query")
    _add_model(attention_answer_query)
    attention_answer_query.add_argument(
        "--generations", type=Path, required=True
    )
    attention_answer_query.add_argument("--output", type=Path, required=True)
    attention_answer_query.add_argument(
        "--site-id", default="answer_query_v3"
    )
    attention_answer_query.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    attention_answer_query.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="all",
    )
    attention_answer_query.add_argument("--max-rows", type=int)
    attention_answer_query.add_argument("--overwrite", action="store_true")
    attention_answer_query.set_defaults(func=command_attention_answer_query)

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

    causal_plan = subparsers.add_parser("causal-plan")
    _add_config(causal_plan)
    causal_plan.add_argument("--source-writes", type=Path, required=True)
    causal_plan.add_argument("--output", type=Path, required=True)
    causal_plan.add_argument(
        "--bank-size",
        type=int,
        help=(
            "Override causal_primary_bank_size when freezing an explicitly "
            "labeled K-sweep plan."
        ),
    )
    causal_plan.add_argument(
        "--anchor-role",
        help=(
            "Rank source-specific writes only at this expanded parser role."
        ),
    )
    causal_plan.add_argument(
        "--target-grammar-class",
        help=(
            "Rank only transitions whose target event has this grammar class."
        ),
    )
    causal_plan.add_argument(
        "--target-retrieval-surface-variant",
        help=(
            "Optionally rank only one parser-audited target surface subtype, "
            "for example rank_before_city_compact."
        ),
    )
    causal_plan.add_argument(
        "--selection-metric",
        choices=[
            "target_source_relative_attention_mass",
            "target_source_attention_mass",
            "source_attention_mass",
            "target_minus_max_wrong_source_attention_mass",
            "source_specific_ov_write_norm",
        ],
        help=(
            "Head-ranking column. The config default is same-site absolute "
            "target-record attention mass; relative mass/top-1 remain audits."
        ),
    )
    causal_plan.add_argument(
        "--minimum-layer",
        type=int,
        help=(
            "Representation-guided lower layer bound for candidate heads."
        ),
    )
    causal_plan.add_argument(
        "--maximum-layer",
        type=int,
        help=(
            "Representation-guided upper layer bound for candidate heads. "
            "Set equal to --minimum-layer for an exact-layer positive control."
        ),
    )
    causal_plan.add_argument(
        "--selection-aggregation",
        choices=["request_equal", "seed_event_mean"],
        default="request_equal",
        help=(
            "How discovery events are pooled before equal-weight seed averaging. "
            "Use seed_event_mean to average every eligible event directly "
            "within each seed, matching the non-thinking selection rule."
        ),
    )
    causal_plan.add_argument(
        "--selection-eligibility-scope",
        choices=["primary", "local"],
        default="primary",
        help=(
            "Which parser-audited source events may contribute to head ranking. "
            "Use 'local' for a pooled all-grammar bank captured with "
            "causal-source-writes --include-secondary."
        ),
    )
    causal_plan.add_argument(
        "--random-control-matching",
        choices=["layer_matched", "global"],
        default="layer_matched",
        help=(
            "Random-bank matching rule. 'layer_matched' preserves the exact "
            "selected layer histogram; 'global' draws distinct same-K banks "
            "from all non-selected heads without constraining layers."
        ),
    )
    causal_plan.add_argument(
        "--development-smoke",
        action="store_true",
        help=(
            "Allow an explicitly non-inferential plan from an incomplete "
            "development-seed source-write smoke."
        ),
    )
    causal_plan.add_argument(
        "--selected-only-smoke",
        action="store_true",
        help=(
            "Freeze literal top-K selected banks without constructing random "
            "banks. This is only for non-inferential exact-site localization "
            "when an exact layer-matched control is combinatorially impossible."
        ),
    )
    causal_plan.add_argument(
        "--confirmation-plan",
        action="store_true",
        help=(
            "Rank once on every frozen source-write development seed and "
            "assign that fixed bank plus matched controls to the registered "
            "fresh causal_confirmation_seeds."
        ),
    )
    causal_plan.add_argument(
        "--full-panel-plan",
        action="store_true",
        help=(
            "Rank once on the available development source scope and assign "
            "one fixed bank plus matched controls to every registered causal "
            "development and confirmation seed. The confirmation subcohort "
            "remains disjoint from ranking; pooled full-panel summaries are "
            "reported separately."
        ),
    )
    causal_plan.set_defaults(func=command_causal_plan)

    answer_query_plan = subparsers.add_parser("answer-query-causal-plan")
    _add_config(answer_query_plan)
    answer_query_plan.add_argument("--attention", type=Path, required=True)
    answer_query_plan.add_argument("--output", type=Path, required=True)
    answer_query_plan.set_defaults(func=command_answer_query_causal_plan)

    pre_city_causal_plan = subparsers.add_parser("pre-city-causal-plan")
    _add_config(pre_city_causal_plan)
    pre_city_causal_plan.add_argument("--attention", type=Path, required=True)
    pre_city_causal_plan.add_argument("--output", type=Path, required=True)
    pre_city_causal_plan.set_defaults(func=command_pre_city_causal_plan)

    response_reference_plan = subparsers.add_parser(
        "response-reference-causal-plan"
    )
    _add_config(response_reference_plan)
    response_reference_plan.add_argument("--attention", type=Path, required=True)
    response_reference_plan.add_argument("--output", type=Path, required=True)
    response_reference_plan.set_defaults(
        func=command_response_reference_causal_plan
    )
    source_writes = subparsers.add_parser("causal-source-writes")
    _add_config(source_writes)
    _add_model(source_writes)
    source_writes.add_argument("--generations", type=Path, required=True)
    source_writes.add_argument("--output", type=Path, required=True)
    source_writes.add_argument("--layers", type=int, nargs="+")
    source_writes.add_argument(
        "--counts",
        type=int,
        nargs="+",
        help="Restrict source-write capture to these registered gold counts.",
    )
    source_writes.add_argument(
        "--final-transition-only",
        action="store_true",
        help="Capture only the registered N-1 to N transition for each row.",
    )
    source_writes.add_argument(
        "--anchor-role",
        help=(
            "Capture one eligible anchor per seed for this expanded parser "
            "role before any repeat."
        ),
    )
    source_writes.add_argument(
        "--target-grammar-class",
        help=(
            "Capture only transitions whose target event has this grammar "
            "class."
        ),
    )
    source_writes.add_argument(
        "--target-retrieval-surface-variant",
        help=(
            "Capture only one parser-audited target retrieval surface "
            "subtype, such as rank_before_city_compact."
        ),
    )
    source_writes.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include locally eligible non-primary grammar cohorts.",
    )
    source_writes.add_argument(
        "--include-block-pre",
        action="store_true",
        help="Include the exploratory non-event-specific block-entry anchor.",
    )
    source_writes.add_argument(
        "--limit",
        type=int,
        help="Deterministic maximum number of anchor captures for a smoke run.",
    )
    source_writes.add_argument(
        "--no-resume", dest="resume", action="store_false"
    )
    source_writes.set_defaults(func=command_causal_source_writes, resume=True)

    causal_heads = subparsers.add_parser("causal-heads")
    _add_config(causal_heads)
    _add_model(causal_heads)
    causal_heads.add_argument("--generations", type=Path, required=True)
    causal_heads.add_argument("--plan", type=Path, required=True)
    causal_heads.add_argument("--output", type=Path, required=True)
    causal_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_heads.add_argument(
        "--head-ablation-scope",
        choices=(
            "registered_query_only",
            "last_pre_city_only",
            "registered_query_through_last_pre_city",
            "city_predictor_chain",
            "registered_query_through_city_prefix",
        ),
        default="registered_query_only",
        help=(
            "Teacher-forced predictor positions at which to lesion the frozen "
            "head bank while scoring the unchanged next-city token span."
        ),
    )
    causal_heads.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include locally eligible non-primary grammar cohorts.",
    )
    causal_heads.add_argument(
        "--include-block-pre",
        action="store_true",
        help="Include the exploratory non-event-specific block-entry anchor.",
    )
    causal_heads.add_argument(
        "--limit",
        type=int,
        help="Deterministic maximum number of distinct anchors for a smoke run.",
    )
    causal_heads.add_argument(
        "--no-resume", dest="resume", action="store_false"
    )
    causal_heads.set_defaults(func=command_causal_heads, resume=True)

    source_edge = subparsers.add_parser("causal-source-edge")
    _add_config(source_edge)
    _add_model(source_edge)
    source_edge.add_argument("--generations", type=Path, required=True)
    source_edge.add_argument("--plan", type=Path, required=True)
    source_edge.add_argument("--output", type=Path, required=True)
    source_edge.add_argument(
        "--anchor-role",
        default="city_pre_d1",
        help="Registered expanded anchor role; city_pre_d1 is the primary estimand.",
    )
    source_edge.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include locally eligible non-primary grammar cohorts.",
    )
    source_edge.add_argument(
        "--include-block-pre",
        action="store_true",
        help="Include exploratory non-event-specific anchors (normally inapplicable).",
    )
    source_edge.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum anchors; selection takes one per seed before any repeat.",
    )
    source_edge.add_argument(
        "--no-resume", dest="resume", action="store_false"
    )
    source_edge.set_defaults(func=command_causal_source_edge, resume=True)

    heads_behavior = subparsers.add_parser("causal-heads-behavior")
    _add_config(heads_behavior)
    _add_model(heads_behavior)
    heads_behavior.add_argument("--generations", type=Path, required=True)
    heads_behavior.add_argument("--plan", type=Path, required=True)
    heads_behavior.add_argument("--output", type=Path, required=True)
    heads_behavior.add_argument(
        "--anchor-role",
        default="city_pre_d1",
        help=(
            "Registered expanded anchor role. The behavioral primary "
            "estimand branches at city_pre_d1."
        ),
    )
    heads_behavior.add_argument(
        "--anchor-routing",
        type=Path,
        help=(
            "Frozen JSON policy mapping target grammar classes to required "
            "and optional semantic anchor roles. When supplied, the same "
            "head bank is ablated at every resolved routed site in one "
            "prefill, and --anchor-role is not used for selection."
        ),
    )
    heads_behavior.add_argument(
        "--target-retrieval-surface-variant",
        help=(
            "Restrict behavior trials to one parser-audited target surface. "
            "For an ordinary run it must match the plan; for an explicit "
            "cross-grammar transfer it describes the behavior-side surface."
        ),
    )
    heads_behavior.add_argument(
        "--behavior-target-grammar-class",
        help=(
            "Apply the frozen bank to this behavior-side grammar instead of "
            "the grammar on which the bank was selected. This requires the "
            "explicit cross-grammar transfer flag."
        ),
    )
    heads_behavior.add_argument(
        "--behavior-all-routed-grammars",
        action="store_true",
        help=(
            "Apply the frozen selection bank to every grammar declared by "
            "--anchor-routing in one shared full-panel run."
        ),
    )
    heads_behavior.add_argument(
        "--allow-cross-grammar-bank-transfer",
        action="store_true",
        help=(
            "Development-only audit that preserves the plan's selection "
            "provenance while applying its frozen heads at another grammar's "
            "native query site."
        ),
    )
    heads_behavior.add_argument(
        "--allow-selection-scope-bank-transfer",
        action="store_true",
        help=(
            "Development-only audit that applies a frozen bank at another "
            "grammar, surface subtype, or exact semantic anchor while "
            "retaining the original selection provenance."
        ),
    )
    heads_behavior.add_argument(
        "--evaluation-split",
        choices=["discovery", "confirmation", "all"],
        default="all",
        help=(
            "Restrict behavior trials by the frozen dataset split label. "
            "This is an execution filter only and does not by itself make "
            "a previously inspected split causally held out."
        ),
    )
    heads_behavior.add_argument(
        "--counts",
        type=int,
        nargs="+",
        help=(
            "Optional registered count subset. This filter is hash-locked in "
            "the causal run manifest."
        ),
    )
    heads_behavior.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "clean",
            "selected_bank",
            "layer_matched_random",
            "global_random",
        ],
        default=["clean", "selected_bank", "layer_matched_random"],
        help=(
            "Behavior arms to run. Use clean selected_bank for an efficient "
            "site-localization screen, then add matched random controls."
        ),
    )
    heads_behavior.add_argument(
        "--reference-results",
        type=Path,
        help=(
            "Optional prior behavioral output whose matching transitions "
            "define the site-sweep cohort."
        ),
    )
    heads_behavior.add_argument(
        "--reference-condition",
        default="selected_bank",
        help="Condition required in --reference-results.",
    )
    heads_behavior.add_argument(
        "--reference-behavior-outcome",
        choices=[
            "correct_next_needle",
            "wrong_gold_needle",
            "wrong_non_gold_city_record",
            "no_identifiable_city_record",
            "no_identifiable_gold_needle",
        ],
        default="correct_next_needle",
        help="Behavior outcome required in --reference-results.",
    )
    heads_behavior.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include locally eligible non-primary grammar cohorts.",
    )
    heads_behavior.add_argument(
        "--include-block-pre",
        action="store_true",
        help="Include exploratory non-event-specific anchors (normally inapplicable).",
    )
    heads_behavior.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum anchors; selection takes one per seed before any repeat.",
    )
    heads_behavior.add_argument(
        "--anchor-sampling",
        choices=["seed_first", "prompt_balanced", "prompt_final_transition"],
        default="seed_first",
        help=(
            "prompt_balanced freezes at most one transition per prompt and "
            "greedily balances grammar/surface/transition coverage; "
            "prompt_final_transition chooses the registered N-1 -> N "
            "transition for each prompt."
        ),
    )
    heads_behavior.add_argument(
        "--freeze-anchor-registry-only",
        action="store_true",
        help=(
            "Resolve and hash the deterministic anchor registry, then stop "
            "before any clean or ablated behavioral forward."
        ),
    )
    heads_behavior.add_argument(
        "--anchor-registry-input",
        type=Path,
        help=(
            "Reuse an already frozen full-panel anchor registry verbatim. "
            "The evaluation split/counts select scheduled trials without "
            "rerunning prompt-balanced anchor choice."
        ),
    )
    heads_behavior.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum greedy continuation length after the registered branch.",
    )
    heads_behavior.add_argument(
        "--decode-head-ablation-steps",
        type=int,
        default=0,
        help=(
            "Also ablate the same head bank during the first N one-token "
            "cached decode forwards after prefill. N=0 preserves the "
            "registered prefill-only estimand; N>0 tests a fixed retry "
            "window; N=-1 keeps the bank off for every decode forward."
        ),
    )
    heads_behavior.add_argument(
        "--no-resume", dest="resume", action="store_false"
    )
    heads_behavior.set_defaults(
        func=command_causal_heads_behavior,
        resume=True,
    )
    causal_heads.add_argument("--include-discovery", action="store_true")
    causal_heads.add_argument(
        "--boundary-policy",
        choices=["strict_registered", "item_end_fallback_v2"],
        default="strict_registered",
        help=(
            "Registered target-boundary policy. item_end_fallback_v2 preserves "
            "the primary boundary and deterministically uses item_end only when "
            "the primary boundary is not literal-token aligned."
        ),
    )
    causal_heads.set_defaults(func=command_causal_heads)

    causal_pre_city_heads = subparsers.add_parser("causal-pre-city-heads")
    _add_config(causal_pre_city_heads)
    _add_model(causal_pre_city_heads)
    causal_pre_city_heads.add_argument("--generations", type=Path, required=True)
    causal_pre_city_heads.add_argument("--plan", type=Path, required=True)
    causal_pre_city_heads.add_argument("--output", type=Path, required=True)
    causal_pre_city_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_pre_city_heads.add_argument(
        "--query-variant",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        required=True,
    )
    causal_pre_city_heads.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_pre_city_heads.add_argument("--include-discovery", action="store_true")
    causal_pre_city_heads.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_pre_city_heads.add_argument("--gold-count", type=int)
    causal_pre_city_heads.add_argument("--max-rows", type=int)
    causal_pre_city_heads.add_argument("--allow-unregistered", action="store_true")
    causal_pre_city_heads.add_argument("--overwrite", action="store_true")
    causal_pre_city_heads.set_defaults(func=command_causal_pre_city_heads)

    causal_pre_city_all_sites = subparsers.add_parser(
        "causal-pre-city-all-sites"
    )
    _add_config(causal_pre_city_all_sites)
    _add_model(causal_pre_city_all_sites)
    causal_pre_city_all_sites.add_argument(
        "--generations", type=Path, required=True
    )
    causal_pre_city_all_sites.add_argument("--plan", type=Path, required=True)
    causal_pre_city_all_sites.add_argument(
        "--output", type=Path, required=True
    )
    causal_pre_city_all_sites.add_argument("--plan-rows", type=int, nargs="+")
    causal_pre_city_all_sites.add_argument(
        "--query-variant",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        required=True,
    )
    causal_pre_city_all_sites.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_pre_city_all_sites.add_argument(
        "--include-discovery", action="store_true"
    )
    causal_pre_city_all_sites.add_argument("--max-rows", type=int)
    causal_pre_city_all_sites.add_argument(
        "--gold-count",
        type=int,
        choices=range(1, 11),
        help="Optional N filter used for exact all-site smoke/audit runs.",
    )
    causal_pre_city_all_sites.add_argument("--overwrite", action="store_true")
    causal_pre_city_all_sites.add_argument(
        "--worker-id",
        default="primary",
        help="Audit label for a cooperative shared-shard worker.",
    )
    causal_pre_city_all_sites.add_argument(
        "--worker-poll-seconds", type=float, default=5.0
    )
    causal_pre_city_all_sites.add_argument(
        "--worker-wait-timeout", type=float, default=86400.0
    )
    causal_pre_city_all_sites.add_argument(
        "--claim-stale-seconds", type=float, default=21600.0
    )
    causal_pre_city_all_sites.add_argument(
        "--allow-unregistered",
        action="store_true",
        help=(
            "Allow isolated supplement seeds outside the primary registry; "
            "outputs remain in a separate caller-provided path."
        ),
    )
    causal_pre_city_all_sites.set_defaults(
        func=command_causal_pre_city_all_sites
    )

    marker_needle_patch = subparsers.add_parser(
        "causal-marker-needle-patch"
    )
    _add_model(marker_needle_patch)
    marker_needle_patch.add_argument(
        "--generations", type=Path, nargs="+", required=True
    )
    marker_needle_patch.add_argument("--pairs", type=Path, required=True)
    marker_needle_patch.add_argument("--output", type=Path, required=True)
    marker_needle_patch.add_argument(
        "--query-variants",
        nargs="+",
        choices=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
        default=["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
    )
    marker_needle_patch.add_argument("--layers", type=int, nargs="+", required=True)
    marker_needle_patch.add_argument(
        "--split", choices=["all", "discovery", "confirmation"], default="all"
    )
    marker_needle_patch.add_argument("--max-pairs", type=int)
    marker_needle_patch.add_argument("--overwrite", action="store_true")
    marker_needle_patch.set_defaults(func=command_causal_marker_needle_patch)

    causal_answer_query_heads = subparsers.add_parser(
        "causal-answer-query-heads"
    )
    _add_model(causal_answer_query_heads)
    causal_answer_query_heads.add_argument(
        "--generations", type=Path, required=True
    )
    causal_answer_query_heads.add_argument("--plan", type=Path, required=True)
    causal_answer_query_heads.add_argument(
        "--output", type=Path, required=True
    )
    causal_answer_query_heads.add_argument(
        "--site-id", default="answer_query_v3"
    )
    causal_answer_query_heads.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    causal_answer_query_heads.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_answer_query_heads.add_argument("--plan-rows", type=int, nargs="+")
    causal_answer_query_heads.add_argument("--max-rows", type=int)
    causal_answer_query_heads.add_argument("--overwrite", action="store_true")
    causal_answer_query_heads.set_defaults(
        func=command_causal_answer_query_heads
    )

    causal_response_reference = subparsers.add_parser(
        "causal-response-reference-heads"
    )
    _add_model(causal_response_reference)
    causal_response_reference.add_argument(
        "--generations", type=Path, required=True
    )
    causal_response_reference.add_argument("--plan", type=Path, required=True)
    causal_response_reference.add_argument("--output", type=Path, required=True)
    causal_response_reference.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one_correct",
    )
    causal_response_reference.add_argument(
        "--split",
        choices=["all", "discovery", "confirmation"],
        default="confirmation",
    )
    causal_response_reference.add_argument(
        "--response-types", nargs="+", choices=list(("bare_or_list", "record_template", "semantic_cue"))
    )
    causal_response_reference.add_argument(
        "--position-variants",
        nargs="+",
        choices=[
            "pre_reference_d1",
            "pre_city_d1",
            "pre_city_d2",
            "pre_city_anchor",
        ],
    )
    causal_response_reference.add_argument(
        "--bank-scopes",
        nargs="+",
        choices=[
            "position_consensus",
            "unified_consensus",
            "response_type_and_position_specific",
        ],
    )
    causal_response_reference.add_argument("--plan-rows", type=int, nargs="+")
    causal_response_reference.add_argument("--max-rows", type=int)
    causal_response_reference.add_argument("--overwrite", action="store_true")
    causal_response_reference.set_defaults(
        func=command_causal_response_reference_heads
    )

    causal_tokens = subparsers.add_parser("causal-tokens")
    _add_config(causal_tokens)
    _add_model(causal_tokens)
    causal_tokens.add_argument("--generations", type=Path, required=True)
    causal_tokens.add_argument("--output", type=Path, required=True)
    causal_tokens.add_argument("--max-new-tokens", type=int, default=16)
    causal_tokens.add_argument("--include-discovery", action="store_true")
    causal_tokens.set_defaults(func=command_causal_tokens)

    causal_context = subparsers.add_parser("causal-context")
    _add_config(causal_context)
    _add_model(causal_context)
    _add_split(causal_context)
    causal_context.add_argument("--generations", type=Path, required=True)
    causal_context.add_argument("--output", type=Path, required=True)
    site_group = causal_context.add_mutually_exclusive_group(required=True)
    site_group.add_argument("--site-kind")
    site_group.add_argument("--site-id")
    causal_context.add_argument(
        "--conditions",
        nargs="+",
        choices=["clean", "trace_only", "matched_nontrace_only"],
        default=["clean", "trace_only", "matched_nontrace_only"],
    )
    causal_context.add_argument("--layers", type=int, nargs="+")
    causal_context.set_defaults(func=command_causal_context)

    causal_writes = subparsers.add_parser("causal-writes")
    _add_config(causal_writes)
    _add_model(causal_writes)
    _add_split(causal_writes, default="all")
    causal_writes.add_argument("--generations", type=Path, required=True)
    causal_writes.add_argument("--output", type=Path, required=True)
    causal_writes.add_argument("--site-id", default="answer_query")
    causal_writes.add_argument("--layers", type=int, nargs="+")
    causal_writes.add_argument("--directions", type=Path)
    causal_writes.add_argument("--save-vectors", action="store_true")
    causal_writes.set_defaults(func=command_causal_writes)

    subspace_fit = subparsers.add_parser("causal-subspace-fit")
    _add_config(subspace_fit)
    subspace_fit.add_argument("--capture-index", type=Path, required=True)
    subspace_fit.add_argument("--output", type=Path, required=True)
    subspace_fit.add_argument("--site-kind", required=True)
    subspace_fit.add_argument("--site-id")
    subspace_fit.add_argument(
        "--cohort",
        choices=["parser_hit", "one_to_one", "one_to_one_correct"],
        default="one_to_one",
    )
    subspace_fit.add_argument("--layers", type=int, nargs="+")
    subspace_fit.add_argument("--rank", type=int, default=3)
    subspace_fit.set_defaults(func=command_causal_subspace_fit)

    causal_subspace = subparsers.add_parser("causal-subspace")
    _add_config(causal_subspace)
    _add_model(causal_subspace)
    _add_split(causal_subspace)
    causal_subspace.add_argument("--generations", type=Path, required=True)
    causal_subspace.add_argument("--basis", type=Path, required=True)
    causal_subspace.add_argument("--output", type=Path, required=True)
    subspace_site = causal_subspace.add_mutually_exclusive_group(required=True)
    subspace_site.add_argument("--site-kind")
    subspace_site.add_argument("--site-id")
    causal_subspace.add_argument("--layers", type=int, nargs="+")
    causal_subspace.add_argument("--doses", type=float, nargs="+", default=[1.0])
    causal_subspace.add_argument("--max-new-tokens", type=int, default=16)
    causal_subspace.set_defaults(func=command_causal_subspace)

    causal_patch = subparsers.add_parser("causal-patch")
    _add_config(causal_patch)
    _add_model(causal_patch)
    causal_patch.add_argument("--generations", type=Path, required=True)
    causal_patch.add_argument("--pairs", type=Path, required=True)
    causal_patch.add_argument("--basis", type=Path)
    causal_patch.add_argument("--output", type=Path, required=True)
    causal_patch.add_argument("--layer", type=int)
    causal_patch.add_argument("--layers", type=int, nargs="+")
    causal_patch.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "self_patch",
            "full_donor_patch",
            "projected_donor_patch",
            "orthogonal_norm_matched",
        ],
        default=[
            "self_patch",
            "full_donor_patch",
            "projected_donor_patch",
            "orthogonal_norm_matched",
        ],
    )
    causal_patch.add_argument("--receiver-site-id", default="answer_query")
    causal_patch.add_argument("--donor-site-id", default="answer_query")
    causal_patch.add_argument("--max-new-tokens", type=int, default=16)
    causal_patch.add_argument("--restartable", action="store_true")
    causal_patch.add_argument("--overwrite", action="store_true")
    causal_patch.set_defaults(func=command_causal_patch)

    causal_analyze = subparsers.add_parser("causal-analyze")
    _add_config(causal_analyze)
    causal_analyze.add_argument("--trials", type=Path, required=True)
    causal_analyze.add_argument("--output", type=Path, required=True)
    causal_analyze.add_argument("--treatment", required=True)
    causal_analyze.add_argument("--control", required=True)
    causal_analyze.add_argument(
        "--outcome", default="target_city_log_probability"
    )
    causal_analyze.add_argument(
        "--mechanism",
        choices=["retrieval_anchor_localization", "progress_transition"],
    )
    causal_analyze.add_argument("--bank-size", type=int)
    causal_analyze.add_argument("--anchor-role")
    causal_analyze.add_argument(
        "--transition-phase", choices=["retrieve", "continue", "stop"]
    )
    causal_analyze.set_defaults(func=command_causal_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
