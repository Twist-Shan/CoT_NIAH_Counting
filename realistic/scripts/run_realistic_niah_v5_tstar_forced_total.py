#!/usr/bin/env python3
"""Force an immediate Total readout from frozen first-pass t-star contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.tstar_prefix import sha256_json  # noqa: E402


SCHEMA = "realistic_niah_v5_tstar_forced_immediate_total_v1"
FORCED_SUFFIX = "\n</think>\n\nTotal: "
ROWS_FILENAME = "forced_immediate_total_rows_v1.jsonl"
CSV_FILENAME = "forced_immediate_total_summary_v1.csv"
PLAN_FILENAME = "frozen_forced_immediate_total_plan_v1.json"
MANIFEST_FILENAME = "forced_immediate_total_manifest_v1.json"
_TOTAL_RE = re.compile(r"(?im)^\s*Total\s*:\s*(-?\d+)\s*$")
_INTEGER_ONLY_RE = re.compile(r"\s*(-?\d+)\s*\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if immutable and path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Frozen file changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def atomic_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, immutable: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    if immutable and path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Frozen file changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Frozen file changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def extract_total(text: str) -> int | None:
    matches = list(_TOTAL_RE.finditer(str(text)))
    return int(matches[-1].group(1)) if matches else None


def extract_immediate_integer(text: str) -> int | None:
    match = _INTEGER_ONLY_RE.fullmatch(str(text))
    return int(match.group(1)) if match else None


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        return [int(item) for item in value]
    return [int(value)]


def _indexed(rows: list[dict[str, Any]], *, label: str) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in output:
            raise ValueError(f"Duplicate {label} seed={seed}")
        output[seed] = row
    return output


def _validate_sources(
    *,
    selected_path: Path,
    selected_manifest: Mapping[str, Any],
    context_path: Path,
    context_manifest: Mapping[str, Any],
) -> tuple[list[int], int]:
    expected_selected = dict(selected_manifest.get("files") or {}).get(
        selected_path.name
    )
    if expected_selected != sha256_file(selected_path):
        raise ValueError("Selected rows hash does not match source manifest")
    expected_context = dict(context_manifest.get("files") or {}).get(
        context_path.name
    )
    if expected_context != sha256_file(context_path):
        raise ValueError("t-star contexts hash does not match context manifest")
    if selected_manifest.get("status") != "FROZEN":
        raise ValueError("Selected cohort manifest is not frozen")
    if context_manifest.get("status") != "FROZEN":
        raise ValueError("t-star context manifest is not frozen")
    fixed_count = int(selected_manifest["fixed_count"])
    if int(context_manifest["fixed_count"]) != fixed_count:
        raise ValueError("Source and context manifests disagree on fixed_count")
    seeds = [int(value) for value in selected_manifest.get("discovery_seeds", ())]
    seeds += [int(value) for value in selected_manifest.get("confirmation_seeds", ())]
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("Forced-total replay requires one frozen 30-seed cohort")
    if seeds != [
        *[int(value) for value in context_manifest.get("discovery_seeds", ())],
        *[int(value) for value in context_manifest.get("confirmation_seeds", ())],
    ]:
        raise ValueError("Source and context manifests disagree on seed order")
    return seeds, fixed_count


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "forced_total_parsed_n": sum(row.get("forced_total") is not None for row in rows),
        "forced_total_gold_correct_n": sum(
            bool(row.get("forced_total_gold_correct")) for row in rows
        ),
        "forced_total_matches_source_n": sum(
            bool(row.get("forced_total_matches_source")) for row in rows
        ),
        "immediate_integer_only_n": sum(
            bool(row.get("immediate_integer_only")) for row in rows
        ),
        "stopped_on_eos_n": sum(bool(row.get("stopped_on_eos")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-rows", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if int(args.max_new_tokens) < 2:
        raise ValueError("--max-new-tokens must be at least 2")

    source_manifest = read_json(args.source_manifest)
    context_manifest = read_json(args.context_manifest)
    seeds, fixed_count = _validate_sources(
        selected_path=args.selected_rows,
        selected_manifest=source_manifest,
        context_path=args.contexts,
        context_manifest=context_manifest,
    )
    source_by_seed = _indexed(read_jsonl(args.selected_rows), label="source")
    context_by_seed = _indexed(read_jsonl(args.contexts), label="context")
    if set(source_by_seed) != set(seeds) or set(context_by_seed) != set(seeds):
        raise ValueError("Replay inputs do not contain the frozen seed set")

    spec = resolve_model_spec(str(args.model))
    plan = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_FORCED_OUTPUTS",
        "model_label": str(args.model),
        "model_id": str(spec.model_id),
        "model_revision": spec.revision,
        "fixed_count": fixed_count,
        "seeds": seeds,
        "discovery_seeds": [int(value) for value in source_manifest["discovery_seeds"]],
        "confirmation_seeds": [
            int(value) for value in source_manifest["confirmation_seeds"]
        ],
        "forced_suffix": FORCED_SUFFIX,
        "forced_suffix_utf8_hex": FORCED_SUFFIX.encode("utf-8").hex(),
        "decoding": {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        },
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attention_backend": str(args.attention_backend),
        "source_selected_rows": str(args.selected_rows.resolve()),
        "source_selected_rows_sha256": sha256_file(args.selected_rows),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "tstar_contexts": str(args.contexts.resolve()),
        "tstar_contexts_sha256": sha256_file(args.contexts),
        "tstar_context_manifest": str(args.context_manifest.resolve()),
        "tstar_context_manifest_sha256": sha256_file(args.context_manifest),
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "interpretation": (
            "standardized forced-stop readout; not an unmodified natural continuation"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / PLAN_FILENAME
    atomic_json(plan_path, plan)
    plan_hash = sha256_file(plan_path)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        seed
        for seed in seeds
        if not (args.resume and (shard_dir / f"seed{seed}.json").is_file())
    ]
    model = tokenizer = None
    if pending:
        model, tokenizer, _adapter = load_registered_model(
            spec,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
        )
    else:
        from realistic_niah_v4.modeling import load_registered_tokenizer

        tokenizer = load_registered_tokenizer(spec, cache_dir=args.cache_dir)

    closure_ids = tuple(
        int(value) for value in tokenizer.encode(FORCED_SUFFIX, add_special_tokens=False)
    )
    decoded_closure = tokenizer.decode(
        list(closure_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_closure != FORCED_SUFFIX:
        raise ValueError("Forced suffix does not round-trip through the tokenizer")

    if pending:
        import torch

        device = model.get_input_embeddings().weight.device
        eos_ids = _eos_ids(model, tokenizer)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = eos_ids[0]
        for ordinal, seed in enumerate(seeds, start=1):
            shard_path = shard_dir / f"seed{seed}.json"
            if args.resume and shard_path.is_file():
                existing = read_json(shard_path)
                if existing.get("plan_sha256") != plan_hash:
                    raise ValueError(f"Existing shard seed={seed} uses another plan")
                print(f"[forced-total] {ordinal}/30 seed={seed} resumed", flush=True)
                continue
            source = source_by_seed[seed]
            context = context_by_seed[seed]
            if sha256_json(source) != context.get("source_row_sha256"):
                raise ValueError(f"Source row hash mismatch for seed={seed}")
            output_end = int(context["output_token_end"])
            expected_context_ids = [
                *[int(value) for value in source["input_ids"]],
                *[int(value) for value in source["output_token_ids"][:output_end]],
            ]
            if expected_context_ids != [int(value) for value in context["input_ids"]]:
                raise ValueError(f"t-star input is not source-exact for seed={seed}")
            forced_input_ids = tuple(expected_context_ids) + closure_ids
            forced_mask = tuple(int(value) for value in context["attention_mask"]) + (
                1,
            ) * len(closure_ids)
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            input_tensor = torch.tensor(
                [forced_input_ids], dtype=torch.long, device=device
            )
            mask_tensor = torch.tensor([forced_mask], dtype=torch.long, device=device)
            kwargs: dict[str, Any] = {
                "input_ids": input_tensor,
                "attention_mask": mask_tensor,
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": False,
                "use_cache": True,
            }
            if pad_token_id is not None:
                kwargs["pad_token_id"] = int(pad_token_id)
            with torch.inference_mode():
                generated = model.generate(**kwargs)
            sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
            continuation_ids = tuple(
                int(value)
                for value in sequences[0, len(forced_input_ids) :]
                .detach()
                .cpu()
                .tolist()
            )
            continuation_raw = tokenizer.decode(
                list(continuation_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            continuation_clean = tokenizer.decode(
                list(continuation_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            forced_total = extract_immediate_integer(continuation_clean)
            source_total = extract_total(str(source.get("clean_output_text", "")))
            gold_total = int(source["gold_count"])
            row = {
                "schema_version": SCHEMA,
                "status": "PASS",
                "plan_sha256": plan_hash,
                "seed": int(seed),
                "split": str(context["split"]),
                "fixed_count": fixed_count,
                "request_id": str(source.get("request_id", "")),
                "source_row_sha256": str(context["source_row_sha256"]),
                "tstar_context_row_sha256": sha256_json(context),
                "t_star_char": int(context["t_star_char"]),
                "output_token_end": output_end,
                "raw_tstar_prefix_text": str(context["raw_prefix_text"]),
                "forced_suffix": FORCED_SUFFIX,
                "forced_suffix_token_ids": list(closure_ids),
                "forced_input_ids": list(forced_input_ids),
                "forced_attention_mask": list(forced_mask),
                "forced_input_ids_sha256": sha256_json(forced_input_ids),
                "forced_continuation_token_ids": list(continuation_ids),
                "forced_continuation_text_raw": continuation_raw,
                "forced_continuation_text_clean": continuation_clean,
                "forced_full_output_text_raw": (
                    str(context["raw_prefix_text"]) + FORCED_SUFFIX + continuation_raw
                ),
                "forced_full_output_text_clean": (
                    str(context["raw_prefix_text"]) + FORCED_SUFFIX + continuation_clean
                ),
                "forced_total": forced_total,
                "source_total": source_total,
                "gold_total": gold_total,
                "forced_total_gold_correct": forced_total == gold_total,
                "forced_total_matches_source": (
                    forced_total is not None and forced_total == source_total
                ),
                "immediate_integer_only": forced_total is not None,
                "stopped_on_eos": bool(
                    continuation_ids and continuation_ids[-1] in set(eos_ids)
                ),
                "max_new_tokens": int(args.max_new_tokens),
                "selection_used_final_answer": False,
                "mechanism_outcomes_accessed": False,
                "interpretation": (
                    "standardized forced-stop readout; not natural continuation"
                ),
            }
            atomic_json(shard_path, row)
            print(
                f"[forced-total] {ordinal}/30 seed={seed} split={context['split']} "
                f"forced={forced_total} source={source_total} gold={gold_total} "
                f"eos={int(row['stopped_on_eos'])}",
                flush=True,
            )

    rows = [read_json(shard_dir / f"seed{seed}.json") for seed in seeds]
    if any(row.get("plan_sha256") != plan_hash for row in rows):
        raise ValueError("One or more shards do not match the frozen plan")
    rows_path = args.output_dir / ROWS_FILENAME
    atomic_jsonl(rows_path, rows)
    csv_rows = [
        {
            "seed": int(row["seed"]),
            "split": str(row["split"]),
            "gold_total": int(row["gold_total"]),
            "source_total": row["source_total"],
            "forced_total": row["forced_total"],
            "forced_total_gold_correct": bool(row["forced_total_gold_correct"]),
            "forced_total_matches_source": bool(row["forced_total_matches_source"]),
            "immediate_integer_only": bool(row["immediate_integer_only"]),
            "stopped_on_eos": bool(row["stopped_on_eos"]),
            "forced_continuation_text_clean": str(
                row["forced_continuation_text_clean"]
            ),
        }
        for row in rows
    ]
    csv_path = args.output_dir / CSV_FILENAME
    atomic_csv(csv_path, csv_rows)
    summary = {
        "all": summarize_rows(rows),
        "discovery": summarize_rows(
            [row for row in rows if row["split"] == "discovery"]
        ),
        "confirmation": summarize_rows(
            [row for row in rows if row["split"] == "confirmation"]
        ),
    }
    manifest = {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "model_label": str(args.model),
        "fixed_count": fixed_count,
        "forced_suffix": FORCED_SUFFIX,
        "plan_sha256": plan_hash,
        "summary": summary,
        "files": {
            PLAN_FILENAME: plan_hash,
            ROWS_FILENAME: sha256_file(rows_path),
            CSV_FILENAME: sha256_file(csv_path),
        },
        "shard_count": len(rows),
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "interpretation": (
            "standardized forced-stop readout; not an unmodified natural continuation"
        ),
    }
    atomic_json(args.output_dir / MANIFEST_FILENAME, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
