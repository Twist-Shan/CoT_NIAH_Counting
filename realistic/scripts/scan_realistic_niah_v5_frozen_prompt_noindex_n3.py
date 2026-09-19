#!/usr/bin/env python3
"""Scan fixed-N=3 first-pass no-index traces in causal split order.

The scan reuses deterministic frozen-prompt generations when available, then
generates missing seeds in ascending order.  Eligibility is format-only.  The
first 20 eligible discovery seeds and first 10 eligible confirmation seeds are
sealed; final-answer correctness and all mechanism outcomes are inaccessible
to the selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset_generation.dynamic_niah import TokenizerAdapter  # noqa: E402
from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v4.stimuli import (  # noqa: E402
    ControlledFreezeSpec,
    build_controlled_family,
)
from realistic_niah_v5.generation import (  # noqa: E402
    V5_USER_TEMPLATE,
    build_v5_user_text,
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_first_occurrence_prefix_clean,
    audit_no_count_enumeration_trace,
)


SCHEMA = "realistic_niah_v5_frozen_prompt_noindex_n3_scan_v4"
MODEL_LABELS = ("Qwen3-8B", "Gemma4-E4B")
SEED_START = 1234
FIXED_COUNT = 3
AUDIT_KEY = "noindex_n3_format_audit"
COHORT_KEY = "noindex_n3_cohort"
LOG_LABEL = "noindex-n3-scan"
PROSPECTIVE_ORIGIN = "prospective_fixed_n3_scan"
ORIGIN_KEY = "noindex_n3_generation_origin"
DISCOVERY_QUOTA = 20
CONFIRMATION_QUOTA = 10
OUTPUT_SUFFIX = "first_pass_noindex_v5"


def assigned_split(seed: int) -> str:
    """Match the existing V5 causal and one-to-one supplement split policy."""

    if seed < SEED_START:
        raise ValueError("Formal scan begins at seed 1234")
    if seed <= 1253:
        return "discovery"
    if seed <= 1263:
        return "confirmation"
    if seed == 1264:
        return "discovery"
    if seed == 1265:
        return "confirmation"
    return "confirmation" if (seed - 1266) % 3 == 2 else "discovery"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
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
    path: Path, rows: Iterable[Mapping[str, Any]], *, immutable: bool = False
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


def load_existing_rows(
    paths: Sequence[Path], *, model: str
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            files = sorted((path / "shards").glob("seed*_N*.json"))
            if not files:
                raise FileNotFoundError(f"No scan shards under {path}")
            source_kind = "scan_shard_directory"
        elif path.is_file():
            files = [path]
            source_kind = "jsonl"
        else:
            raise FileNotFoundError(path)
        used = 0
        source_hashes: dict[str, str] = {}
        candidate_rows: list[dict[str, Any]] = []
        for file in files:
            source_hashes[file.name] = sha256_file(file)
            if source_kind == "scan_shard_directory":
                value = json.loads(file.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"{file} is not a JSON object")
                candidate_rows.append(value)
            else:
                candidate_rows.extend(read_jsonl(file))
        for row in candidate_rows:
            if (
                str(row.get("model_label")) != model
                or int(row.get("gold_count", -1)) != FIXED_COUNT
                or int(row.get("seed", -1)) < SEED_START
            ):
                continue
            seed = int(row["seed"])
            previous = by_seed.get(seed)
            if previous is not None:
                old_identity = (
                    str(previous.get("request_id")),
                    sha256_json(previous.get("input_ids", ())),
                    sha256_json(previous.get("output_token_ids", ())),
                )
                new_identity = (
                    str(row.get("request_id")),
                    sha256_json(row.get("input_ids", ())),
                    sha256_json(row.get("output_token_ids", ())),
                )
                if old_identity != new_identity:
                    raise ValueError(f"Conflicting deterministic row for seed={seed}")
                continue
            by_seed[seed] = dict(row)
            used += 1
        sources.append(
            {
                "path": str(path.resolve()),
                "source_kind": source_kind,
                "source_file_count": len(files),
                "source_files_sha256": source_hashes,
                "inventory_sha256": sha256_json(source_hashes),
                "reused_n3_rows": used,
            }
        )
    return by_seed, sources


def frozen_prompt_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    actual = str(row.get("user_text", ""))
    if "passage" in row:
        expected = build_v5_user_text(
            str(row["passage"]), entity_domain=str(row.get("entity_domain", "city"))
        )
        reconstruction = "passage_field"
    else:
        prefix, suffix = V5_USER_TEMPLATE.split("{passage}")
        if not actual.startswith(prefix) or not actual.endswith(suffix):
            raise ValueError("PROMPT_INTEGRITY_FAILURE: archived user text lacks V5 envelope")
        passage = actual[len(prefix) : len(actual) - len(suffix)] if suffix else actual[len(prefix) :]
        expected = V5_USER_TEMPLATE.format(passage=passage)
        reconstruction = "canonical_template_envelope"
    rendered = str(row.get("rendered_prompt", ""))
    expected_decoding = {
        "max_new_tokens": 4096,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    if actual != expected:
        raise ValueError("PROMPT_INTEGRITY_FAILURE: user text is not the V5 template")
    if rendered.count(actual) != 1:
        raise ValueError("PROMPT_INTEGRITY_FAILURE: archived rendered prompt changed")
    if str(row.get("prompt_mode")) != "native_thinking":
        raise ValueError("PROMPT_INTEGRITY_FAILURE: prompt mode changed")
    if row.get("decoding") != expected_decoding:
        raise ValueError("PROMPT_INTEGRITY_FAILURE: decoding changed")
    if int(row.get("prompt_token_count", -1)) != len(row.get("input_ids", ())):
        raise ValueError("PROMPT_INTEGRITY_FAILURE: archived input length changed")
    if "NATURAL_UNNUMBERED" in str(row.get("request_id", "")):
        raise ValueError("PROMPT_INTEGRITY_FAILURE: prompt-conditioned row is disallowed")
    return {
        "status": "PASS",
        "user_text_byte_exact": True,
        "extra_user_instruction": False,
        "extra_system_message": False,
        "extra_assistant_prefix": False,
        "reconstruction": reconstruction,
        "user_text_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "input_ids_sha256": sha256_json(row.get("input_ids", ())),
    }


def format_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    prefix = audit_first_occurrence_prefix_clean(row)
    legacy = audit_no_count_enumeration_trace(row)
    parser = row["trace_parse"]["parser"]
    marker = str(parser.get("marker_kind", ""))
    primary = bool(prefix["prefix_clean_eligible"])
    global_clean = bool(prefix["global_clean_eligible"])
    return {
        **prefix,
        "scan_eligibility_version": OUTPUT_SUFFIX,
        "marker_kind": marker,
        "parser_marker_kind": marker,
        "trace_one_to_one": bool(parser.get("trace_one_to_one")),
        "item_count": int(parser.get("item_count", -1)),
        "legacy_registered_trace_audit": legacy,
        "primary_eligible_prefix_clean": primary,
        "global_clean_no_running_index": global_clean,
        "primary_eligible_no_running_index": primary,
        "strict_eligible_no_explicit_count_cue": global_clean,
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
    }


def choose_rows(
    attempted_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = {"discovery": [], "confirmation": []}
    quotas = {"discovery": DISCOVERY_QUOTA, "confirmation": CONFIRMATION_QUOTA}
    ledger: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(attempted_rows, key=lambda value: int(value["seed"])):
        seed = int(row["seed"])
        if seed in seen:
            raise ValueError(f"Duplicate attempted seed={seed}")
        seen.add(seed)
        split = assigned_split(seed)
        audit = row[AUDIT_KEY]
        eligible = bool(audit["primary_eligible_prefix_clean"])
        is_selected = eligible and len(selected[split]) < quotas[split]
        if is_selected:
            chosen = dict(row)
            chosen[COHORT_KEY] = {
                "split": split,
                "rank_within_split": len(selected[split]) + 1,
                "selection_basis": "format_only",
                "selection_population": "first_pass_noindex_enumeration",
                "fixed_count": FIXED_COUNT,
                "selection_used_final_answer": False,
                "mechanism_outcomes_accessed": False,
            }
            selected[split].append(chosen)
        ledger.append(
            {
                "seed": seed,
                "split": split,
                "fixed_count": FIXED_COUNT,
                "marker_kind": audit["marker_kind"],
                "primary_eligible": eligible,
                "prefix_clean_eligible": eligible,
                "global_clean_eligible": bool(
                    audit["global_clean_no_running_index"]
                ),
                "strict_eligible": bool(
                    audit["strict_eligible_no_explicit_count_cue"]
                ),
                "selected": is_selected,
                "reasons": list(audit["reasons"]),
                "selection_used_final_answer": False,
                "mechanism_outcomes_accessed": False,
            }
        )
    return selected["discovery"], selected["confirmation"], ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--existing-generations", type=Path, nargs="+", required=True)
    parser.add_argument("--max-seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if int(args.max_seed) < 1266:
        raise ValueError("max-seed safety bound is too small")

    args.output.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output / f"attempt_ledger_{OUTPUT_SUFFIX}.jsonl"
    existing, sources = load_existing_rows(
        tuple(args.existing_generations), model=str(args.model)
    )
    plan = {
        "schema_version": SCHEMA,
        "status": "FROZEN_BEFORE_MECHANISM_OUTCOMES",
        "model_label": str(args.model),
        "seed_start": SEED_START,
        "fixed_count": FIXED_COUNT,
        "minimum_transitions": FIXED_COUNT - 1,
        "discovery_quota": DISCOVERY_QUOTA,
        "confirmation_quota": CONFIRMATION_QUOTA,
        "base_split": {
            "discovery": [1234, 1253],
            "confirmation": [1254, 1263],
        },
        "supplement_split_policy": (
            "1264 discovery; 1265 confirmation; from 1266 every third seed "
            "with offset 2 is confirmation and the other two are discovery"
        ),
        "seed_selection": "ascending within fixed split; format-only",
        "primary_population": (
            "first complete score-supported gold-record pass contains each gold "
            "record exactly once and has no explicit count/index cue through its "
            "K-th unique occurrence; later rethink or recap is allowed"
        ),
        "primary_endpoint": "t_star=end_of_Kth_first_score_supported_gold_mention",
        "strict_sensitivity": (
            "global reasoning has no per-record index; terminal aggregate total allowed"
        ),
        "prompt_contract": "byte_identical_V5_USER_TEMPLATE",
        "decoding": {
            "max_new_tokens": 4096,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        "max_seed_safety_bound": int(args.max_seed),
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "existing_sources": sources,
    }
    plan_path = args.output / f"frozen_scan_plan_{OUTPUT_SUFFIX}.json"
    atomic_json(plan_path, plan, immutable=True)

    all_seeds = tuple(range(SEED_START, int(args.max_seed) + 1))
    discovery_seeds = tuple(seed for seed in all_seeds if assigned_split(seed) == "discovery")
    confirmation_seeds = tuple(
        seed for seed in all_seeds if assigned_split(seed) == "confirmation"
    )
    config = V4Config(
        seeds=all_seeds,
        discovery_seeds=discovery_seeds,
        confirmation_seeds=confirmation_seeds,
    )
    config.validate()
    canonical_tokenizer = TokenizerAdapter(
        config.canonical_tokenizer,
        revision=config.canonical_tokenizer_revision,
        cache_dir=str(args.cache_dir),
    )
    if canonical_tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Pinned canonical tokenizer unavailable: "
            + str(canonical_tokenizer.load_error)
        )
    freeze_spec = ControlledFreezeSpec(
        config=config,
        haystack_dir=str(ROOT / "data" / "haystacks" / "paul_graham"),
        entities_path=str(ROOT / "data" / "entities" / "cities.csv"),
        fact_templates_path=str(
            ROOT / "data" / "templates" / "niah_fact_single_template.txt"
        ),
        tokenizer_cache_dir=str(args.cache_dir),
    )

    attempted: list[dict[str, Any]] = []
    model = tokenizer = spec = None
    started = time.perf_counter()
    for seed in all_seeds:
        shard = shard_dir / f"seed{seed}_N{FIXED_COUNT}.json"
        if args.resume and shard.exists():
            row = json.loads(shard.read_text(encoding="utf-8"))
            origin = str(
                row.get(
                    ORIGIN_KEY,
                    row.get("noindex_n3_generation_origin", "resumed_shard"),
                )
            )
            if (
                origin == "prospective_fixed_n3_scan"
                and PROSPECTIVE_ORIGIN != "prospective_fixed_n3_scan"
            ):
                origin = PROSPECTIVE_ORIGIN
        elif seed in existing:
            row = dict(existing[seed])
            origin = "reused_frozen_native_generation"
            row["frozen_prompt_audit"] = frozen_prompt_audit(row)
            row[AUDIT_KEY] = format_audit(row)
            row[ORIGIN_KEY] = origin
            atomic_json(shard, row)
        else:
            if model is None:
                spec = resolve_model_spec(str(args.model))
                model, tokenizer, _adapter = load_registered_model(
                    spec,
                    cache_dir=args.cache_dir,
                    device_map=args.device_map,
                    torch_dtype=args.torch_dtype,
                    attention_backend=args.attention_backend,
                )
            family, _metadata = build_controlled_family(
                variant="v4.4",
                seed=seed,
                tokenizer=canonical_tokenizer,
                freeze_spec=freeze_spec,
                active_counts=(FIXED_COUNT,),
            )
            if len(family) != 1 or int(family[0]["gold_count"]) != FIXED_COUNT:
                raise RuntimeError(
                    f"Seed {seed} did not produce one frozen N={FIXED_COUNT} stimulus"
                )
            stimulus = dict(family[0])
            prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=spec)
            expected_user = build_v5_user_text(
                str(stimulus["passage"]),
                entity_domain=str(stimulus.get("entity_domain", "city")),
            )
            if prompt.user_text != expected_user:
                raise ValueError("PROMPT_INTEGRITY_FAILURE before generation")
            row = generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=DecodingSpec(),
                sampling_seed=seed,
            )
            origin = PROSPECTIVE_ORIGIN
            row["frozen_prompt_audit"] = frozen_prompt_audit(row)
            row[AUDIT_KEY] = format_audit(row)
            row[ORIGIN_KEY] = origin
            atomic_json(shard, row)
        if int(row.get("seed", -1)) != seed or int(row.get("gold_count", -1)) != FIXED_COUNT:
            raise ValueError("Shard identity changed")
        if row.get("frozen_prompt_audit", {}).get("status") != "PASS":
            raise ValueError("Shard lacks frozen prompt PASS")
        corrected_audit = format_audit(row)
        if row.get(AUDIT_KEY) != corrected_audit or row.get(ORIGIN_KEY) != origin:
            row[AUDIT_KEY] = corrected_audit
            row[ORIGIN_KEY] = origin
            atomic_json(shard, row)
        attempted.append(row)
        discovery, confirmation, ledger = choose_rows(attempted)
        atomic_jsonl(ledger_path, ledger)
        audit = row[AUDIT_KEY]
        print(
            f"[{LOG_LABEL}] seed={seed} split={assigned_split(seed)} "
            f"origin={origin} marker={audit['marker_kind']} "
            f"prefix_clean={int(audit['primary_eligible_prefix_clean'])} "
            f"global_clean={int(audit['global_clean_no_running_index'])} "
            f"hits={len(discovery)}/{DISCOVERY_QUOTA}d+"
            f"{len(confirmation)}/{CONFIRMATION_QUOTA}c",
            flush=True,
        )
        if len(discovery) == DISCOVERY_QUOTA and len(confirmation) == CONFIRMATION_QUOTA:
            break
    else:
        raise RuntimeError("Reached max-seed before filling the frozen 20/10 cohort")

    selected = [*discovery, *confirmation]
    discovery_path = args.output / f"discovery_rows_{OUTPUT_SUFFIX}.jsonl"
    confirmation_path = args.output / f"confirmation_rows_{OUTPUT_SUFFIX}.jsonl"
    selected_path = args.output / f"selected_rows_{OUTPUT_SUFFIX}.jsonl"
    atomic_jsonl(discovery_path, discovery, immutable=True)
    atomic_jsonl(confirmation_path, confirmation, immutable=True)
    atomic_jsonl(selected_path, selected, immutable=True)
    manifest = {
        "schema_version": SCHEMA,
        "status": "FROZEN",
        "model_label": str(args.model),
        "fixed_count": FIXED_COUNT,
        "discovery_seeds": [int(row["seed"]) for row in discovery],
        "confirmation_seeds": [int(row["seed"]) for row in confirmation],
        "last_scanned_seed": int(attempted[-1]["seed"]),
        "attempted_seed_count": len(attempted),
        "strict_sensitivity": {
            "discovery_n": sum(
                bool(row[AUDIT_KEY]["strict_eligible_no_explicit_count_cue"])
                for row in discovery
            ),
            "confirmation_n": sum(
                bool(row[AUDIT_KEY]["strict_eligible_no_explicit_count_cue"])
                for row in confirmation
            ),
        },
        "selection_population": "first_pass_noindex_enumeration",
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            plan_path.name: sha256_file(plan_path),
            discovery_path.name: sha256_file(discovery_path),
            confirmation_path.name: sha256_file(confirmation_path),
            selected_path.name: sha256_file(selected_path),
            ledger_path.name: sha256_file(ledger_path),
        },
    }
    atomic_json(
        args.output / f"cohort_manifest_{OUTPUT_SUFFIX}.json",
        manifest,
        immutable=True,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
