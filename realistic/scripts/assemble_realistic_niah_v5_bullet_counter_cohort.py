#!/usr/bin/env python3
"""Generate and freeze 20+10 complete marker-scrubbable N=10 lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


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
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    audit_complete_marker_scrubbable_list,
)
from realistic_niah_v5.generation import (  # noqa: E402
    build_v5_user_text,
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
)


SCHEMA_VERSION = "realistic_niah_v5_marker_scrubbed_list_cohort_v2"
SEED_START = 1234
DISCOVERY_SIZE = 20
CONFIRMATION_SIZE = 10
TARGET_ELIGIBLE = DISCOVERY_SIZE + CONFIRMATION_SIZE


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_existing_rows(
    paths: tuple[Path, ...], *, model_label: str, seed_start: int
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        sources = sorted(path.glob("seed*_N10.json")) if path.is_dir() else [path]
        loaded_rows = (
            [json.loads(source.read_text(encoding="utf-8")) for source in sources]
            if path.is_dir()
            else read_jsonl(path)
        )
        for row in loaded_rows:
            if (
                str(row.get("model_label")) != str(model_label)
                or int(row.get("gold_count", -1)) != 10
                or int(row.get("seed", -1)) < int(seed_start)
            ):
                continue
            seed = int(row["seed"])
            previous = rows.get(seed)
            if previous is not None:
                identity = (
                    str(previous.get("request_id")),
                    _sha256_json(previous.get("input_ids", ())),
                    _sha256_json(previous.get("output_token_ids", ())),
                )
                candidate = (
                    str(row.get("request_id")),
                    _sha256_json(row.get("input_ids", ())),
                    _sha256_json(row.get("output_token_ids", ())),
                )
                if identity != candidate:
                    raise ValueError(
                        f"Conflicting existing deterministic generations for seed {seed}"
                    )
                continue
            rows[seed] = dict(row)
    return rows


def _prospective_freeze_spec(
    *, seed_start: int, max_seed: int, tokenizer_cache_dir: Path
) -> tuple[ControlledFreezeSpec, TokenizerAdapter]:
    seeds = tuple(range(int(seed_start), int(max_seed) + 1))
    if len(seeds) < 2:
        raise ValueError("Prospective seed registry must contain at least two seeds")
    base = V4Config()
    config = replace(
        base,
        seeds=seeds,
        discovery_seeds=seeds[:-1],
        confirmation_seeds=seeds[-1:],
    )
    freeze_spec = ControlledFreezeSpec(
        config=config,
        haystack_dir=str(ROOT / "data" / "haystacks" / "paul_graham"),
        entities_path=str(ROOT / "data" / "entities" / "cities.csv"),
        fact_templates_path=str(
            ROOT / "data" / "templates" / "niah_fact_single_template.txt"
        ),
        tokenizer_cache_dir=str(tokenizer_cache_dir),
    )
    freeze_spec.validate()
    canonical_tokenizer = TokenizerAdapter(
        config.canonical_tokenizer,
        revision=config.canonical_tokenizer_revision,
        cache_dir=str(tokenizer_cache_dir),
    )
    if canonical_tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Canonical Qwen stimulus tokenizer did not load from Hugging Face cache: "
            f"{canonical_tokenizer.load_error}"
        )
    return freeze_spec, canonical_tokenizer


def _build_n10_stimulus(
    seed: int,
    *,
    freeze_spec: ControlledFreezeSpec,
    canonical_tokenizer: TokenizerAdapter,
) -> dict[str, Any]:
    rows, metadata = build_controlled_family(
        variant="v4.4",
        seed=int(seed),
        tokenizer=canonical_tokenizer,
        freeze_spec=freeze_spec,
        active_counts=(10,),
    )
    if len(rows) != 1 or int(rows[0].get("gold_count", -1)) != 10:
        raise RuntimeError("Prospective V4.4 builder did not return one N=10 stimulus")
    stimulus = dict(rows[0])
    stimulus["prospective_family_metadata_sha256"] = _sha256_json(metadata)
    return stimulus


def _assert_frozen_prompt(stimulus: Mapping[str, Any], prompt: Any) -> dict[str, Any]:
    expected = build_v5_user_text(
        str(stimulus["passage"]),
        entity_domain=str(stimulus.get("entity_domain", "city")),
    )
    if str(prompt.user_text) != expected:
        raise ValueError("PROMPT_INTEGRITY_FAILURE: V5 user template changed")
    return {
        "status": "PASS",
        "user_text_byte_exact": True,
        "extra_user_instruction": False,
        "extra_system_message": False,
        "extra_assistant_prefix": False,
        "user_text_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(
            str(prompt.rendered_prompt).encode("utf-8")
        ).hexdigest(),
        "input_ids_sha256": _sha256_json(prompt.input_ids),
    }


def _cohort_manifest(
    *, model_label: str, eligible_rows: list[dict[str, Any]], scanned_rows: int
) -> dict[str, Any]:
    if len(eligible_rows) != TARGET_ELIGIBLE:
        raise ValueError("Cannot freeze a cohort before 30 eligible rows exist")
    ordered = sorted(eligible_rows, key=lambda row: int(row["seed"]))
    selected: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        role = "discovery" if rank <= DISCOVERY_SIZE else "confirmation"
        selected.append(
            {
                "selection_rank": rank,
                "cohort_role": role,
                "seed": int(row["seed"]),
                "request_id": str(row["request_id"]),
                "request_id_sha256": hashlib.sha256(
                    str(row["request_id"]).encode("utf-8")
                ).hexdigest(),
                "input_ids_sha256": _sha256_json(row.get("input_ids", ())),
                "output_token_ids_sha256": _sha256_json(
                    row.get("output_token_ids", ())
                ),
                "raw_output_text_sha256": hashlib.sha256(
                    str(row.get("raw_output_text", "")).encode("utf-8")
                ).hexdigest(),
                "generation_origin": str(
                    row.get("bullet_counter_generation_origin", "unknown")
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN",
        "model_label": str(model_label),
        "source_gold_count": 10,
        "seed_scan_start": SEED_START,
        "missing_seed_policy": "increase_seed_by_one_until_30_format_eligible_rows",
        "eligible_selection_order": "ascending_seed",
        "discovery_size": DISCOVERY_SIZE,
        "confirmation_size": CONFIRMATION_SIZE,
        "selected_seed_count": TARGET_ELIGIBLE,
        "last_selected_seed": int(selected[-1]["seed"]),
        "scanned_row_count": int(scanned_rows),
        "eligibility": "complete_contiguous_one_to_one_marker_scrubbable_10_item_list",
        "selection_uses_final_answer": False,
        "patch_outcomes_accessed": False,
        "prompt_modified": False,
        "rows": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--existing-generations", type=Path, nargs="*", default=())
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--max-seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if int(args.seed_start) != SEED_START:
        raise ValueError("Formal seed scan must start at 1234")
    if int(args.max_seed) < SEED_START + TARGET_ELIGIBLE:
        raise ValueError("max-seed safety bound is too small")
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / "generation_shards"
    shards.mkdir(parents=True, exist_ok=True)
    cohort_path = args.output / "frozen_cohort_manifest.json"
    if cohort_path.exists():
        frozen = json.loads(cohort_path.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "FROZEN"
            or frozen.get("model_label") != str(args.model)
            or len(frozen.get("rows", ())) != TARGET_ELIGIBLE
        ):
            raise ValueError("Existing cohort manifest is not the registered frozen cohort")
        print(f"[bullet-cohort] already frozen: {cohort_path}", flush=True)
        return

    existing = _load_existing_rows(
        tuple(args.existing_generations),
        model_label=str(args.model),
        seed_start=int(args.seed_start),
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "model_label": str(args.model),
        "seed_start": SEED_START,
        "max_seed_safety_bound": int(args.max_seed),
        "source_gold_count": 10,
        "target_eligible": TARGET_ELIGIBLE,
        "discovery_size": DISCOVERY_SIZE,
        "confirmation_size": CONFIRMATION_SIZE,
        "prompt_contract": "byte_identical_V5_USER_TEMPLATE",
        "decoding": {
            "max_new_tokens": 4096,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        "eligibility_uses_final_answer": False,
        "patch_outcomes_accessed": False,
    }
    plan_path = args.output / "generation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing prospective generation plan changed")
    else:
        _atomic_json(plan_path, plan)

    # Audit every already available deterministic row before loading the model.
    eligible: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    next_missing_seed: int | None = None
    for seed in range(SEED_START, int(args.max_seed) + 1):
        shard = shards / f"seed{seed}_N10.json"
        if shard.exists():
            row = json.loads(shard.read_text(encoding="utf-8"))
        elif seed in existing:
            row = dict(existing[seed])
            row["bullet_counter_generation_origin"] = "reused_frozen_native_generation"
            audit = audit_complete_marker_scrubbable_list(row)
            row["bullet_counter_cohort_audit"] = audit
            _atomic_json(shard, row)
        else:
            next_missing_seed = seed
            break
        audit = row.get("bullet_counter_cohort_audit")
        if not isinstance(audit, Mapping) or audit.get("schema_version") != "marker_scrubbable_list_raw_audit_v2":
            audit = audit_complete_marker_scrubbable_list(row)
            row["bullet_counter_cohort_audit"] = audit
            _atomic_json(shard, row)
        scanned.append(row)
        if bool(audit.get("eligible")):
            eligible.append(row)
        print(
            f"[bullet-cohort] reused seed={seed} eligible={int(bool(audit.get('eligible')))} "
            f"hits={len(eligible)}/{TARGET_ELIGIBLE}",
            flush=True,
        )
        if len(eligible) == TARGET_ELIGIBLE:
            break

    if len(eligible) < TARGET_ELIGIBLE:
        if next_missing_seed is None:
            raise RuntimeError("Reached max-seed before the cohort was complete")
        freeze_spec, canonical_tokenizer = _prospective_freeze_spec(
            seed_start=SEED_START,
            max_seed=int(args.max_seed),
            tokenizer_cache_dir=args.cache_dir,
        )
        spec = resolve_model_spec(str(args.model))
        model, tokenizer, _adapter = load_registered_model(
            spec,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
        )
        decoding = DecodingSpec()
        started = time.perf_counter()
        for seed in range(int(next_missing_seed), int(args.max_seed) + 1):
            shard = shards / f"seed{seed}_N10.json"
            if args.resume and shard.exists():
                row = json.loads(shard.read_text(encoding="utf-8"))
            else:
                stimulus = _build_n10_stimulus(
                    seed,
                    freeze_spec=freeze_spec,
                    canonical_tokenizer=canonical_tokenizer,
                )
                prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=spec)
                prompt_audit = _assert_frozen_prompt(stimulus, prompt)
                row = generate_native_trace(
                    model,
                    tokenizer,
                    prompt,
                    decoding=decoding,
                    sampling_seed=int(seed),
                )
                row["frozen_prompt_audit"] = prompt_audit
                row["bullet_counter_generation_origin"] = "prospective_seed_extension"
                row["bullet_counter_cohort_audit"] = (
                    audit_complete_marker_scrubbable_list(row)
                )
                _atomic_json(shard, row)
            audit = row["bullet_counter_cohort_audit"]
            scanned.append(row)
            if bool(audit["eligible"]):
                eligible.append(row)
            print(
                f"[bullet-cohort] generated seed={seed} "
                f"marker={audit.get('marker_kind')} items={audit.get('parsed_item_count')} "
                f"eligible={int(bool(audit['eligible']))} "
                f"hits={len(eligible)}/{TARGET_ELIGIBLE} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if len(eligible) == TARGET_ELIGIBLE:
                break

    if len(eligible) != TARGET_ELIGIBLE:
        _atomic_jsonl(args.output / "scanned_generations.jsonl", scanned)
        _atomic_json(
            args.output / "incomplete_manifest.json",
            {
                **plan,
                "status": "INCOMPLETE_MAX_SEED_REACHED",
                "scanned_row_count": len(scanned),
                "eligible_row_count": len(eligible),
                "last_scanned_seed": int(scanned[-1]["seed"]) if scanned else None,
            },
        )
        raise RuntimeError("max-seed reached before 30 eligible list traces")

    eligible.sort(key=lambda row: int(row["seed"]))
    selected_seed_set = {int(row["seed"]) for row in eligible}
    selected_rows = [row for row in scanned if int(row["seed"]) in selected_seed_set]
    _atomic_jsonl(args.output / "eligible_generations.jsonl", selected_rows)
    manifest = _cohort_manifest(
        model_label=str(args.model),
        eligible_rows=selected_rows,
        scanned_rows=len(scanned),
    )
    _atomic_json(cohort_path, manifest)
    print(
        f"[bullet-cohort] FROZEN model={args.model} "
        f"last_seed={manifest['last_selected_seed']} "
        f"scanned={manifest['scanned_row_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
