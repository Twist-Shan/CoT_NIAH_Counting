#!/usr/bin/env python3
"""Generate an outcome-blind natural-bullet rate pilot with the frozen prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.run_realistic_niah_v5_count_stream import _atomic_json, _atomic_jsonl  # noqa: E402
from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.generation import (  # noqa: E402
    NativePrompt,
    build_v5_user_text,
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_prompt(
    stimulus: Mapping[str, Any], prompt: NativePrompt
) -> dict[str, Any]:
    """Require the preregistered V5 user text with no added assistant prefix."""

    if "passage" not in stimulus:
        raise ValueError("Frozen-prompt pilot stimulus has no passage")
    expected = build_v5_user_text(
        str(stimulus["passage"]),
        entity_domain=str(stimulus.get("entity_domain", "city")),
    )
    if prompt.user_text != expected:
        raise ValueError("PROMPT_INTEGRITY_FAILURE: user_text differs from V5_USER_TEMPLATE")
    return {
        "status": "PASS",
        "user_text_byte_exact": True,
        "extra_user_instruction": False,
        "extra_system_message": False,
        "extra_assistant_prefix": False,
        "user_text_sha256": _sha256_bytes(prompt.user_text.encode("utf-8")),
        "rendered_prompt_sha256": _sha256_bytes(prompt.rendered_prompt.encode("utf-8")),
        "input_ids_sha256": _sha256_bytes(
            json.dumps(list(prompt.input_ids), separators=(",", ":")).encode("utf-8")
        ),
    }


def _load_rows(
    path: Path,
    *,
    expected_seeds: tuple[int, ...],
    counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    seed_set = set(expected_seeds)
    count_set = set(counts)
    rows = [
        row
        for row in read_jsonl(path)
        if int(row.get("seed", -1)) in seed_set
        and int(row.get("gold_count", -1)) in count_set
    ]
    expected = {(seed, count) for seed in expected_seeds for count in counts}
    actual = {(int(row["seed"]), int(row["gold_count"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Frozen-prompt pilot stimulus seed/count panel changed")
    seed_rank = {seed: rank for rank, seed in enumerate(expected_seeds)}
    count_rank = {count: rank for rank, count in enumerate(counts)}
    rows.sort(key=lambda row: (seed_rank[int(row["seed"])], count_rank[int(row["gold_count"])]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(10, 9))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    expected_seeds = tuple(int(value) for value in args.expected_seeds)
    counts = tuple(int(value) for value in args.counts)
    if len(expected_seeds) != 100 or len(set(expected_seeds)) != 100:
        raise ValueError("Rate pilot is frozen to 100 unique seeds")
    if counts != (10, 9):
        raise ValueError("Rate pilot is frozen to N=10 then N=9")
    rows = _load_rows(args.stimuli, expected_seeds=expected_seeds, counts=counts)
    plan = {
        "schema_version": "realistic_niah_v5_frozen_prompt_bullet_rate_plan_v1",
        "model_label": str(args.model),
        "seeds": list(expected_seeds),
        "seed_count": 100,
        "count_order": list(counts),
        "generation_row_count": len(rows),
        "stimuli_sha256": _sha256_file(args.stimuli),
        "prompt_contract": "byte_identical_V5_USER_TEMPLATE",
        "extra_user_instruction": False,
        "extra_system_message": False,
        "extra_assistant_prefix": False,
        "decoding": {
            "max_new_tokens": 4096,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        "primary_success_rule": (
            "seed succeeds iff N=10 or N=9 is a complete one-to-one natural "
            "trace using bullet, audit-sentence, evidence-sequence, or "
            "completion-recap grammar without item indices; a pre-item "
            "final/progress total is allowed because it is constant across the "
            "item-k diagonal"
        ),
        "secondary_strict_rule": (
            "primary success plus no explicit record enumeration or progress "
            "total in any registered item causal prefix"
        ),
        "count_priority_for_future_selection": [10, 9],
        "patch_outcomes_accessed": False,
        "outcome_blind": True,
        "selection_rank_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "frozen_rate_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing frozen-prompt pilot plan changed")
    else:
        _atomic_json(plan_path, plan)

    spec = resolve_model_spec(str(args.model))
    model, tokenizer, _adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    decoding = DecodingSpec()
    shard_dir = args.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    newly_completed = 0
    for index, stimulus in enumerate(rows, start=1):
        seed = int(stimulus["seed"])
        count = int(stimulus["gold_count"])
        shard = shard_dir / f"seed{seed}_N{count}.json"
        if args.resume and shard.exists():
            generated = json.loads(shard.read_text(encoding="utf-8"))
            if generated.get("frozen_prompt_audit", {}).get("status") != "PASS":
                raise ValueError("Resumed pilot shard lacks frozen prompt PASS")
        else:
            prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=spec)
            prompt_audit = assert_frozen_prompt(stimulus, prompt)
            generated = generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=decoding,
                sampling_seed=seed,
            )
            if generated["user_text"] != prompt.user_text:
                raise RuntimeError("PROMPT_INTEGRITY_FAILURE after generation")
            format_audit = audit_no_count_enumeration_trace(generated)
            parser_record = generated["trace_parse"]["parser"]
            marker_kind = str(parser_record["marker_kind"])
            one_to_one = bool(parser_record.get("trace_one_to_one"))
            item_count = int(parser_record.get("item_count", -1))
            reasons = tuple(str(value) for value in format_audit["reasons"])
            non_progress_reasons = tuple(
                value
                for value in reasons
                if not value.endswith("explicit_progress_total_in_causal_prefix")
            )
            accepted_markers = {
                "bullet",
                "audit_sentence",
                "evidence_sequence",
                "completion_recap",
            }
            primary_no_index = bool(
                marker_kind in accepted_markers
                and one_to_one
                and item_count == int(generated["gold_count"])
                and not non_progress_reasons
            )
            generated["frozen_prompt_audit"] = prompt_audit
            generated["natural_bullet_rate_audit"] = {
                **format_audit,
                "marker_kind": marker_kind,
                "trace_one_to_one": one_to_one,
                "item_count": item_count,
                "primary_eligible_no_index": primary_no_index,
                "primary_eligible_bullet": bool(
                    primary_no_index and marker_kind == "bullet"
                ),
                "strict_eligible_bullet": bool(
                    primary_no_index
                    and marker_kind == "bullet"
                    and format_audit["eligible"]
                ),
                "patch_outcomes_accessed": False,
            }
            _atomic_json(shard, generated)
            newly_completed += 1
        generated_rows.append(generated)
        audit = generated["natural_bullet_rate_audit"]
        print(
            f"[frozen-prompt-bullet-pilot] {index}/{len(rows)} "
            f"seed={seed} N={count} marker={audit['marker_kind']} "
            f"primary={int(audit['primary_eligible_no_index'])} "
            f"bullet={int(audit['primary_eligible_bullet'])} "
            f"strict={int(audit['strict_eligible_bullet'])}",
            flush=True,
        )

    successes_by_seed: dict[int, list[int]] = {}
    bullet_successes_by_seed: dict[int, list[int]] = {}
    strict_successes_by_seed: dict[int, list[int]] = {}
    for row in generated_rows:
        if row["natural_bullet_rate_audit"]["primary_eligible_no_index"]:
            successes_by_seed.setdefault(int(row["seed"]), []).append(int(row["gold_count"]))
        if row["natural_bullet_rate_audit"]["primary_eligible_bullet"]:
            bullet_successes_by_seed.setdefault(int(row["seed"]), []).append(
                int(row["gold_count"])
            )
        if row["natural_bullet_rate_audit"]["strict_eligible_bullet"]:
            strict_successes_by_seed.setdefault(int(row["seed"]), []).append(
                int(row["gold_count"])
            )
    _atomic_jsonl(args.output / "generations.jsonl", generated_rows)
    _atomic_json(
        args.output / "rate_manifest.json",
        {
            "schema_version": "realistic_niah_v5_frozen_prompt_bullet_rate_manifest_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "seed_count": len(expected_seeds),
            "generation_row_count": len(generated_rows),
            "primary_eligible_no_index_row_count": sum(
                bool(row["natural_bullet_rate_audit"]["primary_eligible_no_index"])
                for row in generated_rows
            ),
            "primary_eligible_no_index_seed_count": len(successes_by_seed),
            "successful_seed_counts": {
                str(seed): sorted(values, reverse=True)
                for seed, values in sorted(successes_by_seed.items())
            },
            "primary_eligible_bullet_row_count": sum(
                bool(row["natural_bullet_rate_audit"]["primary_eligible_bullet"])
                for row in generated_rows
            ),
            "primary_eligible_bullet_seed_count": len(bullet_successes_by_seed),
            "strict_eligible_bullet_row_count": sum(
                bool(row["natural_bullet_rate_audit"]["strict_eligible_bullet"])
                for row in generated_rows
            ),
            "strict_eligible_bullet_seed_count": len(strict_successes_by_seed),
            "all_prompts_byte_exact": all(
                row["frozen_prompt_audit"]["status"] == "PASS"
                for row in generated_rows
            ),
            "newly_completed_rows": newly_completed,
            "elapsed_seconds": time.perf_counter() - started,
            "patch_outcomes_accessed": False,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )


if __name__ == "__main__":
    main()
