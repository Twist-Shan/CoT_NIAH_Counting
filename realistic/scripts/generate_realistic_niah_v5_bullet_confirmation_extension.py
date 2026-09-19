#!/usr/bin/env python3
"""Prospectively freeze a fresh outcome-blind Qwen N=10 list confirmation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.bullet_counterfactual_restore import (  # noqa: E402
    audit_complete_marker_scrubbable_list,
)
from realistic_niah_v5.generation import (  # noqa: E402
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402
from scripts.assemble_realistic_niah_v5_bullet_counter_cohort import (  # noqa: E402
    _assert_frozen_prompt,
    _build_n10_stimulus,
    _prospective_freeze_spec,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1304)
    parser.add_argument("--target-eligible", type=int, default=10)
    parser.add_argument("--max-seed", type=int, default=1500)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if int(args.seed_start) <= 1303:
        raise ValueError("Fresh confirmation extension must begin after seed 1303")
    if int(args.target_eligible) != 10:
        raise ValueError("Fresh confirmation size is frozen to 10")
    if int(args.max_seed) < int(args.seed_start) + 10:
        raise ValueError("max-seed bound is too small")
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / "generation_shards"
    shards.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "frozen_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "FROZEN" or len(manifest.get("rows", ())) != 10:
            raise ValueError("Existing extension manifest is not a valid frozen set")
        print(f"[fresh-confirmation] already frozen: {manifest_path}", flush=True)
        return

    plan = {
        "schema_version": "bullet_counter_fresh_confirmation_plan_v1",
        "model_label": "Qwen3-8B",
        "gold_count": 10,
        "seed_start": int(args.seed_start),
        "missing_seed_policy": "advance_seed_by_one_until_10_format_eligible_rows",
        "target_eligible": 10,
        "eligibility": "complete_contiguous_one_to_one_marker_scrubbable_10_item_list",
        "eligibility_uses_final_answer": False,
        "patch_outcomes_accessed": False,
        "construction_and_sites_frozen_before_generation": True,
        "decoding": {
            "max_new_tokens": 4096,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
        },
    }
    plan_path = args.output / "generation_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing prospective plan differs")
    else:
        _atomic_json(plan_path, plan)

    freeze_spec, canonical_tokenizer = _prospective_freeze_spec(
        seed_start=1234,
        max_seed=int(args.max_seed),
        tokenizer_cache_dir=args.cache_dir,
    )
    spec = resolve_model_spec("Qwen3-8B")
    model, tokenizer, _adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    decoding = DecodingSpec()
    eligible: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in range(int(args.seed_start), int(args.max_seed) + 1):
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
            row = generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=decoding,
                sampling_seed=seed,
            )
            row["frozen_prompt_audit"] = _assert_frozen_prompt(stimulus, prompt)
            row["bullet_counter_generation_origin"] = (
                "prospective_fresh_confirmation_extension"
            )
            row["bullet_counter_cohort_audit"] = (
                audit_complete_marker_scrubbable_list(row)
            )
            _atomic_json(shard, row)
        audit = row.get("bullet_counter_cohort_audit")
        if not isinstance(audit, dict):
            audit = audit_complete_marker_scrubbable_list(row)
            row["bullet_counter_cohort_audit"] = audit
            _atomic_json(shard, row)
        scanned.append(row)
        if bool(audit["eligible"]):
            eligible.append(row)
        print(
            f"[fresh-confirmation] seed={seed} marker={audit.get('marker_kind')} "
            f"items={audit.get('parsed_item_count')} eligible={int(bool(audit['eligible']))} "
            f"hits={len(eligible)}/10 elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
        if len(eligible) == 10:
            break

    _atomic_jsonl(args.output / "scanned_generations.jsonl", scanned)
    if len(eligible) != 10:
        raise RuntimeError("max-seed reached before 10 fresh eligible traces")
    for rank, row in enumerate(eligible, start=1):
        row["fresh_confirmation_rank"] = rank
    _atomic_jsonl(args.output / "eligible_generations.jsonl", eligible)
    manifest_rows = [
        {
            "rank": int(row["fresh_confirmation_rank"]),
            "seed": int(row["seed"]),
            "request_id": str(row["request_id"]),
            "input_ids_sha256": _sha256_json(row.get("input_ids", ())),
            "output_token_ids_sha256": _sha256_json(row.get("output_token_ids", ())),
            "marker_kind": str(row["bullet_counter_cohort_audit"]["marker_kind"]),
        }
        for row in eligible
    ]
    _atomic_json(
        manifest_path,
        {
            **plan,
            "status": "FROZEN",
            "scanned_seed_count": len(scanned),
            "last_scanned_seed": int(scanned[-1]["seed"]),
            "rows": manifest_rows,
        },
    )
    print(f"[fresh-confirmation] FROZEN seeds={[r['seed'] for r in eligible]}", flush=True)


if __name__ == "__main__":
    main()
