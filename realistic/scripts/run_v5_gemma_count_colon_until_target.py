#!/usr/bin/env python3
"""Generate isolated Gemma N=10 traces until a frozen ``(Count: k)`` target is met."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    load_registered_model,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.generation import (  # noqa: E402
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl, write_jsonl  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402


SCHEMA = "realistic_niah_v5_gemma_count_colon_until_target_v1"
MODEL = "Gemma4-E4B"
ORDER_SALT = "gemma_count_colon_candidate_order_v1"


def _key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row.get("gold_count", 10))


def _is_count_colon_trace(row: Mapping[str, Any]) -> bool:
    parsed = dict(row.get("trace_parse", {}))
    parser = dict(parsed.get("parser", {}))
    if not bool(parsed.get("exact_count")):
        return False
    if str(parser.get("trace_category")) != "one_to_one":
        return False
    if int(parser.get("item_count", -1)) != 10:
        return False
    events = list(dict(parsed.get("episode_parse", {})).get("events", ()))
    if len(events) != 10:
        return False
    for occurrence, event in enumerate(events, start=1):
        if int(event.get("rank", -1)) != occurrence:
            return False
        if str(event.get("association")) != "rank_after_city":
            return False
        if str(event.get("evidence_family")) != "inline_count":
            return False
        surface = str(event.get("evidence_surface", "")).strip()
        if re.fullmatch(rf"Count:\s*{occurrence}\)", surface) is None:
            return False
    return True


def _load_decoding(config_path: Path) -> DecodingSpec:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoding = DecodingSpec(**dict(config.get("decoding", {})))
    decoding.validate()
    return decoding


def _candidate_order(row: Mapping[str, Any]) -> tuple[str, int]:
    seed = int(row["seed"])
    digest = hashlib.sha256(f"{ORDER_SALT}:{seed}".encode("utf-8")).hexdigest()
    return digest, seed


def run(args: argparse.Namespace) -> dict[str, Any]:
    stimuli = [
        row
        for row in read_jsonl(args.stimuli)
        if int(row.get("gold_count", -1)) == 10
    ]
    stimuli.sort(key=_candidate_order)
    if len({_key(row) for row in stimuli}) != len(stimuli):
        raise ValueError("Candidate stimuli contain duplicate seed/count keys")
    existing = read_jsonl(args.output) if args.output.is_file() else []
    existing_by_key = {_key(row): row for row in existing}
    if len(existing_by_key) != len(existing):
        raise ValueError("Existing generation file contains duplicate keys")
    accepted = [row for row in existing if _is_count_colon_trace(row)]
    if len(accepted) >= args.target_accepted:
        pending: list[dict[str, Any]] = []
    else:
        pending = [row for row in stimuli if _key(row) not in existing_by_key]

    model_spec = resolve_model_spec(MODEL)
    model, tokenizer, _adapter = load_registered_model(
        model_spec,
        cache_dir=str(args.cache_dir),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    output_by_key = dict(existing_by_key)
    print(
        f"[Gemma count-colon] pool={len(stimuli)} existing={len(existing)} "
        f"accepted={len(accepted)} target={args.target_accepted}",
        flush=True,
    )
    for index, stimulus in enumerate(pending, start=1):
        prompt = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=model_spec)
        generated = generate_native_trace(
            model,
            tokenizer,
            prompt,
            decoding=_load_decoding(args.config),
            sampling_seed=int(stimulus["seed"]),
        )
        generated["split"] = str(stimulus["split"])
        generated["supplemental"] = True
        generated["supplement_schema_version"] = SCHEMA
        output_by_key[_key(stimulus)] = generated
        ordered = [
            output_by_key[_key(row)]
            for row in stimuli
            if _key(row) in output_by_key
        ]
        write_jsonl(args.output, ordered)
        accepted = [row for row in ordered if _is_count_colon_trace(row)]
        parser = dict(dict(generated.get("trace_parse", {})).get("parser", {}))
        print(
            f"[Gemma count-colon] generated={index} seed={stimulus['seed']} "
            f"category={parser.get('trace_category')} accepted={len(accepted)}/"
            f"{args.target_accepted}",
            flush=True,
        )
        if len(accepted) >= args.target_accepted:
            break
    if len(accepted) < args.target_accepted:
        raise RuntimeError(
            f"Candidate pool exhausted at {len(accepted)}/{args.target_accepted}"
        )
    accepted_rows = [
        {
            "model_label": MODEL,
            "request_id": str(row["request_id"]),
            "seed": int(row["seed"]),
            "source_split": str(row["split"]),
            "gold_count": 10,
            "surface_family": "count_colon",
        }
        for row in accepted[: args.target_accepted]
    ]
    args.accepted_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in accepted_rows) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": SCHEMA,
        "candidate_order_salt": ORDER_SALT,
        "generated_rows": len(output_by_key),
        "accepted_rows": len(accepted_rows),
        "accepted_seeds": [row["seed"] for row in accepted_rows],
        "output": str(args.output),
        "accepted_output": str(args.accepted_output),
    }
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-accepted", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.accepted_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.target_accepted < 1:
        raise ValueError("target_accepted must be positive")
    run(args)


if __name__ == "__main__":
    main()
