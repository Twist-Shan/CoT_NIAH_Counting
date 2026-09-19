#!/usr/bin/env python3
"""Generate an isolated, resumable N=10 native-thinking seed supplement.

This runner deliberately does not relax ``V5Config.validate``: the registered
300-row experiment remains frozen to seeds 1234..1263.  Instead it validates a
supplement manifest locally and writes to a separate output tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.generation import (  # noqa: E402
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl, write_jsonl  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402


SCHEMA = "realistic_niah_v5_pure_trace_n10_supplement_runner_v1"
SOURCE_SCHEMA = "realistic_niah_v5_config_v2"


def _load_supplement_config(path: Path) -> tuple[tuple[int, ...], tuple[int, ...], DecodingSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != SOURCE_SCHEMA:
        raise ValueError(f"Unsupported source config schema in {path}")
    discovery = tuple(int(value) for value in payload["discovery_seeds"])
    confirmation = tuple(int(value) for value in payload["confirmation_seeds"])
    if not discovery or not confirmation:
        raise ValueError("Supplement requires non-empty discovery and confirmation seeds")
    if len(set(discovery)) != len(discovery) or len(set(confirmation)) != len(confirmation):
        raise ValueError("Supplement seed roles must be unique")
    if set(discovery) & set(confirmation):
        raise ValueError("Supplement discovery and confirmation seeds overlap")
    if min(discovery + confirmation) <= 1263:
        raise ValueError("Supplement seeds must be disjoint from registered seeds 1234..1263")
    counts = tuple(int(value) for value in payload.get("counts", ()))
    if counts != tuple(range(1, 11)):
        raise ValueError("Source stimulus design must retain counts 1..10")
    decoding = DecodingSpec(**dict(payload.get("decoding", {})))
    decoding.validate()
    return discovery, confirmation, decoding


def _trajectory_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row.get("gold_count", 10))


def generate(args: argparse.Namespace) -> dict[str, Any]:
    discovery, confirmation, decoding = _load_supplement_config(args.config)
    split_by_seed = {
        **{seed: "discovery" for seed in discovery},
        **{seed: "confirmation" for seed in confirmation},
    }
    stimuli = []
    for row in read_jsonl(args.stimuli):
        seed = int(row["seed"])
        count = int(row["gold_count"])
        if seed not in split_by_seed or count != 10:
            continue
        value = dict(row)
        expected_split = split_by_seed[seed]
        if str(value.get("split")) != expected_split:
            raise ValueError(
                f"Stimulus seed {seed} split {value.get('split')!r} != {expected_split!r}"
            )
        value["split"] = expected_split
        stimuli.append(value)
    expected_keys = {(seed, 10) for seed in discovery + confirmation}
    observed_keys = {_trajectory_key(row) for row in stimuli}
    if observed_keys != expected_keys or len(stimuli) != len(expected_keys):
        raise ValueError(
            f"Supplement stimulus panel mismatch: rows={len(stimuli)}, "
            f"keys={len(observed_keys)}, expected={len(expected_keys)}"
        )

    existing = read_jsonl(args.output) if args.output.is_file() else []
    existing_by_key = {_trajectory_key(row): row for row in existing}
    if len(existing_by_key) != len(existing):
        raise ValueError(f"Duplicate existing trajectories in {args.output}")
    unexpected_existing = sorted(set(existing_by_key) - expected_keys)
    if unexpected_existing:
        raise ValueError(f"Unexpected existing supplement keys: {unexpected_existing}")

    model_spec = resolve_model_spec(args.model)
    model, tokenizer, _adapter = load_registered_model(
        model_spec,
        cache_dir=str(args.cache_dir),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    output_by_key = dict(existing_by_key)
    pending = [row for row in stimuli if _trajectory_key(row) not in output_by_key]
    print(
        f"[supplement generate] model={args.model} total={len(stimuli)} "
        f"existing={len(existing)} pending={len(pending)}",
        flush=True,
    )
    for index, stimulus in enumerate(pending, start=1):
        prompt = render_native_prompt(
            stimulus,
            tokenizer=tokenizer,
            model_spec=model_spec,
        )
        generated = generate_native_trace(
            model,
            tokenizer,
            prompt,
            decoding=decoding,
            sampling_seed=int(stimulus["seed"]),
        )
        generated["split"] = str(stimulus["split"])
        generated["supplemental"] = True
        generated["supplement_schema_version"] = SCHEMA
        output_by_key[_trajectory_key(stimulus)] = generated
        ordered = [output_by_key[_trajectory_key(row)] for row in stimuli if _trajectory_key(row) in output_by_key]
        write_jsonl(args.output, ordered)
        parser = generated.get("trace_parse", {}).get("parser", {})
        print(
            f"[supplement generate] {index}/{len(pending)} seed={stimulus['seed']} "
            f"category={parser.get('trace_category')} exact={parser.get('exact_count')}",
            flush=True,
        )

    result = {
        "schema_version": SCHEMA,
        "model_label": args.model,
        "discovery_seeds": list(discovery),
        "confirmation_seeds": list(confirmation),
        "rows": len(output_by_key),
        "output": str(args.output),
    }
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend")
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
