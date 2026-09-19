#!/usr/bin/env python3
"""Generate a controlled Gemma N=10 nested-record grammar cohort.

The registered natural prompt remains untouched.  This isolated sensitivity
cohort asks Gemma to emit ``Record k: (city, score)`` and retains the native
thinking grammar ``*   Record k: (city, score)``.  The running index therefore
precedes the retrieved content and ``item_end`` is the closing parenthesis
after the score, rather than a token adjacent to the index.  Freezing the
nested-bullet shell also prevents plain and bulleted record lines from forming
a second nuisance cluster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.generation import (  # noqa: E402
    _chat_template_kwargs,
    _flat,
    _offsets,
    _token_span,
    generate_native_trace,
    render_native_prompt,
)
from realistic_niah_v5.pipeline import read_jsonl, write_jsonl  # noqa: E402
from realistic_niah_v5.spec import DecodingSpec  # noqa: E402


SCHEMA = "realistic_niah_v5_gemma_prefix_record_control_v1"
MODEL = "Gemma4-E4B"
ORDER_SALT = "gemma_prefix_record_control_candidate_order_v1"
USER_TEMPLATE = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

Find every city-score audit record. During reasoning, output exactly one line
per found record, in passage order, using only this grammar:
Record <running index>: (<city>, <score>)

Start the running index at 1 and increment it by 1. Do not add arrows, quotes,
other labels, or commentary. After the final record,
output exactly one additional line:
Total: <integer>"""


def _key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row.get("gold_count", 10))


def _load_decoding(path: Path) -> DecodingSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding = DecodingSpec(**dict(payload.get("decoding", {})))
    decoding.validate()
    return decoding


def _candidate_order(row: Mapping[str, Any]) -> tuple[str, int]:
    seed = int(row["seed"])
    digest = hashlib.sha256(f"{ORDER_SALT}:{seed}".encode()).hexdigest()
    return digest, seed


def _controlled_prompt(stimulus: Mapping[str, Any], tokenizer: Any, model_spec: Any) -> Any:
    base = render_native_prompt(stimulus, tokenizer=tokenizer, model_spec=model_spec)
    passage = str(stimulus["passage"])
    user_text = USER_TEMPLATE.format(passage=passage)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec),
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = tuple(_flat(encoded["input_ids"]))
    attention_mask = tuple(_flat(encoded.get("attention_mask", [1] * len(input_ids))))
    offsets = _offsets(encoded["offset_mapping"])
    passage_start = rendered.find(passage)
    if passage_start < 0 or rendered.find(passage, passage_start + 1) >= 0:
        raise RuntimeError("Controlled prompt must contain one exact passage")
    prompt_record_spans = []
    for record in stimulus["active_needle_spans"]:
        start, end = _token_span(
            offsets,
            passage_start + int(record["char_start"]),
            passage_start + int(record["char_end"]),
        )
        prompt_record_spans.append(
            {
                "slot_index": int(record["slot_index"]),
                "city": str(record["city"]),
                "entity": str(record.get("entity", record["city"])),
                "entity_domain": "city",
                "score": int(record["score"]),
                "start": int(start),
                "end": int(end),
                "kind": "active_prompt_record",
            }
        )
    return replace(
        base,
        user_text=user_text,
        rendered_prompt=rendered,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_record_spans=tuple(prompt_record_spans),
    )


def _is_exact_prefix_record(row: Mapping[str, Any]) -> bool:
    parsed = dict(row.get("trace_parse", {}))
    parser = dict(parsed.get("parser", {}))
    if not bool(parsed.get("exact_count")):
        return False
    if str(parser.get("trace_category")) != "one_to_one":
        return False
    episode = dict(parsed.get("episode_parse", {}))
    sequences = list(episode.get("sequences", ()))
    selected_index = int(episode.get("selected_sequence_index", -1))
    if 0 <= selected_index < len(sequences):
        events = list(dict(sequences[selected_index]).get("events", ()))
    else:
        events = list(episode.get("events", ()))
    if len(events) != 10:
        return False
    raw = str(row.get("raw_output_text", ""))
    gold = {str(value["city"]): int(value["score"]) for value in row["gold_records"]}
    for occurrence, raw_event in enumerate(events, start=1):
        event = dict(raw_event)
        if int(event.get("rank", -1)) != occurrence:
            return False
        # The frozen parser calls a prefix rank and city in the same semantic
        # line ``same_unit``; the causal registry expands this to
        # ``same_unit_rank_before_city``.
        if str(event.get("association")) != "same_unit":
            return False
        if str(event.get("evidence_family")) != "inline_count":
            return False
        city = str(event.get("city", ""))
        if city not in gold:
            return False
        # Freeze both the semantic grammar and its local Markdown shell:
        # ``*   Record k: (city, score)``.  Record k must be the selected rank
        # evidence, and both city and score occur after it within the same
        # completed semantic item.  This excludes suffix Count:k,
        # city-before-rank traces, plain Record lines, bold/italic variants,
        # and punctuation variants that otherwise create nuisance clusters.
        surface = str(event.get("evidence_surface", "")).strip()
        if surface != f"Record {occurrence}:":
            return False
        start = int(event["semantic_start_char"])
        end = int(event["semantic_end_char"])
        item = raw[start:end].strip()
        city_at = item.find(city)
        score_matches = list(re.finditer(rf"\b{gold[city]}\b", item))
        if city_at < 0 or not score_matches or score_matches[-1].start() <= city_at:
            return False
        exact_item = re.fullmatch(
            rf"\*   Record {occurrence}: \("
            rf"{re.escape(city)}, {gold[city]}\)",
            item,
        )
        if exact_item is None:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    stimuli = [
        row
        for path in args.stimuli
        for row in read_jsonl(path)
        if int(row.get("gold_count", -1)) == 10
    ]
    stimuli.sort(key=_candidate_order)
    if len({_key(row) for row in stimuli}) != len(stimuli):
        raise ValueError("Candidate stimuli contain duplicate keys")
    existing = read_jsonl(args.output) if args.output.is_file() else []
    existing_by_key = {_key(row): row for row in existing}
    if len(existing_by_key) != len(existing):
        raise ValueError("Existing generation file contains duplicate keys")
    accepted = [row for row in existing if _is_exact_prefix_record(row)]
    pending = [row for row in stimuli if _key(row) not in existing_by_key]

    model_spec = resolve_model_spec(MODEL)
    model, tokenizer, _adapter = load_registered_model(
        model_spec,
        cache_dir=str(args.cache_dir),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    decoding = _load_decoding(args.config)
    output_by_key = dict(existing_by_key)
    print(
        f"[Gemma prefix-record] pool={len(stimuli)} existing={len(existing)} "
        f"accepted={len(accepted)} target={args.target_accepted}",
        flush=True,
    )
    for index, stimulus in enumerate(pending, start=1):
        prompt = _controlled_prompt(stimulus, tokenizer, model_spec)
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
        generated["controlled_trace_grammar"] = "Record k: (city, score)"
        output_by_key[_key(stimulus)] = generated
        ordered = [
            output_by_key[_key(row)] for row in stimuli if _key(row) in output_by_key
        ]
        write_jsonl(args.output, ordered)
        accepted = [row for row in ordered if _is_exact_prefix_record(row)]
        print(
            f"[Gemma prefix-record] generated={index} seed={stimulus['seed']} "
            f"accepted={len(accepted)}/{args.target_accepted}",
            flush=True,
        )
        if len(accepted) >= args.target_accepted:
            break
    if len(accepted) < args.target_accepted:
        raise RuntimeError(f"Candidate pool exhausted at {len(accepted)}/{args.target_accepted}")

    accepted.sort(key=lambda row: _candidate_order(row))
    accepted = accepted[: args.target_accepted]
    rows = [
        {
            "model_label": MODEL,
            "request_id": str(row["request_id"]),
            "seed": int(row["seed"]),
            "source_split": str(row["split"]),
            "gold_count": 10,
            "grammar_class": "same_unit_rank_before_city",
            "marker_kind": "inline_count",
            "surface_family": "controlled_nested_bullet_record_prefix_bare_paren",
            "endpoint_family": "bare_closing_parenthesis_after_city_and_score",
        }
        for row in accepted
    ]
    write_jsonl(args.accepted_output, rows)
    result = {
        "schema_version": SCHEMA,
        "controlled_prompt": True,
        "surface_grammar": (
            "exact nested bullet '*   Record k: (city, score)'; "
            "every item ends in bare )"
        ),
        "candidate_order_salt": ORDER_SALT,
        "generated_rows": len(output_by_key),
        "accepted_rows": len(rows),
        "accepted_seeds": [row["seed"] for row in rows],
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
    parser.add_argument("--stimuli", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-accepted", type=int, default=30)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend")
    args = parser.parse_args()
    for path in (args.output, args.accepted_output, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
