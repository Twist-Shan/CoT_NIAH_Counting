#!/usr/bin/env python3
"""Inspect the literal baseline token immediately before the numeric answer."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.spec import resolve_model_spec


TOTAL_ANSWER = re.compile(r"(?is)(?<!\w)Total\s*:\s*(?P<answer>[+-]?\d+)")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    spec = resolve_model_spec(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    rows = read_jsonl(args.generations)
    row = next(
        row
        for row in rows
        if str(row.get("model_label")) == args.model
        and bool(row.get("trace_parse", {}).get("parser", {}).get("trace_one_to_one"))
    )
    ids = [int(value) for value in row["output_token_ids"]]
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    matches = list(TOTAL_ANSWER.finditer(decoded))
    if not matches:
        raise RuntimeError("No decoded Total: <integer> answer found")
    answer_start = matches[-1].start("answer")
    numeric_prefix_len = None
    previous_text = ""
    for prefix_len in range(1, len(ids) + 1):
        text = tokenizer.decode(
            ids[:prefix_len],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if len(text) > answer_start and any(
            char.isdigit() for char in text[answer_start:]
        ):
            numeric_prefix_len = prefix_len
            break
        previous_text = text
    if numeric_prefix_len is None or numeric_prefix_len < 2:
        raise RuntimeError("Cannot isolate the first numeric answer token")
    numeric_index = numeric_prefix_len - 1
    query_index = numeric_index - 1
    window = []
    for index in range(max(0, query_index - 3), min(len(ids), numeric_index + 4)):
        window.append(
            {
                "output_index": index,
                "token_id": ids[index],
                "token_text": tokenizer.decode(
                    [ids[index]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
    payload = {
        "model": args.model,
        "request_id": row.get("request_id"),
        "decoded_tail": decoded[max(0, answer_start - 40) : answer_start + 30],
        "answer_char_start": answer_start,
        "numeric_token_output_index": numeric_index,
        "query_token_output_index": query_index,
        "query_token_id": ids[query_index],
        "query_token_text": tokenizer.decode(
            [ids[query_index]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "numeric_token_id": ids[numeric_index],
        "numeric_token_text": tokenizer.decode(
            [ids[numeric_index]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "decoded_prefix_before_numeric": previous_text[-80:],
        "window": window,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
