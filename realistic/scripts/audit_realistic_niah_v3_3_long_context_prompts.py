from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from realistic_niah.prompts import (
    COMMON_COUNTING_CUE,
    build_messages,
    query_block,
    render_generation_prompt,
)
from realistic_niah.runner import decoding_config, request_id
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3_1.spec import MODEL_SPECS
from realistic_niah_v3_3_long_context.integrity import (
    sha256_file,
    validate_frozen_dataset,
)
from realistic_niah_v3_3_long_context.spec import (
    EXPECTED_REQUESTS,
    FORMAL_PROMPT_MODES,
    MAX_MODEL_LEN,
    MODEL_IDS,
    MODEL_LABELS,
    MODEL_REVISIONS,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    PREFLIGHT_LENGTH,
    PREFLIGHT_NEEDLE_COUNT,
    PREFLIGHT_SEED,
    PROTOCOL_VERSION,
    SEEDS,
    V33_LONG_CONTEXT_RUN_PROTOCOL,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _expected_messages(passage: str, mode: str) -> list[dict[str, str]]:
    content = (
        f"{COMMON_COUNTING_CUE}\n\n"
        f"<passage>\n{passage}\n</passage>\n\n"
        f"{query_block(mode)}"
    )
    return [{"role": "user", "content": content}]


def audit_prompts(
    *,
    dataset_dir: str | Path,
    output_path: str | Path,
    examples_path: str | Path,
    cache_dir: str | Path | None = None,
    expected_dataset_seal_sha256: str | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    stimuli_path = dataset / "stimuli.jsonl"
    dataset_audit = validate_frozen_dataset(
        dataset,
        expected_seal_sha256=expected_dataset_seal_sha256,
    )
    request_ids: set[str] = set()
    request_digest = hashlib.sha256()
    messages_digest = hashlib.sha256()
    maximum_rows: list[dict[str, Any]] = []
    preflight_row: dict[str, Any] | None = None
    observed_stimuli = 0

    with stimuli_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            observed_stimuli += 1
            length = int(row["target_passage_tokens"])
            count = int(row["num_needles"])
            seed = int(row["seed"])
            if (
                row.get("protocol_version") != PROTOCOL_VERSION
                or length not in PASSAGE_LENGTHS
                or count not in NEEDLE_COUNTS
                or seed not in SEEDS
            ):
                raise RuntimeError(f"Unregistered stimulus at line {line_number}")
            passage = str(row["passage"])
            if length == PREFLIGHT_LENGTH:
                maximum_rows.append(row)
            if (
                length == PREFLIGHT_LENGTH
                and count == PREFLIGHT_NEEDLE_COUNT
                and seed == PREFLIGHT_SEED
            ):
                preflight_row = row

            for model_label in MODEL_LABELS:
                model_spec = MODEL_SPECS[model_label]
                for mode in FORMAL_PROMPT_MODES:
                    messages = build_messages(
                        passage,
                        prompt_mode=mode,
                        query_layout=QUERY_LAYOUT,
                    )
                    if messages != _expected_messages(passage, mode):
                        raise RuntimeError(
                            f"Prompt contract mismatch at line {line_number}: "
                            f"model={model_label}, mode={mode}"
                        )
                    identifier = request_id(
                        model_spec=model_spec,
                        prompt_mode=mode,
                        query_layout=QUERY_LAYOUT,
                        stimulus_id=str(row["stimulus_id"]),
                        namespace=(
                            V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace
                        ),
                    )
                    if identifier in request_ids:
                        raise RuntimeError(f"Duplicate request ID: {identifier}")
                    request_ids.add(identifier)
                    request_digest.update(f"{identifier}\n".encode("utf-8"))
                    messages_digest.update(
                        json.dumps(
                            messages,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    messages_digest.update(b"\n")

    if observed_stimuli != 3_780 or len(request_ids) != EXPECTED_REQUESTS:
        raise RuntimeError(
            "Prompt request accounting failed: "
            f"stimuli={observed_stimuli}, requests={len(request_ids)}"
        )
    if len(maximum_rows) != len(NEEDLE_COUNTS) * len(SEEDS):
        raise RuntimeError("Maximum-length audit subset is incomplete")
    if preflight_row is None:
        raise RuntimeError("Registered maximum-context preflight row is missing")

    render_audits: list[dict[str, Any]] = []
    prompt_examples: list[dict[str, Any]] = []
    for model_label in MODEL_LABELS:
        model_spec = MODEL_SPECS[model_label]
        revision = MODEL_REVISIONS[model_label]
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_IDS[model_label],
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        for mode in FORMAL_PROMPT_MODES:
            decode = decoding_config(model_spec, mode)
            minimum_input_tokens: int | None = None
            maximum_input_tokens = -1
            maximum_total_budget = -1
            maximum_stimulus_id = ""
            for row in maximum_rows:
                messages = build_messages(
                    str(row["passage"]),
                    prompt_mode=mode,
                    query_layout=QUERY_LAYOUT,
                )
                rendered = render_generation_prompt(
                    tokenizer,
                    messages,
                    model_spec=model_spec,
                    prompt_mode=mode,
                )
                input_tokens = len(
                    tokenizer.encode(rendered, add_special_tokens=False)
                )
                total_budget = input_tokens + decode.max_tokens
                if total_budget > MAX_MODEL_LEN:
                    raise RuntimeError(
                        f"Prompt exceeds engine budget: model={model_label}, "
                        f"mode={mode}, stimulus={row['stimulus_id']}, "
                        f"total={total_budget}, limit={MAX_MODEL_LEN}"
                    )
                minimum_input_tokens = (
                    input_tokens
                    if minimum_input_tokens is None
                    else min(minimum_input_tokens, input_tokens)
                )
                if input_tokens > maximum_input_tokens:
                    maximum_input_tokens = input_tokens
                    maximum_total_budget = total_budget
                    maximum_stimulus_id = str(row["stimulus_id"])
                if row is preflight_row:
                    prompt_examples.append(
                        {
                            "schema_version": (
                                "realistic_niah_prompt_example_v3_3_long_context"
                            ),
                            "protocol_version": PROTOCOL_VERSION,
                            "model_label": model_label,
                            "model_id": MODEL_IDS[model_label],
                            "model_revision": revision,
                            "prompt_mode": mode,
                            "stimulus_id": str(row["stimulus_id"]),
                            "input_tokens": input_tokens,
                            "max_output_tokens": decode.max_tokens,
                            "total_budget": total_budget,
                            "rendered_prompt": rendered,
                        }
                    )
            render_audits.append(
                {
                    "model_label": model_label,
                    "model_id": MODEL_IDS[model_label],
                    "model_revision": revision,
                    "prompt_mode": mode,
                    "rows_checked": len(maximum_rows),
                    "minimum_input_tokens": minimum_input_tokens,
                    "maximum_input_tokens": maximum_input_tokens,
                    "maximum_output_tokens": decode.max_tokens,
                    "maximum_total_budget": maximum_total_budget,
                    "maximum_stimulus_id": maximum_stimulus_id,
                    "engine_max_model_len": MAX_MODEL_LEN,
                    "passed": True,
                }
            )

    _atomic_jsonl(Path(examples_path).resolve(), prompt_examples)
    audit = {
        "schema_version": "realistic_niah_prompt_audit_v3_3_long_context",
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "dataset_seal_sha256": dataset_audit["seal_sha256"],
        "stimuli": observed_stimuli,
        "requests": len(request_ids),
        "unique_request_ids": len(request_ids),
        "request_ids_sha256": request_digest.hexdigest(),
        "messages_sha256": messages_digest.hexdigest(),
        "prompt_examples": len(prompt_examples),
        "prompt_examples_path": str(Path(examples_path).resolve()),
        "maximum_length_rows_checked_per_model_mode": len(maximum_rows),
        "render_audits": render_audits,
    }
    _atomic_json(Path(output_path).resolve(), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every V3.3 prompt contract and tokenize every 100k prompt."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--expected-dataset-seal-sha256")
    args = parser.parse_args()
    audit = audit_prompts(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        examples_path=args.examples,
        cache_dir=args.cache_dir,
        expected_dataset_seal_sha256=args.expected_dataset_seal_sha256,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
