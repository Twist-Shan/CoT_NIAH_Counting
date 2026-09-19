from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from realistic_niah.parsing import evaluate_generation
from realistic_niah_v4.spec import V4ModelSpec

from .parsing import parse_trace_record
from .spec import DecodingSpec, GENERATION_SCHEMA_VERSION, PROMPT_MODES


COMMON_COUNTING_CUE = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score."""

ENUMERATION_QUERY_BLOCKS = {
    "enumeration_index": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin the first item with "1. ", the second with "2. ", and continue with ordinary digits.
After each number, write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.""",
    "enumeration_bullet": """\
How many city-score audit records are in the passage?
List each occurrence once, in passage order.
Begin each item with "-", then write the actual city name, then ": ", then the actual numeric score.
Use only actual values from the passage; do not output placeholders or angle brackets.
Then report the number listed:
Total: <integer>
Do not include any other text.""",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructuredEnumerationPrompt:
    stimulus_id: str
    design_variant: str
    seed: int
    split: str
    gold_count: int
    model_label: str
    model_family: str
    model_id: str
    model_revision: str
    model_loader_class: str
    chat_template_control: str
    prompt_mode: str
    user_text: str
    rendered_prompt: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    gold_records: tuple[dict[str, Any], ...]
    prompt_record_spans: tuple[dict[str, Any], ...]

    @property
    def prompt_token_count(self) -> int:
        return len(self.input_ids)


def build_v6_user_text(passage: str, *, prompt_mode: str) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported V6 prompt mode: {prompt_mode!r}")
    passage_block = f"<passage>\n{passage}\n</passage>"
    return (
        f"{COMMON_COUNTING_CUE}\n\n{passage_block}\n\n"
        f"{ENUMERATION_QUERY_BLOCKS[prompt_mode]}"
    )


def _chat_template_kwargs(model_spec: V4ModelSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_spec.chat_template_control == "enable_thinking_kwarg":
        kwargs["enable_thinking"] = False
    elif model_spec.chat_template_control == "thinking_kwarg":
        kwargs["thinking"] = False
    else:
        raise ValueError(
            "V6 requires an explicit switchable thinking control, got "
            f"{model_spec.chat_template_control!r}"
        )
    return kwargs


def _flat(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("V6 generation requires batch size one")
        values = values[0]
    return [int(value) for value in values]


def _offsets(values: Any) -> list[tuple[int, int]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(
        values[0][0], list
    ):
        if len(values) != 1:
            raise ValueError("V6 prompt offsets require batch size one")
        values = values[0]
    return [(int(start), int(end)) for start, end in values]


def _token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < int(char_end) and end > int(char_start)
    ]
    if not indices:
        raise RuntimeError(
            f"No prompt tokens overlap record chars [{char_start}, {char_end})"
        )
    return indices[0], indices[-1] + 1


def render_structured_prompt(
    stimulus: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_spec: V4ModelSpec,
    prompt_mode: str,
) -> StructuredEnumerationPrompt:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported V6 prompt mode: {prompt_mode!r}")
    if str(stimulus.get("design_variant")) != "v4.4":
        raise ValueError("V6 is fixed to frozen V4.4 stimuli")
    passage = str(stimulus["passage"])
    user_text = build_v6_user_text(passage, prompt_mode=prompt_mode)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec),
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Chat template did not return a V6 prompt string")
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("V6 requires fast-tokenizer offset mappings")
    input_ids = tuple(_flat(encoded["input_ids"]))
    attention_mask = tuple(
        _flat(encoded.get("attention_mask", [1] * len(input_ids)))
    )
    offsets = _offsets(encoded["offset_mapping"])
    if not input_ids or len(attention_mask) != len(input_ids):
        raise RuntimeError("V6 rendered prompt tokenization is invalid")
    if len(offsets) != len(input_ids):
        raise RuntimeError("V6 prompt IDs and offsets have different lengths")

    gold = tuple(dict(value) for value in stimulus["gold_pairs"])
    active_spans = tuple(dict(value) for value in stimulus["active_needle_spans"])
    gold_count = int(stimulus["gold_count"])
    if len(gold) != gold_count or len(active_spans) != gold_count:
        raise RuntimeError("Frozen stimulus gold records/count disagree")
    passage_start = rendered.find(passage)
    if passage_start < 0 or rendered.find(passage, passage_start + 1) >= 0:
        raise RuntimeError("Rendered V6 prompt must contain one exact passage")
    prompt_record_spans: list[dict[str, Any]] = []
    for record in active_spans:
        start, end = _token_span(
            offsets,
            passage_start + int(record["char_start"]),
            passage_start + int(record["char_end"]),
        )
        prompt_record_spans.append(
            {
                "slot_index": int(record["slot_index"]),
                "city": str(record["city"]),
                "score": int(record["score"]),
                "start": int(start),
                "end": int(end),
                "kind": "active_prompt_record",
            }
        )
    return StructuredEnumerationPrompt(
        stimulus_id=str(stimulus["stimulus_id"]),
        design_variant="v4.4",
        seed=int(stimulus["seed"]),
        split=str(stimulus["split"]),
        gold_count=gold_count,
        model_label=model_spec.label,
        model_family=str(model_spec.family),
        model_id=str(model_spec.model_id),
        model_revision=str(model_spec.revision),
        model_loader_class=str(model_spec.loader_class),
        chat_template_control=str(model_spec.chat_template_control),
        prompt_mode=prompt_mode,
        user_text=user_text,
        rendered_prompt=rendered,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gold_records=gold,
        prompt_record_spans=tuple(prompt_record_spans),
    )


def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        return [int(item) for item in value]
    return [int(value)]


@torch.inference_mode()
def generate_structured_enumeration(
    model: Any,
    tokenizer: Any,
    prompt: StructuredEnumerationPrompt,
    *,
    decoding: DecodingSpec,
    sampling_seed: int,
) -> dict[str, Any]:
    decoding.validate()
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [prompt.attention_mask], dtype=torch.long, device=device
    )
    torch.manual_seed(int(sampling_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(sampling_seed))
    eos_ids = _eos_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = eos_ids[0]
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(decoding.max_new_tokens),
        "do_sample": bool(decoding.do_sample),
        "use_cache": True,
    }
    if decoding.do_sample:
        kwargs.update(
            temperature=float(decoding.temperature),
            top_p=float(decoding.top_p),
            top_k=int(decoding.top_k),
        )
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    started = time.perf_counter()
    generated = model.generate(**kwargs)
    elapsed = time.perf_counter() - started
    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    continuation = tuple(
        int(value)
        for value in sequences[0, len(prompt.input_ids) :].detach().cpu().tolist()
    )
    if not continuation:
        raise RuntimeError("V6 generation returned an empty continuation")
    raw = tokenizer.decode(
        continuation,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    stopped_on_eos = bool(eos_ids and continuation[-1] in set(eos_ids))
    truncated = len(continuation) >= decoding.max_new_tokens and not stopped_on_eos
    finish_reason = "length" if truncated else "stop"
    evaluation = evaluate_generation(
        raw,
        prompt_mode=prompt.prompt_mode,
        reasoning_expected=False,
        gold_pairs=list(prompt.gold_records),
        finish_reason=finish_reason,
        output_tokens=len(continuation),
        max_output_tokens=int(decoding.max_new_tokens),
    )
    row: dict[str, Any] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "protocol_version": "realistic_niah_v6_structured_enumeration_replication_v1",
        "request_id": (
            f"{prompt.model_label}/{prompt.prompt_mode}/v6/{prompt.stimulus_id}"
        ),
        "stimulus_id": prompt.stimulus_id,
        "design_variant": prompt.design_variant,
        "model_label": prompt.model_label,
        "model_family": prompt.model_family,
        "model_id": prompt.model_id,
        "model_revision": prompt.model_revision,
        "model_loader_class": prompt.model_loader_class,
        "chat_template_control": prompt.chat_template_control,
        "entity_domain": "city",
        "prompt_mode": prompt.prompt_mode,
        "native_thinking": False,
        "chat_template_thinking_enabled": False,
        "seed": prompt.seed,
        "split": prompt.split,
        "gold_count": prompt.gold_count,
        "gold_records": list(prompt.gold_records),
        "prompt_record_spans": list(prompt.prompt_record_spans),
        "user_text": prompt.user_text,
        "user_text_sha256": _sha256_text(prompt.user_text),
        "rendered_prompt": prompt.rendered_prompt,
        "rendered_prompt_sha256": _sha256_text(prompt.rendered_prompt),
        "input_ids": list(prompt.input_ids),
        "attention_mask": list(prompt.attention_mask),
        "prompt_token_count": prompt.prompt_token_count,
        "output_token_ids": list(continuation),
        "output_tokens": len(continuation),
        "raw_output_text": raw,
        "clean_output_text": clean,
        "sampling_seed": int(sampling_seed),
        "decoding": asdict(decoding),
        "generation_eos_token_ids": eos_ids,
        "stopped_on_eos": stopped_on_eos,
        "generation_truncated": truncated,
        "finish_reason": finish_reason,
        "elapsed_seconds": elapsed,
        "generation_eval": evaluation,
        "generation_contract_sha256": hashlib.sha256(
            json.dumps(
                {
                    "prompt_mode": prompt.prompt_mode,
                    "thinking": False,
                    "decoding": asdict(decoding),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    row["trace_parse"] = parse_trace_record(row)
    return row
