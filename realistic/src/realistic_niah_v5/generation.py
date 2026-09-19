from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from realistic_niah.entity_domains import native_user_text, resolve_entity_domain
from realistic_niah_v4.spec import V4ModelSpec

from .parsing import parse_trace_record
from .spec import DecodingSpec


GENERATION_SCHEMA_VERSION = "realistic_niah_v5_generation_v1"

V5_USER_TEMPLATE = """\
You will need to count all city-score audit records in the passage below.
A city-score audit record names one city and gives that city's numeric score.

<passage>
{passage}
</passage>

How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: <integer>"""


@dataclass(frozen=True)
class NativePrompt:
    stimulus_id: str
    design_variant: str
    seed: int
    split: str
    gold_count: int
    model_label: str
    model_family: str
    entity_domain: str
    user_text: str
    rendered_prompt: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    gold_records: tuple[dict[str, Any], ...]
    prompt_record_spans: tuple[dict[str, Any], ...]

    @property
    def prompt_token_count(self) -> int:
        return len(self.input_ids)


def build_v5_user_text(passage: str, *, entity_domain: str = "city") -> str:
    domain = resolve_entity_domain(entity_domain)
    if domain.name == "city":
        # Preserve the preregistered city prompt byte-for-byte.
        return V5_USER_TEMPLATE.format(passage=str(passage))
    return native_user_text(str(passage), entity_domain=domain.name)


def _chat_template_kwargs(model_spec: V4ModelSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_spec.chat_template_control == "enable_thinking_kwarg":
        kwargs["enable_thinking"] = True
    elif model_spec.chat_template_control == "thinking_kwarg":
        kwargs["thinking"] = True
    else:
        raise ValueError(
            "V5 requires a registered native-thinking chat-template control, got "
            f"{model_spec.chat_template_control!r}"
        )
    if getattr(model_spec, "system_prompt_strategy", "none") != "none":
        raise ValueError("V5 does not inject an unregistered system prompt")
    return kwargs


def _flat(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("V5 generation requires batch size one")
        values = values[0]
    return [int(value) for value in values]


def _offsets(values: Any) -> list[tuple[int, int]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        if len(values) != 1:
            raise ValueError("V5 prompt offsets require batch size one")
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


def render_native_prompt(
    stimulus: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_spec: V4ModelSpec,
) -> NativePrompt:
    if str(stimulus.get("design_variant")) != "v4.4":
        raise ValueError("V5 is fixed to frozen V4.4 stimuli")
    passage = str(stimulus["passage"])
    entity_domain = resolve_entity_domain(stimulus.get("entity_domain"))
    user_text = build_v5_user_text(
        passage,
        entity_domain=entity_domain.name,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        **_chat_template_kwargs(model_spec),
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Chat template did not return V5 prompt text")
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("V5 targeted retrieval requires fast-tokenizer offsets")
    input_ids = tuple(_flat(encoded["input_ids"]))
    offsets = _offsets(encoded["offset_mapping"])
    attention_mask = tuple(
        _flat(encoded.get("attention_mask", [1] * len(input_ids)))
    )
    if not input_ids or len(attention_mask) != len(input_ids):
        raise RuntimeError("V5 rendered prompt tokenization is invalid")
    if len(offsets) != len(input_ids):
        raise RuntimeError("V5 prompt IDs and offsets have different lengths")
    gold = tuple(dict(value) for value in stimulus["gold_pairs"])
    if len(gold) != int(stimulus["gold_count"]):
        raise RuntimeError("Frozen stimulus gold pairs/count disagree")
    active_spans = list(stimulus.get("active_needle_spans", ()))
    if len(active_spans) != int(stimulus["gold_count"]):
        raise RuntimeError(
            "Frozen V4.4 active_needle_spans are required for targeted retrieval"
        )
    passage_start = rendered.find(passage)
    if passage_start < 0 or rendered.find(passage, passage_start + 1) >= 0:
        raise RuntimeError("Rendered native prompt must contain one exact passage")
    prompt_record_spans = []
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
                "entity": str(record.get("entity", record["city"])),
                "entity_domain": entity_domain.name,
                "score": int(record["score"]),
                "start": int(start),
                "end": int(end),
                "kind": "active_prompt_record",
            }
        )
    family = "qwen3" if "qwen" in model_spec.label.lower() else "gemma4"
    return NativePrompt(
        stimulus_id=str(stimulus["stimulus_id"]),
        design_variant="v4.4",
        seed=int(stimulus["seed"]),
        split=str(stimulus["split"]),
        gold_count=int(stimulus["gold_count"]),
        model_label=model_spec.label,
        model_family=family,
        entity_domain=entity_domain.name,
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
def generate_native_trace(
    model: Any,
    tokenizer: Any,
    prompt: NativePrompt,
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
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(decoding.max_new_tokens),
        "do_sample": bool(decoding.do_sample),
        "use_cache": True,
    }
    if decoding.do_sample:
        generation_kwargs.update(
            {
                "temperature": float(decoding.temperature),
                "top_p": float(decoding.top_p),
                "top_k": int(decoding.top_k),
            }
        )
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    started = time.perf_counter()
    generated = model.generate(**generation_kwargs)
    elapsed = time.perf_counter() - started
    sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
    continuation = tuple(
        int(value)
        for value in sequences[0, len(prompt.input_ids) :].detach().cpu().tolist()
    )
    if not continuation:
        raise RuntimeError("V5 generation returned an empty continuation")
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
    row: dict[str, Any] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "request_id": (
            f"{prompt.model_label}/native_thinking/v5/"
            f"{prompt.stimulus_id}"
        ),
        "stimulus_id": prompt.stimulus_id,
        "design_variant": prompt.design_variant,
        "model_label": prompt.model_label,
        "model_family": prompt.model_family,
        "entity_domain": prompt.entity_domain,
        "prompt_mode": "native_thinking",
        "seed": prompt.seed,
        "split": prompt.split,
        "gold_count": prompt.gold_count,
        "gold_records": list(prompt.gold_records),
        "prompt_record_spans": list(prompt.prompt_record_spans),
        "user_text": prompt.user_text,
        "rendered_prompt": prompt.rendered_prompt,
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
        "generation_truncated": (
            len(continuation) >= decoding.max_new_tokens and not stopped_on_eos
        ),
        "elapsed_seconds": elapsed,
    }
    row["trace_parse"] = parse_trace_record(row)
    return row
