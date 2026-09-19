from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.response_eval import parse_model_output

THINK_END_MARKER = "</think>"


def _flat_ids(value: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in value]


def _token_prefix_char_lengths(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    lengths: list[int] = []
    for i in range(1, len(token_ids) + 1):
        lengths.append(len(tokenizer.decode(list(token_ids[:i]), skip_special_tokens=False)))
    return lengths


def _token_index_at_or_after_char(tokenizer: Any, token_ids: Sequence[int], char_pos: int) -> int:
    if char_pos <= 0:
        return 0
    for idx, prefix_len in enumerate(_token_prefix_char_lengths(tokenizer, token_ids), start=1):
        if prefix_len >= char_pos:
            return idx
    return len(token_ids)


def locate_final_answer_start(
    generated_ids: torch.Tensor | Sequence[int],
    tokenizer: Any,
    *,
    task_type: str,
) -> tuple[int, dict[str, Any]]:
    """Locate the generated-token index where the final JSON answer begins.

    The preferred Qwen3 thinking-mode path starts the search after the final
    ``</think>`` marker. If the marker is absent or no schema JSON is found after
    it, fall back to a token-by-token schema parse.
    """

    ids = _flat_ids(generated_ids)
    full_text = tokenizer.decode(ids, skip_special_tokens=False)
    marker_idx = full_text.rfind(THINK_END_MARKER)
    marker_token_end = 0
    used_marker = False
    if marker_idx >= 0:
        marker_token_end = _token_index_at_or_after_char(
            tokenizer, ids, marker_idx + len(THINK_END_MARKER)
        )
        used_marker = True

    def scan(start: int) -> int | None:
        for token_idx in range(max(0, int(start)), len(ids)):
            suffix = tokenizer.decode(ids[token_idx:], skip_special_tokens=True)
            if not suffix.lstrip().startswith("{"):
                continue
            parsed = parse_model_output(suffix, task_type)
            if parsed.get("parse_mode") != "parse_fail" and parsed.get("count") is not None:
                return token_idx
        return None

    answer_start = scan(marker_token_end)
    fallback_used = not used_marker
    if answer_start is None:
        answer_start = scan(0)
        fallback_used = True
    if answer_start is None:
        answer_start = len(ids)
        fallback_used = True

    return answer_start, {
        "thinking_end_marker_found": used_marker,
        "thinking_end_marker_token_end": int(marker_token_end),
        "fallback_parse_used": bool(fallback_used),
        "final_answer_found": int(answer_start) < len(ids),
    }


def split_generated_cot(
    generated_ids: torch.Tensor | Sequence[int],
    tokenizer: Any,
    *,
    task_type: str,
    eos_token_id: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = _flat_ids(generated_ids)
    eos_positions = [i for i, tok in enumerate(ids) if eos_token_id is not None and int(tok) == int(eos_token_id)]
    eos_index = eos_positions[0] if eos_positions else None
    decode_limit = eos_index if eos_index is not None else len(ids)
    answer_start, locate_info = locate_final_answer_start(ids[:decode_limit], tokenizer, task_type=task_type)
    reasoning_ids = ids[:answer_start]
    final_answer_ids = ids[answer_start:decode_limit]
    eos_ids = [] if eos_index is None else [ids[eos_index]]
    hit_max_without_eos = eos_index is None and len(ids) >= int(max_new_tokens)
    return {
        "reasoning_ids": reasoning_ids,
        "final_answer_ids": final_answer_ids,
        "eos_ids": eos_ids,
        "answer_start_generated_index": int(answer_start),
        "eos_index": None if eos_index is None else int(eos_index),
        "eos_hit": eos_index is not None,
        "hit_max_new_tokens_without_eos": bool(hit_max_without_eos),
        **locate_info,
    }


def cot_tensor_path(tensors_dir: str | Path, example_id: int) -> Path:
    return Path(tensors_dir) / f"inputs_cot_{int(example_id)}.pt"


def cot_info_path(tables_dir: str | Path) -> Path:
    return Path(tables_dir) / "cot_info.json"


def save_cot_generation_artifacts(
    *,
    tensors_dir: str | Path,
    tables_dir: str | Path,
    example_id: int,
    row: dict[str, Any],
    prompt_input_ids: torch.Tensor,
    generated_ids: torch.Tensor | Sequence[int],
    tokenizer: Any,
    task_type: str,
    max_new_tokens: int,
    prompt_text: str,
) -> dict[str, Any]:
    tensors_dir = Path(tensors_dir)
    tables_dir = Path(tables_dir)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = _flat_ids(prompt_input_ids)
    gen_ids = _flat_ids(generated_ids)
    split = split_generated_cot(
        gen_ids,
        tokenizer,
        task_type=task_type,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        max_new_tokens=max_new_tokens,
    )
    extended_ids = prompt_ids + split["reasoning_ids"]
    payload = {
        "schema_version": "counting_cot_inputs_v1",
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "task_type": task_type,
        "input_ids": torch.tensor([extended_ids], dtype=torch.long),
        "prompt_input_ids": torch.tensor([prompt_ids], dtype=torch.long),
        "generated_ids": torch.tensor([gen_ids], dtype=torch.long),
        "reasoning_ids": torch.tensor([split["reasoning_ids"]], dtype=torch.long),
        "final_answer_ids": torch.tensor([split["final_answer_ids"]], dtype=torch.long),
        "eos_ids": torch.tensor([split["eos_ids"]], dtype=torch.long),
        "prompt_tokens": len(prompt_ids),
        "reasoning_tokens_before_final_answer": len(split["reasoning_ids"]),
        "final_answer_tokens": len(split["final_answer_ids"]),
        "eos_tokens": len(split["eos_ids"]),
        "total_new_tokens": len(gen_ids),
        "extended_input_tokens": len(extended_ids),
        "max_new_tokens": int(max_new_tokens),
        **{k: v for k, v in split.items() if not k.endswith("_ids")},
    }
    tensor_path = cot_tensor_path(tensors_dir, example_id)
    torch.save(payload, tensor_path)

    text_path = tables_dir / f"input_generate_cot_{int(example_id)}.txt"
    text_path.write_text(
        "PROMPT\n======\n"
        + prompt_text
        + "\n\nREASONING BEFORE FINAL ANSWER\n=============================\n"
        + tokenizer.decode(split["reasoning_ids"], skip_special_tokens=False)
        + "\n\nFINAL ANSWER\n============\n"
        + tokenizer.decode(split["final_answer_ids"], skip_special_tokens=False)
        + "\n",
        encoding="utf-8",
    )

    info = {
        k: v
        for k, v in payload.items()
        if not isinstance(v, torch.Tensor)
    } | {"tensor_path": str(tensor_path), "text_path": str(text_path)}
    return info


def write_cot_info(infos: Sequence[dict[str, Any]], tables_dir: str | Path) -> Path:
    path = cot_info_path(tables_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(info["example_id"]): dict(info) for info in infos}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_cot_payload(tensors_dir: str | Path, example_id: int) -> dict[str, Any]:
    path = cot_tensor_path(tensors_dir, example_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing CoT extended input tensor: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "input_ids" not in payload:
        raise ValueError(f"Invalid CoT tensor payload at {path}")
    return payload


def build_controlled_extended_input(
    *,
    tokenizer: Any,
    controlled_messages: list[dict[str, str]],
    thinking_mode: bool,
    reasoning_ids: torch.Tensor | Sequence[int],
) -> tuple[torch.Tensor, str]:
    controlled_text = apply_generation_chat_template(
        tokenizer, controlled_messages, thinking_mode=thinking_mode
    )
    encoded = tokenizer(controlled_text, return_tensors="pt")
    controlled_prompt_ids = _flat_ids(encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids)
    extended = controlled_prompt_ids + _flat_ids(reasoning_ids)
    return torch.tensor([extended], dtype=torch.long), controlled_text
