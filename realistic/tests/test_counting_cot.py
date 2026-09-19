from pathlib import Path

import torch

from counting.cot import (
    locate_final_answer_start,
    save_cot_generation_artifacts,
    split_generated_cot,
    write_cot_info,
)


class FakeTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().reshape(-1).tolist()
        text = "".join(chr(int(i)) for i in ids)
        if skip_special_tokens:
            text = text.replace("\x00", "")
        return text


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_locate_final_answer_start_prefers_think_marker() -> None:
    tokenizer = FakeTokenizer()
    generated = _ids('reasoning with {not json}</think>\n{"count":3}\x00')

    start, info = locate_final_answer_start(generated, tokenizer, task_type="match_count")

    assert tokenizer.decode(generated[start:], skip_special_tokens=True).lstrip().startswith('{"count":3}')
    assert info["thinking_end_marker_found"] is True
    assert info["final_answer_found"] is True


def test_locate_final_answer_start_falls_back_to_schema_parse() -> None:
    tokenizer = FakeTokenizer()
    generated = _ids('reasoning first\n{"count":2}\x00')

    start, info = locate_final_answer_start(generated, tokenizer, task_type="literal_count")

    assert tokenizer.decode(generated[start:], skip_special_tokens=True).lstrip().startswith('{"count":2}')
    assert info["thinking_end_marker_found"] is False
    assert info["fallback_parse_used"] is True


def test_split_generated_cot_records_missing_eos_at_max() -> None:
    tokenizer = FakeTokenizer()
    generated = _ids('</think>{"count":1}')

    split = split_generated_cot(
        generated,
        tokenizer,
        task_type="match_count",
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=len(generated),
    )

    assert split["eos_hit"] is False
    assert split["hit_max_new_tokens_without_eos"] is True
    assert split["final_answer_ids"] == _ids('{"count":1}')


def test_save_cot_generation_artifacts_writes_tensor_info_and_text(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    info = save_cot_generation_artifacts(
        tensors_dir=tmp_path / "tensors",
        tables_dir=tmp_path / "tables",
        example_id=0,
        row={"id": "row0"},
        prompt_input_ids=torch.tensor([_ids("prompt")], dtype=torch.long),
        generated_ids=torch.tensor(_ids("think</think>{\"count\":4}\x00"), dtype=torch.long),
        tokenizer=tokenizer,
        task_type="match_count",
        max_new_tokens=64,
        prompt_text="prompt",
    )
    path = write_cot_info([info], tmp_path / "tables")

    payload = torch.load(tmp_path / "tensors" / "inputs_cot_0.pt", map_location="cpu")
    assert payload["prompt_tokens"] == len("prompt")
    assert payload["extended_input_tokens"] == len("prompt") + len("think</think>")
    assert payload["final_answer_tokens"] == len('{"count":4}')
    assert path.exists()
    assert (tmp_path / "tables" / "input_generate_cot_0.txt").exists()
