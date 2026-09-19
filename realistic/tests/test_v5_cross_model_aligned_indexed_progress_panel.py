from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.build_realistic_niah_v5_cross_model_aligned_indexed_progress_panel import (
    build_panel,
)


class FakeTokenizer:
    """Lossless character tokenizer sufficient for cohort unit tests."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(value) for value in text]

    def decode(
        self,
        ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in ids)


def _row(model: str, seed: int) -> dict:
    records = [
        {"city": f"City{index}", "score": 50 + index, "slot_index": index}
        for index in range(1, 11)
    ]
    return {
        "request_id": f"old/{model}/{seed}",
        "stimulus_id": f"V4_4_T10000_N10_seed{seed}",
        "design_variant": "v4.4",
        "model_label": model,
        "model_family": "qwen3" if model == "Qwen3-8B" else "gemma4",
        "prompt_mode": "native_thinking",
        "seed": seed,
        "split": "discovery",
        "gold_count": 10,
        "gold_records": records,
        "rendered_prompt": "prompt",
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "generation_eos_token_ids": [],
    }


def test_cross_model_indexed_panel_has_exact_keys_and_surface() -> None:
    seeds = (1234, 1254)
    tokenizer = FakeTokenizer()
    output, shared = build_panel(
        qwen_rows=[_row("Qwen3-8B", seed) for seed in seeds],
        gemma_rows=[_row("Gemma4-E4B", seed) for seed in seeds],
        qwen_tokenizer=tokenizer,
        gemma_tokenizer=tokenizer,
        seeds=seeds,
        gold_count=10,
    )
    assert [(row["phase"], row["seed"], row["gold_count"]) for row in shared] == [
        ("discovery", 1234, 10),
        ("confirmation", 1254, 10),
    ]
    qwen_keys = [
        (row["split"], row["seed"], row["gold_count"])
        for row in output["Qwen3-8B"]
    ]
    gemma_keys = [
        (row["split"], row["seed"], row["gold_count"])
        for row in output["Gemma4-E4B"]
    ]
    assert qwen_keys == gemma_keys
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for row in output[model]:
            audit = row["indexed_progress_control_format_audit"]
            assert audit["primary_eligible_indexed_positive_control"] is True
            assert tuple(row["trace_parse"]["parser"]["item_markers"]) == tuple(
                range(1, 11)
            )
    assert (
        output["Qwen3-8B"][0]["controlled_indexed_surface_text"]
        == output["Gemma4-E4B"][0]["controlled_indexed_surface_text"]
    )


def test_cross_model_indexed_panel_rejects_stimulus_mismatch() -> None:
    qwen = _row("Qwen3-8B", 1234)
    gemma = _row("Gemma4-E4B", 1234)
    gemma["gold_records"][0]["score"] = 999
    tokenizer = FakeTokenizer()
    try:
        build_panel(
            qwen_rows=[qwen],
            gemma_rows=[gemma],
            qwen_tokenizer=tokenizer,
            gemma_tokenizer=tokenizer,
            seeds=(1234,),
            gold_count=10,
        )
    except ValueError as error:
        assert "stimulus mismatch" in str(error)
    else:
        raise AssertionError("Expected a cross-model stimulus mismatch")
