from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from realistic_niah_v4.prompts import TokenSpan
from realistic_niah_v5.encoding import NativeTraceEncoding
from realistic_niah_v5.indexed_counter_patch import (
    audit_original_explicit_progress_row,
    build_minimal_item_early_stop_encoding,
    capture_decoder_block_input_states,
    minimal_terminal_suffix_token_ids,
    prefill_with_single_decoder_block_input_replacement,
)
from scripts.analyze_realistic_niah_v5_indexed_counter_early_stop_patch import analyze


class _CharTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) + 1 for value in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(
        self, token_ids: list[int] | tuple[int, ...], *, skip_special_tokens: bool = False
    ) -> str:
        assert not skip_special_tokens
        return "".join(chr(int(value) - 1) for value in token_ids)


def _encoding() -> NativeTraceEncoding:
    spans = tuple(
        TokenSpan(
            slot_index=index,
            start=start,
            end=end,
            active=True,
            kind="native_trace_item",
            canonical_length=end - start,
            model_token_length=end - start,
        )
        for index, (start, end) in enumerate(((4, 7), (7, 10)), start=1)
    )
    return NativeTraceEncoding(
        stimulus_id="s",
        request_id="r",
        design_variant="native_thinking",
        seed=1234,
        split="development",
        count=10,
        model_label="Qwen3-8B",
        model_family="qwen3",
        answer_format="Total: <integer>",
        text="",
        generation_prompt="",
        input_ids=tuple(range(14)),
        attention_mask=(1,) * 14,
        query_position=13,
        prompt_token_count=4,
        raw_prefix_text="",
        selected_site={},
        prompt_record_spans=(),
        trace_item_spans=spans,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=tuple((value, str(value)) for value in range(1, 11)),
        count_candidate_answer_token_ids=tuple((value, (value,)) for value in range(1, 11)),
        count_candidate_token_ids=tuple((value, (value, 99)) for value in range(1, 11)),
    )


def test_minimal_suffix_removes_natural_recap_and_answer_digit() -> None:
    raw = "<think>1. A\nThere are ten records.</think>\n\nTotal: 10"
    close = raw.index("</think>")
    query_start = raw.index("Total:")
    row = {
        "raw_output_text": raw,
        "output_token_ids": _CharTokenizer().encode(raw, add_special_tokens=False),
        "trace_parse": {
            "parser": {"reasoning_end_char": close},
            "char_sites": [
                {
                    "site_kind": "answer_query_v3",
                    "char_start": query_start,
                    "char_end": query_start + len("Total:"),
                }
            ],
        },
    }
    token_ids, audit = minimal_terminal_suffix_token_ids(row, _CharTokenizer())
    decoded = "".join(chr(value - 1) for value in token_ids)
    assert decoded == "</think>\n\nTotal: "
    assert "ten" not in decoded.lower()
    assert "10" not in decoded
    assert audit["natural_recap_removed"] is True
    assert audit["interstitial_nonwhitespace_removed"] is False
    assert audit["minimal_terminal_suffix_fragment_count"] == 1


def test_minimal_suffix_removes_gemma_final_channel_recap() -> None:
    raw = (
        "<think>1. A</think><channel|>"
        "The passage contains 8 city-score audit records.\n\nTotal: 8<turn|>"
    )
    close = raw.index("<channel|>")
    row = {
        "raw_output_text": raw,
        "output_token_ids": _CharTokenizer().encode(raw, add_special_tokens=False),
        "trace_parse": {"parser": {"reasoning_end_char": close}},
    }
    token_ids, audit = minimal_terminal_suffix_token_ids(row, _CharTokenizer())
    decoded = _CharTokenizer().decode(token_ids, skip_special_tokens=False)
    assert decoded == "<channel|>Total: "
    assert "8" not in decoded
    assert audit["interstitial_nonwhitespace_removed"] is True
    assert audit["removed_interstitial_contains_candidate_digit"] is True
    assert audit["minimal_terminal_suffix_fragment_count"] == 2
    assert (
        audit["terminal_suffix_source"]
        == "saved_output_token_ids_channel_close_plus_Total_query_fragments"
    )


def test_early_stop_keeps_item_prefix_and_deletes_future_items() -> None:
    encoding = _encoding()
    registry = SimpleNamespace(trace_items=((4, 7), (7, 10)), query_position=13)
    stopped, audit = build_minimal_item_early_stop_encoding(
        encoding,
        registry,
        target_occurrence=1,
        terminal_suffix_token_ids=(50, 51, 52),
    )
    assert stopped.input_ids == encoding.input_ids[:7] + (50, 51, 52)
    assert stopped.query_position == 9
    assert len(stopped.trace_item_spans) == 1
    assert audit["future_trace_tokens_present"] is False
    assert audit["future_trace_items_removed"] == 1


def test_explicit_progress_audit_accepts_partial_contiguous_episode() -> None:
    row = {
        "gold_count": 10,
        "trace_parse": {
            "parser": {
                "item_start_chars": [10, 20, 30],
                "item_end_chars": [15, 25, 35],
                "item_count": 3,
                "marker_kind": "inline_count",
                "trace_one_to_one": False,
            }
        },
    }
    audit = audit_original_explicit_progress_row(row)
    assert audit["eligible"] is True
    assert audit["parsed_item_count"] == 3
    assert audit["explicit_progress_marker"] is True
    assert audit["available_span_analysis"] is True


class _ToyBlock(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = float(increment)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.increment


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 2)
        self.layers = nn.ModuleList([_ToyBlock(1.0), _ToyBlock(2.0), _ToyBlock(3.0)])
        with torch.no_grad():
            values = torch.arange(32, dtype=torch.float32)
            self.embedding.weight.copy_(torch.stack((values, -values), dim=1))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        **_kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(
            logits=hidden,
            past_key_values=() if use_cache else None,
        )


def test_single_block_input_patch_recomputes_all_upper_layers() -> None:
    model = _ToyModel()
    adapter = SimpleNamespace(
        layers=model.layers,
        attentions=(),
        num_layers=len(model.layers),
    )
    clean = _encoding()
    receiver = NativeTraceEncoding(
        **{
            **clean.__dict__,
            "input_ids": clean.input_ids[:4] + (0, 0, 0) + clean.input_ids[7:],
        }
    )
    positions = (4, 5, 6)
    captured = capture_decoder_block_input_states(
        model, adapter, clean, positions, layers=(1,)
    )
    prefill, applications, realized = (
        prefill_with_single_decoder_block_input_replacement(
            model,
            adapter,
            receiver,
            positions=positions,
            layer=1,
            replacement_states=captured[1],
        )
    )
    clean_output = model(
        input_ids=torch.tensor([clean.input_ids]),
        attention_mask=torch.tensor([clean.attention_mask]),
        use_cache=True,
    )
    assert applications == 1
    assert realized > 0.0
    assert torch.equal(
        prefill.logits[0, list(positions)],
        clean_output.logits[0, list(positions)],
    )


def _score_string(target: int | None, strength: float) -> str:
    scores = [0.0] * 10
    if target is not None:
        scores[target - 1] = float(strength)
    return ",".join(str(value) for value in scores)


def test_analysis_freezes_later_layer_after_equal_exact_restoration() -> None:
    rows = []
    for seed in range(1234, 1254):
        for target in range(1, 11):
            common = {
                "seed": seed,
                "target_occurrence": target,
                "marker_kind": "inline_count",
            }
            rows.extend(
                [
                    {
                        **common,
                        "condition": "clean_early_stop_reference",
                        "source_layer": -1,
                        "candidate_log_scores": _score_string(target, 3.0),
                    },
                    {
                        **common,
                        "condition": "corrupt_early_stop_reference",
                        "source_layer": -1,
                        "candidate_log_scores": _score_string(None, 0.0),
                    },
                ]
            )
            for layer, strength in ((0, 2.0), (4, 4.0)):
                rows.extend(
                    [
                        {
                            **common,
                            "condition": "clean_item_restore_into_corrupt",
                            "source_layer": layer,
                            "candidate_log_scores": _score_string(target, strength),
                        },
                        {
                            **common,
                            "condition": "corrupt_item_ablate_into_clean",
                            "source_layer": layer,
                            "candidate_log_scores": _score_string(None, 0.0),
                        },
                    ]
                )
    _layers, _derived, occurrence, _seed_effects, result = analyze(
        pd.DataFrame(rows),
        phase="discovery",
        frozen_layer=None,
        expected_seed_order=tuple(range(1234, 1254)),
    )
    assert result["selected_layer"] == 4
    assert result["old_html_explicit_progress_state_restoration_pass"] is True
    assert result["ablation_secondary_support"] is True
    assert set(occurrence["target_occurrence"]) == set(range(1, 11))
