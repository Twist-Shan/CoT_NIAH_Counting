from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from realistic_niah_v5.counting_mechanism_transfer import (
    align_region_groups,
    build_first_pass_tstar_answer_source_registry,
    build_immediate_count_query_encoding,
    candidate_metrics,
    countscope_blank_encoding,
    continued_count_expected,
    item_candidate_tokens,
    maximum_latent_count_expected,
    occurrence_region_positions,
    paper_causal_influence,
    prefill_with_block_input_intervention,
    prompt_scrubbed_encoding,
    score_count_candidate_sequences,
)
from realistic_niah_v5.count_stream import build_answer_source_registry
from realistic_niah_v5.unnumbered_counter_restore import (
    audit_first_occurrence_prefix_clean,
)
from scripts.analyze_realistic_niah_v5_counting_mechanism_transfer import (
    analyze,
    markdown,
)
from scripts import run_realistic_niah_v5_counting_mechanism_transfer as runner
from scripts.freeze_realistic_niah_v5_counting_mechanism_confirmation import (
    freeze_confirmation_config,
    score_geometry_bands,
)


class _Registry:
    trace_items = ((10, 15), (20, 27))

    def positions(self, group: str):
        if group == "trace_markers":
            return (10, 20)
        raise KeyError(group)


class _CharacterTokenizer:
    all_special_ids: tuple[int, ...] = ()

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert not add_special_tokens
        result = {
            "input_ids": [ord(value) for value in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) for value in text]

    def decode(
        self,
        values: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in values)

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return "".join(str(message["content"]) for message in messages) + "<eos>"


def _unnumbered_row() -> dict[str, object]:
    header = "Inspect the records carefully and report all matching cities. " * 3 + "\n"
    first = "Chicago received a score of 72.\n"
    second = "Baku received a score of 98.\n"
    prompt = header + first + second
    first_start = len(header)
    second_start = first_start + len(first)
    raw = (
        "<think>\n"
        "- Chicago received a score of 72.\n"
        "- Baku received a score of 98.\n"
        "</think>\nTotal: 2"
    )
    return {
        "request_id": "counting-transfer-unnumbered-test",
        "stimulus_id": "counting-transfer-unnumbered-test",
        "model_label": "Qwen3-8B",
        "model_family": "qwen3",
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [ord(value) for value in prompt],
        "attention_mask": [1] * len(prompt),
        "rendered_prompt": prompt,
        "prompt_record_spans": [
            {
                "slot_index": 1,
                "city": "Chicago",
                "score": 72,
                "start": first_start,
                "end": second_start,
            },
            {
                "slot_index": 2,
                "city": "Baku",
                "score": 98,
                "start": second_start,
                "end": len(prompt),
            },
        ],
        "gold_records": [
            {"city": "Chicago", "score": 72},
            {"city": "Baku", "score": 98},
        ],
        "gold_count": 2,
        "seed": 1234,
        "split": "discovery",
    }


def _first_pass_row_with_future_recap() -> dict[str, object]:
    row = _unnumbered_row()
    raw = (
        "<think>\n"
        "- Chicago received a score of 72.\n"
        "- Baku received a score of 98.\n"
        "The matching cities are Chicago and Baku.\n"
        "</think>\nTotal: 2"
    )
    row["raw_output_text"] = raw
    row["output_token_ids"] = [ord(value) for value in raw]
    audit = audit_first_occurrence_prefix_clean(row)
    assert audit["eligible"] is True
    row["noindex_n2_format_audit"] = {
        **audit,
        "primary_eligible_prefix_clean": True,
        "strict_eligible_no_explicit_count_cue": True,
    }
    row["noindex_n2_cohort"] = {
        "fixed_count": 2,
        "selection_population": "first_pass_noindex_enumeration",
        "selection_used_final_answer": False,
        "mechanism_outcomes_accessed": False,
        "split": "confirmation",
    }
    return row


def test_occurrence_regions_separate_marker_payload_and_closing() -> None:
    registry = _Registry()
    assert occurrence_region_positions(registry, 1, "marker") == (10,)
    assert occurrence_region_positions(registry, 1, "opening") == (11,)
    assert occurrence_region_positions(registry, 1, "payload") == (11, 12, 13)
    assert occurrence_region_positions(registry, 1, "closing") == (14,)
    assert occurrence_region_positions(registry, 1, "post_item") == (15,)
    assert occurrence_region_positions(registry, 1, "nonmarker") == (11, 12, 13, 14)
    assert occurrence_region_positions(registry, 1, "full") == (10, 11, 12, 13, 14)


def test_alignment_resamples_inside_each_event_without_crossing() -> None:
    aligned = align_region_groups(
        ((1, 2), (5, 6, 7)),
        ((10, 11, 12, 13), (20, 21)),
    )
    assert aligned.receiver_positions == (10, 11, 12, 13, 20, 21)
    assert aligned.donor_positions == (1, 1, 2, 2, 5, 7)
    assert aligned.source_group_widths == (2, 3)
    assert aligned.receiver_group_widths == (4, 2)
    aligned.validate()


def test_unnumbered_registry_scrub_and_countscope_receiver_end_to_end() -> None:
    tokenizer = _CharacterTokenizer()
    clean, registry = build_answer_source_registry(
        _unnumbered_row(), tokenizer, candidate_counts=tuple(range(1, 19))
    )
    assert [value for value, _ids in clean.count_candidate_token_ids] == list(
        range(1, 19)
    )
    assert len(registry.trace_items) == 2
    assert len(registry.trace_markers) == 2

    prompt_scrubbed, prompt_audit = prompt_scrubbed_encoding(
        clean, registry, tokenizer, random_seed=7
    )
    assert prompt_audit["prompt_record_changed_token_count"] > 0
    assert all(
        prompt_scrubbed.input_ids[position] == clean.input_ids[position]
        for position in registry.positions("trace_items")
    )

    receiver, receiver_audit = countscope_blank_encoding(
        clean, registry, tokenizer, receiver_occurrence=1, random_seed=11
    )
    assert receiver_audit["future_trace_items_removed"] == 1
    assert receiver_audit["item_alphanumeric_semantics_scrubbed"] is True
    assert receiver_audit["post_event_original_suffix_tokens_retained"] == 0
    assert receiver_audit["early_stop_suffix_mode"] == (
        "minimal_literal_reasoning_close_and_total_query"
    )
    assert receiver.query_position < clean.query_position
    assert [value for value, _ids in receiver.count_candidate_token_ids] == list(
        range(1, 19)
    )

    immediate, immediate_audit = build_immediate_count_query_encoding(
        clean, registry, tokenizer, target_occurrence=1
    )
    assert immediate.input_ids[: registry.trace_items[0][1]] == clean.input_ids[
        : registry.trace_items[0][1]
    ]
    assert immediate_audit["post_event_original_suffix_tokens_retained"] == 0
    assert immediate_audit["terminal_suffix_contains_candidate_digit"] is False
    assert immediate_audit["terminal_suffix_contains_candidate_word"] is False


def test_first_pass_registry_excludes_future_recap_and_registers_separator() -> None:
    tokenizer = _CharacterTokenizer()
    row = _first_pass_row_with_future_recap()
    clean, registry = build_first_pass_tstar_answer_source_registry(
        row, tokenizer, candidate_counts=tuple(range(1, 7))
    )
    audit = row["noindex_n2_format_audit"]
    assert clean.split == "confirmation"
    assert clean.count == 2
    assert len(registry.trace_items) == 2
    assert occurrence_region_positions(registry, 2, "marker")
    assert clean.raw_prefix_text.startswith(
        str(row["raw_output_text"])[: int(audit["t_star_char"])]
    )
    assert "The matching cities" not in clean.raw_prefix_text
    assert clean.raw_prefix_text.endswith("\n</think>\n\nTotal:")
    assert clean.selected_site["future_recap_available_to_context"] is False


def test_expected_count_hypotheses_match_paper_formulas() -> None:
    assert continued_count_expected(5, 4, 2) == 7
    assert maximum_latent_count_expected(3, 7, 2) == 5
    assert maximum_latent_count_expected(8, 5, 2) == 8
    with pytest.raises(ValueError):
        continued_count_expected(1, 4, 2)


def test_native_item_candidates_support_dynamic_trace_length() -> None:
    encoding = SimpleNamespace(input_ids=tuple(range(20)))
    registry = SimpleNamespace(trace_items=((2, 5), (5, 9), (9, 14)))
    assert item_candidate_tokens(encoding, registry) == {
        1: (2, 3, 4),
        2: (5, 6, 7, 8),
        3: (9, 10, 11, 12, 13),
    }

    single_token_registry = SimpleNamespace(trace_items=((2, 3), (5, 9)))
    assert item_candidate_tokens(encoding, single_token_registry) == {
        1: (2,),
        2: (5, 6, 7, 8),
    }


def test_count_candidate_scorer_supports_more_than_ten_candidates() -> None:
    class RepeatablePast:
        repeated = 1

        def batch_repeat_interleave(self, count: int) -> None:
            self.repeated = int(count)

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branch_batches: list[int] = []

        def forward(
            self,
            input_ids,
            attention_mask,
            past_key_values,
            use_cache=False,
            position_ids=None,
            cache_position=None,
        ):
            del attention_mask, use_cache, position_ids, cache_position
            assert past_key_values.repeated in {2, 4}
            batch, length = input_ids.shape
            self.branch_batches.append(int(batch))
            return SimpleNamespace(logits=torch.zeros(batch, length, 64))

    encoding = SimpleNamespace(
        attention_mask=(1, 1, 1, 1),
        sequence_length=4,
        count_candidate_token_ids=tuple(
            (count, (count, 63)) for count in range(1, 19)
        ),
    )
    prefill = SimpleNamespace(
        logits=torch.zeros(1, 1, 64),
        past_key_values=RepeatablePast(),
    )
    model = ToyModel()
    result = score_count_candidate_sequences(model, encoding, prefill)
    assert sorted(result.candidate_log_scores) == list(range(1, 19))
    assert model.branch_batches == [4, 4, 4, 4, 2]
    assert prefill.past_key_values.repeated == 1


def test_maximum_count_per_seed_selection_is_outcome_blind(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "rows.jsonl"
    values = [
        {"seed": 1, "gold_count": 3, "request_id": "small"},
        {"seed": 1, "gold_count": 7, "request_id": "large"},
        {"seed": 2, "gold_count": 4, "request_id": "only"},
    ]
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "audit_no_count_enumeration_trace",
        lambda _row: {"eligible": True, "reasons": []},
    )
    selected = runner._read_rows(
        path, selection_mode="maximum_gold_count_per_seed"
    )
    assert selected[1]["request_id"] == "large"
    assert selected[2]["request_id"] == "only"
    with pytest.raises(ValueError, match="Duplicate generation seed"):
        runner._read_rows(path, selection_mode="unique_seed")


def test_general_candidate_metrics_and_paper_ci() -> None:
    baseline = candidate_metrics(
        {1: -3.0, 2: -2.0, 3: 0.0, 4: -4.0},
        target_count=4,
        original_count=3,
    )
    patched = candidate_metrics(
        {1: -3.0, 2: -2.0, 3: -4.0, 4: 0.0},
        target_count=4,
        original_count=3,
    )
    assert patched["predicted_count_among_candidates"] == 4
    assert patched["target_is_candidate_argmax"] is True
    expected = 0.5 * (
        (patched["target_probability"] - baseline["target_probability"])
        + (
            baseline["candidate_probabilities_by_count"]["3"]
            - patched["candidate_probabilities_by_count"]["3"]
        )
    )
    assert paper_causal_influence(
        baseline, patched, expected_count=4, original_count=3
    ) == pytest.approx(expected)


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1.0


class _FakeModel(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        torch.nn.init.zeros_(self.embedding.weight)
        self.layers = layers

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_attentions: bool = False,
        **_kwargs,
    ):
        del attention_mask, use_cache, output_attentions
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden, past_key_values=object())


def test_preblock_replace_rescales_and_applies_once() -> None:
    layers = torch.nn.ModuleList([_Block(), _Block()])
    model = _FakeModel(layers)
    adapter = SimpleNamespace(num_layers=2, layers=layers, attentions=())
    encoding = SimpleNamespace(
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        sequence_length=3,
    )
    replacements = {0: torch.tensor([[2.0, 0.0, 0.0, 0.0]])}
    _prefill, captures, applications, realized = (
        prefill_with_block_input_intervention(
            model,
            adapter,
            encoding,
            positions=(1,),
            layer_values=replacements,
            intervention_kind="replace",
            readout_layers=(0, 1),
            readout_positions=(1,),
            norm_rescale_replacement=False,
        )
    )
    assert applications == {0: 1}
    assert realized[0] == pytest.approx(2.0)
    assert captures[0][0].tolist() == pytest.approx([2.0, 0.0, 0.0, 0.0])
    assert captures[1][0].tolist() == pytest.approx([3.0, 1.0, 1.0, 1.0])


def test_analysis_keeps_k1_and_k2_separate() -> None:
    rows = []
    for k, adoption in ((1, False), (2, True)):
        rows.append(
            {
                "experiment": "continued_counting",
                "condition": "last_k_source_to_first_k_target",
                "seed": 1234,
                "region": "marker",
                "k": k,
                "source_end_occurrence": 5,
                "paper_ci": 0.2 * k,
                "target_probability": 0.1 * k,
                "target_is_candidate_argmax": adoption,
                "greedy_target_adoption": adoption,
                "boundary_countscope_readouts": [
                    {
                        "hop_after_patch": 1,
                        "paper_ci": 0.1 * k,
                        "target_probability": 0.1,
                        "target_is_candidate_argmax": adoption,
                        "greedy_target_adoption": adoption,
                    }
                ],
                "successor_readout": {
                    "donor_successor_occurrence": 6,
                    "predicted_occurrence_mean_logprob": 6 if adoption else 2,
                    "greedy_donor_successor_adoption": adoption,
                    "target_delta_mean_logprob_margin": float(k),
                },
            }
        )
    summary = analyze(rows, bootstrap_samples=100, random_seed=7)
    assert [row["k"] for row in summary["continued_final"]] == [1, 2]
    assert summary["continued_final"][0]["metrics"][
        "greedy_target_adoption"
    ]["mean"] == 0.0
    assert summary["continued_final"][1]["metrics"][
        "greedy_target_adoption"
    ]["mean"] == 1.0
    assert "Continued counting: final answer" in markdown(summary)


def test_geometry_band_freeze_uses_only_heldout_discovery_panel() -> None:
    rows = []
    values = {
        "early": {
            "position_difference": 0.1,
            "opposite_position_difference": 0.0,
            "norm_matched_orthogonal": 0.0,
        },
        "late": {
            "position_difference": 0.7,
            "opposite_position_difference": -0.1,
            "norm_matched_orthogonal": 0.1,
        },
    }
    for band, by_condition in values.items():
        for seed in (10, 11):
            for condition, score in by_condition.items():
                rows.append(
                    {
                        "experiment": "linear_additivity",
                        "seed": seed,
                        "steering_band": band,
                        "receiver_occurrence": 2,
                        "position_difference": 1,
                        "condition": condition,
                        "paper_ci": score,
                        "donor_aligned_expected_shift": score,
                        "target_is_candidate_argmax": score > 0.5,
                        "greedy_target_adoption": score > 0.5,
                    }
                )
    selected, scores = score_geometry_bands(
        rows, expected_eval_seeds=(10, 11)
    )
    assert selected == "late"
    assert len(scores) == 2

    discovery = {
        "phase": "first_pass_n3_discovery",
        "seeds": [1, 2, 3],
        "experiments": {
            "countscope": {"seeds": [1, 2, 3]},
            "linear_additivity": {
                "fit_seeds": [1, 2],
                "eval_seeds": [3],
                "layer_bands": {"early": [1], "late": [2]},
            },
        },
        "cohort_contract": {"confirmation_seeds_reserved": [20, 21]},
    }
    frozen = freeze_confirmation_config(
        discovery,
        selected_band=selected,
        selection_audit={"selected_band": selected},
    )
    assert frozen["phase"] == "first_pass_n3_confirmation"
    assert frozen["seeds"] == [20, 21]
    assert frozen["experiments"]["countscope"]["seeds"] == [20, 21]
    assert frozen["experiments"]["linear_additivity"]["fit_seeds"] == [1, 2, 3]
    assert frozen["experiments"]["linear_additivity"]["eval_seeds"] == [20, 21]
    assert frozen["experiments"]["linear_additivity"]["layer_bands"] == {
        "late": [2]
    }
