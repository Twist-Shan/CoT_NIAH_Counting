from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
import torch
from torch import nn

from realistic_niah_v4.spec import MODEL_SPECS
from realistic_niah_v4_4_2.attention import (
    REGION_NAMES,
    apply_saved_rope,
    reconstruct_attention_summary,
)
from realistic_niah_v4_4_2.capture import _attention_metadata, kv_source_layers
from realistic_niah_v4_4_2.aggregate import (
    LEGACY_MODE_COMPARISON,
    PAIR_SPECS,
    attention_paired_effects,
    hidden_paired_effects,
)
from realistic_niah_v4_4_2.prompts import build_user_text, render_trace_prompt
from realistic_niah_v4_4_2.spec import V442Config
from realistic_niah_v4_4_2.trace import locate_trace_boundaries


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def _id(self, value: str) -> int:
        if value not in self.vocabulary:
            self.vocabulary[value] = len(self.vocabulary) + 1
        return self.vocabulary[value]

    def apply_chat_template(self, messages, **kwargs):
        enabled = kwargs.get("enable_thinking", kwargs.get("thinking", False))
        suffix = "<assistant><think>" if enabled else "<assistant>"
        return "<user>" + str(messages[0]["content"]) + suffix

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        result = {
            "input_ids": [self._id(value) for value in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, ids, **_kwargs):
        inverse = {value: key for key, value in self.vocabulary.items()}
        return "".join(inverse[int(value)] for value in ids)


def _stimulus() -> dict:
    passage = "AAA city score BBB"
    start = passage.index("city")
    return {
        "stimulus_id": "V4_4_T10000_N1_seed1234",
        "design_variant": "v4.4",
        "seed": 1234,
        "split": "discovery",
        "gold_count": 1,
        "passage": passage,
        "slots": [
            {
                "slot_index": 0,
                "char_start": start,
                "char_end": start + 4,
                "active": True,
                "content_kind": "needle",
                "canonical_token_length": 4,
            }
        ],
    }


def test_registered_config_keeps_v3_native_decoding() -> None:
    config = V442Config()
    config.validate()
    assert config.seeds == tuple(range(1234, 1244))
    assert config.legacy_baseline_mode == "rerun_all"
    assert any(name == "cue_present_mode_effect" for name, _, _ in PAIR_SPECS)
    assert config.native_max_new_tokens == 4096
    assert config.decoding("Qwen3-8B", "native_thinking") == {
        "do_sample": True,
        "max_new_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert config.decoding("Gemma4-E4B", "native_thinking")["top_k"] == 64


def test_prompt_grid_changes_only_cue_and_native_flag() -> None:
    tokenizer = CharacterTokenizer()
    stimulus = _stimulus()
    cue_native = render_trace_prompt(
        stimulus,
        tokenizer=tokenizer,
        model_spec=MODEL_SPECS["Qwen3-8B"],
        mode="native_thinking",
        prompt_variant="cue_present",
    )
    no_cue_native = render_trace_prompt(
        stimulus,
        tokenizer=tokenizer,
        model_spec=MODEL_SPECS["Qwen3-8B"],
        mode="native_thinking",
        prompt_variant="cue_absent",
    )
    cue_nonthinking = render_trace_prompt(
        stimulus,
        tokenizer=tokenizer,
        model_spec=MODEL_SPECS["Qwen3-8B"],
        mode="nonthinking",
        prompt_variant="cue_present",
    )
    assert "You will need to count" in cue_native.user_text
    assert "You will need to count" not in no_cue_native.user_text
    assert cue_native.user_text == cue_nonthinking.user_text
    assert cue_native.assistant_prefix_span is None
    assert cue_nonthinking.assistant_prefix_span is not None
    assert cue_nonthinking.model_text.endswith("Total:")
    assert cue_native.model_text.endswith("<think>")
    assert len(cue_native.needle_spans) == 1


@pytest.mark.parametrize(
    ("family", "text", "trace_end_text"),
    [
        ("qwen3", "count one</think>\nTotal:1", "count one"),
        ("gemma4", "count one<channel|>Total:1", "count one"),
    ],
)
def test_trace_boundaries_are_token_exact(family, text, trace_end_text) -> None:
    tokenizer = CharacterTokenizer()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    boundary = locate_trace_boundaries(
        tokenizer, ids, mode="native_thinking", model_family=family
    )
    assert boundary.boundary_status == "ok"
    assert tokenizer.decode(ids[: boundary.trace_end]) == trace_end_text
    assert tokenizer.decode(ids[boundary.answer_query_start : boundary.answer_query_end]).strip() == "Total:"


def test_unterminated_trace_is_explicit() -> None:
    tokenizer = CharacterTokenizer()
    ids = tokenizer("still counting", add_special_tokens=False)["input_ids"]
    boundary = locate_trace_boundaries(
        tokenizer, ids, mode="native_thinking", model_family="qwen3"
    )
    assert boundary.boundary_status == "unterminated_trace"
    assert boundary.trace_end == len(ids)
    assert boundary.final_start == len(ids)


def test_partial_rope_preserves_unrotated_suffix() -> None:
    value = torch.tensor([[[1.0, 2.0, 7.0, 8.0]]])
    cos = torch.zeros(1, 2)
    sin = torch.ones(1, 2)
    result = apply_saved_rope(value, cos, sin)
    assert result[0, 0, :2].tolist() == pytest.approx([-2.0, 1.0])
    assert result[0, 0, 2:].tolist() == [7.0, 8.0]


class _Attention(nn.Module):
    def __init__(self, layer_type: str, *, has_k: bool):
        super().__init__()
        self.layer_type = layer_type
        self.q_norm = nn.Identity()
        if has_k:
            self.k_norm = nn.Identity()


def test_shared_kv_layers_resolve_within_layer_type() -> None:
    attentions = (
        _Attention("sliding_attention", has_k=True),
        _Attention("full_attention", has_k=True),
        _Attention("sliding_attention", has_k=False),
        _Attention("full_attention", has_k=False),
    )
    adapter = SimpleNamespace(attentions=attentions)
    assert kv_source_layers(adapter, (0, 1, 2, 3)) == {0: 0, 1: 1, 2: 0, 3: 1}


def test_qwen_layer_type_and_sliding_window_are_read_from_config() -> None:
    attention = _Attention("unused", has_k=True)
    del attention.layer_type
    attention.layer_idx = 1
    attention.config = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"], sliding_window=512
    )
    attention.sliding_window = 512
    attention.head_dim = 128
    metadata = _attention_metadata(attention, layer=1, kv_source_layer=1)
    assert metadata["layer_type"] == "sliding_attention"
    assert metadata["is_sliding"] is True
    assert metadata["sliding_window"] == 512


def test_attention_reconstruction_writes_binned_maps(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    manifest = {
        "sequence_length": 6,
        "prompt_token_count": 3,
        "query_roles": ["trace", "trace", "answer_query"],
        "query_positions": [3, 4, 5],
        "boundaries": {
            "trace_start": 0,
            "trace_end": 2,
            "final_start": 2,
            "final_end": 3,
        },
        "needle_spans": [[1, 2]],
        "needle_end_positions": [1],
        "cue_span": [0, 1],
        "passage_span": [1, 2],
        "question_span": [2, 3],
        "layers": [
            {
                "layer": 0,
                "k_file": "kv_source_00_k_norm.pt",
                "scaling": 1.0,
                "softcap": None,
                "is_sliding": False,
                "sliding_window": None,
            }
        ],
    }
    (root / "capture_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(torch.tensor([3, 4, 5]), root / "query_positions.pt")
    q = torch.ones(3, 1, 2)
    k = torch.ones(6, 1, 2)
    torch.save(q, root / "layer_00_q_norm.pt")
    torch.save(k, root / "kv_source_00_k_norm.pt")
    torch.save(
        {"cos": torch.ones(6, 2), "sin": torch.zeros(6, 2)},
        root / "layer_00_rope.pt",
    )
    config = V442Config(attention_trace_bins=2)
    metadata = reconstruct_attention_summary(root, config=config)
    payload = torch.load(
        root / "attention_summary.pt", map_location="cpu", weights_only=True
    )
    layer = payload["layers"][0]
    assert metadata["trace_bins"] == 2
    assert payload["region_names"] == REGION_NAMES
    assert layer["trace_region_map"].shape == (2, len(REGION_NAMES))
    assert layer["trace_to_trace_map"].shape == (2, 2)
    assert layer["answer_query_last_region"].shape == (1, len(REGION_NAMES))
    assert torch.isfinite(layer["trace_region_map"]).all()


def test_legacy_v44_bridge_uses_final_total_query_and_needle_mass(
    tmp_path: Path,
) -> None:
    native = (
        tmp_path
        / "conditions"
        / "Qwen3-8B"
        / "cue_present"
        / "native_thinking"
        / "discovery"
        / "toy"
        / "capture"
    )
    native.mkdir(parents=True)
    manifest = {
        "model_label": "Qwen3-8B",
        "stimulus_id": "toy",
        "mode": "native_thinking",
        "prompt_variant": "cue_present",
        "seed": 1234,
        "split": "discovery",
        "gold_count": 1,
        "query_roles": ["trace", "answer_query", "answer_query"],
        "layers": [{"layer": 0}],
    }
    (native / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    # The bridge must use the final marker token [0, 1], not the first [1, 0].
    torch.save(
        torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]),
        native / "layer_00_hidden.pt",
    )
    attention = torch.zeros(1, len(REGION_NAMES))
    attention[0, REGION_NAMES.index("needle_span")] = 0.7
    torch.save(
        {
            "region_names": REGION_NAMES,
            "layers": {0: {"answer_query_last_region": attention}},
        },
        native / "attention_summary.pt",
    )

    legacy_hidden = tmp_path / "legacy_hidden"
    (legacy_hidden / "shards" / "v4.4").mkdir(parents=True)
    with (legacy_hidden / "shards" / "v4.4" / "toy.npz").open("wb") as handle:
        np.savez(
            handle,
            layer_indices=np.asarray([0]),
            query_states=np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
    (legacy_hidden / "capture_index.jsonl").write_text(
        json.dumps(
            {
                "stimulus_id": "toy",
                "design_variant": "v4.4",
                "shard_path": "shards/v4.4/toy.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_attention = tmp_path / "attention_head_detail.csv"
    pd.DataFrame(
        [
            {
                "stimulus_id": "toy",
                "design_variant": "v4.4",
                "seed": 1234,
                "split": "discovery",
                "count": 1,
                "layer": 0,
                "head": 0,
                "broad_mass": 0.2,
            }
        ]
    ).to_csv(legacy_attention, index=False)
    (tmp_path / "legacy_v4_4_reference.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "model_label": "Qwen3-8B",
                        "artifact": "representation/answer_query_all_layers_v1/capture_index.jsonl",
                        "path": str(legacy_hidden / "capture_index.jsonl"),
                    },
                    {
                        "model_label": "Qwen3-8B",
                        "artifact": "attention/analysis/attention_head_detail.csv",
                        "path": str(legacy_attention),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    hidden = hidden_paired_effects(tmp_path)
    row = hidden[hidden["comparison"] == LEGACY_MODE_COMPARISON].iloc[0]
    assert row["cosine"] == pytest.approx(0.0)
    attention_rows = attention_paired_effects(tmp_path)
    row = attention_rows[
        attention_rows["comparison"] == LEGACY_MODE_COMPARISON
    ].iloc[0]
    assert row["left_mass"] == pytest.approx(0.2)
    assert row["right_mass"] == pytest.approx(0.7)
