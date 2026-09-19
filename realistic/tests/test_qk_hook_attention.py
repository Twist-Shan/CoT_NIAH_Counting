import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from dataset_generation.qk_hook_attention import analyze_qk_qwen3 as analyze
from dataset_generation.qk_hook_attention import capture_qk_qwen3 as capture


class DummyTokenizer:
    all_special_ids = [101, 102]

    def __call__(self, text, add_special_tokens=False):
        # Tiny deterministic tokenizer for marker-span tests.
        vocab = {"needle": [7], " needle": [8, 7], "answer": [9], " answer": [8, 9]}
        return SimpleNamespace(input_ids=vocab.get(text, [ord(c) % 50 for c in text]))


class TinySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 4, bias=False)
        self.k_proj = nn.Linear(3, 2, bias=False)
        self.q_norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
        self.k_norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=TinySelfAttention())])


def tiny_qwen3_config(**overrides):
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_dropout": 0.0,
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def write_tiny_qwen3_capture_cache(cache_dir: Path, model, input_ids, attention_mask, position_ids):
    torch.save(input_ids.cpu(), cache_dir / "input_ids.pt")
    torch.save(attention_mask.cpu(), cache_dir / "attention_mask.pt")
    torch.save(position_ids.cpu(), cache_dir / "position_ids.pt")
    capture.save_layer_norms(model, [0], cache_dir)
    metadata = {
        "seq_len": int(input_ids.shape[1]),
        "batch_size": 1,
        "target_layers": [0],
        "special_token_ids": [],
        "model_config": model.config.to_dict(),
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, default=str), encoding="utf-8")



def dense_stats(q, k, query_positions, spans, key_padding_mask=None, scaling=1.0, local_windows=()):
    scores = q[query_positions].float() @ k.float().T * scaling
    T = k.shape[0]
    keys = torch.arange(T)
    allowed = keys[None, :] <= torch.tensor(query_positions)[:, None]
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[None, :]
    scores = scores.masked_fill(~allowed, -torch.inf)
    probs = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)
    span_mass = {}
    for name, intervals in spans.items():
        mask = torch.zeros(T, dtype=torch.bool)
        for start, end in intervals:
            mask[start:end] = True
        span_mass[name] = probs[:, mask].sum(dim=-1)
    local_mass = {}
    for w in local_windows:
        vals = []
        for row, pos in enumerate(query_positions):
            start = max(0, pos - w + 1)
            vals.append(probs[row, start : pos + 1].sum())
        local_mass[str(w)] = torch.stack(vals)
    return probs, entropy, span_mass, local_mass


def write_synthetic_cache(cache_dir: Path):
    T, hq, hkv, d = 5, 4, 2, 2
    q_raw = torch.arange(T * hq * d, dtype=torch.float32).view(1, T, hq * d) / 10
    k_raw = torch.arange(T * hkv * d, dtype=torch.float32).view(1, T, hkv * d) / 7
    torch.save(q_raw, cache_dir / "layer_00_q_raw.pt")
    torch.save(k_raw, cache_dir / "layer_00_k_raw.pt")
    torch.save(
        {
            "q_norm_weight": torch.ones(d),
            "k_norm_weight": torch.ones(d),
            "q_norm_eps": 1e-6,
            "k_norm_eps": 1e-6,
        },
        cache_dir / "layer_00_qk_norms.pt",
    )
    torch.save(torch.arange(T).view(1, T), cache_dir / "input_ids.pt")
    torch.save(torch.ones(1, T, dtype=torch.long), cache_dir / "attention_mask.pt")
    torch.save(torch.arange(T, dtype=torch.long).view(1, T), cache_dir / "position_ids.pt")
    (cache_dir / "tokens.json").write_text(json.dumps([f"tok{i}" for i in range(T)]), encoding="utf-8")
    metadata = {
        "seq_len": T,
        "special_token_ids": [0],
        "model_config": {
            "num_attention_heads": hq,
            "num_key_value_heads": hkv,
            "head_dim": d,
            "hidden_size": hq * d,
            "rope_theta": 10000.0,
            "rope_parameters": {"rope_type": "default"},
            "rope_scaling": None,
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    spec = {
        "query_positions": {"last": [-1], "middle": [2]},
        "spans": {"first_two": [[0, 2]], "single": [3, 4]},
        "local_windows": [1, 3],
        "window_received": [{"name": "first_two_later", "span_name": "first_two", "query_start": 2, "query_end": None}],
    }
    (cache_dir / "analysis_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return metadata, spec


def test_capture_parse_layers_and_analysis_spec_helpers():
    assert capture.parse_layers("0, 2:7:2, 2") == [0, 2, 4, 6]
    assert capture.find_all_subsequences([1, 2, 1, 2, 1], [1, 2]) == [(0, 2), (2, 4)]

    spec = capture.build_analysis_spec(input_ids=[101, 3, 8, 7, 4, 102], tokenizer=DummyTokenizer(), markers={"needle": "needle"})
    assert spec["spans"]["bos"] == [[0, 1]]
    assert spec["spans"]["special_tokens"] == [[0, 1], [5, 6]]
    assert spec["spans"]["needle"] == [[2, 4], [3, 4]]
    assert spec["query_positions"]["after_needle"] == [4, 4]


def test_capture_hook_saves_qk_with_requested_dtype(tmp_path):
    model = TinyModel()
    handles = capture.install_qk_hooks(model=model, target_layers=[0], out_dir=tmp_path, save_dtype=torch.float16)
    try:
        x = torch.randn(1, 3, 3)
        attn = model.model.layers[0].self_attn
        _ = attn.q_proj(x)
        _ = attn.k_proj(x)
    finally:
        for handle in handles:
            handle.remove()

    q = torch.load(tmp_path / "layer_00_q_raw.pt", map_location="cpu")
    k = torch.load(tmp_path / "layer_00_k_raw.pt", map_location="cpu")
    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert q.shape == (1, 3, 4)
    assert k.shape == (1, 3, 2)
    assert capture.SAVE_DTYPE_MAP["auto"] is None
    assert capture.SAVE_DTYPE_MAP["same"] is None


def test_compute_query_block_stats_matches_dense_reference_with_small_blocks():
    torch.manual_seed(0)
    T, d = 9, 4
    q = torch.randn(T, d)
    k = torch.randn(T, d)
    query_positions = torch.tensor([0, 3, 8])
    spans = {"prefix": [(0, 2)], "middle": [(3, 6)], "empty": []}
    key_padding_mask = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    scaling = d**-0.5
    stats = analyze.compute_query_block_stats(
        q_block=q[query_positions],
        k=k,
        query_positions=query_positions,
        spans=spans,
        key_padding_mask=key_padding_mask,
        scaling=scaling,
        key_block_size=2,
        topk=3,
        local_windows=[1, 4],
        return_rows=True,
    )
    probs, entropy, span_mass, local_mass = dense_stats(
        q, k, query_positions.tolist(), spans, key_padding_mask=key_padding_mask, scaling=scaling, local_windows=[1, 4]
    )
    assert torch.allclose(stats["entropy"], entropy, atol=1e-6)
    for name in spans:
        assert torch.allclose(stats["span_mass"][name], span_mass[name], atol=1e-6)
    for name in ["1", "4"]:
        assert torch.allclose(stats["local_mass"][name], local_mass[name], atol=1e-6)
    assert torch.allclose(stats["rows"], probs, atol=1e-6)

    top_probs, top_idx = probs.topk(k=3, dim=-1)
    # Rows with fewer than top-k allowed keys may contain arbitrary zero-probability
    # tie entries. Compare only positive-probability top-k entries exactly.
    positive = top_probs > 0
    assert torch.equal(stats["topk_indices"][positive], top_idx[positive])
    assert torch.allclose(stats["topk_probs"], top_probs, atol=1e-6)


def test_compute_query_block_stats_handles_fully_masked_rows_without_nan_topk():
    q = torch.randn(1, 3)
    k = torch.randn(4, 3)
    stats = analyze.compute_query_block_stats(
        q_block=q,
        k=k,
        query_positions=torch.tensor([0]),
        spans={"all": [(0, 4)]},
        key_padding_mask=torch.zeros(4, dtype=torch.bool),
        scaling=1.0,
        key_block_size=2,
        topk=2,
        local_windows=[2],
        return_rows=True,
    )
    assert math.isnan(float(stats["entropy"][0]))
    assert stats["span_mass"]["all"].item() == 0.0
    assert stats["local_mass"]["2"].item() == 0.0
    assert torch.equal(stats["topk_probs"], torch.zeros_like(stats["topk_probs"]))
    assert torch.equal(stats["rows"], torch.zeros_like(stats["rows"]))


@pytest.mark.parametrize("head, expected_kv_head", [(0, 0), (1, 0), (2, 1), (3, 1)])
def test_reconstruct_single_head_qk_maps_gqa_heads(tmp_path, head, expected_kv_head):
    write_synthetic_cache(tmp_path)
    q, k, info = analyze.reconstruct_single_head_qk(
        cache_dir=tmp_path, layer=0, head=head, device=torch.device("cpu"), compute_dtype=torch.float32
    )
    assert q.shape == (5, 2)
    assert k.shape == (5, 2)
    assert info["kv_head"] == expected_kv_head
    assert info["num_key_value_groups"] == 2


def test_summarize_and_window_received_match_dense_reference(tmp_path):
    _, spec = write_synthetic_cache(tmp_path)
    q, k, info = analyze.reconstruct_single_head_qk(
        cache_dir=tmp_path, layer=0, head=3, device=torch.device("cpu"), compute_dtype=torch.float32
    )
    selected, row_path = analyze.summarize_selected_rows(
        q=q,
        k=k,
        info=info,
        cache_dir=tmp_path,
        spec=spec,
        layer=0,
        head=3,
        key_block_size=2,
        topk=2,
        save_full_rows=True,
    )
    assert row_path is not None and row_path.exists()
    assert selected["query_positions_by_label"] == {"last": [4], "middle": [2]}
    assert set(selected["rows"].keys()) == {"2", "4"}

    q_positions = [2, 4]
    probs, entropy, span_mass, local_mass = dense_stats(
        q, k, q_positions, analyze.normalize_spans(spec, 5), key_padding_mask=torch.ones(5, dtype=torch.bool), scaling=info["scaling"], local_windows=[1, 3]
    )
    for row_index, pos in enumerate(q_positions):
        out = selected["rows"][str(pos)]
        assert out["entropy_nats"] == pytest.approx(float(entropy[row_index]), abs=1e-6)
        assert out["span_mass"]["first_two"] == pytest.approx(float(span_mass["first_two"][row_index]), abs=1e-6)
        assert out["local_window_mass"]["3"] == pytest.approx(float(local_mass["3"][row_index]), abs=1e-6)

    received = analyze.average_window_received(
        q=q, k=k, info=info, cache_dir=tmp_path, spec=spec, key_block_size=2, query_block_size=1
    )
    # query_start=2, query_end=None covers positions 2,3,4.
    _, _, dense_received, _ = dense_stats(
        q,
        k,
        [2, 3, 4],
        {"target": [(0, 2)]},
        key_padding_mask=torch.ones(5, dtype=torch.bool),
        scaling=info["scaling"],
    )
    assert received["first_two_later"]["num_queries"] == 3
    assert received["first_two_later"]["mean_mass"] == pytest.approx(float(dense_received["target"].mean()), abs=1e-6)


def test_analyze_main_writes_json(tmp_path, monkeypatch):
    write_synthetic_cache(tmp_path)
    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_qk_qwen3.py",
            "--cache-dir",
            str(tmp_path),
            "--layer",
            "0",
            "--head",
            "1",
            "--device",
            "cpu",
            "--key-block-size",
            "2",
            "--query-block-size",
            "1",
            "--topk",
            "2",
            "--out-json",
            str(out_json),
        ],
    )
    analyze.main()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["layer"] == 0
    assert payload["head"] == 1
    assert payload["kv_head"] == 0
    assert "selected_query_rows" in payload
    assert "window_received" in payload


def test_invalid_reconstruction_metadata_is_rejected(tmp_path):
    write_synthetic_cache(tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    meta["model_config"]["num_attention_heads"] = 3
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        analyze.reconstruct_single_head_qk(
            cache_dir=tmp_path, layer=0, head=0, device=torch.device("cpu"), compute_dtype=torch.float32
        )


def test_reconstructed_tiny_qwen3_attention_matches_hf_eager_attention(tmp_path):
    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(tiny_qwen3_config())
    model.config._attn_implementation = "eager"
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    write_tiny_qwen3_capture_cache(tmp_path, model, input_ids, attention_mask, position_ids)

    handles = capture.install_qk_hooks(model=model, target_layers=[0], out_dir=tmp_path, save_dtype=torch.float32)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    hf_attention = outputs.attentions[0][0].detach().cpu()
    assert hf_attention.shape == (4, 5, 5)

    for head in range(model.config.num_attention_heads):
        q, k, info = analyze.reconstruct_single_head_qk(
            cache_dir=tmp_path,
            layer=0,
            head=head,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )
        stats = analyze.compute_query_block_stats(
            q_block=q,
            k=k,
            query_positions=torch.arange(input_ids.shape[1]),
            spans={},
            key_padding_mask=attention_mask[0].bool(),
            scaling=float(info["scaling"]),
            key_block_size=2,
            topk=0,
            local_windows=(),
            return_rows=True,
        )
        assert torch.allclose(stats["rows"], hf_attention[head], atol=1e-6)


def test_reconstruct_rejects_sliding_window_layers(tmp_path):
    write_synthetic_cache(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"].update(
        {
            "use_sliding_window": True,
            "sliding_window": 2,
            "layer_types": ["sliding_attention"],
        }
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(NotImplementedError, match="sliding-window"):
        analyze.reconstruct_single_head_qk(
            cache_dir=tmp_path,
            layer=0,
            head=0,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )


def test_reconstruct_allows_full_attention_layer_when_global_sliding_window_is_set(tmp_path):
    write_synthetic_cache(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["model_config"].update(
        {
            "use_sliding_window": True,
            "sliding_window": 2,
            "layer_types": ["full_attention"],
        }
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    q, k, info = analyze.reconstruct_single_head_qk(
        cache_dir=tmp_path,
        layer=0,
        head=0,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    assert q.shape == (5, 2)
    assert k.shape == (5, 2)
    assert info["kv_head"] == 0


def test_reconstruct_reads_consolidated_qk_metadata(tmp_path):
    cache_dir = tmp_path / "input_0"
    cache_dir.mkdir()
    write_synthetic_cache(cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    spec = json.loads((cache_dir / "analysis_spec.json").read_text(encoding="utf-8"))
    tokens = json.loads((cache_dir / "tokens.json").read_text(encoding="utf-8"))
    (tmp_path / "qk_cache_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "qk_cache_metadata_v1",
                "examples": {
                    "input_0": {
                        "example_idx": 0,
                        "cache_dir": "input_0",
                        "metadata": metadata,
                        "analysis_spec": spec,
                        "tokens": tokens,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "metadata.json").unlink()
    (cache_dir / "analysis_spec.json").unlink()
    (cache_dir / "tokens.json").unlink()

    q, k, info = analyze.reconstruct_single_head_qk(
        cache_dir=cache_dir,
        layer=0,
        head=0,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )

    assert q.shape == (5, 2)
    assert k.shape == (5, 2)
    assert info["seq_len"] == 5
    assert analyze.load_cache_analysis_spec(cache_dir)["query_positions"]["last"] == [-1]
    assert analyze.load_cache_tokens(cache_dir) == tokens
