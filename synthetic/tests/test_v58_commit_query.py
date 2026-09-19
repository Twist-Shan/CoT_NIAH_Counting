import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_v58_commit_query import make_condition_states, forward, pair_metadata, routing_metrics
from synthetic_counting_v20.config import preset_config
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v20.model import build_model
from synthetic_counting_v20.v10_port_analysis import _residual_patch


def fixture_model():
    cfg = replace(preset_config("main", device="cpu"), n_layer=4, n_head=8, n_embd=64, n_inner=128, trace_format="separator")
    vocab = V20Vocab.build(cfg, "abc ")
    chars = "abcabcabca"
    tokens = [character_token(c) for c in chars]
    example = V20Example("counting_task", tokens, "test", 0, 10, "test_prompt",
                         set_id="abc", needle_characters=("a", "b", "c"), rendered_set_order=("a", "b", "c"),
                         needle_positions=tuple(range(10)), needle_markers=tuple(tokens), count=10)
    return build_model(cfg, vocab).eval(), vocab, example, render_v20(example, vocab, "thinking")


def test_subspace_transplant_and_control_norms():
    torch.manual_seed(7)
    basis = torch.linalg.qr(torch.randn(32, 3)).Q.T
    receiver, donor = torch.randn(32), torch.randn(32)
    vectors, norms = make_condition_states(receiver, donor, basis, "pair", 1)
    assert torch.allclose(basis @ vectors["count_subspace_transplant"], basis @ donor, atol=1e-5)
    assert torch.allclose(basis @ vectors["norm_matched_orthogonal_patch"], basis @ receiver, atol=1e-5)
    assert abs(float((vectors["norm_matched_orthogonal_patch"]-receiver).norm())-norms["count_delta_norm"])<1e-5
    for i in range(3):
        assert abs(float((vectors[f"full_norm_orthogonal_r{i}"]-receiver).norm())-norms["full_delta_norm"])<1e-5


def test_adjacent_pair_ordinal_and_prompt_position():
    _, _, example, item = fixture_model()
    meta = pair_metadata(example, item, 4, 1, "confirmation")
    assert meta["receiver_successor"] == 5 and meta["donor_successor"] == 6
    assert meta["receiver_source_position"] == item.prompt_needle_positions[4]
    assert meta["donor_commit_position"] == item.spans.trace_marker_positions[4]


def test_aligned_filler_preserves_needles_and_matches_absolute_item_position():
    from run_v58_aligned_item_query import aligned_donor
    _,vocab,example,_=fixture_model()
    example=replace(example,seq_tokens=example.seq_tokens+[character_token(" ")]*4)
    item=render_v20(example,vocab,"thinking")
    for offset in [-1,1]:
        donor=aligned_donor(example,offset)
        assert donor.needle_markers==example.needle_markers
        assert tuple(donor.seq_tokens[i] for i in donor.needle_positions)==example.needle_markers
        rendered=render_v20(donor,vocab,"thinking")
        assert rendered.spans.trace_marker_positions[4+offset-1]==item.spans.trace_marker_positions[4-1]


def test_late_commit_cannot_change_next_query_qk_ranking():
    model,vocab,example,item = fixture_model()
    meta = pair_metadata(example,item,4,1,"confirmation")
    seq=item.input_ids[:meta["commit_position"]+1]+[vocab.token_to_id["<Sep>"]]
    clean=forward(model,vocab,[seq],attention=True,hidden=True)
    case={"meta":meta,"positions":item.prompt_needle_positions}
    cm=routing_metrics(clean,0,len(seq)-1,case)
    for layer in (3,4):
        vector=clean.hidden_states[layer][0,meta["commit_position"]]+torch.arange(64)*.07
        with _residual_patch(model,layer,[meta["commit_position"]],vector[None]):
            patched=forward(model,vocab,[seq],attention=True)
        pm=routing_metrics(patched,0,len(seq)-1,case)
        for head in (5,0,1,4):
            assert abs(pm[f"L4H{head}_qk_margin"]-cm[f"L4H{head}_qk_margin"])<2e-5
        if layer==4:
            assert torch.equal(clean.logits[0,-1],patched.logits[0,-1])
            assert pm["routing_y"]==cm["routing_y"]


def test_two_token_self_patch_does_not_touch_next_query():
    from run_v58_commit_query import patch_batch
    model,vocab,example,item=fixture_model()
    meta=pair_metadata(example,item,4,1,"confirmation")
    seq=item.input_ids[:meta["commit_position"]+1]+[vocab.token_to_id["<Sep>"]]
    clean=forward(model,vocab,[seq],hidden=True)
    positions=[meta["commit_position"]-1,meta["commit_position"]]
    case={"meta":meta,"condition":"self_patch","patch_span":2,
          "vector":clean.hidden_states[1][0,positions].flatten()}
    with patch_batch(model,1,[case],"cpu"):
        patched=forward(model,vocab,[seq])
    assert torch.allclose(clean.logits,patched.logits,atol=1e-6)


def test_checkpoint_provenance_uses_indexed_dense_fallback(tmp_path):
    import pandas as pd
    from run_v58_commit_query import checkpoint_source
    root=tmp_path/"checkpoints/rope/thinking"
    root.mkdir(parents=True)
    (root/"shard.pt").touch()
    pd.DataFrame([{"step":10000,"shard":"shard.pt"}]).to_csv(root/"snapshot_index.csv",index=False)
    assert checkpoint_source(tmp_path,10000)==root/"shard.pt"
