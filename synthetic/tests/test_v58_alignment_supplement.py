from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v58_alignment_core import Engine, changed_source, controls_for, fit_space, make_record, paired_bootstrap, rank_discovery, removal
from test_v58_commit_query import fixture_model
from synthetic_counting_v20.data import character_token


def fixture():
    model, vocab, example, item = fixture_model()
    example = replace(example, seq_tokens=example.seq_tokens + [character_token(c) for c in "  a  b  c   "])
    # Only these added spaces are ordinary; preserve original registered count.
    example = replace(example, seq_tokens=example.seq_tokens[:10]+[character_token(" ")]*12)
    return model, vocab, make_record(example, vocab, "thinking", "confirmation", 0)


def test_norm_and_orthogonality_match_realized_projection():
    rng = np.random.default_rng(3)
    space = fit_space(rng.normal(size=(100, 64)), np.repeat(np.arange(1, 11), 10))
    x = torch.from_numpy(rng.normal(size=64)).float()
    a, o = removal(x, space, "aligned"), removal(x, space, "orthogonal")
    assert torch.allclose(a.norm(), o.norm(), atol=1e-5)
    assert (space["u"] @ o).norm() < 1e-5


def test_no_fake_control_repeats():
    bank = [(4, i) for i in range(4)]
    assert len(controls_for(bank, 4)) == 1
    controls = controls_for(bank, 2)
    assert len(controls) == 3
    assert len({tuple(x) for x in controls}) == 3
    assert all(not set(x) & set(bank[:2]) for x in controls)


def test_constant_embedding_has_valid_control_basis_without_false_geometry():
    space = fit_space(np.zeros((100, 64)), np.repeat(np.arange(1, 11), 10))
    assert (space["u"] @ space["v"].T).norm() < 1e-5
    assert removal(torch.zeros(64), space, "orthogonal").norm() == 0


def test_self_patch_and_full_embedding_restoration():
    model, vocab, r = fixture()
    e = Engine(model, vocab, [(1, 0)], 2)
    c = e.run([r], capture=True)[0]
    a = {"patch": [(2, [r["q"]], c["hidden"][2][[r["q"]]])]}
    assert abs(e.run([r], [a])[0]["margin"]-c["margin"]) < 1e-5
    a = {"seq": changed_source(r, "needles"), "patch": [(0, r["needles"], c["hidden"][0][r["needles"]])]}
    assert abs(e.run([r], [a])[0]["margin"]-c["margin"]) < 1e-5


def test_final_query_mask_cannot_change_later_fixed_token_state():
    model, vocab, r = fixture()
    e = Engine(model, vocab, [(4, 0)], 2)
    c = e.run([r], capture=True)[0]
    a = {"mask": [(4, 0), (4, 1)], "mask_queries": [r["markers"][-1]-1]}
    p = e.run([r], [a], capture=True)[0]
    assert torch.equal(c["hidden"][4][r["markers"][-1]], p["hidden"][4][r["markers"][-1]])


def test_discovery_ranking_shape_and_finite_scores():
    model, vocab, r = fixture()
    ranks = rank_discovery(model, vocab, [r])
    assert len(ranks["targeted"]) == len(ranks["broad"]) == 32
    assert all(np.isfinite(score) for l, h, score in ranks["targeted"])


def test_bootstrap_counts_blocks_not_trial_repeats():
    x = paired_bootstrap([1, 1, 0, 0], [0, 0, 1, 1])
    assert x["effect"] == .5 and x["blocks"] == 2 and x["pairs"] == 4
