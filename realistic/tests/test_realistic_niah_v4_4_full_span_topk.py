from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from realistic_niah_v4_4_4.gemma_full_span_residual_spec import (
    GemmaFullSpanResidualConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    path = ROOT / "scripts" / "run_realistic_niah_v4_causal_v2.py"
    spec = importlib.util.spec_from_file_location("_full_span_topk_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_span_ablation_selection_freezes_geometric_grid() -> None:
    runner = _runner_module()
    selection = (
        ROOT
        / "configs"
        / "realistic_niah_v4_4_full_span_topk_confirmation_selection.json"
    )
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        _payload, plan = runner._ablation_confirmation_plan(
            selection, model_label=model, repo_root=ROOT
        )
        assert plan["top_ns"] == (1, 2, 4, 8, 16, 32)
        assert plan["seeds"] == tuple(range(1316, 1336))
        assert plan["emit_dual_population"] is True


def test_gemma_full_span_residual_configs_are_nested() -> None:
    previous: tuple[tuple[int, int], ...] = ()
    for top_n in (1, 2, 4, 8, 16):
        config = GemmaFullSpanResidualConfig.from_json(
            ROOT
            / "configs"
            / f"realistic_niah_v4_4_4_gemma_full_span_residual_k{top_n}.json"
        )
        sites = tuple((site.layer, site.head) for site in config.candidate_sites)
        assert len(sites) == top_n
        assert sites[: len(previous)] == previous
        assert all(site.layer < min(config.candidate_mediator_layers) for site in config.candidate_sites)
        previous = sites
