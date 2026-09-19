from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_realistic_niah_v4_causal_v2.py"
SELECTION_PATH = (
    ROOT
    / "configs"
    / "realistic_niah_v4_4_ablation_seed_extrapolation_selection.json"
)
CONFIG_PATH = (
    ROOT / "configs" / "realistic_niah_v4_4_ablation_seed_extrapolation.json"
)
LEGACY_SELECTION_PATH = (
    ROOT
    / "configs"
    / "realistic_niah_v4_causal_v2_ablation_confirmation_selection.json"
)


def _runner_module():
    name = "_seed_extrapolation_runner_for_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normalized_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_registered_seed_grid_is_a_disjoint_contiguous_suffix() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    seeds = selection["confirmation_design"]["seeds"]
    assert seeds == list(range(1296, 1316))
    assert seeds == config["seeds"]
    assert min(seeds) > selection["discovery_source"]["prior_seed_end_inclusive"]
    assert selection["confirmation_design"]["counts"] == [1, 2, 3, 4, 5]


def test_selection_sources_are_hash_locked() -> None:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    source = ROOT / selection["discovery_source"]["table"]
    candidates = ROOT / selection["discovery_source"]["candidate_summary_table"]
    assert _normalized_sha(source) == selection["discovery_source"]["table_sha256"]
    assert (
        _normalized_sha(candidates)
        == selection["discovery_source"]["candidate_summary_table_sha256"]
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [("Qwen3-8B", (2, 4)), ("Gemma4-E4B", (1, 2))],
)
def test_runner_reads_two_frozen_doses_without_reselection(
    model: str, expected: tuple[int, int]
) -> None:
    runner = _runner_module()
    _, plan = runner._ablation_confirmation_plan(
        SELECTION_PATH,
        model_label=model,
        repo_root=ROOT,
    )
    assert plan["top_ns"] == expected
    assert plan["seeds"] == tuple(range(1296, 1316))
    assert plan["counts"] == (1, 2, 3, 4, 5)
    assert plan["selection_status"] == "frozen_before_seed_extrapolation"
    assert plan["evidence_split"] == "independent_seed_extrapolation"
    assert plan["emit_dual_population"] is True


def test_runner_rejects_any_reused_seed(tmp_path: Path) -> None:
    runner = _runner_module()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    selection["confirmation_design"]["seeds"] = list(range(1295, 1315))
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps a previously used seed range"):
        runner._ablation_confirmation_plan(
            path,
            model_label="Qwen3-8B",
            repo_root=ROOT,
        )


def test_legacy_single_dose_confirmation_still_parses() -> None:
    runner = _runner_module()
    _, qwen = runner._ablation_confirmation_plan(
        LEGACY_SELECTION_PATH,
        model_label="Qwen3-8B",
        repo_root=ROOT,
    )
    _, gemma = runner._ablation_confirmation_plan(
        LEGACY_SELECTION_PATH,
        model_label="Gemma4-E4B",
        repo_root=ROOT,
    )
    assert qwen["top_ns"] == (8,)
    assert gemma["top_ns"] == (6,)
    assert qwen["emit_dual_population"] is False
    assert gemma["emit_dual_population"] is False
