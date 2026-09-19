from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_v5_native_causal_chain_v2_report.py"
SPEC = importlib.util.spec_from_file_location("v5_native_causal_chain_v2_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holm_is_monotone_in_sorted_raw_p_order() -> None:
    raw = [0.04, 0.001, 0.03, 0.2]
    adjusted = MODULE.holm(raw)
    ordered = [adjusted[index] for index in sorted(range(len(raw)), key=raw.__getitem__)]
    assert ordered == sorted(ordered)
    assert adjusted[1] == 0.004
    assert adjusted[3] == 0.2


def test_registered_alignment_targets_are_symmetric() -> None:
    assert sum(MODULE.EXPECTED_GAPS.values()) == 56
    assert MODULE.EXPECTED_GAPS[-1] == MODULE.EXPECTED_GAPS[1] == 24
    assert len(MODULE.EXPECTED_MARKER_STRATA) == 10
