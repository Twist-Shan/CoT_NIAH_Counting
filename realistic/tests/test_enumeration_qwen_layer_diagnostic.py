"""Reject misleading reuse of the pre-N10 or differently selected baseline."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("qwen_layer_diagnostic", Path(__file__).resolve().parents[1] / "scripts/launch_enumeration_qwen_layer_diagnostic.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def evidence(tmp_path):
    selection = {"confirmation_used_for_selection": False, "cells": [
        {"model": "Qwen3-8B", "mode": "enumeration_index", "layer_one_based": 30, "confirmation_seeds": [1620, 1621]}]}
    path = tmp_path / "selection_manifest.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    digest = mod.sha(path)
    cfg = {"model": "Qwen3-8B", "mode": "enumeration_index", "reuse_layer_one_based": 30,
           "primary_selection_manifest_sha256": digest}
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"n10_selection_manifest_sha256": digest}), encoding="utf-8")
    return tmp_path, cfg, {"selected_seeds": [1620, 1621]}, runtime


def test_verified_n10_reference_is_accepted(evidence):
    assert mod.validate_n10_selection(*evidence)[1] == evidence[1]["primary_selection_manifest_sha256"]


@pytest.mark.parametrize("fault", ["selection_hash", "reference_layer", "seed_order", "old_runtime", "confirmation_selection"])
def test_incompatible_reference_is_rejected(evidence, fault):
    folder, cfg, cell, runtime = evidence
    if fault == "selection_hash":
        cfg["primary_selection_manifest_sha256"] = "0" * 64
    elif fault == "reference_layer":
        cfg["reuse_layer_one_based"] = 19
    elif fault == "seed_order":
        cell["selected_seeds"].reverse()
    elif fault == "old_runtime":
        runtime.write_text(json.dumps({"n10_selection_manifest_sha256": "old"}), encoding="utf-8")
    else:
        path = folder / "selection_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["confirmation_used_for_selection"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        cfg["primary_selection_manifest_sha256"] = mod.sha(path)
    with pytest.raises(ValueError):
        mod.validate_n10_selection(*evidence)
