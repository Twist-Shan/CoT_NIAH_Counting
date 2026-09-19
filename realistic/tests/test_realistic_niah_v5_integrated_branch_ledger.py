from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEDGER = _module(
    "integrated_branch_ledger",
    ROOT / "scripts" / "finalize_realistic_niah_v5_integrated_branch_ledger.py",
)
REPORT = _module(
    "native_count_chain_report_for_ledger",
    ROOT / "scripts" / "build_v5_native_count_chain_report.py",
)
ASSEMBLER = _module(
    "native_count_chain_evidence_assembler",
    ROOT / "scripts" / "assemble_realistic_niah_v5_count_chain_evidence.py",
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metric(estimate: float = 0.0) -> dict[str, float]:
    return {"estimate": estimate, "ci_low": estimate - 0.1, "ci_high": estimate + 0.1}


def _restoration_claims() -> dict:
    return {
        "gates": {
            "targeted_receiver_damage": _metric(),
            "clean_state_restores_selected_receiver": _metric(),
            "readout_cut_occludes_restoration": _metric(),
            "cut_restoration_residual_equivalence": _metric(),
        }
    }


def _target_meta(model: str) -> dict:
    return {
        "model_label": model,
        "bank_size": 125 if model == "Qwen3-8B" else 8,
        "selection_rank_used": False,
    }


def _make_spec(tmp_path: Path, *, pass_final: bool = False, branch_count: int = 4) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    names = list(LEDGER.EXPECTED_BRANCHES[:branch_count])
    branches = []
    for index, name in enumerate(names):
        status = "PASS" if pass_final and index == len(names) - 1 else "DISCOVERY_GATE_FAIL"
        complete = {
            "model_label": "Qwen3-8B",
            "status": status,
            "discovery_claim_gates": _restoration_claims(),
        }
        discovery = {
            "status": "PASS",
            "seed_count": 20,
            "applicable_seed_count": 20,
            "selection_rank_used": False,
            "applicable_sample_count": 1,
            "planned_sample_count": 1,
        }
        complete_path = tmp_path / f"{name}_complete.json"
        discovery_path = tmp_path / f"{name}_discovery.json"
        _write(complete_path, complete)
        _write(discovery_path, discovery)
        branch = {
            "name": name,
            "complete": complete_path.name,
            "discovery_audit": discovery_path.name,
            "confirmation_audit": None,
        }
        if "restoration" in name:
            branch["mediator_geometry"] = (
                "full_span" if name == "fullspan_restoration" else "suffix8"
            )
        if status == "PASS":
            confirmation = {
                "status": "PASS",
                "seed_count": 10,
                "applicable_seed_count": 10,
                "selection_rank_used": False,
                "applicable_sample_count": 1,
                "planned_sample_count": 1,
            }
            confirmation_path = tmp_path / f"{name}_confirmation.json"
            _write(confirmation_path, confirmation)
            complete["confirmation_claim_gates"] = _restoration_claims()
            _write(complete_path, complete)
            branch["confirmation_audit"] = confirmation_path.name
        branches.append(branch)
    return {"model_label": "Qwen3-8B", "branches": branches}


def test_finalizer_accepts_all_four_terminal_failures(tmp_path: Path) -> None:
    integrated, discovery, confirmation = LEDGER.finalize(
        _make_spec(tmp_path), base=tmp_path
    )
    assert integrated["status"] == "PRE_REGISTERED_BRANCHES_EXHAUSTED"
    assert integrated["pre_registered_branches_exhausted"] is True
    assert integrated["mediator_geometry"] == "full_span"
    assert len(integrated["branch_outcomes"]) == 4
    assert discovery["seed_count"] == 20
    assert confirmation is None
    assert "unsupported bridge" in REPORT._chain(
        "Qwen3-8B", integrated, _target_meta("Qwen3-8B")
    )


def test_finalizer_accepts_confirmed_final_branch(tmp_path: Path) -> None:
    integrated, discovery, confirmation = LEDGER.finalize(
        _make_spec(tmp_path, pass_final=True), base=tmp_path
    )
    assert integrated["status"] == "PASS"
    assert integrated["integrated_mediator_restoration_pass"] is True
    assert integrated["passed_branch"] == "fullspan_restoration"
    assert discovery["seed_count"] == 20
    assert confirmation is not None and confirmation["seed_count"] == 10


def test_finalizer_rejects_premature_exhaustion(tmp_path: Path) -> None:
    try:
        LEDGER.finalize(_make_spec(tmp_path, branch_count=3), base=tmp_path)
    except ValueError as exc:
        assert "all four" in str(exc)
    else:
        raise AssertionError("Premature branch exhaustion must be rejected")


def test_report_contract_accepts_audited_exhaustion(tmp_path: Path) -> None:
    integrated, discovery, _confirmation = LEDGER.finalize(
        _make_spec(tmp_path), base=tmp_path
    )
    evidence = {}
    for model in REPORT.MODELS:
        model_integrated = dict(integrated)
        model_integrated["model_label"] = model
        evidence[f"{model}:targeted"] = {"status": "PASS"}
        evidence[f"{model}:targeted_plan_meta"] = _target_meta(model)
        evidence[f"{model}:readout"] = {"status": "PASS"}
        evidence[f"{model}:integrated"] = model_integrated
        evidence[f"{model}:integrated_discovery_audit"] = discovery
    REPORT._assert_contract(evidence)


def test_report_renders_broken_chain_for_exhausted_models(tmp_path: Path) -> None:
    integrated, discovery, _confirmation = LEDGER.finalize(
        _make_spec(tmp_path), base=tmp_path
    )
    targeted_gates = {
        "clean_endpoint_adequacy": _metric(1.0),
        "targeted_bank_changes_final_count": _metric(0.4),
        "retrieval_failure_propagates_to_count": _metric(0.6),
    }
    qwen_readout = {
        "storage_main_effect": _metric(4.0),
        "residual_relay_contribution": _metric(2.0),
        "direct_reread_contribution_after_relay": _metric(1.0),
        "joint_cut_residual_equivalence": _metric(0.05),
    }
    gemma_readout = {
        "storage_main_effect": _metric(3.0),
        "trace_source_specific_occlusion": _metric(-2.0),
        "trace_mask_residual_equivalence": _metric(0.04),
    }
    evidence = {}
    for model, readout_gates in (
        ("Qwen3-8B", qwen_readout),
        ("Gemma4-E4B", gemma_readout),
    ):
        model_integrated = dict(integrated)
        model_integrated["model_label"] = model
        evidence[f"{model}:targeted"] = {
            "status": "PASS",
            "confirmation": {"gates": targeted_gates},
        }
        evidence[f"{model}:targeted_plan_meta"] = _target_meta(model)
        evidence[f"{model}:readout"] = {
            "status": "PASS",
            "confirmation_claim_gates": {"gates": readout_gates},
        }
        evidence[f"{model}:integrated"] = model_integrated
        evidence[f"{model}:integrated_discovery_audit"] = discovery
    REPORT._assert_contract(evidence)
    document = REPORT.build(evidence, {})
    assert document.count("unsupported bridge") == 2
    assert "不支持完整串行中介链" in document


def test_report_renders_model_specific_mixed_outcome(tmp_path: Path) -> None:
    q_integrated, q_discovery, q_confirmation = LEDGER.finalize(
        _make_spec(tmp_path / "qwen", pass_final=True), base=tmp_path / "qwen"
    )
    g_integrated, g_discovery, _ = LEDGER.finalize(
        _make_spec(tmp_path / "gemma"), base=tmp_path / "gemma"
    )
    targeted_gates = {
        "clean_endpoint_adequacy": _metric(1.0),
        "targeted_bank_changes_final_count": _metric(0.4),
        "retrieval_failure_propagates_to_count": _metric(0.6),
    }
    evidence = {
        "Qwen3-8B:targeted": {"status": "PASS", "confirmation": {"gates": targeted_gates}},
        "Gemma4-E4B:targeted": {"status": "PASS", "confirmation": {"gates": targeted_gates}},
        "Qwen3-8B:targeted_plan_meta": _target_meta("Qwen3-8B"),
        "Gemma4-E4B:targeted_plan_meta": _target_meta("Gemma4-E4B"),
        "Qwen3-8B:readout": {
            "status": "PASS",
            "confirmation_claim_gates": {
                "gates": {
                    "storage_main_effect": _metric(4.0),
                    "residual_relay_contribution": _metric(2.0),
                    "direct_reread_contribution_after_relay": _metric(1.0),
                    "joint_cut_residual_equivalence": _metric(0.05),
                }
            },
        },
        "Gemma4-E4B:readout": {
            "status": "PASS",
            "confirmation_claim_gates": {
                "gates": {
                    "storage_main_effect": _metric(3.0),
                    "trace_source_specific_occlusion": _metric(-2.0),
                    "trace_mask_residual_equivalence": _metric(0.04),
                }
            },
        },
        "Qwen3-8B:integrated": q_integrated,
        "Gemma4-E4B:integrated": {**g_integrated, "model_label": "Gemma4-E4B"},
        "Qwen3-8B:integrated_discovery_audit": q_discovery,
        "Qwen3-8B:integrated_confirmation_audit": q_confirmation,
        "Gemma4-E4B:integrated_discovery_audit": g_discovery,
    }
    REPORT._assert_contract(evidence)
    document = REPORT.build(evidence, {})
    assert document.count("unsupported bridge") == 1
    assert "Qwen3-8B 获得完整 confirmation 链" in document
    assert "Gemma4-E4B 只确认了链条两端" in document


def test_evidence_assembler_audits_all_components_and_branches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "evidence"
    for model in ASSEMBLER.MODELS:
        plan = (
            source
            / "work/v5_native_count_stream/targeted_count_chain_20d10c_20260821"
            / model
            / "frozen_targeted_count_plan.csv"
        )
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(
            "bank_size,condition\n"
            + ("125" if model == "Qwen3-8B" else "8")
            + ",selected_bank\n",
            encoding="utf-8",
        )
        for label, paths in ASSEMBLER._component_paths(source, model).items():
            complete, discovery, confirmation = paths
            _write(complete, {"status": "PASS", "model_label": model})
            _write(
                discovery,
                {
                    "status": "PASS",
                    "seed_count": 20,
                    "selection_rank_used": False,
                },
            )
            _write(
                confirmation,
                {"status": "PASS", "seed_count": 10, "selection_rank_used": False},
            )
        for name, branch_root, _geometry in ASSEMBLER._branch_roots(source, model):
            prefix = "restoration" if "restoration" in name else "integrated_bridge"
            _write(
                branch_root / f"{prefix}_complete.json",
                {
                    "status": "DISCOVERY_GATE_FAIL",
                    "model_label": model,
                    "discovery_claim_gates": _restoration_claims(),
                },
            )
            _write(
                branch_root / f"{prefix}_analysis_discovery/audit.json",
                {
                    "status": "PASS",
                    "seed_count": 20,
                    "applicable_seed_count": 20,
                    "selection_rank_used": False,
                    "applicable_sample_count": 1,
                    "planned_sample_count": 1,
                },
            )
    manifest = ASSEMBLER.assemble(source, output)
    assert manifest["status"] == "PASS"
    assert manifest["models"] == {
        "Qwen3-8B": "PRE_REGISTERED_BRANCHES_EXHAUSTED",
        "Gemma4-E4B": "PRE_REGISTERED_BRANCHES_EXHAUSTED",
    }
    for model in ASSEMBLER.MODELS:
        integrated = json.loads(
            (output / model / "integrated_complete.json").read_text(encoding="utf-8")
        )
        assert integrated["pre_registered_branches_exhausted"] is True
        assert len(integrated["branch_outcomes"]) == 4
