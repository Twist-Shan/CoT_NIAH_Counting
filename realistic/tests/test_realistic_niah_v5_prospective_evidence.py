from __future__ import annotations

import csv
import hashlib
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


ASSEMBLER = _module(
    "prospective_count_chain_evidence",
    ROOT / "scripts/assemble_realistic_niah_v5_prospective_count_chain_evidence.py",
)
REPORT = _module(
    "prospective_count_chain_report",
    ROOT / "scripts/build_v5_native_count_chain_report.py",
)
COMPLETION = _module(
    "prospective_count_chain_completion_audit",
    ROOT / "scripts/audit_realistic_niah_v5_prospective_completion.py",
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metric(value: float) -> dict[str, float]:
    return {"estimate": value, "ci_low": value - 0.05, "ci_high": value + 0.05}


def _build_report(evidence_root: Path, report: Path) -> None:
    evidence, hashes = REPORT._read_evidence(evidence_root)
    REPORT._assert_contract(evidence)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(REPORT.build(evidence, hashes), encoding="utf-8")
    _write(
        report.with_suffix(".manifest.json"),
        {
            "status": "PASS",
            "output": str(report.resolve()),
            "output_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "evidence_sha256": hashes,
        },
    )


def _protocol(model: str, k: int, digest: str) -> dict:
    return {
        "model_label": model,
        "bank": {"size": k, "selected_bank_sha256": digest},
        "stages": [
            {
                "name": "targeted_retrieval_to_final_count",
                "root": f"runs/{model}/endpoint",
            },
            {
                "name": "targeted_retrieval_to_terminal_state_to_readout",
                "root": f"runs/{model}/bridge",
            },
        ],
    }


def _make_terminal_pass(repo: Path, model: str, k: int, digest: str) -> Path:
    endpoint = repo / f"runs/{model}/endpoint"
    bridge = repo / f"runs/{model}/bridge"
    plan = endpoint / "frozen_targeted_count_plan.csv"
    plan.parent.mkdir(parents=True, exist_ok=True)
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "bank_size", "bank_sha256"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition": "selected_bank",
                "bank_size": k,
                "bank_sha256": digest,
            }
        )
    _write(
        plan.with_suffix(".audit.json"),
        {
            "status": "FROZEN_NEW",
            "model_label": model,
            "bank_size": k,
            "selected_bank_sha256": digest,
            "outcome_blind": True,
            "selection_rank_used": False,
            "historical_artifacts_modified": False,
        },
    )
    _write(
        endpoint / "targeted_count_complete.json",
        {
            "status": "PASS",
            "model_label": model,
            "confirmation": {
                "gates": {
                    "clean_endpoint_adequacy": _metric(0.9),
                    "targeted_bank_changes_final_count": _metric(0.4),
                    "retrieval_failure_propagates_to_count": _metric(0.5),
                }
            },
        },
    )
    for phase, seeds in (("discovery", 20), ("confirmation", 10)):
        _write(
            endpoint / f"targeted_count_analysis_{phase}/audit.json",
            {
                "status": "PASS",
                "seed_count": seeds,
                "selection_rank_used": False,
            },
        )
    _write(
        bridge / "restoration_complete.json",
        {
            "status": "PASS",
            "model_label": model,
            "targeted_bank_size": k,
            "targeted_bank_sha256": digest,
            "integrated_mediator_restoration_pass": True,
            "mediator_geometry": "full_span",
            "confirmation_claim_gates": {
                "gates": {
                    "targeted_receiver_damage": _metric(2.0),
                    "clean_state_restores_selected_receiver": _metric(1.0),
                    "readout_cut_occludes_restoration": _metric(0.8),
                    "cut_restoration_residual_equivalence": _metric(0.05),
                }
            },
        },
    )
    for phase, seeds in (("discovery", 20), ("confirmation", 10)):
        _write(
            bridge / f"restoration_analysis_{phase}/audit.json",
            {
                "status": "PASS",
                "seed_count": seeds,
                "applicable_seed_count": seeds,
                "selection_rank_used": False,
                "applicable_sample_count": seeds,
                "planned_sample_count": seeds,
            },
        )
    protocol_path = repo / f"configs/{model}.json"
    _write(protocol_path, _protocol(model, k, digest))
    return protocol_path


def _make_historical_evidence(root: Path) -> None:
    _write(root / "evidence_manifest.json", {"status": "PASS"})
    _write(
        root / "Qwen3-8B/readout_complete.json",
        {
            "status": "PASS",
            "confirmation_claim_gates": {
                "gates": {
                    "storage_main_effect": _metric(3.0),
                    "residual_relay_contribution": _metric(1.0),
                    "direct_reread_contribution_after_relay": _metric(1.0),
                    "joint_cut_residual_equivalence": _metric(0.05),
                }
            },
        },
    )
    _write(
        root / "Gemma4-E4B/readout_complete.json",
        {
            "status": "PASS",
            "confirmation_claim_gates": {
                "gates": {
                    "storage_main_effect": _metric(3.0),
                    "trace_source_specific_occlusion": _metric(1.0),
                    "trace_mask_residual_equivalence": _metric(0.05),
                }
            },
        },
    )


def _make_complete_historical_primary(root: Path) -> None:
    _make_historical_evidence(root)
    targeted_gates = {
        "clean_endpoint_adequacy": _metric(0.9),
        "targeted_bank_changes_final_count": _metric(0.4),
        "retrieval_failure_propagates_to_count": _metric(0.5),
    }
    restoration_gates = {
        "targeted_receiver_damage": _metric(1.0),
        "clean_state_restores_selected_receiver": _metric(0.3),
        "readout_cut_occludes_restoration": _metric(0.2),
        "cut_restoration_residual_equivalence": _metric(0.4),
    }
    branch_names = [
        "exact_query_transfer",
        "persistent_transfer",
        "suffix8_restoration",
        "fullspan_restoration",
    ]
    for model, k in (("Qwen3-8B", 125), ("Gemma4-E4B", 8)):
        target = root / model
        _write(
            target / "targeted_complete.json",
            {
                "status": "PASS",
                "confirmation": {"gates": targeted_gates},
            },
        )
        _write(
            target / "targeted_plan_meta.json",
            {
                "model_label": model,
                "bank_size": k,
                "selection_rank_used": False,
            },
        )
        _write(
            target / "integrated_complete.json",
            {
                "status": "PRE_REGISTERED_BRANCHES_EXHAUSTED",
                "model_label": model,
                "integrated_serial_bridge_pass": False,
                "integrated_mediator_restoration_pass": False,
                "pre_registered_branches_exhausted": True,
                "mediator_geometry": "full_span",
                "branch_outcomes": [
                    {"name": name, "status": "DISCOVERY_GATE_FAIL"}
                    for name in branch_names
                ],
                "final_branch_claim_gates": {"gates": restoration_gates},
            },
        )
        _write(
            target / "integrated_discovery_audit.json",
            {
                "status": "PASS",
                "seed_count": 20,
                "applicable_seed_count": 20,
                "selection_rank_used": False,
                "applicable_sample_count": 20,
                "planned_sample_count": 20,
            },
        )


def _make_terminal_discovery_negative(
    repo: Path, model: str, k: int, digest: str
) -> Path:
    endpoint = repo / f"runs/{model}/endpoint"
    plan = endpoint / "frozen_targeted_count_plan.csv"
    plan.parent.mkdir(parents=True, exist_ok=True)
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "bank_size", "bank_sha256"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition": "selected_bank",
                "bank_size": k,
                "bank_sha256": digest,
            }
        )
    _write(
        plan.with_suffix(".audit.json"),
        {
            "status": "FROZEN_NEW",
            "model_label": model,
            "bank_size": k,
            "selected_bank_sha256": digest,
            "outcome_blind": True,
            "selection_rank_used": False,
            "historical_artifacts_modified": False,
        },
    )
    _write(
        endpoint / "targeted_count_complete.json",
        {"status": "DISCOVERY_NEGATIVE"},
    )
    _write(
        endpoint / "targeted_count_analysis_discovery/audit.json",
        {"status": "PASS", "seed_count": 20, "selection_rank_used": False},
    )
    protocol_path = repo / f"configs/{model}.json"
    _write(protocol_path, _protocol(model, k, digest))
    return protocol_path


def _install_claim_contract(repo: Path) -> None:
    source = (
        ROOT
        / "configs/realistic_niah_v5_prospective_count_chain_claim_contract_v1.json"
    )
    target = (
        repo
        / "configs/realistic_niah_v5_prospective_count_chain_claim_contract_v1.json"
    )
    value = json.loads(source.read_text(encoding="utf-8"))
    value["models"]["Qwen3-8B"]["selected_bank_sha256"] = "qwen128"
    value["models"]["Gemma4-E4B"]["selected_bank_sha256"] = "gemma6"
    _write(target, value)
    bridge_source = ROOT / "src/realistic_niah_v5/integrated_bridge.py"
    bridge_target = repo / "src/realistic_niah_v5/integrated_bridge.py"
    bridge_target.parent.mkdir(parents=True, exist_ok=True)
    bridge_target.write_bytes(bridge_source.read_bytes())
    ledger_source = (
        ROOT / "configs/realistic_niah_v5_integrated_bridge_metadata_fix_v1.json"
    )
    ledger_target = (
        repo / "configs/realistic_niah_v5_integrated_bridge_metadata_fix_v1.json"
    )
    ledger_target.write_bytes(ledger_source.read_bytes())


def test_prospective_evidence_overlays_confirmed_same_bank_chains(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    output = tmp_path / "prospective"
    _make_historical_evidence(historical)
    _install_claim_contract(repo)
    protocols = [
        _make_terminal_pass(repo, "Qwen3-8B", 128, "qwen128"),
        _make_terminal_pass(repo, "Gemma4-E4B", 6, "gemma6"),
    ]

    manifest = ASSEMBLER.assemble(
        repo_root=repo,
        historical_evidence_root=historical,
        protocol_paths=protocols,
        output=output,
    )
    assert manifest["prospective_extensions"] == {
        "Qwen3-8B": "PASS",
        "Gemma4-E4B": "PASS",
    }
    evidence, hashes = REPORT._read_evidence(output)
    REPORT._assert_contract(evidence)
    document = REPORT.build(evidence, hashes)
    assert "K128" in document
    assert "K6" in document
    assert "Qwen 与 Gemma 均支持" in document
    assert "Result-independent provenance fix" in document
    assert document.count("prospective_same_bank_fullspan_restoration") == 0
    report = tmp_path / "report.html"
    _build_report(output, report)
    completion = COMPLETION.audit(output, report)
    assert completion["status"] == "PASS"
    assert completion["cross_model_full_chain_confirmed"] is True


def test_prospective_evidence_rejects_stale_metadata_fix_source(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    _make_historical_evidence(historical)
    _install_claim_contract(repo)
    protocols = [
        _make_terminal_pass(repo, "Qwen3-8B", 128, "qwen128"),
        _make_terminal_pass(repo, "Gemma4-E4B", 6, "gemma6"),
    ]
    (repo / "src/realistic_niah_v5/integrated_bridge.py").write_text(
        "# stale source\n", encoding="utf-8"
    )
    try:
        ASSEMBLER.assemble(
            repo_root=repo,
            historical_evidence_root=historical,
            protocol_paths=protocols,
            output=tmp_path / "prospective",
        )
    except ValueError as exc:
        assert "does not match metadata-fix ledger" in str(exc)
    else:
        raise AssertionError("A stale bridge source must fail evidence assembly")


def test_prospective_evidence_rejects_duplicate_model_protocols(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    _make_historical_evidence(historical)
    _install_claim_contract(repo)
    protocol = _make_terminal_pass(repo, "Qwen3-8B", 128, "qwen128")
    try:
        ASSEMBLER.assemble(
            repo_root=repo,
            historical_evidence_root=historical,
            protocol_paths=[protocol, protocol],
            output=tmp_path / "prospective",
        )
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate model protocols must be rejected")


def test_prospective_evidence_preserves_model_specific_exhaustion(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    output = tmp_path / "prospective"
    _make_complete_historical_primary(historical)
    _install_claim_contract(repo)
    protocols = [
        _make_terminal_pass(repo, "Qwen3-8B", 128, "qwen128"),
        _make_terminal_discovery_negative(repo, "Gemma4-E4B", 6, "gemma6"),
    ]

    manifest = ASSEMBLER.assemble(
        repo_root=repo,
        historical_evidence_root=historical,
        protocol_paths=protocols,
        output=output,
    )
    assert manifest["prospective_extensions"] == {
        "Qwen3-8B": "PASS",
        "Gemma4-E4B": "PROTOCOL_EXHAUSTED",
    }
    evidence, hashes = REPORT._read_evidence(output)
    REPORT._assert_contract(evidence)
    assert evidence["Qwen3-8B:targeted_plan_meta"]["bank_size"] == 128
    assert evidence["Gemma4-E4B:targeted_plan_meta"]["bank_size"] == 8
    document = REPORT.build(evidence, hashes)
    assert "Qwen3-8B 获得完整 confirmation 链" in document
    assert "Gemma4-E4B 只确认了链条两端" in document
    assert "PROTOCOL_EXHAUSTED" in document
    report = tmp_path / "report.html"
    _build_report(output, report)
    completion = COMPLETION.audit(output, report)
    assert completion["cross_model_full_chain_confirmed"] is False
    assert (
        completion["models"]["Qwen3-8B"]["classification"]
        == "MODEL_FULL_CHAIN_CONFIRMED"
    )
    assert (
        completion["models"]["Gemma4-E4B"]["classification"]
        == "PROSPECTIVE_BANK_ENDPOINT_NOT_SUPPORTED"
    )


def test_nonterminal_extension_does_not_create_partial_evidence_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    historical = tmp_path / "historical"
    output = tmp_path / "prospective"
    _make_complete_historical_primary(historical)
    _install_claim_contract(repo)
    qwen = _make_terminal_pass(repo, "Qwen3-8B", 128, "qwen128")
    gemma = repo / "configs/Gemma4-E4B.json"
    _write(gemma, _protocol("Gemma4-E4B", 6, "gemma6"))
    try:
        ASSEMBLER.assemble(
            repo_root=repo,
            historical_evidence_root=historical,
            protocol_paths=[qwen, gemma],
            output=output,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Nonterminal extension must not assemble evidence")
    assert not output.exists()


def test_frozen_claim_contract_matches_prospective_defaults() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs/realistic_niah_v5_prospective_count_chain_claim_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["shared_protocol"]["discovery_seeds"] == list(
        range(1234, 1254)
    )
    assert contract["shared_protocol"]["confirmation_seeds"] == list(
        range(1254, 1264)
    )
    assert contract["shared_protocol"]["outcome_blind"] is True
    assert contract["shared_protocol"]["selection_rank_used"] is False
    assert contract["models"]["Qwen3-8B"]["prospective_bank_size"] == 128
    assert contract["models"]["Gemma4-E4B"]["prospective_bank_size"] == 6
    assert "final allowed" in contract["exhaustion_rule"]
