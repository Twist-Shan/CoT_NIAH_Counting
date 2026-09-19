from __future__ import annotations

import hashlib
import json

import pandas as pd

from realistic_niah_v4.causal_generation import (
    causal_v2_monotonic_span_source_indices,
)
from realistic_niah_v4.causal_v2_audit import (
    AuditCollector,
    _audit_prompt_alignment,
    _audit_transport_metrics,
    _completed_formal_root,
)
from realistic_niah_v4.causal_v2 import (
    CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
    CausalV2Design,
)


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _alignment_fields(receiver_length: int, donor_length: int) -> dict[str, object]:
    source_indices = causal_v2_monotonic_span_source_indices(
        receiver_length,
        donor_length,
    )
    distinct = len(set(source_indices))
    if receiver_length == 1 or donor_length == 1:
        max_error = 0.0
    else:
        max_error = max(
            abs(
                source_index / (donor_length - 1)
                - receiver_index / (receiver_length - 1)
            )
            for receiver_index, source_index in enumerate(source_indices)
        )
    return {
        "receiver_model_token_length": receiver_length,
        "donor_model_token_length": donor_length,
        "absolute_model_token_length_delta": abs(receiver_length - donor_length),
        "exact_model_token_alignment": receiver_length == donor_length,
        "alignment_policy": CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
        "mapping_supported": True,
        "source_index_map": json.dumps(list(source_indices)),
        "mapped_receiver_token_count": receiver_length,
        "distinct_source_token_count": distinct,
        "source_tokens_dropped": donor_length - distinct,
        "source_tokens_reused": receiver_length - distinct,
        "max_normalized_position_error": max_error,
    }


def test_stage_audit_checks_directory_and_complete_design_hash(tmp_path) -> None:
    design = {
        "profile": "formal",
        "behavior_metric": "strict_greedy_complete_numeric_generation",
        "design_variant": "v4.4",
        "implementation_sha256": "a" * 64,
    }
    digest = _hash(design)
    root = tmp_path / "answer_patching" / f"screen_{digest}"
    root.mkdir(parents=True)
    (root / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (root / "complete.json").write_text(
        json.dumps({"status": "complete", "design_hash": digest}),
        encoding="utf-8",
    )
    audit = AuditCollector()
    observed = _completed_formal_root(
        tmp_path,
        family="answer_patching",
        phase="screen",
        required=True,
        audit=audit,
    )
    assert observed == root
    assert not audit.errors


def test_transport_audit_recomputes_failure_aware_metric() -> None:
    frame = pd.DataFrame(
        [
            {
                "receiver_count": 5,
                "donor_count": 10,
                "baseline_predicted_count": 5,
                "patched_predicted_count": 10,
                "normalized_transport": 1.0,
                "target_conformity": 1.0,
                "strict_normalized_transport": 1.0,
                "strict_target_conformity": 1.0,
                "transport_numeric_valid": True,
                "status": "ok",
            },
            {
                "receiver_count": 10,
                "donor_count": 5,
                "baseline_predicted_count": 10,
                "patched_predicted_count": float("nan"),
                "normalized_transport": float("nan"),
                "target_conformity": float("nan"),
                "strict_normalized_transport": 0.0,
                "strict_target_conformity": 0.0,
                "transport_numeric_valid": False,
                "status": "ok",
            },
        ]
    )
    audit = AuditCollector()
    _audit_transport_metrics(frame, audit, "toy")
    assert not audit.errors
    frame.loc[0, "normalized_transport"] = 0.5
    failed = AuditCollector()
    _audit_transport_metrics(frame, failed, "toy")
    assert failed.errors[0]["name"] == "toy.normalized_transport_recomputed"


def test_prompt_alignment_audit_accepts_registered_remapping_but_not_unsupported(
    tmp_path,
) -> None:
    design = CausalV2Design()
    rows = []
    for seed in (*design.screen_seeds, *design.confirmation_seeds):
        for receiver, donor in design.directed_pairs:
            lower, upper = sorted((receiver, donor))
            for slot in range(lower + 1, upper + 1):
                rows.append(
                    {
                        "seed": seed,
                        "receiver_count": receiver,
                        "donor_count": donor,
                        "k": upper - lower,
                        "slot_index": slot,
                        **_alignment_fields(3, 3),
                    }
                )
    root = tmp_path / "preflight"
    root.mkdir()
    path = root / "prompt_full_span_alignment.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    audit = AuditCollector()
    _audit_prompt_alignment(root, design, audit)
    assert not audit.errors
    frame = pd.read_csv(path)
    for key, value in _alignment_fields(3, 4).items():
        frame.loc[0, key] = value
    frame.to_csv(path, index=False)
    remapped = AuditCollector()
    _audit_prompt_alignment(root, design, remapped)
    assert not remapped.errors
    frame.loc[0, "mapping_supported"] = False
    frame.to_csv(path, index=False)
    failed = AuditCollector()
    _audit_prompt_alignment(root, design, failed)
    assert [item["name"] for item in failed.errors] == [
        "prompt_alignment.all_changed_slots_supported"
    ]
