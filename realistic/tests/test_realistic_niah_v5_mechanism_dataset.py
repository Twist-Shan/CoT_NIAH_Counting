from __future__ import annotations

from realistic_niah_v5.mechanism_dataset import (
    audit_paired_records,
    audit_shared_geometry_records,
    build_mode_contracts,
    build_non_thinking_user_text,
    causal_extension_registry,
    paired_record,
    render_mode_user_text,
    shared_geometry_record,
)


def _stimulus() -> dict:
    text = "Excerpt: City record."
    passage = f"before {text} after"
    start = passage.index(text)
    import hashlib

    return {
        "schema_version": "realistic_niah_v4_stimulus_v1",
        "protocol_version": "realistic_niah_v4_nonthinking_v3",
        "stimulus_id": "V4_4_T10000_N1_seed1234",
        "design_variant": "v4.4",
        "seed": 1234,
        "split": "discovery",
        "target_passage_tokens": 10000,
        "canonical_passage_tokens": 10000,
        "canonical_tokenizer": "Qwen/Qwen3-8B",
        "canonical_tokenizer_revision": "revision",
        "num_needles": 1,
        "gold_count": 1,
        "passage": passage,
        "passage_sha256": hashlib.sha256(passage.encode("utf-8")).hexdigest(),
        "gold_pairs": [{"slot_index": 1, "city": "Oslo", "score": 7}],
        "slots": [{"slot_index": 1, "active": True}],
        "active_needle_spans": [
            {
                "slot_index": 1,
                "needle_id": "N1",
                "text": text,
                "city": "Oslo",
                "score": 7,
                "canonical_span_start": 8,
                "canonical_span_end": 11,
                "canonical_token_length": 3,
                "char_start": start,
                "char_end": start + len(text),
            }
        ],
        "hard_negative_spans": [],
        "design": {"family_key": "unit"},
        "haystack": {"source": "unit"},
    }


def test_paired_modes_share_exact_v44_backbone() -> None:
    stimulus = _stimulus()
    non = paired_record(stimulus, mode="non_thinking")
    native = paired_record(stimulus, mode="native_thinking")
    audit = audit_paired_records([non], [native])
    assert audit["passed"] is True
    assert non["backbone_sha256"] == native["backbone_sha256"]
    assert non["passage"] == native["passage"]
    assert non["active_needle_spans"] == native["active_needle_spans"]
    assert non["chat_template_thinking_enabled"] is False
    assert native["chat_template_thinking_enabled"] is True
    assert non["expected_final_line"] == "Total:1"
    assert native["expected_final_line"] == "Total: 1"


def test_non_thinking_prompt_is_exact_v4_numeric_layout() -> None:
    prompt = build_non_thinking_user_text("PASSAGE")
    assert prompt.count("PASSAGE") == 1
    assert prompt.endswith("Total:<integer>")
    assert "with no space after the colon" in prompt


def test_shared_geometry_row_stores_backbone_once_and_two_mode_views() -> None:
    stimulus = _stimulus()
    contracts = build_mode_contracts()
    row = shared_geometry_record(stimulus, mode_contracts=contracts)
    audit = audit_shared_geometry_records([row], mode_contracts=contracts)

    assert audit["passed"] is True
    assert audit["geometry_shared_rows"] == 1
    assert audit["prompt_hash_checks"] == 2
    assert row["row_id"] == row["pair_id"] == stimulus["stimulus_id"]
    assert row["passage"] == stimulus["passage"]
    assert row["available_modes"] == ["non_thinking", "native_thinking"]
    assert set(row["mode_views"]) == {"non_thinking", "native_thinking"}
    assert "user_text" not in row
    assert "messages" not in row


def test_mode_contracts_reconstruct_registered_prompts() -> None:
    passage = "PASSAGE"
    contracts = build_mode_contracts()
    non = render_mode_user_text(contracts["non_thinking"], passage)
    native = render_mode_user_text(contracts["native_thinking"], passage)

    assert non == build_non_thinking_user_text(passage)
    assert non.count(passage) == 1
    assert native.count(passage) == 1
    assert contracts["non_thinking"]["chat_template_thinking_enabled"] is False
    assert contracts["native_thinking"]["chat_template_thinking_enabled"] is True
    assert "flag-only contrast" in contracts["native_thinking"]["treatment_note"]


def test_shared_geometry_audit_detects_mode_hash_tampering() -> None:
    import copy

    contracts = build_mode_contracts()
    row = shared_geometry_record(_stimulus(), mode_contracts=contracts)
    tampered = copy.deepcopy(row)
    tampered["mode_views"]["native_thinking"]["user_text_sha256"] = "0" * 64

    audit = audit_shared_geometry_records([tampered], mode_contracts=contracts)
    assert audit["passed"] is False
    assert any("native_thinking prompt SHA mismatch" in error for error in audit["errors"])


def test_causal_registry_keeps_mode_extensions_separate() -> None:
    registry = causal_extension_registry()
    assert registry["modes"]["non_thinking"]["extensions"] == []
    assert registry["modes"]["native_thinking"]["extensions"] == []
    assert "{mode}" in registry["layout"]
    assert any("shared paired confirmation" in rule for rule in registry["rules"])
    template = registry["extension_entry_template"]
    assert set(registry["required_entry_fields"]).issubset(template)
    assert template["files"]["discovery"].startswith("data/causal/<mode>/")
    assert template["files"]["confirmation"].startswith("data/causal/<mode>/")
