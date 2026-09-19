from __future__ import annotations

import hashlib
import json

from realistic_niah.entity_domains import (
    native_user_text,
    nonthinking_query_text,
)
from realistic_niah_v5.domain_transfer import (
    audit_transformed_stimulus,
    entity_panel,
    transform_stimulus,
    write_capture_site_index,
)
from realistic_niah_v5.generation import V5_USER_TEMPLATE, build_v5_user_text
from realistic_niah_v5.parsing import find_trace_count_sequence


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_row() -> dict:
    first = (
        "\u2029Excerpt:\nIn the 2024 city score audit, Madrid received a score "
        "of 71.\nEnd excerpt.\u2029"
    )
    second = (
        "\u2029Excerpt:\nIn the 2024 city score audit, Seoul received a score "
        "of 84.\nEnd excerpt.\u2029"
    )
    prefix = "alpha "
    middle = " beta "
    suffix = " gamma hard-negative tail"
    passage = prefix + first + middle + second + suffix
    first_start = len(prefix)
    first_end = first_start + len(first)
    second_start = first_end + len(middle)
    second_end = second_start + len(second)
    negative_start = second_end + len(" gamma ")
    return {
        "schema_version": "realistic_niah_v4_stimulus_v1",
        "stimulus_id": "V4_4_T10000_N2_seed1254",
        "design_variant": "v4.4",
        "split": "confirmation",
        "seed": 1254,
        "gold_count": 2,
        "target_passage_tokens": 10000,
        "protocol_version": "realistic_niah_v4_nonthinking_v3",
        "passage": passage,
        "passage_sha256": _sha(passage),
        "active_needle_spans": [
            {
                "slot_index": 1,
                "city": "Madrid",
                "score": 71,
                "char_start": first_start,
                "char_end": first_end,
                "text": first,
            },
            {
                "slot_index": 2,
                "city": "Seoul",
                "score": 84,
                "char_start": second_start,
                "char_end": second_end,
                "text": second,
            },
        ],
        "gold_pairs": [
            {"slot_index": 1, "city": "Madrid", "score": 71},
            {"slot_index": 2, "city": "Seoul", "score": 84},
        ],
        "slots": [
            {
                "slot_index": 1,
                "active": True,
                "city": "Madrid",
                "score": 71,
                "content_kind": "needle",
                "char_start": first_start,
                "char_end": first_end,
                "text": first,
                "content_text": first,
                "canonical_span_start": 10,
                "canonical_span_end": 20,
                "canonical_token_length": 10,
            },
            {
                "slot_index": 2,
                "active": True,
                "city": "Seoul",
                "score": 84,
                "content_kind": "needle",
                "char_start": second_start,
                "char_end": second_end,
                "text": second,
                "content_text": second,
                "canonical_span_start": 30,
                "canonical_span_end": 40,
                "canonical_token_length": 10,
            },
        ],
        "hard_negative_spans": [
            {
                "slot_index": 11,
                "char_start": negative_start,
                "char_end": negative_start + len("hard-negative"),
                "canonical_span_start": 50,
                "canonical_span_end": 52,
                "canonical_token_length": 2,
            }
        ],
    }


def test_domain_prompt_templates_preserve_city_and_replace_task_lexicon() -> None:
    passage = "test passage"
    assert build_v5_user_text(passage) == V5_USER_TEMPLATE.format(passage=passage)
    flower = build_v5_user_text(passage, entity_domain="flower")
    assert flower == native_user_text(passage, entity_domain="flower")
    assert "flower-score audit records" in flower
    assert "city-score" not in flower
    animal = nonthinking_query_text(entity_domain="animal", answer_format="numeric")
    assert animal.startswith("How many animal-score audit records")
    assert animal.endswith("Total:<integer>")
    assert "An animal-score audit record" in native_user_text(
        passage, entity_domain="animal"
    )


def test_transform_stimulus_preserves_scores_and_recomputes_offsets() -> None:
    source = _source_row()
    transformed = transform_stimulus(source, entity_domain="flower")
    audit_transformed_stimulus(transformed)
    assert transformed["source_stimulus_id"] == source["stimulus_id"]
    assert transformed["entity_domain"] == "flower"
    assert [row["score"] for row in transformed["gold_pairs"]] == [71, 84]
    assert "city score audit" not in transformed["passage"]
    assert transformed["passage"].count("flower score audit") == 2
    for span in transformed["active_needle_spans"]:
        assert transformed["passage"][span["char_start"] : span["char_end"]] == span[
            "text"
        ]
    negative = transformed["hard_negative_spans"][0]
    assert (
        transformed["passage"][negative["char_start"] : negative["char_end"]]
        == "hard-negative"
    )


def test_entity_panel_is_seed_deterministic_and_domain_specific() -> None:
    assert entity_panel("flower", 1254) == entity_panel("flower", 1254)
    assert entity_panel("flower", 1254) != entity_panel("flower", 1255)
    assert set(entity_panel("flower", 1254)).isdisjoint(
        entity_panel("animal", 1254)
    )


def test_legacy_city_registry_parses_noncity_entity_names() -> None:
    trace = "1. Rose: 71\n2. Tulip: 84\nTotal: 2"
    parsed = find_trace_count_sequence(
        trace,
        model_family="qwen3",
        gold_records=[
            {"city": "Rose", "score": 71},
            {"city": "Tulip", "score": 84},
        ],
    )
    assert parsed.detected
    assert parsed.item_count == 2
    assert parsed.item_gold_cities == ("Rose", "Tulip")


def test_truncated_generation_rescue_rule_is_strictly_higher_ceiling() -> None:
    from types import SimpleNamespace

    from realistic_niah_v5.domain_transfer import _should_regenerate_censored

    truncated = {"generation_truncated": True, "output_tokens": 4096}
    complete = {"generation_truncated": False, "output_tokens": 400}
    assert _should_regenerate_censored(
        truncated, SimpleNamespace(max_new_tokens=16384)
    )
    assert not _should_regenerate_censored(
        truncated, SimpleNamespace(max_new_tokens=4096)
    )
    assert not _should_regenerate_censored(
        complete, SimpleNamespace(max_new_tokens=16384)
    )


def test_capture_site_index_flattens_ragged_running_path(tmp_path) -> None:
    shard = tmp_path / "shards" / "example"
    shard.mkdir(parents=True)
    (shard / "states.npz").write_bytes(b"placeholder")
    manifest = {
        "layers": [0, 1],
        "site_states_shape": [3, 2, 4],
        "site_rows": [
            {
                "site_id": "item_end:1",
                "site_kind": "item_end",
                "endpoint_token": 101,
            },
            {
                "site_id": "item_end:2",
                "site_kind": "item_end",
                "endpoint_token": 147,
            },
            {
                "site_id": "answer_query_v3",
                "site_kind": "answer_query_v3",
                "endpoint_token": 211,
            },
        ],
    }
    (shard / "capture_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    capture_row = {
        "row_index": 0,
        "request_id": "request-1",
        "stimulus_id": "stimulus-1",
        "source_stimulus_id": "source-1",
        "model_label": "Qwen3-8B",
        "mode": "native-thinking",
        "entity_domain": "flower",
        "seed": 1254,
        "split": "confirmation",
        "gold_count": 1,
        "manifest_path": "shards/example/capture_manifest.json",
        "states_path": "shards/example/states.npz",
    }
    (tmp_path / "capture_index.jsonl").write_text(
        json.dumps(capture_row) + "\n", encoding="utf-8"
    )

    site_index, site_manifest = write_capture_site_index(tmp_path)
    rows = [json.loads(line) for line in site_index.read_text().splitlines()]
    summary = json.loads(site_manifest.read_text())

    assert [row["state_axis"] for row in rows] == [0, 1, 2]
    assert [row["site_kind"] for row in rows] == [
        "item_end",
        "item_end",
        "answer_query_v3",
    ]
    assert all(row["states_path"] == "shards/example/states.npz" for row in rows)
    assert summary["running_site_rows"] == 2
    assert summary["answer_site_rows"] == 1
    assert summary["layer_registries"] == {"Qwen3-8B": [0, 1]}
