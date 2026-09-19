from __future__ import annotations

import json

import realistic_niah_v5.capture as capture_module
from realistic_niah_v5.spec import V5Config


def test_capture_shards_records_no_site_exclusion_and_continues(
    monkeypatch, tmp_path
) -> None:
    records = [
        {
            "request_id": "skip-me",
            "stimulus_id": "V4_4_T10000_N5_seed1238",
            "model_label": "Qwen3-8B",
            "seed": 1238,
            "split": "discovery",
        },
        {
            "request_id": "capture-me",
            "stimulus_id": "V4_4_T10000_N10_seed1238",
            "model_label": "Qwen3-8B",
            "seed": 1238,
            "split": "discovery",
        },
    ]

    def fake_capture_record(_model, _adapter, _tokenizer, row, **_kwargs):
        if row["request_id"] == "skip-me":
            raise capture_module.NoAlignedTraceSitesError(
                "No aligned registered V5 trace sites"
            )
        return {
            "stimulus_id": row["stimulus_id"],
            "model_label": row["model_label"],
            "seed": row["seed"],
            "split": row["split"],
            "gold_count": 10,
            "parsed_count": 10,
            "exact_count": True,
            "parser": {
                "trace_one_to_one": True,
                "trace_category": "one_to_one",
                "marker_kind": "indexed",
            },
            "site_rows": [{"site_id": "item_end:1"}],
            "elapsed_seconds": 0.01,
        }

    def fake_parse_record(row):
        return {
            "stimulus_id": row["stimulus_id"],
            "model_label": row["model_label"],
            "seed": row["seed"],
            "split": row["split"],
            "gold_count": 5,
            "parsed_count": 5,
            "exact_count": True,
            "parser": {
                "detected": False,
                "status": "no_terminated_gold_city_list",
                "trace_one_to_one": False,
                "trace_category": "partial_unique",
            },
        }

    monkeypatch.setattr(capture_module, "capture_trace_record", fake_capture_record)
    monkeypatch.setattr(capture_module, "parse_trace_record", fake_parse_record)

    index_path = capture_module.capture_trace_shards(
        None,
        None,
        None,
        records,
        config=V5Config(),
        output_dir=tmp_path,
    )

    index_rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    exclusion_rows = [
        json.loads(line)
        for line in (tmp_path / "capture_exclusions.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "capture_manifest.json").read_text())

    assert [row["request_id"] for row in index_rows] == ["capture-me"]
    assert [row["request_id"] for row in exclusion_rows] == ["skip-me"]
    assert exclusion_rows[0]["reason_code"] == (
        "no_aligned_registered_v5_trace_sites"
    )
    assert exclusion_rows[0]["parser_status"] == (
        "no_terminated_gold_city_list"
    )
    assert summary["rows"] == 1
    assert summary["input_rows"] == 2
    assert summary["excluded_rows"] == 1


def test_capture_shards_does_not_hide_unexpected_errors(
    monkeypatch, tmp_path
) -> None:
    def fail_unexpectedly(*_args, **_kwargs):
        raise RuntimeError("CUDA failure")

    monkeypatch.setattr(capture_module, "capture_trace_record", fail_unexpectedly)

    try:
        capture_module.capture_trace_shards(
            None,
            None,
            None,
            [{"request_id": "broken"}],
            config=V5Config(),
            output_dir=tmp_path,
        )
    except RuntimeError as error:
        assert str(error) == "CUDA failure"
    else:
        raise AssertionError("Unexpected errors must still fail the capture job")
