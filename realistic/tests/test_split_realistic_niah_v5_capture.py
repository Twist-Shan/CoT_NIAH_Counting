from __future__ import annotations

import json

import numpy as np

from scripts.split_realistic_niah_v5_capture import filter_capture


def test_filter_capture_extracts_one_answer_site_per_trajectory(tmp_path) -> None:
    source = tmp_path / "source"
    source_rows = []
    for row_number, (split, seed, gold_count) in enumerate(
        (("discovery", 1, 1), ("confirmation", 2, 2))
    ):
        shard = source / "shards" / f"r{row_number}"
        shard.mkdir(parents=True)
        np.savez(
            shard / "states.npz",
            layer_indices=np.asarray([0, 1]),
            site_states=np.arange(3 * 2 * 4).reshape(3, 2, 4),
        )
        manifest = {
            "schema_version": "source-v1",
            "site_rows": [
                {"site_kind": "item_end", "occurrence": 1},
                {"site_kind": "post_boundary", "occurrence": 1},
                {"site_kind": "answer_query_v3", "occurrence": None},
            ],
        }
        (shard / "capture_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        source_rows.append(
            {
                "schema_version": "source-v1",
                "request_id": f"r{row_number}",
                "split": split,
                "seed": seed,
                "gold_count": gold_count,
                "manifest_path": f"shards/r{row_number}/capture_manifest.json",
                "states_path": f"shards/r{row_number}/states.npz",
            }
        )
    source_index = source / "capture_index.jsonl"
    source_index.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )

    output = tmp_path / "answer"
    result = filter_capture(
        source_index, output, site_kinds=["answer_query_v3"]
    )

    rows = [json.loads(line) for line in result.read_text().splitlines()]
    assert len(rows) == 2
    manifest = json.loads(
        (output / rows[0]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert [site["site_kind"] for site in manifest["site_rows"]] == [
        "answer_query_v3"
    ]
    with np.load(output / rows[0]["states_path"], allow_pickle=False) as archive:
        assert archive["site_states"].shape == (1, 2, 4)
        np.testing.assert_array_equal(
            archive["site_states"][0], np.arange(3 * 2 * 4).reshape(3, 2, 4)[2]
        )
    audit = json.loads(
        (output / "capture_manifest.json").read_text(encoding="utf-8")
    )
    assert audit["rows"] == 2
    assert audit["sites_per_trajectory_min"] == 1
    assert audit["split_trajectory_counts"] == {
        "confirmation": 1,
        "discovery": 1,
    }

    # A completed split is restart-safe and does not rewrite or duplicate rows.
    assert (
        filter_capture(source_index, output, site_kinds=["answer_query_v3"])
        == result
    )
