from __future__ import annotations

import json

import numpy as np

from scripts.audit_realistic_niah_v5_geometry_capture import audit_capture


def test_geometry_capture_audit_accepts_exact_full_panel(tmp_path) -> None:
    rows = []
    row_number = 0
    for seed in range(1234, 1264):
        split = "discovery" if seed < 1254 else "confirmation"
        for gold_count in range(1, 11):
            shard = tmp_path / "shards" / f"r{row_number}"
            shard.mkdir(parents=True)
            item_count = gold_count - 1 if gold_count == 10 else gold_count
            sites = []
            for occurrence in range(1, item_count + 1):
                sites.extend(
                    {
                        "site_id": f"{site_kind}:{occurrence}",
                        "site_kind": site_kind,
                        "occurrence": occurrence,
                        "alignment_eligible": True,
                        "endpoint_token": occurrence,
                    }
                    for site_kind in (
                        "pre_city",
                        "city_end",
                        "city_unit_end",
                        "item_end",
                        "post_boundary",
                    )
                )
            sites.append(
                {
                    "site_id": "answer_query_v3",
                    "site_kind": "answer_query_v3",
                    "occurrence": None,
                    "alignment_eligible": True,
                    "endpoint_token": item_count + 1,
                }
            )
            layers = [0, 1]
            shape = (len(sites), len(layers), 4)
            np.savez(
                shard / "states.npz",
                layer_indices=np.asarray(layers),
                site_states=np.zeros(shape, dtype=np.float16),
            )
            (shard / "capture_manifest.json").write_text(
                json.dumps(
                    {
                        "request_id": f"r{row_number}",
                        "stimulus_id": f"s{row_number}",
                        "model_label": "Toy",
                        "split": split,
                        "seed": seed,
                        "gold_count": gold_count,
                        "selected_site_kinds": [
                            "pre_city",
                            "city_end",
                            "city_unit_end",
                            "item_end",
                            "post_boundary",
                            "answer_query_v3",
                        ],
                        "layers": layers,
                        "site_rows": sites,
                        "site_states_shape": list(shape),
                    }
                ),
                encoding="utf-8",
            )
            rows.append(
                {
                    "request_id": f"r{row_number}",
                    "stimulus_id": f"s{row_number}",
                    "model_label": "Toy",
                    "split": split,
                    "seed": seed,
                    "gold_count": gold_count,
                    "trace_item_count": item_count,
                    "trace_category": "partial_unique",
                    "marker_kind": "inline_count",
                    "manifest_path": f"shards/r{row_number}/capture_manifest.json",
                    "states_path": f"shards/r{row_number}/states.npz",
                }
            )
            row_number += 1
    index = tmp_path / "capture_index.jsonl"
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    audit = audit_capture(index)

    assert audit["status"] == "pass"
    assert audit["trajectory_rows"] == 300
    assert audit["answer_query_v3_rows"] == 300
    assert audit["running_state_rows"] == 1620
