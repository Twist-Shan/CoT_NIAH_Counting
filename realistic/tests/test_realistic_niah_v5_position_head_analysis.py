from __future__ import annotations

import json

import pandas as pd

from realistic_niah_v5.position_head_analysis import analyze_position_heads


def _attention() -> pd.DataFrame:
    rows = []
    for split, seeds in (("discovery", [1, 2]), ("confirmation", [3, 4])):
        for seed in seeds:
            for occurrence in (1, 2, 3):
                for layer, head in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    first_specialist = (layer, head) == (0, 0)
                    raw = (
                        0.8
                        if first_specialist and occurrence == 1
                        else 0.6
                        if (layer, head) == (1, 1) and occurrence >= 2
                        else 0.1
                    )
                    rows.append(
                        {
                            "request_id": f"{split}-{seed}",
                            "model_label": "Toy",
                            "seed": seed,
                            "split": split,
                            "gold_count": 3,
                            "trace_one_to_one": True,
                            "mechanism": "targeted_retrieval",
                            "occurrence": occurrence,
                            "layer": layer,
                            "head": head,
                            "target_needle_raw_mass": raw,
                            "target_needle_relative_mass": raw / 0.9,
                            "target_needle_top1": raw > 0.5,
                        }
                    )
    return pd.DataFrame(rows)


def test_position_head_analysis_separates_selection_and_confirmation(tmp_path):
    attention_path = tmp_path / "attention.csv"
    _attention().to_csv(attention_path, index=False)
    registry = {
        "model_label": "Toy",
        "first_locator_definition": (
            "mean(first needle span mass - mean(other needle span masses))"
        ),
        "rankings": {
            "first_locator": [
                {"rank": 1, "layer": 0, "head": 1},
                {"rank": 2, "layer": 1, "head": 0},
            ]
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    paths = analyze_position_heads(
        attention_path,
        registry_path,
        tmp_path / "out",
        bank_sizes=[1, 2],
        bootstrap_samples=100,
    )
    rankings = pd.read_csv(paths["rankings"])
    first = rankings.loc[rankings["region"].eq("first")].sort_values("rank")
    later = rankings.loc[rankings["region"].eq("later")].sort_values("rank")
    assert (int(first.iloc[0]["layer"]), int(first.iloc[0]["head"])) == (0, 0)
    assert (int(later.iloc[0]["layer"]), int(later.iloc[0]["head"])) == (1, 1)
    summary = pd.read_csv(paths["position_summary"])
    raw = summary.loc[
        summary["metric"].eq("target_needle_raw_mass")
        & summary["bank_size"].eq(1)
    ].iloc[0]
    assert raw["estimate"] > 0
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["confirmation_used_for_selection"] is False
    assert audit["mass_contract"] == [
        "target_needle_raw_mass",
        "target_needle_relative_mass",
    ]
