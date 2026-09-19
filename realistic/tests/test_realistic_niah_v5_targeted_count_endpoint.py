from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_realistic_niah_v5_targeted_count_endpoint import (
    _read_shards,
    analyze,
    parse_final_total,
)
from scripts.build_realistic_niah_v5_targeted_count_plan import build_plan
from scripts.freeze_realistic_niah_v5_targeted_default_plan import freeze
from scripts.analyze_realistic_niah_v5_terminal_relay_mediation import (
    GEOMETRY_REASON,
    main as relay_analysis_main,
    relay_claim_gates,
    relay_pair_effects,
)
from scripts.analyze_realistic_niah_v5_complementary_readout import (
    complementary_claim_gates,
    complementary_pair_effects,
)
from scripts.run_realistic_niah_v5 import (
    _archive_invalid_behavior_shard,
    _atomic_jsonl,
    _load_completed_behavior_shard,
)
from scripts.inspect_realistic_niah_v5_targeted_count_partial_snapshot import (
    inspect as inspect_partial_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def test_behavior_resume_rejects_and_archives_malformed_shard(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shards" / "trial_example.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        '{"trial_id":"trial_example","completion_text":"cut',
        encoding="utf-8",
    )

    rows, error = _load_completed_behavior_shard(
        shard, expected_trial_id="trial_example"
    )
    assert rows is None
    assert "JSONDecodeError" in str(error)
    archived = _archive_invalid_behavior_shard(shard)
    assert archived.read_bytes() == shard.read_bytes()
    assert archived.parent.name == "corrupt_shards"

    _atomic_jsonl(
        shard,
        [
            {
                "trial_id": "trial_example",
                "trial_complete": True,
                "behavior_outcome": "correct_next_needle",
            }
        ],
    )
    rows, error = _load_completed_behavior_shard(
        shard, expected_trial_id="trial_example"
    )
    assert error is None
    assert rows is not None and rows[0]["trial_complete"] is True
    assert not list(shard.parent.glob("*.tmp"))


def test_targeted_reader_preserves_unicode_paragraph_separator(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shards" / "trial_unicode.jsonl"
    shard.parent.mkdir(parents=True)
    payload = {
        "trial_id": "trial_unicode",
        "completion_text": "before\u2029inside\u2028after",
    }
    shard.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded = _read_shards(tmp_path)
    assert len(loaded) == 1
    assert loaded.loc[0, "completion_text"] == payload["completion_text"]


def test_parse_final_total_uses_last_explicit_total() -> None:
    assert parse_final_total("draft Total: 2\nfinal Total: 3") == 3
    assert parse_final_total("I counted three records") is None


def test_targeted_count_plan_is_layer_matched_and_selected_excluded() -> None:
    selected = [[2, 1], [2, 3], [4, 0]]
    import hashlib
    import json

    selection = {
        "model_label": "toy",
        "sample_panel": {
            "seeds": list(range(30)),
            "discovery_seeds": list(range(20)),
        },
        "development_selection": {
            "primary_bank_heads": selected,
            "primary_bank_sha256": hashlib.sha256(
                json.dumps(selected).encode("utf-8")
            ).hexdigest(),
            "head_ranking_source_anchor": "p0_item_end",
            "head_ranking_metric": "frozen_metric",
            "head_ranking_source_grammar": "grammar",
        },
    }
    plan = build_plan(
        selection, heads_per_layer=6, random_repeats=3, random_seed=7
    )
    selected_set = {tuple(value) for value in selected}
    for raw in plan.loc[plan["condition"].eq("layer_matched_random"), "heads"]:
        control = json.loads(raw)
        assert not ({tuple(value) for value in control} & selected_set)
        assert [value[0] for value in control].count(2) == 2
        assert [value[0] for value in control].count(4) == 1


@pytest.mark.parametrize(
    ("model", "expected_k", "expected_sha"),
    [
        (
            "Qwen3-8B",
            128,
            "ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b",
        ),
        (
            "Gemma4-E4B",
            6,
            "2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1",
        ),
    ],
)
def test_prospective_targeted_defaults_freeze_exact_bank(
    tmp_path: Path, model: str, expected_k: int, expected_sha: str
) -> None:
    output = tmp_path / model / "frozen_targeted_count_plan.csv"
    audit = freeze(
        root=ROOT,
        defaults_path=(
            ROOT
            / "configs/realistic_niah_v5_targeted_retrieval_prospective_defaults_v1.json"
        ),
        model=model,
        output=output,
    )
    assert audit["status"] == "FROZEN_NEW"
    assert audit["bank_size"] == expected_k
    assert audit["selected_bank_sha256"] == expected_sha
    assert audit["selection_rank_used"] is False
    plan = pd.read_csv(output)
    selected = plan.loc[plan["condition"].eq("selected_bank")].iloc[0]
    assert int(selected["bank_size"]) == expected_k
    assert selected["bank_sha256"] == expected_sha
    assert "selection_rank" not in plan.columns
    assert freeze(
        root=ROOT,
        defaults_path=(
            ROOT
            / "configs/realistic_niah_v5_targeted_retrieval_prospective_defaults_v1.json"
        ),
        model=model,
        output=output,
    )["status"] == "REUSED_IDENTICAL"


def test_prospective_targeted_default_refuses_mismatched_existing_plan(
    tmp_path: Path,
) -> None:
    output = tmp_path / "frozen_targeted_count_plan.csv"
    output.write_text("bank_size,condition\n125,selected_bank\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="mismatched frozen plan"):
        freeze(
            root=ROOT,
            defaults_path=(
                ROOT
                / "configs/realistic_niah_v5_targeted_retrieval_prospective_defaults_v1.json"
            ),
            model="Qwen3-8B",
            output=output,
        )


def test_targeted_count_endpoint_passes_synthetic_serial_propagation() -> None:
    rows = []
    for seed in range(1234, 1254):
        for count in range(2, 11):
            base = {
                "request_id": f"request-{seed}-{count}",
                "seed": seed,
                "gold_count": count,
                "from_occurrence": count - 1,
                "to_occurrence": count,
                "split": "discovery",
                "status": "ok",
                "head_ablation_decode_steps_requested": -1,
                "generation_truncated": False,
            }
            rows.append(
                {
                    **base,
                    "condition": "clean",
                    "repeat": 0,
                    "completion_text": f"Total: {count}",
                    "correct_next_needle": True,
                }
            )
            rows.append(
                {
                    **base,
                    "condition": "selected_bank",
                    "repeat": 0,
                    "completion_text": f"Total: {count - 1}",
                    "correct_next_needle": False,
                }
            )
            for repeat in range(1, 4):
                rows.append(
                    {
                        **base,
                        "condition": "layer_matched_random",
                        "repeat": repeat,
                        "completion_text": f"Total: {count}",
                        "correct_next_needle": True,
                    }
                )
    _scored, seed_effects, claims = analyze(
        pd.DataFrame(rows),
        phase="discovery",
        bootstrap_samples=200,
        random_seed=11,
    )
    assert len(seed_effects) == 20 * 5
    assert claims["targeted_to_count_pass"] is True
    assert claims["gates"]["exact_minus_one"]["pass"] is True


def test_partial_snapshot_uses_only_complete_five_arm_anchors() -> None:
    rows = []
    for seed in (1234, 1235):
        base = {
            "request_id": f"request-{seed}",
            "seed": seed,
            "gold_count": 5,
            "from_occurrence": 4,
            "to_occurrence": 5,
            "split": "discovery",
            "status": "ok",
            "head_ablation_decode_steps_requested": -1,
        }
        rows.append(
            {
                **base,
                "condition": "clean",
                "completion_text": "Total: 5",
                "correct_next_needle": True,
            }
        )
        rows.append(
            {
                **base,
                "condition": "selected_bank",
                "completion_text": "Total: 4",
                "correct_next_needle": False,
            }
        )
        for _repeat in range(3):
            rows.append(
                {
                    **base,
                    "condition": "layer_matched_random",
                    "completion_text": "Total: 5",
                    "correct_next_needle": True,
                }
            )
    rows.append(
        {
            **base,
            "request_id": "incomplete-anchor",
            "condition": "clean",
            "completion_text": "Total: 5",
            "correct_next_needle": True,
        }
    )
    snapshot = inspect_partial_snapshot(
        pd.DataFrame(rows),
        bootstrap_samples=200,
        random_seed=19,
        expected_anchor_count_by_seed={1234: 1, 1235: 2},
    )
    assert snapshot["formal_discovery_gate_evaluated"] is False
    assert snapshot["observed_seed_count"] == 1
    assert snapshot["complete_anchor_count"] == 1
    assert snapshot["discarded_incomplete_anchor_trial_row_count"] == 1
    assert snapshot["excluded_partial_seed_trial_row_count"] == 5
    assert snapshot["partially_observed_seeds"] == {
        "1235": {"observed_complete_anchors": 1, "expected_frozen_anchors": 2}
    }
    assert snapshot["snapshot_all_primary_pass"] is True


def test_terminal_relay_factorial_passes_synthetic_nested_reset() -> None:
    rows = []
    for seed in range(1234, 1254):
        for count in range(5, 11):
            common = {
                "model_label": "Qwen3-8B",
                "seed": seed,
                "request_id": f"request-{seed}-{count}",
                "gold_count": count,
                "mechanism_split": "development",
                "pair_sha256": f"pair-{seed}-{count}",
                "donor_offset": -3,
                "source_layer": 19,
                "relay_layer": 26,
            }
            values = {
                ("self_patch", "natural_relay"): 5.0,
                ("full_donor_patch", "natural_relay"): 1.0,
                ("self_patch", "answer_query_clean_reset"): 5.0,
                ("full_donor_patch", "answer_query_clean_reset"): 3.0,
                ("self_patch", "post_terminal_suffix_clean_reset"): 5.0,
                ("full_donor_patch", "post_terminal_suffix_clean_reset"): 5.0,
            }
            for (source, relay), value in values.items():
                exact_count = 1.0 if source == "self_patch" else 0.0
                if relay == "post_terminal_suffix_clean_reset":
                    exact_count = 1.0
                rows.append(
                    {
                        **common,
                        "source_condition": source,
                        "relay_condition": relay,
                        "correct_count_margin": value,
                        "exact_count": exact_count,
                    }
                )
    effects = relay_pair_effects(pd.DataFrame(rows))
    claims = relay_claim_gates(
        effects,
        phase="discovery",
        bootstrap_samples=200,
        random_seed=5,
    )
    assert claims["residual_relay_pass"] is True
    assert claims["gates"]["answer_query_only_mediation"]["pass"] is True
    assert claims["greedy_exact_count_support_pass"] is True


def test_terminal_relay_keeps_planned_seed_that_is_fully_geometry_na(
    tmp_path: Path,
) -> None:
    rows = []
    for seed in range(1254, 1264):
        pair_sha256 = f"pair-{seed}"
        is_geometry_na = seed == 1259
        for source_condition in ("self_patch", "full_donor_patch"):
            for relay_condition in (
                "natural_relay",
                "answer_query_clean_reset",
                "post_terminal_suffix_clean_reset",
            ):
                row = {
                    "model_label": "Qwen3-8B",
                    "seed": seed,
                    "request_id": f"request-{seed}",
                    "gold_count": 10,
                    "mechanism_split": "confirmation",
                    "pair_sha256": pair_sha256,
                    "donor_offset": -1,
                    "source_layer": 19,
                    "relay_layer": 26,
                    "source_condition": source_condition,
                    "relay_condition": relay_condition,
                    "status": "not_applicable" if is_geometry_na else "ok",
                }
                if is_geometry_na:
                    row["exclusion_reason"] = GEOMETRY_REASON
                else:
                    self_value = 5.0
                    donor_values = {
                        "natural_relay": 1.0,
                        "answer_query_clean_reset": 3.0,
                        "post_terminal_suffix_clean_reset": 5.0,
                    }
                    row["correct_count_margin"] = (
                        self_value
                        if source_condition == "self_patch"
                        else donor_values[relay_condition]
                    )
                    row["exact_count"] = float(
                        source_condition == "self_patch"
                        or relay_condition == "post_terminal_suffix_clean_reset"
                    )
                rows.append(row)
    trials = tmp_path / "trials"
    shards = trials / "shards"
    shards.mkdir(parents=True)
    (shards / "all.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    relay_analysis_main(
        [
            "--trials",
            str(trials),
            "--output",
            str(output),
            "--phase",
            "confirmation",
            "--bootstrap-samples",
            "200",
            "--random-seed",
            "7",
        ]
    )
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["planned_seed_count"] == 10
    assert audit["seed_count"] == 9
    assert audit["geometry_not_applicable_full_seed_count"] == 1
    assert audit["geometry_not_applicable_full_seeds"] == [1259]


def test_complementary_readout_requires_both_qwen_routes_to_be_cut() -> None:
    rows = []
    damages = {
        ("natural_relay", "clean"): 4.0,
        ("natural_relay", "block_trace_items_matched_control"): 4.0,
        ("natural_relay", "block_trace_items"): 2.0,
        (
            "post_terminal_suffix_clean_reset",
            "block_trace_items_matched_control",
        ): 2.0,
        ("post_terminal_suffix_clean_reset", "clean"): 2.0,
        ("post_terminal_suffix_clean_reset", "block_trace_items"): 0.0,
    }
    for seed in range(1234, 1254):
        common = {
            "model_label": "Qwen3-8B",
            "seed": seed,
            "request_id": f"request-{seed}",
            "gold_count": 10,
            "mechanism_split": "development",
            "pair_sha256": f"pair-{seed}",
            "donor_offset": -1,
            "source_layer": 19,
            "relay_layer": 26,
            "status": "ok",
        }
        for patch in ("self_patch", "full_donor_patch"):
            for relay in (
                "natural_relay",
                "post_terminal_suffix_clean_reset",
            ):
                for mask in (
                    "clean",
                    "block_trace_items",
                    "block_trace_items_matched_control",
                ):
                    damage = damages[(relay, mask)]
                    rows.append(
                        {
                            **common,
                            "patch_condition": patch,
                            "relay_condition": relay,
                            "mask_condition": mask,
                            "correct_count_margin": (
                                10.0 if patch == "self_patch" else 10.0 - damage
                            ),
                            "exact_count": (
                                1.0
                                if patch == "self_patch"
                                or (
                                    relay
                                    == "post_terminal_suffix_clean_reset"
                                    and mask == "block_trace_items"
                                )
                                else 0.0
                            ),
                        }
                    )
    effects = complementary_pair_effects(pd.DataFrame(rows))
    claims = complementary_claim_gates(
        effects,
        phase="discovery",
        bootstrap_samples=300,
        random_seed=9,
    )
    assert claims["complementary_readout_pass"] is True
    assert claims["gates"]["source_only_leaves_residual"]["estimate"] == 2.0
    assert claims["gates"]["relay_only_leaves_residual"]["estimate"] == 2.0
    assert claims["gates"]["joint_cut_residual_equivalence"]["estimate"] == 0.0
