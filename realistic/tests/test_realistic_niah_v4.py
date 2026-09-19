from __future__ import annotations

import json
import re
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from realistic_niah_v4.attention import (
    _validate_raw_attention_shard,
    _write_raw_attention_shard,
    broad_attention_metrics,
)
from realistic_niah_v4.attention_outcomes import (
    POOL_METRICS,
    confirmation_outcome_effects,
    discovery_seed_bootstrap_stability,
    layer_pooling_metrics,
    nested_increment_diagnostics,
    rank_broad_candidates,
)
from realistic_niah_v4.behavior import (
    label_generated_completion,
    parse_numeric_completion,
)
from realistic_niah_v4.causal import count_logit_metrics
from realistic_niah_v4.causal_generation import (
    RESIDUAL_PATCH_SITES,
    _residual_site_states,
    intervention_outcome,
    run_generation_head_ablation,
)
from realistic_niah_v4.geometric_steering import (
    CountCentroidBundle,
    LayerSetSteeringPlan,
    _layer_set_centroid_delta_states,
    centroid_geometry_tables,
    chord_point,
    fit_count_centroids,
    layer_set_steering_plan_scores,
    load_centroid_bundle,
    polyline_point,
    save_centroid_bundle,
    select_layer_set_steering_plans,
)
from realistic_niah_v4.modeling import (
    _attention_tensor,
    capture_post_block_states,
    capture_span_states,
    discover_decoder_adapter,
    generate_with_head_ablation,
    generate_with_residual_interventions,
    position_attention_outputs,
    query_attention_rows,
    run_last_logits,
    run_with_head_ablation,
    run_with_residual_patch,
)
from realistic_niah_v4.pipeline import (
    _causal_design_root,
    run_labeled_attention_analysis,
)
from realistic_niah_v4.prompts import (
    PromptEncoding,
    TokenSpan,
    render_v4_prompt,
)
from realistic_niah_v4.representation import (
    analyze_representation_captures,
    capture_answer_query_representation_shards,
    capture_representation_shards,
    label_representation_analysis_by_generation,
)
from realistic_niah_v4.prompt_counter_attention import (
    prompt_counter_attention_metrics,
)
from realistic_niah_v4.spec import (
    DESIGN_VARIANTS,
    MODEL_SPECS,
    V4Config,
)
from realistic_niah_v4.stimuli import (
    ControlledFreezeSpec,
    _assemble_tokens,
    _remap_filler_positions_for_exact_roundtrip,
    audit_v4_grid,
    freeze_v4_grid,
    load_stimuli,
)


class _ByteFragmentTokenizer:
    """Tiny byte tokenizer with one Unicode scalar split across two tokens."""

    _bytes = {
        0: b"A",
        1: b"\xc2",
        2: b"\x97",
        3: b"B",
        9: b"\xef\xbf\xbd",
        100: b"X",
        101: b"Y",
    }

    def decode(self, tokens: list[int]) -> str:
        return b"".join(self._bytes[token] for token in tokens).decode(
            "utf-8", errors="replace"
        )

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for character in text:
            encoded.extend(
                {
                    "A": [0],
                    "\x97": [1, 2],
                    "B": [3],
                    "\ufffd": [9],
                    "X": [100],
                    "Y": [101],
                }[character]
            )
        return encoded


def test_insertion_boundary_remaps_split_unicode_scalar_deterministically() -> None:
    tokenizer = _ByteFragmentTokenizer()
    base = [0, 1, 2, 3]
    needles = [{"tokens": [100], "inserted_tokens": [101], "token_length": 1}]
    positions, starts, remaps = _remap_filler_positions_for_exact_roundtrip(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        base_tokens=base,
        ordered_needles=needles,
        nominal_filler_positions=(2,),
        nominal_final_starts=(2,),
        widths=(1,),
        minimum_separation=1,
    )
    assert positions == (1,)
    assert starts == (1,)
    assert remaps == [
        {
            "slot_index": 1,
            "nominal_filler_position": 2,
            "realized_filler_position": 1,
            "nominal_final_start": 2,
            "realized_final_start": 1,
            "delta_tokens": -1,
        }
    ]
    for active_count in (0, 1):
        assembled, _ = _assemble_tokens(
            base_tokens=base,
            ordered_needles=needles,
            active_count=active_count,
            filler_positions=positions,
        )
        assert tokenizer.encode(tokenizer.decode(assembled)) == assembled


def _small_config() -> V4Config:
    config = V4Config(
        target_passage_tokens=300,
        needle_counts=(1, 2, 3),
        seeds=(11, 12, 13, 14),
        discovery_seeds=(11, 12),
        confirmation_seeds=(13, 14),
        canonical_tokenizer="simple",
        canonical_tokenizer_revision="test-only",
        fixed_slot_depths=(0.2, 0.5, 0.8),
        randomized_position_min_separation_tokens=20,
        representation_count=3,
        patch_count_pairs=((1, 2), (2, 3)),
        steering_count_pairs=((1, 2), (2, 3)),
    )
    config.validate()
    return config


@pytest.fixture()
def small_grid(tmp_path: Path) -> tuple[V4Config, Path]:
    haystack = tmp_path / "haystack"
    haystack.mkdir()
    for index in range(4):
        text = " ".join(f"essay{index}_token{token % 97}" for token in range(1_200))
        (haystack / f"essay_{index}.txt").write_text(text, encoding="utf-8")
    entities = tmp_path / "cities.csv"
    entities.write_text(
        "entity,region,category\n"
        + "".join(
            f"City{index},Region{index % 3},Category{index % 2}\n"
            for index in range(30)
        ),
        encoding="utf-8",
    )
    template = tmp_path / "template.txt"
    template.write_text(
        "In the {year} city score audit, {entity} received a score of {score}.\n",
        encoding="utf-8",
    )
    config = _small_config()
    output = tmp_path / "frozen"
    freeze_v4_grid(
        output_dir=output,
        freeze_spec=ControlledFreezeSpec(
            config=config,
            haystack_dir=str(haystack),
            entities_path=str(entities),
            fact_templates_path=str(template),
            minimum_filler_tokens=32,
            hard_negative_gap_tokens=2,
        ),
        require_huggingface_tokenizer=False,
    )
    return config, output


def test_registered_v4_grid_accounting() -> None:
    config = V4Config()
    assert config.design_variants == DESIGN_VARIANTS
    assert len(config.seeds) == 30
    assert len(config.discovery_seeds) == 20
    assert len(config.confirmation_seeds) == 10
    assert (
        len(config.design_variants) * len(config.seeds) * len(config.needle_counts)
        == 1_200
    )
    assert set(config.model_labels) == {"Qwen3-8B", "Gemma4-E4B"}


def test_labeled_attention_pipeline_reaches_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import realistic_niah_v4.pipeline as pipeline

    config = V4Config()
    monkeypatch.setattr(
        pipeline.V4Config,
        "from_json",
        classmethod(lambda _cls, _path: config),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_model_spec",
        lambda _label: SimpleNamespace(label="Qwen3-8B"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_registered_tokenizer",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(pipeline, "select_stimuli", lambda *_a, **_k: [{"row": 1}])
    monkeypatch.setattr(
        pipeline,
        "render_encodings",
        lambda *_a, **_k: iter(["encoded"]),
    )
    observed: dict[str, object] = {}

    def fake_attention(**kwargs):
        observed.update(kwargs)
        return {"table": tmp_path / "attention.csv"}

    monkeypatch.setattr(pipeline, "analyze_labeled_attention", fake_attention)
    monkeypatch.setattr(
        pipeline,
        "label_representation_analysis_by_generation",
        lambda **_kwargs: {"labels": tmp_path / "labels.csv"},
    )
    result = run_labeled_attention_analysis(
        stimuli_path=tmp_path / "stimuli.jsonl",
        config_path=tmp_path / "config.json",
        output_dir=tmp_path / "run",
        model_label="Qwen3-8B",
        answer_format="numeric",
    )
    assert list(observed["encodings"]) == ["encoded"]
    assert result["table"].endswith("attention.csv")
    assert result["representation_labels"].endswith("labels.csv")


def test_causal_design_hash_separates_smoke_and_formal(tmp_path: Path) -> None:
    first = _causal_design_root(
        tmp_path,
        "generation_head_ablation_v1",
        {"layers": [0], "top_ns": [1]},
    )
    repeated = _causal_design_root(
        tmp_path,
        "generation_head_ablation_v1",
        {"layers": [0], "top_ns": [1]},
    )
    formal = _causal_design_root(
        tmp_path,
        "generation_head_ablation_v1",
        {"layers": [0, 8, 16], "top_ns": [1, 2, 4, 8]},
    )
    assert first == repeated
    assert first != formal
    assert json.loads((first / "design.json").read_text())["top_ns"] == [1]


def test_four_panel_freeze_contract_and_audit(
    small_grid: tuple[V4Config, Path],
) -> None:
    config, output = small_grid
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    families = manifest["families"]
    assert manifest["rows"] == 4 * 4 * 3

    def selected(variant: str) -> list[dict[str, object]]:
        return [row for row in families if row["design_variant"] == variant]

    v41 = selected("v4.1")
    assert len({tuple(row["slot_final_starts"]) for row in v41}) == 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v41}) == 1

    v42 = selected("v4.2")
    assert len({tuple(row["slot_final_starts"]) for row in v42}) > 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v42}) == 1

    v43 = selected("v4.3")
    assert len({tuple(row["slot_final_starts"]) for row in v43}) > 1
    assert len({tuple(row["catalog_set_fingerprint"]) for row in v43}) == 1
    assert len({tuple(row["catalog_order_fingerprint"]) for row in v43}) > 1

    v44 = selected("v4.4")
    assert len({tuple(row["catalog_set_fingerprint"]) for row in v44}) > 1

    keyed = {(row["design_variant"], row["seed"]): row for row in families}
    for seed in config.seeds:
        assert (
            keyed[("v4.2", seed)]["slot_final_starts"]
            == keyed[("v4.3", seed)]["slot_final_starts"]
            == keyed[("v4.4", seed)]["slot_final_starts"]
        )
        assert (
            keyed[("v4.3", seed)]["content_permutation_zero_based"]
            == keyed[("v4.4", seed)]["content_permutation_zero_based"]
        )

    rows = load_stimuli(output / "stimuli.jsonl")
    family_rows = [
        row for row in rows if row["design_variant"] == "v4.3" and row["seed"] == 13
    ]
    assert [row["gold_count"] for row in family_rows] == [1, 2, 3]
    assert all(row["canonical_passage_tokens"] == 300 for row in family_rows)
    starts = {
        tuple(slot["canonical_span_start"] for slot in row["slots"])
        for row in family_rows
    }
    assert len(starts) == 1
    for row in family_rows:
        assert [slot["active"] for slot in row["slots"]] == [
            index < row["gold_count"] for index in range(3)
        ]
        assert row["passage"].count("Excerpt:") == 3
        assert row["passage"].count("End excerpt.") == 3
    assert all(
        family["nested_token_identity_outside_toggled_slot"] is True
        for family in families
    )
    for lower, higher in pairwise(family_rows):
        toggled = higher["slots"][higher["gold_count"] - 1]
        start = int(toggled["canonical_span_start"])
        end = int(toggled["canonical_span_end"])
        lower_tokens = lower["passage"].split()
        higher_tokens = higher["passage"].split()
        assert lower_tokens[:start] == higher_tokens[:start]
        assert lower_tokens[end:] == higher_tokens[end:]

    audit = audit_v4_grid(
        stimuli_path=output / "stimuli.jsonl",
        manifest_path=output / "manifest.json",
        require_huggingface_tokenizer=False,
    )
    assert audit["passed"], audit["errors"]


def test_broad_metric_rewards_uniform_span_coverage() -> None:
    needles = tuple(
        TokenSpan(index, index * 3, index * 3 + 1, True, "needle", 1, 1)
        for index in range(1, 4)
    )
    negatives = tuple(
        TokenSpan(index, 20 + index, 21 + index, False, "negative", 1, 1)
        for index in range(1, 4)
    )
    uniform = np.zeros(32)
    concentrated = np.zeros(32)
    uniform[[3, 6, 9]] = 0.2
    concentrated[[3, 6, 9]] = [0.58, 0.01, 0.01]
    uniform[0] = 0.4
    concentrated[0] = 0.4
    broad = broad_attention_metrics(uniform, needles, negatives)
    narrow = broad_attention_metrics(concentrated, needles, negatives)
    assert broad["broad_mass"] == pytest.approx(narrow["broad_mass"])
    assert broad["broad_coverage"] == pytest.approx(1.0)
    assert broad["broad_primary"] > narrow["broad_primary"]
    assert broad["broad_contrast"] > 0


def test_numeric_generation_labels_use_actual_strict_completion() -> None:
    ten = label_generated_completion("10\n", gold_count=10)
    assert ten["outcome_group"] == "correct"
    assert ten["parsed_count"] == 10
    assert ten["count_error"] == 0

    under = label_generated_completion("8", gold_count=10)
    assert under["outcome_group"] == "wrong"
    assert under["error_direction"] == "undercount"
    assert under["omission_count"] == 2

    verbose = label_generated_completion("The answer is 10.", gold_count=10)
    assert verbose["outcome_group"] == "invalid"
    assert verbose["parsed_count"] is None
    assert verbose["extracted_count"] == 10

    out_of_range = parse_numeric_completion("11")
    assert not out_of_range["format_valid"]
    assert out_of_range["parsed_count"] is None


def test_span_end_mean_and_sum_attention_metrics_are_distinct() -> None:
    needles = (
        TokenSpan(1, 2, 4, True, "needle", 2, 2),
        TokenSpan(2, 6, 8, True, "needle", 2, 2),
    )
    negatives = (
        TokenSpan(1, 0, 2, False, "hard_negative", 2, 2),
        TokenSpan(2, 8, 10, False, "hard_negative", 2, 2),
    )
    rows = np.zeros((2, 12), dtype=np.float32)
    rows[0, [2, 3, 6, 7]] = [0.01, 0.20, 0.01, 0.20]
    rows[1, [2, 3, 6, 7]] = [0.20, 0.01, 0.20, 0.01]
    rows[:, [0, 1, 8, 9]] = 0.001
    metrics = layer_pooling_metrics(rows, needles, negatives, key_start=0)
    assert metrics["span_end"]["pool_sum"][0] == pytest.approx(0.4)
    assert metrics["span_end"]["pool_sum"][1] == pytest.approx(0.02)
    assert metrics["span_mean"]["pool_sum"][0] == pytest.approx(0.21)
    assert metrics["span_mean"]["pool_sum"][1] == pytest.approx(0.21)
    assert np.allclose(metrics["span_mean"]["pool_coverage"], 1.0)
    assert metrics["span_sum"]["pool_sum"][0] == pytest.approx(0.42)
    assert metrics["span_sum"]["pool_sum"][1] == pytest.approx(0.42)
    assert np.allclose(metrics["span_sum"]["pool_coverage"], 1.0)
    # Span-sum retains literal full-span mass, but its negative-control
    # contrast and enrichment are computed from per-token density.  For
    # equal-length spans they therefore match span-mean exactly.
    assert np.allclose(
        metrics["span_sum"]["pool_contrast"],
        metrics["span_mean"]["pool_contrast"],
    )
    assert np.allclose(
        metrics["span_sum"]["pool_enrichment"],
        metrics["span_mean"]["pool_enrichment"],
    )


def test_vectorized_attention_outcome_bootstrap_preserves_estimand() -> None:
    rows = []
    for seed in (11, 12, 13):
        for count in (2, 3):
            for outcome, base in (("correct", 0.8), ("wrong", 0.5)):
                rows.append(
                    {
                        "stimulus_id": f"{seed}-{count}-{outcome}",
                        "split": "confirmation",
                        "design_variant": "v4.1",
                        "pooling": "span_sum",
                        "layer": 0,
                        "head": 0,
                        "seed": seed,
                        "count": count,
                        "outcome_group": outcome,
                        "pool_primary": base + 0.001 * seed,
                        "pool_coverage": base + 0.001 * seed,
                        "pool_enrichment": base + 0.001 * seed,
                        "pool_min_to_mean": base + 0.001 * seed,
                    }
                )
    result = confirmation_outcome_effects(
        pd.DataFrame(rows),
        {("v4.1", "span_sum"): ((0, 0),)},
        bootstrap_replicates=100,
        seed=17,
    )
    assert len(result) == 4
    assert set(result["counts_with_both_outcomes"]) == {2}
    assert np.allclose(result["wrong_minus_correct_count_adjusted"], -0.3)
    assert set(result["bootstrap_valid_replicates"]) == {100}

    one_seed = confirmation_outcome_effects(
        pd.DataFrame(rows).query("seed == 11"),
        {("v4.1", "span_sum"): ((0, 0),)},
        bootstrap_replicates=100,
        seed=17,
    )
    assert set(one_seed["bootstrap_valid_replicates"]) == {0}


def test_broad_candidate_ranking_requires_full_grid_visibility() -> None:
    rows = []
    for head in (0, 1):
        for seed, visible in ((11, True), (12, head == 0)):
            row = {
                "stimulus_id": f"s{seed}h{head}",
                "model_label": "toy",
                "design_variant": "v4.1",
                "pooling": "span_end",
                "split": "discovery",
                "count": 2,
                "seed": seed,
                "layer": 0,
                "head": head,
                "layer_type": "full_attention" if head == 0 else "sliding_attention",
                "all_needles_visible": visible,
                "all_hard_negatives_visible": visible,
            }
            row.update({metric: 0.5 for metric in POOL_METRICS})
            row["pool_contrast"] = 0.1
            row["pool_enrichment"] = 2.0
            rows.append(row)
    summary, rankings = rank_broad_candidates(pd.DataFrame(rows), top_k=8)
    assert rankings[("v4.1", "span_end")] == [(0, 0)]
    sliding = summary[summary["head"] == 1].iloc[0]
    assert sliding["full_visibility_rate"] == pytest.approx(0.5)
    assert not bool(sliding["is_broad_candidate"])
    stability = discovery_seed_bootstrap_stability(
        pd.DataFrame(rows), summary, top_k=1, replicates=20
    )
    assert len(stability) == 1
    assert stability.iloc[0]["top_k_selection_frequency"] == pytest.approx(1.0)


def test_representation_rows_inherit_actual_generation_labels(tmp_path: Path) -> None:
    analysis = tmp_path / "representation" / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True)
    label_rows = []
    seed_rows = []
    point_rows = []
    for offset, variant in enumerate(DESIGN_VARIANTS):
        seed = 1254 + offset
        outcome = "correct" if offset % 2 == 0 else "wrong"
        prediction = 10 if outcome == "correct" else 9
        label_rows.append(
            {
                "stimulus_id": f"{variant}-{seed}",
                "design_variant": variant,
                "model_label": "toy",
                "seed": seed,
                "split": "confirmation",
                "gold_count": 10,
                "outcome_group": outcome,
                "is_correct": outcome == "correct",
                "format_valid": True,
                "parsed_count": prediction,
                "count_error": prediction - 10,
            }
        )
        seed_rows.append(
            {
                "model_label": "toy",
                "design_variant": variant,
                "pooling": "span_end",
                "layer": 1,
                "seed": seed,
                "probe_mae": float(offset),
                "curve_residual_rms": float(offset + 1),
                "curve_residual_to_signal": float(offset + 2),
            }
        )
        for count_index in range(1, 11):
            point_rows.append(
                {
                    "design_variant": variant,
                    "pooling": "span_end",
                    "layer": 1,
                    "seed": seed,
                    "split": "confirmation",
                    "count_index": count_index,
                    "pc1": float(count_index),
                    "pc2": float(offset),
                }
            )
    labels_path = tmp_path / "generation_labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    pd.DataFrame(seed_rows).to_csv(
        analysis / "representation_confirmation_by_seed.csv", index=False
    )
    pd.DataFrame(point_rows).to_csv(
        figures / "shared_pca_span_end_layer_1.csv", index=False
    )
    outputs = label_representation_analysis_by_generation(
        analysis_dir=analysis,
        generation_labels_path=labels_path,
        output_dir=analysis / "outcomes",
    )
    labeled = pd.read_csv(outputs["confirmation_by_seed_labeled"])
    assert set(labeled["outcome_group"]) == {"correct", "wrong"}
    assert outputs["manifest"].exists()
    assert (
        analysis
        / "outcomes"
        / "figures"
        / "shared_pca_span_end_layer_1_by_outcome.png"
    ).exists()


def test_nested_increment_diagnostic_targets_newly_activated_needle() -> None:
    prompts = pd.DataFrame(
        [
            {
                "stimulus_id": "n1",
                "model_label": "toy",
                "design_variant": "v4.1",
                "seed": 11,
                "pooling": "span_end",
                "split": "confirmation",
                "count": 1,
                "predicted_count": 1,
            },
            {
                "stimulus_id": "n2",
                "model_label": "toy",
                "design_variant": "v4.1",
                "seed": 11,
                "pooling": "span_end",
                "split": "confirmation",
                "count": 2,
                "predicted_count": 1,
            },
        ]
    )
    occurrences = pd.DataFrame(
        [
            {
                "stimulus_id": "n1",
                "pooling": "span_end",
                "count": 1,
                "occurrence_index": 1,
                "slot_index": 1,
                "normalized_depth": 0.2,
                "ensemble_raw_attention": 0.4,
                "ensemble_normalized_share": 1.0,
                "low_attention_rank": 1,
                "correct_discovery_q10_share": 0.5,
                "below_correct_discovery_q10": False,
            },
            {
                "stimulus_id": "n2",
                "pooling": "span_end",
                "count": 2,
                "occurrence_index": 1,
                "slot_index": 1,
                "normalized_depth": 0.2,
                "ensemble_raw_attention": 0.4,
                "ensemble_normalized_share": 1.8,
                "low_attention_rank": 2,
                "correct_discovery_q10_share": 0.5,
                "below_correct_discovery_q10": False,
            },
            {
                "stimulus_id": "n2",
                "pooling": "span_end",
                "count": 2,
                "occurrence_index": 2,
                "slot_index": 2,
                "normalized_depth": 0.8,
                "ensemble_raw_attention": 0.02,
                "ensemble_normalized_share": 0.2,
                "low_attention_rank": 1,
                "correct_discovery_q10_share": 0.5,
                "below_correct_discovery_q10": True,
            },
        ]
    )
    diagnostics, summary = nested_increment_diagnostics(occurrences, prompts)
    row = diagnostics[diagnostics["count"] == 2].iloc[0]
    assert row["increment_status"] == "failed_to_increment"
    assert row["new_needle_normalized_share"] == pytest.approx(0.2)
    assert bool(row["new_needle_below_correct_discovery_q10"])
    assert summary.iloc[0]["transitions"] == 1


class WhitespaceFastTokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary) + 1
        return self.vocabulary[token]

    def apply_chat_template(self, messages, **kwargs) -> str:
        add_generation_prompt = bool(kwargs.get("add_generation_prompt", False))
        if add_generation_prompt:
            if any(message["role"] != "user" for message in messages):
                raise ValueError("Toy generation prompts contain only the user turn")
            return "\n".join(str(message["content"]) for message in messages) + (
                "\n<assistant>"
            )
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Toy completed chats must end with an assistant turn")
        prefix = "\n".join(
            str(message["content"]) for message in messages[:-1]
        )
        return (
            prefix
            + "\n<assistant>"
            + str(messages[-1]["content"])
            + "\n<eos>"
        )

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        # Separate punctuation and every decimal digit so appending "10" to
        # "Total:" leaves the already-rendered prefix tokens unchanged.
        matches = list(re.finditer(r"[A-Za-z_]+|\d|[^\w\s]", text))
        result = {
            "input_ids": [self._id(match.group(0)) for match in matches],
            "attention_mask": [1] * len(matches),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result


def test_prompt_maps_spans_and_numeric_count_sequences(
    small_grid: tuple[V4Config, Path],
) -> None:
    config, output = small_grid
    row = next(
        item
        for item in load_stimuli(output / "stimuli.jsonl")
        if item["design_variant"] == "v4.1" and item["gold_count"] == 3
    )
    encoding = render_v4_prompt(
        row,
        tokenizer=WhitespaceFastTokenizer(),
        model_spec=MODEL_SPECS["Qwen3-8B"],
        config=config,
        answer_format="numeric",
    )
    assert encoding.design_variant == "v4.1"
    assert encoding.query_position == encoding.sequence_length - 1
    assert len(encoding.needle_spans) == 3
    assert len(encoding.hard_negative_spans) == 3
    assert len(dict(encoding.count_candidate_token_ids)) == 3
    assert encoding.answer_format == "numeric"
    assert "ordinary decimal digits" in encoding.generation_prompt
    assert encoding.text.endswith("Total:")
    assert all(
        len(scored_ids) > len(dict(encoding.count_candidate_answer_token_ids)[count])
        for count, scored_ids in encoding.count_candidate_token_ids
    )
    assert all(span.start < span.end for span in encoding.needle_spans)


class ToyAttention(nn.Module):
    def __init__(self, hidden_size: int = 4, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_type = "global"
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.o_proj(hidden)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.self_attn(hidden) + torch.tanh(self.mlp(hidden))


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=4,
            num_attention_heads=2,
        )
        self.embed = nn.Embedding(40, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(), ToyBlock()])
        self.lm_head = nn.Linear(4, 40, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        logits_to_keep=None,
        **_kwargs,
    ):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep) :]
        return SimpleNamespace(logits=logits)


class FixedNumericDecodeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, _token_ids, **_kwargs):
        return "2"


def _toy_encoding(input_ids: tuple[int, ...]) -> PromptEncoding:
    spans = (
        TokenSpan(1, 1, 2, True, "needle", 1, 1),
        TokenSpan(2, 2, 3, True, "needle", 1, 1),
    )
    return PromptEncoding(
        stimulus_id="toy",
        design_variant="v4.1",
        seed=1,
        split="confirmation",
        count=2,
        model_label="toy",
        answer_format="numeric",
        text="toy",
        generation_prompt="toy",
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        query_position=len(input_ids) - 1,
        slot_spans=spans,
        needle_spans=spans,
        hard_negative_spans=(),
        count_candidate_texts=((1, "1"), (2, "2"), (3, "3")),
        count_candidate_answer_token_ids=((1, (10,)), (2, (11,)), (3, (12,))),
        count_candidate_token_ids=((1, (10, 1)), (2, (11, 1)), (3, (12, 1))),
    )


def test_answer_query_representation_capture_is_all_layer_and_restartable(
    tmp_path: Path,
) -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    encoding = _toy_encoding((1, 2, 3, 4))
    output = tmp_path / "answer_query_all_layers_v1"

    index_path = capture_answer_query_representation_shards(
        model,
        adapter,
        [encoding],
        output_dir=output,
        save_dtype="float32",
    )
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["layer_indices"] == [0, 1]
    assert records[0]["position"] == "prompt_final_total_query"
    shard = output / records[0]["shard_path"]
    with np.load(shard, allow_pickle=False) as saved:
        assert saved["layer_indices"].tolist() == [0, 1]
        assert saved["query_states"].shape == (2, 4)
        assert saved["query_position"].tolist() == [3]
        first_capture = np.asarray(saved["query_states"]).copy()

    restarted = capture_answer_query_representation_shards(
        model,
        adapter,
        [encoding],
        output_dir=output,
        save_dtype="float32",
    )
    assert restarted == index_path
    with np.load(shard, allow_pickle=False) as saved:
        assert np.array_equal(saved["query_states"], first_capture)
    manifest = json.loads(
        (output / "capture_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["rows"] == 1
    assert manifest["layer_indices"] == [0, 1]
    assert manifest["full_sequence_hidden_states_materialized"] is False


def test_prompt_representation_capture_accepts_variable_gold_counts(
    tmp_path: Path,
) -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    count_two = _toy_encoding((1, 2, 3, 4))
    count_one = replace(
        count_two,
        stimulus_id="toy_n1",
        count=1,
        needle_spans=count_two.needle_spans[:1],
    )
    count_two = replace(count_two, stimulus_id="toy_n2")
    output = tmp_path / "prompt_all_counts"

    index_path = capture_representation_shards(
        model,
        adapter,
        [count_one, count_two],
        output_dir=output,
        save_dtype="float32",
    )
    records = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["count"] for row in records] == [1, 2]
    assert [row["array_shape"] for row in records] == [[2, 1, 4], [2, 2, 4]]
    for row in records:
        with np.load(output / row["shard_path"], allow_pickle=False) as saved:
            assert saved["span_end"].shape[1] == row["count"]
            assert saved["span_mean"].shape == saved["span_end"].shape
    manifest = json.loads(
        (output / "capture_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["rows"] == 2
    assert manifest["counts"] == [1, 2]


def test_prompt_counter_attention_metrics_separate_query_and_key_pooling() -> None:
    adapter = discover_decoder_adapter(ToyLM().eval())
    encoding = _toy_encoding((1, 2, 3, 4))
    frame = prompt_counter_attention_metrics(
        [
            np.asarray([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]),
            np.asarray([[0.3, 0.7], [0.6, 0.4]]),
        ],
        [0, 1],
        adapter=adapter,
        encoding=encoding,
        occurrence_index=1,
    )
    assert len(frame) == 4
    assert set(frame["query_site"]) == {"needle_end"}
    assert set(frame["query_occurrence"]) == {2}
    assert set(frame["visible_needle_endpoints"]) == {2}
    assert set(frame["visible_complete_needle_spans"]) == {2}
    selected = frame[(frame["layer"] == 0) & (frame["head"] == 0)].iloc[0]
    assert selected["needle_end_total_mass"] == pytest.approx(0.9)
    assert selected["needle_span_total_mass"] == pytest.approx(0.9)
    assert selected["needle_end_current_share"] == pytest.approx(0.7 / 0.9)
    assert selected["current_endpoint_self_mass"] == pytest.approx(0.7)
    assert selected["prior_needle_endpoint_mass"] == pytest.approx(0.2)
    assert selected["non_needle_mass"] == pytest.approx(0.1)
    assert np.isfinite(frame["row_entropy_nats"]).all()


def test_decoder_hooks_ablation_and_residual_patch() -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    assert adapter.num_layers == 2
    encoding = _toy_encoding((1, 2, 3, 4))
    captured = capture_span_states(model, adapter, encoding)
    assert captured["span_end"].shape == (2, 2, 4)
    assert captured["span_mean"].shape == (2, 2, 4)

    baseline = run_last_logits(model, encoding)
    ablated = run_with_head_ablation(
        model,
        adapter,
        encoding,
        [(0, 0)],
        scope="answer_query",
    )
    assert not torch.allclose(baseline, ablated)

    donor = _toy_encoding((5, 6, 7, 8))
    donor_states = capture_span_states(
        model,
        adapter,
        donor,
        spans=[TokenSpan(1, 3, 4, True, "query", 1, 1)],
        layers=[0],
    )["span_end"][0]
    patched = run_with_residual_patch(
        model,
        adapter,
        encoding,
        layer=0,
        receiver_positions=[encoding.query_position],
        donor_states=donor_states,
    )
    assert patched.shape == baseline.shape
    assert not torch.allclose(baseline, patched)

    metrics = count_logit_metrics(baseline, encoding)
    assert {
        metrics["predicted_count_among_candidates"],
        metrics["gold_count"],
    }.issubset({1, 2, 3})
    assert 0.0 <= metrics["correct_count_probability"] <= 1.0


def test_complete_generation_intervention_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import realistic_niah_v4.modeling as modeling

    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    receiver = _toy_encoding((1, 2, 3, 4))
    donor = replace(
        _toy_encoding((5, 6, 7, 8)),
        stimulus_id="toy_donor",
        count=3,
    )

    def fake_generation(
        fake_model,
        _tokenizer,
        encoding,
        *,
        max_new_tokens=16,
    ):
        del max_new_tokens
        input_ids = torch.tensor([encoding.input_ids], dtype=torch.long)
        attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long)
        output = fake_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return {
            "completion_text": "10",
            "completion_text_raw": "10",
            "generated_token_ids": [31, 30],
            "generation_truncated": False,
            "probe_logits": output.logits[0, -1].detach().clone(),
        }

    monkeypatch.setattr(modeling, "generate_answer_completion", fake_generation)
    baseline = fake_generation(model, None, receiver)["probe_logits"]
    ablated = generate_with_head_ablation(
        model,
        None,
        adapter,
        receiver,
        [(0, 0)],
        scope="answer_query",
    )
    assert not torch.allclose(baseline, ablated["probe_logits"])

    _donor_logits, donor_states = capture_post_block_states(
        model,
        adapter,
        donor,
        [donor.query_position],
        layers=[0],
    )
    patched_residual = generate_with_residual_interventions(
        model,
        None,
        adapter,
        receiver,
        {0: ([receiver.query_position], donor_states[0])},
    )
    assert not torch.allclose(baseline, patched_residual["probe_logits"])
    after = fake_generation(model, None, receiver)["probe_logits"]
    assert torch.allclose(baseline, after)


def test_complete_numeric_outcome_handles_multitoken_ten() -> None:
    encoding = replace(
        _toy_encoding((1, 2, 3, 4)),
        stimulus_id="toy_ten",
        count=10,
    )
    baseline = {
        "model_label": "toy",
        "design_variant": "v4.1",
        "seed": 1,
        "gold_count": 10,
        "outcome_group": "wrong",
        "is_correct": False,
        "format_valid": True,
        "parsed_count": 9,
        "count_error": -1,
    }
    outcome = intervention_outcome(
        {
            "completion_text": "10",
            "completion_text_raw": "10",
            "generated_token_ids": [31, 30],
            "generation_truncated": False,
        },
        encoding,
        baseline,
    )
    assert outcome["patched_predicted_count"] == 10
    assert outcome["patched_is_correct"] is True
    assert outcome["generated_count_shift"] == 1
    assert json.loads(outcome["patched_generated_token_ids"]) == [31, 30]


def test_generation_ablation_can_select_span_end_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import realistic_niah_v4.causal_generation as causal_generation

    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    encoding = _toy_encoding((1, 2, 3, 4))
    calls: list[tuple[tuple[tuple[int, int], ...], str]] = []

    def fake_ablation(
        _model,
        _tokenizer,
        _adapter,
        _encoding,
        heads,
        *,
        scope,
        max_new_tokens,
    ):
        del max_new_tokens
        calls.append((tuple(heads), str(scope)))
        return {
            "completion_text": "2",
            "completion_text_raw": "2",
            "generated_token_ids": [11, 1],
            "generation_truncated": False,
            "intervention_hook_applications": {"0": 1},
        }

    monkeypatch.setattr(
        causal_generation, "generate_with_head_ablation", fake_ablation
    )
    detail = run_generation_head_ablation(
        model,
        None,
        adapter,
        [encoding],
        baseline_labels={
            encoding.stimulus_id: {
                "model_label": "toy",
                "design_variant": "v4.1",
                "seed": 1,
                "gold_count": 2,
                "outcome_group": "correct",
                "is_correct": True,
                "format_valid": True,
                "parsed_count": 2,
                "count_error": 0,
            }
        },
        rankings={("v4.1", "span_end"): [(0, 0), (1, 0)]},
        poolings=("span_end",),
        top_ns=(1,),
        random_replicates=1,
        scopes=("answer_query",),
    )
    assert set(detail["pooling"]) == {"span_end"}
    assert set(detail["condition"]) == {"ranked", "layer_matched_random"}
    assert len(calls) == 2
    assert all(scope == "answer_query" for _, scope in calls)


def test_geometric_chord_and_polyline_are_distinct(tmp_path: Path) -> None:
    centroids = np.asarray(
        [[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32
    )
    bundle = CountCentroidBundle(
        variants=("v4.1",),
        layers=(0,),
        counts=(1, 2, 3),
        centroids=centroids,
        sample_counts=np.asarray([[2, 2, 2]], dtype=np.int32),
        discovery_seeds=(11, 12),
    )
    bundle.validate()
    chord, chord_count = chord_point(
        bundle,
        variant="v4.1",
        layer=0,
        receiver_count=1,
        target_count=3,
        alpha=0.5,
    )
    curve, curve_count = polyline_point(
        bundle,
        variant="v4.1",
        layer=0,
        receiver_count=1,
        target_count=3,
        alpha=0.5,
    )
    assert torch.allclose(chord, torch.tensor([0.5, 0.5]))
    assert torch.allclose(curve, torch.tensor([1.0, 0.0]))
    assert chord_count == curve_count == 2.0

    path = save_centroid_bundle(bundle, tmp_path / "centroids.npz")
    loaded = load_centroid_bundle(path)
    assert np.array_equal(loaded.centroids, bundle.centroids)
    summary, adjacent = centroid_geometry_tables(loaded)
    assert len(summary) == 1
    assert len(adjacent) == 2
    assert summary.iloc[0]["path_tortuosity"] == pytest.approx(2**0.5)


def test_layer_set_centroid_delta_uses_each_layers_full_state() -> None:
    bundle = CountCentroidBundle(
        variants=("v4.1",),
        layers=(1, 2),
        counts=(1, 2),
        centroids=np.asarray(
            [
                [
                    [[1.0, 2.0], [3.0, 6.0]],
                    [[0.0, 0.0], [2.0, -2.0]],
                ]
            ],
            dtype=np.float32,
        ),
        sample_counts=np.asarray([[2, 2]], dtype=np.int32),
        discovery_seeds=(11, 12),
    )
    plan = LayerSetSteeringPlan(layers=(1, 2), alpha=0.5)
    plan.validate(bundle)
    replacements, deltas = _layer_set_centroid_delta_states(
        bundle,
        {1: torch.tensor([10.0, 10.0]), 2: torch.tensor([20.0, 20.0])},
        variant="v4.1",
        receiver_count=1,
        target_count=2,
        plan=plan,
    )
    assert plan.protocol == "multi_layer"
    assert torch.allclose(deltas[1], torch.tensor([1.0, 2.0]))
    assert torch.allclose(deltas[2], torch.tensor([1.0, -1.0]))
    assert torch.allclose(replacements[1], torch.tensor([11.0, 12.0]))
    assert torch.allclose(replacements[2], torch.tensor([21.0, 19.0]))


def test_layer_set_discovery_selection_maximizes_worst_panel() -> None:
    rows = []
    plans = {
        ("single_layer", "1", 0.5): [0.8, 0.8, 0.8, -0.2],
        ("single_layer", "2", 1.0): [0.3, 0.3, 0.3, 0.3],
        ("multi_layer", "1+2", 0.5): [0.5, 0.5, 0.5, 0.5],
        ("multi_layer", "2+3", 1.0): [0.7, 0.7, 0.7, -0.1],
    }
    for (protocol, layer_set, alpha), effects in plans.items():
        for variant_index, (variant, effect) in enumerate(
            zip(("v4.1", "v4.2", "v4.3", "v4.4"), effects)
        ):
            for seed in (11, 12):
                common = {
                    "model_label": "toy",
                    "design_variant": variant,
                    "seed": seed,
                    "receiver_stimulus_id": f"r-{variant_index}-{seed}",
                    "target_stimulus_id": f"t-{variant_index}-{seed}",
                    "receiver_count": 7,
                    "target_count": 8,
                    "target_direction": "increase",
                    "steering_protocol": protocol,
                    "layer_set": layer_set,
                    "alpha": alpha,
                    "patched_format_valid": True,
                    "moved_toward_donor_gold": effect > 0,
                    "follows_donor_gold": False,
                }
                rows.append(
                    {
                        **common,
                        "condition": "geometric",
                        "direction_aligned_generated_count_shift": effect,
                    }
                )
                rows.append(
                    {
                        **common,
                        "condition": "orthogonal_norm_matched_random",
                        "direction_aligned_generated_count_shift": 0.0,
                    }
                )
    detail = pd.DataFrame(rows)
    scores = layer_set_steering_plan_scores(detail)
    selected = select_layer_set_steering_plans(detail)["selected"]
    assert not scores.empty
    assert selected["single_layer"]["layer_set"] == "2"
    assert selected["multi_layer"]["layer_set"] == "1+2"


def test_count_centroids_require_complete_discovery_grid(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    rows = []
    for seed in (11, 12):
        for count in (1, 2, 3):
            relative = Path("shards") / f"seed{seed}_count{count}.npz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                layer_indices=np.asarray([0], dtype=np.int64),
                query_states=np.asarray(
                    [[float(count), float(seed - 10)]], dtype=np.float16
                ),
            )
            rows.append(
                {
                    "stimulus_id": f"seed{seed}_count{count}",
                    "design_variant": "v4.1",
                    "seed": seed,
                    "split": "discovery",
                    "count": count,
                    "shard_path": relative.as_posix(),
                }
            )
    bundle = fit_count_centroids(
        rows,
        capture_root=root,
        variants=("v4.1",),
        layers=(0,),
        counts=(1, 2, 3),
        discovery_seeds=(11, 12),
    )
    assert torch.allclose(bundle.state("v4.1", 0, 2), torch.tensor([2.0, 1.5]))
    with pytest.raises(ValueError, match="discovery grid mismatch"):
        fit_count_centroids(
            rows[:-1],
            capture_root=root,
            variants=("v4.1",),
            layers=(0,),
            counts=(1, 2, 3),
            discovery_seeds=(11, 12),
        )


def test_residual_span_patch_uses_token_states_not_span_mean() -> None:
    assert RESIDUAL_PATCH_SITES == (
        "answer_query",
        "toggled_needle_end",
        "toggled_needle_span",
    )
    donor = {0: torch.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]])}
    receiver = {0: torch.tensor([[8.0, 8.0], [5.0, 6.0], [7.0, 8.0]])}
    positions, states, ok, reason = _residual_site_states(
        site="toggled_needle_span",
        donor_states=donor,
        receiver_states=receiver,
        donor_span_offset=1,
        donor_span_length=2,
        receiver_span_offset=1,
        receiver_span_length=2,
        receiver_positions=[10, 20, 21],
        layer=0,
    )
    assert ok is True and reason == ""
    assert positions == (20, 21)
    assert torch.equal(states, donor[0][1:3])

    _positions, _states, ok, reason = _residual_site_states(
        site="toggled_needle_span",
        donor_states=donor,
        receiver_states={0: torch.tensor([[8.0, 8.0], [5.0, 6.0]])},
        donor_span_offset=1,
        donor_span_length=2,
        receiver_span_offset=1,
        receiver_span_length=1,
        receiver_positions=[10, 20],
        layer=0,
    )
    assert ok is False
    assert reason == "model_token_length_mismatch"


def test_raw_answer_query_attention_round_trip(tmp_path: Path) -> None:
    model = ToyLM().eval()
    adapter = discover_decoder_adapter(model)
    encoding = _toy_encoding((1, 2, 3, 4))
    raw_rows = [
        torch.softmax(
            torch.arange(
                adapter.num_heads[layer] * encoding.sequence_length,
                dtype=torch.float32,
            ).reshape(adapter.num_heads[layer], encoding.sequence_length),
            dim=-1,
        )
        for layer in range(adapter.num_layers)
    ]
    shard = tmp_path / "raw_attention.npz"
    _write_raw_attention_shard(
        shard,
        attention_rows=raw_rows,
        key_starts=[0] * adapter.num_layers,
        adapter=adapter,
        encoding=encoding,
        save_dtype="float16",
    )
    _validate_raw_attention_shard(
        shard,
        adapter=adapter,
        encoding=encoding,
    )
    with np.load(shard, allow_pickle=False) as saved:
        assert saved["layer_000"].dtype == np.float16
        assert saved["layer_000"].shape == (
            adapter.num_heads[0],
            encoding.sequence_length,
        )


def test_attention_tensor_accepts_single_block_query() -> None:
    tensor = torch.zeros(1, 2, 1, 1, 7)
    assert _attention_tensor(tensor).shape == (1, 2, 1, 7)
    with pytest.raises(RuntimeError, match="more than one query cell"):
        _attention_tensor(torch.zeros(1, 2, 2, 1, 7))


def test_tiny_transformers_architectures_support_v4_hooks() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "AutoModelForMultimodalLM"):
        pytest.skip("Installed Transformers predates the Gemma 4 API")
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4AudioConfig,
            Gemma4Config,
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )
    except ImportError:
        pytest.skip("Installed Transformers lacks a registered V4 architecture")

    spans = tuple(
        TokenSpan(index, position, position + 1, True, "needle", 1, 1)
        for index, position in enumerate((2, 5, 8), start=1)
    )
    negatives = tuple(
        TokenSpan(index, position, position + 1, False, "negative", 1, 1)
        for index, position in enumerate((3, 6, 9), start=1)
    )

    def encoding(label: str) -> PromptEncoding:
        return PromptEncoding(
            stimulus_id=f"tiny_{label}",
            design_variant="v4.1",
            seed=1,
            split="confirmation",
            count=3,
            model_label=label,
            answer_format="numeric",
            text="",
            generation_prompt="",
            input_ids=tuple(range(4, 16)),
            attention_mask=(1,) * 12,
            query_position=11,
            slot_spans=spans,
            needle_spans=spans,
            hard_negative_spans=negatives,
            count_candidate_texts=tuple(
                (count, str(count)) for count in range(1, 11)
            ),
            count_candidate_answer_token_ids=tuple(
                (count, (20 + count,)) for count in range(1, 11)
            ),
            count_candidate_token_ids=tuple(
                (count, (20 + count, 1)) for count in range(1, 11)
            ),
        )

    models = [
        (
            "qwen",
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=4,
                    max_position_embeddings=128,
                    layer_types=["full_attention", "full_attention"],
                )
            ),
        ),
        (
            "gemma4",
            Gemma4ForConditionalGeneration(
                Gemma4Config(
                    text_config=Gemma4TextConfig(
                        vocab_size=64,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=4,
                        global_head_dim=4,
                        num_global_key_value_heads=2,
                        max_position_embeddings=128,
                        sliding_window=8,
                        layer_types=["sliding_attention", "full_attention"],
                        vocab_size_per_layer_input=64,
                        hidden_size_per_layer_input=0,
                        num_kv_shared_layers=0,
                    ),
                    vision_config=Gemma4VisionConfig(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        head_dim=4,
                        max_position_embeddings=128,
                        patch_size=4,
                        position_embedding_size=64,
                    ),
                    audio_config=Gemma4AudioConfig(
                        hidden_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        subsampling_conv_channels=(4, 4),
                        output_proj_dims=16,
                    ),
                    boi_token_id=50,
                    eoi_token_id=51,
                    image_token_id=52,
                    video_token_id=53,
                    boa_token_id=54,
                    eoa_token_index=55,
                    audio_token_id=56,
                )
            ),
        ),
    ]
    for label, model in models:
        model.eval()
        adapter = discover_decoder_adapter(model)
        if label == "gemma4":
            assert adapter.layer_container_name == "model.language_model.layers"
        item = encoding(label)
        baseline = run_last_logits(model, item)
        captured = capture_span_states(model, adapter, item)
        rows, key_starts = query_attention_rows(model, adapter, item)
        prompt_rows, prompt_key_starts, _ = position_attention_outputs(
            model,
            adapter,
            item,
            item.needle_spans[-1].end - 1,
        )
        ablated = run_with_head_ablation(
            model, adapter, item, [(0, 0)], scope="answer_query"
        )
        _, states = capture_post_block_states(
            model,
            adapter,
            item,
            [item.query_position],
            layers=[0],
        )
        patched = run_with_residual_patch(
            model,
            adapter,
            item,
            layer=0,
            receiver_positions=[item.query_position],
            donor_states=states[0] + 0.1,
        )
        generated_ablation = generate_with_head_ablation(
            model,
            FixedNumericDecodeTokenizer(),
            adapter,
            item,
            [(0, 0)],
            max_new_tokens=1,
        )
        generated_residual_patch = generate_with_residual_interventions(
            model,
            FixedNumericDecodeTokenizer(),
            adapter,
            item,
            {0: ([item.query_position], states[0] + 0.1)},
            max_new_tokens=1,
        )
        assert captured["span_end"].shape == (2, 3, 16)
        expected_attention_shapes = (
            [(4, 8), (4, 12)] if label == "gemma4" else [(4, 12), (4, 12)]
        )
        assert [tuple(row.shape) for row in rows] == expected_attention_shapes
        assert key_starts == ([4, 0] if label == "gemma4" else [0, 0])
        prompt_key_end = item.needle_spans[-1].end
        assert all(
            key_start + row.shape[-1] == prompt_key_end
            for row, key_start in zip(prompt_rows, prompt_key_starts)
        )
        assert all(
            torch.isfinite(row).all() and torch.allclose(
                row.sum(dim=-1),
                torch.ones(row.shape[0]),
                atol=1e-5,
            )
            for row in prompt_rows
        )
        assert not torch.allclose(baseline, ablated)
        assert not torch.allclose(baseline, patched)
        assert generated_ablation["completion_text"] == "2"
        assert generated_residual_patch["completion_text"] == "2"
        assert generated_residual_patch["intervention_hook_applications"] == {
            "0": 1
        }


def test_synthetic_representation_analysis_recovers_count_curve(
    tmp_path: Path,
) -> None:
    config = V4Config(
        seeds=(1, 2, 3, 4, 5, 6, 7, 8),
        discovery_seeds=(1, 2, 3, 4, 5, 6),
        confirmation_seeds=(7, 8),
    )
    config.validate()
    capture = tmp_path / "capture"
    index_rows = []
    direction = np.asarray([1.0, -0.5, 0.25, 0.1, -0.2, 0.4])
    noise_scale = {
        "v4.1": 0.01,
        "v4.2": 0.03,
        "v4.3": 0.12,
        "v4.4": 0.35,
    }
    for variant_index, variant in enumerate(config.design_variants):
        for seed in config.seeds:
            rng = np.random.default_rng(1000 * variant_index + seed)
            array = np.empty((2, 10, 6), dtype=np.float32)
            array[0] = rng.normal(size=(10, 6))
            for count in range(1, 11):
                array[1, count - 1] = float(count) * direction + rng.normal(
                    scale=noise_scale[variant], size=direction.shape
                )
            relative = Path("shards") / variant / f"{variant}_{seed}.npz"
            shard = capture / relative
            shard.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                shard,
                layer_indices=np.asarray([0, 1]),
                span_end=array,
                span_mean=array + 0.001,
            )
            index_rows.append(
                {
                    "schema_version": "realistic_niah_v4_capture_v1",
                    "stimulus_id": f"{variant}_{seed}",
                    "design_variant": variant,
                    "model_label": "synthetic",
                    "answer_format": "numeric",
                    "seed": seed,
                    "split": (
                        "discovery"
                        if seed in config.discovery_seeds
                        else "confirmation"
                    ),
                    "count": 10,
                    "sequence_length": 100,
                    "query_position": 99,
                    "poolings": ["span_end", "span_mean"],
                    "array_shape": [2, 10, 6],
                    "save_dtype": "float32",
                    "shard_path": relative.as_posix(),
                }
            )
    index_path = capture / "capture_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    outputs = analyze_representation_captures(
        capture_index_path=index_path,
        output_dir=tmp_path / "analysis",
        config=config,
    )
    metrics = np.genfromtxt(
        outputs["layer_metrics"],
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    matching = [
        row
        for row in metrics
        if row["design_variant"] == "v4.1"
        and row["pooling"] == "span_end"
        and int(row["layer"]) == 1
    ]
    assert len(matching) == 1
    assert float(matching[0]["confirmation_r2"]) > 0.95
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["primary_layer_selection"]["layers"]["span_end"] == 1
    assert (
        tmp_path / "analysis" / "figures" / "shared_pca_span_end_layer_1.png"
    ).exists()
