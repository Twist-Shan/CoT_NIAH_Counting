from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .attention import Head, matched_random_heads
from .behavior import count_logit_metrics
from .modeling import (
    DecoderAdapter,
    capture_post_block_states,
    run_last_logits,
    run_with_head_ablation,
    run_with_residual_patch,
)
from .prompts import PromptEncoding


PATCH_SITES = (
    "answer_query",
    "toggled_needle_end",
    "toggled_needle_span",
)


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _base_metadata(encoding: PromptEncoding) -> dict[str, Any]:
    return {
        "stimulus_id": encoding.stimulus_id,
        "design_variant": encoding.design_variant,
        "model_label": encoding.model_label,
        "seed": int(encoding.seed),
        "split": encoding.split,
        "count": int(encoding.count),
        "sequence_length": int(encoding.sequence_length),
    }


def run_head_ablation_experiment(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    rankings: Mapping[str, Sequence[Head]],
    top_ns: Sequence[int],
    random_replicates: int,
    scope: str = "answer_query",
) -> pd.DataFrame:
    """Compare discovery-ranked broad heads with layer-matched random heads."""

    rows: list[dict[str, Any]] = []
    control_sets: dict[tuple[str, int, int], list[Head]] = {}
    for variant, ranking in rankings.items():
        for top_n in top_ns:
            selected = list(ranking[: int(top_n)])
            if len(selected) != int(top_n):
                raise ValueError(
                    f"{variant} ranking has fewer than top_n={top_n} heads"
                )
            for replicate in range(int(random_replicates)):
                control_sets[(variant, int(top_n), replicate)] = matched_random_heads(
                    selected,
                    adapter,
                    seed=_stable_seed(f"{variant}:top{top_n}:random{replicate}"),
                )
    for example_index, encoding in enumerate(encodings):
        if encoding.design_variant not in rankings:
            raise KeyError(f"No broad-head ranking for {encoding.design_variant}")
        baseline_logits = run_last_logits(model, encoding)
        baseline = count_logit_metrics(baseline_logits, encoding)
        metadata = _base_metadata(encoding)
        rows.append(
            {
                **metadata,
                "example_index": int(example_index),
                "condition": "baseline",
                "top_n": 0,
                "random_replicate": -1,
                "scope": scope,
                "heads": "",
                **baseline,
                "delta_correct_margin": 0.0,
                "delta_expected_count": 0.0,
                "expected_count_absolute_error": abs(
                    float(baseline["expected_count"]) - encoding.count
                ),
                "delta_expected_count_absolute_error": 0.0,
            }
        )
        ranking = list(rankings[encoding.design_variant])
        for top_n in top_ns:
            selected = ranking[: int(top_n)]
            conditions: list[tuple[str, int, Sequence[Head]]] = [
                ("broad", -1, selected)
            ]
            conditions.extend(
                (
                    "layer_matched_random",
                    replicate,
                    control_sets[(encoding.design_variant, int(top_n), replicate)],
                )
                for replicate in range(int(random_replicates))
            )
            for condition, replicate, heads in conditions:
                logits = run_with_head_ablation(
                    model,
                    adapter,
                    encoding,
                    heads,
                    scope=scope,
                )
                metrics = count_logit_metrics(logits, encoding)
                rows.append(
                    {
                        **metadata,
                        "example_index": int(example_index),
                        "condition": condition,
                        "top_n": int(top_n),
                        "random_replicate": int(replicate),
                        "scope": scope,
                        "heads": ",".join(
                            f"L{int(layer)}H{int(head)}" for layer, head in heads
                        ),
                        **metrics,
                        "delta_correct_margin": float(
                            metrics["correct_count_margin"]
                            - baseline["correct_count_margin"]
                        ),
                        "delta_expected_count": float(
                            metrics["expected_count"] - baseline["expected_count"]
                        ),
                        "expected_count_absolute_error": abs(
                            float(metrics["expected_count"]) - encoding.count
                        ),
                        "delta_expected_count_absolute_error": float(
                            abs(float(metrics["expected_count"]) - encoding.count)
                            - abs(float(baseline["expected_count"]) - encoding.count)
                        ),
                    }
                )
        print(
            "[v4 ablation] "
            f"{example_index + 1}/{len(encodings)} "
            f"{encoding.design_variant} seed={encoding.seed} N={encoding.count}",
            flush=True,
        )
    if not rows:
        raise ValueError("No V4 ablation encodings were supplied")
    return pd.DataFrame(rows)


def summarize_head_ablation(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_label",
        "design_variant",
        "condition",
        "top_n",
        "seed",
        "count",
        "delta_correct_margin",
        "delta_expected_count",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Ablation table is missing columns: {missing}")
    experimental = detail[detail["condition"] != "baseline"].copy()
    if experimental.empty:
        raise ValueError("Ablation table has no experimental rows")
    groups = [
        "model_label",
        "design_variant",
        "condition",
        "top_n",
    ]
    summary = experimental.groupby(groups, as_index=False).agg(
        examples=("stimulus_id", "count"),
        seeds=("seed", "nunique"),
        mean_delta_correct_margin=("delta_correct_margin", "mean"),
        sd_delta_correct_margin=("delta_correct_margin", "std"),
        mean_delta_expected_count=("delta_expected_count", "mean"),
        mean_delta_expected_count_absolute_error=(
            "delta_expected_count_absolute_error",
            "mean",
        ),
    )
    # Pandas aggregation cannot compare predicted/gold columns directly.
    accuracy = (
        experimental.assign(
            candidate_correct=(
                experimental["predicted_count_among_candidates"]
                == experimental["gold_count"]
            ).astype(float)
        )
        .groupby(groups, as_index=False)["candidate_correct"]
        .mean()
        .rename(columns={"candidate_correct": "candidate_accuracy"})
    )
    summary = summary.merge(accuracy, on=groups, how="left")
    return summary


def compare_head_ablation_to_random(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    """Paired seed-cluster comparison of broad and random-head damage."""

    experimental = detail[detail["condition"] != "baseline"].copy()
    identifier = [
        "model_label",
        "design_variant",
        "stimulus_id",
        "seed",
        "count",
        "top_n",
    ]
    broad = experimental[experimental["condition"] == "broad"]
    random_rows = experimental[experimental["condition"] == "layer_matched_random"]
    metrics = [
        "delta_correct_margin",
        "delta_expected_count_absolute_error",
    ]
    random_mean = (
        random_rows.groupby(identifier, as_index=False)[metrics]
        .mean()
        .rename(columns={metric: f"{metric}_random_mean" for metric in metrics})
    )
    paired = broad.merge(random_mean, on=identifier, how="inner")
    output: list[dict[str, Any]] = []
    groups = ["model_label", "design_variant", "top_n"]
    for keys, frame in paired.groupby(groups, sort=True):
        model_label, variant, top_n = keys
        for metric in metrics:
            difference = frame[metric] - frame[f"{metric}_random_mean"]
            seed_means = (
                frame.assign(difference=difference)
                .groupby("seed")["difference"]
                .mean()
                .to_numpy(dtype=float)
            )
            rng = np.random.default_rng(_stable_seed(f"{variant}:{top_n}:{metric}"))
            indices = rng.integers(
                0,
                len(seed_means),
                size=(int(bootstrap_repetitions), len(seed_means)),
            )
            bootstrap = seed_means[indices].mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            output.append(
                {
                    "model_label": model_label,
                    "design_variant": variant,
                    "top_n": int(top_n),
                    "metric": metric,
                    "broad_minus_random_mean": float(seed_means.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "confirmation_seeds": len(seed_means),
                    "bootstrap_repetitions": int(bootstrap_repetitions),
                }
            )
    return pd.DataFrame(output)


def _validate_patch_pair(
    receiver: PromptEncoding,
    donor: PromptEncoding,
) -> None:
    if receiver.design_variant != donor.design_variant:
        raise ValueError("Patch donor and receiver must use one design variant")
    if receiver.seed != donor.seed:
        raise ValueError("Patch donor and receiver must share a seed")
    if receiver.model_label != donor.model_label:
        raise ValueError("Patch donor and receiver must share a model")
    if receiver.count >= donor.count:
        raise ValueError("Patch direction must be lower-count to higher-count")
    if receiver.count_candidate_token_ids != donor.count_candidate_token_ids:
        raise ValueError("Patch pair has different count-token candidates")


def _patch_score(
    logits: torch.Tensor,
    encoding: PromptEncoding,
    *,
    receiver_count: int,
    donor_count: int,
) -> float:
    candidate_ids = dict(encoding.count_candidate_token_ids)
    return float(
        logits[candidate_ids[int(donor_count)]]
        - logits[candidate_ids[int(receiver_count)]]
    )


def _recovery_fraction(
    receiver_score: float,
    donor_score: float,
    patched_score: float,
) -> float:
    denominator = donor_score - receiver_score
    return (
        (patched_score - receiver_score) / denominator
        if abs(denominator) > 1e-12
        else math.nan
    )


def run_hidden_patching_experiment(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    layers: Sequence[int],
    count_pairs: Sequence[tuple[int, int]],
    sites: Sequence[str] = PATCH_SITES,
) -> pd.DataFrame:
    """Patch donor post-block residuals into a paired receiver prompt.

    Sites:
      - answer_query: donor's final `Total:` query state.
      - toggled_needle_end: final token of the newly activated slot.
      - toggled_needle_span: all slot tokens, only when model-token lengths match.
    """

    requested_sites = tuple(str(site) for site in sites)
    if (
        not requested_sites
        or len(set(requested_sites)) != len(requested_sites)
        or any(site not in PATCH_SITES for site in requested_sites)
    ):
        raise ValueError(f"Invalid V4 patch sites: {requested_sites}")
    by_key = {
        (
            encoding.design_variant,
            int(encoding.seed),
            int(encoding.count),
        ): encoding
        for encoding in encodings
    }
    rows: list[dict[str, Any]] = []
    for variant in sorted({encoding.design_variant for encoding in encodings}):
        seeds = sorted(
            {
                int(encoding.seed)
                for encoding in encodings
                if encoding.design_variant == variant
            }
        )
        for seed in seeds:
            for receiver_count, donor_count in count_pairs:
                receiver = by_key.get((variant, seed, int(receiver_count)))
                donor = by_key.get((variant, seed, int(donor_count)))
                if receiver is None or donor is None:
                    raise KeyError(
                        f"Missing patch pair {variant}, seed={seed}, "
                        f"N={receiver_count}->{donor_count}"
                    )
                _validate_patch_pair(receiver, donor)
                toggled_slot_index = int(donor_count)
                donor_span = donor.slot_spans[toggled_slot_index - 1]
                receiver_span = receiver.slot_spans[toggled_slot_index - 1]
                donor_positions = [int(donor.query_position)] + list(
                    range(int(donor_span.start), int(donor_span.end))
                )
                donor_logits, donor_states = capture_post_block_states(
                    model,
                    adapter,
                    donor,
                    donor_positions,
                    layers=layers,
                )
                receiver_logits = run_last_logits(model, receiver)
                receiver_score = _patch_score(
                    receiver_logits,
                    receiver,
                    receiver_count=receiver_count,
                    donor_count=donor_count,
                )
                donor_score = _patch_score(
                    donor_logits,
                    donor,
                    receiver_count=receiver_count,
                    donor_count=donor_count,
                )
                receiver_metrics = count_logit_metrics(receiver_logits, receiver)
                donor_metrics = count_logit_metrics(donor_logits, donor)
                span_lengths_match = (
                    donor_span.model_token_length == receiver_span.model_token_length
                )
                for layer in layers:
                    layer = int(layer)
                    states = donor_states[layer]
                    site_specs: dict[
                        str, tuple[Sequence[int], torch.Tensor, bool, str]
                    ] = {
                        "answer_query": (
                            [receiver.query_position],
                            states[0:1],
                            True,
                            "",
                        ),
                        "toggled_needle_end": (
                            [receiver_span.end - 1],
                            states[-1:],
                            True,
                            "",
                        ),
                        "toggled_needle_span": (
                            list(range(receiver_span.start, receiver_span.end)),
                            states[1:],
                            span_lengths_match,
                            (
                                ""
                                if span_lengths_match
                                else "model_token_length_mismatch"
                            ),
                        ),
                    }
                    selected_site_specs = [
                        (
                            site,
                            *site_specs[site],
                        )
                        for site in requested_sites
                    ]
                    for (
                        site,
                        receiver_positions,
                        patch_states,
                        executable,
                        skip_reason,
                    ) in selected_site_specs:
                        metadata = {
                            "model_label": receiver.model_label,
                            "design_variant": variant,
                            "seed": int(seed),
                            "split": receiver.split,
                            "receiver_stimulus_id": receiver.stimulus_id,
                            "donor_stimulus_id": donor.stimulus_id,
                            "receiver_count": int(receiver_count),
                            "donor_count": int(donor_count),
                            "toggled_slot_index": toggled_slot_index,
                            "layer": layer,
                            "site": site,
                            "receiver_positions": ",".join(
                                str(int(value)) for value in receiver_positions
                            ),
                            "donor_model_token_length": int(
                                donor_span.model_token_length
                            ),
                            "receiver_model_token_length": int(
                                receiver_span.model_token_length
                            ),
                            "status": "ok" if executable else "skipped",
                            "skip_reason": skip_reason,
                            "receiver_pair_logit_score": receiver_score,
                            "donor_pair_logit_score": donor_score,
                            "receiver_expected_count": float(
                                receiver_metrics["expected_count"]
                            ),
                            "donor_expected_count": float(
                                donor_metrics["expected_count"]
                            ),
                        }
                        if not executable:
                            rows.append(
                                {
                                    **metadata,
                                    "patched_pair_logit_score": math.nan,
                                    "pair_logit_score_delta": math.nan,
                                    "pair_logit_recovery_fraction": math.nan,
                                    "patched_expected_count": math.nan,
                                    "expected_count_delta": math.nan,
                                }
                            )
                            continue
                        patched_logits = run_with_residual_patch(
                            model,
                            adapter,
                            receiver,
                            layer=layer,
                            receiver_positions=receiver_positions,
                            donor_states=patch_states,
                        )
                        patched_metrics = count_logit_metrics(patched_logits, receiver)
                        patched_score = _patch_score(
                            patched_logits,
                            receiver,
                            receiver_count=receiver_count,
                            donor_count=donor_count,
                        )
                        rows.append(
                            {
                                **metadata,
                                "patched_pair_logit_score": patched_score,
                                "pair_logit_score_delta": (
                                    patched_score - receiver_score
                                ),
                                "pair_logit_recovery_fraction": (
                                    _recovery_fraction(
                                        receiver_score,
                                        donor_score,
                                        patched_score,
                                    )
                                ),
                                "patched_expected_count": float(
                                    patched_metrics["expected_count"]
                                ),
                                "expected_count_delta": float(
                                    patched_metrics["expected_count"]
                                    - receiver_metrics["expected_count"]
                                ),
                            }
                        )
                print(
                    "[v4 patch] "
                    f"{variant} seed={seed} "
                    f"N={receiver_count}->{donor_count}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No V4 hidden-state patch rows were produced")
    return pd.DataFrame(rows)


def summarize_hidden_patching(detail: pd.DataFrame) -> pd.DataFrame:
    successful = detail[detail["status"] == "ok"].copy()
    if successful.empty:
        raise ValueError("No successful V4 hidden-state patches")
    groups = [
        "model_label",
        "design_variant",
        "site",
        "layer",
        "receiver_count",
        "donor_count",
    ]
    return successful.groupby(groups, as_index=False).agg(
        examples=("seed", "count"),
        seeds=("seed", "nunique"),
        mean_pair_logit_score_delta=("pair_logit_score_delta", "mean"),
        sd_pair_logit_score_delta=("pair_logit_score_delta", "std"),
        mean_pair_logit_recovery_fraction=(
            "pair_logit_recovery_fraction",
            "mean",
        ),
        mean_expected_count_delta=("expected_count_delta", "mean"),
    )
