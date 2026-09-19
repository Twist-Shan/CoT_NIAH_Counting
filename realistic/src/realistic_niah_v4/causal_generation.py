from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attention import Head, matched_random_heads
from .behavior import label_generated_completion
from .causal_v2 import (
    CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
    normalized_transport_metrics,
)
from .modeling import (
    DecoderAdapter,
    capture_post_block_states,
    generate_with_head_ablation,
    generate_with_residual_interventions,
)
from .prompts import PromptEncoding

POOLINGS = ("span_end", "span_mean")
ABLATION_SCOPES = ("answer_query", "global")
RESIDUAL_PATCH_SITES = (
    "answer_query",
    "toggled_needle_end",
    "toggled_needle_span",
)
NEEDLE_PATCH_PROTOCOLS = ("single_layer", "cumulative_from_layer")


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def load_generation_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "stimulus_id",
        "model_label",
        "design_variant",
        "seed",
        "gold_count",
        "outcome_group",
        "is_correct",
        "format_valid",
        "parsed_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Generation labels are missing columns: {missing}")
    if frame["stimulus_id"].duplicated().any():
        raise ValueError("Generation labels contain duplicate stimulus IDs")
    return {str(row["stimulus_id"]): row.to_dict() for _, row in frame.iterrows()}


def _completion_from_generation_label(label: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the greedy completion payload already stored for a baseline.

    A self-patch writes the receiver's own post-block residual back to the
    same semantic position.  Once a small set of real hook executions has
    verified that identity invariant, the remaining dense self controls can
    reuse this payload instead of repeating an identical 10k-token prefill.
    """

    completion_text = str(label.get("completion_text", ""))
    raw_text = label.get("completion_text_raw", completion_text)
    if raw_text is None or (isinstance(raw_text, float) and math.isnan(raw_text)):
        raw_text = completion_text
    token_ids = label.get("generated_token_ids")
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    if token_ids is None or (isinstance(token_ids, float) and math.isnan(token_ids)):
        raise ValueError("Generation label lacks baseline generated_token_ids")
    if not isinstance(token_ids, Sequence) or isinstance(
        token_ids, (str, bytes, bytearray)
    ):
        raise TypeError("Baseline generated_token_ids must be a sequence")
    return {
        "completion_text": completion_text,
        "completion_text_raw": str(raw_text),
        "generated_token_ids": [int(value) for value in token_ids],
        "generation_truncated": _as_bool(label.get("generation_truncated", False)),
    }


def _causal_v2_completion_cache_key(
    *,
    condition: str,
    receiver: PromptEncoding,
    state_source: PromptEncoding,
    site: str,
    protocol: str,
    start_layer: int,
    slot_indices: Sequence[int],
) -> tuple[Any, ...]:
    """Identify interventions that are computationally exactly equivalent.

    In particular, an answer-query same-count control does not depend on the
    nominal donor or k: it patches the same receiver query with the same
    other-seed, same-count state at the same layer(s).  Treating every donor
    label as a new generation is pseudoreplication and wastes a full prefill.
    Pair-specific normalized metrics are still computed after cache reuse.
    """

    base = (
        str(condition),
        str(receiver.stimulus_id),
        str(state_source.stimulus_id),
        str(site),
        str(protocol),
        int(start_layer),
    )
    if str(condition) == "same_count_seed" and str(site) == "answer_query":
        return ("answer_same_count_equivalent", *base)
    return ("exact_intervention", *base, tuple(int(value) for value in slot_indices))


def load_broad_rankings(
    rankings_dir: str | Path,
    *,
    variants: Sequence[str],
    poolings: Sequence[str] = POOLINGS,
) -> dict[tuple[str, str], list[Head]]:
    root = Path(rankings_dir)
    result: dict[tuple[str, str], list[Head]] = {}
    for variant in variants:
        for pooling in poolings:
            if pooling not in POOLINGS:
                raise ValueError(f"Unknown broad-head pooling: {pooling}")
            path = root / f"{str(variant).replace('.', '_')}_{pooling}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("design_variant")) != str(variant):
                raise ValueError(f"Ranking variant mismatch: {path}")
            if str(payload.get("pooling")) != str(pooling):
                raise ValueError(f"Ranking pooling mismatch: {path}")
            if str(payload.get("selection_split")) != "discovery":
                raise ValueError(f"Causal head selection must use discovery: {path}")
            heads = payload.get("top_heads")
            if not isinstance(heads, list) or not heads:
                raise ValueError(f"Ranking contains no eligible broad heads: {path}")
            result[(str(variant), str(pooling))] = [
                (int(item["layer"]), int(item["head"])) for item in heads
            ]
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _validate_baseline_label(
    encoding: PromptEncoding,
    label: Mapping[str, Any],
) -> None:
    observed = {
        "model_label": str(label["model_label"]),
        "design_variant": str(label["design_variant"]),
        "seed": int(label["seed"]),
        "gold_count": int(label["gold_count"]),
    }
    expected = {
        "model_label": encoding.model_label,
        "design_variant": encoding.design_variant,
        "seed": int(encoding.seed),
        "gold_count": int(encoding.count),
    }
    if observed != expected:
        raise ValueError(
            f"Behavior/encoding mismatch for {encoding.stimulus_id}: "
            f"observed={observed} expected={expected}"
        )


def _base_metadata(encoding: PromptEncoding) -> dict[str, Any]:
    return {
        "stimulus_id": encoding.stimulus_id,
        "model_label": encoding.model_label,
        "design_variant": encoding.design_variant,
        "seed": int(encoding.seed),
        "split": encoding.split,
        "gold_count": int(encoding.count),
        "sequence_length": int(encoding.sequence_length),
        "query_position": int(encoding.query_position),
    }


def _baseline_metadata(label: Mapping[str, Any]) -> dict[str, Any]:
    predicted = _optional_int(label.get("parsed_count"))
    count_error = _optional_int(label.get("count_error"))
    return {
        "baseline_outcome": str(label["outcome_group"]),
        "baseline_is_correct": _as_bool(label["is_correct"]),
        "baseline_format_valid": _as_bool(label["format_valid"]),
        "baseline_predicted_count": predicted,
        "baseline_count_error": count_error,
        "baseline_absolute_error": (
            abs(int(count_error)) if count_error is not None else math.nan
        ),
        "baseline_completion_text": str(label.get("completion_text", "")),
    }


def intervention_outcome(
    completion: Mapping[str, Any],
    encoding: PromptEncoding,
    baseline_label: Mapping[str, Any],
    *,
    valid_counts: Sequence[int] = tuple(range(1, 11)),
) -> dict[str, Any]:
    labeled = label_generated_completion(
        str(completion["completion_text"]),
        gold_count=int(encoding.count),
        valid_counts=tuple(int(value) for value in valid_counts),
    )
    baseline_pred = _optional_int(baseline_label.get("parsed_count"))
    patched_pred = _optional_int(labeled.get("parsed_count"))
    baseline_error = _optional_int(baseline_label.get("count_error"))
    patched_error = _optional_int(labeled.get("count_error"))
    both_valid = baseline_pred is not None and patched_pred is not None
    return {
        "patched_outcome": str(labeled["outcome_group"]),
        "patched_is_correct": bool(labeled["is_correct"]),
        "patched_format_valid": bool(labeled["format_valid"]),
        "patched_predicted_count": patched_pred,
        "patched_count_error": patched_error,
        "patched_error_direction": str(labeled["error_direction"]),
        "patched_completion_text": str(completion["completion_text"]),
        "patched_completion_text_raw": str(completion["completion_text_raw"]),
        "patched_generated_token_ids": json.dumps(
            [int(value) for value in completion["generated_token_ids"]]
        ),
        "patched_generation_truncated": bool(completion["generation_truncated"]),
        "prediction_changed": (
            bool(patched_pred != baseline_pred) if both_valid else math.nan
        ),
        "generated_count_shift": (
            int(patched_pred) - int(baseline_pred) if both_valid else math.nan
        ),
        "accuracy_delta": float(bool(labeled["is_correct"]))
        - float(_as_bool(baseline_label["is_correct"])),
        "absolute_error_delta": (
            abs(int(patched_error)) - abs(int(baseline_error))
            if patched_error is not None and baseline_error is not None
            else math.nan
        ),
        "intervention_hook_applications": json.dumps(
            completion.get("intervention_hook_applications", {}), sort_keys=True
        ),
    }


def _control_sets(
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    adapter: DecoderAdapter,
    *,
    top_ns: Sequence[int],
    random_replicates: int,
) -> dict[tuple[str, str, int, int], list[Head]]:
    controls: dict[tuple[str, str, int, int], list[Head]] = {}
    for (variant, pooling), ranking in rankings.items():
        for top_n in top_ns:
            selected = list(ranking[: int(top_n)])
            if len(selected) != int(top_n):
                raise ValueError(
                    f"{variant}/{pooling} has fewer than top_n={top_n} eligible heads"
                )
            for replicate in range(int(random_replicates)):
                controls[(variant, pooling, int(top_n), replicate)] = (
                    matched_random_heads(
                        selected,
                        adapter,
                        seed=_stable_seed(
                            f"{variant}:{pooling}:top{top_n}:random{replicate}"
                        ),
                    )
                )
    return controls


def run_generation_head_ablation(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    poolings: Sequence[str] = POOLINGS,
    top_ns: Sequence[int] = (1, 2, 4, 8),
    random_replicates: int = 3,
    scopes: Sequence[str] = ABLATION_SCOPES,
    max_new_tokens: int = 16,
) -> pd.DataFrame:
    """Run global and answer-query-local head ablation with real generation."""

    normalized_scopes = tuple(str(scope) for scope in scopes)
    if not normalized_scopes or any(
        scope not in ABLATION_SCOPES for scope in normalized_scopes
    ):
        raise ValueError(f"Invalid ablation scopes: {normalized_scopes}")
    normalized_poolings = tuple(str(pooling) for pooling in poolings)
    if (
        not normalized_poolings
        or len(set(normalized_poolings)) != len(normalized_poolings)
        or any(pooling not in POOLINGS for pooling in normalized_poolings)
    ):
        raise ValueError(f"Invalid ablation poolings: {normalized_poolings}")
    normalized_top_ns = tuple(sorted({int(value) for value in top_ns}))
    if not normalized_top_ns or normalized_top_ns[0] <= 0:
        raise ValueError("Ablation top_ns must be positive")
    controls = _control_sets(
        rankings,
        adapter,
        top_ns=normalized_top_ns,
        random_replicates=int(random_replicates),
    )
    rows: list[dict[str, Any]] = []
    for example_index, encoding in enumerate(encodings):
        if encoding.split != "confirmation":
            raise ValueError("Causal evaluation must use confirmation encodings")
        label = baseline_labels.get(encoding.stimulus_id)
        if label is None:
            raise KeyError(f"Missing baseline generation label: {encoding.stimulus_id}")
        _validate_baseline_label(encoding, label)
        cache: dict[tuple[str, tuple[Head, ...]], dict[str, Any]] = {}
        for pooling in normalized_poolings:
            ranking = list(rankings[(encoding.design_variant, pooling)])
            for top_n in normalized_top_ns:
                selected = tuple(ranking[:top_n])
                conditions: list[tuple[str, int, tuple[Head, ...]]] = [
                    ("ranked", -1, selected)
                ]
                conditions.extend(
                    (
                        "layer_matched_random",
                        replicate,
                        tuple(
                            controls[
                                (
                                    encoding.design_variant,
                                    pooling,
                                    top_n,
                                    replicate,
                                )
                            ]
                        ),
                    )
                    for replicate in range(int(random_replicates))
                )
                for scope in normalized_scopes:
                    for condition, replicate, heads in conditions:
                        cache_key = (scope, heads)
                        if cache_key not in cache:
                            cache[cache_key] = generate_with_head_ablation(
                                model,
                                tokenizer,
                                adapter,
                                encoding,
                                heads,
                                scope=scope,
                                max_new_tokens=max_new_tokens,
                            )
                        rows.append(
                            {
                                **_base_metadata(encoding),
                                **_baseline_metadata(label),
                                "example_index": int(example_index),
                                "pooling": pooling,
                                "scope": scope,
                                "condition": condition,
                                "top_n": int(top_n),
                                "random_replicate": int(replicate),
                                "heads": ",".join(
                                    f"L{layer}H{head}" for layer, head in heads
                                ),
                                **intervention_outcome(
                                    cache[cache_key], encoding, label
                                ),
                            }
                        )
        print(
            "[v4 causal ablation] "
            f"{example_index + 1}/{len(encodings)} "
            f"{encoding.design_variant} seed={encoding.seed} N={encoding.count}",
            flush=True,
        )
    if not rows:
        raise ValueError("No generation head-ablation rows were produced")
    return pd.DataFrame(rows)


def summarize_generation_head_ablation(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "model_label",
        "design_variant",
        "pooling",
        "scope",
        "condition",
        "top_n",
        "baseline_outcome",
    ]
    return (
        detail.groupby(groups, as_index=False, dropna=False)
        .agg(
            examples=("stimulus_id", "size"),
            seeds=("seed", "nunique"),
            baseline_accuracy=("baseline_is_correct", "mean"),
            patched_accuracy=("patched_is_correct", "mean"),
            patched_valid_rate=("patched_format_valid", "mean"),
            mean_accuracy_delta=("accuracy_delta", "mean"),
            mean_generated_count_shift=("generated_count_shift", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
        )
        .sort_values(groups)
    )


def compare_ranked_ablation_to_random(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    identifiers = [
        "model_label",
        "design_variant",
        "pooling",
        "scope",
        "stimulus_id",
        "seed",
        "gold_count",
        "top_n",
        "baseline_outcome",
    ]
    metrics = ["accuracy_delta", "absolute_error_delta", "prediction_changed"]
    ranked = detail[detail["condition"] == "ranked"].copy()
    random = detail[detail["condition"] == "layer_matched_random"].copy()
    random_mean = (
        random.groupby(identifiers, as_index=False)[metrics]
        .mean()
        .rename(columns={metric: f"{metric}_random_mean" for metric in metrics})
    )
    paired = ranked.merge(random_mean, on=identifiers, how="inner")
    rows: list[dict[str, Any]] = []
    group_columns = [
        "model_label",
        "design_variant",
        "pooling",
        "scope",
        "top_n",
        "baseline_outcome",
    ]
    for keys, frame in paired.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        for metric in metrics:
            differences = frame[metric] - frame[f"{metric}_random_mean"]
            seed_means = (
                frame.assign(difference=differences)
                .groupby("seed")["difference"]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(seed_means) == 0:
                low = high = mean = math.nan
            else:
                rng = np.random.default_rng(
                    _stable_seed(
                        ":".join(str(metadata[name]) for name in group_columns)
                        + f":{metric}"
                    )
                )
                indices = rng.integers(
                    0,
                    len(seed_means),
                    size=(int(bootstrap_repetitions), len(seed_means)),
                )
                distribution = seed_means[indices].mean(axis=1)
                low, high = np.quantile(distribution, [0.025, 0.975])
                mean = float(seed_means.mean())
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "ranked_minus_random_mean": mean,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "confirmation_seeds": len(seed_means),
                    "bootstrap_repetitions": int(bootstrap_repetitions),
                }
            )
    return pd.DataFrame(rows)


def _causal_v2_control_sets(
    rankings: Mapping[str, Sequence[Head]],
    adapter: DecoderAdapter,
    *,
    top_ns: Sequence[int],
    random_replicates: int,
) -> dict[tuple[str, int, int], tuple[Head, ...]]:
    """Freeze unbiased per-layer random banks for each ranked top-k size.

    Each random bank is drawn without replacement from all heads in exactly
    the same layers and with exactly the same per-layer counts as the ranked
    bank.  Ranked/random overlap is allowed and reported: forbidding the full
    discovery candidate union can make a top-32 layer match mathematically
    impossible when phenotype heads cluster within a layer.
    """
    controls: dict[tuple[str, int, int], tuple[Head, ...]] = {}
    for bank, ranking in sorted(rankings.items()):
        normalized = tuple((int(layer), int(head)) for layer, head in ranking)
        for top_n in top_ns:
            selected = normalized[: int(top_n)]
            if len(selected) != int(top_n):
                raise ValueError(f"{bank} has fewer than top_n={top_n} heads")
            for replicate in range(int(random_replicates)):
                controls[(str(bank), int(top_n), int(replicate))] = tuple(
                    matched_random_heads(
                        selected,
                        adapter,
                        seed=_stable_seed(
                            f"causal-v2:{bank}:top{top_n}:random:{replicate}"
                        ),
                        exclude_selected=False,
                    )
                )
    return controls


def run_generation_head_ablation_v2(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Head]],
    top_ns: Sequence[int] = tuple(range(1, 33)),
    random_replicates: int = 3,
    require_full_sweep: bool = True,
    require_correct_baseline: bool = False,
    valid_counts: Sequence[int] = tuple(range(0, 11)),
    max_new_tokens: int = 16,
) -> pd.DataFrame:
    """Run the V4.4 answer-query-only broad/first-locator top-k sweep."""

    expected_banks = {"broad_aggregation", "first_locator"}
    observed_banks = set(rankings)
    if require_full_sweep and observed_banks != expected_banks:
        raise ValueError(
            "Causal-v2 discovery ablation requires broad_aggregation and "
            "first_locator banks"
        )
    if not require_full_sweep and (
        not observed_banks or not observed_banks.issubset(expected_banks)
    ):
        raise ValueError(
            "Causal-v2 confirmation ablation requires a nonempty subset of "
            "the registered broad_aggregation/first_locator banks"
        )
    normalized_top_ns = tuple(sorted({int(value) for value in top_ns}))
    if require_full_sweep and normalized_top_ns != tuple(range(1, 33)):
        raise ValueError("Causal-v2 ablation must sweep every top-k from 1 to 32")
    if (
        not normalized_top_ns
        or min(normalized_top_ns) < 1
        or max(normalized_top_ns) > 32
    ):
        raise ValueError("Causal-v2 ablation top-k values must lie in [1, 32]")
    if int(random_replicates) < 1:
        raise ValueError("Causal-v2 ablation requires matched random controls")
    controls = _causal_v2_control_sets(
        rankings,
        adapter,
        top_ns=normalized_top_ns,
        random_replicates=int(random_replicates),
    )
    rows: list[dict[str, Any]] = []
    for example_index, encoding in enumerate(encodings):
        if encoding.design_variant != "v4.4":
            raise ValueError("Causal-v2 ablation is restricted to v4.4")
        label = baseline_labels.get(encoding.stimulus_id)
        if label is None:
            raise KeyError(f"Missing baseline generation label: {encoding.stimulus_id}")
        _validate_baseline_label(encoding, label)
        if require_correct_baseline and (
            str(label.get("outcome_group")) != "correct"
            or not _as_bool(label.get("format_valid"))
        ):
            raise ValueError(
                "Correct-only head ablation received an ineligible clean "
                f"baseline: stimulus_id={encoding.stimulus_id}, "
                f"outcome={label.get('outcome_group')!r}, "
                f"format_valid={label.get('format_valid')!r}"
            )
        cache: dict[tuple[Head, ...], dict[str, Any]] = {}
        for bank, ranking in sorted(rankings.items()):
            normalized_ranking = tuple(
                (int(layer), int(head)) for layer, head in ranking
            )
            for top_n in normalized_top_ns:
                selected = normalized_ranking[:top_n]
                conditions: list[tuple[str, int, tuple[Head, ...]]] = [
                    ("ranked", -1, selected)
                ]
                conditions.extend(
                    (
                        "layer_matched_random",
                        replicate,
                        controls[(bank, top_n, replicate)],
                    )
                    for replicate in range(int(random_replicates))
                )
                for condition, replicate, heads in conditions:
                    if heads not in cache:
                        cache[heads] = generate_with_head_ablation(
                            model,
                            tokenizer,
                            adapter,
                            encoding,
                            heads,
                            scope="answer_query",
                            max_new_tokens=max_new_tokens,
                        )
                    rows.append(
                        {
                            **_base_metadata(encoding),
                            **_baseline_metadata(label),
                            "example_index": int(example_index),
                            "head_bank": str(bank),
                            "scope": "answer_query",
                            "condition": condition,
                            "top_n": int(top_n),
                            "random_replicate": int(replicate),
                            "heads": ",".join(
                                f"L{layer}H{head}" for layer, head in heads
                            ),
                            "ranked_random_head_overlap": int(
                                len(set(heads) & set(selected))
                                if condition == "layer_matched_random"
                                else len(selected)
                            ),
                            "random_sampling_population": (
                                "all_heads_in_matched_layers_without_replacement"
                                if condition == "layer_matched_random"
                                else "discovery_ranked_prefix"
                            ),
                            "analysis_population": (
                                "clean_correct_only"
                                if require_correct_baseline
                                else "all_examples"
                            ),
                            **intervention_outcome(
                                cache[heads],
                                encoding,
                                label,
                                valid_counts=valid_counts,
                            ),
                        }
                    )
        print(
            "[v4 causal-v2 ablation] "
            f"{example_index + 1}/{len(encodings)} seed={encoding.seed} "
            f"N={encoding.count}",
            flush=True,
        )
    if not rows:
        raise ValueError("No causal-v2 head-ablation rows were produced")
    return pd.DataFrame(rows)


def summarize_generation_head_ablation_v2(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "model_label",
        "design_variant",
        "head_bank",
        "condition",
        "top_n",
        "baseline_outcome",
    ]
    return (
        detail.groupby(groups, as_index=False, dropna=False)
        .agg(
            examples=("stimulus_id", "size"),
            seeds=("seed", "nunique"),
            baseline_accuracy=("baseline_is_correct", "mean"),
            patched_accuracy=("patched_is_correct", "mean"),
            patched_valid_rate=("patched_format_valid", "mean"),
            mean_accuracy_delta=("accuracy_delta", "mean"),
            mean_generated_count_shift=("generated_count_shift", "mean"),
            mean_absolute_error_delta=("absolute_error_delta", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
        )
        .sort_values(groups)
        .reset_index(drop=True)
    )


def compare_head_ablation_v2_to_random(
    detail: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    identifiers = [
        "model_label",
        "stimulus_id",
        "seed",
        "gold_count",
        "head_bank",
        "top_n",
        "baseline_outcome",
    ]
    metrics = ["accuracy_delta", "absolute_error_delta", "prediction_changed"]
    ranked = detail[detail["condition"].astype(str).eq("ranked")].copy()
    random = detail[detail["condition"].astype(str).eq("layer_matched_random")].copy()
    random_mean = (
        random.groupby(identifiers, as_index=False, dropna=False)[metrics]
        .mean()
        .rename(columns={metric: f"{metric}_random_mean" for metric in metrics})
    )
    paired = ranked.merge(
        random_mean, on=identifiers, how="inner", validate="one_to_one"
    )
    if len(paired) != len(ranked):
        raise ValueError("Causal-v2 ablation has incomplete matched random controls")
    group_columns = [
        "model_label",
        "head_bank",
        "top_n",
        "baseline_outcome",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        for metric in metrics:
            seed_means = (
                frame.assign(difference=frame[metric] - frame[f"{metric}_random_mean"])
                .groupby("seed")["difference"]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(seed_means) == 0:
                mean = low = high = math.nan
            else:
                rng = np.random.default_rng(
                    _stable_seed(
                        "causal-v2-ablation:"
                        + ":".join(str(metadata[name]) for name in group_columns)
                        + f":{metric}"
                    )
                )
                indices = rng.integers(
                    0,
                    len(seed_means),
                    size=(int(bootstrap_repetitions), len(seed_means)),
                )
                distribution = seed_means[indices].mean(axis=1)
                low, high = np.quantile(distribution, [0.025, 0.975])
                mean = float(seed_means.mean())
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "ranked_minus_random_mean": mean,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "screen_seeds": int(len(seed_means)),
                    "bootstrap_repetitions": int(bootstrap_repetitions),
                }
            )
    return pd.DataFrame(rows)


def _encoding_map(
    encodings: Sequence[PromptEncoding],
) -> dict[tuple[str, int, int], PromptEncoding]:
    result = {
        (item.design_variant, int(item.seed), int(item.count)): item
        for item in encodings
    }
    if len(result) != len(encodings):
        raise ValueError("Causal encoding grid is not unique by variant/seed/count")
    return result


def _validate_pair(receiver: PromptEncoding, donor: PromptEncoding) -> None:
    if receiver.design_variant != donor.design_variant:
        raise ValueError("Donor and receiver must share a design variant")
    if receiver.seed != donor.seed:
        raise ValueError("Donor and receiver must share a seed")
    if receiver.model_label != donor.model_label:
        raise ValueError("Donor and receiver must share a model")
    if receiver.count == donor.count:
        raise ValueError("Donor and receiver counts must differ")


def transport_fields(
    outcome: Mapping[str, Any],
    *,
    receiver_label: Mapping[str, Any],
    donor_label: Mapping[str, Any],
    receiver_count: int,
    donor_count: int,
) -> dict[str, Any]:
    patched = _optional_int(outcome.get("patched_predicted_count"))
    receiver_prediction = _optional_int(receiver_label.get("parsed_count"))
    donor_prediction = _optional_int(donor_label.get("parsed_count"))
    gold_offset = int(donor_count) - int(receiver_count)
    predicted_offset = (
        int(donor_prediction) - int(receiver_prediction)
        if donor_prediction is not None and receiver_prediction is not None
        else math.nan
    )
    generated_shift = outcome.get("generated_count_shift", math.nan)
    normalized = normalized_transport_metrics(
        baseline_prediction=receiver_prediction,
        intervened_prediction=patched,
        receiver_count=int(receiver_count),
        target_count=int(donor_count),
    )
    return {
        "gold_count_offset": int(gold_offset),
        "baseline_prediction_offset": predicted_offset,
        "gold_transport_fraction": (
            float(generated_shift) / float(gold_offset)
            if np.isfinite(generated_shift) and gold_offset != 0
            else math.nan
        ),
        "baseline_prediction_transport_fraction": (
            float(generated_shift) / float(predicted_offset)
            if np.isfinite(generated_shift)
            and np.isfinite(predicted_offset)
            and abs(float(predicted_offset)) > 1e-12
            else math.nan
        ),
        "follows_donor_gold": (
            bool(patched == int(donor_count)) if patched is not None else math.nan
        ),
        "follows_receiver_gold": (
            bool(patched == int(receiver_count)) if patched is not None else math.nan
        ),
        "follows_donor_prediction": (
            bool(patched == donor_prediction)
            if patched is not None and donor_prediction is not None
            else math.nan
        ),
        "moved_toward_donor_gold": (
            abs(int(patched) - int(donor_count))
            < abs(int(receiver_prediction) - int(donor_count))
            if patched is not None and receiver_prediction is not None
            else math.nan
        ),
        **normalized,
    }


def _ols_slope(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    finite = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2 or finite[x].nunique() < 2:
        return math.nan, math.nan, math.nan
    xv = finite[x].to_numpy(dtype=float)
    yv = finite[y].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(xv)), xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    fitted = design @ beta
    denominator = float(np.sum((yv - yv.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum((yv - fitted) ** 2)) / denominator
        if denominator > 1e-12
        else math.nan
    )
    return float(beta[1]), float(beta[0]), float(r2)


def _residual_site_states(
    *,
    site: str,
    donor_states: Mapping[int, Any],
    receiver_states: Mapping[int, Any],
    donor_span_offset: int,
    donor_span_length: int,
    receiver_span_offset: int,
    receiver_span_length: int,
    receiver_positions: Sequence[int],
    layer: int,
) -> tuple[tuple[int, ...], Any, bool, str]:
    donor = donor_states[int(layer)]
    receiver = receiver_states[int(layer)]
    if site == "answer_query":
        return (int(receiver_positions[0]),), donor[0:1], True, ""
    donor_span = donor[
        int(donor_span_offset) : int(donor_span_offset + donor_span_length)
    ]
    receiver_span = receiver[
        int(receiver_span_offset) : int(receiver_span_offset + receiver_span_length)
    ]
    span_positions = tuple(int(value) for value in receiver_positions[1:])
    if site == "toggled_needle_end":
        return (span_positions[-1],), donor_span[-1:], True, ""
    if site == "toggled_needle_span":
        if int(donor_span.shape[0]) != int(donor_span_length):
            raise RuntimeError("Captured donor span length does not match metadata")
        if int(receiver_span.shape[0]) != int(receiver_span_length):
            raise RuntimeError("Captured receiver span length does not match metadata")
        if donor_span_length != receiver_span_length:
            return span_positions, donor_span, False, "model_token_length_mismatch"
        return span_positions, donor_span, True, ""
    raise ValueError(f"Unknown residual patch site: {site}")


def run_generation_residual_patching(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    count_pairs: Sequence[tuple[int, int]],
    start_layers: Sequence[int],
    sites: Sequence[str] = RESIDUAL_PATCH_SITES,
    needle_protocols: Sequence[str] = NEEDLE_PATCH_PROTOCOLS,
    max_new_tokens: int = 16,
) -> pd.DataFrame:
    """Run V10 final-query patches and realistic matched-slot patches.

    Final-query patches replace one post-block residual at one layer. Needle
    interventions follow the older realistic protocol and clamp the matched
    donor slot from the selected start layer through the final block. Both
    lower<-higher insertion and higher<-lower removal are supported.
    """

    requested_sites = tuple(str(site) for site in sites)
    if not requested_sites or any(
        site not in RESIDUAL_PATCH_SITES for site in requested_sites
    ):
        raise ValueError(f"Invalid residual patch sites: {requested_sites}")
    requested_protocols = tuple(str(protocol) for protocol in needle_protocols)
    if not requested_protocols or any(
        protocol not in NEEDLE_PATCH_PROTOCOLS for protocol in requested_protocols
    ):
        raise ValueError(f"Invalid needle patch protocols: {requested_protocols}")
    starts = tuple(sorted({int(layer) for layer in start_layers}))
    if not starts or starts[0] < 0 or starts[-1] >= adapter.num_layers:
        raise ValueError("Residual patch start layers are invalid")
    by_key = _encoding_map(encodings)
    rows: list[dict[str, Any]] = []
    variants = sorted({item.design_variant for item in encodings})
    seeds = sorted({int(item.seed) for item in encodings})
    for variant in variants:
        for seed in seeds:
            state_cache: dict[tuple[str, int], tuple[list[int], dict[int, Any]]] = {}
            for receiver_count, donor_count in count_pairs:
                receiver = by_key.get((variant, seed, int(receiver_count)))
                donor = by_key.get((variant, seed, int(donor_count)))
                if receiver is None or donor is None:
                    raise KeyError(
                        f"Missing residual pair {variant} seed={seed} "
                        f"N={receiver_count}<-{donor_count}"
                    )
                _validate_pair(receiver, donor)
                receiver_label = baseline_labels[receiver.stimulus_id]
                donor_label = baseline_labels[donor.stimulus_id]
                _validate_baseline_label(receiver, receiver_label)
                _validate_baseline_label(donor, donor_label)
                toggled_slot = max(int(receiver_count), int(donor_count))
                if abs(int(donor_count) - int(receiver_count)) != 1:
                    needle_sites_allowed = False
                else:
                    needle_sites_allowed = True
                receiver_span = receiver.slot_spans[toggled_slot - 1]
                donor_span = donor.slot_spans[toggled_slot - 1]
                receiver_positions = [int(receiver.query_position)] + list(
                    range(int(receiver_span.start), int(receiver_span.end))
                )
                donor_positions = [int(donor.query_position)] + list(
                    range(int(donor_span.start), int(donor_span.end))
                )
                donor_cache_key = (donor.stimulus_id, int(toggled_slot))
                if donor_cache_key not in state_cache:
                    _, captured = capture_post_block_states(
                        model, adapter, donor, donor_positions
                    )
                    state_cache[donor_cache_key] = (donor_positions, captured)
                receiver_cache_key = (receiver.stimulus_id, int(toggled_slot))
                if receiver_cache_key not in state_cache:
                    _, captured = capture_post_block_states(
                        model, adapter, receiver, receiver_positions
                    )
                    state_cache[receiver_cache_key] = (
                        receiver_positions,
                        captured,
                    )
                cached_donor_positions, donor_states = state_cache[donor_cache_key]
                cached_receiver_positions, receiver_states = state_cache[
                    receiver_cache_key
                ]
                if cached_donor_positions != donor_positions:
                    raise RuntimeError("Cached donor semantic positions changed")
                if cached_receiver_positions != receiver_positions:
                    raise RuntimeError("Cached receiver semantic positions changed")
                donor_span_offset = 1
                receiver_span_offset = 1
                donor_span_length = int(donor_span.model_token_length)
                receiver_span_length = int(receiver_span.model_token_length)
                if donor_span_length != int(donor_span.end - donor_span.start):
                    raise RuntimeError(
                        "Donor span token-length metadata is inconsistent"
                    )
                if receiver_span_length != int(receiver_span.end - receiver_span.start):
                    raise RuntimeError(
                        "Receiver span token-length metadata is inconsistent"
                    )
                direction = (
                    "needle_insertion"
                    if int(donor_count) > int(receiver_count)
                    else "needle_removal"
                )
                for site in requested_sites:
                    if site != "answer_query" and not needle_sites_allowed:
                        continue
                    for start_layer in starts:
                        protocols = (
                            ("single_layer",)
                            if site == "answer_query"
                            else requested_protocols
                        )
                        for protocol in protocols:
                            intervention_layers = (
                                (int(start_layer),)
                                if protocol == "single_layer"
                                else tuple(range(int(start_layer), adapter.num_layers))
                            )
                            interventions: dict[int, tuple[Sequence[int], Any]] = {}
                            executable = True
                            skip_reason = ""
                            for layer in intervention_layers:
                                positions, states, ok, reason = _residual_site_states(
                                    site=site,
                                    donor_states=donor_states,
                                    receiver_states=receiver_states,
                                    donor_span_offset=donor_span_offset,
                                    donor_span_length=donor_span_length,
                                    receiver_span_offset=receiver_span_offset,
                                    receiver_span_length=receiver_span_length,
                                    receiver_positions=receiver_positions,
                                    layer=layer,
                                )
                                if not ok:
                                    executable = False
                                    skip_reason = reason
                                    break
                                interventions[int(layer)] = (positions, states)
                            metadata = {
                                **_base_metadata(receiver),
                                **_baseline_metadata(receiver_label),
                                "receiver_stimulus_id": receiver.stimulus_id,
                                "donor_stimulus_id": donor.stimulus_id,
                                "receiver_count": int(receiver_count),
                                "donor_count": int(donor_count),
                                "donor_baseline_outcome": str(
                                    donor_label["outcome_group"]
                                ),
                                "donor_baseline_predicted_count": _optional_int(
                                    donor_label.get("parsed_count")
                                ),
                                "direction": direction,
                                "site": site,
                                "patch_protocol": protocol,
                                "start_layer": int(start_layer),
                                "patched_layer_count": len(intervention_layers),
                                "toggled_slot_index": int(toggled_slot),
                                "donor_span_active": bool(donor_span.active),
                                "receiver_span_active": bool(receiver_span.active),
                                "donor_model_token_length": donor_span_length,
                                "receiver_model_token_length": receiver_span_length,
                                "status": "ok" if executable else "skipped",
                                "skip_reason": skip_reason,
                            }
                            if not executable:
                                rows.append(metadata)
                                continue
                            completion = generate_with_residual_interventions(
                                model,
                                tokenizer,
                                adapter,
                                receiver,
                                interventions,
                                max_new_tokens=max_new_tokens,
                            )
                            outcome = intervention_outcome(
                                completion, receiver, receiver_label
                            )
                            rows.append(
                                {
                                    **metadata,
                                    **outcome,
                                    **transport_fields(
                                        outcome,
                                        receiver_label=receiver_label,
                                        donor_label=donor_label,
                                        receiver_count=int(receiver_count),
                                        donor_count=int(donor_count),
                                    ),
                                }
                            )
                print(
                    "[v4 causal residual patch] "
                    f"{variant} seed={seed} N={receiver_count}<-{donor_count}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No generation residual-patching rows were produced")
    return pd.DataFrame(rows)


def summarize_generation_residual_patching(detail: pd.DataFrame) -> pd.DataFrame:
    successful = detail[detail["status"] == "ok"].copy()
    if successful.empty:
        raise ValueError("No successful generation residual patches")
    groups = [
        "model_label",
        "design_variant",
        "site",
        "patch_protocol",
        "start_layer",
        "direction",
        "baseline_outcome",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in successful.groupby(groups, sort=True, dropna=False):
        slope, intercept, r2 = _ols_slope(
            frame, "gold_count_offset", "generated_count_shift"
        )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "examples": len(frame),
                "seeds": int(frame["seed"].nunique()),
                "patched_valid_rate": float(frame["patched_format_valid"].mean()),
                "patched_accuracy": float(frame["patched_is_correct"].mean()),
                "prediction_changed_rate": float(frame["prediction_changed"].mean()),
                "mean_generated_count_shift": float(
                    frame["generated_count_shift"].mean()
                ),
                "follows_donor_gold_rate": float(frame["follows_donor_gold"].mean()),
                "moved_toward_donor_gold_rate": float(
                    frame["moved_toward_donor_gold"].mean()
                ),
                "gold_transport_slope": slope,
                "gold_transport_intercept": intercept,
                "gold_transport_r2": r2,
            }
        )
    return pd.DataFrame(rows)


def _causal_v2_semantic_positions(
    encoding: PromptEncoding,
    slot_indices: Sequence[int],
) -> tuple[tuple[int, ...], dict[int, tuple[int, int]]]:
    """Return query + selected slot positions and slices into captured states."""

    positions: list[int] = [int(encoding.query_position)]
    slices: dict[int, tuple[int, int]] = {}
    by_slot = {int(span.slot_index): span for span in encoding.slot_spans}
    for raw_slot in slot_indices:
        slot = int(raw_slot)
        if slot not in by_slot:
            raise KeyError(f"Encoding {encoding.stimulus_id} has no slot {slot}")
        span = by_slot[slot]
        start = len(positions)
        positions.extend(range(int(span.start), int(span.end)))
        end = len(positions)
        if end <= start:
            raise RuntimeError(f"Slot {slot} has an empty model-token span")
        slices[slot] = (start, end)
    if len(set(positions)) != len(positions):
        raise RuntimeError("Causal-v2 semantic patch positions overlap")
    return tuple(positions), slices


def causal_v2_monotonic_span_source_indices(
    receiver_length: int,
    source_length: int,
) -> tuple[int, ...]:
    """Map every receiver span token to one complete source-token state.

    For non-degenerate spans, receiver index ``j`` maps to the nearest source
    index at the same normalized position.  Integer arithmetic implements
    deterministic round-half-up behavior, preserves both endpoints, and
    reduces exactly to the identity map when the two lengths agree.  No
    hidden-state interpolation or averaging is performed.
    """

    receiver_length = int(receiver_length)
    source_length = int(source_length)
    if receiver_length <= 0 or source_length <= 0:
        raise ValueError("Causal-v2 span lengths must both be positive")
    if receiver_length == 1:
        return ((source_length - 1) // 2,)
    if source_length == 1:
        return (0,) * receiver_length
    denominator = 2 * (receiver_length - 1)
    return tuple(
        (2 * receiver_index * (source_length - 1) + (receiver_length - 1))
        // denominator
        for receiver_index in range(receiver_length)
    )


def _causal_v2_span_mapping_diagnostics(
    receiver_length: int,
    source_length: int,
) -> dict[str, Any]:
    receiver_length = int(receiver_length)
    source_length = int(source_length)
    source_indices = causal_v2_monotonic_span_source_indices(
        receiver_length,
        source_length,
    )
    distinct_source_tokens = len(set(source_indices))
    if receiver_length == 1 or source_length == 1:
        max_normalized_position_error = 0.0
    else:
        max_normalized_position_error = max(
            abs(
                source_index / (source_length - 1)
                - receiver_index / (receiver_length - 1)
            )
            for receiver_index, source_index in enumerate(source_indices)
        )
    return {
        "mapping_supported": True,
        "source_index_map": json.dumps(list(source_indices)),
        "mapped_receiver_token_count": receiver_length,
        "distinct_source_token_count": distinct_source_tokens,
        "source_tokens_dropped": source_length - distinct_source_tokens,
        "source_tokens_reused": receiver_length - distinct_source_tokens,
        "max_normalized_position_error": float(max_normalized_position_error),
    }


def _causal_v2_patch_payload(
    *,
    site: str,
    layer_states: Any,
    state_slices: Mapping[int, tuple[int, int]],
    receiver: PromptEncoding,
    source: PromptEncoding,
    slot_indices: Sequence[int],
    alignment_policy: str = CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
) -> tuple[tuple[int, ...], Any, bool, str]:
    if alignment_policy != CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY:
        raise ValueError(
            f"Unknown causal-v2 full-span alignment policy: {alignment_policy}"
        )
    if site == "answer_query":
        return (int(receiver.query_position),), layer_states[0:1], True, ""
    receiver_spans = {int(span.slot_index): span for span in receiver.slot_spans}
    source_spans = {int(span.slot_index): span for span in source.slot_spans}
    receiver_positions: list[int] = []
    source_rows: list[Any] = []
    for raw_slot in slot_indices:
        slot = int(raw_slot)
        receiver_span = receiver_spans[slot]
        source_span = source_spans[slot]
        state_start, state_end = state_slices[slot]
        source_state = layer_states[state_start:state_end]
        if int(source_state.shape[0]) != int(source_span.model_token_length):
            raise RuntimeError("Captured source span length disagrees with metadata")
        if site == "toggled_needle_end":
            receiver_positions.append(int(receiver_span.end) - 1)
            source_rows.append(source_state[-1:])
            continue
        if site != "toggled_needle_span":
            raise ValueError(f"Unknown causal-v2 patch site: {site}")
        source_indices = causal_v2_monotonic_span_source_indices(
            int(receiver_span.model_token_length),
            int(source_span.model_token_length),
        )
        receiver_positions.extend(
            range(int(receiver_span.start), int(receiver_span.end))
        )
        source_rows.append(source_state[list(source_indices)])
    if not source_rows:
        raise RuntimeError("Causal-v2 prompt patch selected no slot states")
    import torch

    states = torch.cat(source_rows, dim=0)
    if int(states.shape[0]) != len(receiver_positions):
        raise RuntimeError("Causal-v2 patch state/position lengths differ")
    return tuple(receiver_positions), states, True, ""


def _next_same_count_source(
    receiver: PromptEncoding,
    by_variant_count: Mapping[tuple[str, int], Sequence[PromptEncoding]],
) -> PromptEncoding:
    candidates = sorted(
        (
            item
            for item in by_variant_count[
                (str(receiver.design_variant), int(receiver.count))
            ]
            if int(item.seed) != int(receiver.seed)
        ),
        key=lambda item: int(item.seed),
    )
    if not candidates:
        raise ValueError("same_count_seed control requires at least two seeds")
    later = [item for item in candidates if int(item.seed) > int(receiver.seed)]
    return later[0] if later else candidates[0]


def causal_v2_prompt_span_alignment_table(
    encodings: Sequence[PromptEncoding],
    *,
    count_pairs: Sequence[tuple[int, int]],
    evaluation_seeds: Sequence[int],
    alignment_policy: str = CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
) -> pd.DataFrame:
    """Preflight and record the registered donor-to-receiver span mapping."""

    if alignment_policy != CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY:
        raise ValueError(
            f"Unknown causal-v2 full-span alignment policy: {alignment_policy}"
        )

    by_key = _encoding_map(encodings)
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(value) for value in evaluation_seeds}):
        for receiver_count, donor_count in count_pairs:
            receiver = by_key.get(("v4.4", seed, int(receiver_count)))
            donor = by_key.get(("v4.4", seed, int(donor_count)))
            if receiver is None or donor is None:
                raise KeyError(
                    f"Missing causal-v2 alignment pair seed={seed} "
                    f"N={receiver_count}<-{donor_count}"
                )
            receiver_spans = {
                int(span.slot_index): span for span in receiver.slot_spans
            }
            donor_spans = {int(span.slot_index): span for span in donor.slot_spans}
            lower = min(int(receiver_count), int(donor_count))
            upper = max(int(receiver_count), int(donor_count))
            for slot in range(lower + 1, upper + 1):
                receiver_length = int(receiver_spans[slot].model_token_length)
                donor_length = int(donor_spans[slot].model_token_length)
                diagnostics = _causal_v2_span_mapping_diagnostics(
                    receiver_length,
                    donor_length,
                )
                rows.append(
                    {
                        "model_label": receiver.model_label,
                        "seed": int(seed),
                        "receiver_count": int(receiver_count),
                        "donor_count": int(donor_count),
                        "k": int(upper - lower),
                        "slot_index": int(slot),
                        "receiver_model_token_length": receiver_length,
                        "donor_model_token_length": donor_length,
                        "absolute_model_token_length_delta": abs(
                            receiver_length - donor_length
                        ),
                        "exact_model_token_alignment": bool(
                            receiver_length == donor_length
                        ),
                        "alignment_policy": alignment_policy,
                        **diagnostics,
                    }
                )
    if not rows:
        raise ValueError("No causal-v2 prompt span alignments were checked")
    return pd.DataFrame(rows)


def run_generation_residual_patching_v2(
    model: Any,
    tokenizer: Any,
    adapter: DecoderAdapter,
    encodings: Sequence[PromptEncoding],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    count_pairs: Sequence[tuple[int, int]],
    start_layers: Sequence[int],
    sites: Sequence[str],
    protocols: Sequence[str] = NEEDLE_PATCH_PROTOCOLS,
    control_conditions: Sequence[str] = ("donor_transport",),
    condition_filter: set[tuple[str, str, int, int]] | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    shared_state_cache: (
        MutableMapping[
            tuple[str, tuple[int, ...]], tuple[tuple[int, ...], dict[int, Any]]
        ]
        | None
    ) = None,
    shared_completion_cache: (
        MutableMapping[tuple[Any, ...], dict[str, Any]] | None
    ) = None,
    identity_execution_filter: set[tuple[int, int, int, str, str, int]] | None = None,
    require_clean_correct_pair: bool = False,
    valid_counts: Sequence[int] = tuple(range(0, 11)),
    max_new_tokens: int = 16,
    full_span_alignment_policy: str = CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY,
) -> pd.DataFrame:
    """Run V4.4 multi-slot +/-k prompt or answer-query residual transport.

    A pair that differs by ``k`` patches all ``k`` nested slots that change
    activation status.  Full-span interventions replace every receiver token
    with one complete donor-token vector under the registered monotonic
    endpoint-preserving map; endpoint interventions replace one final token
    per changed slot.  Both sites support a one-layer replacement and the
    registered L-to-final cumulative clamp.  The answer-query site patches
    only the final ``Total:`` prompt position.
    """

    if full_span_alignment_policy != CAUSAL_V2_FULL_SPAN_ALIGNMENT_POLICY:
        raise ValueError(
            "Unknown causal-v2 full-span alignment policy: "
            f"{full_span_alignment_policy}"
        )

    requested_sites = tuple(str(site) for site in sites)
    if not requested_sites or any(
        site not in RESIDUAL_PATCH_SITES for site in requested_sites
    ):
        raise ValueError(f"Invalid causal-v2 residual patch sites: {requested_sites}")
    requested_protocols = tuple(str(protocol) for protocol in protocols)
    if not requested_protocols or any(
        protocol not in NEEDLE_PATCH_PROTOCOLS for protocol in requested_protocols
    ):
        raise ValueError(f"Invalid causal-v2 patch protocols: {requested_protocols}")
    conditions = tuple(str(value) for value in control_conditions)
    allowed_conditions = {"donor_transport", "self_patch", "same_count_seed"}
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or any(value not in allowed_conditions for value in conditions)
    ):
        raise ValueError(f"Invalid causal-v2 patch conditions: {conditions}")
    if "donor_transport" not in conditions:
        raise ValueError("Causal-v2 patching must include donor_transport")
    starts = tuple(sorted({int(layer) for layer in start_layers}))
    if not starts or starts[0] < 0 or starts[-1] >= adapter.num_layers:
        raise ValueError("Causal-v2 residual patch layers are invalid")
    directed_pairs = tuple((int(left), int(right)) for left, right in count_pairs)
    if not directed_pairs or any(
        left == right or abs(right - left) not in {1, 3, 5}
        for left, right in directed_pairs
    ):
        raise ValueError("Causal-v2 pairs must be distinct with k in {1, 3, 5}")
    by_key = _encoding_map(encodings)
    by_variant_count: dict[tuple[str, int], list[PromptEncoding]] = {}
    for encoding in encodings:
        by_variant_count.setdefault(
            (str(encoding.design_variant), int(encoding.count)), []
        ).append(encoding)
    rows: list[dict[str, Any]] = []
    variants = sorted({str(item.design_variant) for item in encodings})
    available_seeds = sorted({int(item.seed) for item in encodings})
    seeds = (
        sorted({int(value) for value in evaluation_seeds})
        if evaluation_seeds is not None
        else available_seeds
    )
    if not seeds or not set(seeds).issubset(set(available_seeds)):
        raise ValueError("Causal-v2 evaluation seeds are missing from encodings")
    state_cache = shared_state_cache if shared_state_cache is not None else {}
    completion_cache = (
        shared_completion_cache if shared_completion_cache is not None else {}
    )
    identity_filter = (
        set(identity_execution_filter)
        if identity_execution_filter is not None
        else set()
    )

    def states_for(
        source: PromptEncoding,
        slot_indices: tuple[int, ...],
    ) -> tuple[dict[int, Any], dict[int, tuple[int, int]]]:
        positions, slices = _causal_v2_semantic_positions(source, slot_indices)
        key = (source.stimulus_id, positions)
        if key not in state_cache:
            _logits, captured = capture_post_block_states(
                model,
                adapter,
                source,
                positions,
                layers=starts if requested_protocols == ("single_layer",) else None,
            )
            state_cache[key] = (positions, captured)
        cached_positions, captured = state_cache[key]
        if cached_positions != positions:
            raise RuntimeError("Causal-v2 cached semantic positions changed")
        return captured, slices

    for variant in variants:
        if variant != "v4.4":
            raise ValueError("Causal-v2 residual patching is restricted to v4.4")
        for seed in seeds:
            for receiver_count, donor_count in directed_pairs:
                receiver = by_key.get((variant, seed, receiver_count))
                donor = by_key.get((variant, seed, donor_count))
                if receiver is None or donor is None:
                    raise KeyError(
                        f"Missing causal-v2 pair {variant} seed={seed} "
                        f"N={receiver_count}<-{donor_count}"
                    )
                _validate_pair(receiver, donor)
                receiver_label = baseline_labels[receiver.stimulus_id]
                donor_label = baseline_labels[donor.stimulus_id]
                _validate_baseline_label(receiver, receiver_label)
                _validate_baseline_label(donor, donor_label)
                if require_clean_correct_pair:
                    ineligible = [
                        ("receiver", receiver.stimulus_id, receiver_label),
                        ("donor", donor.stimulus_id, donor_label),
                    ]
                    failures = [
                        (
                            role,
                            stimulus_id,
                            label.get("outcome_group"),
                            label.get("format_valid"),
                        )
                        for role, stimulus_id, label in ineligible
                        if str(label.get("outcome_group")) != "correct"
                        or not _as_bool(label.get("format_valid"))
                    ]
                    if failures:
                        raise ValueError(
                            "Correct-only residual patching received an ineligible "
                            f"receiver/donor pair: {failures}"
                        )
                lower = min(receiver_count, donor_count)
                upper = max(receiver_count, donor_count)
                slot_indices = tuple(range(lower + 1, upper + 1))
                k = upper - lower
                if len(slot_indices) != k:
                    raise RuntimeError("Causal-v2 changed-slot accounting failed")
                source_conditions: list[tuple[str, PromptEncoding]] = []
                for condition in conditions:
                    if condition == "donor_transport":
                        source_conditions.append((condition, donor))
                    elif condition == "self_patch":
                        source_conditions.append((condition, receiver))
                    else:
                        source_conditions.append(
                            (
                                condition,
                                _next_same_count_source(receiver, by_variant_count),
                            )
                        )
                for condition, state_source in source_conditions:
                    source_label = baseline_labels[state_source.stimulus_id]
                    _validate_baseline_label(state_source, source_label)
                    source_payload: (
                        tuple[dict[int, Any], dict[int, tuple[int, int]]] | None
                    ) = None
                    for site in requested_sites:
                        for start_layer in starts:
                            for protocol in requested_protocols:
                                filter_key = (site, protocol, int(start_layer), int(k))
                                if (
                                    condition_filter is not None
                                    and filter_key not in condition_filter
                                ):
                                    continue
                                intervention_layers = (
                                    (int(start_layer),)
                                    if protocol == "single_layer"
                                    else tuple(
                                        range(int(start_layer), adapter.num_layers)
                                    )
                                )
                                identity_key = (
                                    int(seed),
                                    int(receiver_count),
                                    int(donor_count),
                                    str(site),
                                    str(protocol),
                                    int(start_layer),
                                )
                                reuse_identity = bool(
                                    condition == "self_patch"
                                    and identity_key not in identity_filter
                                )
                                cache_key = _causal_v2_completion_cache_key(
                                    condition=condition,
                                    receiver=receiver,
                                    state_source=state_source,
                                    site=site,
                                    protocol=protocol,
                                    start_layer=start_layer,
                                    slot_indices=slot_indices,
                                )
                                interventions: dict[int, tuple[Sequence[int], Any]] = {}
                                executable = True
                                skip_reason = ""
                                cache_hit = bool(
                                    not reuse_identity and cache_key in completion_cache
                                )
                                if not reuse_identity and not cache_hit:
                                    if source_payload is None:
                                        source_payload = states_for(
                                            state_source, slot_indices
                                        )
                                    source_states, source_slices = source_payload
                                    for layer in intervention_layers:
                                        if layer not in source_states:
                                            raise RuntimeError(
                                                "Causal-v2 source state cache lacks "
                                                f"layer {layer}"
                                            )
                                        positions, states, ok, reason = (
                                            _causal_v2_patch_payload(
                                                site=site,
                                                layer_states=source_states[layer],
                                                state_slices=source_slices,
                                                receiver=receiver,
                                                source=state_source,
                                                slot_indices=slot_indices,
                                                alignment_policy=(
                                                    full_span_alignment_policy
                                                ),
                                            )
                                        )
                                        if not ok:
                                            executable = False
                                            skip_reason = reason
                                            break
                                        interventions[int(layer)] = (positions, states)
                                if reuse_identity:
                                    generation_executed = False
                                    generation_reuse_mode = "baseline_identity_reuse"
                                elif cache_hit:
                                    generation_executed = False
                                    generation_reuse_mode = (
                                        "equivalent_intervention_cache"
                                    )
                                elif condition == "self_patch":
                                    generation_executed = True
                                    generation_reuse_mode = (
                                        "executed_identity_preflight"
                                    )
                                else:
                                    generation_executed = True
                                    generation_reuse_mode = "fresh_intervention"
                                metadata = {
                                    **_base_metadata(receiver),
                                    **_baseline_metadata(receiver_label),
                                    "receiver_stimulus_id": receiver.stimulus_id,
                                    "donor_stimulus_id": donor.stimulus_id,
                                    "state_donor_stimulus_id": state_source.stimulus_id,
                                    "receiver_count": int(receiver_count),
                                    "donor_count": int(donor_count),
                                    "state_donor_count": int(state_source.count),
                                    "donor_baseline_outcome": str(
                                        donor_label["outcome_group"]
                                    ),
                                    "donor_baseline_predicted_count": _optional_int(
                                        donor_label.get("parsed_count")
                                    ),
                                    "state_donor_baseline_outcome": str(
                                        source_label["outcome_group"]
                                    ),
                                    "condition": condition,
                                    "direction": (
                                        "needle_insertion"
                                        if donor_count > receiver_count
                                        else "needle_removal"
                                    ),
                                    "target_direction": (
                                        "increase"
                                        if donor_count > receiver_count
                                        else "decrease"
                                    ),
                                    "k": int(k),
                                    "site": site,
                                    "patch_protocol": protocol,
                                    "start_layer": int(start_layer),
                                    "patched_layer_count": len(intervention_layers),
                                    "changed_slot_indices": json.dumps(
                                        list(slot_indices)
                                    ),
                                    "changed_slot_count": len(slot_indices),
                                    "full_span_alignment_policy": (
                                        full_span_alignment_policy
                                    ),
                                    "generation_executed": generation_executed,
                                    "generation_reuse_mode": generation_reuse_mode,
                                    "analysis_population": (
                                        "clean_correct_pair_only"
                                        if require_clean_correct_pair
                                        else "all_examples"
                                    ),
                                    "status": "ok" if executable else "skipped",
                                    "skip_reason": skip_reason,
                                }
                                if not executable:
                                    rows.append(metadata)
                                    continue
                                if reuse_identity:
                                    completion = _completion_from_generation_label(
                                        receiver_label
                                    )
                                elif cache_hit:
                                    completion = completion_cache[cache_key]
                                else:
                                    completion = generate_with_residual_interventions(
                                        model,
                                        tokenizer,
                                        adapter,
                                        receiver,
                                        interventions,
                                        max_new_tokens=max_new_tokens,
                                    )
                                    completion_cache[cache_key] = dict(completion)
                                    if condition == "self_patch":
                                        expected = _completion_from_generation_label(
                                            receiver_label
                                        )
                                        observed_ids = [
                                            int(value)
                                            for value in completion[
                                                "generated_token_ids"
                                            ]
                                        ]
                                        if (
                                            str(completion["completion_text"])
                                            != expected["completion_text"]
                                            or observed_ids
                                            != expected["generated_token_ids"]
                                        ):
                                            raise RuntimeError(
                                                "Self-patch identity preflight changed "
                                                "the greedy completion: "
                                                f"key={identity_key}"
                                            )
                                outcome = intervention_outcome(
                                    completion,
                                    receiver,
                                    receiver_label,
                                    valid_counts=valid_counts,
                                )
                                rows.append(
                                    {
                                        **metadata,
                                        **outcome,
                                        **transport_fields(
                                            outcome,
                                            receiver_label=receiver_label,
                                            donor_label=donor_label,
                                            receiver_count=receiver_count,
                                            donor_count=donor_count,
                                        ),
                                    }
                                )
                print(
                    "[v4 causal-v2 patch] "
                    f"{variant} seed={seed} N={receiver_count}<-{donor_count} k={k}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No causal-v2 residual-patching rows were produced")
    return pd.DataFrame(rows)


def summarize_generation_residual_patching_v2(detail: pd.DataFrame) -> pd.DataFrame:
    successful = detail[detail["status"].astype(str).eq("ok")].copy()
    if successful.empty:
        raise ValueError("No successful causal-v2 residual patches")
    groups = [
        "model_label",
        "site",
        "patch_protocol",
        "start_layer",
        "k",
        "target_direction",
        "condition",
        "baseline_outcome",
    ]
    return (
        successful.groupby(groups, as_index=False, dropna=False)
        .agg(
            examples=("receiver_stimulus_id", "size"),
            seeds=("seed", "nunique"),
            patched_valid_rate=("patched_format_valid", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
            mean_generated_count_shift=("generated_count_shift", "mean"),
            mean_normalized_transport=("normalized_transport", "mean"),
            mean_strict_normalized_transport=("strict_normalized_transport", "mean"),
            mean_target_conformity=("target_conformity", "mean"),
            mean_strict_target_conformity=("strict_target_conformity", "mean"),
            target_hit_rate=("strict_target_hit", "mean"),
            average_patch_target_accuracy=("strict_target_hit", "mean"),
            average_post_patch_receiver_accuracy=("patched_is_correct", "mean"),
        )
        .sort_values(groups)
        .reset_index(drop=True)
    )
