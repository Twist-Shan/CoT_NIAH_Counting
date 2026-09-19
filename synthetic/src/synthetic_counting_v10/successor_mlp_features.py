from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import V10Config
from .core import Vocab, count_bin, render
from .report_followups import load_run
from .successor_conversion import _tensor_from_output
from .successor_patching import (
    DIRECTION_CLOSE,
    DIRECTION_CONTINUE,
    _two_token_margin,
    successor_nested_example_pair,
)
from .training import load_final_model


DEFAULT_SUPPORT_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
FEATURE_FAMILIES = ("ranked_feature_replacement", "random_feature_replacement")
DIRECTION_FAMILIES = ("sparse_mean_direction", "random_sparse_mean_direction")


def _analysis_layers(cfg: V10Config) -> tuple[int, ...]:
    """Return the last two 0-based layers (Layer 3-4 in the main 4-layer model)."""
    return tuple(range(max(0, int(cfg.n_layer) - 2), int(cfg.n_layer)))


def _replace_tensor_output(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError(f"Expected tensor or tensor-first tuple, got {type(output)!r}")


@torch.no_grad()
def capture_mlp_intermediates(
    model,
    input_ids: torch.Tensor,
    position: int,
    layers: Iterable[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Capture post-GELU MLP intermediate vectors at one semantic query row."""
    states: dict[int, torch.Tensor] = {}
    handles = []

    def capture(layer: int):
        def hook(_module, _args, output):
            hidden = _tensor_from_output(output)
            states[int(layer)] = hidden[0, int(position)].detach().clone()

        return hook

    for layer in layers:
        handles.append(model.transformer.h[int(layer)].mlp.act.register_forward_hook(capture(layer)))
    try:
        logits = model(input_ids=input_ids).logits[0, int(position)].detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return logits, states


@torch.no_grad()
def patched_mlp_intermediate_forward(
    model,
    receiver_ids: torch.Tensor,
    receiver_position: int,
    layer: int,
    replacements: list[torch.Tensor],
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    """Run post-GELU feature replacements and return final logits at the query row."""
    if not replacements:
        return torch.empty((0, model.config.vocab_size), device=receiver_ids.device)
    outputs: list[torch.Tensor] = []
    module = model.transformer.h[int(layer)].mlp.act
    for start in range(0, len(replacements), int(batch_size)):
        chunk = replacements[start : start + int(batch_size)]
        replacement = torch.stack(chunk).to(device=receiver_ids.device)
        repeated_ids = receiver_ids.repeat(len(chunk), 1)

        def patch(_module, _args, output):
            hidden = _tensor_from_output(output)
            patched = hidden.clone()
            patched[:, int(receiver_position), :] = replacement.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            return _replace_tensor_output(output, patched)

        handle = module.register_forward_hook(patch)
        try:
            logits = model(input_ids=repeated_ids).logits[:, int(receiver_position)].detach()
        finally:
            handle.remove()
        outputs.append(logits)
    return torch.cat(outputs, dim=0)


def _decision_specs(
    vocab: Vocab,
    k: int,
    short_ids: torch.Tensor,
    long_ids: torch.Tensor,
    position: int,
    short_logits: torch.Tensor,
    long_logits: torch.Tensor,
    short_features: dict[int, torch.Tensor],
    long_features: dict[int, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    continue_id = vocab.number_id(k + 1)
    close_id = vocab.think_close_id
    return (
        {
            "direction": DIRECTION_CONTINUE,
            "receiver_ids": short_ids,
            "clean_logits": long_logits,
            "corrupt_logits": short_logits,
            "clean_features": long_features,
            "corrupt_features": short_features,
            "receiver_position": position,
            "target_id": continue_id,
            "alternative_id": close_id,
        },
        {
            "direction": DIRECTION_CLOSE,
            "receiver_ids": long_ids,
            "clean_logits": short_logits,
            "corrupt_logits": long_logits,
            "clean_features": short_features,
            "corrupt_features": long_features,
            "receiver_position": position,
            "target_id": close_id,
            "alternative_id": continue_id,
        },
    )


def _feature_margin_coefficients(model, layer: int, target_id: int, alternative_id: int) -> torch.Tensor:
    """Linear pre-ln_f coefficient from each post-GELU feature to a token margin."""
    unembedding_delta = (
        model.lm_head.weight[int(target_id)] - model.lm_head.weight[int(alternative_id)]
    )
    # Hugging Face GPT-2 Conv1D stores c_proj.weight as [n_inner, n_embd].
    return model.transformer.h[int(layer)].mlp.c_proj.weight @ unembedding_delta


def _pair_payloads(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    example_index: int,
    k: int,
    layers: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    model_device = next(model.parameters()).device
    seed = cfg.seed + 1_241_000 + int(k) * 1_000 + int(example_index)
    short_example, long_example = successor_nested_example_pair(cfg, vocab, int(k), seed)
    short_item = render(short_example, vocab, "thinking")
    long_item = render(long_example, vocab, "thinking")
    short_position = int(short_item.spans.trace_marker_positions[int(k) - 1])
    long_position = int(long_item.spans.trace_marker_positions[int(k) - 1])
    if short_position != long_position:
        raise AssertionError("nested successor query positions must align")
    short_ids = torch.tensor([short_item.input_ids], dtype=torch.long, device=model_device)
    long_ids = torch.tensor([long_item.input_ids], dtype=torch.long, device=model_device)
    short_logits, short_features = capture_mlp_intermediates(
        model, short_ids, short_position, layers
    )
    long_logits, long_features = capture_mlp_intermediates(model, long_ids, long_position, layers)
    return _decision_specs(
        vocab,
        int(k),
        short_ids,
        long_ids,
        short_position,
        short_logits,
        long_logits,
        short_features,
        long_features,
    )


@torch.no_grad()
def fit_mlp_feature_statistics(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    fit_examples_per_k: int = 2,
    layers: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], dict[str, torch.Tensor]]]:
    """Fit paired feature rankings and mean clean-minus-corrupt directions."""
    layers = layers or _analysis_layers(cfg)
    accumulators: dict[tuple[str, str, int], dict[str, Any]] = {}
    progress = tqdm(
        total=(cfg.count_max - cfg.count_min) * int(fit_examples_per_k),
        desc="v10 fit Layer 3-4 MLP features",
    )
    for k in range(cfg.count_min, cfg.count_max):
        for example_index in range(int(fit_examples_per_k)):
            for spec in _pair_payloads(
                model,
                cfg,
                vocab,
                example_index=example_index,
                k=k,
                layers=layers,
            ):
                for layer in layers:
                    clean = spec["clean_features"][int(layer)].float()
                    corrupt = spec["corrupt_features"][int(layer)].float()
                    delta = clean - corrupt
                    coeff = _feature_margin_coefficients(
                        model,
                        int(layer),
                        int(spec["target_id"]),
                        int(spec["alternative_id"]),
                    ).detach().float()
                    evidence = delta * coeff
                    key = (str(spec["direction"]), count_bin(k), int(layer))
                    if key not in accumulators:
                        zeros = torch.zeros_like(delta, device="cpu")
                        accumulators[key] = {
                            "n": 0,
                            "delta_sum": zeros.clone(),
                            "delta_abs_sum": zeros.clone(),
                            "coeff_sum": zeros.clone(),
                            "evidence_sum": zeros.clone(),
                            "evidence_abs_sum": zeros.clone(),
                            "evidence_sq_sum": zeros.clone(),
                            "positive_count": zeros.clone(),
                        }
                    acc = accumulators[key]
                    acc["n"] += 1
                    for name, value in (
                        ("delta_sum", delta),
                        ("delta_abs_sum", delta.abs()),
                        ("coeff_sum", coeff),
                        ("evidence_sum", evidence),
                        ("evidence_abs_sum", evidence.abs()),
                        ("evidence_sq_sum", evidence.square()),
                        ("positive_count", evidence.gt(0).float()),
                    ):
                        acc[name] += value.detach().cpu()
            progress.update(1)
    progress.close()

    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    for key, acc in accumulators.items():
        direction, bin_name, layer = key
        n = int(acc["n"])
        mean_delta = acc["delta_sum"] / max(1, n)
        mean_evidence = acc["evidence_sum"] / max(1, n)
        evidence_var = (acc["evidence_sq_sum"] / max(1, n) - mean_evidence.square()).clamp_min(0)
        ranking = torch.argsort(mean_evidence, descending=True)
        sparse_ranking = torch.argsort(mean_evidence.abs(), descending=True)
        fitted[key] = {
            "mean_delta": mean_delta,
            "mean_evidence": mean_evidence,
            "ranking": ranking,
            "sparse_ranking": sparse_ranking,
        }
        for feature in range(mean_delta.numel()):
            rows.append(
                {
                    "direction": direction,
                    "count_bin": bin_name,
                    "layer": int(layer),
                    "feature": int(feature),
                    "n_fit_pairs": n,
                    "mean_activation_delta": float(mean_delta[feature]),
                    "mean_abs_activation_delta": float(acc["delta_abs_sum"][feature] / n),
                    "mean_logit_coefficient": float(acc["coeff_sum"][feature] / n),
                    "mean_projected_evidence": float(mean_evidence[feature]),
                    "mean_abs_projected_evidence": float(acc["evidence_abs_sum"][feature] / n),
                    "projected_evidence_std": float(evidence_var[feature].sqrt()),
                    "positive_evidence_rate": float(acc["positive_count"][feature] / n),
                    "signed_rank": int((ranking == feature).nonzero(as_tuple=False)[0, 0]) + 1,
                    "absolute_rank": int((sparse_ranking == feature).nonzero(as_tuple=False)[0, 0]) + 1,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["direction", "count_bin", "layer", "signed_rank"]
    )
    return frame.reset_index(drop=True), fitted


def summarize_feature_concentration(
    feature_stats: pd.DataFrame,
    support_sizes: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in feature_stats.groupby(["direction", "count_bin", "layer"]):
        ordered = frame.sort_values("signed_rank")
        positive_total = ordered.mean_projected_evidence.clip(lower=0).sum()
        absolute_total = ordered.mean_abs_projected_evidence.sum()
        for size in support_sizes:
            n = min(int(size), len(ordered))
            top = ordered.head(n)
            rows.append(
                {
                    "direction": keys[0],
                    "count_bin": keys[1],
                    "layer": int(keys[2]),
                    "support_size": n,
                    "positive_evidence_fraction": (
                        float(top.mean_projected_evidence.clip(lower=0).sum() / positive_total)
                        if positive_total > 0
                        else float("nan")
                    ),
                    "absolute_evidence_fraction": (
                        float(top.mean_abs_projected_evidence.sum() / absolute_total)
                        if absolute_total > 0
                        else float("nan")
                    ),
                    "top_features": " ".join(str(value) for value in top.feature.tolist()),
                }
            )
    return pd.DataFrame(rows)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    norm = value.norm()
    return value / norm if float(norm) > 1e-12 else torch.zeros_like(value)


def _condition_replacements(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    fitted: dict[str, torch.Tensor],
    support_sizes: tuple[int, ...],
    *,
    random_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    n_inner = int(clean.numel())
    sizes = tuple(sorted({min(int(value), n_inner) for value in support_sizes if int(value) > 0}))
    ranking = fitted["ranking"].long()
    sparse_ranking = fitted["sparse_ranking"].long()
    mean_delta = fitted["mean_delta"].to(dtype=clean.dtype)
    pair_delta = clean - corrupt
    conditions: list[dict[str, Any]] = []
    replacements: list[torch.Tensor] = []

    def append(family: str, size: int, replicate: int, replacement: torch.Tensor, support: torch.Tensor):
        conditions.append(
            {
                "family": family,
                "support_size": int(size),
                "replicate": int(replicate),
                "support": support,
            }
        )
        replacements.append(replacement)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_orders = [torch.randperm(n_inner, generator=generator) for _ in range(int(random_replicates))]
    for size in sizes:
        support = ranking[:size]
        replacement = corrupt.clone()
        replacement[support.to(clean.device)] = clean[support.to(clean.device)]
        append("ranked_feature_replacement", size, -1, replacement, support)
        if size < n_inner:
            for replicate, order in enumerate(random_orders):
                random_support = order[:size]
                random_replacement = corrupt.clone()
                random_replacement[random_support.to(clean.device)] = clean[
                    random_support.to(clean.device)
                ]
                append(
                    "random_feature_replacement",
                    size,
                    replicate,
                    random_replacement,
                    random_support,
                )

    for size in sizes:
        support = sparse_ranking[:size]
        direction = torch.zeros_like(mean_delta)
        direction[support] = mean_delta[support]
        direction = _normalized(direction).to(clean.device)
        amplitude = torch.dot(pair_delta, direction)
        append(
            "sparse_mean_direction",
            size,
            -1,
            corrupt + amplitude * direction,
            support,
        )
        if size < n_inner:
            for replicate, order in enumerate(random_orders):
                random_support = order[:size]
                random_direction = torch.zeros_like(mean_delta)
                random_direction[random_support] = mean_delta[random_support]
                random_direction = _normalized(random_direction).to(clean.device)
                random_amplitude = torch.dot(pair_delta, random_direction)
                append(
                    "random_sparse_mean_direction",
                    size,
                    replicate,
                    corrupt + random_amplitude * random_direction,
                    random_support,
                )
    return conditions, replacements


@torch.no_grad()
def evaluate_mlp_feature_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    fitted: dict[tuple[str, str, int], dict[str, torch.Tensor]],
    *,
    eval_examples_per_k: int = 2,
    fit_examples_per_k: int = 2,
    support_sizes: tuple[int, ...] = DEFAULT_SUPPORT_SIZES,
    random_replicates: int = 4,
    layers: tuple[int, ...] | None = None,
    patch_batch_size: int = 32,
) -> pd.DataFrame:
    layers = layers or _analysis_layers(cfg)
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=(cfg.count_max - cfg.count_min) * int(eval_examples_per_k) * 2 * len(layers),
        desc="v10 causal Layer 3-4 MLP feature patching",
    )
    for k in range(cfg.count_min, cfg.count_max):
        for eval_index in range(int(eval_examples_per_k)):
            example_index = int(fit_examples_per_k) + eval_index
            for spec in _pair_payloads(
                model,
                cfg,
                vocab,
                example_index=example_index,
                k=k,
                layers=layers,
            ):
                clean_margin = _two_token_margin(
                    spec["clean_logits"], spec["target_id"], spec["alternative_id"]
                )
                corrupt_margin = _two_token_margin(
                    spec["corrupt_logits"], spec["target_id"], spec["alternative_id"]
                )
                denominator = clean_margin - corrupt_margin
                for layer in layers:
                    clean = spec["clean_features"][int(layer)].float()
                    corrupt = spec["corrupt_features"][int(layer)].float()
                    key = (str(spec["direction"]), count_bin(k), int(layer))
                    conditions, replacements = _condition_replacements(
                        clean,
                        corrupt,
                        fitted[key],
                        support_sizes,
                        random_replicates=random_replicates,
                        seed=cfg.seed + 1_771_000 + k * 100 + layer,
                    )
                    logits = patched_mlp_intermediate_forward(
                        model,
                        spec["receiver_ids"],
                        int(spec["receiver_position"]),
                        int(layer),
                        replacements,
                        batch_size=patch_batch_size,
                    )
                    coeff = _feature_margin_coefficients(
                        model,
                        int(layer),
                        int(spec["target_id"]),
                        int(spec["alternative_id"]),
                    ).detach().float()
                    pair_delta = clean - corrupt
                    pair_norm_sq = float(pair_delta.square().sum())
                    for condition, patched_logits, replacement in zip(
                        conditions, logits, replacements
                    ):
                        patched_margin = _two_token_margin(
                            patched_logits, spec["target_id"], spec["alternative_id"]
                        )
                        transported = replacement.to(corrupt.device) - corrupt
                        support = condition.pop("support")
                        rows.append(
                            {
                                "decision_k": int(k),
                                "count_bin": count_bin(k),
                                "example_index": int(example_index),
                                "split": "held_out_eval",
                                "direction": str(spec["direction"]),
                                "layer": int(layer),
                                **condition,
                                "support_signature": ",".join(
                                    str(int(value)) for value in support.tolist()
                                ),
                                "clean_margin": clean_margin,
                                "corrupt_margin": corrupt_margin,
                                "patched_margin": patched_margin,
                                "margin_shift": patched_margin - corrupt_margin,
                                "normalized_recovery": (
                                    (patched_margin - corrupt_margin) / denominator
                                    if abs(denominator) > 1e-8
                                    else float("nan")
                                ),
                                "patched_target_correct": float(patched_margin > 0),
                                "linearized_direct_margin_shift": float(
                                    torch.dot(transported.float(), coeff)
                                ),
                                "pair_delta_fraction_transported": (
                                    float(transported.square().sum() / pair_norm_sq)
                                    if pair_norm_sq > 1e-12
                                    else float("nan")
                                ),
                            }
                        )
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarize_mlp_feature_patching(detail: pd.DataFrame) -> pd.DataFrame:
    detail = detail.copy()
    detail["pair_key"] = (
        detail["decision_k"].astype(str)
        + ":"
        + detail["example_index"].astype(str)
    )
    return (
        detail.groupby(
            ["direction", "count_bin", "layer", "family", "support_size"],
            as_index=False,
        )
        .agg(
            n_rows=("decision_k", "size"),
            n_decision_values=("decision_k", "nunique"),
            n_pairs=("pair_key", "nunique"),
            normalized_recovery=("normalized_recovery", "mean"),
            normalized_recovery_std=("normalized_recovery", "std"),
            patched_target_correct=("patched_target_correct", "mean"),
            margin_shift=("margin_shift", "mean"),
            linearized_direct_margin_shift=("linearized_direct_margin_shift", "mean"),
            pair_delta_fraction_transported=("pair_delta_fraction_transported", "mean"),
        )
        .sort_values(["direction", "count_bin", "layer", "family", "support_size"])
        .reset_index(drop=True)
    )


def run_successor_mlp_features(
    run_dir: str | Path,
    *,
    fit_examples_per_k: int = 2,
    eval_examples_per_k: int = 2,
    support_sizes: tuple[int, ...] = DEFAULT_SUPPORT_SIZES,
    random_replicates: int = 4,
    patch_batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "successor_mlp_features"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {
        "mlp_feature_statistics": tables / "mlp_feature_statistics.csv",
        "mlp_feature_concentration": tables / "mlp_feature_concentration.csv",
        "mlp_feature_patching": tables / "mlp_feature_patching.csv",
        "mlp_feature_patching_summary": tables / "mlp_feature_patching_summary.csv",
    }
    if not overwrite and all(path.exists() for path in paths.values()):
        return {name: pd.read_csv(path) for name, path in paths.items()}

    cfg, vocab = load_run(run_dir, device=device)
    model = load_final_model(cfg, vocab, run_dir, "thinking")
    layers = _analysis_layers(cfg)
    n_inner = int(model.config.n_inner or model.transformer.h[0].mlp.c_fc.weight.shape[1])
    clipped_sizes = tuple(sorted({min(int(value), n_inner) for value in support_sizes if value > 0}))
    feature_stats, fitted = fit_mlp_feature_statistics(
        model,
        cfg,
        vocab,
        fit_examples_per_k=fit_examples_per_k,
        layers=layers,
    )
    concentration = summarize_feature_concentration(feature_stats, clipped_sizes)
    patch_detail = evaluate_mlp_feature_patching(
        model,
        cfg,
        vocab,
        fitted,
        eval_examples_per_k=eval_examples_per_k,
        fit_examples_per_k=fit_examples_per_k,
        support_sizes=clipped_sizes,
        random_replicates=random_replicates,
        layers=layers,
        patch_batch_size=patch_batch_size,
    )
    outputs = {
        "mlp_feature_statistics": feature_stats,
        "mlp_feature_concentration": concentration,
        "mlp_feature_patching": patch_detail,
        "mlp_feature_patching_summary": summarize_mlp_feature_patching(patch_detail),
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "device": cfg.device,
        "layers_zero_based": list(layers),
        "layers_report_labels": [int(layer) + 1 for layer in layers],
        "mlp_intermediate": "post-GELU c_fc features before c_proj",
        "n_inner": n_inner,
        "fit_examples_per_k": int(fit_examples_per_k),
        "eval_examples_per_k": int(eval_examples_per_k),
        "fit_eval_separation": "disjoint example_index values for every decision k",
        "support_sizes": list(clipped_sizes),
        "random_replicates": int(random_replicates),
        "feature_ranking": (
            "mean over fit pairs of (clean-corrupt post-GELU activation) times the "
            "feature's linear c_proj-to-target-minus-alternative unembedding coefficient"
        ),
        "feature_patch": "replace only selected corrupt post-GELU coordinates with clean values",
        "sparse_direction": (
            "mean clean-corrupt activation direction restricted to top-|projected evidence| "
            "coordinates; patch transports the held-out pair projection on that direction"
        ),
        "query": "teacher-forced trace marker M_k",
        "directions": [DIRECTION_CONTINUE, DIRECTION_CLOSE],
        "autoregressive_rollout": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
