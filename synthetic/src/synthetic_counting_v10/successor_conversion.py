from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import V10Config
from .core import Vocab, count_bin, render
from .report_followups import load_run
from .successor_patching import (
    DIRECTION_CLOSE,
    DIRECTION_CONTINUE,
    _two_token_margin,
    successor_nested_example_pair,
)
from .training import load_final_model


RESIDUAL_STAGES = ("resid_pre", "post_attn", "post_mlp")
COMPONENT_STAGES = ("attn_out", "mlp_out")
INTERVENTIONS = (
    "attn_direct_residual",
    "attn_native_mlp",
    "mlp_out_only",
    "attn_plus_mlp",
    "post_attn_state",
    "post_mlp_state",
)


def _tensor_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor or tensor-first tuple, got {type(output)!r}")


def _replace_tensor_output(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError(f"Expected tensor or tensor-first tuple, got {type(output)!r}")


def _replace_position(hidden: torch.Tensor, position: int, value: torch.Tensor) -> torch.Tensor:
    patched = hidden.clone()
    patched[:, int(position), :] = value.to(device=hidden.device, dtype=hidden.dtype)
    return patched


@torch.no_grad()
def capture_sublayer_states(
    model,
    input_ids: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor]]:
    """Capture residual and component vectors at one semantic query row."""
    states: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def block_pre(layer: int) -> Callable:
        def hook(_module, args):
            hidden = args[0]
            states[(layer, "resid_pre")] = hidden[0, int(position)].detach().clone()

        return hook

    def tensor_output(layer: int, name: str) -> Callable:
        def hook(_module, _args, output):
            hidden = _tensor_from_output(output)
            states[(layer, name)] = hidden[0, int(position)].detach().clone()

        return hook

    def tensor_pre(layer: int, name: str) -> Callable:
        def hook(_module, args):
            hidden = args[0]
            states[(layer, name)] = hidden[0, int(position)].detach().clone()

        return hook

    for layer, block in enumerate(model.transformer.h):
        handles.extend(
            [
                block.register_forward_pre_hook(block_pre(layer)),
                block.attn.register_forward_hook(tensor_output(layer, "attn_out")),
                block.ln_2.register_forward_pre_hook(tensor_pre(layer, "post_attn")),
                block.mlp.register_forward_hook(tensor_output(layer, "mlp_out")),
                block.register_forward_hook(tensor_output(layer, "post_mlp")),
            ]
        )
    try:
        logits = model(input_ids=input_ids).logits[0, int(position)].detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return logits, states


@torch.no_grad()
def _residual_logit_margin(
    model,
    state: torch.Tensor,
    target_id: int,
    alternative_id: int,
) -> float:
    normalized = model.transformer.ln_f(state.unsqueeze(0))
    logits = model.lm_head(normalized)[0]
    return _two_token_margin(logits, target_id, alternative_id)


@torch.no_grad()
def _component_logit_margin(
    model,
    component: torch.Tensor,
    target_id: int,
    alternative_id: int,
) -> float:
    # The tied unembedding is linear, so this measures the additive component's
    # direct target-versus-alternative contribution. It deliberately omits ln_f.
    logits = model.lm_head(component.unsqueeze(0))[0]
    return _two_token_margin(logits, target_id, alternative_id)


@torch.no_grad()
def patched_sublayer_forward(
    model,
    receiver_ids: torch.Tensor,
    receiver_position: int,
    donor_states: dict[tuple[int, str], torch.Tensor],
    receiver_states: dict[tuple[int, str], torch.Tensor],
    layer: int,
    intervention: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one layer-local sublayer intervention at the receiver query.

    Returns final logits and the MLP output actually used at the patched layer.
    """
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown successor-conversion intervention: {intervention}")
    block = model.transformer.h[int(layer)]
    handles = []
    observed_mlp: dict[str, torch.Tensor] = {}

    def block_pre_hook(_module, args):
        hidden = args[0]
        replacement = donor_states[(int(layer), "resid_pre")]
        patched = _replace_position(hidden, receiver_position, replacement)
        return (patched, *args[1:])

    def output_patch(name: str, replacement_source: str) -> Callable:
        def hook(_module, _args, output):
            hidden = _tensor_from_output(output)
            replacement_states = donor_states if replacement_source == "donor" else receiver_states
            patched = _replace_position(
                hidden,
                receiver_position,
                replacement_states[(int(layer), name)],
            )
            return _replace_tensor_output(output, patched)

        return hook

    def observe_mlp(_module, _args, output):
        hidden = _tensor_from_output(output)
        observed_mlp["value"] = hidden[0, int(receiver_position)].detach().clone()

    if intervention == "attn_direct_residual":
        handles.append(block.attn.register_forward_hook(output_patch("attn_out", "donor")))
        # Keep the MLP addend at its corrupt baseline, isolating the direct
        # residual path of the clean attention output.
        handles.append(block.mlp.register_forward_hook(output_patch("mlp_out", "receiver")))
    elif intervention == "attn_native_mlp":
        handles.append(block.attn.register_forward_hook(output_patch("attn_out", "donor")))
    elif intervention == "mlp_out_only":
        handles.append(block.mlp.register_forward_hook(output_patch("mlp_out", "donor")))
    elif intervention == "attn_plus_mlp":
        handles.append(block.attn.register_forward_hook(output_patch("attn_out", "donor")))
        handles.append(block.mlp.register_forward_hook(output_patch("mlp_out", "donor")))
    elif intervention == "post_attn_state":
        handles.append(block.register_forward_pre_hook(block_pre_hook))
        handles.append(block.attn.register_forward_hook(output_patch("attn_out", "donor")))
    elif intervention == "post_mlp_state":
        handles.append(block.register_forward_hook(output_patch("post_mlp", "donor")))

    # Registered after patch hooks so this sees the vector actually used.
    handles.append(block.mlp.register_forward_hook(observe_mlp))
    try:
        logits = model(input_ids=receiver_ids).logits[0, int(receiver_position)].detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    mlp_value = observed_mlp.get("value", receiver_states[(int(layer), "mlp_out")])
    return logits, mlp_value


def _direction_specs(
    vocab: Vocab,
    k: int,
    short_ids: torch.Tensor,
    long_ids: torch.Tensor,
    short_position: int,
    long_position: int,
    short_logits: torch.Tensor,
    long_logits: torch.Tensor,
    short_states: dict[tuple[int, str], torch.Tensor],
    long_states: dict[tuple[int, str], torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    continue_id = vocab.number_id(k + 1)
    close_id = vocab.think_close_id
    return (
        {
            "direction": DIRECTION_CONTINUE,
            "receiver_ids": short_ids,
            "clean_logits": long_logits,
            "corrupt_logits": short_logits,
            "clean_states": long_states,
            "receiver_states": short_states,
            "receiver_position": short_position,
            "target_id": continue_id,
            "alternative_id": close_id,
        },
        {
            "direction": DIRECTION_CLOSE,
            "receiver_ids": long_ids,
            "clean_logits": short_logits,
            "corrupt_logits": long_logits,
            "clean_states": short_states,
            "receiver_states": long_states,
            "receiver_position": long_position,
            "target_id": close_id,
            "alternative_id": continue_id,
        },
    )


@torch.no_grad()
def run_successor_conversion_rows(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_k: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure and causally patch attention-to-MLP successor conversion."""
    logit_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    model_device = next(model.parameters()).device
    k_values = range(cfg.count_min, cfg.count_max)
    progress = tqdm(
        total=len(k_values) * int(examples_per_k) * 2,
        desc="v10 successor attention-to-MLP conversion",
    )

    for k in k_values:
        for example_index in range(int(examples_per_k)):
            seed = cfg.seed + 981_000 + k * 1_000 + example_index
            short_example, long_example = successor_nested_example_pair(cfg, vocab, k, seed)
            short_item = render(short_example, vocab, "thinking")
            long_item = render(long_example, vocab, "thinking")
            short_position = int(short_item.spans.trace_marker_positions[k - 1])
            long_position = int(long_item.spans.trace_marker_positions[k - 1])
            if short_position != long_position:
                raise AssertionError("nested successor query positions must align")
            short_ids = torch.tensor([short_item.input_ids], dtype=torch.long, device=model_device)
            long_ids = torch.tensor([long_item.input_ids], dtype=torch.long, device=model_device)
            short_logits, short_states = capture_sublayer_states(model, short_ids, short_position)
            long_logits, long_states = capture_sublayer_states(model, long_ids, long_position)

            for spec in _direction_specs(
                vocab,
                k,
                short_ids,
                long_ids,
                short_position,
                long_position,
                short_logits,
                long_logits,
                short_states,
                long_states,
            ):
                clean_margin = _two_token_margin(
                    spec["clean_logits"], spec["target_id"], spec["alternative_id"]
                )
                corrupt_margin = _two_token_margin(
                    spec["corrupt_logits"], spec["target_id"], spec["alternative_id"]
                )
                for layer in range(cfg.n_layer):
                    for state_kind in ("clean", "corrupt"):
                        states = (
                            spec["clean_states"]
                            if state_kind == "clean"
                            else spec["receiver_states"]
                        )
                        for stage in RESIDUAL_STAGES:
                            logit_rows.append(
                                {
                                    "decision_k": k,
                                    "count_bin": count_bin(k),
                                    "example_index": example_index,
                                    "direction": spec["direction"],
                                    "layer": layer,
                                    "state_kind": state_kind,
                                    "stage": stage,
                                    "stage_type": "residual_logit_lens",
                                    "target_margin": _residual_logit_margin(
                                        model,
                                        states[(layer, stage)],
                                        spec["target_id"],
                                        spec["alternative_id"],
                                    ),
                                }
                            )
                        for stage in COMPONENT_STAGES:
                            logit_rows.append(
                                {
                                    "decision_k": k,
                                    "count_bin": count_bin(k),
                                    "example_index": example_index,
                                    "direction": spec["direction"],
                                    "layer": layer,
                                    "state_kind": state_kind,
                                    "stage": stage,
                                    "stage_type": "direct_component_unembedding",
                                    "target_margin": _component_logit_margin(
                                        model,
                                        states[(layer, stage)],
                                        spec["target_id"],
                                        spec["alternative_id"],
                                    ),
                                }
                            )

                    for intervention in INTERVENTIONS:
                        patched_logits, patched_mlp = patched_sublayer_forward(
                            model,
                            spec["receiver_ids"],
                            int(spec["receiver_position"]),
                            spec["clean_states"],
                            spec["receiver_states"],
                            layer,
                            intervention,
                        )
                        patched_margin = _two_token_margin(
                            patched_logits, spec["target_id"], spec["alternative_id"]
                        )
                        denominator = clean_margin - corrupt_margin
                        patch_rows.append(
                            {
                                "decision_k": k,
                                "count_bin": count_bin(k),
                                "example_index": example_index,
                                "direction": spec["direction"],
                                "layer": layer,
                                "intervention": intervention,
                                "clean_margin": clean_margin,
                                "corrupt_margin": corrupt_margin,
                                "patched_margin": patched_margin,
                                "normalized_recovery": (
                                    (patched_margin - corrupt_margin) / denominator
                                    if abs(denominator) > 1e-8
                                    else float("nan")
                                ),
                                "clean_target_correct": float(clean_margin > 0),
                                "corrupt_target_correct": float(corrupt_margin > 0),
                                "patched_target_correct": float(patched_margin > 0),
                                "patched_mlp_direct_margin": _component_logit_margin(
                                    model,
                                    patched_mlp,
                                    spec["target_id"],
                                    spec["alternative_id"],
                                ),
                            }
                        )
                progress.update(1)
    progress.close()
    return pd.DataFrame(logit_rows), pd.DataFrame(patch_rows)


def summarize_successor_logit_lens(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(
            ["direction", "count_bin", "layer", "stage", "stage_type", "state_kind"],
            as_index=False,
        )
        .agg(n_pairs=("decision_k", "size"), target_margin=("target_margin", "mean"))
    )
    clean = summary[summary.state_kind == "clean"].drop(columns="state_kind").rename(
        columns={"target_margin": "clean_margin"}
    )
    corrupt = summary[summary.state_kind == "corrupt"].drop(columns="state_kind").rename(
        columns={"target_margin": "corrupt_margin", "n_pairs": "n_corrupt_pairs"}
    )
    merged = clean.merge(
        corrupt,
        on=["direction", "count_bin", "layer", "stage", "stage_type"],
        how="outer",
    )
    merged["evidence_gap"] = merged["clean_margin"] - merged["corrupt_margin"]
    return merged.sort_values(["direction", "count_bin", "layer", "stage_type", "stage"])


def summarize_successor_conversion(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["direction", "count_bin", "layer", "intervention"], as_index=False)
        .agg(
            n_pairs=("decision_k", "size"),
            clean_margin=("clean_margin", "mean"),
            corrupt_margin=("corrupt_margin", "mean"),
            patched_margin=("patched_margin", "mean"),
            normalized_recovery=("normalized_recovery", "mean"),
            clean_target_correct=("clean_target_correct", "mean"),
            corrupt_target_correct=("corrupt_target_correct", "mean"),
            patched_target_correct=("patched_target_correct", "mean"),
            patched_mlp_direct_margin=("patched_mlp_direct_margin", "mean"),
        )
        .sort_values(["direction", "count_bin", "layer", "intervention"])
        .reset_index(drop=True)
    )


def run_successor_conversion(
    run_dir: str | Path,
    *,
    examples_per_k: int = 2,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "successor_conversion"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {
        "successor_stage_logit_lens": tables / "successor_stage_logit_lens.csv",
        "successor_stage_logit_lens_summary": tables / "successor_stage_logit_lens_summary.csv",
        "successor_sublayer_patching": tables / "successor_sublayer_patching.csv",
        "successor_sublayer_patching_summary": tables / "successor_sublayer_patching_summary.csv",
    }
    if not overwrite and all(path.exists() for path in paths.values()):
        return {name: pd.read_csv(path) for name, path in paths.items()}

    cfg, vocab = load_run(run_dir, device=device)
    model = load_final_model(cfg, vocab, run_dir, "thinking")
    logit_detail, patch_detail = run_successor_conversion_rows(
        model,
        cfg,
        vocab,
        examples_per_k=examples_per_k,
    )
    outputs = {
        "successor_stage_logit_lens": logit_detail,
        "successor_stage_logit_lens_summary": summarize_successor_logit_lens(logit_detail),
        "successor_sublayer_patching": patch_detail,
        "successor_sublayer_patching_summary": summarize_successor_conversion(patch_detail),
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "device": cfg.device,
        "examples_per_k": int(examples_per_k),
        "decision_k_min": int(cfg.count_min),
        "decision_k_max": int(cfg.count_max - 1),
        "directions": [DIRECTION_CONTINUE, DIRECTION_CLOSE],
        "residual_stages": list(RESIDUAL_STAGES),
        "component_stages": list(COMPONENT_STAGES),
        "interventions": list(INTERVENTIONS),
        "query": "teacher-forced trace marker M_k",
        "pair_definition": (
            "count-k and count-(k+1) prompts share noise, first-k needles, and trace through M_k"
        ),
        "residual_logit_lens": "ln_f(state) followed by tied unembedding",
        "component_margin": "linear unembedding of additive component without ln_f",
        "autoregressive_rollout": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
