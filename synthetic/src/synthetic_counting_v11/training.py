from __future__ import annotations

import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import ExperimentConfig, config_from_dict
from .data import (
    Example,
    FixedExamplePool,
    Vocab,
    WindowWithoutReplacementSampler,
    balanced_examples,
    collate,
    component_loss_values,
    count_sampling_probabilities,
    count_prediction,
    load_or_create_fixed_pool,
    make_example,
    render,
    split_target_window_starts,
    shifted_token_losses,
)
from .model import TinyPositionCausalLM, build_model


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sync_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(target)


def learning_rate(cfg: ExperimentConfig, step: int) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _autocast_context(cfg: ExperimentConfig):
    enabled = (
        cfg.precision == "bf16"
        and str(cfg.device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _paired_model(
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
) -> TinyPositionCausalLM:
    """Pair all shared initial weights across APE/RoPE/RPE variants."""

    torch.manual_seed(cfg.seed + 17)
    canonical = build_model(cfg, vocab, "rope", "cpu")
    canonical_state = canonical.state_dict()
    torch.manual_seed(cfg.seed + 17)
    model = build_model(cfg, vocab, position_encoding, "cpu")
    state = model.state_dict()
    for name, value in canonical_state.items():
        if name in state and state[name].shape == value.shape:
            state[name] = value.clone()
    model.load_state_dict(state)
    return model.to(cfg.device)


def _checkpoint_root(run_dir: Path, position_encoding: str, mode: str) -> Path:
    return run_dir / "checkpoints" / position_encoding / mode


def _latest_checkpoint(root: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("step_*/checkpoint.pt"):
        try:
            candidates.append((int(path.parent.name.removeprefix("step_")), path))
        except ValueError:
            continue
    return max(candidates, default=None, key=lambda item: item[0])


def _cpu_byte_rng_state(state: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize RNG state loaded with a CUDA map location for PyTorch setters."""

    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(state).__name__}")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng_states(payload: dict[str, Any], rng: random.Random) -> None:
    rng.setstate(payload["python_rng_state"])
    torch.set_rng_state(
        _cpu_byte_rng_state(payload["torch_rng_state"], name="torch_rng_state")
    )
    cuda_states = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(state, name=f"cuda_rng_state_all[{index}]")
                for index, state in enumerate(cuda_states)
            ]
        )


def _save_checkpoint(
    model: TinyPositionCausalLM,
    optimizer: AdamW,
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    mode: str,
    step: int,
    rng: random.Random,
    run_dir: Path,
    sync_run_dir: Path | None,
    *,
    label: str | None = None,
    window_sampler: WindowWithoutReplacementSampler | None = None,
) -> Path:
    root = _checkpoint_root(run_dir, position_encoding, mode)
    directory = root / (label or f"step_{step:06d}")
    path = directory / "checkpoint.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "position_encoding": position_encoding,
        "mode": mode,
        "step": int(step),
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "vocab_fingerprint": vocab.fingerprint,
        "window_sampler_state": (
            window_sampler.state_dict() if window_sampler is not None else None
        ),
    }
    _atomic_torch_save(payload, path)
    latest = root / "latest.json"
    latest.write_text(
        json.dumps({"step": int(step), "checkpoint": str(path.relative_to(run_dir))}, indent=2),
        encoding="utf-8",
    )
    if sync_run_dir is not None:
        for source in (path, latest):
            target = sync_run_dir / source.relative_to(run_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return path


def _sample_training_batch(
    cfg: ExperimentConfig,
    vocab: Vocab,
    mode: str,
    rng: random.Random,
    fixed_pool: FixedExamplePool | None,
    window_sampler: WindowWithoutReplacementSampler | None,
) -> list:
    if window_sampler is not None:
        examples = window_sampler.sample(cfg.batch_size)
    elif fixed_pool is not None:
        examples = fixed_pool.sample(rng, cfg.batch_size, vocab)
    else:
        examples = [make_example(cfg, vocab, rng) for _ in range(cfg.batch_size)]
    return [render(example, vocab, mode) for example in examples]


def _parse_generation(tokens: list[str], mode: str, vocab: Vocab, example: Example) -> dict[str, Any]:
    predicted_count = None
    if "<Ans>" in tokens:
        answer_index = tokens.index("<Ans>")
        if answer_index + 1 < len(tokens) and tokens[answer_index + 1] in vocab.numbers:
            predicted_count = vocab.numbers.index(tokens[answer_index + 1]) + 1
    trace: list[str] = []
    if mode == "thinking" and "<Think>" in tokens:
        start = tokens.index("<Think>") + 1
        end = tokens.index("</Think>") if "</Think>" in tokens[start:] else len(tokens)
        trace = tokens[start:end]
    expected_trace = [
        token
        for k, marker in enumerate(example.needle_markers, start=1)
        for token in (vocab.number_token(k), marker)
    ]
    # Trace grammar is always index, marker, index, marker, ... . In v16 the
    # marker is a native Shakespeare character token rather than an <M*> token.
    generated_markers = trace[1::2]
    marker_matches = sum(
        int(index < len(generated_markers) and generated_markers[index] == marker)
        for index, marker in enumerate(example.needle_markers)
    )
    return {
        "ar_pred_count": predicted_count,
        "ar_accuracy": float(predicted_count == example.count),
        "ar_abs_error": abs(predicted_count - example.count) if predicted_count is not None else np.nan,
        "trace_exact": float(trace == expected_trace) if mode == "thinking" else np.nan,
        "trace_marker_recall": (
            marker_matches / max(1, example.count) if mode == "thinking" else np.nan
        ),
        "generated_tokens": " ".join(tokens),
    }


@torch.no_grad()
def autoregressive_evaluate(
    model: TinyPositionCausalLM,
    cfg: ExperimentConfig,
    vocab: Vocab,
    mode: str,
    examples: list[Example],
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prefixes = []
        for example in chunk:
            item = render(example, vocab, mode)
            stop = item.spans.ans_pos + 1 if mode == "nonthinking" else item.spans.think_pos + 1
            prefixes.append(item.tokens[:stop])
        generated = torch.tensor([vocab.encode(prefix) for prefix in prefixes], device=cfg.device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = 2 if mode == "nonthinking" else 2 * cfg.count_max + 5
        for _ in range(max_new_tokens):
            next_ids = model(input_ids=generated).logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            for row_index, token_id in enumerate(next_ids.tolist()):
                if done[row_index]:
                    continue
                parsed = _parse_generation(vocab.decode(generated[row_index].tolist()), mode, vocab, chunk[row_index])
                if parsed["ar_pred_count"] is not None or token_id == vocab.eos_id:
                    done[row_index] = True
            if bool(done.all()):
                break
        for row_index, example in enumerate(chunk):
            parsed = _parse_generation(vocab.decode(generated[row_index].tolist()), mode, vocab, example)
            rows.append(
                {
                    "count": example.count,
                    "count_bin": cfg.count_bin(example.count),
                    **parsed,
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(
    model: TinyPositionCausalLM,
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    mode: str,
    examples: list[Example],
    *,
    step: int,
    run_ar: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    component_parts: dict[str, list[float]] = {}
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render(example, vocab, mode) for example in chunk]
        ids, labels, attention_mask = collate(rendered, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=attention_mask)
        total_loss, token_losses = shifted_token_losses(output.logits, labels)
        component_parts.setdefault("total", []).append(float(total_loss.detach().cpu()))
        for name, value in component_loss_values(token_losses, rendered).items():
            component_parts.setdefault(name, []).append(value)
        for row_index, (example, item) in enumerate(zip(chunk, rendered)):
            predicted, _, _ = count_prediction(output.logits[row_index, item.spans.ans_pos], vocab)
            marker_correct: list[float] = []
            index_correct: list[float] = []
            if mode == "thinking":
                for index, marker_position in enumerate(item.spans.trace_marker_positions):
                    query_position = item.spans.trace_index_positions[index]
                    marker_correct.append(
                        float(int(output.logits[row_index, query_position].argmax()) == item.input_ids[marker_position])
                    )
                for index, index_position in enumerate(item.spans.trace_index_positions):
                    query_position = item.spans.think_pos if index == 0 else item.spans.trace_marker_positions[index - 1]
                    index_correct.append(
                        float(int(output.logits[row_index, query_position].argmax()) == item.input_ids[index_position])
                    )
            rows.append(
                {
                    "example_id": int(start + row_index),
                    "step": int(step),
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "count": example.count,
                    "count_bin": cfg.count_bin(example.count),
                    "tf_pred_count": predicted,
                    "tf_final_accuracy": float(predicted == example.count),
                    "tf_trace_marker_accuracy": float(np.mean(marker_correct)) if marker_correct else np.nan,
                    "tf_trace_index_accuracy": float(np.mean(index_correct)) if index_correct else np.nan,
                }
            )
    loss_row = {
        "step": int(step),
        "position_encoding": position_encoding,
        "mode": mode,
        **{f"eval_{name}_loss": float(np.mean(values)) for name, values in component_parts.items()},
    }
    ar_frame = pd.DataFrame()
    if run_ar:
        ar_examples = balanced_examples(
            cfg,
            vocab,
            cfg.ar_examples_per_count,
            cfg.seed + 90_000 + step,
        )
        ar_frame = autoregressive_evaluate(model, cfg, vocab, mode, ar_examples)
        ar_frame.insert(0, "mode", mode)
        ar_frame.insert(0, "position_encoding", position_encoding)
        ar_frame.insert(0, "step", int(step))
    return pd.DataFrame(rows), pd.DataFrame([loss_row]), ar_frame


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _append_unique(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.empty:
        return
    combined = pd.concat((_read_table(path), frame), ignore_index=True)
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    _atomic_csv(combined, path)


def train_variant(
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    mode: str,
    run_dir: Path,
    eval_examples: list[Example],
    fixed_pool: FixedExamplePool | None,
    *,
    sync_run_dir: Path | None,
    skip_completed: bool,
) -> None:
    root = _checkpoint_root(run_dir, position_encoding, mode)
    final_path = root / "final" / "checkpoint.pt"
    if skip_completed and final_path.exists():
        print(f"[skip] {position_encoding}/{mode}: final checkpoint exists", flush=True)
        return
    model = _paired_model(cfg, vocab, position_encoding)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    # Pair the prompt stream across output modes. Rendering still changes the
    # supervised completion, but each optimization step sees the same haystack.
    rng = random.Random(cfg.seed)
    window_sampler = (
        WindowWithoutReplacementSampler(
            cfg,
            vocab,
            split="train",
            seed=cfg.seed + 31_000,
        )
        if cfg.training_data_mode == "split_window_index"
        else None
    )
    start_step = 0
    latest = _latest_checkpoint(root) if skip_completed else None
    if latest is not None:
        start_step, path = latest
        payload = torch.load(path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _restore_rng_states(payload, rng)
        if window_sampler is not None:
            sampler_state = payload.get("window_sampler_state")
            if sampler_state is None:
                raise ValueError(
                    "Cannot resume v16.1 exactly: checkpoint has no window_sampler_state. "
                    "Use a fresh run name or remove the incomplete local checkpoint."
                )
            window_sampler.load_state_dict(sampler_state)
        print(f"[resume] {position_encoding}/{mode} from step {start_step}", flush=True)

    train_path = run_dir / "tables" / "train_metrics.csv"
    eval_path = run_dir / "tables" / "eval_detail.csv"
    eval_loss_path = run_dir / "tables" / "eval_losses.csv"
    ar_path = run_dir / "tables" / "autoregressive_detail.csv"
    progress = tqdm(
        range(start_step + 1, cfg.train_steps + 1),
        desc=f"{cfg.version} {position_encoding}/{mode}",
        initial=start_step,
        total=cfg.train_steps,
    )
    for step in progress:
        model.train()
        rendered = _sample_training_batch(
            cfg,
            vocab,
            mode,
            rng,
            fixed_pool,
            window_sampler,
        )
        ids, labels, attention_mask = collate(rendered, vocab, cfg.device)
        with _autocast_context(cfg):
            output = model(input_ids=ids, attention_mask=attention_mask)
            loss, token_losses = shifted_token_losses(output.logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        rate = learning_rate(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if step % cfg.log_every == 0 or step == 1 or step == cfg.train_steps:
            components = component_loss_values(token_losses, rendered)
            batch_counts = np.asarray([item.count for item in rendered], dtype=np.float64)
            row = pd.DataFrame(
                [
                    {
                        "step": step,
                        "position_encoding": position_encoding,
                        "mode": mode,
                        "train_total_loss": float(loss.detach().cpu()),
                        "learning_rate": rate,
                        "gradient_norm": gradient_norm,
                        "batch_count_mean": float(batch_counts.mean()),
                        **{
                            f"batch_fraction_count_{lo}_{hi}": float(
                                np.mean((batch_counts >= lo) & (batch_counts <= hi))
                            )
                            for lo, hi in cfg.count_bins
                        },
                        **{f"train_{name}_loss": value for name, value in components.items()},
                    }
                ]
            )
            _append_unique(train_path, row, ["position_encoding", "mode", "step"])
            progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{rate:.1e}")
        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            detail, losses, ar = evaluate_model(
                model,
                cfg,
                vocab,
                position_encoding,
                mode,
                eval_examples,
                step=step,
                run_ar=(step % cfg.ar_eval_every == 0 or step == cfg.train_steps),
            )
            _append_unique(
                eval_path,
                detail,
                ["position_encoding", "mode", "step", "example_id"],
            )
            _append_unique(eval_loss_path, losses, ["position_encoding", "mode", "step"])
            if not ar.empty:
                ar.insert(3, "row_id", np.arange(len(ar)))
                _append_unique(ar_path, ar, ["position_encoding", "mode", "step", "row_id"])
        if step % cfg.checkpoint_every == 0:
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                vocab,
                position_encoding,
                mode,
                step,
                rng,
                run_dir,
                sync_run_dir,
                window_sampler=window_sampler,
            )
    _save_checkpoint(
        model,
        optimizer,
        cfg,
        vocab,
        position_encoding,
        mode,
        cfg.train_steps,
        rng,
        run_dir,
        sync_run_dir,
        label="final",
        window_sampler=window_sampler,
    )


def summarize_learning_tables(run_dir: Path) -> None:
    detail = _read_table(run_dir / "tables" / "eval_detail.csv")
    if not detail.empty:
        by_count = (
            detail.groupby(["position_encoding", "mode", "step", "count", "count_bin"], as_index=False)
            .agg(
                tf_final_accuracy=("tf_final_accuracy", "mean"),
                tf_trace_marker_accuracy=("tf_trace_marker_accuracy", "mean"),
                tf_trace_index_accuracy=("tf_trace_index_accuracy", "mean"),
            )
        )
        by_bin = (
            detail.groupby(["position_encoding", "mode", "step", "count_bin"], as_index=False)
            .agg(
                tf_final_accuracy=("tf_final_accuracy", "mean"),
                tf_trace_marker_accuracy=("tf_trace_marker_accuracy", "mean"),
                tf_trace_index_accuracy=("tf_trace_index_accuracy", "mean"),
            )
        )
        _atomic_csv(by_count, run_dir / "tables" / "eval_by_count.csv")
        _atomic_csv(by_bin, run_dir / "tables" / "eval_by_bin.csv")
    ar = _read_table(run_dir / "tables" / "autoregressive_detail.csv")
    if not ar.empty:
        summary = (
            ar.groupby(["position_encoding", "mode", "step", "count_bin"], as_index=False)
            .agg(
                ar_final_accuracy=("ar_accuracy", "mean"),
                ar_abs_error=("ar_abs_error", "mean"),
                trace_exact=("trace_exact", "mean"),
                trace_marker_recall=("trace_marker_recall", "mean"),
            )
        )
        _atomic_csv(summary, run_dir / "tables" / "autoregressive_by_bin.csv")


def train_all_models(
    cfg: ExperimentConfig,
    vocab: Vocab,
    run_dir: Path,
    *,
    sync_run_dir: Path | None,
    skip_completed: bool,
) -> None:
    probabilities = count_sampling_probabilities(cfg)
    window_summary = pd.DataFrame()
    if cfg.training_data_mode == "split_window_index":
        rows: list[dict[str, Any]] = []
        for split in ("train", "validation", "test"):
            for (target_token, count), starts in split_target_window_starts(cfg, split).items():
                rows.append(
                    {
                        "split": split,
                        "target_token": target_token,
                        "count": int(count),
                        "candidate_windows": int(len(starts)),
                        "eligible_for_training": bool(
                            split == "train" and len(starts) >= cfg.min_candidate_windows
                        ),
                    }
                )
        window_summary = pd.DataFrame(rows)
        _atomic_csv(window_summary, run_dir / "tables" / "window_index_summary.csv")
        eligible = window_summary[window_summary["eligible_for_training"]]
        totals = eligible.groupby("count")["candidate_windows"].sum()
        denominator = float(totals.sum())
        if denominator <= 0:
            raise ValueError(
                "v16.1 has no eligible training windows. Lower "
                "min_candidate_windows, shorten seq_len, or choose more target characters."
            )
        probabilities = np.asarray(
            [float(totals.get(count, 0)) / denominator for count in range(cfg.count_min, cfg.count_max + 1)]
        )
    distribution = pd.DataFrame(
        {
            "count": np.arange(cfg.count_min, cfg.count_max + 1),
            "probability": probabilities,
            "count_sampling": cfg.count_sampling,
            "power_alpha": cfg.power_alpha,
            "exponential_beta": cfg.exponential_beta,
            "sampling_unit": (
                "eligible (target,count,window_start), without replacement per epoch"
                if cfg.training_data_mode == "split_window_index"
                else "streamed example"
            ),
        }
    )
    distribution["count_bin"] = distribution["count"].map(cfg.count_bin)
    _atomic_csv(distribution, run_dir / "tables" / "training_count_distribution.csv")
    fixed_pool = None
    if cfg.training_data_mode == "fixed":
        fixed_pool = load_or_create_fixed_pool(run_dir / "data" / "fixed_train_dataset.npz", cfg, vocab)
    eval_examples = balanced_examples(
        cfg,
        vocab,
        cfg.eval_examples_per_count,
        cfg.seed + 70_000,
    )
    specifications: list[dict[str, Any]] = []
    for position_encoding, mode in cfg.model_variants:
        probe_model = _paired_model(cfg, vocab, position_encoding)
        specifications.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "parameters": probe_model.parameter_count(),
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "n_inner": cfg.n_inner,
            }
        )
        del probe_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[train] {position_encoding}/{mode}", flush=True)
        train_variant(
            cfg,
            vocab,
            position_encoding,
            mode,
            run_dir,
            eval_examples,
            fixed_pool,
            sync_run_dir=sync_run_dir,
            skip_completed=skip_completed,
        )
    _atomic_csv(pd.DataFrame(specifications), run_dir / "tables" / "model_specifications.csv")
    summarize_learning_tables(run_dir)
    if sync_run_dir is not None:
        sync_tree(run_dir, sync_run_dir)


def load_final_model(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    device: str | torch.device | None = None,
) -> tuple[ExperimentConfig, Vocab, TinyPositionCausalLM]:
    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if device is not None:
        cfg = ExperimentConfig(**{**cfg.__dict__, "device": str(device)})
    vocab = Vocab.load(run_dir / "vocab.json")
    model = build_model(cfg, vocab, position_encoding, cfg.device)
    path = _checkpoint_root(run_dir, position_encoding, mode) / "final" / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {path}")
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return cfg, vocab, model.eval()
