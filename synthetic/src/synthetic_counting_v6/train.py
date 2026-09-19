from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

from .config import MODEL_TYPES, V6Config
from .data import BaseExample, sample_example
from .model import causal_lm_loss, count_logits, make_model, save_pretrained_checkpoint
from .render import render_for_model
from .vocab import Vocab


def collate_rendered(rendered, vocab: Vocab, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in rendered)
    input_ids = torch.full((len(rendered), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(rendered), max_len), -100, dtype=torch.long)
    for row_idx, item in enumerate(rendered):
        input_ids[row_idx, : len(item.input_ids)] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row_idx, : len(item.labels)] = torch.tensor(item.labels, dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def learning_rate_at_step(step: int, cfg: V6Config) -> float:
    base_lr = float(cfg.train.learning_rate)
    warmup = int(cfg.train.warmup_steps)
    total = max(1, int(cfg.train.train_steps))
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(progress * math.pi))


def _batch_examples(cfg: V6Config, rng: random.Random, step: int) -> list[BaseExample]:
    return [
        sample_example(
            cfg.seq_len,
            rng,
            min_count=cfg.min_count,
            max_count=cfg.max_count,
            seed=cfg.seed * 1_000_000 + step * 10_000 + idx,
        )
        for idx in range(int(cfg.train.batch_size))
    ]


def _final_answer_loss(logits: torch.Tensor, rendered, vocab: Vocab) -> torch.Tensor:
    row_indices = torch.arange(logits.size(0), device=logits.device)
    ans_positions = torch.tensor([item.spans.ans_pos for item in rendered], dtype=torch.long, device=logits.device)
    gold = torch.tensor([item.input_ids[item.spans.final_count_pos] for item in rendered], dtype=torch.long, device=logits.device)
    logits_at_ans = logits[row_indices, ans_positions]
    restricted = count_logits(logits_at_ans, vocab)
    gold_offsets = torch.tensor([vocab.count_ids.index(int(token_id)) for token_id in gold.detach().cpu().tolist()], dtype=torch.long, device=logits.device)
    return F.cross_entropy(restricted, gold_offsets)


def train_step(model, optimizer, cfg: V6Config, vocab: Vocab, examples: list[BaseExample], model_type: str, lr: float) -> dict[str, float]:
    for group in optimizer.param_groups:
        group["lr"] = lr
    rendered = [render_for_model(example, vocab, model_type) for example in examples]
    input_ids, labels = collate_rendered(rendered, vocab, cfg.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=(input_ids != vocab.pad_id).long(), use_cache=False)
    loss = causal_lm_loss(out.logits, labels)
    final_loss = _final_answer_loss(out.logits, rendered, vocab)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
    optimizer.step()
    trace_positions = sum(len(item.spans.trace_token_positions) for item in rendered)
    return {
        "train_loss": float(loss.detach().cpu()),
        "train_completion_loss": float(loss.detach().cpu()),
        "train_trace_loss": float(loss.detach().cpu()) if trace_positions else math.nan,
        "train_final_answer_loss": float(final_loss.detach().cpu()),
        "learning_rate": float(lr),
    }


def init_models_and_optimizers(cfg: V6Config, vocab: Vocab) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(int(cfg.seed))
    models = {model_type: make_model(cfg, vocab, cfg.device) for model_type in MODEL_TYPES}
    optimizers = {
        model_type: AdamW(
            model.parameters(),
            lr=float(cfg.train.learning_rate),
            betas=tuple(cfg.train.betas),
            weight_decay=float(cfg.train.weight_decay),
        )
        for model_type, model in models.items()
    }
    return models, optimizers


def final_checkpoint_exists(run_dir: Path) -> bool:
    return all((run_dir / "checkpoints" / "final" / model_type / "config.json").exists() for model_type in MODEL_TYPES)


def save_checkpoint_group(models: dict[str, Any], run_dir: Path, name: str, cfg: V6Config) -> None:
    for model_type, model in models.items():
        save_pretrained_checkpoint(
            model,
            run_dir / "checkpoints" / name / model_type,
            metadata={"model_type": model_type, "config": cfg.to_dict(), "checkpoint_name": name},
        )


def load_final_models(cfg: V6Config, vocab: Vocab, run_dir: Path) -> dict[str, Any]:
    models = {}
    for model_type in MODEL_TYPES:
        path = run_dir / "checkpoints" / "final" / model_type
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"missing v6 checkpoint: {path}")
        model = GPT2LMHeadModel.from_pretrained(path).to(cfg.device)
        model.eval()
        models[model_type] = model
    return models


def train_models(
    cfg: V6Config,
    vocab: Vocab,
    run_dir: Path,
    skip_completed: bool = True,
    eval_examples: list[BaseExample] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_csv = run_dir / "metrics" / "metrics_train.csv"
    if skip_completed and final_checkpoint_exists(run_dir) and train_csv.exists():
        return load_final_models(cfg, vocab, run_dir), pd.read_csv(train_csv)

    from .evaluation import evaluate_all

    models, optimizers = init_models_and_optimizers(cfg, vocab)
    rng = random.Random(int(cfg.seed))
    rows: list[dict[str, Any]] = []
    eval_by_count_frames: list[pd.DataFrame] = []
    eval_by_bin_frames: list[pd.DataFrame] = []
    progress = tqdm(range(1, int(cfg.train.train_steps) + 1), desc="v6 training", leave=True)
    for step in progress:
        lr = learning_rate_at_step(step - 1, cfg)
        examples = _batch_examples(cfg, rng, step)
        step_rows = []
        for model_type in MODEL_TYPES:
            metrics = train_step(models[model_type], optimizers[model_type], cfg, vocab, examples, model_type, lr)
            row = {"step": step, "model_type": model_type, **metrics}
            step_rows.append(row)
            if step == 1 or step % int(cfg.train.log_every) == 0:
                rows.append(row)
        if step % max(1, int(cfg.train.log_every)) == 0:
            progress.set_postfix(
                non=f"{step_rows[0]['train_loss']:.4f}",
                sep=f"{step_rows[1]['train_loss']:.4f}",
                lr=f"{lr:.2e}",
            )
        if cfg.train.save_every and step % int(cfg.train.save_every) == 0:
            save_checkpoint_group(models, run_dir, f"step_{step:06d}", cfg)
        if eval_examples is not None and (step == 1 or step % int(cfg.train.eval_every) == 0 or step == int(cfg.train.train_steps)):
            detail, by_count, by_bin = evaluate_all(
                models,
                eval_examples,
                vocab,
                cfg.device,
                step=step,
                batch_size=max(1, min(64, int(cfg.train.batch_size))),
            )
            eval_by_count_frames.append(by_count)
            eval_by_bin_frames.append(by_bin)
            by_count_df = pd.concat(eval_by_count_frames, ignore_index=True)
            by_bin_df = pd.concat(eval_by_bin_frames, ignore_index=True)
            by_count_df.to_csv(run_dir / "metrics" / "metrics_eval_by_count.csv", index=False)
            by_bin_df.to_csv(run_dir / "metrics" / "metrics_eval_by_bin.csv", index=False)
            by_count_df.to_csv(run_dir / "metrics_eval_by_count.csv", index=False)
            by_bin_df.to_csv(run_dir / "metrics_eval_by_bin.csv", index=False)
            detail.to_csv(run_dir / "metrics" / f"metrics_eval_detail_step_{step:06d}.csv", index=False)
    save_checkpoint_group(models, run_dir, "final", cfg)
    train_df = pd.DataFrame(rows)
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    train_df.to_csv(run_dir / "metrics_train.csv", index=False)
    (run_dir / "checkpoints" / "manifest.json").write_text(
        json.dumps({"final": str(run_dir / "checkpoints" / "final"), "model_types": list(MODEL_TYPES)}, indent=2),
        encoding="utf-8",
    )
    return models, train_df
