from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .data import IGNORE_INDEX, RenderedExample, render_example, sample_example
from .model import causal_lm_loss, make_model
from .vocab import Vocab


def learning_rate_at_step(step: int, train_cfg: dict[str, Any]) -> float:
    base_lr = float(train_cfg["lr"])
    warmup_steps = int(train_cfg["warmup_steps"])
    train_steps = int(train_cfg["train_steps"])
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def collate_rendered(
    rendered: list[RenderedExample],
    vocab: Vocab,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in rendered)
    input_ids = torch.full((len(rendered), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(rendered), max_len), IGNORE_INDEX, dtype=torch.long)
    for row_idx, item in enumerate(rendered):
        input_ids[row_idx, : len(item.input_ids)] = torch.tensor(item.input_ids, dtype=torch.long)
        labels[row_idx, : len(item.labels)] = torch.tensor(item.labels, dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def sample_training_batch(train_cfg: dict[str, Any], vocab: Vocab, rng: random.Random, cfg: dict[str, Any]) -> list[RenderedExample]:
    rendered: list[RenderedExample] = []
    for _ in range(int(train_cfg["batch_size"])):
        example = sample_example(
            int(train_cfg["seq_len"]),
            rng,
            int(train_cfg["count_min"]),
            int(train_cfg["count_max"]),
        )
        variant = "thinking" if rng.random() < float(train_cfg["thinking_fraction"]) else "nonthinking"
        rendered.append(
            render_example(
                example,
                variant,
                vocab,
                trace_indices=bool(cfg["trace_indices"]),
            )
        )
    return rendered


def _component_mask(rendered: list[RenderedExample], max_ce_len: int, component: str) -> torch.Tensor:
    mask = torch.zeros((len(rendered), max_ce_len), dtype=torch.bool)
    for row_idx, item in enumerate(rendered):
        positions: list[int] = []
        if component == "thinking_trace" and item.variant == "thinking":
            positions = item.spans.trace_token_positions + [item.spans.think_close_pos]
        elif component == "thinking_final_count" and item.variant == "thinking":
            positions = [item.spans.count_pos]
        elif component == "nonthinking_final_count" and item.variant == "nonthinking":
            positions = [item.spans.count_pos]
        elif component == "nonthinking_close" and item.variant == "nonthinking":
            positions = [item.spans.think_close_pos]
        for target_pos in positions:
            ce_pos = target_pos - 1
            if 0 <= ce_pos < max_ce_len:
                mask[row_idx, ce_pos] = True
    return mask


def _masked_mean(ce: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(ce.device)
    if not mask.any():
        return float("nan")
    return float(ce[mask].mean().detach().cpu())


def save_checkpoint(model, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(model, path: str | Path, device: str | torch.device):
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state_dict"])
    return model


def checkpoint_steps(run_dir: Path, train_steps: int) -> list[tuple[int, Path]]:
    ckpt_dir = run_dir / "checkpoints"
    steps: list[tuple[int, Path]] = []
    for path in sorted(ckpt_dir.glob("step_*.pt")):
        try:
            step = int(path.stem.split("_", 1)[1])
        except ValueError:
            continue
        steps.append((step, path))
    final_path = ckpt_dir / "final.pt"
    if final_path.exists() and not any(step == int(train_steps) for step, _ in steps):
        steps.append((int(train_steps), final_path))
    return sorted(steps, key=lambda item: item[0])


def train_model(cfg: dict[str, Any], vocab: Vocab, run_dir: Path) -> pd.DataFrame:
    train_cfg = cfg["train"]
    device = cfg["device"]
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(train_cfg["seed"]))
    rng = random.Random(int(train_cfg["seed"]))
    model = make_model(cfg["model"], device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    rows: list[dict[str, Any]] = []
    progress = tqdm(range(1, int(train_cfg["train_steps"]) + 1), desc="v5 train", leave=True)
    for step in progress:
        lr = learning_rate_at_step(step - 1, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rendered = sample_training_batch(train_cfg, vocab, rng, cfg)
        input_ids, labels = collate_rendered(rendered, vocab, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask=(input_ids != vocab.pad_id).long())
        loss, ce = causal_lm_loss(out.logits, labels, ignore_index=IGNORE_INDEX)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()

        if step == 1 or step % int(train_cfg["log_every"]) == 0:
            max_ce_len = ce.size(1)
            row = {
                "step": step,
                "loss_total": float(loss.detach().cpu()),
                "loss_thinking_trace": _masked_mean(ce, _component_mask(rendered, max_ce_len, "thinking_trace")),
                "loss_thinking_final_count": _masked_mean(ce, _component_mask(rendered, max_ce_len, "thinking_final_count")),
                "loss_nonthinking_close": _masked_mean(ce, _component_mask(rendered, max_ce_len, "nonthinking_close")),
                "loss_nonthinking_final_count": _masked_mean(ce, _component_mask(rendered, max_ce_len, "nonthinking_final_count")),
                "lr": lr,
            }
            rows.append(row)
            progress.set_postfix(loss=f"{row['loss_total']:.4f}", lr=f"{lr:.2e}")

        should_save = step == int(train_cfg["train_steps"])
        if int(train_cfg["checkpoint_every"]) > 0 and step % int(train_cfg["checkpoint_every"]) == 0:
            should_save = True
        if int(train_cfg["eval_every"]) > 0 and step % int(train_cfg["eval_every"]) == 0:
            should_save = True
        if should_save:
            save_checkpoint(model, ckpt_dir / f"step_{step}.pt", {"step": step, "config": cfg})

    save_checkpoint(model, ckpt_dir / "final.pt", {"step": int(train_cfg["train_steps"]), "config": cfg})
    train_log = pd.DataFrame(rows)
    train_log.to_csv(tables / "train_log.csv", index=False)
    (ckpt_dir / "metadata.json").write_text(
        json.dumps({"final_checkpoint": str(ckpt_dir / "final.pt"), "train_steps": int(train_cfg["train_steps"])}, indent=2),
        encoding="utf-8",
    )
    return train_log
