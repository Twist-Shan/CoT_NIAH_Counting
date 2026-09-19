from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .data import sample_example
from .model import make_model, weighted_next_token_loss
from .objectives import build_training_weights
from .render import render_for_model
from .vocab import Vocab


def collate_training_batch(
    examples,
    model_type: str,
    vocab: Vocab,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = [render_for_model(example, vocab, model_type) for example in examples]
    max_len = max(len(item.tokens) for item in rendered)
    input_ids = torch.full((len(rendered), max_len), vocab.pad_id, dtype=torch.long)
    weights = torch.zeros((len(rendered), max_len), dtype=torch.float32)
    for row_idx, item in enumerate(rendered):
        input_ids[row_idx, : len(item.tokens)] = torch.tensor(item.tokens, dtype=torch.long)
        weights[row_idx, : len(item.tokens)] = build_training_weights(item.tokens, item.spans, model_type)
    return input_ids.to(device), weights.to(device)


def learning_rate_at_step(step: int, cfg: dict[str, Any]) -> float:
    base_lr = float(cfg["learning_rate"])
    warmup_steps = int(cfg["warmup_steps"])
    train_steps = int(cfg["train_steps"])
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def save_checkpoint(model, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(model, path: Path, device: str | torch.device):
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state_dict"])
    return model


def train_model(
    cfg: dict[str, Any],
    model_type: str,
    seed: int,
    vocab: Vocab,
    run_dir: Path,
    skip_completed: bool = True,
) -> tuple[Path, pd.DataFrame]:
    device = cfg["device"]
    model_dir = run_dir / "checkpoints" / f"{model_type}_seed{seed}"
    final_path = model_dir / "final.pt"
    train_log_path = run_dir / "metrics" / f"train_log_{model_type}_seed{seed}.csv"
    if skip_completed and final_path.exists() and train_log_path.exists():
        return final_path, pd.read_csv(train_log_path)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = make_model(cfg, device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=tuple(cfg["betas"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    rows: list[dict[str, Any]] = []
    progress = tqdm(range(1, int(cfg["train_steps"]) + 1), desc=f"{model_type} seed={seed}", leave=True)
    for step in progress:
        lr = learning_rate_at_step(step - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [sample_example(int(cfg["train_seq_len"]), rng) for _ in range(int(cfg["batch_size"]))]
        input_ids, weights = collate_training_batch(batch, model_type, vocab, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids)
        loss, ce = weighted_next_token_loss(out.logits, input_ids, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
        optimizer.step()

        if step % int(cfg["log_every"]) == 0 or step == 1:
            with torch.no_grad():
                final_mask = torch.zeros_like(weights[:, :-1], dtype=torch.bool)
                eos_mask = torch.zeros_like(weights[:, :-1], dtype=torch.bool)
                trace_mask = torch.zeros_like(weights[:, :-1], dtype=torch.bool)
                for row_idx, example in enumerate(batch):
                    rendered = render_for_model(example, vocab, model_type)
                    final_mask[row_idx, rendered.spans.ans_pos] = True
                    eos_mask[row_idx, rendered.spans.final_count_pos] = True
                    if model_type == "thinking":
                        if rendered.spans.think_open_pos is not None:
                            trace_mask[row_idx, rendered.spans.think_open_pos] = True
                        for pos in rendered.spans.trace_token_positions:
                            if pos < trace_mask.size(1):
                                trace_mask[row_idx, pos] = True
                def mean_mask(mask):
                    return float(ce[mask].mean().detach().cpu()) if mask.any() else float("nan")
                row = {
                    "step": step,
                    "model_type": model_type,
                    "seed": seed,
                    "train_total_loss": float(loss.detach().cpu()),
                    "train_final_count_ce": mean_mask(final_mask),
                    "train_trace_ce": mean_mask(trace_mask),
                    "train_eos_ce": mean_mask(eos_mask),
                    "learning_rate": lr,
                }
                rows.append(row)
                progress.set_postfix(loss=f"{row['train_total_loss']:.4f}", lr=f"{lr:.2e}")

        save_step = False
        if int(cfg["checkpoint_every"]) > 0 and step % int(cfg["checkpoint_every"]) == 0:
            save_step = True
        if int(cfg["eval_every"]) > 0 and step % int(cfg["eval_every"]) == 0:
            save_step = True
        if save_step:
            save_checkpoint(
                model,
                model_dir / f"step_{step}.pt",
                {"model_type": model_type, "seed": seed, "step": step, "config": cfg},
            )

    save_checkpoint(model, final_path, {"model_type": model_type, "seed": seed, "step": int(cfg["train_steps"]), "config": cfg})
    train_log = pd.DataFrame(rows)
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    train_log.to_csv(train_log_path, index=False)
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_type": model_type, "seed": seed, "final_checkpoint": str(final_path)}, indent=2),
        encoding="utf-8",
    )
    return final_path, train_log


def checkpoint_steps_for_model(run_dir: Path, model_type: str, seed: int, train_steps: int) -> list[tuple[int, Path]]:
    model_dir = run_dir / "checkpoints" / f"{model_type}_seed{seed}"
    steps: list[tuple[int, Path]] = []
    for path in sorted(model_dir.glob("step_*.pt")):
        try:
            step = int(path.stem.split("_", 1)[1])
        except ValueError:
            continue
        steps.append((step, path))
    final_path = model_dir / "final.pt"
    if final_path.exists() and not any(step == train_steps for step, _ in steps):
        steps.append((train_steps, final_path))
    return sorted(steps, key=lambda item: item[0])
