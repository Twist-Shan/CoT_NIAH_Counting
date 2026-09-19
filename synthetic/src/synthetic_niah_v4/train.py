from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import MODEL_TYPES, V4Config
from .data import sample_example
from .model import causal_lm_loss, make_model
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


def learning_rate_at_step(step: int, cfg: V4Config) -> float:
    base_lr = float(cfg.train.lr)
    warmup = int(cfg.train.warmup_steps)
    total = max(1, int(cfg.train.steps))
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def checkpoint_path(run_dir: Path, model_type: str, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{model_type}_seed{seed}" / "final.pt"


def _expected_model_signature(cfg: V4Config, vocab: Vocab) -> dict[str, int]:
    return {
        "vocab_size": len(vocab.id_to_token),
        "n_layer": int(cfg.model.n_layer),
        "n_head": int(cfg.model.n_head),
        "n_embd": int(cfg.model.n_embd),
        "n_positions": int(cfg.model.n_positions),
    }


def _metadata_model_signature(metadata: dict[str, Any]) -> dict[str, int] | None:
    config = metadata.get("config") or {}
    model = config.get("model") or {}
    if not model:
        return None
    return {
        "vocab_size": int(model.get("vocab_size", 90)),
        "n_layer": int(model.get("n_layer", -1)),
        "n_head": int(model.get("n_head", -1)),
        "n_embd": int(model.get("n_embd", -1)),
        "n_positions": int(model.get("n_positions", -1)),
    }


def checkpoint_is_compatible(path: Path, cfg: V4Config, vocab: Vocab) -> bool:
    if not path.exists():
        return False
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        metadata = obj.get("metadata") or {}
        saved = _metadata_model_signature(metadata)
        expected = _expected_model_signature(cfg, vocab)
        if saved is not None:
            return saved == expected
        state = obj.get("model_state_dict") or {}
        emb = state.get("gpt2.transformer.wte.weight")
        if emb is None:
            return False
        return tuple(emb.shape) == (expected["vocab_size"], expected["n_embd"])
    except Exception as exc:
        print(f"[v4] ignoring unreadable checkpoint {path}: {exc}", flush=True)
        return False


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


def train_model(cfg: V4Config, model_type: str, seed: int, vocab: Vocab, run_dir: Path, skip_completed: bool = True) -> pd.DataFrame:
    final_path = checkpoint_path(run_dir, model_type, seed)
    log_path = run_dir / "metrics" / f"train_log_{model_type}_seed{seed}.csv"
    if skip_completed and final_path.exists() and log_path.exists() and checkpoint_is_compatible(final_path, cfg, vocab):
        return pd.read_csv(log_path)
    if skip_completed and final_path.exists() and not checkpoint_is_compatible(final_path, cfg, vocab):
        print(f"[v4 train] existing checkpoint is incompatible with current config; retraining: {final_path}", flush=True)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = make_model(cfg, len(vocab.id_to_token), cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    rows: list[dict[str, Any]] = []
    progress = tqdm(range(1, int(cfg.train.steps) + 1), desc=f"v4 {model_type} seed={seed}", leave=True)
    for step in progress:
        lr = learning_rate_at_step(step - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [
            sample_example(cfg.seq_len, rng, cfg.count_min, cfg.count_max, seed=seed * 1_000_000 + step * 1000 + idx)
            for idx in range(int(cfg.train.batch_size))
        ]
        rendered = [render_for_model(example, vocab, model_type) for example in batch]
        input_ids, labels = collate_rendered(rendered, vocab, cfg.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, attention_mask=(input_ids != vocab.pad_id).long())
        loss = causal_lm_loss(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        optimizer.step()
        if step == 1 or step % max(1, int(cfg.train.log_every)) == 0:
            row = {
                "step": step,
                "model_type": model_type,
                "seed": seed,
                "train_loss": float(loss.detach().cpu()),
                "learning_rate": lr,
            }
            rows.append(row)
            progress.set_postfix(loss=f"{row['train_loss']:.4f}", lr=f"{lr:.2e}")
        if cfg.train.checkpoint_every and step % int(cfg.train.checkpoint_every) == 0:
            save_checkpoint(model, final_path.parent / f"step_{step}.pt", {"config": cfg.to_dict(), "step": step})

    save_checkpoint(model, final_path, {"config": cfg.to_dict(), "step": int(cfg.train.steps), "model_type": model_type, "seed": seed})
    log = pd.DataFrame(rows)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    (final_path.parent / "metadata.json").write_text(
        json.dumps({"model_type": model_type, "seed": seed, "checkpoint": str(final_path)}, indent=2),
        encoding="utf-8",
    )
    return log


def train_all(cfg: V4Config, vocab: Vocab, run_dir: Path, skip_completed: bool = True) -> pd.DataFrame:
    frames = []
    for seed in cfg.seeds:
        for model_type in MODEL_TYPES:
            print(f"[v4 train] {model_type} seed={seed}", flush=True)
            frames.append(train_model(cfg, model_type, seed, vocab, run_dir, skip_completed=skip_completed))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        out.to_csv(run_dir / "metrics" / "train_log.csv", index=False)
    return out


def load_models(cfg: V4Config, vocab: Vocab, run_dir: Path, seed: int | None = None) -> dict[str, Any]:
    selected_seed = int(seed if seed is not None else cfg.seeds[0])
    models = {}
    for model_type in MODEL_TYPES:
        path = checkpoint_path(run_dir, model_type, selected_seed)
        if not checkpoint_is_compatible(path, cfg, vocab):
            raise RuntimeError(
                f"Checkpoint is missing or incompatible with the current v4 config: {path}. "
                "Use a separate --run-name/--out-root for debug vs main, or rerun training."
            )
        model = make_model(cfg, len(vocab.id_to_token), cfg.device)
        load_checkpoint(model, path, cfg.device)
        model.eval()
        models[model_type] = model
    return models
