from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from synthetic_niah_v5.model import make_model


IGNORE_INDEX = -100


@dataclass
class SweepConfig:
    experiment: str
    preset: str = "debug"
    seed: int = 1234
    seq_len: int = 256
    train_count_min: int = 1
    train_count_max: int = 10
    eval_count_min: int = 1
    eval_count_max: int = 10
    noise_vocab_size: int = 64
    marker_vocab_size: int = 10
    train_steps: int = 2000
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    log_every: int = 50
    checkpoint_every: int = 0
    eval_examples_per_count: int = 100
    ar_examples_per_count: int = 40
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_inner: int | None = None
    task_variant: str = "plain_marker_count"
    distractor_min: int = 0
    distractor_max: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def max_count(self) -> int:
        return max(int(self.train_count_max), int(self.eval_count_max))

    @property
    def n_positions(self) -> int:
        return int(self.seq_len) + 2 * int(self.max_count) + 12

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size) * int(self.grad_accum_steps)


class Vocab:
    def __init__(self, cfg: SweepConfig):
        self.special = ["<PAD>", "<BOS>", "<EOS>", "<Think>", "</Think>", "<Ans>"]
        if cfg.task_variant == "conditional_pairs":
            self.special += ["<Query>", "<Needle>", "<Decoy>"]
        self.noise = [f"<N{i}>" for i in range(cfg.noise_vocab_size)]
        self.markers = [f"<M{i}>" for i in range(cfg.marker_vocab_size)]
        # Keep v7/v8 directly comparable with v2: trace indices and final
        # answers use the same numeric-token embeddings and unembeddings.
        self.numbers = [f"<{i}>" for i in range(1, cfg.max_count + 1)]
        self.id_to_token = self.special + self.noise + self.markers + self.numbers
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def think_id(self) -> int:
        return self.token_to_id["<Think>"]

    @property
    def think_end_id(self) -> int:
        return self.token_to_id["</Think>"]

    @property
    def ans_id(self) -> int:
        return self.token_to_id["<Ans>"]

    def count_token(self, value: int) -> str:
        return f"<{int(value)}>"

    def index_token(self, value: int) -> str:
        return self.count_token(value)

    @property
    def count_ids(self) -> list[int]:
        return [self.token_to_id[t] for t in self.numbers]

    def to_json(self) -> dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "numeric_tokens": self.numbers,
            "shared_trace_and_answer_numbers": True,
            "task_variant": "conditional_pairs" if "<Query>" in self.token_to_id else "plain_marker_count",
        }


@dataclass
class Example:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    query_marker: str | None = None
    distractor_count: int = 0
    target_pair_positions: list[int] = field(default_factory=list)
    pair_type_counts: dict[str, int] = field(default_factory=dict)
    query_marker_budget: int | None = None
    needle_qualifier_budget: int | None = None


def _other_marker(vocab: Vocab, query_marker: str, rng: random.Random) -> str:
    choices = [marker for marker in vocab.markers if marker != query_marker]
    if not choices:
        raise ValueError("conditional_pairs requires at least two marker token types")
    return rng.choice(choices)


def _make_conditional_example(
    cfg: SweepConfig,
    vocab: Vocab,
    rng: random.Random,
    count: int | None,
) -> Example:
    target_count = rng.randint(cfg.train_count_min, cfg.train_count_max) if count is None else int(count)
    if not 1 <= target_count <= cfg.max_count:
        raise ValueError(f"count must be in [1, {cfg.max_count}], got {target_count}")

    query_marker = rng.choice(vocab.markers)
    # Marginal frequencies are sampled independently of the gold count. This
    # removes both count(query_marker) and count(<Needle>) as answer shortcuts.
    budget_low = cfg.max_count + 2
    budget_high = cfg.max_count + 4
    query_budget = rng.randint(budget_low, budget_high)
    needle_budget = rng.randint(budget_low, budget_high)
    query_decoys = query_budget - target_count
    other_needles = needle_budget - target_count

    required_distractors = query_decoys + other_needles + 2
    distractor_low = max(int(cfg.distractor_min), required_distractors)
    if distractor_low > int(cfg.distractor_max):
        raise ValueError(
            "distractor_max cannot support count-independent marginals: "
            f"need at least {distractor_low}, got {cfg.distractor_max}"
        )
    distractor_count = rng.randint(distractor_low, int(cfg.distractor_max))
    other_decoys = distractor_count - query_decoys - other_needles

    target_payloads = rng.sample(vocab.noise, target_count)
    distractor_payloads = [token for token in vocab.noise if token not in set(target_payloads)]

    blocks: list[tuple[str, list[str]]] = []
    blocks.extend(
        ("target", [query_marker, "<Needle>", payload])
        for payload in target_payloads
    )
    blocks.extend(
        ("query_decoy", [query_marker, "<Decoy>", rng.choice(distractor_payloads)])
        for _ in range(query_decoys)
    )
    blocks.extend(
        (
            "other_needle",
            [_other_marker(vocab, query_marker, rng), "<Needle>", rng.choice(distractor_payloads)],
        )
        for _ in range(other_needles)
    )
    blocks.extend(
        (
            "other_decoy",
            [_other_marker(vocab, query_marker, rng), "<Decoy>", rng.choice(distractor_payloads)],
        )
        for _ in range(other_decoys)
    )
    occupied = 2 + 3 * len(blocks)
    if occupied > cfg.seq_len:
        raise ValueError(f"seq_len={cfg.seq_len} cannot fit query plus {len(blocks)} event pairs")
    blocks.extend(("noise", [rng.choice(distractor_payloads)]) for _ in range(cfg.seq_len - occupied))
    rng.shuffle(blocks)

    sequence = ["<Query>", query_marker]
    target_positions: list[int] = []
    ordered_target_payloads: list[str] = []
    for block_type, block_tokens in blocks:
        start = len(sequence)
        sequence.extend(block_tokens)
        if block_type == "target":
            target_positions.append(start)
            ordered_target_payloads.append(block_tokens[2])

    return Example(
        seq_tokens=sequence,
        count=target_count,
        needle_positions=[position + 2 for position in target_positions],
        needle_markers=ordered_target_payloads,
        query_marker=query_marker,
        distractor_count=distractor_count,
        target_pair_positions=target_positions,
        pair_type_counts={
            "target": target_count,
            "query_decoy": query_decoys,
            "other_needle": other_needles,
            "other_decoy": other_decoys,
        },
        query_marker_budget=query_budget,
        needle_qualifier_budget=needle_budget,
    )


def make_example(cfg: SweepConfig, vocab: Vocab, rng: random.Random, count: int | None = None) -> Example:
    if cfg.task_variant == "conditional_pairs":
        return _make_conditional_example(cfg, vocab, rng, count)
    c = rng.randint(cfg.train_count_min, cfg.train_count_max) if count is None else int(count)
    if c > cfg.seq_len:
        raise ValueError("count cannot exceed seq_len")
    positions = sorted(rng.sample(range(cfg.seq_len), c))
    markers = [rng.choice(vocab.markers) for _ in positions]
    seq = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]
    for pos, marker in zip(positions, markers):
        seq[pos] = marker
    return Example(seq, c, positions, markers)


def balanced_examples(cfg: SweepConfig, vocab: Vocab, examples_per_count: int, seed: int, *, eval_counts: bool = False) -> list[Example]:
    rng = random.Random(seed)
    lo = cfg.eval_count_min if eval_counts else cfg.train_count_min
    hi = cfg.eval_count_max if eval_counts else cfg.train_count_max
    examples: list[Example] = []
    for count in range(lo, hi + 1):
        for _ in range(examples_per_count):
            examples.append(make_example(cfg, vocab, rng, count=count))
    rng.shuffle(examples)
    return examples


def render(ex: Example, vocab: Vocab, mode: str) -> dict[str, Any]:
    labels: list[int]
    if mode == "nonthinking":
        tokens = ["<BOS>"] + ex.seq_tokens + ["<Ans>", vocab.count_token(ex.count), "<EOS>"]
        ans_pos = 1 + len(ex.seq_tokens)
        labels = [IGNORE_INDEX] * len(tokens)
        labels[ans_pos + 1] = vocab.token_to_id[vocab.count_token(ex.count)]
        labels[ans_pos + 2] = vocab.eos_id
        anchors = {"ans_pos": ans_pos, "count_pos": ans_pos + 1}
    elif mode == "thinking":
        trace: list[str] = []
        if ex.query_marker is not None:
            for k, payload in enumerate(ex.needle_markers, start=1):
                trace.extend([vocab.index_token(k), payload])
        else:
            for k, marker in enumerate(ex.needle_markers, start=1):
                trace.extend([vocab.index_token(k), marker])
        tokens = ["<BOS>"] + ex.seq_tokens + ["<Think>"] + trace + ["</Think>", "<Ans>", vocab.count_token(ex.count), "<EOS>"]
        think_pos = 1 + len(ex.seq_tokens)
        close_pos = think_pos + 1 + len(trace)
        ans_pos = close_pos + 1
        labels = [IGNORE_INDEX] * len(tokens)
        for pos in range(think_pos + 1, len(tokens)):
            labels[pos] = vocab.token_to_id[tokens[pos]]
        anchors = {"think_pos": think_pos, "close_pos": close_pos, "ans_pos": ans_pos, "count_pos": ans_pos + 1}
    else:
        raise ValueError(mode)
    return {"tokens": tokens, "input_ids": vocab.encode(tokens), "labels": labels, "anchors": anchors, "count": ex.count}


def collate(items: list[dict[str, Any]], vocab: Vocab, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, item in enumerate(items):
        ids[i, : len(item["input_ids"])] = torch.tensor(item["input_ids"])
        labels[i, : len(item["labels"])] = torch.tensor(item["labels"])
    return ids.to(device), labels.to(device)


def lr_at(step: int, cfg: SweepConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_model_cfg(cfg: SweepConfig, vocab: Vocab) -> dict[str, Any]:
    return {
        "vocab_size": len(vocab.id_to_token),
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "n_inner": cfg.n_inner if cfg.n_inner is not None else 4 * cfg.n_embd,
        "n_positions": cfg.n_positions,
        "n_ctx": cfg.n_positions,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "use_cache": False,
        "bos_token_id": vocab.bos_id,
        "eos_token_id": vocab.eos_id,
        "pad_token_id": vocab.pad_id,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sync_file(local_path: Path, run_dir: Path, sync_run_dir: Path | None) -> None:
    if sync_run_dir is None:
        return
    destination = sync_run_dir / local_path.relative_to(run_dir)
    _atomic_copy(local_path, destination)


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("step_"))
    except ValueError:
        return -1


def _latest_checkpoint(
    local_mode_dir: Path,
    sync_mode_dir: Path | None,
) -> tuple[int, Path] | None:
    candidates = list(local_mode_dir.glob("step_*/checkpoint.pt"))
    if sync_mode_dir is not None and sync_mode_dir.exists():
        candidates.extend(sync_mode_dir.glob("step_*/checkpoint.pt"))
    valid = [(_checkpoint_step(path), path) for path in candidates if _checkpoint_step(path) >= 0]
    return max(valid, key=lambda item: item[0]) if valid else None


def _training_row(
    cfg: SweepConfig,
    mode: str,
    step: int,
    loss: float,
    lr: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "mode": mode,
        "loss": loss,
        "lr": lr,
        "micro_batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch_size": cfg.effective_batch_size,
    }


def _save_training_checkpoint(
    *,
    cfg: SweepConfig,
    model: torch.nn.Module,
    optimizer: AdamW,
    mode: str,
    step: int,
    rng: random.Random,
    rows: list[dict[str, Any]],
    run_dir: Path,
    sync_run_dir: Path | None,
) -> Path:
    mode_dir = run_dir / "checkpoints" / mode
    checkpoint_path = mode_dir / f"step_{step:06d}" / "checkpoint.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "mode": mode,
        "step": step,
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _atomic_torch_save(payload, checkpoint_path)

    train_path = run_dir / "tables" / f"train_{mode}.csv"
    _atomic_dataframe_csv(pd.DataFrame(rows), train_path)
    latest_path = mode_dir / "latest.json"
    latest_path.write_text(
        json.dumps({"step": step, "checkpoint": str(checkpoint_path.relative_to(run_dir))}, indent=2),
        encoding="utf-8",
    )
    for artifact in [checkpoint_path, train_path, latest_path]:
        _sync_file(artifact, run_dir, sync_run_dir)
    if sync_run_dir is not None:
        print(f"[{mode}] checkpoint step {step} synced to {sync_run_dir}", flush=True)
    return checkpoint_path


def train_one(
    cfg: SweepConfig,
    vocab: Vocab,
    mode: str,
    run_dir: Path,
    sync_run_dir: Path | None = None,
) -> pd.DataFrame:
    mode_dir = run_dir / "checkpoints" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    train_path = run_dir / "tables" / f"train_{mode}.csv"
    final_path = mode_dir / "model.pt"
    sync_mode_dir = sync_run_dir / "checkpoints" / mode if sync_run_dir is not None else None

    if not final_path.exists() and sync_mode_dir is not None:
        sync_final = sync_mode_dir / "model.pt"
        if sync_final.exists():
            _atomic_copy(sync_final, final_path)
        sync_train = sync_run_dir / "tables" / f"train_{mode}.csv"
        if sync_train.exists() and not train_path.exists():
            _atomic_copy(sync_train, train_path)
    if final_path.exists():
        return pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()

    torch.manual_seed(cfg.seed + (0 if mode == "nonthinking" else 17))
    rng = random.Random(cfg.seed + (100 if mode == "nonthinking" else 200))
    model = make_model(build_model_cfg(cfg, vocab), cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    rows: list[dict[str, Any]] = []
    start_step = 0

    latest = _latest_checkpoint(mode_dir, sync_mode_dir)
    if latest is not None:
        start_step, resume_path = latest
        local_resume_path = mode_dir / f"step_{start_step:06d}" / "checkpoint.pt"
        if resume_path != local_resume_path:
            _atomic_copy(resume_path, local_resume_path)
        payload = _torch_load(local_resume_path, cfg.device)
        model.load_state_dict(payload["model_state_dict"])
        opt.load_state_dict(payload["optimizer_state_dict"])
        rng.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        if not train_path.exists() and sync_run_dir is not None:
            sync_train = sync_run_dir / "tables" / f"train_{mode}.csv"
            if sync_train.exists():
                _atomic_copy(sync_train, train_path)
        if train_path.exists():
            rows = pd.read_csv(train_path).query("step <= @start_step").to_dict("records")
        print(f"[{mode}] resuming from step {start_step}: {resume_path}", flush=True)

    pbar = tqdm(
        range(start_step + 1, cfg.train_steps + 1),
        total=cfg.train_steps,
        initial=start_step,
        desc=f"{cfg.experiment} {mode}",
        leave=True,
    )
    for step in pbar:
        lr = lr_at(step - 1, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = [render(make_example(cfg, vocab, rng), vocab, mode) for _ in range(cfg.batch_size)]
            ids, labels = collate(batch, vocab, cfg.device)
            out = model(input_ids=ids, attention_mask=(ids != vocab.pad_id).long())
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            (loss / cfg.grad_accum_steps).backward()
            mean_loss += float(loss.detach().cpu()) / cfg.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        should_checkpoint = cfg.checkpoint_every > 0 and (
            step % cfg.checkpoint_every == 0 or step == cfg.train_steps
        )
        should_log = step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps
        if should_log or should_checkpoint:
            row = _training_row(cfg, mode, step, mean_loss, lr)
            if not rows or int(rows[-1]["step"]) != step:
                rows.append(row)
            pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.1e}")
        if should_checkpoint:
            _save_training_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=opt,
                mode=mode,
                step=step,
                rng=rng,
                rows=rows,
                run_dir=run_dir,
                sync_run_dir=sync_run_dir,
            )

    final_payload = {"model_state_dict": model.state_dict(), "cfg": asdict(cfg), "mode": mode}
    _atomic_torch_save(final_payload, final_path)
    train_df = pd.DataFrame(rows)
    _atomic_dataframe_csv(train_df, train_path)
    _sync_file(final_path, run_dir, sync_run_dir)
    _sync_file(train_path, run_dir, sync_run_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_df


def load_model(cfg: SweepConfig, vocab: Vocab, mode: str, run_dir: Path):
    model = make_model(build_model_cfg(cfg, vocab), cfg.device)
    obj = torch.load(run_dir / "checkpoints" / mode / "model.pt", map_location=cfg.device)
    model.load_state_dict(obj["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_one(cfg: SweepConfig, vocab: Vocab, mode: str, run_dir: Path) -> pd.DataFrame:
    model = load_model(cfg, vocab, mode, run_dir)
    examples = balanced_examples(cfg, vocab, cfg.eval_examples_per_count, cfg.seed + 9000, eval_counts=True)
    rows: list[dict[str, Any]] = []
    ar_seen = {count: 0 for count in range(cfg.eval_count_min, cfg.eval_count_max + 1)}
    for ex in tqdm(examples, desc=f"eval {mode}", leave=False):
        r = render(ex, vocab, mode)
        ids = torch.tensor([r["input_ids"]], dtype=torch.long, device=cfg.device)
        logits = model(input_ids=ids).logits[0, r["anchors"]["ans_pos"]]
        pred = int(logits[vocab.count_ids].argmax().item()) + 1
        ar_pred = None
        ar_steps = None
        if ar_seen[ex.count] < cfg.ar_examples_per_count:
            ar_seen[ex.count] += 1
            ar_pred, ar_steps = autoregressive_predict_count(model, vocab, ex, cfg, mode)
        rows.append(
            {
                "mode": mode,
                "count": ex.count,
                "tf_pred_count": pred,
                "tf_accuracy": float(pred == ex.count),
                "tf_abs_error": abs(pred - ex.count),
                "ar_pred_count": ar_pred,
                "ar_accuracy": float(ar_pred == ex.count) if ar_pred is not None else None,
                "ar_abs_error": abs(ar_pred - ex.count) if ar_pred is not None else None,
                "ar_generated_steps": ar_steps,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "tables" / f"eval_{mode}_examples.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


@torch.no_grad()
def autoregressive_predict_count(model, vocab: Vocab, ex: Example, cfg: SweepConfig, mode: str) -> tuple[int | None, int]:
    if mode == "nonthinking":
        generated = vocab.encode(["<BOS>"] + ex.seq_tokens + ["<Ans>"])
        seen_ans = True
        max_new = 2
    else:
        generated = vocab.encode(["<BOS>"] + ex.seq_tokens + ["<Think>"])
        seen_ans = False
        max_new = 2 * cfg.max_count + 8
    count_lookup = {tok_id: count for count, tok_id in enumerate(vocab.count_ids, start=1)}
    for step in range(1, max_new + 1):
        ids = torch.tensor([generated], dtype=torch.long, device=cfg.device)
        next_id = int(model(input_ids=ids).logits[0, -1].argmax().item())
        generated.append(next_id)
        if seen_ans and next_id in count_lookup:
            return count_lookup[next_id], step
        if next_id == vocab.ans_id:
            seen_ans = True
        if next_id == vocab.eos_id and not seen_ans:
            return None, step
    return None, max_new


def summarize_eval(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(dfs, ignore_index=True)
    summary = (
        all_df.groupby(["mode", "count"], as_index=False)
        .agg(
            n=("tf_accuracy", "size"),
            n_ar=("ar_accuracy", "count"),
            tf_accuracy=("tf_accuracy", "mean"),
            ar_accuracy=("ar_accuracy", "mean"),
            tf_mae=("tf_abs_error", "mean"),
            ar_mae=("ar_abs_error", "mean"),
        )
        .sort_values(["mode", "count"])
    )
    summary["accuracy"] = summary["ar_accuracy"].fillna(summary["tf_accuracy"])
    summary["mae"] = summary["ar_mae"].fillna(summary["tf_mae"])
    summary["validation_split"] = summary["count"].map(validation_split)
    return summary


def validation_split(count: int) -> str:
    count = int(count)
    if count <= 10:
        return "val_1_10"
    if count <= 20:
        return "val_11_20"
    if count <= 30:
        return "val_21_30"
    return "val_31_plus"


def summarize_validation_splits(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["mode", "validation_split"], as_index=False)
        .agg(
            count_min=("count", "min"),
            count_max=("count", "max"),
            n_counts=("count", "size"),
            n_tf=("n", "sum"),
            n_ar=("n_ar", "sum"),
            tf_accuracy=("tf_accuracy", "mean"),
            ar_accuracy=("ar_accuracy", "mean"),
            tf_mae=("tf_mae", "mean"),
            ar_mae=("ar_mae", "mean"),
            accuracy=("accuracy", "mean"),
            mae=("mae", "mean"),
        )
        .sort_values(["mode", "count_min"])
        .reset_index(drop=True)
    )


def preset_configs(experiment: str, preset: str) -> list[SweepConfig]:
    if preset == "debug":
        debug_max_count = 30 if experiment == "v8" else (4 if experiment == "v9" else 10)
        debug_model = (
            {"n_layer": 3, "n_head": 3, "n_embd": 48, "n_inner": 96}
            if experiment == "v9"
            else {"n_layer": 2, "n_head": 2, "n_embd": 64}
        )
        base = SweepConfig(
            experiment=experiment,
            preset=preset,
            seq_len=64 if experiment == "v9" else 48,
            train_count_max=debug_max_count,
            eval_count_max=debug_max_count,
            task_variant="conditional_pairs" if experiment == "v9" else "plain_marker_count",
            distractor_min=8 if experiment == "v9" else 0,
            distractor_max=16 if experiment == "v9" else 0,
            train_steps=4,
            batch_size=8,
            grad_accum_steps=1,
            eval_examples_per_count=2,
            ar_examples_per_count=2,
            **debug_model,
        )
        return [base]
    if experiment == "v7":
        return [
            SweepConfig(
                experiment=experiment,
                preset=preset,
                seq_len=2048,
                train_count_max=10,
                eval_count_max=10,
                train_steps=10000,
                batch_size=16,
                grad_accum_steps=8,
                checkpoint_every=2000,
                eval_examples_per_count=150,
                ar_examples_per_count=30,
            ),
        ]
    if experiment == "v8":
        return [
            SweepConfig(
                experiment=experiment,
                preset=preset,
                seq_len=256,
                train_count_max=30,
                eval_count_max=30,
                train_steps=10000,
                batch_size=128,
                grad_accum_steps=1,
                eval_examples_per_count=200,
                ar_examples_per_count=50,
            ),
        ]
    if experiment == "v9":
        return [
            SweepConfig(
                experiment=experiment,
                preset=preset,
                seq_len=256,
                train_count_min=1,
                train_count_max=10,
                eval_count_min=1,
                eval_count_max=10,
                train_steps=10000,
                batch_size=128,
                grad_accum_steps=1,
                checkpoint_every=2000,
                eval_examples_per_count=200,
                ar_examples_per_count=50,
                n_layer=3,
                n_head=3,
                n_embd=48,
                n_inner=96,
                task_variant="conditional_pairs",
                distractor_min=32,
                distractor_max=64,
            ),
        ]
    raise ValueError(experiment)


def run_one_config(
    cfg: SweepConfig,
    out_root: Path,
    skip_completed: bool = True,
    checkpoint_sync_root: Path | None = None,
) -> Path:
    capacity_suffix = (
        f"_cap{cfg.n_layer}L{cfg.n_head}H_d{cfg.n_embd}_ff{cfg.n_inner}"
        if cfg.experiment == "v9"
        else ""
    )
    task_suffix = "_conditional_pairs" if cfg.task_variant == "conditional_pairs" else ""
    name = (
        f"{cfg.experiment}_{cfg.preset}_L{cfg.seq_len}_"
        f"train{cfg.train_count_min}-{cfg.train_count_max}_"
        f"eval{cfg.eval_count_min}-{cfg.eval_count_max}{task_suffix}"
        f"{capacity_suffix}_sharednum_seed{cfg.seed}"
    )
    run_dir = out_root / name
    sync_run_dir = checkpoint_sync_root / name if checkpoint_sync_root is not None else None
    for sub in ["tables", "figures", "checkpoints", "report"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    vocab = Vocab(cfg)
    vocab_path = run_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab.to_json(), indent=2), encoding="utf-8")
    if sync_run_dir is not None:
        _sync_file(config_path, run_dir, sync_run_dir)
        _sync_file(vocab_path, run_dir, sync_run_dir)
    eval_dfs = []
    for mode in ["nonthinking", "thinking"]:
        if not skip_completed or not (run_dir / "checkpoints" / mode / "model.pt").exists():
            train_one(cfg, vocab, mode, run_dir, sync_run_dir=sync_run_dir)
        eval_path = run_dir / "tables" / f"eval_{mode}_examples.csv"
        if skip_completed and eval_path.exists():
            eval_dfs.append(pd.read_csv(eval_path))
        else:
            eval_dfs.append(evaluate_one(cfg, vocab, mode, run_dir))
    summary = summarize_eval(eval_dfs)
    summary.to_csv(run_dir / "tables" / "eval_by_count.csv", index=False)
    split_summary = summarize_validation_splits(summary)
    split_summary.to_csv(run_dir / "tables" / "eval_by_validation_split.csv", index=False)
    make_plots(run_dir, summary, split_summary)
    make_report(run_dir, cfg, summary, split_summary)
    return run_dir


def make_plots(run_dir: Path, summary: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, group in summary.sort_values("count").groupby("mode"):
        ax.plot(group["count"], group["accuracy"], marker="o", label=mode)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("gold needle count")
    ax.set_ylabel("final-count accuracy")
    ax.set_title("Accuracy by needle count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "accuracy_by_count.png", dpi=180)
    plt.close(fig)

    pivot = summary.pivot(index="mode", columns="count", values="accuracy")
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel("gold needle count")
    ax.set_ylabel("model mode")
    for y, row in enumerate(values):
        for x, val in enumerate(row):
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Accuracy heatmap by mode and count")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    split_order = ["val_1_10", "val_11_20", "val_21_30", "val_31_plus"]
    available = [name for name in split_order if name in set(split_summary["validation_split"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    x = np.arange(len(available))
    for offset, (mode, group) in zip(
        (-width / 2, width / 2), split_summary.groupby("mode", sort=True)
    ):
        values = (
            group.set_index("validation_split")
            .reindex(available)["accuracy"]
            .to_numpy(dtype=float)
        )
        bars = ax.bar(x + offset, values, width=width, label=mode)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x, labels=available)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("balanced validation split by gold needle count")
    ax.set_ylabel("autoregressive final-count accuracy")
    ax.set_title("Accuracy by validation count range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "accuracy_by_validation_split.png", dpi=180)
    plt.close(fig)


def _img(name: str) -> str:
    return f"<img src='../figures/{name}' style='max-width:100%;border:1px solid #ddd;border-radius:8px'>"


def make_report(
    run_dir: Path,
    cfg: SweepConfig,
    summary: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    wide = summary.pivot(index="count", columns="mode", values="accuracy").reset_index()
    if {"thinking", "nonthinking"}.issubset(wide.columns):
        wide["cot_minus_nonthinking"] = wide["thinking"] - wide["nonthinking"]
    threshold_rows = []
    for mode, g in summary.groupby("mode"):
        bad = g[g["accuracy"] < 0.9]
        threshold_rows.append({"mode": mode, "first_count_below_0.9": int(bad["count"].iloc[0]) if not bad.empty else "none"})
    threshold = pd.DataFrame(threshold_rows)
    descriptions = {
        "v7": "v7 increases prompt length while keeping count 1-10.",
        "v8": "v8 increases the needle-count range and reports balanced count bins.",
        "v9": (
            "v9 uses a query-conditioned pair-counting task at length 256 and target "
            "count 1-10. It counts only <M_q> <Needle> pairs while rejecting query "
            "decoys, other-marker needles, and other decoys. Query-marker and Needle "
            "marginal frequencies are sampled independently of the answer. The model "
            "has 3 layers, 3 heads, d_model 48, and MLP 96."
        ),
    }
    task_details = ""
    if cfg.task_variant == "conditional_pairs":
        task_details = pd.DataFrame(
            [
                {"record type": "target", "form": "<M_q> <Needle> payload", "counted": "yes"},
                {"record type": "query decoy", "form": "<M_q> <Decoy> payload", "counted": "no"},
                {"record type": "other needle", "form": "<M_j> <Needle> payload", "counted": "no"},
                {"record type": "other decoy", "form": "<M_j> <Decoy> payload", "counted": "no"},
            ]
        ).to_html(index=False)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{cfg.experiment} report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;line-height:1.55;max-width:1050px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f4f7fb}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}</style></head><body>
<h1>{cfg.experiment} synthetic counting sweep</h1>
<p>{descriptions[cfg.experiment]}</p>
<h2>Task semantics</h2>{task_details}
<h2>Config</h2><pre>{json.dumps(asdict(cfg), indent=2)}</pre>
<div class="grid"><div>{_img('accuracy_by_count.png')}</div><div>{_img('accuracy_heatmap.png')}</div><div>{_img('accuracy_by_validation_split.png')}</div></div>
<h2>CoT advantage by count</h2>{wide.to_html(index=False)}
<h2>First count below 0.9 accuracy</h2>{threshold.to_html(index=False)}
<h2>Balanced validation ranges</h2>{split_summary.to_html(index=False)}
<h2>Raw summary</h2>{summary.to_html(index=False)}
</body></html>"""
    (run_dir / "report" / "report.html").write_text(html, encoding="utf-8")


def run_sweep(
    experiment: str,
    preset: str,
    out_root: str | Path,
    *,
    skip_completed: bool = True,
    device: str | None = None,
    checkpoint_sync_root: str | Path | None = None,
) -> pd.DataFrame:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    sync_root = Path(checkpoint_sync_root) if checkpoint_sync_root is not None else None
    if sync_root is not None:
        sync_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_split_rows = []
    for cfg in preset_configs(experiment, preset):
        if device:
            cfg.device = device
        run_dir = run_one_config(
            cfg,
            out_root,
            skip_completed=skip_completed,
            checkpoint_sync_root=sync_root,
        )
        summary = pd.read_csv(run_dir / "tables" / "eval_by_count.csv")
        summary.insert(0, "run_dir", str(run_dir))
        summary.insert(1, "setting", run_dir.name)
        all_rows.append(summary)
        split_summary = pd.read_csv(run_dir / "tables" / "eval_by_validation_split.csv")
        split_summary.insert(0, "run_dir", str(run_dir))
        split_summary.insert(1, "setting", run_dir.name)
        all_split_rows.append(split_summary)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_root / f"{experiment}_{preset}_combined_eval_by_count.csv", index=False)
    combined_splits = pd.concat(all_split_rows, ignore_index=True)
    combined_splits.to_csv(
        out_root / f"{experiment}_{preset}_combined_eval_by_validation_split.csv",
        index=False,
    )
    return combined


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v7/v8/v9 synthetic counting sweeps")
    p.add_argument("--experiment", choices=["v7", "v8", "v9"], required=True)
    p.add_argument("--preset", choices=["debug", "main"], default="debug")
    p.add_argument("--out-root", default="runs/synthetic_counting_sweeps")
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint-sync-root", default=None)
    p.add_argument("--no-skip-completed", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_sweep(
        args.experiment,
        args.preset,
        args.out_root,
        skip_completed=not args.no_skip_completed,
        device=args.device,
        checkpoint_sync_root=args.checkpoint_sync_root,
    )


if __name__ == "__main__":
    main()
