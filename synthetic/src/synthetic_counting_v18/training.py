from __future__ import annotations

import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import ReferenceConfig, RunSpec
from .data import (
    ANSWER,
    END,
    PAD,
    START,
    count_from_token,
    collate,
    index_from_token,
    is_marker_token,
    render,
    sample_example,
)
from .model import ReferenceTransformer


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                if path.exists():
                    path.unlink()
                temporary.replace(path)
                return
            time.sleep(0.05 * (attempt + 1))


def append_rows(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
    if not rows:
        return
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    frame = pd.concat((old, pd.DataFrame(rows)), ignore_index=True)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    atomic_csv(frame, path)


def rate(cfg: ReferenceConfig, step: int) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def autocast(cfg: ReferenceConfig):
    enabled = (
        cfg.precision == "bf16"
        and str(cfg.device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def save_checkpoint(
    model: ReferenceTransformer,
    optimizer: AdamW,
    step: int,
    rng: random.Random,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "python_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def restore_checkpoint(
    model: ReferenceTransformer,
    optimizer: AdamW,
    rng: random.Random,
    path: Path,
    device: str,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([state.cpu().to(torch.uint8) for state in payload["cuda_rng"]])
    return int(payload["step"])


@torch.no_grad()
def greedy_batch(
    model: ReferenceTransformer,
    spec: RunSpec,
    contexts: list[tuple[int, ...]],
) -> list[tuple[int | None, int | None, list[int], list[int]]]:
    """Free-run a same-task batch while allowing examples to stop independently."""

    if not contexts:
        return []
    model.eval()
    generated = [list(context) + ([ANSWER] if spec.mode == "direct" else [START]) for context in contexts]
    prefix_lengths = [len(context) + 1 for context in contexts]
    active = [True] * len(generated)
    maximum_new = 1 if spec.mode == "direct" else 2 * spec.eval_max_count + 2
    for _ in range(maximum_new):
        if not any(active):
            break
        lengths = [len(tokens) for tokens in generated]
        width = max(lengths)
        device = next(model.parameters()).device
        ids = torch.full((len(generated), width), PAD, dtype=torch.long, device=device)
        valid = torch.zeros((len(generated), width), dtype=torch.long, device=device)
        for row, tokens in enumerate(generated):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            valid[row, : len(tokens)] = 1
        with autocast(model.cfg):
            logits = model(ids, valid).logits
        next_tokens = logits[
            torch.arange(len(generated), device=device),
            torch.tensor(lengths, device=device) - 1,
        ].argmax(-1).tolist()
        for row, token in enumerate(next_tokens):
            if not active[row]:
                continue
            generated[row].append(int(token))
            suffix = generated[row][prefix_lengths[row] :]
            if spec.mode == "direct" or (
                END in suffix and count_from_token(int(token), spec.eval_max_count) is not None
            ):
                active[row] = False

    results: list[tuple[int | None, int | None, list[int], list[int]]] = []
    for row, tokens in enumerate(generated):
        suffix = tokens[prefix_lengths[row] :]
        if spec.mode == "direct":
            results.append((None, count_from_token(suffix[0], spec.eval_max_count) if suffix else None, [], suffix))
            continue
        cursor = 0
        expected_index = 1
        generated_markers: list[int] = []
        valid_trace = True
        while cursor < len(suffix) and suffix[cursor] != END:
            if cursor + 1 >= len(suffix):
                valid_trace = False
                break
            index = index_from_token(suffix[cursor], spec.eval_max_count)
            marker = suffix[cursor + 1]
            if index != expected_index or not is_marker_token(marker):
                valid_trace = False
                break
            generated_markers.append(marker)
            expected_index += 1
            cursor += 2
        enumeration = (
            len(generated_markers)
            if valid_trace and cursor < len(suffix) and suffix[cursor] == END
            else None
        )
        token_count = (
            count_from_token(suffix[cursor + 1], spec.eval_max_count)
            if enumeration is not None and cursor + 1 < len(suffix)
            else None
        )
        results.append((enumeration, token_count, generated_markers, suffix))
    return results


@torch.no_grad()
def greedy_one(
    model: ReferenceTransformer,
    spec: RunSpec,
    context: tuple[int, ...],
) -> tuple[int | None, int | None, list[int], list[int]]:
    return greedy_batch(model, spec, [context])[0]


@torch.no_grad()
def evaluate(
    model: ReferenceTransformer,
    cfg: ReferenceConfig,
    spec: RunSpec,
    *,
    step: int,
    examples_per_count: int | None = None,
    total_examples: int | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(cfg.seed + 100_000 + step)
    counts: list[int] = []
    if examples_per_count is not None:
        counts = [count for count in range(1, spec.eval_max_count + 1) for _ in range(examples_per_count)]
    elif total_examples is not None:
        counts = [rng.randint(1, spec.eval_max_count) for _ in range(total_examples)]
    rows: list[dict[str, object]] = []
    examples = [sample_example(cfg, spec, rng, count=count) for count in counts]
    maximum_new = 1 if spec.mode == "direct" else 2 * spec.eval_max_count + 2
    tokens_per_example = spec.context_length + maximum_new + 1
    effective_batch_size = min(
        cfg.eval_batch_size,
        max(1, cfg.eval_token_budget // tokens_per_example),
    )
    for batch_start in range(0, len(examples), effective_batch_size):
        batch = examples[batch_start : batch_start + effective_batch_size]
        outputs = greedy_batch(model, spec, [example.context for example in batch])
        for offset, (example, output) in enumerate(zip(batch, outputs)):
            index = batch_start + offset
            enumeration, token_count, generated_markers, generated = output
            marker_matches = sum(
                predicted == gold
                for predicted, gold in zip(generated_markers, example.needle_markers)
            )
            trace_marker_accuracy = (
                marker_matches / example.count if spec.mode == "cot" else np.nan
            )
            trace_exact = (
                enumeration == example.count
                and tuple(generated_markers) == example.needle_markers
            ) if spec.mode == "cot" else False
            # The primary metric is the final scalar answer for both models.
            primary = token_count
            rows.append({
                "run_name": spec.name,
                "step": step,
                "row_id": index,
                "mode": spec.mode,
                "distribution": spec.distribution,
                "context_length": spec.context_length,
                "train_max_count": spec.train_max_count,
                "eval_max_count": spec.eval_max_count,
                "alpha": spec.alpha,
                "gold_count": example.count,
                "count_band": (
                    "count_1_32" if example.count <= 32 else
                    "count_33_64" if example.count <= 64 else
                    "count_65_96" if example.count <= 96 else
                    "count_97_128"
                ),
                "enumeration_count": enumeration,
                "token_count": token_count,
                "enumeration_accuracy": (
                    float(enumeration == example.count) if spec.mode == "cot" else np.nan
                ),
                "trace_marker_accuracy": trace_marker_accuracy,
                "trace_exact_accuracy": float(trace_exact) if spec.mode == "cot" else np.nan,
                "token_accuracy": float(token_count == example.count),
                "primary_accuracy": float(primary == example.count),
                "primary_abs_error": (
                    abs(primary - example.count) if primary is not None else np.nan
                ),
                "generated": " ".join(map(str, generated)),
            })
    return rows


def _restore_synced_checkpoint(
    suite_dir: Path,
    spec: RunSpec,
    checkpoint_sync_root: Path | None,
) -> None:
    if checkpoint_sync_root is None:
        return
    local = suite_dir / "runs" / spec.name / "checkpoints"
    remote = checkpoint_sync_root / suite_dir.name / "runs" / spec.name / "checkpoints"
    local.mkdir(parents=True, exist_ok=True)
    for name in ("final.pt", "latest.pt"):
        source, target = remote / name, local / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
            print(f"[restore] {spec.name}: {name} from {source}", flush=True)


def train_run(
    cfg: ReferenceConfig,
    spec: RunSpec,
    suite_dir: Path,
    *,
    skip_completed: bool,
    checkpoint_sync_root: Path | None,
) -> None:
    run_dir = suite_dir / "runs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec.__dict__, indent=2), encoding="utf-8")
    if skip_completed:
        _restore_synced_checkpoint(suite_dir, spec, checkpoint_sync_root)
    final = run_dir / "checkpoints" / "final.pt"
    final_table = suite_dir / "tables" / "final_detail.csv"
    final_is_recorded = False
    if final_table.exists() and final_table.stat().st_size:
        recorded = pd.read_csv(final_table, usecols=["run_name"])
        final_is_recorded = bool((recorded["run_name"] == spec.name).any())
    if skip_completed and final.exists() and final_is_recorded:
        print(f"[skip] {spec.name}: final checkpoint and evaluation exist", flush=True)
        return
    torch.manual_seed(cfg.seed + 17)
    model = ReferenceTransformer(cfg).to(cfg.device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    rng = random.Random(cfg.seed)
    start = 0
    if skip_completed and final.exists():
        restore_checkpoint(model, optimizer, rng, final, cfg.device)
        print(f"[evaluate] {spec.name}: restored final checkpoint", flush=True)
        rows = evaluate(
            model,
            cfg,
            spec,
            step=cfg.train_steps,
            examples_per_count=cfg.final_examples_per_count,
        )
        append_rows(final_table, rows, ["run_name", "step", "row_id"])
        return
    latest = run_dir / "checkpoints" / "latest.pt"
    if skip_completed and latest.exists():
        start = restore_checkpoint(model, optimizer, rng, latest, cfg.device)
        print(f"[resume] {spec.name} at step {start}", flush=True)
    progress = tqdm(range(start + 1, cfg.train_steps + 1), initial=start, total=cfg.train_steps, desc=spec.name)
    for step in progress:
        rendered = [render(sample_example(cfg, spec, rng), spec.mode) for _ in range(cfg.batch_size)]
        ids, labels, valid = collate(rendered, cfg.device)
        model.train()
        with autocast(cfg):
            loss = token_loss(model(ids, valid).logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        lr = rate(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            append_rows(
                suite_dir / "tables" / "training_metrics.csv",
                [{"run_name": spec.name, "step": step, "loss": float(loss.detach()), "lr": lr, "grad_norm": grad_norm}],
                ["run_name", "step"],
            )
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            rows = evaluate(model, cfg, spec, step=step, total_examples=cfg.dynamics_examples)
            append_rows(suite_dir / "tables" / "dynamics_detail.csv", rows, ["run_name", "step", "row_id"])
        if step % cfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, rng, latest)
            save_checkpoint(model, optimizer, step, rng, run_dir / "checkpoints" / f"step_{step:06d}.pt")
            if checkpoint_sync_root is not None:
                target = checkpoint_sync_root / suite_dir.name / "runs" / spec.name / "checkpoints"
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(latest, target / "latest.pt")
    save_checkpoint(model, optimizer, cfg.train_steps, rng, final)
    if checkpoint_sync_root is not None:
        target = checkpoint_sync_root / suite_dir.name / "runs" / spec.name / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, target / "final.pt")
    rows = evaluate(model, cfg, spec, step=cfg.train_steps, examples_per_count=cfg.final_examples_per_count)
    append_rows(suite_dir / "tables" / "final_detail.csv", rows, ["run_name", "step", "row_id"])


def summarize(suite_dir: Path) -> None:
    for source, target in (("dynamics_detail.csv", "dynamics_summary.csv"), ("final_detail.csv", "final_summary.csv")):
        path = suite_dir / "tables" / source
        if not path.exists() or not path.stat().st_size:
            continue
        frame = pd.read_csv(path)
        keys = ["run_name", "step", "mode", "distribution", "context_length", "train_max_count", "eval_max_count", "alpha"]
        summary = frame.groupby(keys, dropna=False, as_index=False).agg(
            primary_accuracy=("primary_accuracy", "mean"),
            enumeration_accuracy=("enumeration_accuracy", "mean"),
            trace_marker_accuracy=("trace_marker_accuracy", "mean"),
            trace_exact_accuracy=("trace_exact_accuracy", "mean"),
            token_accuracy=("token_accuracy", "mean"),
            primary_mae=("primary_abs_error", "mean"),
        )
        atomic_csv(summary, suite_dir / "tables" / target)
    dynamics_path = suite_dir / "tables" / "dynamics_detail.csv"
    if dynamics_path.exists() and dynamics_path.stat().st_size:
        frame = pd.read_csv(dynamics_path)
        by_band = frame.groupby(
            ["run_name", "step", "mode", "distribution", "count_band"],
            as_index=False,
        ).agg(
            n_examples=("primary_accuracy", "size"),
            primary_accuracy=("primary_accuracy", "mean"),
            enumeration_accuracy=("enumeration_accuracy", "mean"),
            trace_marker_accuracy=("trace_marker_accuracy", "mean"),
            trace_exact_accuracy=("trace_exact_accuracy", "mean"),
            token_accuracy=("token_accuracy", "mean"),
            primary_mae=("primary_abs_error", "mean"),
        )
        atomic_csv(by_band, suite_dir / "tables" / "dynamics_by_band.csv")
        by_count = frame.groupby(
            ["run_name", "step", "mode", "distribution", "gold_count"],
            as_index=False,
        ).agg(
            n_examples=("primary_accuracy", "size"),
            primary_accuracy=("primary_accuracy", "mean"),
            enumeration_accuracy=("enumeration_accuracy", "mean"),
            trace_marker_accuracy=("trace_marker_accuracy", "mean"),
            trace_exact_accuracy=("trace_exact_accuracy", "mean"),
            token_accuracy=("token_accuracy", "mean"),
            primary_mae=("primary_abs_error", "mean"),
        )
        atomic_csv(by_count, suite_dir / "tables" / "dynamics_by_count.csv")
    detail_path = suite_dir / "tables" / "final_detail.csv"
    if detail_path.exists() and detail_path.stat().st_size:
        frame = pd.read_csv(detail_path)
        by_count = frame.groupby(["run_name", "mode", "gold_count"], as_index=False).agg(
            primary_accuracy=("primary_accuracy", "mean"),
            enumeration_accuracy=("enumeration_accuracy", "mean"),
            trace_marker_accuracy=("trace_marker_accuracy", "mean"),
            trace_exact_accuracy=("trace_exact_accuracy", "mean"),
            token_accuracy=("token_accuracy", "mean"),
            primary_mae=("primary_abs_error", "mean"),
        )
        atomic_csv(by_count, suite_dir / "tables" / "final_by_count.csv")
        by_band = frame.groupby(["run_name", "mode", "count_band"], as_index=False).agg(
            n_examples=("primary_accuracy", "size"),
            primary_accuracy=("primary_accuracy", "mean"),
            enumeration_accuracy=("enumeration_accuracy", "mean"),
            trace_marker_accuracy=("trace_marker_accuracy", "mean"),
            trace_exact_accuracy=("trace_exact_accuracy", "mean"),
            token_accuracy=("token_accuracy", "mean"),
            primary_mae=("primary_abs_error", "mean"),
        )
        atomic_csv(by_band, suite_dir / "tables" / "final_by_band.csv")
