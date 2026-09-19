from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import V10Config
from .core import (
    Example,
    Rendered,
    Vocab,
    atomic_torch_save,
    balanced_examples,
    build_model,
    collate,
    component_loss_values,
    count_bin,
    count_prediction,
    learning_rate,
    make_example,
    render,
    shifted_token_losses,
    torch_load,
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _sync_artifacts(paths: list[Path], run_dir: Path, sync_run_dir: Path | None) -> None:
    if sync_run_dir is None:
        return
    for path in paths:
        if path.exists():
            _copy_atomic(path, sync_run_dir / path.relative_to(run_dir))


def _latest_checkpoint(mode_dir: Path, sync_mode_dir: Path | None) -> tuple[int, Path] | None:
    candidates = list(mode_dir.glob("step_*/checkpoint.pt"))
    if sync_mode_dir is not None and sync_mode_dir.exists():
        candidates.extend(sync_mode_dir.glob("step_*/checkpoint.pt"))
    valid: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            valid.append((int(path.parent.name.removeprefix("step_")), path))
        except ValueError:
            continue
    return max(valid, key=lambda item: item[0]) if valid else None


def _save_checkpoint(
    model,
    optimizer: AdamW,
    cfg: V10Config,
    vocab: Vocab,
    mode: str,
    step: int,
    rng: random.Random,
    run_dir: Path,
    sync_run_dir: Path | None,
    *,
    label: str | None = None,
) -> Path:
    directory = run_dir / "checkpoints" / mode / (label or f"step_{step:06d}")
    path = directory / "checkpoint.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "vocab": {
            "token_to_id": vocab.token_to_id,
            "id_to_token": vocab.id_to_token,
            "numbers": vocab.numbers,
            "markers": vocab.markers,
            "noise": vocab.noise,
        },
        "mode": mode,
        "step": int(step),
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_torch_save(payload, path)
    latest = run_dir / "checkpoints" / mode / "latest.json"
    latest.write_text(
        json.dumps({"step": int(step), "checkpoint": str(path.relative_to(run_dir))}, indent=2),
        encoding="utf-8",
    )
    _sync_artifacts([path, latest], run_dir, sync_run_dir)
    return path


def _sample_training_batch(
    cfg: V10Config,
    vocab: Vocab,
    mode: str,
    rng: random.Random,
) -> list[Rendered]:
    return [render(make_example(cfg, vocab, rng), vocab, mode) for _ in range(cfg.batch_size)]


def _parse_generated(tokens: list[str], mode: str, vocab: Vocab, example: Example) -> dict[str, Any]:
    count = None
    if "<Ans>" in tokens:
        pos = tokens.index("<Ans>")
        if pos + 1 < len(tokens):
            token = tokens[pos + 1]
            if token in vocab.numbers:
                count = vocab.numbers.index(token) + 1
    trace: list[str] = []
    if mode == "thinking":
        start = tokens.index("<Think>") + 1 if "<Think>" in tokens else 0
        end = tokens.index("</Think>") if "</Think>" in tokens else len(tokens)
        trace = tokens[start:end]
    expected_trace = [
        token
        for k, marker in enumerate(example.needle_markers, start=1)
        for token in (vocab.number_token(k), marker)
    ]
    generated_markers = [token for token in trace if token in vocab.markers]
    expected_markers = list(example.needle_markers)
    marker_match = sum(
        int(index < len(generated_markers) and generated_markers[index] == marker)
        for index, marker in enumerate(expected_markers)
    )
    return {
        "ar_pred_count": count,
        "ar_accuracy": float(count == example.count),
        "ar_abs_error": abs(count - example.count) if count is not None else np.nan,
        "trace_exact": float(trace == expected_trace) if mode == "thinking" else np.nan,
        "trace_marker_recall": marker_match / max(1, len(expected_markers)) if mode == "thinking" else np.nan,
        "generated_tokens": " ".join(tokens),
    }


@torch.no_grad()
def autoregressive_batch(
    model,
    cfg: V10Config,
    vocab: Vocab,
    mode: str,
    examples: list[Example],
    batch_size: int = 32,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        if mode == "nonthinking":
            prefixes = [["<BOS>", *ex.seq_tokens, "<Ans>"] for ex in chunk]
            max_new = 2
        else:
            prefixes = [["<BOS>", *ex.seq_tokens, "<Think>"] for ex in chunk]
            max_new = 2 * cfg.count_max + 5
        generated = torch.tensor([vocab.encode(prefix) for prefix in prefixes], dtype=torch.long, device=cfg.device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        for _ in range(max_new):
            logits = model(input_ids=generated).logits[:, -1]
            next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat([generated, next_ids[:, None]], dim=1)
            for row_idx, token_id in enumerate(next_ids.tolist()):
                if done[row_idx]:
                    continue
                tokens = vocab.decode(generated[row_idx].tolist())
                parsed = _parse_generated(tokens, mode, vocab, chunk[row_idx])
                if parsed["ar_pred_count"] is not None or token_id == vocab.eos_id:
                    done[row_idx] = True
            if bool(done.all()):
                break
        for row_idx, ex in enumerate(chunk):
            tokens = vocab.decode(generated[row_idx].tolist())
            rows.append({"count": ex.count, "count_bin": count_bin(ex.count), **_parse_generated(tokens, mode, vocab, ex)})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model(
    model,
    cfg: V10Config,
    vocab: Vocab,
    mode: str,
    examples: list[Example],
    *,
    step: int,
    run_ar: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    component_parts: dict[str, list[float]] = {}
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render(ex, vocab, mode) for ex in chunk]
        ids, labels, attention_mask = collate(rendered, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=attention_mask)
        total_loss, token_losses = shifted_token_losses(output.logits, labels)
        components = component_loss_values(token_losses, rendered)
        component_parts.setdefault("total", []).append(float(total_loss.detach().cpu()))
        for name, value in components.items():
            component_parts.setdefault(name, []).append(value)
        for row_idx, (ex, item) in enumerate(zip(chunk, rendered)):
            logits = output.logits[row_idx, item.spans.ans_pos]
            pred, expected, _ = count_prediction(logits, vocab)
            marker_correct: list[float] = []
            index_correct: list[float] = []
            if mode == "thinking":
                for index, marker_pos in enumerate(item.spans.trace_marker_positions):
                    query_pos = item.spans.trace_index_positions[index]
                    marker_correct.append(float(int(output.logits[row_idx, query_pos].argmax()) == item.input_ids[marker_pos]))
                for index, index_pos in enumerate(item.spans.trace_index_positions):
                    query_pos = item.spans.think_pos if index == 0 else item.spans.trace_marker_positions[index - 1]
                    index_correct.append(float(int(output.logits[row_idx, query_pos].argmax()) == item.input_ids[index_pos]))
            rows.append(
                {
                    "step": step,
                    "mode": mode,
                    "example_idx": start + row_idx,
                    "count": ex.count,
                    "count_bin": count_bin(ex.count),
                    "tf_pred_count": pred,
                    "tf_expected_count": expected,
                    "tf_accuracy": float(pred == ex.count),
                    "tf_abs_error": abs(pred - ex.count),
                    "tf_trace_marker_accuracy": float(np.mean(marker_correct)) if marker_correct else np.nan,
                    "tf_trace_index_accuracy": float(np.mean(index_correct)) if index_correct else np.nan,
                }
            )
    detail = pd.DataFrame(rows)
    if run_ar:
        selected: list[Example] = []
        seen = {count: 0 for count in range(cfg.count_min, cfg.count_max + 1)}
        for ex in examples:
            if seen[ex.count] < cfg.ar_examples_per_count:
                seen[ex.count] += 1
                selected.append(ex)
        ar = autoregressive_batch(model, cfg, vocab, mode, selected)
        ar.insert(0, "ar_example_idx", range(len(ar)))
        ar.insert(0, "step", step)
        ar.insert(1, "mode", mode)
        ar_summary = ar.groupby(["count", "count_bin"], as_index=False).agg(
            ar_n=("ar_accuracy", "size"),
            ar_accuracy=("ar_accuracy", "mean"),
            ar_mae=("ar_abs_error", "mean"),
            trace_exact=("trace_exact", "mean"),
            trace_marker_recall=("trace_marker_recall", "mean"),
        )
        detail = detail.merge(ar_summary, on=["count", "count_bin"], how="left")
    else:
        for column in ("ar_n", "ar_accuracy", "ar_mae", "trace_exact", "trace_marker_recall"):
            detail[column] = np.nan
    loss_rows = [
        {
            "step": step,
            "mode": mode,
            "component": component,
            "loss": float(np.mean(values)),
            "n_batches": len(values),
        }
        for component, values in component_parts.items()
    ]
    model.train()
    return detail, pd.DataFrame(loss_rows)


def summarize_eval(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_count = (
        detail.groupby(["step", "mode", "count", "count_bin"], as_index=False)
        .agg(
            n=("tf_accuracy", "size"),
            tf_accuracy=("tf_accuracy", "mean"),
            tf_mae=("tf_abs_error", "mean"),
            tf_trace_marker_accuracy=("tf_trace_marker_accuracy", "mean"),
            tf_trace_index_accuracy=("tf_trace_index_accuracy", "mean"),
            ar_n=("ar_n", "max"),
            ar_accuracy=("ar_accuracy", "max"),
            ar_mae=("ar_mae", "max"),
            trace_exact=("trace_exact", "max"),
            trace_marker_recall=("trace_marker_recall", "max"),
        )
        .sort_values(["mode", "step", "count"])
    )
    by_bin = (
        by_count.groupby(["step", "mode", "count_bin"], as_index=False)
        .agg(
            count_min=("count", "min"),
            count_max=("count", "max"),
            tf_accuracy=("tf_accuracy", "mean"),
            tf_mae=("tf_mae", "mean"),
            ar_accuracy=("ar_accuracy", "mean"),
            ar_mae=("ar_mae", "mean"),
            trace_exact=("trace_exact", "mean"),
            trace_marker_recall=("trace_marker_recall", "mean"),
        )
        .sort_values(["mode", "step", "count_min"])
    )
    return by_count, by_bin


def train_mode(
    cfg: V10Config,
    vocab: Vocab,
    mode: str,
    run_dir: Path,
    *,
    sync_run_dir: Path | None = None,
    skip_completed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_dir = run_dir / "checkpoints" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    final_path = mode_dir / "final" / "model.pt"
    train_path = run_dir / "tables" / f"train_{mode}.csv"
    eval_detail_path = run_dir / "tables" / f"eval_dynamics_examples_{mode}.csv"
    eval_loss_path = run_dir / "tables" / f"eval_dynamics_losses_{mode}.csv"
    if skip_completed and final_path.exists():
        return (
            pd.read_csv(train_path) if train_path.exists() else pd.DataFrame(),
            pd.read_csv(eval_detail_path) if eval_detail_path.exists() else pd.DataFrame(),
            pd.read_csv(eval_loss_path) if eval_loss_path.exists() else pd.DataFrame(),
        )

    torch.manual_seed(cfg.seed + (0 if mode == "nonthinking" else 17))
    rng = random.Random(cfg.seed + (100 if mode == "nonthinking" else 200))
    model = build_model(cfg, vocab).train()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    train_rows: list[dict[str, Any]] = []
    eval_frames: list[pd.DataFrame] = []
    eval_loss_frames: list[pd.DataFrame] = []
    start_step = 0
    best_score = -float("inf")
    stale_evals = 0

    sync_mode_dir = sync_run_dir / "checkpoints" / mode if sync_run_dir is not None else None
    latest = _latest_checkpoint(mode_dir, sync_mode_dir)
    if latest is not None:
        start_step, source = latest
        local = mode_dir / f"step_{start_step:06d}" / "checkpoint.pt"
        if source != local:
            _copy_atomic(source, local)
        payload = torch_load(local, cfg.device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        rng.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        if train_path.exists():
            train_rows = pd.read_csv(train_path).query("step <= @start_step").to_dict("records")
        if eval_detail_path.exists():
            eval_frames = [pd.read_csv(eval_detail_path).query("step <= @start_step")]
        if eval_loss_path.exists():
            eval_loss_frames = [pd.read_csv(eval_loss_path).query("step <= @start_step")]
        print(f"[v10:{mode}] resume step {start_step} from {source}", flush=True)

    eval_examples = balanced_examples(cfg, vocab, cfg.eval_examples_per_count, cfg.seed + 50_000)
    if start_step == 0 and not eval_frames:
        baseline_detail, baseline_losses = evaluate_model(
            model,
            cfg,
            vocab,
            mode,
            eval_examples,
            step=0,
            run_ar=False,
        )
        eval_frames.append(baseline_detail)
        eval_loss_frames.append(baseline_losses)
        _atomic_csv(baseline_detail, eval_detail_path)
        _atomic_csv(baseline_losses, eval_loss_path)
    pbar = tqdm(range(start_step + 1, cfg.train_steps + 1), initial=start_step, total=cfg.train_steps, desc=f"v10 {mode}")
    for step in pbar:
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        component_accumulator: dict[str, list[float]] = {}
        total_value = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = _sample_training_batch(cfg, vocab, mode, rng)
            ids, labels, attention_mask = collate(batch, vocab, cfg.device)
            output = model(input_ids=ids, attention_mask=attention_mask)
            loss, token_losses = shifted_token_losses(output.logits, labels)
            (loss / cfg.grad_accum_steps).backward()
            total_value += float(loss.detach().cpu()) / cfg.grad_accum_steps
            for name, value in component_loss_values(token_losses, batch).items():
                component_accumulator.setdefault(name, []).append(value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        should_log = step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps
        should_eval = step % cfg.eval_every == 0 or step == cfg.train_steps
        should_checkpoint = step % cfg.checkpoint_every == 0 or step == cfg.train_steps
        if should_log:
            row: dict[str, Any] = {"step": step, "mode": mode, "total_loss": total_value, "lr": lr}
            row.update({f"{name}_loss": float(np.mean(values)) for name, values in component_accumulator.items()})
            train_rows.append(row)
            pbar.set_postfix(loss=f"{total_value:.4f}", lr=f"{lr:.1e}")
        if should_eval:
            run_ar = step % cfg.ar_eval_every == 0 or step == cfg.train_steps
            detail, losses = evaluate_model(model, cfg, vocab, mode, eval_examples, step=step, run_ar=run_ar)
            eval_frames.append(detail)
            eval_loss_frames.append(losses)
            all_eval = pd.concat(eval_frames, ignore_index=True)
            all_eval_loss = pd.concat(eval_loss_frames, ignore_index=True)
            _atomic_csv(all_eval, eval_detail_path)
            _atomic_csv(all_eval_loss, eval_loss_path)
            score = float(detail.groupby("count_bin")["tf_accuracy"].mean().mean())
            if score > best_score + cfg.early_stop_min_delta:
                best_score = score
                stale_evals = 0
                _save_checkpoint(model, optimizer, cfg, vocab, mode, step, rng, run_dir, sync_run_dir, label="best")
            else:
                stale_evals += 1
        if should_checkpoint:
            _save_checkpoint(model, optimizer, cfg, vocab, mode, step, rng, run_dir, sync_run_dir)
            _atomic_csv(pd.DataFrame(train_rows), train_path)
            _sync_artifacts([train_path, eval_detail_path, eval_loss_path], run_dir, sync_run_dir)
        if cfg.early_stop_patience > 0 and stale_evals >= cfg.early_stop_patience:
            print(f"[v10:{mode}] optional early stop at step {step}", flush=True)
            break

    final_payload = {"model_state_dict": model.state_dict(), "config": cfg.to_dict(), "mode": mode}
    atomic_torch_save(final_payload, final_path)
    _atomic_csv(pd.DataFrame(train_rows), train_path)
    final_eval = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    final_eval_loss = pd.concat(eval_loss_frames, ignore_index=True) if eval_loss_frames else pd.DataFrame()
    _atomic_csv(final_eval, eval_detail_path)
    _atomic_csv(final_eval_loss, eval_loss_path)
    _sync_artifacts([final_path, train_path, eval_detail_path, eval_loss_path], run_dir, sync_run_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(train_rows), final_eval, final_eval_loss


def load_final_model(cfg: V10Config, vocab: Vocab, run_dir: Path, mode: str):
    model = build_model(cfg, vocab)
    path = run_dir / "checkpoints" / mode / "final" / "model.pt"
    payload = torch_load(path, cfg.device)
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def train_both_models(
    cfg: V10Config,
    vocab: Vocab,
    run_dir: Path,
    *,
    sync_run_dir: Path | None = None,
    skip_completed: bool = True,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    all_eval: list[pd.DataFrame] = []
    all_losses: list[pd.DataFrame] = []
    for mode in cfg.modes:
        train, detail, losses = train_mode(
            cfg,
            vocab,
            mode,
            run_dir,
            sync_run_dir=sync_run_dir,
            skip_completed=skip_completed,
        )
        outputs[f"train_{mode}"] = train
        all_eval.append(detail)
        all_losses.append(losses)
    detail = pd.concat(all_eval, ignore_index=True)
    losses = pd.concat(all_losses, ignore_index=True)
    by_count, by_bin = summarize_eval(detail)
    _atomic_csv(detail, run_dir / "tables" / "eval_dynamics_examples.csv")
    _atomic_csv(losses, run_dir / "tables" / "eval_dynamics_losses.csv")
    _atomic_csv(by_count, run_dir / "tables" / "eval_dynamics_by_count.csv")
    _atomic_csv(by_bin, run_dir / "tables" / "eval_dynamics_by_bin.csv")
    outputs.update({"eval_examples": detail, "eval_losses": losses, "eval_by_count": by_count, "eval_by_bin": by_bin})
    return outputs
