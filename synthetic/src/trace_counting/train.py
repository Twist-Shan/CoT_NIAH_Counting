from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .dataset import TraceCollator, dataset_for_split
from .io_utils import append_jsonl, ensure_dir, save_json, save_yaml
from .loss_masks import ID_TO_SEGMENT
from .model import build_model_from_config, load_model_config
from .tokenizer import VocabTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = loss_weights[:, 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    ce = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = (shift_labels != -100).float()
    weighted_valid = valid * shift_weights
    numerator = (ce * weighted_valid).sum()
    denominator = weighted_valid.sum()
    loss = numerator / denominator.clamp_min(1.0)
    stats: dict[str, Any] = {
        "total_weighted_loss": float(loss.detach().cpu()),
        "weighted_tokens": float(denominator.detach().cpu()),
    }

    if segment_ids is not None:
        shift_segments = segment_ids[:, 1:].contiguous()
        for segment_id, name in ID_TO_SEGMENT.items():
            mask = (shift_segments == segment_id).float() * weighted_valid
            denom = mask.sum()
            if float(denom.detach().cpu()) == 0.0:
                stats[name] = None
            else:
                stats[name] = float(((ce * mask).sum() / denom).detach().cpu())
    return loss, stats


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def evaluate_weighted_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    max_batches: int = 20,
    use_bf16: bool = False,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, list[float]] = {"total_weighted_loss": []}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            _, stats = weighted_causal_lm_loss(
                outputs.logits,
                batch["labels"],
                batch["loss_weights"],
                batch["segment_ids"],
            )
        for key, value in stats.items():
            if key == "weighted_tokens" or value is None:
                continue
            totals.setdefault(key, []).append(float(value))
    model.train()
    return {f"val_{key}": float(np.mean(values)) if values else None for key, values in totals.items()}


@torch.no_grad()
def quick_teacher_forced_accuracy(
    model: torch.nn.Module,
    examples: list[dict],
    tokenizer: VocabTokenizer,
    device: str,
    *,
    limit: int = 512,
    use_bf16: bool = False,
) -> float:
    from .eval import is_repeat_count_format, teacher_forced_predict_repeated_count

    model.eval()
    count_ids = torch.tensor(tokenizer.count_token_ids, dtype=torch.long, device=device)
    correct = 0
    total = 0
    for example in examples[:limit]:
        if is_repeat_count_format(example.get("task_format", "think_trace")):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = teacher_forced_predict_repeated_count(model, tokenizer, example, device)
            correct += int(bool(pred["correct"]))
            total += 1
            continue
        ans_idx = example["spans"]["ans_idx"]
        input_ids = torch.tensor([tokenizer.encode(example["full_tokens"][: ans_idx + 1])], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
        pred_count = int(torch.argmax(logits.index_select(0, count_ids)).item())
        correct += int(pred_count == int(example["count"]))
        total += 1
    model.train()
    return correct / max(total, 1)


def save_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    out_dir: Path,
    name: str,
) -> Path:
    checkpoint_dir = ensure_dir(out_dir / "checkpoints" / name)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save(checkpoint_dir / "vocab.json")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


def _run_label(args: argparse.Namespace, model_config: dict) -> str:
    model_name = model_config.get("model_name", "model")
    weighted = ""
    if args.loss_mask in {"full_sequence_final_weighted", "completion_final_weighted"}:
        weighted = f", fw={args.final_weight:g}"
    return f"{model_name} | seed={args.seed} | {args.loss_mask}{weighted}"


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    tokenizer.save(out_dir / "vocab.json")
    model_config = load_model_config(args.model_config)
    model = build_model_from_config(model_config, tokenizer).to(device)
    model.train()
    run_label = _run_label(args, model_config)

    train_dataset = dataset_for_split(
        data_dir,
        "train",
        tokenizer,
        loss_mask=args.loss_mask,
        final_weight=args.final_weight,
        eos_weight=args.eos_weight,
        final_count_only_include_eos=args.final_count_only_include_eos,
    )
    val_dataset = dataset_for_split(
        data_dir,
        "val_id",
        tokenizer,
        loss_mask=args.loss_mask,
        final_weight=args.final_weight,
        eos_weight=args.eos_weight,
        final_count_only_include_eos=args.final_count_only_include_eos,
        limit=args.eval_limit,
    )
    collator = TraceCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    use_bf16 = bool(
        device.startswith("cuda") and torch.cuda.is_bf16_supported() and args.precision == "bf16_if_available_else_fp32"
    )

    run_config = {
        "data_dir": str(data_dir),
        "model_config": str(args.model_config),
        "model": model_config,
        "loss_mask": args.loss_mask,
        "final_weight": args.final_weight,
        "eos_weight": args.eos_weight,
        "final_count_only_include_eos": args.final_count_only_include_eos,
        "seed": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "precision": "bf16" if use_bf16 else "fp32",
    }
    save_yaml(run_config, out_dir / "config.yaml")
    save_json(run_config, out_dir / "config.json")
    metadata_path = data_dir / "dataset_metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, out_dir / "dataset_metadata.json")

    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(
        "\n".join(
            [
                "",
                "=" * 88,
                f"Training: {run_label}",
                f"Data: {data_dir}",
                f"Output: {out_dir}",
                f"Device: {device} | precision: {'bf16' if use_bf16 else 'fp32'}",
                f"Steps: {args.max_steps} | batch_size: {args.batch_size} | train_examples: {len(train_dataset)}",
                "=" * 88,
            ]
        ),
        flush=True,
    )

    step = 0
    micro_step = 0
    last_stats: dict[str, Any] = {}
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc=run_label, dynamic_ncols=True, leave=True)
    while step < args.max_steps:
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss, stats = weighted_causal_lm_loss(
                    outputs.logits,
                    batch["labels"],
                    batch["loss_weights"],
                    batch["segment_ids"],
                )
                scaled_loss = loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            scaled_loss.backward()
            micro_step += 1
            last_stats = stats
            if micro_step % args.grad_accum_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)

            should_log = step == 1 or step % args.log_every == 0 or step == args.max_steps
            should_eval = step % args.eval_every == 0 or step == args.max_steps
            lr = float(scheduler.get_last_lr()[0])
            should_print_progress = (
                step == 1
                or step == args.max_steps
                or (args.progress_every > 0 and step % args.progress_every == 0)
            )
            if should_log:
                progress.set_postfix(
                    {
                        "loss": f"{last_stats['total_weighted_loss']:.4f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{float(grad_norm.detach().cpu()):.2f}",
                    },
                    refresh=False,
                )
            if should_print_progress:
                elapsed = time.time() - start_time
                steps_per_sec = step / max(elapsed, 1e-9)
                remaining = max(args.max_steps - step, 0) / max(steps_per_sec, 1e-9)
                print(
                    "[train] "
                    f"{run_label} step={step}/{args.max_steps} "
                    f"loss={last_stats['total_weighted_loss']:.4f} "
                    f"lr={lr:.2e} "
                    f"{steps_per_sec:.2f} step/s "
                    f"eta={remaining / 60:.1f}m",
                    flush=True,
                )
            if should_log or should_eval:
                row = {
                    "step": step,
                    "epoch_fraction": step * args.batch_size / max(len(train_dataset), 1),
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "elapsed_sec": time.time() - start_time,
                    **last_stats,
                }
                if should_eval:
                    progress.write(f"[eval] {run_label} step={step}: running validation...")
                    row.update(evaluate_weighted_loss(model, val_loader, device, max_batches=args.eval_batches, use_bf16=use_bf16))
                    row["val_tf_count_acc"] = quick_teacher_forced_accuracy(
                        model,
                        val_dataset.examples,
                        tokenizer,
                        device,
                        limit=args.eval_limit,
                        use_bf16=use_bf16,
                    )
                    progress.write(
                        "[eval] "
                        f"{run_label} step={step}: "
                        f"val_loss={row.get('val_total_weighted_loss')} "
                        f"val_tf_acc={row.get('val_tf_count_acc')}"
                    )
                    progress.set_postfix(
                        {
                            "loss": f"{last_stats['total_weighted_loss']:.4f}",
                            "val": (
                                "nan"
                                if row.get("val_total_weighted_loss") is None
                                else f"{row['val_total_weighted_loss']:.4f}"
                            ),
                            "tf": f"{row['val_tf_count_acc']:.3f}",
                            "lr": f"{lr:.2e}",
                        },
                        refresh=False,
                    )
                append_jsonl(row, log_path)

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_dir = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    out_dir=out_dir,
                    name=f"step_{step:08d}",
                )
                progress.write(f"[save] {run_label} step={step}: {checkpoint_dir}")
            if step >= args.max_steps:
                break
    progress.close()
    final_dir = save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        out_dir=out_dir,
        name="final",
    )
    print(f"saved final checkpoint to {final_dir}")
    return final_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GPT-2 style decoder from scratch on trace-counting data.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--loss_mask", required=True)
    parser.add_argument("--final_weight", type=float, default=10.0)
    parser.add_argument("--eos_weight", type=float, default=1.0)
    parser.add_argument("--final_count_only_include_eos", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--eval_limit", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--precision", default="bf16_if_available_else_fp32", choices=["bf16_if_available_else_fp32", "fp32"])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
