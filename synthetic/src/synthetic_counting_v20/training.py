from __future__ import annotations

import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import V20Config, config_from_dict
from .data import (
    CorpusSplit,
    V20Example,
    V20Rendered,
    V20Vocab,
    collate_v20,
    collate_v20_loss_weights,
    component_target_positions,
    load_corpus_text,
    make_training_example,
    make_v20_example,
    render_v20,
    shifted_v20_token_losses,
)
from .model import TinyPositionCausalLM, build_model
from .needle_pool import NeedlePool, load_needle_pool
from .timing import record_duration_event, timed_event


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _append_unique(path: Path, frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.empty:
        return
    combined = pd.concat((_read_table(path), frame), ignore_index=True)
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    atomic_csv(combined, path)


def sync_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            _copy_if_changed(path, target)


def learning_rate(cfg: V20Config, step: int) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    decay_steps = cfg.train_steps if cfg.lr_decay_steps is None else cfg.lr_decay_steps
    progress = (step - cfg.warmup_steps) / max(1, decay_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def training_loss_phase(cfg: V20Config, step: int) -> str:
    """Return the objective phase for an absolute optimizer step."""

    return "all_sequence" if step <= cfg.max_steps_for_language_pred else "task_output"


def scheduled_sampling_probability(cfg: V20Config, step: int, mode: str) -> float:
    """Return the predeclared linear roll-in probability for this update.

    Non-thinking has no generated continuation input between its fixed
    ``<Ans>`` prefix and the answer target, so the same rule degenerates to
    ordinary teacher forcing for that mode.  Thinking ramps linearly from zero
    immediately after the language-model phase to the configured maximum at
    the final optimizer step.
    """

    maximum = float(cfg.task_output_scheduled_sampling_max_probability)
    if mode != "thinking" or maximum == 0.0 or step <= cfg.max_steps_for_language_pred:
        return 0.0
    task_steps = max(1, cfg.train_steps - cfg.max_steps_for_language_pred)
    progress = (step - cfg.max_steps_for_language_pred) / task_steps
    return maximum * min(1.0, max(0.0, float(progress)))


@torch.no_grad()
def scheduled_sampling_inputs(
    model: TinyPositionCausalLM,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rendered: list[V20Rendered],
    probability: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Roll model predictions into generated continuation input positions.

    The first pass predicts token ``t`` from the original prefix through
    ``t-1``.  Selected continuation inputs are replaced by that prediction;
    the second, gradient-carrying pass still uses the original gold labels.
    Thus neither the serialized trace nor any supervision target changes.

    Eligible positions begin immediately after the fixed ``<Think>`` prefix
    and end at ``<Ans>`` (inclusive).  The final count itself is a target, not
    a roll-in input.  Non-thinking examples have no eligible positions.
    """

    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("scheduled-sampling probability must be finite and in [0, 1]")
    if ids.shape != attention_mask.shape or ids.ndim != 2:
        raise ValueError("ids and attention_mask must be identically shaped rank-2 tensors")
    if ids.shape[0] != len(rendered):
        raise ValueError("rendered batch size must match ids")

    eligible = torch.zeros_like(ids, dtype=torch.bool)
    for row, item in enumerate(rendered):
        if item.spans is None or item.mode != "thinking":
            continue
        if item.spans.think_pos is None:
            raise ValueError("Thinking rows require a <Think> position")
        start = int(item.spans.think_pos) + 1
        stop = int(item.spans.ans_pos) + 1
        eligible[row, start:stop] = True
    eligible &= attention_mask.bool()
    eligible_count = int(eligible.sum().item())
    empty_stats: dict[str, float | int] = {
        "eligible_tokens": eligible_count,
        "selected_tokens": 0,
        "changed_tokens": 0,
        "changed_fraction": 0.0,
    }
    if value == 0.0 or eligible_count == 0:
        return ids, empty_stats

    logits = model(input_ids=ids, attention_mask=attention_mask).logits
    if logits.shape[:2] != ids.shape:
        raise ValueError("model logits must align with ids in batch and sequence dimensions")
    predicted_next = logits[:, :-1].argmax(dim=-1)
    rollin_ids = torch.empty_like(ids)
    rollin_ids[:, 0] = ids[:, 0]
    rollin_ids[:, 1:] = predicted_next
    selected = eligible & (torch.rand(ids.shape, device=ids.device) < value)
    corrupted = ids.clone()
    corrupted[selected] = rollin_ids[selected]
    changed = selected & corrupted.ne(ids)
    selected_count = int(selected.sum().item())
    changed_count = int(changed.sum().item())
    return corrupted, {
        "eligible_tokens": eligible_count,
        "selected_tokens": selected_count,
        "changed_tokens": changed_count,
        "changed_fraction": changed_count / max(1, eligible_count),
    }


def component_normalized_task_output_loss(
    token_losses: torch.Tensor,
    active: torch.Tensor,
    loss_weights: torch.Tensor,
    rendered: list[V20Rendered],
    cfg: V20Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine per-example count, trace, and structure means.

    ``token_losses`` and ``active`` are shifted by one position relative to the
    rendered sequence, whereas ``loss_weights`` is unshifted.  Only positions
    selected by the task-output mask participate.  By default, trace
    delimiters and marker identities form one region; all remaining active
    output tokens (tags, ``<Ans>``, and ``<EOS>``) form the structure region.
    A grammar-balanced configuration instead keeps trace delimiters in
    structure and reserves trace for retrieved marker identities.

    Averaging within each region and example before the batch mean ensures that
    a count token has the same objective coefficient in Non-thinking and
    Thinking, regardless of trace length.
    """

    if token_losses.shape != active.shape:
        raise ValueError("token_losses and active must have identical shifted shapes")
    if loss_weights.shape != (token_losses.shape[0], token_losses.shape[1] + 1):
        raise ValueError("loss_weights must be unshifted by exactly one target position")
    if len(rendered) != token_losses.shape[0]:
        raise ValueError("rendered batch size must match token_losses")

    region_names = ("final_count", "trace", "structure")
    region_masks_cpu = torch.zeros(
        (len(region_names), token_losses.shape[0], token_losses.shape[1]),
        dtype=torch.bool,
    )
    thinking_rows = 0
    task_rows = 0
    for row, item in enumerate(rendered):
        if item.spans is None:
            continue
        task_rows += 1
        if item.mode == "thinking":
            thinking_rows += 1
        count_indices = [position - 1 for position in item.spans.count_positions]
        trace_positions = list(item.spans.trace_marker_positions)
        if not cfg.task_output_trace_delimiters_as_structure:
            trace_positions[:0] = [
                position
                for group in item.spans.trace_index_token_groups
                for position in group
            ]
        trace_indices = [position - 1 for position in trace_positions]
        output_start = (
            item.spans.ans_pos
            if item.mode == "nonthinking"
            else item.spans.think_pos
        )
        if output_start is None:
            raise ValueError("task-output rows require an output start position")
        region_masks_cpu[0, row, count_indices] = True
        if trace_indices:
            region_masks_cpu[1, row, trace_indices] = True
        region_masks_cpu[2, row, output_start - 1 : len(item.tokens) - 1] = True

    # Count and trace positions were initially included in the continuous
    # output range.  Removing them leaves tags, <Ans>, and <EOS> as structure.
    region_masks_cpu[2] &= ~(region_masks_cpu[0] | region_masks_cpu[1])
    cpu_valid_rows = region_masks_cpu.any(dim=-1)
    if (
        task_rows == 0
        or int(cpu_valid_rows[0].sum()) != task_rows
        or int(cpu_valid_rows[2].sum()) != task_rows
    ):
        raise ValueError("component-normalized task loss requires count and structure targets")
    if thinking_rows and int(cpu_valid_rows[1].sum()) != thinking_rows:
        raise ValueError("every Thinking row must contain an active trace region")

    objective_mask = ((loss_weights[:, 1:] > 0) & active).unsqueeze(0)
    region_masks = region_masks_cpu.to(device=token_losses.device) & objective_mask
    region_counts = region_masks.sum(dim=-1)
    valid_rows = region_counts > 0
    region_row_means = (
        (token_losses.unsqueeze(0) * region_masks.to(dtype=token_losses.dtype)).sum(dim=-1)
        / region_counts.clamp_min(1).to(dtype=token_losses.dtype)
    )
    region_means = {
        "final_count": region_row_means[0, valid_rows[0]].mean(),
        "structure": region_row_means[2, valid_rows[2]].mean(),
    }
    if thinking_rows:
        region_means["trace"] = region_row_means[1, valid_rows[1]].mean()
    total = (
        float(cfg.task_output_count_weight) * region_means["final_count"]
        + float(cfg.task_output_structure_weight) * region_means["structure"]
    )
    if "trace" in region_means:
        total = total + float(cfg.task_output_trace_weight) * region_means["trace"]
    return total, region_means


def answer_query_supervised_contrastive_loss(
    final_hidden: torch.Tensor,
    rendered: list[V20Rendered],
    *,
    temperature: float,
) -> torch.Tensor:
    """Compress count classes at the native ``<Ans>`` query representation.

    This is a training-only supervised contrastive objective.  It adds no
    decoder, parameters, tokens, or inference-time rule: the ordinary LM head
    must still predict the atomic answer token from the same query state.
    Anchors without a same-count positive in the current batch are ignored.
    """

    if final_hidden.ndim != 3 or final_hidden.shape[0] != len(rendered):
        raise ValueError("final_hidden must be [batch, sequence, hidden]")
    rows: list[int] = []
    positions: list[int] = []
    labels: list[int] = []
    for row, item in enumerate(rendered):
        if item.spans is None or item.count is None:
            continue
        rows.append(row)
        positions.append(int(item.spans.ans_pos))
        labels.append(int(item.count))
    if len(rows) < 2:
        return final_hidden.sum() * 0.0

    row_index = torch.tensor(rows, device=final_hidden.device, dtype=torch.long)
    position_index = torch.tensor(
        positions, device=final_hidden.device, dtype=torch.long
    )
    label = torch.tensor(labels, device=final_hidden.device, dtype=torch.long)
    query = final_hidden[row_index, position_index].float()
    query = F.normalize(query, dim=-1)
    scores = query @ query.transpose(0, 1)
    scores = scores / float(temperature)
    diagonal = torch.eye(len(rows), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diagonal, float("-inf"))
    log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    positives = label[:, None].eq(label[None, :]) & ~diagonal
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return final_hidden.sum() * 0.0
    positive_log_prob = torch.where(
        positives, log_prob, torch.zeros_like(log_prob)
    ).sum(dim=1)
    return -(positive_log_prob[valid] / positive_count[valid]).mean()


def _autocast_context(cfg: V20Config):
    if (
        cfg.precision == "bf16"
        and str(cfg.device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def paired_v20_model(
    cfg: V20Config, vocab: V20Vocab, position_encoding: str
) -> TinyPositionCausalLM:
    """Build the deterministic RoPE initialization shared by both modes/versions."""

    return build_model(cfg, vocab, position_encoding, cfg.device)


def _checkpoint_root(run_dir: Path, position_encoding: str, mode: str) -> Path:
    return run_dir / "checkpoints" / position_encoding / mode


def _latest_checkpoint(root: Path) -> tuple[int, Path] | None:
    path = root / "recovery" / "latest.pt"
    metadata = root / "latest.json"
    if not path.exists() or not metadata.exists():
        return None
    value = json.loads(metadata.read_text(encoding="utf-8"))
    return int(value["step"]), path


def checkpoint_steps(
    run_dir: str | Path, position_encoding: str, mode: str
) -> list[tuple[int, Path]]:
    """Return dense scientific snapshots as ``(step, shard_path)`` pairs."""

    root = _checkpoint_root(Path(run_dir), position_encoding, mode)
    index = root / "snapshot_index.csv"
    if not index.exists():
        return []
    frame = pd.read_csv(index)
    return [
        (int(row.step), root / str(row.shard))
        for row in frame.sort_values("step").itertuples(index=False)
    ]


def planned_checkpoint_steps(cfg: V20Config) -> list[int]:
    """Numeric snapshots: initialization, cadence, objective boundary, and final."""

    values = {0, int(cfg.train_steps)}
    values.update(range(cfg.checkpoint_every, cfg.train_steps + 1, cfg.checkpoint_every))
    if cfg.max_steps_for_language_pred <= cfg.train_steps:
        values.add(int(cfg.max_steps_for_language_pred))
    return sorted(values)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _copy_if_changed(source: Path, target: Path) -> None:
    """Copy one artifact atomically without rewriting an identical Drive file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_stat = source.stat()
        target_stat = target.stat()
        # Immutable snapshot shards are normally skipped by the size test.  For
        # mutable files such as recovery/latest.pt and CSV indexes, also require
        # the destination to be at least as new as the source; full checkpoints
        # often have identical byte sizes from one save to the next.
        if (
            target_stat.st_size == source_stat.st_size
            and target_stat.st_mtime_ns >= source_stat.st_mtime_ns
        ):
            return
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def _snapshot_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _compact_state_dict(model: TinyPositionCausalLM, dtype: str) -> dict[str, torch.Tensor]:
    target = _snapshot_dtype(dtype)
    return {
        name: tensor.detach().to(device="cpu", dtype=target if tensor.is_floating_point() else tensor.dtype).clone()
        for name, tensor in model.state_dict().items()
    }


class DenseSnapshotWriter:
    """Buffer 100-step model-only snapshots and write one shard per recovery window."""

    def __init__(
        self,
        root: Path,
        cfg: V20Config,
        position_encoding: str,
        mode: str,
        sync_run_dir: Path | None,
        run_dir: Path,
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.position_encoding = position_encoding
        self.mode = mode
        self.sync_run_dir = sync_run_dir
        self.run_dir = run_dir
        self.pending: dict[int, dict[str, torch.Tensor]] = {}
        self.index_path = root / "snapshot_index.csv"
        self.existing = set()
        if self.index_path.exists() and self.index_path.stat().st_size:
            self.existing = set(pd.read_csv(self.index_path)["step"].astype(int))

    def add(self, model: TinyPositionCausalLM, step: int, *, force: bool = False) -> None:
        step = int(step)
        if step in self.existing:
            return
        self.pending[step] = _compact_state_dict(model, self.cfg.snapshot_dtype)
        if force or step == 0 or step % self.cfg.snapshot_shard_every == 0 or step == self.cfg.train_steps:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        steps = sorted(self.pending)
        relative = Path("snapshots") / f"steps_{steps[0]:06d}_{steps[-1]:06d}.pt"
        path = self.root / relative
        payload = {
            "format": "v20_dense_model_snapshot_shard",
            "version": self.cfg.version,
            "position_encoding": self.position_encoding,
            "mode": self.mode,
            "snapshot_dtype": self.cfg.snapshot_dtype,
            "steps": steps,
            "model_state_dicts": {str(step): self.pending[step] for step in steps},
        }
        _atomic_torch_save(payload, path)
        old = _read_table(self.index_path)
        rows = pd.DataFrame(
            {
                "step": steps,
                "shard": [str(relative).replace("\\", "/")] * len(steps),
                "snapshot_dtype": [self.cfg.snapshot_dtype] * len(steps),
                "shard_bytes": [path.stat().st_size] * len(steps),
            }
        )
        combined = pd.concat((old, rows), ignore_index=True).drop_duplicates("step", keep="last")
        atomic_csv(combined.sort_values("step"), self.index_path)
        self.existing.update(steps)
        self.pending.clear()
        if self.sync_run_dir is not None:
            for source in (path, self.index_path):
                _copy_if_changed(source, self.sync_run_dir / source.relative_to(self.run_dir))


def load_dense_snapshot_state(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    step: int,
) -> dict[str, torch.Tensor]:
    root = _checkpoint_root(Path(run_dir), position_encoding, mode)
    index = pd.read_csv(root / "snapshot_index.csv")
    row = index[index["step"].astype(int) == int(step)]
    if row.empty:
        raise FileNotFoundError(f"dense snapshot step {step} is not indexed for {position_encoding}/{mode}")
    path = root / str(row.iloc[-1]["shard"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "v20_dense_model_snapshot_shard":
        raise ValueError(f"unrecognized dense snapshot format in {path}")
    if payload.get("position_encoding") != position_encoding or payload.get("mode") != mode:
        raise ValueError("dense snapshot shard identity does not match its index path")
    if int(step) not in {int(value) for value in payload.get("steps", ())}:
        raise ValueError(f"dense snapshot shard {path} does not declare step {step}")
    return dict(payload["model_state_dicts"][str(int(step))])


def _save_checkpoint(
    model: TinyPositionCausalLM,
    optimizer: AdamW,
    cfg: V20Config,
    vocab: V20Vocab,
    pool: NeedlePool,
    split: CorpusSplit,
    position_encoding: str,
    mode: str,
    step: int,
    rng: random.Random,
    run_dir: Path,
    sync_run_dir: Path | None,
    *,
    label: str | None = None,
    sampling_state: dict[str, Any] | None = None,
) -> Path:
    root = _checkpoint_root(run_dir, position_encoding, mode)
    if label == "final":
        path = root / "final" / "checkpoint.pt"
    elif label == "pinned":
        path = root / "pinned" / f"step_{step:06d}" / "checkpoint.pt"
    else:
        path = root / "recovery" / "latest.pt"
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
        "pool_fingerprint": pool.pool_fingerprint,
        "split_fingerprint": split.split_fingerprint,
        "sampling_state": sampling_state,
    }
    block = "checkpoint_write_final" if label == "final" else "checkpoint_write"
    with timed_event(
        run_dir,
        scope="training",
        block=block,
        position_encoding=position_encoding,
        mode=mode,
        step=step,
        device=cfg.device,
    ):
        detail_prefix = "checkpoint_final" if label == "final" else "checkpoint"
        with timed_event(
            run_dir,
            scope="training",
            block=f"{detail_prefix}_serialize",
            position_encoding=position_encoding,
            mode=mode,
            step=step,
            device=cfg.device,
        ):
            _atomic_torch_save(payload, path)
            latest = root / "latest.json"
            latest.write_text(
                json.dumps(
                    {"step": int(step), "checkpoint": str(path.relative_to(run_dir))},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if sync_run_dir is not None:
            with timed_event(
                run_dir,
                scope="training",
                block=f"{detail_prefix}_drive_sync",
                position_encoding=position_encoding,
                mode=mode,
                step=step,
                device=cfg.device,
            ):
                for source in (path, latest):
                    _copy_if_changed(source, sync_run_dir / source.relative_to(run_dir))
    return path


def _restore_checkpoint(
    payload: dict[str, Any],
    model: TinyPositionCausalLM,
    optimizer: AdamW,
    rng: random.Random,
    pool: NeedlePool,
    split: CorpusSplit,
    vocab: V20Vocab,
    cfg: V20Config,
    position_encoding: str,
    mode: str,
    expected_step: int,
) -> dict[str, Any] | None:
    payload_cfg = config_from_dict(dict(payload.get("config", {})))
    if replace(payload_cfg, device=cfg.device) != cfg:
        raise ValueError("checkpoint config does not match the requested training config")
    if payload.get("position_encoding") != position_encoding:
        raise ValueError("checkpoint position encoding does not match the requested variant")
    if payload.get("mode") != mode:
        raise ValueError("checkpoint mode does not match the requested variant")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("checkpoint payload step does not match its checkpoint directory")
    if payload.get("pool_fingerprint") != pool.pool_fingerprint:
        raise ValueError("checkpoint needle-pool fingerprint mismatch")
    if payload.get("split_fingerprint") != split.split_fingerprint:
        raise ValueError("checkpoint corpus-split fingerprint mismatch")
    if payload.get("vocab_fingerprint") != vocab.fingerprint:
        raise ValueError("checkpoint vocabulary fingerprint mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].detach().to("cpu", dtype=torch.uint8).contiguous())
    cuda_states = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all(
            [state.detach().to("cpu", dtype=torch.uint8).contiguous() for state in cuda_states]
        )
    state = payload.get("sampling_state")
    return dict(state) if isinstance(state, dict) else None


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 700 else math.inf


def _component_example_means(
    losses: torch.Tensor,
    rendered: list[V20Rendered],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row, item in enumerate(rendered):
        for name, positions in component_target_positions(item).items():
            active_positions = [position for position in positions if position > 0]
            if not active_positions:
                continue
            values = [float(losses[row, position - 1].detach().cpu()) for position in active_positions]
            result.setdefault(name, []).append(float(np.mean(values)))
    return result


@torch.no_grad()
def evaluate_loss_suite(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: list[V20Example],
    *,
    position_encoding: str,
    mode: str,
    step: int,
    curve_source: str,
    suite: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calculate the equal-sequence mean exactly, independent of evaluation batches."""

    model.eval()
    sequence_nll: list[float] = []
    sequence_tokens: list[int] = []
    kinds: list[str] = []
    component_values: dict[str, list[float]] = {}
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render_v20(example, vocab, mode) for example in chunk]
        ids, labels, attention_mask = collate_v20(rendered, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=attention_mask)
        _, losses, active = shifted_v20_token_losses(output.logits, labels)
        for row, item in enumerate(rendered):
            token_count = int(active[row].sum().detach().cpu())
            nll = float((losses[row] * active[row]).sum().detach().cpu())
            sequence_nll.append(nll)
            sequence_tokens.append(token_count)
            kinds.append(item.example_kind)
        for name, values in _component_example_means(losses, rendered).items():
            component_values.setdefault(name, []).extend(values)
    nll_array = np.asarray(sequence_nll, dtype=np.float64)
    token_array = np.asarray(sequence_tokens, dtype=np.int64)
    if np.any(token_array <= 0):
        raise RuntimeError("every loss-suite sequence must contain an active next-token target")
    per_example = nll_array / token_array
    token_weighted = float(nll_array.sum() / token_array.sum())
    example_mean = float(per_example.mean())
    is_task = np.asarray([kind == "counting_task" for kind in kinds])
    task_tokens = int(token_array[is_task].sum())
    row = {
        "step": int(step),
        "position_encoding": position_encoding,
        "mode": mode,
        "curve_source": curve_source,
        "source_region": examples[0].corpus_region,
        "suite": suite,
        "task_occurrence_ratio": cfg.task_occurrence_ratio,
        "realized_task_example_ratio": float(is_task.mean()),
        "realized_task_token_ratio": task_tokens / int(token_array.sum()),
        "num_examples": len(examples),
        "active_tokens": int(token_array.sum()),
        "token_weighted_cross_entropy": token_weighted,
        "example_mean_cross_entropy": example_mean,
        "token_weighted_perplexity": _perplexity(token_weighted),
        "example_mean_perplexity": _perplexity(example_mean),
    }
    components = [
        {
            "step": int(step),
            "position_encoding": position_encoding,
            "mode": mode,
            "curve_source": curve_source,
            "source_region": examples[0].corpus_region,
            "suite": suite,
            "component": name,
            "num_contributing_examples": len(values),
            "example_mean_cross_entropy": float(np.mean(values)),
        }
        for name, values in sorted(component_values.items())
    ]
    return row, components


def evaluate_curve_suites(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    curve_suites: dict[str, dict[str, list[V20Example]]],
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for source in ("train", "heldout"):
        for suite in ("raw", "task", "mixture"):
            row, components = evaluate_loss_suite(
                model,
                cfg,
                vocab,
                curve_suites[source][suite],
                position_encoding=position_encoding,
                mode=mode,
                step=step,
                curve_source=source,
                suite=suite,
            )
            rows.append(row)
            component_rows.extend(components)
    return pd.DataFrame(rows), pd.DataFrame(component_rows)


def _count_prediction(logits: torch.Tensor, vocab: V20Vocab) -> int:
    if vocab.count_tokenization != "atomic":
        raise ValueError("single-logit count prediction is defined only for atomic v20")
    number_ids = torch.tensor(vocab.number_ids, device=logits.device)
    return int(logits[number_ids].argmax()) + 1


def _teacher_forced_number(
    logits: torch.Tensor,
    item: V20Rendered,
    vocab: V20Vocab,
) -> tuple[int | None, float]:
    assert item.spans is not None
    predicted_tokens = [
        vocab.id_to_token[int(logits[position - 1].argmax())]
        for position in item.spans.count_positions
    ]
    predicted = vocab.decode_number_tokens(predicted_tokens)
    exact = float(
        all(
            int(logits[position - 1].argmax()) == item.input_ids[position]
            for position in item.spans.count_positions
        )
    )
    return predicted, exact


@torch.no_grad()
def teacher_forced_task_evaluation(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: list[V20Example],
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        rendered = [render_v20(example, vocab, mode) for example in chunk]
        ids, _, mask = collate_v20(rendered, vocab, cfg.device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for row_index, (example, item) in enumerate(zip(chunk, rendered)):
            assert item.spans is not None and example.count is not None
            predicted, final_exact = _teacher_forced_number(logits[row_index], item, vocab)
            marker_correct: list[float] = []
            query_correct: list[float] = []
            marker_correct_by_token: dict[str, list[float]] = {}
            if mode == "thinking":
                for marker_position in item.spans.trace_marker_positions:
                    correct = float(
                        int(logits[row_index, marker_position - 1].argmax())
                        == item.input_ids[marker_position]
                    )
                    marker_correct.append(correct)
                    marker_correct_by_token.setdefault(item.tokens[marker_position], []).append(correct)
                for group in item.spans.trace_query_token_groups:
                    query_correct.append(
                        float(
                            all(
                                int(logits[row_index, position - 1].argmax())
                                == item.input_ids[position]
                                for position in group
                            )
                        )
                    )
            baseline = min(cfg.count_max_threshold, max(1, round(cfg.seq_len * float(example.set_frequency_sum))))
            row = {
                "step": step,
                "position_encoding": position_encoding,
                "mode": mode,
                "example_id": start + row_index,
                "set_id": example.set_id,
                "count": example.count,
                "count_bin": cfg.count_bin(example.count),
                "corpus_region": example.corpus_region,
                "corpus_start": example.corpus_start,
                "corpus_end": example.corpus_end,
                "prompt_sha256": example.prompt_sha256,
                "set_frequency_sum": example.set_frequency_sum,
                "set_frequency_bin": example.set_frequency_bin,
                "filter_attempts": example.filter_attempts,
                "tf_pred_count": predicted,
                "tf_final_accuracy": final_exact,
                "tf_trace_marker_accuracy": float(np.mean(marker_correct)) if marker_correct else np.nan,
                "tf_trace_query_accuracy": float(np.mean(query_correct)) if query_correct else np.nan,
                "tf_trace_index_accuracy": (
                    float(np.mean(query_correct))
                    if query_correct and vocab.trace_format == "indexed"
                    else np.nan
                ),
                "tf_trace_delimiter_accuracy": (
                    float(np.mean(query_correct))
                    if query_correct and vocab.trace_format == "separator"
                    else np.nan
                ),
                "frequency_baseline_count": baseline,
                "frequency_baseline_accuracy": float(baseline == example.count),
            }
            for index, character in enumerate(example.needle_characters or (), start=1):
                row[f"character_{index}_codepoint"] = ord(character)
                row[f"character_{index}_count"] = (example.per_character_counts or ())[index - 1]
                values = marker_correct_by_token.get(
                    f"<CH_{ord(character):04X}>", []
                )
                row[f"character_{index}_trace_marker_accuracy"] = (
                    float(np.mean(values)) if values else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _parse_generation(tokens: list[str], vocab: V20Vocab, example: V20Example, mode: str) -> dict[str, Any]:
    predicted = None
    if "<Ans>" in tokens:
        index = tokens.index("<Ans>")
        number: list[str] = []
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor] in vocab.numbers:
            number.append(tokens[cursor])
            cursor += 1
        predicted = vocab.decode_number_tokens(number)
    trace: list[str] = []
    trace_started = mode == "thinking" and "<Think>" in tokens
    trace_closed = False
    if trace_started:
        start = tokens.index("<Think>") + 1
        trace_closed = "</Think>" in tokens[start:]
        end = tokens.index("</Think>", start) if trace_closed else len(tokens)
        trace = tokens[start:end]
    if vocab.trace_format == "separator":
        expected = [
            token
            for marker in example.needle_markers
            for token in ("<Sep>", marker)
        ]
    else:
        expected = [
            token
            for index, marker in enumerate(example.needle_markers, start=1)
            for token in (*vocab.number_tokens(index), marker)
        ]
    generated_markers: list[str] = []
    cursor = 0
    # A generation that never enters the trace is not a well-formed empty
    # trace.  Closure is reported separately so syntax and stopping can be
    # diagnosed independently.
    trace_format_valid = trace_started
    if vocab.trace_format == "separator":
        while cursor < len(trace):
            if trace[cursor] != "<Sep>" or cursor + 1 >= len(trace):
                trace_format_valid = False
                break
            marker = trace[cursor + 1]
            if marker not in vocab.character_tokens:
                trace_format_valid = False
                break
            generated_markers.append(marker)
            cursor += 2
    else:
        while cursor < len(trace):
            number: list[str] = []
            while cursor < len(trace) and trace[cursor] in vocab.numbers:
                number.append(trace[cursor])
                cursor += 1
            if not number or cursor >= len(trace):
                trace_format_valid = False
                break
            marker = trace[cursor]
            cursor += 1
            if marker not in vocab.character_tokens:
                trace_format_valid = False
                break
            generated_markers.append(marker)
    trace_format_valid = trace_format_valid and cursor == len(trace)
    matches = sum(
        int(index < len(generated_markers) and generated_markers[index] == marker)
        for index, marker in enumerate(example.needle_markers)
    )
    return {
        "ar_pred_count": predicted,
        "ar_answered": float(predicted is not None),
        "ar_accuracy": float(predicted == example.count),
        "ar_abs_error": abs(predicted - int(example.count)) if predicted is not None else np.nan,
        "ar_abs_error_with_missing_penalty": (
            abs(predicted - int(example.count)) if predicted is not None else int(example.count)
        ),
        "trace_exact": float(trace == expected) if mode == "thinking" else np.nan,
        "trace_ordered_marker_accuracy": (
            matches / max(1, int(example.count)) if mode == "thinking" else np.nan
        ),
        "trace_generated_marker_count": (
            len(generated_markers) if mode == "thinking" else np.nan
        ),
        "trace_marker_count_accuracy": (
            float(len(generated_markers) == int(example.count))
            if mode == "thinking"
            else np.nan
        ),
        "trace_format_valid": (
            float(trace_format_valid) if mode == "thinking" else np.nan
        ),
        "trace_closed": float(trace_closed) if mode == "thinking" else np.nan,
        "trace_delimiter_count": (
            trace.count("<Sep>")
            if mode == "thinking" and vocab.trace_format == "separator"
            else np.nan
        ),
        "trace_delimiter_count_accuracy": (
            float(trace.count("<Sep>") == int(example.count))
            if mode == "thinking" and vocab.trace_format == "separator"
            else np.nan
        ),
        # Backward-compatible alias. This is positional accuracy, not set-level recall.
        "trace_marker_recall": matches / max(1, int(example.count)) if mode == "thinking" else np.nan,
        "generated_tokens": " ".join(tokens),
    }


@torch.no_grad()
def autoregressive_task_evaluation(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: list[V20Example],
    *,
    position_encoding: str,
    mode: str,
    step: int,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    if not examples:
        return pd.DataFrame(rows)
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prefixes: list[list[str]] = []
        for example in chunk:
            item = render_v20(example, vocab, mode)
            assert item.spans is not None
            stop = item.spans.ans_pos + 1 if mode == "nonthinking" else item.spans.think_pos + 1
            prefixes.append(item.tokens[:stop])
        generated = torch.tensor([vocab.encode(tokens) for tokens in prefixes], device=cfg.device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = 4 if mode == "nonthinking" else cfg.max_render_len - len(prefixes[0]) + 2
        for _ in range(max_new_tokens):
            next_ids = model(input_ids=generated).logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            for index, token_id in enumerate(next_ids.tolist()):
                if token_id == vocab.eos_id:
                    done[index] = True
            if bool(done.all()):
                break
        for index, example in enumerate(chunk):
            rows.append(
                {
                    "step": step,
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "row_id": start + index,
                    "set_id": example.set_id,
                    "count": example.count,
                    "count_bin": cfg.count_bin(int(example.count)),
                    "corpus_region": example.corpus_region,
                    "corpus_start": example.corpus_start,
                    "prompt_sha256": example.prompt_sha256,
                    **_parse_generation(vocab.decode(generated[index]), vocab, example, mode),
                }
            )
    return pd.DataFrame(rows)


def _retain_joint_starts(
    relative_starts: np.ndarray,
    max_starts_per_cell: int | None,
) -> np.ndarray:
    """Apply the configured deterministic within-cell support policy."""

    if max_starts_per_cell is None or relative_starts.size <= max_starts_per_cell:
        return relative_starts
    retained = np.linspace(
        0,
        relative_starts.size - 1,
        max_starts_per_cell,
        dtype=np.int64,
    )
    return relative_starts[retained]


@dataclass(frozen=True)
class _JointSetCountSampler:
    cells: tuple[tuple[int, int], ...]
    probabilities: tuple[float, ...]
    starts: dict[tuple[int, int], np.ndarray]
    plan: pd.DataFrame


_JOINT_SET_COUNT_SAMPLER_CACHE: dict[tuple[Any, ...], _JointSetCountSampler] = {}


def _maximum_entropy_cell_probabilities(
    support: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
) -> np.ndarray:
    """Return the max-entropy table with uniform row and column marginals.

    ``support`` is a boolean set-by-count matrix. Structural-zero cells remain
    zero. Iterative proportional fitting is deterministic and either reaches
    the requested marginals or fails loudly, rather than silently dropping
    needle sets that do not support every count.
    """

    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or not support.size:
        raise ValueError("set-count support must be a nonempty 2D matrix")
    if not support.any(axis=1).all():
        raise ValueError("every needle set must support at least one count")
    if not support.any(axis=0).all():
        raise ValueError("every requested count must be supported by at least one set")
    row_target = np.full(support.shape[0], 1.0 / support.shape[0], dtype=np.float64)
    column_target = np.full(support.shape[1], 1.0 / support.shape[1], dtype=np.float64)
    probabilities = support.astype(np.float64)
    probabilities /= probabilities.sum()
    for _ in range(max_iterations):
        row_sums = probabilities.sum(axis=1)
        probabilities *= (row_target / row_sums)[:, None]
        column_sums = probabilities.sum(axis=0)
        probabilities *= (column_target / column_sums)[None, :]
        error = max(
            float(np.abs(probabilities.sum(axis=1) - row_target).max()),
            float(np.abs(probabilities.sum(axis=0) - column_target).max()),
        )
        if error <= tolerance:
            break
    else:
        raise RuntimeError(
            "feasible set-count support did not converge to uniform marginals; "
            f"final error={error:.3e}"
        )
    probabilities[~support] = 0.0
    probabilities /= probabilities.sum()
    return probabilities


def _joint_set_count_sampler(
    cfg: V20Config,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
) -> _JointSetCountSampler:
    key = (
        split.corpus_sha256,
        split.split_fingerprint,
        pool.pool_fingerprint,
        split.train.start,
        split.train.end,
        cfg.seq_len,
        cfg.count_max_threshold,
        cfg.joint_sampler_max_starts_per_cell,
    )
    cached = _JOINT_SET_COUNT_SAMPLER_CACHE.get(key)
    if cached is not None:
        return cached

    region = split.train
    region_text = text[region.start : region.end]
    if len(region_text) < cfg.seq_len:
        raise ValueError("training corpus region is shorter than seq_len")
    codepoints = np.fromiter(
        (ord(character) for character in region_text),
        dtype=np.int32,
        count=len(region_text),
    )
    support = np.zeros(
        (len(pool.sets), cfg.count_max_threshold),
        dtype=bool,
    )
    starts_by_cell: dict[tuple[int, int], np.ndarray] = {}
    full_window_counts: dict[tuple[int, int], int] = {}
    for set_index, needle_set in enumerate(pool.sets):
        membership = np.zeros(len(codepoints), dtype=bool)
        for character in needle_set.characters:
            membership |= codepoints == ord(character)
        prefix = np.empty(len(membership) + 1, dtype=np.int32)
        prefix[0] = 0
        np.cumsum(membership, dtype=np.int32, out=prefix[1:])
        window_counts = prefix[cfg.seq_len :] - prefix[: -cfg.seq_len]
        for count in range(1, cfg.count_max_threshold + 1):
            relative_starts = np.flatnonzero(window_counts == count).astype(
                np.int32,
                copy=False,
            )
            full_window_counts[(set_index, count)] = int(relative_starts.size)
            if not relative_starts.size:
                continue
            support[set_index, count - 1] = True
            relative_starts = _retain_joint_starts(
                relative_starts,
                cfg.joint_sampler_max_starts_per_cell,
            )
            starts_by_cell[(set_index, count)] = (
                relative_starts + int(region.start)
            ).astype(np.int32, copy=False)

    probabilities = _maximum_entropy_cell_probabilities(support)
    cells: list[tuple[int, int]] = []
    cell_probabilities: list[float] = []
    rows: list[dict[str, Any]] = []
    row_marginals = probabilities.sum(axis=1)
    count_marginals = probabilities.sum(axis=0)
    for set_index, needle_set in enumerate(pool.sets):
        for count in range(1, cfg.count_max_threshold + 1):
            cell = (set_index, count)
            probability = float(probabilities[set_index, count - 1])
            starts = starts_by_cell.get(cell)
            if probability > 0:
                if starts is None or not len(starts):
                    raise RuntimeError("positive set-count cell has no retained starts")
                cells.append(cell)
                cell_probabilities.append(probability)
            rows.append(
                {
                    "set_id": needle_set.set_id,
                    "set_index": set_index,
                    "count": count,
                    "feasible": bool(support[set_index, count - 1]),
                    "full_window_count": full_window_counts[cell],
                    "retained_window_count": 0 if starts is None else int(len(starts)),
                    "within_cell_sampling_policy": (
                        "all_legal_starts"
                        if cfg.joint_sampler_max_starts_per_cell is None
                        else (
                            "deterministic_evenly_spaced_cap_"
                            f"{cfg.joint_sampler_max_starts_per_cell}"
                        )
                    ),
                    "target_probability": probability,
                    "target_set_marginal": float(row_marginals[set_index]),
                    "target_count_marginal": float(count_marginals[count - 1]),
                }
            )
    sampler = _JointSetCountSampler(
        cells=tuple(cells),
        probabilities=tuple(cell_probabilities),
        starts=starts_by_cell,
        plan=pd.DataFrame(rows),
    )
    _JOINT_SET_COUNT_SAMPLER_CACHE[key] = sampler
    return sampler


def _maxent_set_count_training_examples(
    cfg: V20Config,
    vocab: V20Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    rng: random.Random,
) -> list[V20Example]:
    sampler = _joint_set_count_sampler(cfg, text, split, pool)
    selections = rng.choices(
        range(len(sampler.cells)),
        weights=sampler.probabilities,
        k=cfg.batch_size,
    )
    examples: list[V20Example] = []
    for selected in selections:
        set_index, target_count = sampler.cells[selected]
        starts = sampler.starts[(set_index, target_count)]
        start = int(starts[rng.randrange(len(starts))])
        example = make_v20_example(
            cfg,
            vocab,
            text,
            split,
            pool,
            rng,
            region_name="train",
            initial_start=start,
            needle_set=pool.sets[set_index],
        )
        if int(example.count or 0) != target_count:
            raise RuntimeError("joint sampler start index produced the wrong count")
        examples.append(example)
    return examples


def _training_batch(
    cfg: V20Config,
    vocab: V20Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    mode: str,
    rng: random.Random,
    *,
    require_task: bool = False,
) -> tuple[list[V20Example], list[V20Rendered]]:
    if cfg.training_count_distribution == "maxent_set_count":
        examples = _maxent_set_count_training_examples(
            cfg,
            vocab,
            text,
            split,
            pool,
            rng,
        )
        return examples, [render_v20(example, vocab, mode) for example in examples]
    if cfg.training_count_distribution == "uniform":
        # Draw the desired semantic counts first, then fill all count buckets
        # from one shared stream of natural candidates.  Sharing rejected
        # candidates across the whole batch is much cheaper than independently
        # rejection-sampling each row (roughly 2-4x baseline work here instead
        # of ~30x for counts 1..30).
        targets = [
            rng.randint(cfg.count_min, cfg.count_max_threshold)
            for _ in range(cfg.batch_size)
        ]
        needed = {count: targets.count(count) for count in set(targets)}
        buckets: dict[int, list[V20Example]] = {count: [] for count in needed}
        proposals = 0
        while any(len(buckets[count]) < amount for count, amount in needed.items()):
            example = make_training_example(cfg, vocab, text, split, pool, rng)
            proposals += 1
            count = int(example.count or 0)
            if count in needed and len(buckets[count]) < needed[count]:
                buckets[count].append(example)
            if proposals >= cfg.candidate_filter_max_attempts:
                missing = {
                    count: needed[count] - len(buckets[count])
                    for count in needed
                    if len(buckets[count]) < needed[count]
                }
                raise RuntimeError(
                    f"could not fill uniform-count training batch after {proposals} proposals: "
                    f"{missing}"
                )
        examples = [buckets[count].pop() for count in targets]
        return examples, [render_v20(example, vocab, mode) for example in examples]
    for _ in range(1_000):
        examples = [make_training_example(cfg, vocab, text, split, pool, rng) for _ in range(cfg.batch_size)]
        if not require_task or any(item.example_kind == "counting_task" for item in examples):
            return examples, [render_v20(example, vocab, mode) for example in examples]
    raise RuntimeError("could not sample a counting task for task-output-only training")


@torch.no_grad()
def prefix_permutation_consistency_evaluation(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: list[V20Example],
    *,
    position_encoding: str,
    mode: str,
) -> pd.DataFrame:
    """Evaluate all six presentations of each unordered three-character set.

    Teacher-forced (``tf_*``) answer logits and genuinely autoregressive
    (``ar_*``) generations are reported separately.  The unprefixed legacy
    columns deliberately mirror the AR result so downstream consumers cannot
    mistake gold-trace conditioning for free-generation robustness.
    """

    rows: list[dict[str, Any]] = []
    all_variants: list[V20Example] = []
    model.eval()
    for example in examples:
        assert example.needle_characters is not None and example.count is not None
        all_variants.extend(
            replace(example, rendered_set_order=tuple(order))
            for order in permutations(example.needle_characters)
        )
    rendered_all = [render_v20(item, vocab, mode) for item in all_variants]
    tf_predictions_flat: list[int] = []
    batch_size = min(cfg.analysis_batch_size, max(1, len(rendered_all)))
    for start in range(0, len(rendered_all), batch_size):
        rendered = rendered_all[start : start + batch_size]
        ids, _, mask = collate_v20(rendered, vocab, cfg.device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for index, item in enumerate(rendered):
            prediction, _ = _teacher_forced_number(logits[index], item, vocab)
            tf_predictions_flat.append(-1 if prediction is None else int(prediction))

    ar_detail = autoregressive_task_evaluation(
        model,
        cfg,
        vocab,
        all_variants,
        position_encoding=position_encoding,
        mode=mode,
        step=cfg.train_steps,
    )
    expected_rows = 6 * len(examples)
    if len(ar_detail) != expected_rows:
        raise RuntimeError(
            f"permutation AR evaluation returned {len(ar_detail)} rows; expected {expected_rows}"
        )
    for example_id, example in enumerate(examples):
        tf_predictions = tf_predictions_flat[6 * example_id : 6 * (example_id + 1)]
        ar_group = ar_detail.iloc[6 * example_id : 6 * (example_id + 1)]
        ar_predictions = [
            None if pd.isna(value) else int(value) for value in ar_group["ar_pred_count"].tolist()
        ]
        ar_accuracies = ar_group["ar_accuracy"].to_numpy(dtype=float)
        rows.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "example_id": example_id,
                "set_id": example.set_id,
                "count": example.count,
                "tf_evaluation_context": "teacher_forced_gold_trace" if mode == "thinking" else "teacher_forced_prompt",
                "tf_prediction_agreement": float(len(set(tf_predictions)) == 1),
                "tf_all_permutations_correct": float(
                    all(value == example.count for value in tf_predictions)
                ),
                "tf_permutation_accuracy": float(
                    np.mean(np.asarray(tf_predictions) == example.count)
                ),
                "tf_predictions_json": json.dumps(tf_predictions),
                "ar_evaluation_context": "free_generation_from_prompt",
                "ar_prediction_agreement": float(len(set(ar_predictions)) == 1),
                "ar_all_permutations_correct": float(np.all(ar_accuracies == 1.0)),
                "ar_permutation_accuracy": float(np.mean(ar_accuracies)),
                "ar_answer_rate": float(ar_group["ar_answered"].mean()),
                "ar_trace_exact_mean": float(ar_group["trace_exact"].mean()),
                "ar_ordered_marker_accuracy_mean": float(
                    ar_group["trace_ordered_marker_accuracy"].mean()
                ),
                "ar_predictions_json": json.dumps(ar_predictions),
                "prediction_agreement": float(len(set(ar_predictions)) == 1),
                "all_permutations_correct": float(np.all(ar_accuracies == 1.0)),
                "permutation_accuracy": float(np.mean(ar_accuracies)),
                "predictions_json": json.dumps(ar_predictions),
            }
        )
    return pd.DataFrame(rows)


def _permutation_examples(
    cfg: V20Config, examples: list[V20Example]
) -> list[V20Example]:
    """Small count-balanced AR suite; six permutations are generated per item."""

    selected: list[V20Example] = []
    for count in range(1, cfg.count_max_threshold + 1):
        values = [example for example in examples if int(example.count or 0) == count]
        chosen = values[: cfg.permutation_examples_per_count]
        if len(chosen) != cfg.permutation_examples_per_count:
            raise ValueError(f"prefix-permutation suite is incomplete for count={count}")
        selected.extend(chosen)
    return selected


def _empty_sampling_state(cfg: V20Config) -> dict[str, Any]:
    return {
        "examples": 0,
        "raw": 0,
        "task": 0,
        "active_tokens": 0,
        "task_active_tokens": 0,
        "filter_attempts": 0,
        "rejected_zero": 0,
        "rejected_over": 0,
        "accepted_counts": {str(value): 0 for value in range(1, cfg.count_max_threshold + 1)},
        "proposed_counts": {},
        "frequency_bins": {},
        "set_ids": {},
        "set_count_cells": {},
    }


def _load_sampling_state(run_dir: Path, position_encoding: str, mode: str, step: int, cfg: V20Config) -> dict[str, Any]:
    if step <= 0:
        return _empty_sampling_state(cfg)
    table = _read_table(run_dir / "tables" / "train_metrics.csv")
    if table.empty:
        return _empty_sampling_state(cfg)
    row = table[
        (table.position_encoding == position_encoding)
        & (table["mode"] == mode)
        & (table.step == step)
    ]
    if row.empty or "cumulative_sampling_json" not in row:
        return _empty_sampling_state(cfg)
    return json.loads(row.iloc[-1]["cumulative_sampling_json"])


def _update_sampling_state(
    state: dict[str, Any], examples: list[V20Example], active_by_example: np.ndarray
) -> None:
    state.setdefault("set_count_cells", {})
    for example, active_tokens in zip(examples, active_by_example):
        state["examples"] += 1
        state["active_tokens"] += int(active_tokens)
        if example.example_kind == "raw_lm":
            state["raw"] += 1
            continue
        state["task"] += 1
        state["task_active_tokens"] += int(active_tokens)
        state["filter_attempts"] += int(example.filter_attempts)
        state["rejected_zero"] += int(example.rejected_zero)
        state["rejected_over"] += int(example.rejected_over_threshold)
        for name, value in (
            ("accepted_counts", example.count),
            ("proposed_counts", example.proposed_count),
            ("frequency_bins", example.set_frequency_bin),
            ("set_ids", example.set_id),
        ):
            key = str(value)
            state[name][key] = int(state[name].get(key, 0)) + 1
        cell_key = f"{example.set_id}|{int(example.count or 0)}"
        state["set_count_cells"][cell_key] = (
            int(state["set_count_cells"].get(cell_key, 0)) + 1
        )


def _write_evaluations(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    curve_suites: dict[str, dict[str, list[V20Example]]],
    run_dir: Path,
    position_encoding: str,
    mode: str,
    step: int,
    *,
    run_ar: bool,
) -> None:
    heldout_task = curve_suites["heldout"]["task"]
    # Dense post-hoc snapshots carry the phase-transition study.  During
    # optimization, evaluate only the paired train/held-out task suites; raw and
    # mixture losses are deferred to the final checkpoint.  This removes the
    # seven-suite evaluation wall-time that dominated v16.2.
    task_suites = (
        ("train", curve_suites["train"]["task"]),
        ("heldout", heldout_task),
    )
    tf_examples = sum(len(examples) for _, examples in task_suites)
    with timed_event(
        run_dir,
        scope="training",
        block="periodic_teacher_forced_evaluation",
        position_encoding=position_encoding,
        mode=mode,
        step=step,
        device=cfg.device,
        num_examples=tf_examples,
    ):
        loss_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for source, examples in task_suites:
            row, components = evaluate_loss_suite(
                model,
                cfg,
                vocab,
                examples,
                position_encoding=position_encoding,
                mode=mode,
                step=step,
                curve_source=source,
                suite="task",
            )
            loss_rows.append(row)
            component_rows.extend(components)
        _append_unique(
            run_dir / "tables" / "eval_loss_curves.csv",
            pd.DataFrame(loss_rows),
            ["step", "position_encoding", "mode", "curve_source", "suite", "task_occurrence_ratio"],
        )
        _append_unique(
            run_dir / "tables" / "eval_loss_components.csv",
            pd.DataFrame(component_rows),
            ["step", "position_encoding", "mode", "curve_source", "suite", "component"],
        )
        detail = teacher_forced_task_evaluation(
            model,
            cfg,
            vocab,
            heldout_task,
            position_encoding=position_encoding,
            mode=mode,
            step=step,
        )
        _append_unique(
            run_dir / "tables" / "eval_detail.csv",
            detail,
            ["step", "position_encoding", "mode", "example_id"],
        )
    if run_ar:
        per_count = cfg.ar_examples_per_count
        ar_examples = []
        for count in range(1, cfg.count_max_threshold + 1):
            ar_examples.extend([item for item in heldout_task if item.count == count][:per_count])
        with timed_event(
            run_dir,
            scope="training",
            block="periodic_autoregressive_evaluation",
            position_encoding=position_encoding,
            mode=mode,
            step=step,
            device=cfg.device,
            num_examples=len(ar_examples),
        ):
            ar = autoregressive_task_evaluation(
                model,
                cfg,
                vocab,
                ar_examples,
                position_encoding=position_encoding,
                mode=mode,
                step=step,
            )
            _append_unique(
                run_dir / "tables" / "autoregressive_detail.csv",
                ar,
                ["step", "position_encoding", "mode", "row_id"],
            )


def _write_final_test(
    model: TinyPositionCausalLM,
    cfg: V20Config,
    vocab: V20Vocab,
    test_suites: dict[str, list[V20Example]],
    run_dir: Path,
    position_encoding: str,
    mode: str,
) -> None:
    rows = []
    for suite in ("raw", "task", "mixture"):
        row, _ = evaluate_loss_suite(
            model,
            cfg,
            vocab,
            test_suites[suite],
            position_encoding=position_encoding,
            mode=mode,
            step=cfg.train_steps,
            curve_source="test",
            suite=suite,
        )
        rows.append(row)
    _append_unique(
        run_dir / "tables" / "test_loss_summary.csv",
        pd.DataFrame(rows),
        ["position_encoding", "mode", "suite", "task_occurrence_ratio"],
    )
    # Periodic AR curves deliberately stay small (``ar_examples_per_count``)
    # because they are evaluated throughout training.  The final checkpoint is
    # evaluated on the disjoint test task suite, whose size is controlled by
    # ``final_examples_per_count`` (50/count in the main preset).  Keeping this
    # in a separate table prevents a high-powered final estimate from being
    # confused with the coarse 2/count learning curve.
    final_ar = autoregressive_task_evaluation(
        model,
        cfg,
        vocab,
        test_suites["task"],
        position_encoding=position_encoding,
        mode=mode,
        step=cfg.train_steps,
    )
    _write_final_autoregressive_artifacts(
        final_ar,
        run_dir,
        examples_per_count=cfg.final_examples_per_count,
    )


def _write_final_autoregressive_artifacts(
    frame: pd.DataFrame,
    run_dir: Path,
    *,
    examples_per_count: int | None,
) -> None:
    """Persist compact sufficient statistics plus a bounded failure sample.

    Full generated traces dominate CSV size and previously made Drive sync and
    notebook inspection fragile.  The complete evaluation table therefore
    keeps numeric outcomes for every prompt, while verbatim generations are
    retained only for up to five failures per count and mode.
    """

    full = frame.copy()
    full["evaluation_split"] = "test_task"
    if examples_per_count is not None:
        full["examples_per_count"] = int(examples_per_count)
    elif "examples_per_count" not in full:
        full["examples_per_count"] = full.groupby(
            ["position_encoding", "mode", "count"]
        )["row_id"].transform("size")
    if "generated_tokens" in full:
        failures = (
            full[full["ar_accuracy"] < 1]
            .sort_values(["position_encoding", "mode", "count", "row_id"])
            .groupby(["position_encoding", "mode", "count"], as_index=False)
            .head(5)
        )
        _append_unique(
            run_dir / "tables" / "final_autoregressive_failures.csv",
            failures,
            ["position_encoding", "mode", "row_id"],
        )
        full = full.drop(columns="generated_tokens")
    detail_path = run_dir / "tables" / "final_autoregressive_detail.csv"
    _append_unique(
        detail_path,
        full,
        ["position_encoding", "mode", "row_id"],
    )
    # Compact an older table that may still carry generations from a previous
    # analysis revision.
    compact = _read_table(detail_path)
    if "generated_tokens" in compact:
        atomic_csv(compact.drop(columns="generated_tokens"), detail_path)


def train_v20_variant(
    cfg: V20Config,
    vocab: V20Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    curve_suites: dict[str, dict[str, list[V20Example]]],
    test_suites: dict[str, list[V20Example]],
    position_encoding: str,
    mode: str,
    run_dir: Path,
    *,
    sync_run_dir: Path | None,
    skip_completed: bool,
) -> None:
    root = _checkpoint_root(run_dir, position_encoding, mode)
    final_path = root / "final" / "checkpoint.pt"
    required_test = run_dir / "tables" / "test_loss_summary.csv"
    required_permutation = run_dir / "tables" / "prefix_permutation_consistency.csv"
    required_final_ar = run_dir / "tables" / "final_autoregressive_detail.csv"
    permutation_examples = _permutation_examples(cfg, curve_suites["heldout"]["task"])
    outputs_complete = False
    if (
        final_path.exists()
        and required_test.exists()
        and required_permutation.exists()
        and required_final_ar.exists()
    ):
        test_table = _read_table(required_test)
        permutation_table = _read_table(required_permutation)
        final_ar_table = _read_table(required_final_ar)
        test_rows = test_table[
            (test_table.position_encoding == position_encoding) & (test_table["mode"] == mode)
        ]
        permutation_rows = permutation_table[
            (permutation_table.position_encoding == position_encoding)
            & (permutation_table["mode"] == mode)
        ]
        final_ar_rows = final_ar_table[
            (final_ar_table.position_encoding == position_encoding)
            & (final_ar_table["mode"] == mode)
        ]
        outputs_complete = (
            set(test_rows.get("suite", pd.Series(dtype=str)).astype(str)) >= {"raw", "task", "mixture"}
            and len(permutation_rows) == len(permutation_examples)
            and {"ar_permutation_accuracy", "tf_permutation_accuracy"}.issubset(permutation_rows.columns)
            and len(final_ar_rows)
            == int(cfg.final_examples_per_count * cfg.count_max_threshold)
            and final_ar_rows.groupby("count").size().eq(cfg.final_examples_per_count).all()
        )
    if skip_completed and final_path.exists() and outputs_complete:
        print(f"[skip] {position_encoding}/{mode}: final checkpoint exists", flush=True)
        return
    if not skip_completed and root.exists() and (
        (root / "recovery" / "latest.pt").exists() or (root / "snapshot_index.csv").exists()
    ):
        raise RuntimeError(
            f"refusing to overwrite existing checkpoints in {root}; "
            "use --skip-completed to resume or choose a new run name"
        )
    if cfg.training_count_distribution == "maxent_set_count":
        sampler = _joint_set_count_sampler(cfg, text, split, pool)
        plan_path = run_dir / "tables" / "training_set_count_sampler_plan.csv"
        atomic_csv(sampler.plan, plan_path)
        feasible = int(sampler.plan["feasible"].sum())
        print(
            "[sampler] maxent_set_count "
            f"feasible_cells={feasible}/{len(sampler.plan)} "
            f"max_set_error="
            f"{float((sampler.plan.groupby('set_id').target_probability.sum() - 1.0 / len(pool.sets)).abs().max()):.3e} "
            f"max_count_error="
            f"{float((sampler.plan.groupby('count').target_probability.sum() - 1.0 / cfg.count_max_threshold).abs().max()):.3e}",
            flush=True,
        )
    model = paired_v20_model(cfg, vocab, position_encoding)
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay
    )
    rng = random.Random(cfg.seed)
    snapshots = DenseSnapshotWriter(
        root, cfg, position_encoding, mode, sync_run_dir, run_dir
    )
    start_step = 0
    latest = _latest_checkpoint(root) if skip_completed else None
    restored_sampling_state: dict[str, Any] | None = None
    if latest is not None:
        start_step, path = latest
        payload = torch.load(path, map_location=cfg.device, weights_only=False)
        restored_sampling_state = _restore_checkpoint(
            payload,
            model,
            optimizer,
            rng,
            pool,
            split,
            vocab,
            cfg,
            position_encoding,
            mode,
            start_step,
        )
        print(f"[resume] {position_encoding}/{mode} from step {start_step}", flush=True)
    else:
        _save_checkpoint(
            model,
            optimizer,
            cfg,
            vocab,
            pool,
            split,
            position_encoding,
            mode,
            0,
            rng,
            run_dir,
            sync_run_dir,
        )
        snapshots.add(model, 0, force=True)
    sampling_state = restored_sampling_state or _load_sampling_state(
        run_dir, position_encoding, mode, start_step, cfg
    )

    if start_step == 0:
        _write_evaluations(
            model, cfg, vocab, curve_suites, run_dir, position_encoding, mode, 0, run_ar=False
        )
    progress = tqdm(
        range(start_step + 1, cfg.train_steps + 1),
        desc=f"{cfg.version} {position_encoding}/{mode}",
        initial=start_step,
        total=cfg.train_steps,
    )
    optimizer_interval_seconds = 0.0
    optimizer_interval_steps = 0
    numeric_checkpoint_steps = set(planned_checkpoint_steps(cfg))
    for step in progress:
        optimizer_started = time.perf_counter()
        model.train()
        loss_phase = training_loss_phase(cfg, step)
        if step == cfg.max_steps_for_language_pred + 1:
            print(
                f"[train] {position_encoding}/{mode}: switching to task-output-only loss at step {step}",
                flush=True,
            )
        examples, rendered = _training_batch(
            cfg,
            vocab,
            text,
            split,
            pool,
            mode,
            rng,
            require_task=loss_phase == "task_output",
        )
        ids, labels, attention_mask = collate_v20(rendered, vocab, cfg.device)
        loss_weights = collate_v20_loss_weights(rendered, cfg, cfg.device, step=step)
        rollin_probability = scheduled_sampling_probability(cfg, step, mode)
        model_input_ids = ids
        rollin_stats: dict[str, float | int] = {
            "eligible_tokens": 0,
            "selected_tokens": 0,
            "changed_tokens": 0,
            "changed_fraction": 0.0,
        }
        if rollin_probability > 0.0:
            with _autocast_context(cfg):
                model_input_ids, rollin_stats = scheduled_sampling_inputs(
                    model,
                    ids,
                    attention_mask,
                    rendered,
                    rollin_probability,
                )
        contrastive_active = (
            loss_phase == "task_output"
            and cfg.answer_query_contrastive_weight > 0
        )
        with _autocast_context(cfg):
            output = model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=contrastive_active,
            )
            token_weighted_loss, token_losses, active = shifted_v20_token_losses(
                output.logits, labels, loss_weights
            )
            component_reduction_active = (
                loss_phase == "task_output"
                and cfg.task_output_loss_reduction == "component_normalized"
            )
            if component_reduction_active:
                loss, objective_region_losses = component_normalized_task_output_loss(
                    token_losses,
                    active,
                    loss_weights,
                    rendered,
                    cfg,
                )
            else:
                loss = token_weighted_loss
                objective_region_losses = {}
            if contrastive_active:
                if output.hidden_states is None:
                    raise RuntimeError(
                        "answer-query contrastive loss requires hidden states"
                    )
                contrastive_loss = answer_query_supervised_contrastive_loss(
                    model.final_norm(output.hidden_states[-1]),
                    rendered,
                    temperature=cfg.answer_query_contrastive_temperature,
                )
                loss = (
                    loss
                    + float(cfg.answer_query_contrastive_weight)
                    * contrastive_loss
                )
            else:
                contrastive_loss = None
        active_by_example = active.sum(dim=1).detach().cpu().numpy()
        _update_sampling_state(sampling_state, examples, active_by_example)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        rate = learning_rate(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        optimizer_interval_seconds += time.perf_counter() - optimizer_started
        optimizer_interval_steps += 1

        if step % cfg.log_every == 0 or step in {1, cfg.train_steps}:
            is_task = np.asarray([example.example_kind == "counting_task" for example in examples])
            task_tokens = int(active_by_example[is_task].sum())
            component_values = _component_example_means(token_losses, rendered)
            active_weights = loss_weights[:, 1:] * active
            active_weight_sum = float(active_weights.sum().detach().cpu())
            objective_active_tokens = int((active_weights > 0).sum().detach().cpu())
            unweighted_loss = float(
                ((token_losses * active).sum() / active.sum().clamp_min(1)).detach().cpu()
            )
            final_count_weight_sum = float(
                sum(
                    loss_weights[row_index, position].detach().cpu()
                    for row_index, item in enumerate(rendered)
                    if item.spans is not None
                    for position in item.spans.count_positions
                )
            )
            trace_weight_sum = float(
                sum(
                    loss_weights[row_index, position].detach().cpu()
                    for row_index, item in enumerate(rendered)
                    if item.spans is not None
                    for position in (
                        *(position for group in item.spans.trace_index_token_groups for position in group),
                        *item.spans.trace_marker_positions,
                    )
                )
            )
            if component_reduction_active:
                coefficient_denominator = (
                    float(cfg.task_output_count_weight)
                    + float(cfg.task_output_structure_weight)
                    + (
                        float(cfg.task_output_trace_weight)
                        if "trace" in objective_region_losses
                        else 0.0
                    )
                )
                final_count_coefficient_share = (
                    float(cfg.task_output_count_weight) / coefficient_denominator
                )
                trace_coefficient_share = (
                    float(cfg.task_output_trace_weight) / coefficient_denominator
                    if "trace" in objective_region_losses
                    else 0.0
                )
                structure_coefficient_share = (
                    float(cfg.task_output_structure_weight) / coefficient_denominator
                )
            else:
                final_count_coefficient_share = float("nan")
                trace_coefficient_share = float("nan")
                structure_coefficient_share = float("nan")
            row: dict[str, Any] = {
                "step": step,
                "position_encoding": position_encoding,
                "mode": mode,
                "train_total_loss": float(loss.detach().cpu()),
                "train_weighted_objective_loss": float(loss.detach().cpu()),
                "train_unweighted_token_loss": unweighted_loss,
                "batch_active_weight_sum": active_weight_sum,
                "batch_final_count_weight_share": final_count_weight_sum / max(1.0, active_weight_sum),
                "batch_cot_trace_weight_share": trace_weight_sum / max(1.0, active_weight_sum),
                "batch_final_count_token_weight_share": final_count_weight_sum
                / max(1.0, active_weight_sum),
                "batch_cot_trace_token_weight_share": trace_weight_sum
                / max(1.0, active_weight_sum),
                "task_output_loss_reduction": cfg.task_output_loss_reduction,
                "component_reduction_active": component_reduction_active,
                "answer_query_contrastive_active": contrastive_active,
                "train_answer_query_contrastive_loss": (
                    float(contrastive_loss.detach().cpu())
                    if contrastive_loss is not None
                    else float("nan")
                ),
                "answer_query_contrastive_weight": float(
                    cfg.answer_query_contrastive_weight
                ),
                "answer_query_contrastive_temperature": float(
                    cfg.answer_query_contrastive_temperature
                ),
                "scheduled_sampling_probability": rollin_probability,
                "scheduled_sampling_eligible_tokens": int(
                    rollin_stats["eligible_tokens"]
                ),
                "scheduled_sampling_selected_tokens": int(
                    rollin_stats["selected_tokens"]
                ),
                "scheduled_sampling_changed_tokens": int(
                    rollin_stats["changed_tokens"]
                ),
                "scheduled_sampling_changed_fraction": float(
                    rollin_stats["changed_fraction"]
                ),
                "batch_final_count_region_coefficient_share": final_count_coefficient_share,
                "batch_trace_region_coefficient_share": trace_coefficient_share,
                "batch_structure_region_coefficient_share": structure_coefficient_share,
                "train_objective_final_count_region_mean_loss": float(
                    objective_region_losses["final_count"].detach().cpu()
                )
                if "final_count" in objective_region_losses
                else float("nan"),
                "train_objective_trace_region_mean_loss": float(
                    objective_region_losses["trace"].detach().cpu()
                )
                if "trace" in objective_region_losses
                else float("nan"),
                "train_objective_structure_region_mean_loss": float(
                    objective_region_losses["structure"].detach().cpu()
                )
                if "structure" in objective_region_losses
                else float("nan"),
                "training_loss_phase": loss_phase,
                "language_prediction_enabled": loss_phase == "all_sequence",
                "batch_objective_active_tokens": objective_active_tokens,
                "batch_task_output_examples": int(is_task.sum()),
                "learning_rate": rate,
                "gradient_norm": gradient_norm,
                "configured_task_example_ratio": cfg.task_occurrence_ratio,
                "batch_realized_task_example_ratio": float(is_task.mean()),
                "batch_realized_task_token_ratio": task_tokens / max(1, int(active_by_example.sum())),
                "batch_active_tokens": int(active_by_example.sum()),
                "batch_filter_attempts_mean": float(
                    np.mean([example.filter_attempts for example in examples if example.example_kind == "counting_task"])
                ) if is_task.any() else 0.0,
                "cumulative_realized_task_example_ratio": sampling_state["task"]
                / max(1, sampling_state["examples"]),
                "cumulative_realized_task_token_ratio": sampling_state["task_active_tokens"]
                / max(1, sampling_state["active_tokens"]),
                "cumulative_sampling_json": json.dumps(sampling_state, sort_keys=True),
            }
            row.update({f"train_{name}_example_mean_loss": float(np.mean(values)) for name, values in component_values.items()})
            _append_unique(
                run_dir / "tables" / "train_metrics.csv",
                pd.DataFrame([row]),
                ["position_encoding", "mode", "step"],
            )
            progress.set_postfix(loss=f"{loss.item():.4f}", task=f"{is_task.mean():.2f}")
            rollin_suffix = (
                f" rollin_p={rollin_probability:.3f} "
                f"rollin_changed={int(rollin_stats['changed_tokens'])}/"
                f"{int(rollin_stats['eligible_tokens'])}"
                if rollin_probability > 0.0
                else ""
            )
            print(
                f"[train] {position_encoding}/{mode} "
                f"step={step}/{cfg.train_steps} loss={loss.item():.4f} "
                f"lr={rate:.3e} grad_norm={gradient_norm:.3f} "
                f"task_ratio={is_task.mean():.2f}{rollin_suffix}",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
                sync_started = time.perf_counter()
                torch.cuda.synchronize(cfg.device)
                optimizer_interval_seconds += time.perf_counter() - sync_started
            record_duration_event(
                run_dir,
                scope="training",
                block="optimizer_interval",
                duration_seconds=optimizer_interval_seconds,
                position_encoding=position_encoding,
                mode=mode,
                step=step,
                device=cfg.device,
                num_examples=optimizer_interval_steps * cfg.batch_size,
                num_batches=optimizer_interval_steps,
            )
            optimizer_interval_seconds = 0.0
            optimizer_interval_steps = 0
            _write_evaluations(
                model,
                cfg,
                vocab,
                curve_suites,
                run_dir,
                position_encoding,
                mode,
                step,
                run_ar=(step % cfg.ar_eval_every == 0 or step == cfg.train_steps),
            )
        if step in numeric_checkpoint_steps:
            snapshots.add(model, step)
        if (
            step % cfg.recovery_every == 0
            or step == cfg.max_steps_for_language_pred
            or step == cfg.train_steps
        ):
            _save_checkpoint(
                model,
                optimizer,
                cfg,
                vocab,
                pool,
                split,
                position_encoding,
                mode,
                step,
                rng,
                run_dir,
                sync_run_dir,
                sampling_state=sampling_state,
            )
            if step == cfg.max_steps_for_language_pred:
                _save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    vocab,
                    pool,
                    split,
                    position_encoding,
                    mode,
                    step,
                    rng,
                    run_dir,
                    sync_run_dir,
                    label="pinned",
                    sampling_state=sampling_state,
                )
    snapshots.flush()
    _save_checkpoint(
        model, optimizer, cfg, vocab, pool, split, position_encoding, mode, cfg.train_steps,
        rng, run_dir, sync_run_dir, label="final", sampling_state=sampling_state,
    )
    with timed_event(
        run_dir,
        scope="training",
        block="final_test",
        position_encoding=position_encoding,
        mode=mode,
        step=cfg.train_steps,
        device=cfg.device,
        num_examples=sum(len(values) for values in test_suites.values()),
    ):
        _write_final_test(model, cfg, vocab, test_suites, run_dir, position_encoding, mode)
    with timed_event(
        run_dir,
        scope="training",
        block="prefix_permutation_evaluation",
        position_encoding=position_encoding,
        mode=mode,
        step=cfg.train_steps,
        device=cfg.device,
    ):
        permutation = prefix_permutation_consistency_evaluation(
            model,
            cfg,
            vocab,
            permutation_examples,
            position_encoding=position_encoding,
            mode=mode,
        )
        _append_unique(
            run_dir / "tables" / "prefix_permutation_consistency.csv",
            permutation,
            ["position_encoding", "mode", "example_id"],
        )


def summarize_learning_tables(run_dir: Path) -> None:
    detail = _read_table(run_dir / "tables" / "eval_detail.csv")
    if not detail.empty:
        tf_aggregations: dict[str, tuple[str, str]] = {
            "tf_final_accuracy": ("tf_final_accuracy", "mean"),
            "tf_trace_marker_accuracy": ("tf_trace_marker_accuracy", "mean"),
            "tf_trace_index_accuracy": ("tf_trace_index_accuracy", "mean"),
            "frequency_baseline_accuracy": ("frequency_baseline_accuracy", "mean"),
        }
        for name in ("tf_trace_query_accuracy", "tf_trace_delimiter_accuracy"):
            if name in detail:
                tf_aggregations[name] = (name, "mean")
        by_count = detail.groupby(
            ["position_encoding", "mode", "step", "count", "count_bin"], as_index=False
        ).agg(**tf_aggregations)
        atomic_csv(by_count, run_dir / "tables" / "eval_by_count.csv")
        by_set = detail.groupby(
            ["position_encoding", "mode", "step", "set_id", "set_frequency_bin"], as_index=False
        ).agg(tf_final_accuracy=("tf_final_accuracy", "mean"), examples=("example_id", "count"))
        atomic_csv(by_set, run_dir / "tables" / "eval_by_set.csv")
    ar = _read_table(run_dir / "tables" / "autoregressive_detail.csv")
    if not ar.empty:
        if "ar_answered" not in ar:
            ar["ar_answered"] = ar["ar_pred_count"].notna().astype(float)
        if "ar_abs_error_with_missing_penalty" not in ar:
            ar["ar_abs_error_with_missing_penalty"] = ar["ar_abs_error"].fillna(
                int(ar["count"].max())
            )
        if "trace_ordered_marker_accuracy" not in ar:
            ar["trace_ordered_marker_accuracy"] = ar["trace_marker_recall"]
        ar_aggregations: dict[str, tuple[str, str]] = {
            "ar_final_accuracy": ("ar_accuracy", "mean"),
            "ar_answer_rate": ("ar_answered", "mean"),
            "ar_abs_error_answered_only": ("ar_abs_error", "mean"),
            "ar_abs_error": ("ar_abs_error", "mean"),
            "ar_abs_error_with_missing_penalty": ("ar_abs_error_with_missing_penalty", "mean"),
            "trace_exact": ("trace_exact", "mean"),
            "trace_ordered_marker_accuracy": ("trace_ordered_marker_accuracy", "mean"),
            "trace_marker_recall": ("trace_marker_recall", "mean"),
        }
        for name in (
            "trace_marker_count_accuracy",
            "trace_format_valid",
            "trace_closed",
            "trace_delimiter_count_accuracy",
        ):
            if name in ar:
                ar_aggregations[name] = (name, "mean")
        summary = ar.groupby(
            ["position_encoding", "mode", "step", "count"], as_index=False
        ).agg(**ar_aggregations)
        atomic_csv(summary, run_dir / "tables" / "autoregressive_by_count.csv")
    final_ar = _read_table(run_dir / "tables" / "final_autoregressive_detail.csv")
    if not final_ar.empty:
        if "ar_answered" not in final_ar:
            final_ar["ar_answered"] = final_ar["ar_pred_count"].notna().astype(float)
        if "trace_ordered_marker_accuracy" not in final_ar:
            final_ar["trace_ordered_marker_accuracy"] = final_ar["trace_marker_recall"]
        final_aggregations: dict[str, tuple[str, str]] = {
            "examples": ("ar_accuracy", "size"),
            "ar_final_accuracy": ("ar_accuracy", "mean"),
            "ar_answer_rate": ("ar_answered", "mean"),
            "ar_abs_error_answered_only": ("ar_abs_error", "mean"),
            "ar_abs_error_with_missing_penalty": (
                "ar_abs_error_with_missing_penalty",
                "mean",
            ),
            "trace_exact": ("trace_exact", "mean"),
            "trace_ordered_marker_accuracy": ("trace_ordered_marker_accuracy", "mean"),
        }
        for name in (
            "trace_marker_count_accuracy",
            "trace_format_valid",
            "trace_closed",
            "trace_delimiter_count_accuracy",
        ):
            if name in final_ar:
                final_aggregations[name] = (name, "mean")
        final_by_count = final_ar.groupby(
            ["position_encoding", "mode", "step", "count"], as_index=False
        ).agg(**final_aggregations)
        atomic_csv(final_by_count, run_dir / "tables" / "final_autoregressive_by_count.csv")

        rows: list[dict[str, Any]] = []
        z = 1.959963984540054
        for (position_encoding, mode, step), group in final_ar.groupby(
            ["position_encoding", "mode", "step"]
        ):
            successes = float(group["ar_accuracy"].sum())
            observations = int(len(group))
            estimate = successes / max(1, observations)
            denominator = 1.0 + z * z / observations
            center = (estimate + z * z / (2 * observations)) / denominator
            radius = z * math.sqrt(
                estimate * (1 - estimate) / observations
                + z * z / (4 * observations * observations)
            ) / denominator
            rows.append(
                {
                    "position_encoding": position_encoding,
                    "mode": mode,
                    "step": int(step),
                    "examples": observations,
                    "examples_per_count": int(group.groupby("count").size().min()),
                    "ar_final_accuracy": estimate,
                    "ar_final_accuracy_wilson95_low": max(0.0, center - radius),
                    "ar_final_accuracy_wilson95_high": min(1.0, center + radius),
                    "ar_answer_rate": float(group["ar_answered"].mean()),
                    "trace_exact": float(group["trace_exact"].mean()),
                    "trace_ordered_marker_accuracy": float(
                        group["trace_ordered_marker_accuracy"].mean()
                    ),
                    **{
                        name: float(group[name].mean())
                        for name in (
                            "trace_marker_count_accuracy",
                            "trace_format_valid",
                            "trace_closed",
                            "trace_delimiter_count_accuracy",
                        )
                        if name in group
                    },
                }
            )
        atomic_csv(
            pd.DataFrame(rows), run_dir / "tables" / "final_autoregressive_summary.csv"
        )
    train = _read_table(run_dir / "tables" / "train_metrics.csv")
    if not train.empty and "cumulative_sampling_json" in train:
        final = train.sort_values("step").groupby(["position_encoding", "mode"], as_index=False).tail(1)
        rows: list[dict[str, Any]] = []
        for _, item in final.iterrows():
            state = json.loads(item["cumulative_sampling_json"])
            for dimension in (
                "accepted_counts",
                "proposed_counts",
                "frequency_bins",
                "set_ids",
                "set_count_cells",
            ):
                for value, count in state.get(dimension, {}).items():
                    rows.append(
                        {
                            "position_encoding": item.position_encoding,
                            "mode": item["mode"],
                            "dimension": dimension,
                            "value": value,
                            "examples": count,
                            "total_training_examples": state["examples"],
                            "raw_examples": state["raw"],
                            "task_examples": state["task"],
                            "rejected_zero": state["rejected_zero"],
                            "rejected_over_threshold": state["rejected_over"],
                            "filter_attempts": state["filter_attempts"],
                        }
                    )
        atomic_csv(pd.DataFrame(rows), run_dir / "tables" / "training_sampling_distribution.csv")


def train_v20_models(
    cfg: V20Config,
    vocab: V20Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    curve_suites: dict[str, dict[str, list[V20Example]]],
    test_suites: dict[str, list[V20Example]],
    run_dir: Path,
    *,
    sync_run_dir: Path | None,
    skip_completed: bool,
) -> None:
    specifications: list[dict[str, Any]] = []
    for position_encoding, mode in cfg.model_variants:
        probe = paired_v20_model(cfg, vocab, position_encoding)
        specifications.append(
            {
                "position_encoding": position_encoding,
                "mode": mode,
                "parameters": probe.parameter_count(),
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_embd": cfg.n_embd,
                "n_inner": cfg.n_inner,
            }
        )
        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[train] {position_encoding}/{mode}", flush=True)
        train_v20_variant(
            cfg,
            vocab,
            text,
            split,
            pool,
            curve_suites,
            test_suites,
            position_encoding,
            mode,
            run_dir,
            sync_run_dir=sync_run_dir,
            skip_completed=skip_completed,
        )
    atomic_csv(pd.DataFrame(specifications), run_dir / "tables" / "model_specifications.csv")
    summarize_learning_tables(run_dir)
    if sync_run_dir is not None:
        sync_tree(run_dir, sync_run_dir)


def load_v20_checkpoint_model(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    *,
    step: int | None = None,
    label: str | None = None,
    device: str | None = None,
) -> tuple[V20Config, V20Vocab, NeedlePool, CorpusSplit, TinyPositionCausalLM]:
    """Load one v20/v21 model checkpoint after validating artifact identities."""

    if (step is None) == (label is None):
        raise ValueError("provide exactly one of step or label")
    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if device is not None:
        cfg = replace(cfg, device=device)
    text = load_corpus_text()
    from .data import load_corpus_split

    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, text)
    vocab = V20Vocab.load(run_dir / "vocab.json")
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    model = build_model(cfg, vocab, position_encoding, cfg.device)
    checkpoint_root = _checkpoint_root(run_dir, position_encoding, mode)
    selected = label if label is not None else f"step_{int(step):06d}"
    path = checkpoint_root / selected / "checkpoint.pt"
    if label == "final" and not path.exists():
        numeric_final = (
            _checkpoint_root(run_dir, position_encoding, mode)
            / f"step_{cfg.train_steps:06d}"
            / "checkpoint.pt"
        )
        if numeric_final.exists():
            path = numeric_final
    if step is not None and not path.exists():
        state = load_dense_snapshot_state(run_dir, position_encoding, mode, int(step))
        model.load_state_dict(state)
        return cfg, vocab, pool, split, model.eval()
    if not path.exists():
        raise FileNotFoundError(f"missing {cfg.version} checkpoint: {path}")
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    payload_cfg = config_from_dict(dict(payload.get("config", {})))
    if payload_cfg != config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8"))):
        raise ValueError("checkpoint config does not match run config")
    if payload.get("position_encoding") != position_encoding or payload.get("mode") != mode:
        raise ValueError("checkpoint position encoding/mode does not match requested variant")
    expected_step = cfg.train_steps if label == "final" else int(step)
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError("checkpoint payload step does not match requested step")
    if payload.get("pool_fingerprint") != pool.pool_fingerprint:
        raise ValueError("checkpoint pool fingerprint mismatch")
    if payload.get("split_fingerprint") != split.split_fingerprint:
        raise ValueError("checkpoint split fingerprint mismatch")
    if payload.get("vocab_fingerprint") != vocab.fingerprint:
        raise ValueError("checkpoint vocabulary fingerprint mismatch")
    model.load_state_dict(payload["model_state_dict"])
    return cfg, vocab, pool, split, model.eval()


def load_final_v20_model(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    device: str | None = None,
) -> tuple[V20Config, V20Vocab, NeedlePool, TinyPositionCausalLM]:
    cfg, vocab, pool, _, model = load_v20_checkpoint_model(
        run_dir,
        position_encoding,
        mode,
        label="final",
        device=device,
    )
    return cfg, vocab, pool, model
