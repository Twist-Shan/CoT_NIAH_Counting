from __future__ import annotations

import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import V16_3Config, config_from_dict
from .data import (
    CorpusSplit,
    V16_3Example,
    V16_3Rendered,
    V16_3Vocab,
    collate_v16_3,
    collate_v16_3_loss_weights,
    component_target_positions,
    load_corpus_text,
    make_training_example,
    render_v16_3,
    shifted_v16_3_token_losses,
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
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, temporary)
            temporary.replace(target)


def learning_rate(cfg: V16_3Config, step: int) -> float:
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def training_loss_phase(cfg: V16_3Config, step: int) -> str:
    """Return the objective phase for an absolute optimizer step."""

    return "all_sequence" if step <= cfg.max_steps_for_language_pred else "task_output"


def _autocast_context(cfg: V16_3Config):
    if (
        cfg.precision == "bf16"
        and str(cfg.device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def paired_v16_3_model(
    cfg: V16_3Config, vocab: V16_3Vocab, position_encoding: str
) -> TinyPositionCausalLM:
    """Give RoPE/RPE and both output modes matching shared initial weights."""

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
            pass
    return max(candidates, key=lambda item: item[0], default=None)


def checkpoint_steps(
    run_dir: str | Path, position_encoding: str, mode: str
) -> list[tuple[int, Path]]:
    """Return validated numeric checkpoint paths in optimizer-step order."""

    root = _checkpoint_root(Path(run_dir), position_encoding, mode)
    result: list[tuple[int, Path]] = []
    for path in root.glob("step_*/checkpoint.pt"):
        try:
            step = int(path.parent.name.removeprefix("step_"))
        except ValueError:
            continue
        result.append((step, path))
    return sorted(result)


def planned_checkpoint_steps(cfg: V16_3Config) -> list[int]:
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


def _save_checkpoint(
    model: TinyPositionCausalLM,
    optimizer: AdamW,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
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
    path = root / (label or f"step_{step:06d}") / "checkpoint.pt"
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
                    target = sync_run_dir / source.relative_to(run_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    return path


def _restore_checkpoint(
    payload: dict[str, Any],
    model: TinyPositionCausalLM,
    optimizer: AdamW,
    rng: random.Random,
    pool: NeedlePool,
    split: CorpusSplit,
    vocab: V16_3Vocab,
    cfg: V16_3Config,
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
    rendered: list[V16_3Rendered],
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
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
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
        rendered = [render_v16_3(example, vocab, mode) for example in chunk]
        ids, labels, attention_mask = collate_v16_3(rendered, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=attention_mask)
        _, losses, active = shifted_v16_3_token_losses(output.logits, labels)
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
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    curve_suites: dict[str, dict[str, list[V16_3Example]]],
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


def _count_prediction(logits: torch.Tensor, vocab: V16_3Vocab) -> int:
    number_ids = torch.tensor(vocab.number_ids, device=logits.device)
    return int(logits[number_ids].argmax()) + 1


@torch.no_grad()
def teacher_forced_task_evaluation(
    model: TinyPositionCausalLM,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
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
        rendered = [render_v16_3(example, vocab, mode) for example in chunk]
        ids, _, mask = collate_v16_3(rendered, vocab, cfg.device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        for row_index, (example, item) in enumerate(zip(chunk, rendered)):
            assert item.spans is not None and example.count is not None
            predicted = _count_prediction(logits[row_index, item.spans.ans_pos], vocab)
            marker_correct: list[float] = []
            index_correct: list[float] = []
            marker_correct_by_token: dict[str, list[float]] = {}
            if mode == "thinking":
                for marker_position in item.spans.trace_marker_positions:
                    correct = float(
                        int(logits[row_index, marker_position - 1].argmax())
                        == item.input_ids[marker_position]
                    )
                    marker_correct.append(correct)
                    marker_correct_by_token.setdefault(item.tokens[marker_position], []).append(correct)
                for index_position in item.spans.trace_index_positions:
                    index_correct.append(
                        float(int(logits[row_index, index_position - 1].argmax()) == item.input_ids[index_position])
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
                "tf_final_accuracy": float(predicted == example.count),
                "tf_trace_marker_accuracy": float(np.mean(marker_correct)) if marker_correct else np.nan,
                "tf_trace_index_accuracy": float(np.mean(index_correct)) if index_correct else np.nan,
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


def _parse_generation(tokens: list[str], vocab: V16_3Vocab, example: V16_3Example, mode: str) -> dict[str, Any]:
    predicted = None
    if "<Ans>" in tokens:
        index = tokens.index("<Ans>")
        if index + 1 < len(tokens) and tokens[index + 1] in vocab.numbers:
            predicted = vocab.numbers.index(tokens[index + 1]) + 1
    trace: list[str] = []
    if mode == "thinking" and "<Think>" in tokens:
        start = tokens.index("<Think>") + 1
        end = tokens.index("</Think>") if "</Think>" in tokens[start:] else len(tokens)
        trace = tokens[start:end]
    expected = [
        token for index, marker in enumerate(example.needle_markers, start=1)
        for token in (vocab.number_token(index), marker)
    ]
    generated_markers = trace[1::2]
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
            abs(predicted - int(example.count)) if predicted is not None else len(vocab.numbers)
        ),
        "trace_exact": float(trace == expected) if mode == "thinking" else np.nan,
        "trace_ordered_marker_accuracy": (
            matches / max(1, int(example.count)) if mode == "thinking" else np.nan
        ),
        # Backward-compatible alias. This is positional accuracy, not set-level recall.
        "trace_marker_recall": matches / max(1, int(example.count)) if mode == "thinking" else np.nan,
        "generated_tokens": " ".join(tokens),
    }


@torch.no_grad()
def autoregressive_task_evaluation(
    model: TinyPositionCausalLM,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
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
            item = render_v16_3(example, vocab, mode)
            assert item.spans is not None
            stop = item.spans.ans_pos + 1 if mode == "nonthinking" else item.spans.think_pos + 1
            prefixes.append(item.tokens[:stop])
        generated = torch.tensor([vocab.encode(tokens) for tokens in prefixes], device=cfg.device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = 2 if mode == "nonthinking" else 2 * cfg.count_max_threshold + 5
        for _ in range(max_new_tokens):
            next_ids = model(input_ids=generated).logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(done, torch.full_like(next_ids, vocab.eos_id), next_ids)
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            for index, token_id in enumerate(next_ids.tolist()):
                if token_id == vocab.eos_id or _parse_generation(vocab.decode(generated[index]), vocab, chunk[index], mode)["ar_pred_count"] is not None:
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


def _training_batch(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    mode: str,
    rng: random.Random,
    *,
    require_task: bool = False,
) -> tuple[list[V16_3Example], list[V16_3Rendered]]:
    for _ in range(1_000):
        examples = [make_training_example(cfg, vocab, text, split, pool, rng) for _ in range(cfg.batch_size)]
        if not require_task or any(item.example_kind == "counting_task" for item in examples):
            return examples, [render_v16_3(example, vocab, mode) for example in examples]
    raise RuntimeError("could not sample a counting task for task-output-only training")


@torch.no_grad()
def prefix_permutation_consistency_evaluation(
    model: TinyPositionCausalLM,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    examples: list[V16_3Example],
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
    all_variants: list[V16_3Example] = []
    model.eval()
    for example in examples:
        assert example.needle_characters is not None and example.count is not None
        all_variants.extend(
            replace(example, rendered_set_order=tuple(order))
            for order in permutations(example.needle_characters)
        )
    rendered_all = [render_v16_3(item, vocab, mode) for item in all_variants]
    tf_predictions_flat: list[int] = []
    batch_size = min(cfg.analysis_batch_size, max(1, len(rendered_all)))
    for start in range(0, len(rendered_all), batch_size):
        rendered = rendered_all[start : start + batch_size]
        ids, _, mask = collate_v16_3(rendered, vocab, cfg.device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        tf_predictions_flat.extend(
            _count_prediction(logits[index, item.spans.ans_pos], vocab)
            for index, item in enumerate(rendered)
        )

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


def _empty_sampling_state(cfg: V16_3Config) -> dict[str, Any]:
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
    }


def _load_sampling_state(run_dir: Path, position_encoding: str, mode: str, step: int, cfg: V16_3Config) -> dict[str, Any]:
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
    state: dict[str, Any], examples: list[V16_3Example], active_by_example: np.ndarray
) -> None:
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


def _write_evaluations(
    model: TinyPositionCausalLM,
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    curve_suites: dict[str, dict[str, list[V16_3Example]]],
    run_dir: Path,
    position_encoding: str,
    mode: str,
    step: int,
    *,
    run_ar: bool,
) -> None:
    heldout_task = curve_suites["heldout"]["task"]
    tf_examples = sum(len(suites[suite]) for suites in curve_suites.values() for suite in ("raw", "task", "mixture")) + len(heldout_task)
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
        loss_rows, component_rows = evaluate_curve_suites(
            model, cfg, vocab, curve_suites, position_encoding=position_encoding, mode=mode, step=step
        )
        _append_unique(
            run_dir / "tables" / "eval_loss_curves.csv",
            loss_rows,
            ["step", "position_encoding", "mode", "curve_source", "suite", "task_occurrence_ratio"],
        )
        _append_unique(
            run_dir / "tables" / "eval_loss_components.csv",
            component_rows,
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
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    test_suites: dict[str, list[V16_3Example]],
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


def train_v16_3_variant(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    curve_suites: dict[str, dict[str, list[V16_3Example]]],
    test_suites: dict[str, list[V16_3Example]],
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
    outputs_complete = False
    if final_path.exists() and required_test.exists() and required_permutation.exists():
        test_table = _read_table(required_test)
        permutation_table = _read_table(required_permutation)
        test_rows = test_table[
            (test_table.position_encoding == position_encoding) & (test_table["mode"] == mode)
        ]
        permutation_rows = permutation_table[
            (permutation_table.position_encoding == position_encoding)
            & (permutation_table["mode"] == mode)
        ]
        outputs_complete = (
            set(test_rows.get("suite", pd.Series(dtype=str)).astype(str)) >= {"raw", "task", "mixture"}
            and len(permutation_rows) == len(curve_suites["heldout"]["task"])
            and {"ar_permutation_accuracy", "tf_permutation_accuracy"}.issubset(permutation_rows.columns)
        )
    if skip_completed and final_path.exists() and outputs_complete:
        print(f"[skip] {position_encoding}/{mode}: final checkpoint exists", flush=True)
        return
    if not skip_completed and root.exists() and any(root.glob("step_*/checkpoint.pt")):
        raise RuntimeError(
            f"refusing to overwrite existing checkpoints in {root}; "
            "use --skip-completed to resume or choose a new run name"
        )
    model = paired_v16_3_model(cfg, vocab, position_encoding)
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay
    )
    rng = random.Random(cfg.seed)
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
    sampling_state = restored_sampling_state or _load_sampling_state(
        run_dir, position_encoding, mode, start_step, cfg
    )

    if start_step == 0:
        _write_evaluations(
            model, cfg, vocab, curve_suites, run_dir, position_encoding, mode, 0, run_ar=False
        )
    progress = tqdm(
        range(start_step + 1, cfg.train_steps + 1),
        desc=f"v16_3 {position_encoding}/{mode}",
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
        ids, labels, attention_mask = collate_v16_3(rendered, vocab, cfg.device)
        loss_weights = collate_v16_3_loss_weights(rendered, cfg, cfg.device, step=step)
        with _autocast_context(cfg):
            output = model(input_ids=ids, attention_mask=attention_mask)
            loss, token_losses, active = shifted_v16_3_token_losses(
                output.logits, labels, loss_weights
            )
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
                    loss_weights[row_index, item.spans.count_pos].detach().cpu()
                    for row_index, item in enumerate(rendered)
                    if item.spans is not None
                )
            )
            trace_weight_sum = float(
                sum(
                    loss_weights[row_index, position].detach().cpu()
                    for row_index, item in enumerate(rendered)
                    if item.spans is not None
                    for position in (
                        *item.spans.trace_index_positions,
                        *item.spans.trace_marker_positions,
                    )
                )
            )
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
            print(
                f"[train] {position_encoding}/{mode} "
                f"step={step}/{cfg.train_steps} loss={loss.item():.4f} "
                f"lr={rate:.3e} grad_norm={gradient_norm:.3f} "
                f"task_ratio={is_task.mean():.2f}",
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
            _save_checkpoint(
                model, optimizer, cfg, vocab, pool, split, position_encoding, mode, step,
                rng, run_dir, sync_run_dir, sampling_state=sampling_state,
            )
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
            curve_suites["heldout"]["task"],
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
        by_count = detail.groupby(
            ["position_encoding", "mode", "step", "count", "count_bin"], as_index=False
        ).agg(
            tf_final_accuracy=("tf_final_accuracy", "mean"),
            tf_trace_marker_accuracy=("tf_trace_marker_accuracy", "mean"),
            tf_trace_index_accuracy=("tf_trace_index_accuracy", "mean"),
            frequency_baseline_accuracy=("frequency_baseline_accuracy", "mean"),
        )
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
        summary = ar.groupby(
            ["position_encoding", "mode", "step", "count"], as_index=False
        ).agg(
            ar_final_accuracy=("ar_accuracy", "mean"),
            ar_answer_rate=("ar_answered", "mean"),
            ar_abs_error_answered_only=("ar_abs_error", "mean"),
            ar_abs_error=("ar_abs_error", "mean"),
            ar_abs_error_with_missing_penalty=("ar_abs_error_with_missing_penalty", "mean"),
            trace_exact=("trace_exact", "mean"),
            trace_ordered_marker_accuracy=("trace_ordered_marker_accuracy", "mean"),
            trace_marker_recall=("trace_marker_recall", "mean"),
        )
        atomic_csv(summary, run_dir / "tables" / "autoregressive_by_count.csv")
    train = _read_table(run_dir / "tables" / "train_metrics.csv")
    if not train.empty and "cumulative_sampling_json" in train:
        final = train.sort_values("step").groupby(["position_encoding", "mode"], as_index=False).tail(1)
        rows: list[dict[str, Any]] = []
        for _, item in final.iterrows():
            state = json.loads(item["cumulative_sampling_json"])
            for dimension in ("accepted_counts", "proposed_counts", "frequency_bins", "set_ids"):
                for value, count in state[dimension].items():
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


def train_v16_3_models(
    cfg: V16_3Config,
    vocab: V16_3Vocab,
    text: str,
    split: CorpusSplit,
    pool: NeedlePool,
    curve_suites: dict[str, dict[str, list[V16_3Example]]],
    test_suites: dict[str, list[V16_3Example]],
    run_dir: Path,
    *,
    sync_run_dir: Path | None,
    skip_completed: bool,
) -> None:
    specifications: list[dict[str, Any]] = []
    for position_encoding, mode in cfg.model_variants:
        probe = paired_v16_3_model(cfg, vocab, position_encoding)
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
        train_v16_3_variant(
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


def load_v16_3_checkpoint_model(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    *,
    step: int | None = None,
    label: str | None = None,
    device: str | None = None,
) -> tuple[V16_3Config, V16_3Vocab, NeedlePool, CorpusSplit, TinyPositionCausalLM]:
    """Load one v16_3 model checkpoint after validating all artifact identities."""

    if (step is None) == (label is None):
        raise ValueError("provide exactly one of step or label")
    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    if device is not None:
        cfg = replace(cfg, device=device)
    text = load_corpus_text()
    from .data import load_corpus_split

    split = load_corpus_split(run_dir / "data" / "corpus_split.json", cfg, text)
    vocab = V16_3Vocab.load(run_dir / "vocab.json")
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    model = build_model(cfg, vocab, position_encoding, cfg.device)
    selected = label if label is not None else f"step_{int(step):06d}"
    path = _checkpoint_root(run_dir, position_encoding, mode) / selected / "checkpoint.pt"
    if label == "final" and not path.exists():
        numeric_final = (
            _checkpoint_root(run_dir, position_encoding, mode)
            / f"step_{cfg.train_steps:06d}"
            / "checkpoint.pt"
        )
        if numeric_final.exists():
            path = numeric_final
    if not path.exists():
        raise FileNotFoundError(f"missing v16_3 checkpoint: {path}")
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


def load_final_v16_3_model(
    run_dir: str | Path,
    position_encoding: str,
    mode: str,
    device: str | None = None,
) -> tuple[V16_3Config, V16_3Vocab, NeedlePool, TinyPositionCausalLM]:
    cfg, vocab, pool, _, model = load_v16_3_checkpoint_model(
        run_dir,
        position_encoding,
        mode,
        label="final",
        device=device,
    )
    return cfg, vocab, pool, model
