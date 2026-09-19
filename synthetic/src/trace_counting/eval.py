from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .io_utils import ensure_dir, read_jsonl, save_json
from .tokenizer import VocabTokenizer, parse_count_token


REPEAT_COUNT_FORMATS = {"think_trace_repeat_count", "answer_only_repeat_count"}
THINK_FORMATS = {"think_trace", "think_trace_repeat_count"}


def is_repeat_count_format(task_format: str) -> bool:
    return task_format in REPEAT_COUNT_FORMATS


def is_think_format(task_format: str) -> bool:
    return task_format in THINK_FORMATS


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and (path / "config.json").exists():
        return path
    if path.exists() and (path / "checkpoints" / "final" / "config.json").exists():
        return path / "checkpoints" / "final"
    if not path.exists() and path.name == "checkpoint_final":
        candidate = path.parent / "checkpoints" / "final"
        if (candidate / "config.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint directory from {path}")


def density_bucket(count: int, seq_len: int) -> str:
    density = count / max(seq_len, 1)
    if count == 0:
        return "zero"
    if density <= 0.02:
        return "<=0.02"
    if density <= 0.05:
        return "<=0.05"
    if density <= 0.10:
        return "<=0.10"
    return ">0.10"


@torch.no_grad()
def teacher_forced_predict(
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    example: dict,
    device: str,
) -> dict[str, Any]:
    task_format = example.get("task_format", "think_trace")
    if is_repeat_count_format(task_format):
        return teacher_forced_predict_repeated_count(model, tokenizer, example, device)
    ans_idx = example["spans"]["ans_idx"]
    prefix_ids = tokenizer.encode(example["full_tokens"][: ans_idx + 1])
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
    count_ids = torch.tensor(tokenizer.count_token_ids, dtype=torch.long, device=device)
    count_logits = logits.index_select(0, count_ids)
    log_probs = F.log_softmax(count_logits, dim=-1)
    pred_count = int(torch.argmax(count_logits).item())
    true_count = int(example["count"])
    return {
        "pred_count": pred_count,
        "count_nll": float(-log_probs[true_count].detach().cpu()),
        "correct": pred_count == true_count,
    }


@torch.no_grad()
def teacher_forced_predict_repeated_count(
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    example: dict,
    device: str,
) -> dict[str, Any]:
    spans = example["spans"]
    ans_idx = int(spans["ans_idx"])
    eos_idx = int(spans["eos_idx"])
    input_ids = torch.tensor([tokenizer.encode(example["full_tokens"][: eos_idx + 1])], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
    target_tokens = example["full_tokens"][ans_idx + 1 : eos_idx + 1]
    pred_ids = torch.argmax(logits[ans_idx:eos_idx], dim=-1).detach().cpu().tolist()
    pred_tokens = tokenizer.decode(pred_ids)
    pred_count = 0
    saw_eos = False
    for token in pred_tokens:
        if token == "<CNT>" and not saw_eos:
            pred_count += 1
        elif token == "<EOS>":
            saw_eos = True
            break
        else:
            break
    true_count = int(example["count"])
    log_probs = F.log_softmax(logits[ans_idx:eos_idx], dim=-1)
    target_ids = tokenizer.encode(target_tokens)
    nll = -log_probs[torch.arange(len(target_ids), device=device), torch.tensor(target_ids, device=device)]
    return {
        "pred_count": pred_count if saw_eos or pred_count > 0 else 0,
        "count_nll": float(nll.mean().detach().cpu()) if len(target_ids) else None,
        "correct": pred_tokens == target_tokens,
        "pred_answer_tokens": pred_tokens,
    }


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    prefix_tokens: list[str],
    device: str,
    *,
    max_new_tokens: int,
) -> list[str]:
    ids = tokenizer.encode(prefix_tokens)
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        ids.append(next_id)
        if next_id == eos_id:
            break
    return tokenizer.decode(ids[len(prefix_tokens) :])


def _find_token(tokens: list[str], token: str, start: int = 0) -> int | None:
    try:
        return tokens.index(token, start)
    except ValueError:
        return None


def parse_generation(tokens: list[str], *, task_format: str = "think_trace") -> dict[str, Any]:
    if task_format == "answer_only_repeat_count":
        return parse_answer_only_repeat_count_generation(tokens)
    if task_format == "think_trace_repeat_count":
        return parse_think_repeat_count_generation(tokens)
    if task_format == "answer_only":
        return parse_answer_only_generation(tokens)
    if task_format != "think_trace":
        raise ValueError(f"Unknown task_format={task_format!r}")
    first_think = _find_token(tokens, "<Think>")
    if first_think is None:
        return {"format_valid": False, "invalid_reason": "missing_first_think", "trace_tokens": [], "pred_count": None}
    second_think = _find_token(tokens, "<Think>", first_think + 1)
    if second_think is None:
        return {"format_valid": False, "invalid_reason": "missing_second_think", "trace_tokens": tokens[first_think + 1 :], "pred_count": None}
    ans_idx = _find_token(tokens, "<ANS>", second_think + 1)
    if ans_idx is None:
        return {
            "format_valid": False,
            "invalid_reason": "missing_ans",
            "trace_tokens": tokens[first_think + 1 : second_think],
            "pred_count": None,
        }
    if ans_idx + 1 >= len(tokens):
        return {
            "format_valid": False,
            "invalid_reason": "missing_count_after_ans",
            "trace_tokens": tokens[first_think + 1 : second_think],
            "pred_count": None,
        }
    answer_token = tokens[ans_idx + 1]
    pred_count = parse_count_token(answer_token)
    if pred_count is None:
        return {
            "format_valid": False,
            "invalid_reason": "invalid_count_token",
            "trace_tokens": tokens[first_think + 1 : second_think],
            "answer_token": answer_token,
            "pred_count": None,
        }
    eos_after_count = ans_idx + 2 < len(tokens) and tokens[ans_idx + 2] == "<EOS>"
    return {
        "format_valid": True,
        "invalid_reason": None,
        "first_think_idx": first_think,
        "second_think_idx": second_think,
        "ans_idx": ans_idx,
        "trace_tokens": tokens[first_think + 1 : second_think],
        "answer_token": answer_token,
        "pred_count": pred_count,
        "eos_after_count": eos_after_count,
    }


def parse_think_repeat_count_generation(tokens: list[str]) -> dict[str, Any]:
    first_think = _find_token(tokens, "<Think>")
    if first_think is None:
        return {"format_valid": False, "invalid_reason": "missing_first_think", "trace_tokens": [], "pred_count": None}
    second_think = _find_token(tokens, "<Think>", first_think + 1)
    if second_think is None:
        return {"format_valid": False, "invalid_reason": "missing_second_think", "trace_tokens": tokens[first_think + 1 :], "pred_count": None}
    ans_idx = _find_token(tokens, "<ANS>", second_think + 1)
    if ans_idx is None:
        return {
            "format_valid": False,
            "invalid_reason": "missing_ans",
            "trace_tokens": tokens[first_think + 1 : second_think],
            "pred_count": None,
        }
    parsed = _parse_repeated_count_answer(tokens, ans_idx)
    parsed.update(
        {
            "first_think_idx": first_think,
            "second_think_idx": second_think,
            "ans_idx": ans_idx,
            "trace_tokens": tokens[first_think + 1 : second_think],
        }
    )
    return parsed


def parse_answer_only_repeat_count_generation(tokens: list[str]) -> dict[str, Any]:
    ans_idx = _find_token(tokens, "<ANS>")
    if ans_idx is None:
        return {"format_valid": False, "invalid_reason": "missing_ans", "trace_tokens": [], "pred_count": None}
    parsed = _parse_repeated_count_answer(tokens, ans_idx)
    parsed.update({"ans_idx": ans_idx, "trace_tokens": []})
    return parsed


def _parse_repeated_count_answer(tokens: list[str], ans_idx: int) -> dict[str, Any]:
    eos_idx = _find_token(tokens, "<EOS>", ans_idx + 1)
    if eos_idx is None:
        answer_tokens = tokens[ans_idx + 1 :]
        valid_prefix = all(token == "<CNT>" for token in answer_tokens)
        return {
            "format_valid": False,
            "invalid_reason": "missing_eos" if valid_prefix else "invalid_count_token",
            "answer_tokens": answer_tokens,
            "pred_count": None,
            "eos_after_count": False,
        }
    answer_tokens = tokens[ans_idx + 1 : eos_idx]
    if any(token != "<CNT>" for token in answer_tokens):
        return {
            "format_valid": False,
            "invalid_reason": "invalid_count_token",
            "answer_tokens": answer_tokens,
            "pred_count": None,
            "eos_after_count": True,
        }
    return {
        "format_valid": True,
        "invalid_reason": None,
        "answer_tokens": answer_tokens,
        "answer_token": "<CNT>",
        "pred_count": len(answer_tokens),
        "eos_after_count": True,
    }


def parse_answer_only_generation(tokens: list[str]) -> dict[str, Any]:
    ans_idx = _find_token(tokens, "<ANS>")
    if ans_idx is None:
        return {"format_valid": False, "invalid_reason": "missing_ans", "trace_tokens": [], "pred_count": None}
    if ans_idx + 1 >= len(tokens):
        return {"format_valid": False, "invalid_reason": "missing_count_after_ans", "trace_tokens": [], "pred_count": None}
    answer_token = tokens[ans_idx + 1]
    pred_count = parse_count_token(answer_token)
    if pred_count is None:
        return {
            "format_valid": False,
            "invalid_reason": "invalid_count_token",
            "trace_tokens": [],
            "answer_token": answer_token,
            "pred_count": None,
        }
    eos_after_count = ans_idx + 2 < len(tokens) and tokens[ans_idx + 2] == "<EOS>"
    return {
        "format_valid": True,
        "invalid_reason": None,
        "ans_idx": ans_idx,
        "trace_tokens": [],
        "answer_token": answer_token,
        "pred_count": pred_count,
        "eos_after_count": eos_after_count,
    }


def trace_metrics(generated_trace: list[str], expected_trace: list[str]) -> dict[str, float | bool | int]:
    trace_exact = generated_trace == expected_trace
    expected_indices = expected_trace[0::2]
    if not expected_indices:
        trace_index_accuracy = 1.0 if not generated_trace else 0.0
    else:
        correct = 0
        for i, expected in enumerate(expected_indices):
            gen_pos = 2 * i
            correct += int(gen_pos < len(generated_trace) and generated_trace[gen_pos] == expected)
        trace_index_accuracy = correct / len(expected_indices)

    expected_markers = Counter(token for token in expected_trace if token in {"X", "Y", "Z"})
    generated_markers = Counter(token for token in generated_trace if token in {"X", "Y", "Z"})
    overlap = sum((expected_markers & generated_markers).values())
    generated_marker_total = sum(generated_markers.values())
    expected_marker_total = sum(expected_markers.values())
    precision = 1.0 if generated_marker_total == 0 and expected_marker_total == 0 else overlap / max(generated_marker_total, 1)
    recall = 1.0 if expected_marker_total == 0 else overlap / expected_marker_total

    generated_indices = [token for token in generated_trace if token.startswith("<I")]
    duplicate_rate = 0.0
    if generated_indices:
        duplicate_rate = (len(generated_indices) - len(set(generated_indices))) / len(generated_indices)
    return {
        "trace_exact_match": bool(trace_exact),
        "trace_index_accuracy": float(trace_index_accuracy),
        "trace_marker_precision": float(precision),
        "trace_marker_recall": float(recall),
        "trace_duplicate_rate": float(duplicate_rate),
        "trace_length_error": int(len(generated_trace) - len(expected_trace)),
    }


def invalid_trace_metrics(generated_trace: list[str], expected_trace: list[str]) -> dict[str, float | bool | int]:
    return {
        "trace_exact_match": False,
        "trace_index_accuracy": 0.0,
        "trace_marker_precision": 0.0,
        "trace_marker_recall": 0.0,
        "trace_duplicate_rate": 0.0,
        "trace_length_error": int(len(generated_trace) - len(expected_trace)),
    }


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def count_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    valid_rows = [row for row in rows if row.get("pred_count") is not None]
    correct = sum(int(row.get("pred_count") == row["true_count"]) for row in rows)
    errors = [abs(int(row["pred_count"]) - int(row["true_count"])) for row in valid_rows]
    under = sum(int(row["pred_count"] < row["true_count"]) for row in valid_rows)
    over = sum(int(row["pred_count"] > row["true_count"]) for row in valid_rows)
    out = {
        "count_accuracy": correct / max(n, 1),
        "mean_absolute_error": _safe_mean(errors),
        "undercount_rate": under / max(n, 1),
        "overcount_rate": over / max(n, 1),
        "invalid_answer_rate": (n - len(valid_rows)) / max(n, 1),
    }
    nlls = [row["count_nll"] for row in rows if row.get("count_nll") is not None and math.isfinite(row["count_nll"])]
    if nlls:
        out["count_nll"] = float(np.mean(nlls))
    return out


def grouped_count_metrics(rows: list[dict], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {group_key: count_metrics(group_rows) for group_key, group_rows in sorted(groups.items())}


def aggregate_trace_metrics(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {}
    metric_names = [
        "trace_exact_match",
        "trace_index_accuracy",
        "trace_marker_precision",
        "trace_marker_recall",
        "trace_duplicate_rate",
    ]
    out = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        out[name] = _safe_mean(values)
    out["format_validity"] = _safe_mean([float(row["format_valid"]) for row in rows])
    out["mean_trace_length_error"] = _safe_mean(
        [abs(float(row["trace_length_error"])) for row in rows if row.get("trace_length_error") is not None]
    )
    return out


def grouped_trace_metrics(rows: list[dict], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {group_key: aggregate_trace_metrics(group_rows) for group_key, group_rows in sorted(groups.items())}


def evaluate_split(
    *,
    model: torch.nn.Module,
    tokenizer: VocabTokenizer,
    examples: list[dict],
    split: str,
    device: str,
    max_new_tokens: int,
    mode: str = "both",
) -> tuple[dict[str, Any], list[dict]]:
    predictions: list[dict] = []
    tf_rows: list[dict] = []
    ar_rows: list[dict] = []
    for example in tqdm(examples, desc=f"eval {split}", dynamic_ncols=True):
        base = {
            "example_id": example["example_id"],
            "split": split,
            "seq_len": int(example["seq_len"]),
            "true_count": int(example["count"]),
            "density_bucket": density_bucket(int(example["count"]), int(example["seq_len"])),
        }
        pred_row = dict(base)
        if mode in {"both", "teacher_forced"}:
            tf = teacher_forced_predict(model, tokenizer, example, device)
            tf_row = {**base, **tf}
            tf_rows.append(tf_row)
            pred_row.update({f"tf_{key}": value for key, value in tf.items()})
        if mode in {"both", "autoregressive"}:
            prefix = example["full_tokens"][: example["spans"]["source_end_exclusive"]]
            generated = greedy_generate(model, tokenizer, prefix, device, max_new_tokens=max_new_tokens)
            task_format = example.get("task_format", "think_trace")
            parsed = parse_generation(generated, task_format=task_format)
            if parsed.get("format_valid", False):
                if task_format in {"answer_only", "answer_only_repeat_count"}:
                    t_metrics = {
                        "trace_exact_match": None,
                        "trace_index_accuracy": None,
                        "trace_marker_precision": None,
                        "trace_marker_recall": None,
                        "trace_duplicate_rate": None,
                        "trace_length_error": None,
                    }
                else:
                    t_metrics = trace_metrics(parsed.get("trace_tokens", []), example["trace_tokens"])
            else:
                t_metrics = invalid_trace_metrics(parsed.get("trace_tokens", []), example["trace_tokens"])
            ar_row = {
                **base,
                "pred_count": parsed.get("pred_count"),
                "format_valid": bool(parsed.get("format_valid", False)),
                "invalid_reason": parsed.get("invalid_reason"),
                **t_metrics,
            }
            ar_rows.append(ar_row)
            pred_row.update(
                {
                    "generated_tokens": generated,
                    "ar_pred_count": parsed.get("pred_count"),
                    "ar_format_valid": parsed.get("format_valid", False),
                    "ar_invalid_reason": parsed.get("invalid_reason"),
                    **{f"ar_{key}": value for key, value in t_metrics.items()},
                }
            )
        predictions.append(pred_row)

    metrics: dict[str, Any] = {"split": split, "n_examples": len(examples)}
    if tf_rows:
        metrics["teacher_forced"] = {
            **count_metrics(tf_rows),
            "accuracy_by_count": grouped_count_metrics(tf_rows, "true_count"),
            "accuracy_by_seq_len": grouped_count_metrics(tf_rows, "seq_len"),
            "accuracy_by_density": grouped_count_metrics(tf_rows, "density_bucket"),
        }
    if ar_rows:
        metrics["autoregressive"] = {
            **count_metrics(ar_rows),
            **aggregate_trace_metrics(ar_rows),
            "accuracy_by_count": grouped_count_metrics(ar_rows, "true_count"),
            "accuracy_by_seq_len": grouped_count_metrics(ar_rows, "seq_len"),
            "accuracy_by_density": grouped_count_metrics(ar_rows, "density_bucket"),
            "trace_by_count": grouped_trace_metrics(ar_rows, "true_count"),
            "trace_by_seq_len": grouped_trace_metrics(ar_rows, "seq_len"),
        }
    return metrics, predictions


def write_predictions(path: Path, predictions: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(__import__("json").dumps(row, sort_keys=True) + "\n")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .model import load_model_from_checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    tokenizer = VocabTokenizer.load(data_dir / "vocab.json")
    model = load_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    max_new_tokens = args.max_new_tokens or (3 * int(tokenizer.metadata.get("max_count", 64)) + 4)

    summary = {}
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        examples = read_jsonl(data_dir / f"{split}.jsonl", limit=args.limit)
        metrics, predictions = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            split=split,
            device=device,
            max_new_tokens=max_new_tokens,
            mode=args.mode,
        )
        save_json(metrics, out_dir / f"{split}_metrics.json")
        write_predictions(out_dir / f"predictions_{split}.jsonl", predictions)
        summary[split] = metrics
    save_json(summary, out_dir / "summary_metrics.json")
    print(f"saved evaluation to {out_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate teacher-forced count readout and autoregressive trace generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--splits", default="val_id,val_length_ood,val_density_shift_low,val_density_shift_high")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--mode", choices=["both", "teacher_forced", "autoregressive"], default="both")
    return parser


def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
