"""Explicit boundaries and matched residual controls for the transfer pilot."""
from __future__ import annotations

import re
import numpy as np

from protocol import encode_ids, token_end


def strip_end_tokens(text: str) -> str:
    return re.sub(r"(?:<\|im_end\|>|<turn\|>|<\|endoftext\|>|<eos>|</s>)\s*$", "", text).strip()


def native_endpoint(tokenizer, rendered: str, generated: dict, answer_prefix: str):
    """Return an exact original-token prefix ending at the final answer colon.

    Never construct an answer prefix from an answer appearing inside reasoning.
    Never retokenize and silently substitute different original generated IDs.
    """
    raw = generated["completion_text_raw"]
    closers = [raw.rfind(s) + len(s) for s in ("</think>", "<channel|>") if s in raw]
    if not closers:
        return None, "missing_native_channel_close"
    after = max(closers)
    matches = list(re.finditer(r"(?m)^\s*" + re.escape(answer_prefix), raw[after:]))
    if len(matches) != 1:
        return None, "missing_or_ambiguous_final_answer_tag"
    stop = len(rendered) + after + matches[0].end()
    full_text = rendered + raw
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    original = list(prompt_ids) + generated["generated_token_ids"]
    if list(encoded["input_ids"]) != original:
        return None, "decoded_text_does_not_reproduce_original_ids"
    end = token_end(encoded["offset_mapping"], stop - 1, stop)
    if encoded["offset_mapping"][end][1] > stop:
        return None, "answer_value_shares_a_token_with_prefix"
    return (encode_ids(original[:end + 1]), encoded["offset_mapping"], full_text, after), None


def trace_record_sites(case: dict, raw: str, rendered_length: int, close_end: int, offsets: list):
    """Conservative natural item-end sites: one city+score record per line.

    Ambiguous, repeated or out-of-order lines are rejected as a cohort; no gold
    answer is used to invent an item or pick between competing episodes.
    """
    reasoning = raw[:close_end]
    lines, cursor = [], 0
    for line in reasoning.splitlines(keepends=True):
        cities = [r for r in case["records"] if re.search(r"(?<!\w)" + re.escape(r["city"]) + r"(?!\w)", line)]
        if len(cities) == 1 and re.search(r"(?<!\d)" + str(cities[0]["score"]) + r"(?!\d)", line):
            record = cities[0]
            pos = token_end(offsets, rendered_length + cursor, rendered_length + cursor + len(line.rstrip("\r\n")))
            lines.append({"position": pos, "ordinal": record["ordinal"],
                          "is_target": record["is_target"],
                          "explicit_index": bool(re.match(r"\s*\d+[.)]", line)),
                          "line": line.rstrip("\r\n")})
        cursor += len(line)
    order = [r["ordinal"] for r in lines]
    if not order:
        return [], "no_unambiguous_city_score_lines"
    if order != sorted(set(order)):
        return [], "repeated_or_out_of_order_natural_records"
    # Report completeness; keep all unambiguous sites without selecting for final correctness.
    return lines, None


def norm_matched_orthogonal(recipient, donor, seed: int):
    recipient, donor = np.asarray(recipient, dtype=np.float32), np.asarray(donor, dtype=np.float32)
    delta = donor - recipient
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Donor displacement is zero/nonfinite")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(delta.shape).astype(np.float32)
    vector -= delta * (np.dot(vector, delta) / np.dot(delta, delta))
    vector *= norm / np.linalg.norm(vector)
    result = recipient + vector
    return result, {"donor_delta_norm": norm, "control_delta_norm": float(np.linalg.norm(result - recipient)),
                    "delta_cosine": float(np.dot(result - recipient, delta) / (np.linalg.norm(result - recipient) * norm))}


def ridge_predict(train_x, train_y, test_x, alpha: float = 10.0):
    """Centered dual ridge, with all centering and fitting restricted to training."""
    x, y, z = (np.asarray(a, dtype=np.float64) for a in (train_x, train_y, test_x))
    if x.ndim != 2 or z.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Invalid probe shapes or insufficient training examples")
    if not all(np.isfinite(a).all() for a in (x, y, z)):
        raise ValueError("Nonfinite probe input")
    center, mean_y = x.mean(0), y.mean(0)
    x = x - center
    # Per-feature scaling would alter the geometry; normalize only average sample norm.
    scale = max(float(np.sqrt((x * x).sum() / len(x))), 1e-12)
    x, z = x / scale, (z - center) / scale
    coef = np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y - mean_y)
    return z @ x.T @ coef + mean_y
