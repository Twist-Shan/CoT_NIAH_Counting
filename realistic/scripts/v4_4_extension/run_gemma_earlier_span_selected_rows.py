from __future__ import annotations

"""Memory-bounded Gemma Earlier-span confirmation for Realistic NiaH V4.4.

The estimand is identical to the existing Qwen experiment:

    attention mass(prior needle full spans)
      - attention mass(equal-length, same-depth ordinary spans)

Discovery data freeze the top-10 layer/head candidates.  Confirmation uses
seeds 1254--1263 and occurrences 2/4/6/8/10.  The only implementation change
is that we reconstruct the single query-to-key softmax row for the frozen
candidates inside the normal SDPA forward instead of materializing every
layer/head attention matrix with ``output_attentions=True``.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


FALLBACK_SRC = Path(
    "runs/"
    "v4_4_counter_channel_20260806/code/src"
)
if FALLBACK_SRC.is_dir() and str(FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(FALLBACK_SRC))

from realistic_niah_v4.modeling import (
    _accepts_keyword,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _temporary_attention_backend,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward


CAPTURE_BACKEND = "v44_selected_query_rows"
MODEL_LABEL = "Gemma4-E4B"
ATTENTION_COLUMNS = (
    "model_label",
    "seed",
    "split",
    "occurrence",
    "rank",
    "layer",
    "head",
    "attention_key_start",
    "prior_needle_span_mass",
    "prior_matched_nonneedle_mass",
    "prior_span_preference",
    "current_needle_span_mass",
    "attention_row_sum",
)

_ORIGINAL_SDPA = ALL_ATTENTION_FUNCTIONS.get_interface(
    "sdpa", eager_attention_forward
)
_SELECTED_HEADS: dict[int, tuple[int, ...]] = {}
_CAPTURED_ROWS: dict[tuple[int, int], torch.Tensor] = {}


def _repeat_kv(key: torch.Tensor, groups: int) -> torch.Tensor:
    if int(groups) == 1:
        return key
    return key.repeat_interleave(int(groups), dim=1)


def selected_query_row_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Delegate to SDPA while reconstructing only frozen query rows."""

    layer = int(getattr(module, "layer_idx"))
    selected = _SELECTED_HEADS.get(layer, ())
    if selected and int(query.shape[-2]) == 1:
        scale = float(scaling) if scaling is not None else int(module.head_dim) ** -0.5
        repeated_key = _repeat_kv(
            key, int(getattr(module, "num_key_value_groups", 1))
        )
        head_index = torch.as_tensor(selected, dtype=torch.long, device=query.device)
        selected_query = query.index_select(1, head_index)
        selected_key = repeated_key.index_select(1, head_index)
        logits = torch.matmul(
            selected_query, selected_key.transpose(2, 3)
        ) * scale
        if softcap is not None:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        if attention_mask is not None:
            logits = logits + attention_mask
        weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        for offset, head in enumerate(selected):
            _CAPTURED_ROWS[(layer, int(head))] = (
                weights[0, offset, 0].detach().float().cpu()
            )

    delegate_kwargs = dict(kwargs)
    if softcap is not None:
        delegate_kwargs["softcap"] = softcap
    return _ORIGINAL_SDPA(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **delegate_kwargs,
    )


ALL_ATTENTION_FUNCTIONS.register(CAPTURE_BACKEND, selected_query_row_attention)


def slice_mass(values: np.ndarray, start: int, end: int, key_start: int) -> float:
    left = max(int(start), int(key_start)) - int(key_start)
    right = min(int(end), int(key_start) + int(values.shape[0])) - int(key_start)
    if right <= left:
        return 0.0
    return float(values[left:right].sum(dtype=np.float64))


def matched_segments(
    encoding: PromptEncoding, occurrence: int
) -> list[tuple[int, int]]:
    spans = encoding.needle_spans[:occurrence]
    forbidden: set[int] = set()
    for span in tuple(encoding.slot_spans) + tuple(encoding.hard_negative_spans):
        forbidden.update(range(int(span.start), int(span.end)))
    used = set(forbidden)
    matched: list[tuple[int, int]] = []
    for span in spans:
        length = int(span.end) - int(span.start)
        found: tuple[int, int] | None = None
        for gap in range(8, 512):
            candidate = int(span.start) - gap - length
            positions = set(range(candidate, candidate + length))
            if candidate >= 1 and not positions.intersection(used):
                found = (candidate, candidate + length)
                used.update(positions)
                break
        if found is None:
            raise RuntimeError("Could not construct depth-matched non-needle segment")
        matched.append(found)
    return matched


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    stats = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(stats) >= observed - 1e-15))


def bootstrap(values: np.ndarray, reps: int = 50_000) -> tuple[float, float]:
    rng = np.random.default_rng(20260806)
    indices = rng.integers(0, len(values), size=(int(reps), len(values)))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running, min(1.0, (len(values) - rank) * float(values[index]))
        )
        adjusted[index] = running
    return adjusted


def discovery_candidates(
    base_run: Path, occurrences: Sequence[int], top_heads: int
) -> pd.DataFrame:
    root = (
        base_run
        / MODEL_LABEL
        / "numeric"
        / "representation"
        / "prompt_counter_attention_v1"
        / "shards"
        / "v4.4"
    )
    parts: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.csv.gz")):
        frame = pd.read_csv(path)
        if frame.empty or str(frame.iloc[0]["split"]) != "discovery":
            continue
        parts.append(frame[frame["query_occurrence"].isin(list(occurrences))])
    if not parts:
        raise RuntimeError(f"No Gemma discovery attention rows under {root}")
    discovery = pd.concat(parts, ignore_index=True)
    ranked = (
        discovery.groupby(["layer", "head"], as_index=False)[
            "prior_needle_span_mass"
        ]
        .mean()
        .sort_values(
            ["prior_needle_span_mass", "layer", "head"],
            ascending=[False, True, True],
        )
        .head(int(top_heads))
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked = ranked.rename(
        columns={"prior_needle_span_mass": "discovery_prior_span_mass"}
    )
    if len(ranked) != int(top_heads):
        raise RuntimeError(f"Expected {top_heads} candidates; got {len(ranked)}")
    return ranked


@torch.inference_mode()
def capture_occurrence(
    model: Any,
    encoding: PromptEncoding,
    occurrence: int,
) -> dict[tuple[int, int], np.ndarray]:
    query_position = int(encoding.needle_spans[occurrence - 1].end) - 1
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefix_output = model(
        input_ids=input_ids[:, :query_position],
        attention_mask=attention_mask[:, :query_position],
        use_cache=True,
        output_attentions=False,
        **_bounded_logits_kwargs(model),
    )
    past = getattr(prefix_output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Prefix forward returned no cache")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, query_position : query_position + 1],
        "attention_mask": attention_mask[:, : query_position + 1],
        "past_key_values": past,
        "use_cache": False,
        "output_attentions": False,
        **_bounded_logits_kwargs(model),
    }
    if _accepts_keyword(model, "position_ids"):
        kwargs["position_ids"] = torch.tensor(
            [[query_position]], dtype=torch.long, device=input_ids.device
        )
    if _accepts_keyword(model, "cache_position"):
        kwargs["cache_position"] = torch.tensor(
            [query_position], dtype=torch.long, device=input_ids.device
        )
    shared = getattr(prefix_output, "shared_kv_states", None)
    if shared is not None and _accepts_keyword(model, "shared_kv_states"):
        kwargs["shared_kv_states"] = shared

    _CAPTURED_ROWS.clear()
    with _temporary_attention_backend(model, CAPTURE_BACKEND):
        model(**kwargs)
    expected = {(layer, head) for layer, heads in _SELECTED_HEADS.items() for head in heads}
    missing = sorted(expected.difference(_CAPTURED_ROWS))
    if missing:
        raise RuntimeError(f"Selected attention rows were not captured: {missing}")
    result = {
        key: value.numpy().astype(np.float64, copy=True)
        for key, value in _CAPTURED_ROWS.items()
    }
    del prefix_output, past, shared
    return result


def summarize(raw: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_effects = (
        raw.groupby(["rank", "layer", "head", "seed"], as_index=False)[
            "prior_span_preference"
        ]
        .mean()
        .sort_values(["rank", "seed"])
    )
    stats: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        selected = seed_effects[
            (seed_effects["layer"] == int(candidate.layer))
            & (seed_effects["head"] == int(candidate.head))
        ]
        values = selected["prior_span_preference"].to_numpy(float)
        low, high = bootstrap(values)
        stats.append(
            {
                "model_label": MODEL_LABEL,
                "rank": int(candidate.rank),
                "layer": int(candidate.layer),
                "head": int(candidate.head),
                "discovery_prior_span_mass": float(
                    candidate.discovery_prior_span_mass
                ),
                "confirmation_preference_mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "p_value": exact_sign_flip(values),
                "seed_count": int(len(values)),
            }
        )
    summary = pd.DataFrame(stats).sort_values("rank").reset_index(drop=True)
    summary["holm_p_within_model"] = holm(summary["p_value"].to_numpy(float))
    seed_effects.insert(0, "model_label", MODEL_LABEL)
    return summary, seed_effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1254, 1264)))
    parser.add_argument("--occurrences", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--top-heads", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_root = args.output / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    candidates = discovery_candidates(args.base_run, args.occurrences, args.top_heads)
    candidates.to_csv(args.output / "discovery_frozen_candidates.csv", index=False)
    global _SELECTED_HEADS
    _SELECTED_HEADS = {
        int(layer): tuple(int(value) for value in part["head"].tolist())
        for layer, part in candidates.groupby("layer", sort=True)
    }

    config = V4Config.from_json(args.v4_config)
    all_stimuli = load_stimuli(args.stimuli)
    stimuli = {
        int(row["seed"]): row
        for row in all_stimuli
        if str(row.get("design_variant")) == "v4.4"
        and int(row.get("gold_count", -1)) == 10
    }
    missing_seeds = sorted(set(args.seeds).difference(stimuli))
    if missing_seeds:
        raise RuntimeError(f"Stimuli missing seeds: {missing_seeds}")

    spec = resolve_model_spec(MODEL_LABEL)
    model, tokenizer, _adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=config.model_torch_dtype,
        attention_backend=config.attention_prefix_backend,
    )
    raw_parts: list[pd.DataFrame] = []
    candidate_lookup = {
        (int(row.layer), int(row.head)): int(row.rank)
        for row in candidates.itertuples(index=False)
    }
    for seed in args.seeds:
        raw_path = raw_root / f"seed_{seed}.csv.gz"
        if raw_path.exists() and not args.overwrite:
            raw_parts.append(pd.read_csv(raw_path))
            continue
        encoding = render_v4_prompt(
            stimuli[int(seed)],
            tokenizer=tokenizer,
            model_spec=spec,
            config=config,
            answer_format="numeric",
        )
        records: list[dict[str, Any]] = []
        for occurrence in args.occurrences:
            matched = matched_segments(encoding, int(occurrence))
            rows = capture_occurrence(model, encoding, int(occurrence))
            current = encoding.needle_spans[int(occurrence) - 1]
            prior = encoding.needle_spans[: int(occurrence) - 1]
            query_position = int(current.end) - 1
            for (layer, head), weights in rows.items():
                key_start = query_position + 1 - int(weights.shape[0])
                prior_needle = sum(
                    slice_mass(weights, int(span.start), int(span.end), key_start)
                    for span in prior
                )
                prior_matched = sum(
                    slice_mass(weights, int(start), int(end), key_start)
                    for start, end in matched[: int(occurrence) - 1]
                )
                current_mass = slice_mass(
                    weights, int(current.start), int(current.end), key_start
                )
                records.append(
                    {
                        "model_label": MODEL_LABEL,
                        "seed": int(seed),
                        "split": "confirmation",
                        "occurrence": int(occurrence),
                        "rank": candidate_lookup[(layer, head)],
                        "layer": int(layer),
                        "head": int(head),
                        "attention_key_start": int(key_start),
                        "prior_needle_span_mass": prior_needle,
                        "prior_matched_nonneedle_mass": prior_matched,
                        "prior_span_preference": prior_needle - prior_matched,
                        "current_needle_span_mass": current_mass,
                        "attention_row_sum": float(weights.sum(dtype=np.float64)),
                    }
                )
            print(
                f"[gemma-earlier-span] seed={seed} occurrence={occurrence} "
                f"rows={len(rows)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        frame = pd.DataFrame(records, columns=ATTENTION_COLUMNS)
        frame.to_csv(raw_path, index=False, compression="gzip")
        raw_parts.append(frame)

    raw = pd.concat(raw_parts, ignore_index=True)
    summary, seed_effects = summarize(raw, candidates)
    raw.to_csv(
        args.output / "gemma_earlier_span_confirmation_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    seed_effects.to_csv(
        args.output / "gemma_earlier_span_head_seed_effects.csv", index=False
    )
    summary.to_csv(
        args.output / "gemma_earlier_span_head_confirmation.csv", index=False
    )

    expected_rows = len(args.seeds) * len(args.occurrences) * int(args.top_heads)
    row_sum_error = float(np.max(np.abs(raw["attention_row_sum"].to_numpy(float) - 1.0)))
    finite = bool(
        np.isfinite(
            raw[
                [
                    "prior_needle_span_mass",
                    "prior_matched_nonneedle_mass",
                    "prior_span_preference",
                    "current_needle_span_mass",
                    "attention_row_sum",
                ]
            ].to_numpy(float)
        ).all()
    )
    passed = (
        len(raw) == expected_rows
        and len(summary) == int(args.top_heads)
        and set(summary["seed_count"].astype(int)) == {len(args.seeds)}
        and finite
        and row_sum_error <= 1e-5
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_gemma_earlier_span_selected_rows_v1",
        "model_label": MODEL_LABEL,
        "measurement_equivalence": (
            "Same discovery-ranked top-10, confirmation seeds, occurrences, full-span "
            "mass, and equal-length same-depth ordinary contrast as Qwen. Only frozen "
            "candidate query rows are reconstructed inside SDPA to avoid quadratic "
            "all-layer attention materialization."
        ),
        "discovery_base_run": str(args.base_run),
        "seeds": list(map(int, args.seeds)),
        "occurrences": list(map(int, args.occurrences)),
        "top_heads": int(args.top_heads),
        "expected_raw_rows": int(expected_rows),
        "observed_raw_rows": int(len(raw)),
        "candidate_count": int(len(summary)),
        "all_finite": finite,
        "max_attention_row_sum_error": row_sum_error,
        "inference": (
            "Per head, average occurrences within each seed; two-sided exact sign-flip "
            "over ten seed effects; Holm correction over ten frozen Gemma heads; "
            "50000 seed bootstrap CI."
        ),
        "status": "PASS" if passed else "FAIL",
    }
    (args.output / "gemma_earlier_span_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    if not passed:
        raise RuntimeError(f"Audit failed: {audit}")


if __name__ == "__main__":
    main()
