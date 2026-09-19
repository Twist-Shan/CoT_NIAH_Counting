"""v10-style representation and causal analyses for the audited v16.3 RoPE run.

This module deliberately separates three evidence levels:

1. descriptive attention and training dynamics (read from the existing audit tables),
2. decodable residual geometry (fit on disjoint train/held-out examples), and
3. interventions on attention-head inputs, MLP features, residual states, and
   length-preserving trace conflicts.

The v10 task used abstract marker tokens and counts 1..30.  v16.3 instead uses
real character tokens and counts 1..10.  The analyses below preserve the
scientific question and intervention semantics while recording those task-level
adaptations in ``analysis_crosswalk.csv``.
"""

from __future__ import annotations

# The user's current Python installation contains optional NumPy-1.x builds of
# pyarrow/numexpr/bottleneck next to NumPy 2.x.  Pandas treats them as optional;
# making their absence explicit avoids noisy binary-compatibility tracebacks.
import sys

for _optional in ("pyarrow", "numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

import contextlib
import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch

# Pandas 3 otherwise selects Arrow-backed strings when a stale pyarrow package
# is discoverable.  The Python string backend is sufficient for CSV artifacts
# and keeps the analysis independent of that optional binary dependency.
pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

from .config import V16_3Config, config_from_dict
from .data import (
    V16_3Example,
    V16_3Rendered,
    V16_3Vocab,
    collate_v16_3,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
    render_v16_3,
    render_v16_3_shortened_trace,
)
from .needle_pool import NeedlePool, load_needle_pool
from .training import load_v16_3_checkpoint_model


Head = tuple[int, int]  # one-based layer, zero-based head


@dataclass(frozen=True)
class PortOptions:
    """Sampling controls chosen to keep the full causal suite reproducible."""

    examples_per_count: int = 4
    centroid_train_per_count: int = 10
    retrieval_selection_examples: int = 10
    retrieval_reporting_examples: int = 24
    random_paths: int = 6
    seed: int = 6162


@dataclass
class RunContext:
    run_dir: Path
    cfg: V16_3Config
    vocab: V16_3Vocab
    pool: NeedlePool
    train_examples: list[V16_3Example]
    heldout_examples: list[V16_3Example]
    models: dict[str, torch.nn.Module]
    device: str


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_context(run_dir: str | Path, device: str | None = None) -> RunContext:
    run_dir = Path(run_dir).resolve()
    config = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(config, device=selected_device)
    if config.position_encodings != ("rope",):
        raise ValueError("the v10 port is defined for the audited RoPE-only v16.3 run")
    vocab = V16_3Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data" / "corpus_split.json", config, corpus)
    pool = load_needle_pool(
        run_dir / "data" / "needle_pool.json",
        config,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, _ = load_suite_manifests(
        run_dir / "data" / "loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    models: dict[str, torch.nn.Module] = {}
    for mode in ("nonthinking", "thinking"):
        _, loaded_vocab, loaded_pool, loaded_split, model = load_v16_3_checkpoint_model(
            run_dir, "rope", mode, step=config.train_steps, device=selected_device
        )
        if loaded_vocab.fingerprint != vocab.fingerprint:
            raise ValueError("loaded vocabulary differs between v16.3 model variants")
        if loaded_pool.pool_fingerprint != pool.pool_fingerprint:
            raise ValueError("loaded needle pool differs between v16.3 model variants")
        if loaded_split.split_fingerprint != split.split_fingerprint:
            raise ValueError("loaded corpus split differs between v16.3 model variants")
        models[mode] = model.eval()
    return RunContext(
        run_dir=run_dir,
        cfg=config,
        vocab=vocab,
        pool=pool,
        train_examples=list(curves["train"]["task"]),
        heldout_examples=list(curves["heldout"]["task"]),
        models=models,
        device=selected_device,
    )


def _balanced(examples: Sequence[V16_3Example], per_count: int) -> list[V16_3Example]:
    buckets: dict[int, list[V16_3Example]] = {count: [] for count in range(1, 11)}
    for example in examples:
        count = int(example.count or 0)
        if count in buckets and len(buckets[count]) < per_count:
            buckets[count].append(example)
    missing = {count: per_count - len(values) for count, values in buckets.items() if len(values) < per_count}
    if missing:
        raise ValueError(f"balanced v10-port sample is incomplete: {missing}")
    return [example for count in range(1, 11) for example in buckets[count]]


def _count_band(count: int) -> str:
    if count <= 3:
        return "1-3"
    if count <= 7:
        return "4-7"
    return "8-10"


def _replace_tokens(item: V16_3Rendered, replacements: dict[int, str], vocab: V16_3Vocab) -> V16_3Rendered:
    tokens = list(item.tokens)
    for position, token in replacements.items():
        if not 0 <= int(position) < len(tokens):
            raise IndexError(f"replacement position {position} is outside rendered sequence")
        tokens[int(position)] = token
    ids = vocab.encode(tokens)
    return replace(item, tokens=tokens, input_ids=ids, labels=list(ids))


def _truncate(item: V16_3Rendered, length: int) -> V16_3Rendered:
    return replace(
        item,
        tokens=item.tokens[:length],
        input_ids=item.input_ids[:length],
        labels=item.labels[:length],
    )


def _non_target_token(example: V16_3Example, vocab: V16_3Vocab) -> str:
    targets = set(example.needle_characters or ())
    for token in vocab.character_tokens:
        character = chr(int(token[4:-1], 16))
        if character not in targets:
            return token
    raise RuntimeError("vocabulary contains no non-target character token")


def _forward(
    model: torch.nn.Module,
    items: Sequence[V16_3Rendered],
    vocab: V16_3Vocab,
    device: str,
    **kwargs: Any,
):
    ids, _, mask = collate_v16_3(list(items), vocab, device)
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=mask, **kwargs)


def _count_metrics(logits: torch.Tensor, items: Sequence[V16_3Rendered], vocab: V16_3Vocab) -> pd.DataFrame:
    number_ids = torch.tensor(vocab.number_ids, device=logits.device)
    rows: list[dict[str, Any]] = []
    for row_index, item in enumerate(items):
        if item.spans is None or item.count is None:
            continue
        values = logits[row_index, item.spans.ans_pos, number_ids].float()
        probabilities = torch.softmax(values, dim=-1)
        expected = float((probabilities * torch.arange(1, 11, device=values.device)).sum().cpu())
        gold_index = int(item.count) - 1
        alternatives = torch.cat((values[:gold_index], values[gold_index + 1 :]))
        rows.append(
            {
                "count": int(item.count),
                "count_band": _count_band(int(item.count)),
                "predicted_count": int(values.argmax().item()) + 1,
                "accuracy": float(values.argmax().item() == gold_index),
                "expected_count": expected,
                "gold_probability_restricted": float(probabilities[gold_index].cpu()),
                "gold_margin": float((values[gold_index] - alternatives.max()).cpu()),
            }
        )
    return pd.DataFrame(rows)


def _paired_count_margin(
    logits: torch.Tensor,
    items: Sequence[V16_3Rendered],
    vocab: V16_3Vocab,
    alternatives: Sequence[int],
) -> np.ndarray:
    margins = []
    for row, (item, alternative) in enumerate(zip(items, alternatives, strict=True)):
        assert item.spans is not None and item.count is not None
        query = item.spans.ans_pos
        gold = vocab.token_to_id[vocab.number_token(int(item.count))]
        other = vocab.token_to_id[vocab.number_token(int(alternative))]
        margins.append(float((logits[row, query, gold] - logits[row, query, other]).detach().cpu()))
    return np.asarray(margins)


def _normalized_recovery(clean: np.ndarray, corrupt: np.ndarray, patched: np.ndarray) -> np.ndarray:
    denominator = clean - corrupt
    return np.divide(
        patched - corrupt,
        denominator,
        out=np.full_like(clean, np.nan, dtype=float),
        where=np.abs(denominator) > 1e-6,
    )


def _capture_attention_inputs(
    model: torch.nn.Module,
    items: Sequence[V16_3Rendered],
    vocab: V16_3Vocab,
    device: str,
) -> tuple[Any, dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.layers, start=1):
        def hook(_module, args, layer_index=layer_index):
            captured[layer_index] = args[0].detach().clone()
        handles.append(layer.attention.output.register_forward_pre_hook(hook))
    try:
        output = _forward(model, items, vocab, device)
    finally:
        for handle in handles:
            handle.remove()
    return output, captured


def _attention_vectors(
    captured: dict[int, torch.Tensor], positions: Sequence[int]
) -> dict[int, torch.Tensor]:
    rows = torch.arange(len(positions), device=next(iter(captured.values())).device)
    return {
        layer: tensor[rows, torch.tensor(positions, device=tensor.device)].detach().clone()
        for layer, tensor in captured.items()
    }


@contextlib.contextmanager
def _local_attention_edit(
    model: torch.nn.Module,
    heads: Sequence[Head],
    positions: Sequence[Sequence[int]],
    donor_vectors: dict[int, torch.Tensor] | None = None,
) -> Iterator[None]:
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []
    head_dim = int(model.config.n_embd // model.config.n_head)
    for layer_index, selected_heads in by_layer.items():
        module = model.layers[layer_index - 1].attention.output

        def hook(_module, args, layer_index=layer_index, selected_heads=tuple(selected_heads)):
            hidden = args[0].clone()
            for row, row_positions in enumerate(positions):
                for position in row_positions:
                    for head in selected_heads:
                        start, end = head * head_dim, (head + 1) * head_dim
                        if donor_vectors is None:
                            hidden[row, int(position), start:end] = 0
                        else:
                            hidden[row, int(position), start:end] = donor_vectors[layer_index][row, start:end]
            return (hidden, *args[1:])

        handles.append(module.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def _position_output_patch(
    module: torch.nn.Module,
    positions: Sequence[int],
    donor_vectors: torch.Tensor,
    feature_indices: Sequence[int] | None = None,
) -> Iterator[None]:
    def hook(_module, _args, output):
        patched = output.clone()
        for row, position in enumerate(positions):
            if feature_indices is None:
                patched[row, int(position)] = donor_vectors[row]
            else:
                index = torch.tensor(feature_indices, device=patched.device, dtype=torch.long)
                patched[row, int(position), index] = donor_vectors[row, index]
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _residual_patch(
    model: torch.nn.Module,
    layer: int,
    positions: Sequence[int],
    donor_vectors: torch.Tensor,
) -> Iterator[None]:
    def hook(_module, _args, output):
        hidden, weights = output
        patched = hidden.clone()
        for row, position in enumerate(positions):
            patched[row, int(position)] = donor_vectors[row]
        return patched, weights

    handle = model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def _prelayer_residual_patch(
    model: torch.nn.Module,
    layer: int,
    positions: Sequence[int],
    donor_vectors: torch.Tensor,
) -> Iterator[None]:
    def hook(_module, args):
        hidden = args[0].clone()
        for row, position in enumerate(positions):
            hidden[row, int(position)] = donor_vectors[row]
        return (hidden, *args[1:])

    handle = model.layers[layer - 1].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _capture_module_vectors(
    model: torch.nn.Module,
    module: torch.nn.Module,
    items: Sequence[V16_3Rendered],
    positions: Sequence[int],
    vocab: V16_3Vocab,
    device: str,
) -> tuple[Any, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def hook(_module, _args, output):
        rows = torch.arange(len(positions), device=output.device)
        captured.append(output[rows, torch.tensor(positions, device=output.device)].detach().clone())

    handle = module.register_forward_hook(hook)
    try:
        output = _forward(model, items, vocab, device)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("module capture did not run exactly once")
    return output, captured[0]


def _capture_sublayer_states(
    model: torch.nn.Module,
    items: Sequence[V16_3Rendered],
    positions: Sequence[int],
    vocab: V16_3Vocab,
    device: str,
) -> tuple[Any, dict[tuple[int, str], torch.Tensor]]:
    values: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    rows: torch.Tensor | None = None
    pos: torch.Tensor | None = None

    for layer_index, layer in enumerate(model.layers, start=1):
        def pre_hook(_module, args, layer_index=layer_index):
            nonlocal rows, pos
            hidden = args[0]
            rows = torch.arange(len(positions), device=hidden.device)
            pos = torch.tensor(positions, device=hidden.device)
            values[(layer_index, "pre")] = hidden[rows, pos].detach().clone()

        def attn_hook(_module, _args, output, layer_index=layer_index):
            assert rows is not None and pos is not None
            values[(layer_index, "attn_component")] = output[rows, pos].detach().clone()

        def mlp_hook(_module, _args, output, layer_index=layer_index):
            assert rows is not None and pos is not None
            values[(layer_index, "mlp_component")] = output[rows, pos].detach().clone()

        handles.extend(
            [
                layer.register_forward_pre_hook(pre_hook),
                layer.attention.output.register_forward_hook(attn_hook),
                layer.mlp[2].register_forward_hook(mlp_hook),
            ]
        )
    try:
        output = _forward(model, items, vocab, device)
    finally:
        for handle in handles:
            handle.remove()
    for layer in range(1, len(model.layers) + 1):
        values[(layer, "post_attn")] = values[(layer, "pre")] + values[(layer, "attn_component")]
        values[(layer, "post_mlp")] = values[(layer, "post_attn")] + values[(layer, "mlp_component")]
    return output, values


def _logit_lens_margin(model: torch.nn.Module, states: torch.Tensor, target: int, alternative: int) -> np.ndarray:
    with torch.inference_mode():
        normalized = model.final_norm(states)
        direction = model.token_embedding.weight[target] - model.token_embedding.weight[alternative]
        return (normalized @ direction).detach().float().cpu().numpy()


def _head_rankings(ctx: RunContext) -> tuple[dict[str, list[Head]], pd.DataFrame]:
    frame = pd.read_csv(ctx.run_dir / "tables" / "checkpoint_attention_summary.csv")
    frame = frame[(frame["step"] == ctx.cfg.train_steps) & (frame["diagnostic_split"] == "head_selection")]
    specifications = {
        "nonthinking_broad": ("nonthinking", "final_answer", "broad_score"),
        "thinking_targeted": ("thinking", "trace_index", "correct_prompt_needle_mass"),
        "thinking_readout": ("thinking", "final_answer", "trace_readout_mass"),
    }
    result: dict[str, list[Head]] = {}
    rows: list[dict[str, Any]] = []
    for role, (mode, query, metric) in specifications.items():
        subset = frame[(frame["mode"] == mode) & (frame["query_kind"] == query)].copy()
        if role == "nonthinking_broad":
            subset["broad_score"] = subset["prompt_needles_mass"] * subset["needle_entropy_normalized"]
        summary = subset.groupby(["layer", "head"], as_index=False)[metric].mean().sort_values(metric, ascending=False)
        result[role] = [(int(row.layer), int(row.head)) for row in summary.itertuples()]
        for rank, row in enumerate(summary.itertuples(), start=1):
            rows.append(
                {
                    "role": role,
                    "rank": rank,
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "selection_metric": metric,
                    "selection_score": float(getattr(row, metric)),
                    "selection_split": "head_selection",
                }
            )
    return result, pd.DataFrame(rows)


def _random_orders(seed: int, count: int) -> list[list[Head]]:
    heads = [(layer, head) for layer in range(1, 5) for head in range(4)]
    rng = random.Random(seed)
    orders = []
    for _ in range(count):
        order = list(heads)
        rng.shuffle(order)
        orders.append(order)
    return orders


def _site_positions(items: Sequence[V16_3Rendered], site: str) -> list[list[int]]:
    result: list[list[int]] = []
    for item in items:
        assert item.spans is not None
        if site == "final_answer":
            result.append([item.spans.ans_pos])
        elif site == "trace_index":
            result.append(list(item.spans.trace_index_positions))
        else:
            raise ValueError(site)
    return result


def _evaluate_mechanism(
    ctx: RunContext,
    mode: str,
    items: Sequence[V16_3Rendered],
    outcome: str,
    *,
    head_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    output = _forward(ctx.models[mode], items, ctx.vocab, ctx.device, head_mask=head_mask)
    if outcome == "final_count":
        metrics = _count_metrics(output.logits, items, ctx.vocab)
        return {
            "accuracy": float(metrics["accuracy"].mean()),
            "margin": float(metrics["gold_margin"].mean()),
            "expected_abs_error": float(np.abs(metrics["expected_count"] - metrics["count"]).mean()),
        }
    if outcome == "trace_marker":
        correct = []
        margins = []
        for row, item in enumerate(items):
            assert item.spans is not None
            for query, target_position in zip(item.spans.trace_index_positions, item.spans.trace_marker_positions, strict=True):
                target = item.input_ids[target_position]
                values = output.logits[row, query].float()
                correct.append(float(int(values.argmax()) == target))
                alternatives = torch.cat((values[:target], values[target + 1 :]))
                margins.append(float((values[target] - alternatives.max()).detach().cpu()))
        return {"accuracy": float(np.mean(correct)), "margin": float(np.mean(margins)), "expected_abs_error": math.nan}
    raise ValueError(outcome)


def run_head_ablation(ctx: RunContext, options: PortOptions, rankings: dict[str, list[Head]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = _balanced(ctx.heldout_examples, options.examples_per_count)
    top_ns = (0, 1, 2, 4, 8, 12, 16)
    mechanisms = {
        "nonthinking_broad": ("nonthinking", "final_answer", "final_count"),
        "thinking_targeted": ("thinking", "trace_index", "trace_marker"),
        "thinking_readout": ("thinking", "final_answer", "final_count"),
    }
    random_orders = _random_orders(options.seed, options.random_paths)
    global_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    for role, (mode, site, outcome) in mechanisms.items():
        items = [render_v16_3(example, ctx.vocab, mode) for example in examples]
        baseline = _evaluate_mechanism(ctx, mode, items, outcome)
        paths: list[tuple[str, int, list[Head]]] = [("ranked", 0, rankings[role])]
        paths.extend(("random", index + 1, order) for index, order in enumerate(random_orders))
        positions = _site_positions(items, site)
        for path_kind, path_id, order in paths:
            for top_n in top_ns:
                selected = order[:top_n]
                mask = torch.ones((4, 4), device=ctx.device)
                for layer, head in selected:
                    mask[layer - 1, head] = 0
                global_metrics = _evaluate_mechanism(ctx, mode, items, outcome, head_mask=mask)
                with _local_attention_edit(ctx.models[mode], selected, positions):
                    local_metrics = _evaluate_mechanism(ctx, mode, items, outcome)
                common = {
                    "role": role,
                    "mode": mode,
                    "site": site,
                    "outcome": outcome,
                    "path_kind": path_kind,
                    "path_id": path_id,
                    "top_n": top_n,
                    "heads": ";".join(f"L{layer}H{head}" for layer, head in selected),
                    "baseline_accuracy": baseline["accuracy"],
                    "baseline_margin": baseline["margin"],
                }
                global_rows.append(
                    {
                        **common,
                        **global_metrics,
                        "accuracy_drop": baseline["accuracy"] - global_metrics["accuracy"],
                        "margin_drop": baseline["margin"] - global_metrics["margin"],
                    }
                )
                local_rows.append(
                    {
                        **common,
                        **local_metrics,
                        "accuracy_drop": baseline["accuracy"] - local_metrics["accuracy"],
                        "margin_drop": baseline["margin"] - local_metrics["margin"],
                    }
                )
    return pd.DataFrame(global_rows), pd.DataFrame(local_rows)


def _retrieval_corruption(example: V16_3Example, vocab: V16_3Vocab, k: int) -> tuple[V16_3Rendered, V16_3Rendered, int, int]:
    clean = render_v16_3(example, vocab, "thinking")
    assert clean.spans is not None and example.needle_characters is not None
    original = example.needle_markers[k - 1]
    alternatives = [
        f"<CH_{ord(char):04X}>" for char in example.needle_characters if f"<CH_{ord(char):04X}>" != original
    ]
    if not alternatives:
        raise RuntimeError("retrieval corruption requires another target-set character")
    replacement = alternatives[0]
    corrupt = _replace_tokens(
        clean,
        {
            clean.prompt_needle_positions[k - 1]: replacement,
            clean.spans.trace_marker_positions[k - 1]: replacement,
        },
        vocab,
    )
    return clean, corrupt, vocab.token_to_id[original], vocab.token_to_id[replacement]


def _marker_margin(logits: torch.Tensor, items: Sequence[V16_3Rendered], ks: Sequence[int], target_ids: Sequence[int], alternative_ids: Sequence[int]) -> np.ndarray:
    values = []
    for row, (item, k, target, alternative) in enumerate(zip(items, ks, target_ids, alternative_ids, strict=True)):
        assert item.spans is not None
        query = item.spans.trace_index_positions[k - 1]
        values.append(float((logits[row, query, target] - logits[row, query, alternative]).detach().cpu()))
    return np.asarray(values)


def run_retrieval_patching(ctx: RunContext, options: PortOptions, ranking: list[Head]) -> pd.DataFrame:
    eligible = [example for example in ctx.heldout_examples if int(example.count or 0) >= 3]
    selected = eligible[: options.retrieval_reporting_examples]
    pairs = [_retrieval_corruption(example, ctx.vocab, max(2, int(example.count or 0) // 2)) for example in selected]
    clean_items = [item[0] for item in pairs]
    corrupt_items = [item[1] for item in pairs]
    ks = [max(2, int(example.count or 0) // 2) for example in selected]
    target_ids = [item[2] for item in pairs]
    alternative_ids = [item[3] for item in pairs]
    clean_output, clean_inputs = _capture_attention_inputs(ctx.models["thinking"], clean_items, ctx.vocab, ctx.device)
    corrupt_output = _forward(ctx.models["thinking"], corrupt_items, ctx.vocab, ctx.device)
    clean_margin = _marker_margin(clean_output.logits, clean_items, ks, target_ids, alternative_ids)
    corrupt_margin = _marker_margin(corrupt_output.logits, corrupt_items, ks, target_ids, alternative_ids)
    donor_vectors = _attention_vectors(clean_inputs, [item.spans.trace_index_positions[k - 1] for item, k in zip(clean_items, ks, strict=True)])
    receiver_positions = [[item.spans.trace_index_positions[k - 1]] for item, k in zip(corrupt_items, ks, strict=True)]
    paths = [("ranked", 0, ranking)] + [
        ("random", index + 1, order) for index, order in enumerate(_random_orders(options.seed + 1, options.random_paths))
    ]
    rows: list[dict[str, Any]] = []
    for path_kind, path_id, order in paths:
        for top_n in (0, 1, 2, 4, 8, 12, 16):
            with _local_attention_edit(ctx.models["thinking"], order[:top_n], receiver_positions, donor_vectors):
                output = _forward(ctx.models["thinking"], corrupt_items, ctx.vocab, ctx.device)
            patched_margin = _marker_margin(output.logits, corrupt_items, ks, target_ids, alternative_ids)
            recovery = _normalized_recovery(clean_margin, corrupt_margin, patched_margin)
            for index, example in enumerate(selected):
                rows.append(
                    {
                        "path_kind": path_kind,
                        "path_id": path_id,
                        "top_n": top_n,
                        "count": int(example.count or 0),
                        "count_band": _count_band(int(example.count or 0)),
                        "k": ks[index],
                        "clean_margin": clean_margin[index],
                        "corrupt_margin": corrupt_margin[index],
                        "patched_margin": patched_margin[index],
                        "normalized_recovery": recovery[index],
                        "clean_correct": float(clean_margin[index] > 0),
                        "corrupt_correct": float(corrupt_margin[index] > 0),
                        "patched_correct": float(patched_margin[index] > 0),
                    }
                )
    return pd.DataFrame(rows)


def _successor_pair(example: V16_3Example, vocab: V16_3Vocab, k: int) -> tuple[V16_3Rendered, V16_3Rendered]:
    clean_full = render_v16_3(example, vocab, "thinking")
    assert clean_full.spans is not None
    query = clean_full.spans.trace_marker_positions[k - 1]
    clean = _truncate(clean_full, query + 1)
    replacements = {
        clean_full.prompt_needle_positions[index]: _non_target_token(example, vocab)
        for index in range(k, int(example.count or 0))
    }
    short = _truncate(_replace_tokens(clean_full, replacements, vocab), query + 1)
    return clean, short


def _two_token_margin(logits: torch.Tensor, positions: Sequence[int], target: int, alternative: int) -> np.ndarray:
    return np.asarray(
        [float((logits[row, int(position), target] - logits[row, int(position), alternative]).detach().cpu()) for row, position in enumerate(positions)]
    )


def run_successor_patching(
    ctx: RunContext, options: PortOptions
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = [example for example in ctx.heldout_examples if int(example.count or 0) >= 4]
    selection_examples = eligible[: options.retrieval_selection_examples]
    reporting_examples = eligible[options.retrieval_selection_examples : options.retrieval_selection_examples + options.retrieval_reporting_examples]
    next_ids = ctx.vocab.number_ids
    close_id = ctx.vocab.token_to_id["</Think>"]

    def prepare(examples: Sequence[V16_3Example]):
        ks = [max(2, min(int(example.count or 0) - 1, int(example.count or 0) // 2)) for example in examples]
        pairs = [_successor_pair(example, ctx.vocab, k) for example, k in zip(examples, ks, strict=True)]
        clean = [pair[0] for pair in pairs]
        short = [pair[1] for pair in pairs]
        positions = [item.spans.trace_marker_positions[k - 1] for item, k in zip(clean, ks, strict=True)]
        targets = [next_ids[k] for k in ks]  # k is zero-based index for token <k+1>
        return ks, clean, short, positions, targets

    def captures(examples: Sequence[V16_3Example]):
        ks, clean, short, positions, targets = prepare(examples)
        clean_output, clean_inputs = _capture_attention_inputs(ctx.models["thinking"], clean, ctx.vocab, ctx.device)
        short_output, short_inputs = _capture_attention_inputs(ctx.models["thinking"], short, ctx.vocab, ctx.device)
        clean_vectors = _attention_vectors(clean_inputs, positions)
        short_vectors = _attention_vectors(short_inputs, positions)
        continue_clean = np.asarray([
            float((clean_output.logits[row, pos, target] - clean_output.logits[row, pos, close_id]).detach().cpu())
            for row, (pos, target) in enumerate(zip(positions, targets, strict=True))
        ])
        continue_short = np.asarray([
            float((short_output.logits[row, pos, target] - short_output.logits[row, pos, close_id]).detach().cpu())
            for row, (pos, target) in enumerate(zip(positions, targets, strict=True))
        ])
        return ks, clean, short, positions, targets, clean_vectors, short_vectors, continue_clean, continue_short

    selection = captures(selection_examples)
    sel_ks, sel_clean, sel_short, sel_pos, sel_targets, sel_clean_vec, sel_short_vec, sel_clean_margin, sel_short_margin = selection
    head_scores: list[dict[str, Any]] = []
    receiver_positions = [[position] for position in sel_pos]
    for head in [(layer, head) for layer in range(1, 5) for head in range(4)]:
        with _local_attention_edit(ctx.models["thinking"], [head], receiver_positions, sel_clean_vec):
            output = _forward(ctx.models["thinking"], sel_short, ctx.vocab, ctx.device)
        patched = np.asarray([
            float((output.logits[row, pos, target] - output.logits[row, pos, close_id]).detach().cpu())
            for row, (pos, target) in enumerate(zip(sel_pos, sel_targets, strict=True))
        ])
        continue_recovery = np.nanmean(_normalized_recovery(sel_clean_margin, sel_short_margin, patched))
        with _local_attention_edit(ctx.models["thinking"], [head], receiver_positions, sel_short_vec):
            reverse = _forward(ctx.models["thinking"], sel_clean, ctx.vocab, ctx.device)
        reverse_continue = np.asarray([
            float((reverse.logits[row, pos, target] - reverse.logits[row, pos, close_id]).detach().cpu())
            for row, (pos, target) in enumerate(zip(sel_pos, sel_targets, strict=True))
        ])
        close_recovery = np.nanmean(_normalized_recovery(-sel_short_margin, -sel_clean_margin, -reverse_continue))
        head_scores.append(
            {
                "layer": head[0],
                "head": head[1],
                "continue_recovery_selection": float(continue_recovery),
                "close_recovery_selection": float(close_recovery),
                "bidirectional_score": float(np.nanmean([continue_recovery, close_recovery])),
            }
        )
    ranking_frame = pd.DataFrame(head_scores).sort_values("bidirectional_score", ascending=False).reset_index(drop=True)
    ranking_frame["rank"] = np.arange(1, len(ranking_frame) + 1)
    ranking = [(int(row.layer), int(row.head)) for row in ranking_frame.itertuples()]

    report = captures(reporting_examples)
    ks, clean, short, positions, targets, clean_vec, short_vec, clean_margin, short_margin = report
    receiver_positions = [[position] for position in positions]
    paths = [("ranked", 0, ranking)] + [
        ("random", index + 1, order) for index, order in enumerate(_random_orders(options.seed + 2, options.random_paths))
    ]
    rows: list[dict[str, Any]] = []
    for direction in ("continue_to_close", "close_to_continue"):
        donor = clean_vec if direction == "continue_to_close" else short_vec
        receiver = short if direction == "continue_to_close" else clean
        clean_reference = clean_margin if direction == "continue_to_close" else -short_margin
        corrupt_reference = short_margin if direction == "continue_to_close" else -clean_margin
        for path_kind, path_id, order in paths:
            for top_n in (0, 1, 2, 4, 8, 12, 16):
                with _local_attention_edit(ctx.models["thinking"], order[:top_n], receiver_positions, donor):
                    output = _forward(ctx.models["thinking"], receiver, ctx.vocab, ctx.device)
                continued = np.asarray([
                    float((output.logits[row, pos, target] - output.logits[row, pos, close_id]).detach().cpu())
                    for row, (pos, target) in enumerate(zip(positions, targets, strict=True))
                ])
                patched_target_margin = continued if direction == "continue_to_close" else -continued
                recovery = _normalized_recovery(clean_reference, corrupt_reference, patched_target_margin)
                for index, example in enumerate(reporting_examples):
                    rows.append(
                        {
                            "direction": direction,
                            "path_kind": path_kind,
                            "path_id": path_id,
                            "top_n": top_n,
                            "count": int(example.count or 0),
                            "count_band": _count_band(int(example.count or 0)),
                            "k": ks[index],
                            "clean_margin": clean_reference[index],
                            "corrupt_margin": corrupt_reference[index],
                            "patched_margin": patched_target_margin[index],
                            "normalized_recovery": recovery[index],
                            "decision_flipped": float(patched_target_margin[index] > 0),
                        }
                    )

    # v10-style residual logit lens and direct component evidence.
    _, clean_states = _capture_sublayer_states(ctx.models["thinking"], clean, positions, ctx.vocab, ctx.device)
    _, short_states = _capture_sublayer_states(ctx.models["thinking"], short, positions, ctx.vocab, ctx.device)
    lens_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for layer in range(1, 5):
        for stage in ("pre", "post_attn", "post_mlp"):
            for index, (target, example) in enumerate(zip(targets, reporting_examples, strict=True)):
                clean_value = _logit_lens_margin(ctx.models["thinking"], clean_states[(layer, stage)][index : index + 1], target, close_id)[0]
                short_value = _logit_lens_margin(ctx.models["thinking"], short_states[(layer, stage)][index : index + 1], target, close_id)[0]
                lens_rows.append(
                    {
                        "layer": layer,
                        "stage": stage,
                        "count": int(example.count or 0),
                        "k": ks[index],
                        "clean_continue_margin": clean_value,
                        "short_continue_margin": short_value,
                        "clean_minus_short": clean_value - short_value,
                    }
                )
        for component in ("attn_component", "mlp_component"):
            for index, (target, example) in enumerate(zip(targets, reporting_examples, strict=True)):
                direction = (
                    ctx.models["thinking"].token_embedding.weight[target]
                    - ctx.models["thinking"].token_embedding.weight[close_id]
                ).detach()
                clean_value = float((clean_states[(layer, component)][index] @ direction).detach().cpu())
                short_value = float((short_states[(layer, component)][index] @ direction).detach().cpu())
                component_rows.append(
                    {
                        "layer": layer,
                        "component": component,
                        "count": int(example.count or 0),
                        "k": ks[index],
                        "clean_direct_margin": clean_value,
                        "short_direct_margin": short_value,
                        "clean_minus_short": clean_value - short_value,
                    }
                )
    return pd.DataFrame(rows), ranking_frame, pd.DataFrame(lens_rows), pd.DataFrame(component_rows)


def _capture_hidden(
    ctx: RunContext,
    mode: str,
    items: Sequence[V16_3Rendered],
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    output = _forward(ctx.models[mode], items, ctx.vocab, ctx.device, output_hidden_states=True)
    assert output.hidden_states is not None
    return output, output.hidden_states


def run_geometry(ctx: RunContext, options: PortOptions) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]:
    """Fit mean-first PCA and train-split count centroids for interventions."""

    coordinate_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    for mode in ("nonthinking", "thinking"):
        part = ctx.run_dir / "analysis" / "checkpoint_dynamics" / "parts" / f"rope_{mode}_step_{ctx.cfg.train_steps:06d}" / "heldout_states.npz"
        archive = np.load(part)
        sites = ("final_answer",) if mode == "nonthinking" else ("final_answer", "trace_index", "trace_marker")
        for site in sites:
            for layer in range(0, 5):
                values = archive[f"{site}__{layer}__x"].astype(np.float64)
                labels = archive[f"{site}__{layer}__y"].astype(int)
                unique = np.unique(labels)
                means = np.stack([values[labels == label].mean(axis=0) for label in unique])
                centered = means - means.mean(axis=0, keepdims=True)
                _, singular, vt = np.linalg.svd(centered, full_matrices=False)
                coordinates = centered @ vt[:6].T
                variance = singular**2
                ratio = variance / max(variance.sum(), 1e-12)
                displacement = np.diff(means, axis=0)
                norms = np.linalg.norm(displacement, axis=1)
                adjacent_cosines = np.sum(displacement[:-1] * displacement[1:], axis=1) / np.maximum(norms[:-1] * norms[1:], 1e-12)
                chord = float(np.linalg.norm(means[-1] - means[0]))
                arc = float(norms.sum())
                geometry_rows.append(
                    {
                        "mode": mode,
                        "site": site,
                        "layer": layer,
                        "classes": len(unique),
                        "pc1_variance": ratio[0],
                        "pc1_to_pc2_variance": ratio[:2].sum(),
                        "pc1_to_pc3_variance": ratio[:3].sum(),
                        "pc1_to_pc6_variance": ratio[:6].sum(),
                        "effective_dimension": float(1.0 / np.maximum(np.sum(ratio**2), 1e-12)),
                        "mean_adjacent_distance": float(norms.mean()),
                        "mean_adjacent_displacement_cosine": float(np.mean(adjacent_cosines)) if len(adjacent_cosines) else math.nan,
                        "path_straightness_chord_over_arc": chord / max(arc, 1e-12),
                    }
                )
                for label, point in zip(unique, coordinates, strict=True):
                    row = {"mode": mode, "site": site, "layer": layer, "label": int(label)}
                    row.update({f"pc{axis + 1}": float(point[axis]) for axis in range(min(6, point.shape[0]))})
                    coordinate_rows.append(row)

    # Train-region final-answer centroids are the only directions used causally.
    train_examples = _balanced(ctx.train_examples, options.centroid_train_per_count)
    for mode in ("nonthinking", "thinking"):
        items = [render_v16_3(example, ctx.vocab, mode) for example in train_examples]
        _, hidden_states = _capture_hidden(ctx, mode, items)
        positions = torch.tensor([item.spans.ans_pos for item in items], device=ctx.device)
        row_index = torch.arange(len(items), device=ctx.device)
        labels = np.asarray([int(example.count or 0) for example in train_examples])
        for layer, hidden in enumerate(hidden_states):
            vectors = hidden[row_index, positions].detach().float().cpu().numpy()
            for count in range(1, 11):
                centroids[(mode, layer, count)] = vectors[labels == count].mean(axis=0)
    return pd.DataFrame(geometry_rows), pd.DataFrame(coordinate_rows), centroids


def _pair_examples(examples: Sequence[V16_3Example], per_count: int = 3) -> list[tuple[V16_3Example, V16_3Example]]:
    buckets: dict[int, list[V16_3Example]] = {count: [] for count in range(1, 11)}
    for example in examples:
        count = int(example.count or 0)
        if count in buckets and len(buckets[count]) < per_count:
            buckets[count].append(example)
    pairs: list[tuple[V16_3Example, V16_3Example]] = []
    for receiver_count in range(2, 10):
        for offset in (-1, 1):
            donor_count = receiver_count + offset
            for index in range(min(len(buckets[receiver_count]), len(buckets[donor_count]))):
                pairs.append((buckets[receiver_count][index], buckets[donor_count][index]))
    return pairs


def run_residual_transport(
    ctx: RunContext,
    centroids: dict[tuple[str, int, int], np.ndarray],
) -> pd.DataFrame:
    pairs = _pair_examples(ctx.heldout_examples)
    rows: list[dict[str, Any]] = []
    for mode in ("nonthinking", "thinking"):
        receiver_items = [render_v16_3(receiver, ctx.vocab, mode) for receiver, _ in pairs]
        donor_items = [render_v16_3(donor, ctx.vocab, mode) for _, donor in pairs]
        baseline_output, receiver_hidden = _capture_hidden(ctx, mode, receiver_items)
        _, donor_hidden = _capture_hidden(ctx, mode, donor_items)
        baseline = _count_metrics(baseline_output.logits, receiver_items, ctx.vocab)
        receiver_positions = [item.spans.ans_pos for item in receiver_items]
        donor_positions = [item.spans.ans_pos for item in donor_items]
        for layer in range(1, 5):
            donor_vectors = torch.stack(
                [donor_hidden[layer][row, position] for row, position in enumerate(donor_positions)]
            )
            interventions: dict[str, torch.Tensor] = {"natural_donor": donor_vectors}
            for alpha in (0.5, 1.0):
                vectors = []
                for row, (receiver, donor) in enumerate(pairs):
                    base = receiver_hidden[layer][row, receiver_positions[row]].detach().float().cpu().numpy()
                    delta = centroids[(mode, layer, int(donor.count or 0))] - centroids[(mode, layer, int(receiver.count or 0))]
                    vectors.append(torch.tensor(base + alpha * delta, device=ctx.device, dtype=receiver_hidden[layer].dtype))
                interventions[f"centroid_delta_alpha_{alpha:g}"] = torch.stack(vectors)
            for intervention, vectors in interventions.items():
                with _residual_patch(ctx.models[mode], layer, receiver_positions, vectors):
                    output = _forward(ctx.models[mode], receiver_items, ctx.vocab, ctx.device)
                metrics = _count_metrics(output.logits, receiver_items, ctx.vocab)
                for index, (receiver, donor) in enumerate(pairs):
                    rows.append(
                        {
                            "mode": mode,
                            "layer": layer,
                            "intervention": intervention,
                            "receiver_count": int(receiver.count or 0),
                            "donor_count": int(donor.count or 0),
                            "offset": int(donor.count or 0) - int(receiver.count or 0),
                            "baseline_expected_count": baseline.iloc[index]["expected_count"],
                            "patched_expected_count": metrics.iloc[index]["expected_count"],
                            "expected_count_shift": metrics.iloc[index]["expected_count"] - baseline.iloc[index]["expected_count"],
                            "patched_accuracy": metrics.iloc[index]["accuracy"],
                        }
                    )
    return pd.DataFrame(rows)


def run_trace_state_patching(ctx: RunContext) -> pd.DataFrame:
    """Patch a donor final-marker state into a receiver interior marker at the same k."""

    buckets: dict[int, list[V16_3Example]] = {count: [] for count in range(1, 11)}
    for example in ctx.heldout_examples:
        count = int(example.count or 0)
        if count in buckets and len(buckets[count]) < 3:
            buckets[count].append(example)
    pairs: list[tuple[V16_3Example, V16_3Example, int]] = []
    for k in range(2, 9):
        receiver_count = min(10, k + 2)
        for donor, receiver in zip(buckets[k], buckets[receiver_count], strict=True):
            pairs.append((donor, receiver, k))
    donor_items = [render_v16_3(donor, ctx.vocab, "thinking") for donor, _, _ in pairs]
    receiver_items = [render_v16_3(receiver, ctx.vocab, "thinking") for _, receiver, _ in pairs]
    baseline_output, receiver_hidden = _capture_hidden(ctx, "thinking", receiver_items)
    _, donor_hidden = _capture_hidden(ctx, "thinking", donor_items)
    positions = [item.spans.trace_marker_positions[k - 1] for item, (_, _, k) in zip(receiver_items, pairs, strict=True)]
    donor_positions = [item.spans.trace_marker_positions[k - 1] for item, (_, _, k) in zip(donor_items, pairs, strict=True)]
    close_id = ctx.vocab.token_to_id["</Think>"]
    next_ids = [ctx.vocab.token_to_id[ctx.vocab.number_token(k + 1)] for _, _, k in pairs]
    baseline_close = np.asarray([
        float((baseline_output.logits[row, pos, close_id] - baseline_output.logits[row, pos, next_id]).detach().cpu())
        for row, (pos, next_id) in enumerate(zip(positions, next_ids, strict=True))
    ])
    rows: list[dict[str, Any]] = []
    for layer in range(1, 5):
        vectors = torch.stack([donor_hidden[layer][row, pos] for row, pos in enumerate(donor_positions)])
        with _residual_patch(ctx.models["thinking"], layer, positions, vectors):
            output = _forward(ctx.models["thinking"], receiver_items, ctx.vocab, ctx.device)
        patched_close = np.asarray([
            float((output.logits[row, pos, close_id] - output.logits[row, pos, next_id]).detach().cpu())
            for row, (pos, next_id) in enumerate(zip(positions, next_ids, strict=True))
        ])
        for index, (donor, receiver, k) in enumerate(pairs):
            rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "donor_total": int(donor.count or 0),
                    "receiver_total": int(receiver.count or 0),
                    "position_matched": True,
                    "baseline_close_margin": baseline_close[index],
                    "patched_close_margin": patched_close[index],
                    "close_margin_shift": patched_close[index] - baseline_close[index],
                    "patched_close_decision": float(patched_close[index] > 0),
                }
            )
    return pd.DataFrame(rows)


def _conflicts(example: V16_3Example, vocab: V16_3Vocab) -> tuple[V16_3Rendered, dict[str, V16_3Rendered]]:
    clean = render_v16_3(example, vocab, "thinking")
    assert clean.spans is not None and example.count is not None and example.count >= 2
    n = int(example.count)
    final_index = clean.spans.trace_index_positions[-1]
    final_marker = clean.spans.trace_marker_positions[-1]
    previous_marker_token = clean.tokens[clean.spans.trace_marker_positions[-2]]
    non_target = _non_target_token(example, vocab)
    return clean, {
        "prompt_minus_one_trace_clean": _replace_tokens(clean, {clean.prompt_needle_positions[-1]: non_target}, vocab),
        "trace_index_minus_one": _replace_tokens(clean, {final_index: vocab.number_token(n - 1)}, vocab),
        "trace_pair_copy_previous": _replace_tokens(
            clean,
            {final_index: vocab.number_token(n - 1), final_marker: previous_marker_token},
            vocab,
        ),
        "marker_identity_control": _replace_tokens(clean, {final_marker: non_target}, vocab),
        "trace_tail_neutral_control": _replace_tokens(
            clean, {final_index: "<Sep>", final_marker: "<Sep>"}, vocab
        ),
        "shortened_trace_position_shifted": render_v16_3_shortened_trace(example, vocab),
    }


def run_trace_conflicts_and_bridge(ctx: RunContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = [example for example in ctx.heldout_examples if int(example.count or 0) >= 2][:36]
    pairs = [_conflicts(example, ctx.vocab) for example in examples]
    clean_items = [pair[0] for pair in pairs]
    clean_output = _forward(ctx.models["thinking"], clean_items, ctx.vocab, ctx.device)
    alternatives = [int(example.count or 0) - 1 for example in examples]
    clean_margin = _paired_count_margin(clean_output.logits, clean_items, ctx.vocab, alternatives)
    behavior_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    variants = list(pairs[0][1])
    for variant in variants:
        corrupt_items = [pair[1][variant] for pair in pairs]
        corrupt_output = _forward(ctx.models["thinking"], corrupt_items, ctx.vocab, ctx.device)
        corrupt_margin = _paired_count_margin(corrupt_output.logits, corrupt_items, ctx.vocab, alternatives)
        corrupt_metrics = _count_metrics(corrupt_output.logits, corrupt_items, ctx.vocab)
        for index, example in enumerate(examples):
            behavior_rows.append(
                {
                    "intervention": variant,
                    "count": int(example.count or 0),
                    "count_band": _count_band(int(example.count or 0)),
                    "clean_margin_n_vs_n_minus_1": clean_margin[index],
                    "intervention_margin_n_vs_n_minus_1": corrupt_margin[index],
                    "margin_change": corrupt_margin[index] - clean_margin[index],
                    "predicted_count": corrupt_metrics.iloc[index]["predicted_count"],
                    "expected_count": corrupt_metrics.iloc[index]["expected_count"],
                    "follows_original_n": float(corrupt_metrics.iloc[index]["predicted_count"] == int(example.count or 0)),
                    "follows_n_minus_1": float(corrupt_metrics.iloc[index]["predicted_count"] == int(example.count or 0) - 1),
                    "length_preserved": variant != "shortened_trace_position_shifted",
                }
            )
        if variant != "shortened_trace_position_shifted":
            continue
        clean_positions = [item.spans.ans_pos for item in clean_items]
        corrupt_positions = [item.spans.ans_pos for item in corrupt_items]
        for layer in range(1, 5):
            modules = {
                "attention_output": ctx.models["thinking"].layers[layer - 1].attention.output,
                "mlp_output": ctx.models["thinking"].layers[layer - 1].mlp[2],
            }
            for component, module in modules.items():
                _, clean_vectors = _capture_module_vectors(
                    ctx.models["thinking"], module, clean_items, clean_positions, ctx.vocab, ctx.device
                )
                with _position_output_patch(module, corrupt_positions, clean_vectors):
                    patched_output = _forward(ctx.models["thinking"], corrupt_items, ctx.vocab, ctx.device)
                patched_margin = _paired_count_margin(patched_output.logits, corrupt_items, ctx.vocab, alternatives)
                recovery = _normalized_recovery(clean_margin, corrupt_margin, patched_margin)
                for index, example in enumerate(examples):
                    bridge_rows.append(
                        {
                            "layer": layer,
                            "component": component,
                            "count": int(example.count or 0),
                            "clean_margin": clean_margin[index],
                            "corrupt_margin": corrupt_margin[index],
                            "patched_margin": patched_margin[index],
                            "normalized_recovery": recovery[index],
                        }
                    )
            clean_hidden_output, clean_hidden = _capture_hidden(ctx, "thinking", clean_items)
            del clean_hidden_output
            vectors = torch.stack([clean_hidden[layer][row, pos] for row, pos in enumerate(clean_positions)])
            with _residual_patch(ctx.models["thinking"], layer, corrupt_positions, vectors):
                patched_output = _forward(ctx.models["thinking"], corrupt_items, ctx.vocab, ctx.device)
            patched_margin = _paired_count_margin(patched_output.logits, corrupt_items, ctx.vocab, alternatives)
            recovery = _normalized_recovery(clean_margin, corrupt_margin, patched_margin)
            for index, example in enumerate(examples):
                bridge_rows.append(
                    {
                        "layer": layer,
                        "component": "post_layer_residual",
                        "count": int(example.count or 0),
                        "clean_margin": clean_margin[index],
                        "corrupt_margin": corrupt_margin[index],
                        "patched_margin": patched_margin[index],
                        "normalized_recovery": recovery[index],
                    }
                )
    return pd.DataFrame(behavior_rows), pd.DataFrame(bridge_rows)


def run_state_to_head_routing(ctx: RunContext, targeted_heads: Sequence[Head]) -> pd.DataFrame:
    """Patch trace-progress residuals before Layer 3 and measure routing redirection."""

    examples = [example for example in ctx.heldout_examples if int(example.count or 0) == 10][:15]
    items = [render_v16_3(example, ctx.vocab, "thinking") for example in examples]
    output = _forward(
        ctx.models["thinking"], items, ctx.vocab, ctx.device, output_hidden_states=True, output_attentions=True
    )
    assert output.hidden_states is not None and output.attentions is not None
    layer = 3
    donor_vectors = []
    receiver_positions = []
    donor_prompt_positions = []
    receiver_prompt_positions = []
    ks = []
    js = []
    for row, item in enumerate(items):
        assert item.spans is not None
        k, j = 3, 8
        receiver_positions.append(item.spans.trace_index_positions[k - 1])
        donor_vectors.append(output.hidden_states[layer - 1][row, item.spans.trace_index_positions[j - 1]])
        donor_prompt_positions.append(item.prompt_needle_positions[j - 1])
        receiver_prompt_positions.append(item.prompt_needle_positions[k - 1])
        ks.append(k)
        js.append(j)
    donor_tensor = torch.stack(donor_vectors)
    with _prelayer_residual_patch(ctx.models["thinking"], layer, receiver_positions, donor_tensor):
        patched = _forward(
            ctx.models["thinking"], items, ctx.vocab, ctx.device, output_attentions=True
        )
    assert patched.attentions is not None
    rows: list[dict[str, Any]] = []
    for head_layer, head in targeted_heads[:4]:
        if head_layer != layer:
            continue
        for row, item in enumerate(items):
            base = output.attentions[layer - 1][row, head, receiver_positions[row]]
            changed = patched.attentions[layer - 1][row, head, receiver_positions[row]]
            base_shift = float((base[donor_prompt_positions[row]] - base[receiver_prompt_positions[row]]).detach().cpu())
            patched_shift = float((changed[donor_prompt_positions[row]] - changed[receiver_prompt_positions[row]]).detach().cpu())
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "receiver_progress": ks[row],
                    "donor_progress": js[row],
                    "baseline_donor_minus_receiver_mass": base_shift,
                    "patched_donor_minus_receiver_mass": patched_shift,
                    "routing_shift": patched_shift - base_shift,
                    "total_count_fixed": 10,
                }
            )
    return pd.DataFrame(rows)


def run_final_head_transport(
    ctx: RunContext,
    options: PortOptions,
    rankings: dict[str, list[Head]],
) -> pd.DataFrame:
    """Transport donor count information through final-query head slices."""

    pairs = _pair_examples(ctx.heldout_examples, per_count=3)
    rows: list[dict[str, Any]] = []
    random_orders = _random_orders(options.seed + 3, options.random_paths)
    for mode, role in (("nonthinking", "nonthinking_broad"), ("thinking", "thinking_readout")):
        receiver_items = [render_v16_3(receiver, ctx.vocab, mode) for receiver, _ in pairs]
        donor_items = [render_v16_3(donor, ctx.vocab, mode) for _, donor in pairs]
        receiver_output = _forward(ctx.models[mode], receiver_items, ctx.vocab, ctx.device)
        baseline = _count_metrics(receiver_output.logits, receiver_items, ctx.vocab)
        _, donor_inputs = _capture_attention_inputs(ctx.models[mode], donor_items, ctx.vocab, ctx.device)
        donor_positions = [item.spans.ans_pos for item in donor_items]
        donor_vectors = _attention_vectors(donor_inputs, donor_positions)
        receiver_positions = [[item.spans.ans_pos] for item in receiver_items]
        paths = [("ranked", 0, rankings[role])] + [
            ("random", index + 1, order) for index, order in enumerate(random_orders)
        ]
        for path_kind, path_id, order in paths:
            for top_n in (0, 1, 2, 4, 8, 12, 16):
                with _local_attention_edit(
                    ctx.models[mode], order[:top_n], receiver_positions, donor_vectors
                ):
                    output = _forward(ctx.models[mode], receiver_items, ctx.vocab, ctx.device)
                metrics = _count_metrics(output.logits, receiver_items, ctx.vocab)
                for index, (receiver, donor) in enumerate(pairs):
                    rows.append(
                        {
                            "mode": mode,
                            "role": role,
                            "path_kind": path_kind,
                            "path_id": path_id,
                            "top_n": top_n,
                            "receiver_count": int(receiver.count or 0),
                            "donor_count": int(donor.count or 0),
                            "offset": int(donor.count or 0) - int(receiver.count or 0),
                            "baseline_expected_count": baseline.iloc[index]["expected_count"],
                            "patched_expected_count": metrics.iloc[index]["expected_count"],
                            "expected_count_shift": metrics.iloc[index]["expected_count"]
                            - baseline.iloc[index]["expected_count"],
                            "patched_accuracy": metrics.iloc[index]["accuracy"],
                        }
                    )
    return pd.DataFrame(rows)


def _successor_dataset(
    ctx: RunContext,
    examples: Sequence[V16_3Example],
) -> tuple[list[int], list[V16_3Rendered], list[V16_3Rendered], list[int], list[int]]:
    ks = [max(2, min(int(example.count or 0) - 1, int(example.count or 0) // 2)) for example in examples]
    pairs = [_successor_pair(example, ctx.vocab, k) for example, k in zip(examples, ks, strict=True)]
    clean = [pair[0] for pair in pairs]
    short = [pair[1] for pair in pairs]
    positions = [item.spans.trace_marker_positions[k - 1] for item, k in zip(clean, ks, strict=True)]
    targets = [ctx.vocab.number_ids[k] for k in ks]
    return ks, clean, short, positions, targets


def run_successor_mlp_features(
    ctx: RunContext,
    options: PortOptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether Layer-3/4 post-GELU features convert continue/close evidence."""

    eligible = [example for example in ctx.heldout_examples if int(example.count or 0) >= 4]
    selection_examples = eligible[: options.retrieval_selection_examples]
    reporting_examples = eligible[
        options.retrieval_selection_examples : options.retrieval_selection_examples
        + options.retrieval_reporting_examples
    ]
    sel_ks, sel_clean, sel_short, sel_pos, sel_targets = _successor_dataset(
        ctx, selection_examples
    )
    ks, clean, short, positions, targets = _successor_dataset(ctx, reporting_examples)
    close_id = ctx.vocab.token_to_id["</Think>"]
    clean_output = _forward(ctx.models["thinking"], clean, ctx.vocab, ctx.device)
    short_output = _forward(ctx.models["thinking"], short, ctx.vocab, ctx.device)
    clean_margin = np.asarray(
        [
            float(
                (
                    clean_output.logits[row, pos, target]
                    - clean_output.logits[row, pos, close_id]
                )
                .detach()
                .cpu()
            )
            for row, (pos, target) in enumerate(zip(positions, targets, strict=True))
        ]
    )
    short_margin = np.asarray(
        [
            float(
                (
                    short_output.logits[row, pos, target]
                    - short_output.logits[row, pos, close_id]
                )
                .detach()
                .cpu()
            )
            for row, (pos, target) in enumerate(zip(positions, targets, strict=True))
        ]
    )
    concentration_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    supports = (0, 1, 4, 16, 64, 256, 1024)
    rng = np.random.default_rng(options.seed + 4)
    for layer in (3, 4):
        module = ctx.models["thinking"].layers[layer - 1].mlp[1]
        _, sel_clean_features = _capture_module_vectors(
            ctx.models["thinking"], module, sel_clean, sel_pos, ctx.vocab, ctx.device
        )
        _, sel_short_features = _capture_module_vectors(
            ctx.models["thinking"], module, sel_short, sel_pos, ctx.vocab, ctx.device
        )
        coefficients = []
        weight = ctx.models["thinking"].layers[layer - 1].mlp[2].weight.detach()
        for target in sel_targets:
            direction = (
                ctx.models["thinking"].token_embedding.weight[target]
                - ctx.models["thinking"].token_embedding.weight[close_id]
            ).detach()
            coefficients.append(weight.T @ direction)
        coefficient = torch.stack(coefficients)
        evidence = (
            (sel_clean_features - sel_short_features) * coefficient
        ).detach().float().cpu().numpy()
        mean_evidence = evidence.mean(axis=0)
        ranking = np.argsort(-mean_evidence)
        absolute_ranking = np.argsort(-np.abs(mean_evidence))
        positive_total = float(np.maximum(mean_evidence, 0).sum())
        absolute_total = float(np.abs(mean_evidence).sum())
        for support in supports[1:]:
            concentration_rows.append(
                {
                    "layer": layer,
                    "support": support,
                    "positive_evidence_fraction": float(
                        np.maximum(mean_evidence[ranking[:support]], 0).sum()
                        / max(positive_total, 1e-12)
                    ),
                    "absolute_evidence_fraction": float(
                        np.abs(mean_evidence[absolute_ranking[:support]]).sum()
                        / max(absolute_total, 1e-12)
                    ),
                    "selection_examples": len(selection_examples),
                }
            )

        _, clean_features = _capture_module_vectors(
            ctx.models["thinking"], module, clean, positions, ctx.vocab, ctx.device
        )
        _, short_features = _capture_module_vectors(
            ctx.models["thinking"], module, short, positions, ctx.vocab, ctx.device
        )
        paths: list[tuple[str, int, np.ndarray]] = [("ranked", 0, ranking)]
        for path_id in range(1, options.random_paths + 1):
            paths.append(("random", path_id, rng.permutation(ctx.cfg.n_inner)))
        for direction_name in ("continue_to_close", "close_to_continue"):
            donor = clean_features if direction_name == "continue_to_close" else short_features
            receiver = short if direction_name == "continue_to_close" else clean
            clean_reference = clean_margin if direction_name == "continue_to_close" else -short_margin
            corrupt_reference = short_margin if direction_name == "continue_to_close" else -clean_margin
            for path_kind, path_id, order in paths:
                for support in supports:
                    indices = [int(value) for value in order[:support]]
                    with _position_output_patch(
                        module, positions, donor, feature_indices=indices
                    ):
                        output = _forward(
                            ctx.models["thinking"], receiver, ctx.vocab, ctx.device
                        )
                    continue_value = np.asarray(
                        [
                            float(
                                (
                                    output.logits[row, pos, target]
                                    - output.logits[row, pos, close_id]
                                )
                                .detach()
                                .cpu()
                            )
                            for row, (pos, target) in enumerate(
                                zip(positions, targets, strict=True)
                            )
                        ]
                    )
                    patched = (
                        continue_value
                        if direction_name == "continue_to_close"
                        else -continue_value
                    )
                    recovery = _normalized_recovery(
                        clean_reference, corrupt_reference, patched
                    )
                    for index, example in enumerate(reporting_examples):
                        patch_rows.append(
                            {
                                "layer": layer,
                                "direction": direction_name,
                                "path_kind": path_kind,
                                "path_id": path_id,
                                "support": support,
                                "count": int(example.count or 0),
                                "k": ks[index],
                                "normalized_recovery": recovery[index],
                                "decision_flipped": float(patched[index] > 0),
                            }
                        )
    return pd.DataFrame(concentration_rows), pd.DataFrame(patch_rows)


def run_head_to_state(
    ctx: RunContext,
    rankings: dict[str, list[Head]],
    centroids: dict[tuple[str, int, int], np.ndarray],
) -> pd.DataFrame:
    """Measure whether local head ablation damages later count-state geometry."""

    examples = _balanced(ctx.heldout_examples, 4)
    rows: list[dict[str, Any]] = []
    for mode, role in (("nonthinking", "nonthinking_broad"), ("thinking", "thinking_readout")):
        items = [render_v16_3(example, ctx.vocab, mode) for example in examples]
        positions = [item.spans.ans_pos for item in items]
        position_lists = [[position] for position in positions]
        for top_n in (0, 1, 2, 4, 8):
            with _local_attention_edit(
                ctx.models[mode], rankings[role][:top_n], position_lists
            ):
                output = _forward(
                    ctx.models[mode],
                    items,
                    ctx.vocab,
                    ctx.device,
                    output_hidden_states=True,
                )
            assert output.hidden_states is not None
            metrics = _count_metrics(output.logits, items, ctx.vocab)
            hidden = output.hidden_states[4]
            for index, example in enumerate(examples):
                count = int(example.count or 0)
                vector = hidden[index, positions[index]].detach().float().cpu().numpy()
                distances = {
                    candidate: float(
                        np.linalg.norm(vector - centroids[(mode, 4, candidate)])
                    )
                    for candidate in range(1, 11)
                }
                prediction = min(distances, key=distances.get)
                wrong_distance = min(
                    distance for candidate, distance in distances.items() if candidate != count
                )
                rows.append(
                    {
                        "mode": mode,
                        "role": role,
                        "top_n": top_n,
                        "count": count,
                        "state_prediction": prediction,
                        "state_accuracy": float(prediction == count),
                        "centroid_margin": wrong_distance - distances[count],
                        "output_accuracy": metrics.iloc[index]["accuracy"],
                        "output_margin": metrics.iloc[index]["gold_margin"],
                    }
                )
    return pd.DataFrame(rows)


def analysis_crosswalk() -> pd.DataFrame:
    rows = [
        ("v10 §4", "learning dynamics by count", "现有 21 个 checkpoint；1–10 exact-count 曲线与机制 milestone", "direct adaptation"),
        ("v10 §5", "broad vs targeted attention", "真实字符集合上的 broad coverage、correct-k mass、diagonal dominance", "marker semantics adapted"),
        ("v10 §6", "2D/3D residual manifolds", "mean-first centroid PCA、样本云、trace index/marker joint trajectory", "direct adaptation"),
        ("v10 §7", "global and position-local head ablation", "三种 role 均含 ranked 与 matched random paths", "count bins changed to 1-3/4-7/8-10"),
        ("v10 §8.2", "retrieval head patching", "集合内字符替换；prompt count 与 token position 均不变", "stronger character-aware corruption"),
        ("v10 §8.3-8.5", "successor/stop and MLP conversion", "未来 occurrence 删除形成同位置 continue/close pair；含 logit lens 与 component evidence", "task-native adaptation"),
        ("v10 §8.6-8.10", "count transport and final bridge", "length-preserving prompt/trace conflicts；attention/MLP/residual clean recovery", "fixed-15 replaced by fixed-length conflict"),
        ("v10 §9", "geometry steering", "独立 train centroids；±1 held-out receivers；alpha=0.5/1", "count range restricted to 1..10"),
        ("v10 §10", "hidden-state patching", "final-answer transport 与同-k final→interior early-close patch", "RoPE position-matched where possible"),
        ("v10 §11", "head↔state bidirectionality", "progress-state transplant before L3；attention routing shift", "same total count=10 control"),
    ]
    return pd.DataFrame(rows, columns=["v10_section", "v10_analysis", "v16_3_implementation", "adaptation_status"])


def run_v10_port_analysis(
    run_dir: str | Path,
    *,
    device: str | None = None,
    options: PortOptions | None = None,
) -> Path:
    options = options or PortOptions()
    ctx = load_context(run_dir, device)
    output_dir = ctx.run_dir / "analysis" / "v10_port"
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rankings, ranking_table = _head_rankings(ctx)
    geometry, coordinates, centroids = run_geometry(ctx, options)
    global_ablation, local_ablation = run_head_ablation(ctx, options, rankings)
    retrieval = run_retrieval_patching(ctx, options, rankings["thinking_targeted"])
    successor, successor_rankings, logit_lens, component_evidence = run_successor_patching(ctx, options)
    feature_concentration, feature_patching = run_successor_mlp_features(ctx, options)
    final_head_transport = run_final_head_transport(ctx, options, rankings)
    residual_transport = run_residual_transport(ctx, centroids)
    trace_state = run_trace_state_patching(ctx)
    conflicts, bridge = run_trace_conflicts_and_bridge(ctx)
    head_to_state = run_head_to_state(ctx, rankings, centroids)
    state_to_head = run_state_to_head_routing(ctx, rankings["thinking_targeted"])

    tables = {
        "analysis_crosswalk.csv": analysis_crosswalk(),
        "head_rankings.csv": ranking_table,
        "representation_geometry.csv": geometry,
        "representation_pca_coordinates.csv": coordinates,
        "global_head_ablation.csv": global_ablation,
        "position_local_head_ablation.csv": local_ablation,
        "retrieval_head_patching.csv": retrieval,
        "successor_head_rankings.csv": successor_rankings,
        "successor_head_patching.csv": successor,
        "successor_residual_logit_lens.csv": logit_lens,
        "successor_component_evidence.csv": component_evidence,
        "successor_mlp_feature_concentration.csv": feature_concentration,
        "successor_mlp_feature_patching.csv": feature_patching,
        "final_query_head_transport.csv": final_head_transport,
        "residual_count_transport.csv": residual_transport,
        "trace_early_stop_patching.csv": trace_state,
        "length_preserving_trace_conflicts.csv": conflicts,
        "final_bridge_component_patching.csv": bridge,
        "head_to_state_geometry.csv": head_to_state,
        "state_to_head_routing.csv": state_to_head,
    }
    for name, frame in tables.items():
        _atomic_csv(frame, table_dir / name)

    manifest = {
        "analysis": "v10_port_for_v16_3",
        "run_id": (ctx.run_dir / "source_run_id.txt").read_text(encoding="utf-8").strip(),
        "position_encoding": "rope",
        "checkpoint_step": ctx.cfg.train_steps,
        "device": ctx.device,
        "options": options.__dict__,
        "tables": [
            {
                "name": name,
                "rows": int(len(frame)),
                "sha256": _file_sha256(table_dir / name),
            }
            for name, frame in tables.items()
        ],
        "causal_identification_notes": [
            "head rankings use the disjoint head_selection split where available",
            "ablation and cumulative patching report deterministic random-path controls",
            "retrieval corruption swaps one target character for another target character, preserving total count and positions",
            "trace conflicts preserve sequence length and final-answer position",
            "train-region centroids are fit independently from held-out intervention receivers",
            "same-k early-stop patching keeps the trace query position matched under RoPE",
        ],
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return output_dir


__all__ = ["PortOptions", "analysis_crosswalk", "load_context", "run_v10_port_analysis"]
