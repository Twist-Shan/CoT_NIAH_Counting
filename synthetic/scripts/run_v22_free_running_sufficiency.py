#!/usr/bin/env python
"""Free-running causal-sufficiency assays for the v22 mode comparison.

Experiment A transplants a full answer-query residual from an adjacent-count
donor into a receiver.  Non-thinking starts directly at ``<Ans>``; Thinking
first generates its own trace up to ``<Ans>`` so the patched answer is evaluated
after a genuinely free-running trace.  Discovery chooses the layer, and the
same layer is frozen for held-out confirmation.

Experiment B intervenes on v22 Thinking progress while generation is paused
after trace item k.  A discovery-fitted running-count centroid delta moves the
marker state from k toward k-1 or k+1 at the same token position.  Generation
then resumes without further teacher forcing.  Equal-norm directions orthogonal
to the centroid span are controls.  Natural donor marker/item-span patches are
also reported as position-confounded upper bounds because the separator grammar
has fixed two-token items and cannot align different k values at one absolute
position.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_v22_modes_ncc import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    SPECS,
    ModeSpec,
    _load_bundle,
    _unique_run,
)
from synthetic_counting_v20.aligned_geometry import capture_mode_geometry  # noqa: E402
from synthetic_counting_v20.data import V20Example, render_v20  # noqa: E402
from synthetic_counting_v20.training import load_v20_checkpoint_model  # noqa: E402
from synthetic_counting_v20.v10_port_analysis import _residual_patch  # noqa: E402


@dataclass(frozen=True)
class AnswerPair:
    receiver: V20Example
    donor: V20Example
    same_count_control: V20Example


def _pad_sequences(sequences: Sequence[Sequence[int]], vocab, device: str):
    width = max(map(len, sequences))
    ids = torch.full(
        (len(sequences), width), vocab.pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = 1
    return ids, mask


@torch.no_grad()
def _forward_sequences(model, sequences, vocab, device: str, *, hidden: bool = False):
    ids, mask = _pad_sequences(sequences, vocab, device)
    return model(
        input_ids=ids,
        attention_mask=mask,
        output_hidden_states=hidden,
    )


def _answer_pairs(
    examples: Sequence[V20Example], *, count_min: int = 1, count_max: int = 30
) -> list[AnswerPair]:
    buckets: dict[int, list[V20Example]] = {}
    for example in examples:
        buckets.setdefault(int(example.count or 0), []).append(example)
    pairs = []
    for receiver_count in range(count_min + 1, count_max):
        receivers = buckets[receiver_count]
        if len(receivers) < 2:
            raise ValueError(f"count={receiver_count}: need two receiver/control rows")
        for offset in (-1, 1):
            donor_count = receiver_count + offset
            if not buckets.get(donor_count):
                raise ValueError(f"count={donor_count}: missing donor row")
            pairs.append(
                AnswerPair(
                    receiver=receivers[0],
                    donor=buckets[donor_count][0],
                    same_count_control=receivers[1],
                )
            )
    return pairs


@torch.no_grad()
def _generate_to_answer(model, cfg, vocab, examples, *, mode: str, device: str):
    """Return one free-running prefix per row, ending at the first <Ans>."""

    sequences = []
    if mode == "nonthinking":
        for example in examples:
            item = render_v20(example, vocab, mode)
            assert item.spans is not None
            sequences.append(vocab.encode(item.tokens[: item.spans.ans_pos + 1]))
        return sequences, [True] * len(sequences)

    for example in examples:
        item = render_v20(example, vocab, mode)
        assert item.spans is not None and item.spans.think_pos is not None
        sequences.append(vocab.encode(item.tokens[: item.spans.think_pos + 1]))
    active = [True] * len(sequences)
    reached = [False] * len(sequences)
    ans_id = vocab.token_to_id["<Ans>"]
    max_new_tokens = 2 * int(cfg.count_max_threshold) + 6
    for _ in range(max_new_tokens):
        if not any(active):
            break
        output = _forward_sequences(model, sequences, vocab, device)
        for row in range(len(sequences)):
            if not active[row]:
                continue
            next_id = int(output.logits[row, len(sequences[row]) - 1].argmax())
            sequences[row].append(next_id)
            if next_id == ans_id:
                reached[row] = True
                active[row] = False
            elif next_id == vocab.eos_id:
                active[row] = False
    return sequences, reached


def _count_from_id(vocab, token_id: int) -> int | None:
    token = vocab.id_to_token[int(token_id)]
    return vocab.decode_number_tokens([token]) if token in vocab.numbers else None


@torch.no_grad()
def _answer_transplant_rows(
    model,
    cfg,
    vocab,
    pairs: Sequence[AnswerPair],
    *,
    mode: str,
    split: str,
    device: str,
) -> pd.DataFrame:
    receiver_examples = [pair.receiver for pair in pairs]
    donor_examples = [pair.donor for pair in pairs]
    control_examples = [pair.same_count_control for pair in pairs]
    receiver_sequences, receiver_reached = _generate_to_answer(
        model, cfg, vocab, receiver_examples, mode=mode, device=device
    )
    donor_sequences, donor_reached = _generate_to_answer(
        model, cfg, vocab, donor_examples, mode=mode, device=device
    )
    control_sequences, control_reached = _generate_to_answer(
        model, cfg, vocab, control_examples, mode=mode, device=device
    )
    eligible = [
        index
        for index in range(len(pairs))
        if receiver_reached[index] and donor_reached[index] and control_reached[index]
    ]
    if not eligible:
        raise RuntimeError(f"{mode}/{split}: no pairs reached <Ans>")
    pairs = [pairs[index] for index in eligible]
    receiver_sequences = [receiver_sequences[index] for index in eligible]
    donor_sequences = [donor_sequences[index] for index in eligible]
    control_sequences = [control_sequences[index] for index in eligible]

    receiver_output = _forward_sequences(
        model, receiver_sequences, vocab, device, hidden=True
    )
    donor_output = _forward_sequences(model, donor_sequences, vocab, device, hidden=True)
    control_output = _forward_sequences(
        model, control_sequences, vocab, device, hidden=True
    )
    assert receiver_output.hidden_states is not None
    assert donor_output.hidden_states is not None
    assert control_output.hidden_states is not None
    receiver_positions = [len(sequence) - 1 for sequence in receiver_sequences]
    donor_positions = [len(sequence) - 1 for sequence in donor_sequences]
    control_positions = [len(sequence) - 1 for sequence in control_sequences]

    rows: list[dict[str, Any]] = []

    def record(layer: int, condition: str, output) -> None:
        for row, pair in enumerate(pairs):
            receiver_count = int(pair.receiver.count or 0)
            donor_count = int(pair.donor.count or 0)
            logits = output.logits[row, receiver_positions[row]].float()
            receiver_id = vocab.token_to_id[vocab.number_token(receiver_count)]
            donor_id = vocab.token_to_id[vocab.number_token(donor_count)]
            predicted_id = int(logits.argmax())
            baseline_logits = receiver_output.logits[
                row, receiver_positions[row]
            ].float()
            rows.append(
                {
                    "split": split,
                    "mode": mode,
                    "layer": int(layer),
                    "condition": condition,
                    "receiver_count": receiver_count,
                    "donor_count": donor_count,
                    "offset": donor_count - receiver_count,
                    "receiver_trace_reached_answer": 1.0,
                    "donor_trace_reached_answer": 1.0,
                    "receiver_generated_prefix_length": len(receiver_sequences[row]),
                    "donor_generated_prefix_length": len(donor_sequences[row]),
                    "baseline_prediction": _count_from_id(
                        vocab, int(baseline_logits.argmax())
                    ),
                    "patched_prediction": _count_from_id(vocab, predicted_id),
                    "patched_token": vocab.id_to_token[predicted_id],
                    "donor_adoption": float(predicted_id == donor_id),
                    "receiver_retention": float(predicted_id == receiver_id),
                    "donor_minus_receiver_margin": float(
                        (logits[donor_id] - logits[receiver_id]).cpu()
                    ),
                    "baseline_donor_minus_receiver_margin": float(
                        (baseline_logits[donor_id] - baseline_logits[receiver_id]).cpu()
                    ),
                    "donor_margin_shift": float(
                        (
                            logits[donor_id]
                            - logits[receiver_id]
                            - baseline_logits[donor_id]
                            + baseline_logits[receiver_id]
                        ).cpu()
                    ),
                }
            )

    record(0, "clean", receiver_output)
    for layer in range(1, cfg.n_layer + 1):
        donor_vectors = torch.stack(
            [
                donor_output.hidden_states[layer][row, position]
                for row, position in enumerate(donor_positions)
            ]
        )
        control_vectors = torch.stack(
            [
                control_output.hidden_states[layer][row, position]
                for row, position in enumerate(control_positions)
            ]
        )
        receiver_vectors = torch.stack(
            [
                receiver_output.hidden_states[layer][row, position]
                for row, position in enumerate(receiver_positions)
            ]
        )
        for condition, vectors in (
            ("self_patch", receiver_vectors),
            ("adjacent_count_donor", donor_vectors),
            ("same_count_context_control", control_vectors),
        ):
            with _residual_patch(model, layer, receiver_positions, vectors):
                output = _forward_sequences(model, receiver_sequences, vocab, device)
            record(layer, condition, output)
    return pd.DataFrame(rows)


def _select_answer_layer(discovery: pd.DataFrame) -> dict[str, int]:
    result = {}
    for mode, frame in discovery.groupby("mode"):
        candidate = frame[frame["condition"].eq("adjacent_count_donor")]
        summary = candidate.groupby("layer", as_index=False).agg(
            donor_margin_shift=("donor_margin_shift", "mean"),
            donor_adoption=("donor_adoption", "mean"),
        )
        winner = summary.sort_values(
            ["donor_margin_shift", "donor_adoption", "layer"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        result[str(mode)] = int(winner["layer"])
    return result


def _centroid_bank(dataset) -> dict[int, dict[int, np.ndarray]]:
    labels = dataset.metadata["occurrence"].to_numpy(dtype=int)
    result: dict[int, dict[int, np.ndarray]] = {}
    for layer, states in dataset.states_by_layer.items():
        result[layer] = {
            label: states[labels == label].mean(axis=0)
            for label in sorted(np.unique(labels))
        }
    return result


def _orthogonal_direction(
    centroids: dict[int, np.ndarray],
    target_norm: float,
    *,
    layer: int,
    k: int,
    shift: int,
) -> np.ndarray:
    matrix = np.stack([centroids[label] for label in sorted(centroids)])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0], 1.0) * 1e-8))
    basis = vh[:rank]
    digest = hashlib.sha256(f"{layer}:{k}:{shift}:orthogonal".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vector = rng.normal(size=matrix.shape[1])
    if len(basis):
        vector = vector - basis.T @ (basis @ vector)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise RuntimeError("failed to construct an orthogonal control direction")
    return vector * (target_norm / norm)


@torch.no_grad()
def _progress_patch_vectors(
    model,
    vocab,
    example: V20Example,
    *,
    layer: int,
    k: int,
    shift: int,
    centroids: dict[int, np.ndarray],
    device: str,
):
    item = render_v20(example, vocab, "thinking")
    assert item.spans is not None
    receiver_marker = int(item.spans.trace_marker_positions[k - 1])
    receiver_query = int(item.spans.trace_query_positions[k - 1])
    donor_k = k + shift
    donor_marker = int(item.spans.trace_marker_positions[donor_k - 1])
    donor_query = int(item.spans.trace_query_positions[donor_k - 1])
    prefix_ids = vocab.encode(item.tokens[: receiver_marker + 1])
    donor_prefix_ids = vocab.encode(item.tokens[: donor_marker + 1])
    receiver_output = _forward_sequences(
        model, [prefix_ids], vocab, device, hidden=True
    )
    donor_output = _forward_sequences(
        model, [donor_prefix_ids], vocab, device, hidden=True
    )
    assert receiver_output.hidden_states is not None
    assert donor_output.hidden_states is not None
    receiver_vector = receiver_output.hidden_states[layer][0, receiver_marker]
    delta = centroids[donor_k] - centroids[k]
    centroid_vector = receiver_vector + torch.tensor(
        delta, device=device, dtype=receiver_vector.dtype
    )
    orthogonal = _orthogonal_direction(
        centroids, float(np.linalg.norm(delta)), layer=layer, k=k, shift=shift
    )
    orthogonal_vector = receiver_vector + torch.tensor(
        orthogonal, device=device, dtype=receiver_vector.dtype
    )
    return {
        "item": item,
        "prefix_ids": prefix_ids,
        "receiver_marker": receiver_marker,
        "receiver_query": receiver_query,
        "self_marker": receiver_vector[None],
        "centroid_marker": centroid_vector[None],
        "orthogonal_marker": orthogonal_vector[None],
        "natural_marker": donor_output.hidden_states[layer][0, donor_marker][None],
        "natural_query": donor_output.hidden_states[layer][0, donor_query][None],
    }


def _progress_context(model, layer: int, payload, condition: str):
    stack = contextlib.ExitStack()
    if condition == "clean":
        return stack
    marker_key = {
        "self_patch": "self_marker",
        "centroid_shift": "centroid_marker",
        "orthogonal_control": "orthogonal_marker",
        "natural_marker_cross_position": "natural_marker",
        "natural_item_span_cross_position": "natural_marker",
    }[condition]
    stack.enter_context(
        _residual_patch(
            model,
            layer,
            [payload["receiver_marker"]],
            payload[marker_key],
        )
    )
    if condition == "natural_item_span_cross_position":
        stack.enter_context(
            _residual_patch(
                model,
                layer,
                [payload["receiver_query"]],
                payload["natural_query"],
            )
        )
    return stack


@torch.no_grad()
def _progress_one(
    model,
    cfg,
    vocab,
    example: V20Example,
    *,
    layer: int,
    k: int,
    shift: int,
    condition: str,
    centroids: dict[int, np.ndarray],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = _progress_patch_vectors(
        model,
        vocab,
        example,
        layer=layer,
        k=k,
        shift=shift,
        centroids=centroids,
        device=device,
    )
    item = payload["item"]
    natural_id = vocab.token_to_id[example.needle_markers[k]]
    donor_id = vocab.token_to_id[example.needle_markers[k + shift]]

    forced = [*payload["prefix_ids"], vocab.token_to_id["<Sep>"]]
    with _progress_context(model, layer, payload, condition):
        forced_output = _forward_sequences(model, [forced], vocab, device)
    forced_logits = forced_output.logits[0, -1].float()

    generated = list(payload["prefix_ids"])
    continuation: list[int] = []
    for _ in range(max_new_tokens):
        with _progress_context(model, layer, payload, condition):
            output = _forward_sequences(model, [generated], vocab, device)
        next_id = int(output.logits[0, -1].argmax())
        generated.append(next_id)
        continuation.append(next_id)
        if next_id == vocab.eos_id:
            break

    continuation_tokens = vocab.decode(continuation)
    markers = []
    cursor = 0
    while cursor + 1 < len(continuation_tokens):
        if continuation_tokens[cursor] != "<Sep>":
            break
        marker = continuation_tokens[cursor + 1]
        if marker not in vocab.character_tokens:
            break
        markers.append(marker)
        cursor += 2
    expected = list(example.needle_markers[k + shift :])
    match_length = 0
    for observed, target in zip(markers, expected, strict=False):
        if observed != target:
            break
        match_length += 1
    generated_count = None
    if "<Ans>" in continuation_tokens:
        ans = continuation_tokens.index("<Ans>")
        if ans + 1 < len(continuation_tokens):
            generated_count = _count_from_id(
                vocab, vocab.token_to_id.get(continuation_tokens[ans + 1], vocab.pad_id)
            )
    first_marker = markers[0] if markers else None
    return {
        "condition": condition,
        "k": int(k),
        "shift": int(shift),
        "donor_progress": int(k + shift),
        "forced_donor_minus_natural_margin": float(
            (forced_logits[donor_id] - forced_logits[natural_id]).cpu()
        ),
        "first_generated_marker": first_marker,
        "donor_first_marker": example.needle_markers[k + shift],
        "natural_first_marker": example.needle_markers[k],
        "donor_first_adoption": float(
            first_marker == example.needle_markers[k + shift]
        ),
        "natural_first_retention": float(first_marker == example.needle_markers[k]),
        "donor_continuation_match_length": int(match_length),
        "donor_first_three_exact": float(
            markers[: min(3, len(expected))] == expected[: min(3, len(expected))]
            and min(3, len(expected)) > 0
        ),
        "generated_final_count": generated_count,
        "generated_final_correct": float(generated_count == int(example.count or 0)),
        "generated_token_count": len(continuation),
        "continuation_tokens": " ".join(continuation_tokens),
    }


def _select_progress_layer(
    model,
    cfg,
    vocab,
    discovery_examples,
    centroid_banks,
    *,
    progress_ks: Sequence[int],
    device: str,
) -> tuple[int, pd.DataFrame]:
    rows = []
    for layer in range(1, cfg.n_layer):
        for example in discovery_examples:
            for k in progress_ks:
                for shift in (-1, 1):
                    if example.needle_markers[k + shift] == example.needle_markers[k]:
                        continue
                    clean = _progress_one(
                        model,
                        cfg,
                        vocab,
                        example,
                        layer=layer,
                        k=k,
                        shift=shift,
                        condition="clean",
                        centroids=centroid_banks[layer],
                        device=device,
                        max_new_tokens=2,
                    )
                    changed = _progress_one(
                        model,
                        cfg,
                        vocab,
                        example,
                        layer=layer,
                        k=k,
                        shift=shift,
                        condition="centroid_shift",
                        centroids=centroid_banks[layer],
                        device=device,
                        max_new_tokens=2,
                    )
                    rows.append(
                        {
                            "layer": layer,
                            "k": k,
                            "shift": shift,
                            "route_shift": (
                                changed["forced_donor_minus_natural_margin"]
                                - clean["forced_donor_minus_natural_margin"]
                            ),
                            "donor_first_adoption": changed["donor_first_adoption"],
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = frame.groupby("layer", as_index=False).agg(
        route_shift=("route_shift", "mean"),
        donor_first_adoption=("donor_first_adoption", "mean"),
    )
    winner = summary.sort_values(
        ["route_shift", "donor_first_adoption", "layer"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return int(winner["layer"]), frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work" / "v22_free_running_sufficiency",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-discovery-examples", type=int, default=10)
    parser.add_argument("--progress-confirmation-examples", type=int, default=8)
    parser.add_argument("--max-progress-new-tokens", type=int, default=28)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--paired-run-prefix", default=None)
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--progress-k", type=int, nargs="+", default=(4, 6, 8))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = SPECS
    if args.paired_run_prefix is not None:
        specs = (
            ModeSpec(
                "nonthinking",
                args.paired_run_prefix,
                "nonthinking",
                "separator",
            ),
            ModeSpec(
                "thinking",
                args.paired_run_prefix,
                "thinking",
                "separator",
            ),
        )

    answer_discovery = []
    answer_confirmation = []
    bundles = {}
    selected_answer_layers: dict[str, int] = {}
    if not args.skip_answer:
        for spec in specs:
            run_dir = _unique_run(args.results_root.resolve(), spec.run_prefix)
            cfg, vocab, train, _selection, reporting = _load_bundle(
                run_dir, device=args.device
            )
            if args.expected_version is not None and cfg.version != args.expected_version:
                raise ValueError(
                    f"{spec.label}: expected version={args.expected_version!r}, "
                    f"got {cfg.version!r}"
                )
            _, checkpoint_vocab, _, _, model = load_v20_checkpoint_model(
                run_dir,
                "rope",
                spec.mode,
                step=cfg.train_steps,
                device=args.device,
            )
            if checkpoint_vocab.fingerprint != vocab.fingerprint:
                raise ValueError(f"{spec.label}: checkpoint vocabulary mismatch")
            model.eval()
            print(f"[{spec.label}] free-running answer discovery", flush=True)
            discovery_rows = _answer_transplant_rows(
                model,
                cfg,
                vocab,
                _answer_pairs(
                    train,
                    count_min=int(cfg.count_min),
                    count_max=int(cfg.count_max_threshold),
                ),
                mode=spec.mode,
                split="discovery",
                device=args.device,
            )
            print(f"[{spec.label}] free-running answer confirmation", flush=True)
            confirmation_rows = _answer_transplant_rows(
                model,
                cfg,
                vocab,
                _answer_pairs(
                    reporting,
                    count_min=int(cfg.count_min),
                    count_max=int(cfg.count_max_threshold),
                ),
                mode=spec.mode,
                split="confirmation",
                device=args.device,
            )
            answer_discovery.append(discovery_rows)
            answer_confirmation.append(confirmation_rows)
            bundles[spec.label] = (cfg, vocab, train, reporting, model)

        discovery_frame = pd.concat(answer_discovery, ignore_index=True)
        confirmation_frame = pd.concat(answer_confirmation, ignore_index=True)
        selected_answer_layers = _select_answer_layer(discovery_frame)
        confirmation_frame["is_discovery_selected_layer"] = confirmation_frame.apply(
            lambda row: float(
                int(row["layer"]) == selected_answer_layers[str(row["mode"])]
                or str(row["condition"]) == "clean"
            ),
            axis=1,
        )
        discovery_frame.to_csv(
            args.output / "answer_transplant_discovery.csv", index=False
        )
        confirmation_frame.to_csv(
            args.output / "answer_transplant_confirmation.csv", index=False
        )
    else:
        selected_path = args.output / "selected_layers.json"
        if selected_path.exists():
            selected_answer_layers = json.loads(
                selected_path.read_text(encoding="utf-8")
            ).get("answer_transplant", {})

    if "thinking" not in bundles:
        spec = next(spec for spec in specs if spec.mode == "thinking")
        run_dir = _unique_run(args.results_root.resolve(), spec.run_prefix)
        cfg, vocab, train, _selection, reporting = _load_bundle(
            run_dir, device=args.device
        )
        if args.expected_version is not None and cfg.version != args.expected_version:
            raise ValueError(
                f"thinking: expected version={args.expected_version!r}, "
                f"got {cfg.version!r}"
            )
        _, checkpoint_vocab, _, _, thinking_model = load_v20_checkpoint_model(
            run_dir,
            "rope",
            "thinking",
            step=cfg.train_steps,
            device=args.device,
        )
        if checkpoint_vocab.fingerprint != vocab.fingerprint:
            raise ValueError("thinking checkpoint vocabulary mismatch")
        thinking_model.eval()
    else:
        cfg, vocab, train, reporting, thinking_model = bundles["thinking"]
    discovery_geometry = capture_mode_geometry(
        thinking_model,
        vocab,
        train,
        mode="thinking",
        split="discovery",
        per_label=10,
        device=args.device,
        batch_size=32,
    )["thinking_item_end"]
    centroid_banks = _centroid_bank(discovery_geometry)
    terminal_count = int(cfg.count_max_threshold)
    progress_ks = tuple(sorted(set(map(int, args.progress_k))))
    invalid_ks = [
        k for k in progress_ks if k - 1 < 0 or k + 1 >= terminal_count
    ]
    if invalid_ks:
        raise ValueError(
            f"progress k values must allow k-1 and k+1 within count={terminal_count}: "
            f"{invalid_ks}"
        )
    discovery_terminal = [
        example for example in train if int(example.count or 0) == terminal_count
    ][: args.progress_discovery_examples]
    confirmation_terminal = [
        example
        for example in reporting
        if int(example.count or 0) == terminal_count
    ][: args.progress_confirmation_examples]
    print("[thinking] selecting progress intervention layer", flush=True)
    selected_progress_layer, progress_discovery = _select_progress_layer(
        thinking_model,
        cfg,
        vocab,
        discovery_terminal,
        centroid_banks,
        progress_ks=progress_ks,
        device=args.device,
    )
    progress_discovery.to_csv(
        args.output / "progress_layer_discovery.csv", index=False
    )
    print(
        f"[thinking] frozen progress layer L{selected_progress_layer}", flush=True
    )
    progress_rows = []
    conditions = (
        "clean",
        "self_patch",
        "centroid_shift",
        "orthogonal_control",
        "natural_marker_cross_position",
        "natural_item_span_cross_position",
    )
    for example_index, example in enumerate(confirmation_terminal):
        for k in progress_ks:
            for shift in (-1, 1):
                if example.needle_markers[k + shift] == example.needle_markers[k]:
                    continue
                for condition in conditions:
                    row = _progress_one(
                        thinking_model,
                        cfg,
                        vocab,
                        example,
                        layer=selected_progress_layer,
                        k=k,
                        shift=shift,
                        condition=condition,
                        centroids=centroid_banks[selected_progress_layer],
                        device=args.device,
                        max_new_tokens=args.max_progress_new_tokens,
                    )
                    progress_rows.append(
                        {
                            "example_index": example_index,
                            "prompt_sha256": example.prompt_sha256,
                            "layer": selected_progress_layer,
                            **row,
                        }
                    )
        print(
            f"[thinking] progress confirmation "
            f"{example_index + 1}/{len(confirmation_terminal)}",
            flush=True,
        )
    pd.DataFrame(progress_rows).to_csv(
        args.output / "progress_rollout_confirmation.csv", index=False
    )
    (args.output / "selected_layers.json").write_text(
        json.dumps(
            {
                "answer_transplant": selected_answer_layers,
                "thinking_progress": selected_progress_layer,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "answer_transplant": (
                    "adjacent-count natural answer-query residual; Thinking trace is "
                    "free-running up to <Ans>; discovery-selected layer frozen"
                ),
                "progress_primary": (
                    "same-position discovery-centroid shift at trace marker; generation "
                    "resumes without teacher forcing"
                ),
                "progress_eligibility": (
                    "donor successor token must differ from natural successor token"
                ),
                "progress_control": (
                    "equal-norm direction orthogonal to discovery count-centroid span"
                ),
                "progress_upper_bounds": (
                    "natural donor marker/item span cross k positions; position-confounded "
                    "because fixed two-token separator items prevent absolute-position matching"
                ),
                "paired_run_prefix": args.paired_run_prefix,
                "expected_version": args.expected_version,
                "terminal_count": terminal_count,
                "progress_k": list(progress_ks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
