#!/usr/bin/env python3
"""Build the Native-thinking P0 targeted-retrieval attention atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "work" / "v5_native_p0_head_atlas_20260820"
DEFAULT_OUTPUT = ROOT / "reports" / "NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html"
DEFAULT_ASSETS = ROOT / "reports" / "v5_native_p0_head_atlas"
EXPECTED_AGGREGATION = "equal_seed_mean_of_within_seed_event_means"


MODEL_ORDER = ("Qwen3-8B", "Gemma4-E4B")
MODEL_K = {"Qwen3-8B": 128, "Gemma4-E4B": 8}
GRAMMAR_LABELS = {
    "adjacent_rank_after_city": "adjacent · city → rank",
    "adjacent_rank_before_city": "adjacent · rank → city",
    "same_unit_rank_after_city": "same unit · city → rank",
    "same_unit_rank_before_city": "same unit · rank → city",
    "structural_explicit_rank_before_city": "structural explicit rank",
    "structural_invariant_bullet": "invariant bullet",
    "structural_unmarked": "unmarked structural",
    "evidence_sequence_unranked": "unranked evidence sequence",
}
ROBUST_SEED_THRESHOLD = 10


def _scope_label(scope: str) -> str:
    if scope == "all":
        return "all trace grammars"
    return GRAMMAR_LABELS.get(scope, scope)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.removeprefix("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


VIRIDIS_STOPS = (
    (0.00, "#1b103d"),
    (0.24, "#3b528b"),
    (0.50, "#21918c"),
    (0.74, "#5ec962"),
    (1.00, "#fde725"),
)


def _color(value: float, maximum: float) -> str:
    ratio = 0.0 if maximum <= 0 else min(1.0, max(0.0, value / maximum))
    for (left_x, left_color), (right_x, right_color) in zip(
        VIRIDIS_STOPS, VIRIDIS_STOPS[1:]
    ):
        if ratio <= right_x:
            width = right_x - left_x
            fraction = 0.0 if width <= 0 else (ratio - left_x) / width
            left = _rgb(left_color)
            right = _rgb(right_color)
            mixed = tuple(
                round(a + fraction * (b - a)) for a, b in zip(left, right)
            )
            return "#" + "".join(f"{channel:02x}" for channel in mixed)
    return VIRIDIS_STOPS[-1][1]


def _head_map_svg(
    model: str,
    grammar: str,
    bundle: dict[str, Any],
    *,
    shared_vmax: float,
) -> str:
    rows = bundle["rows"]
    maximum_layer = max(int(row["layer"]) for row in rows)
    maximum_head = max(int(row["head"]) for row in rows)
    layers = maximum_layer + 1
    heads = maximum_head + 1
    cell_width = 14 if heads >= 16 else 28
    cell_height = 14 if layers <= 38 else 11
    left = 58
    top = 34
    right = 92
    bottom = 58
    plot_width = heads * cell_width
    plot_height = layers * cell_height
    width = left + plot_width + right
    height = top + plot_height + bottom
    score_by_head = {
        (int(row["layer"]), int(row["head"])): float(row["score"])
        for row in rows
    }
    rank_by_head = {
        (int(row["layer"]), int(row["head"])): int(row["rank"])
        for row in rows
    }
    selected_k = int(MODEL_K[model])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="head-map" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(model)} {_escape(grammar)} P0 targeted retrieval head map">',
        '<rect width="100%" height="100%" rx="14" fill="#101927"/>',
    ]
    for layer in range(layers):
        for head in range(heads):
            score = score_by_head.get((layer, head), 0.0)
            rank = rank_by_head.get((layer, head), layers * heads + 1)
            x = left + head * cell_width
            y = top + layer * cell_height
            selected = rank <= selected_k
            stroke = "#f8fafc" if selected else "#243348"
            stroke_width = 1.25 if selected else 0.45
            title = (
                f"{model} · {grammar} · L{layer}H{head} · "
                f"score={score:.6f} · rank={rank}"
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" fill="{_color(score, shared_vmax)}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"><title>{_escape(title)}</title></rect>'
            )
            if rank <= 8 and cell_width >= 14 and cell_height >= 11:
                font_size = 7 if cell_width < 20 else 9
                parts.append(
                    f'<text x="{x + cell_width / 2:.2f}" y="{y + cell_height * 0.72:.2f}" '
                    f'text-anchor="middle" font-size="{font_size}" font-weight="800" '
                    f'fill="#ffffff">{rank}</text>'
                )
    for head in range(heads):
        if head % (4 if heads >= 16 else 1) == 0 or head == heads - 1:
            x = left + (head + 0.5) * cell_width
            parts.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-size="10" fill="#a7b6cb">H{head}</text>'
            )
    for layer in range(layers):
        if layer % 5 == 0 or layer == layers - 1:
            y = top + (layer + 0.65) * cell_height
            parts.append(
                f'<text x="{left - 10}" y="{y:.2f}" text-anchor="end" '
                f'font-size="10" fill="#a7b6cb">L{layer}</text>'
            )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" text-anchor="middle" '
            'font-size="12" font-weight="700" fill="#d8e2ee">attention head</text>',
            f'<text transform="translate(14 {top + plot_height / 2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12" font-weight="700" fill="#d8e2ee">decoder layer</text>',
        ]
    )
    legend_x = left + plot_width + 30
    legend_y = top
    legend_height = min(250, plot_height)
    steps = 60
    for index in range(steps):
        fraction = index / max(1, steps - 1)
        y = legend_y + (1 - fraction) * legend_height
        parts.append(
            f'<rect x="{legend_x}" y="{y:.2f}" width="14" height="{legend_height / steps + 0.8:.2f}" '
            f'fill="{_color(fraction * shared_vmax, shared_vmax)}"/>'
        )
    for fraction in (0.0, 0.5, 1.0):
        y = legend_y + (1 - fraction) * legend_height + 4
        parts.append(
            f'<text x="{legend_x + 21}" y="{y:.2f}" font-size="9" fill="#b7c5d6">'
            f'{fraction * shared_vmax:.3f}</text>'
        )
    parts.append(
        f'<text transform="translate({legend_x + 65} {legend_y + legend_height / 2:.2f}) rotate(-90)" '
        'text-anchor="middle" font-size="10" fill="#d8e2ee">P0 targeted retrieval score</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _ordinal_head_svg(
    model: str,
    scope: str,
    bundle: dict[str, Any],
    *,
    shared_vmax: float,
) -> str:
    heads = sorted(bundle["selected_heads"], key=lambda row: int(row["rank"]))
    ordinals = list(range(2, 11))
    value_by_cell = {
        (int(row["target_ordinal"]), int(row["layer"]), int(row["head"])): row
        for row in bundle["ordinal_rows"]
    }
    cell_width = 14 if len(heads) > 16 else 58
    cell_height = 36
    left = 104
    top = 34
    right = 112
    bottom = 116
    plot_width = len(heads) * cell_width
    plot_height = len(ordinals) * cell_height
    width = left + plot_width + right
    height = top + plot_height + bottom
    explicit_width = f"width:{width}px;max-width:none" if len(heads) > 16 else "width:100%;max-width:760px"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="ordinal-map" style="{explicit_width}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_escape(model)} {_escape(scope)} target needle ordinal by ranked head">',
        f'<title>{_escape(model)} · {_escape(scope)} · target needle ordinal × ranked head</title>',
        '<desc>Columns are ranked layer-head identities; rows are the ordinal of the next needle retrieved at exact P0. Color is raw attention mass to the correct prompt record span.</desc>',
        '<rect width="100%" height="100%" rx="14" fill="#101927"/>',
    ]
    for column, head_row in enumerate(heads):
        layer = int(head_row["layer"])
        head = int(head_row["head"])
        rank = int(head_row["rank"])
        for row_index, ordinal in enumerate(ordinals):
            x = left + column * cell_width
            y = top + row_index * cell_height
            cell = value_by_cell.get((ordinal, layer, head))
            if cell is None or cell.get("value") is None:
                fill = "#293548"
                title = (
                    f"{model} · {scope} · L{layer}H{head} · rank={rank} · "
                    f"needle #{ordinal} · no eligible events"
                )
            else:
                value = float(cell["value"])
                fill = _color(value, shared_vmax)
                title = (
                    f"{model} · {scope} · L{layer}H{head} · rank={rank} · "
                    f"needle #{ordinal} · mass={value:.6f} · "
                    f"seeds={cell['n_seeds']} · events={cell['n_events']}"
                )
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" fill="{fill}" stroke="#243348" '
                f'stroke-width="0.6"><title>{_escape(title)}</title></rect>'
            )
        if column > 0 and column % 16 == 0:
            x = left + column * cell_width
            parts.append(
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + plot_height}" '
                'stroke="#d8e2ee" stroke-opacity="0.48" stroke-width="1.2"/>'
            )
        x = left + (column + 0.5) * cell_width
        label_y = top + plot_height + 12
        parts.append(
            f'<text transform="translate({x:.2f} {label_y}) rotate(65)" '
            f'text-anchor="start" font-size="{8 if len(heads) > 16 else 10}" '
            f'fill="#b7c5d6">L{layer}H{head}</text>'
        )
    for row_index, ordinal in enumerate(ordinals):
        y = top + (row_index + 0.64) * cell_height
        parts.append(
            f'<text x="{left - 12}" y="{y:.2f}" text-anchor="end" '
            f'font-size="11" fill="#c9d5e4">needle #{ordinal}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" '
            'text-anchor="middle" font-size="12" font-weight="700" fill="#d8e2ee">ranked retrieval heads (LxHy)</text>',
            f'<text transform="translate(17 {top + plot_height / 2:.2f}) rotate(-90)" '
            'text-anchor="middle" font-size="12" font-weight="700" fill="#d8e2ee">target needle ordinal</text>',
        ]
    )
    legend_x = left + plot_width + 28
    legend_y = top
    legend_height = min(250, plot_height)
    steps = 60
    for index in range(steps):
        fraction = index / max(1, steps - 1)
        y = legend_y + (1 - fraction) * legend_height
        parts.append(
            f'<rect x="{legend_x}" y="{y:.2f}" width="14" '
            f'height="{legend_height / steps + 0.8:.2f}" '
            f'fill="{_color(fraction * shared_vmax, shared_vmax)}"/>'
        )
    for fraction in (0.0, 0.5, 1.0):
        y = legend_y + (1 - fraction) * legend_height + 4
        parts.append(
            f'<text x="{legend_x + 21}" y="{y:.2f}" font-size="9" '
            f'fill="#b7c5d6">{fraction * shared_vmax:.3f}</text>'
        )
    parts.append(
        f'<text transform="translate({legend_x + 66} {legend_y + legend_height / 2:.2f}) rotate(-90)" '
        'text-anchor="middle" font-size="10" fill="#d8e2ee">correct-needle raw attention mass</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _event_target_mass(event: dict[str, Any]) -> float:
    return sum(float(row["mass"]) for row in event["records"] if row["is_target"])


def _event_needle_total(event: dict[str, Any]) -> float:
    return sum(float(row["mass"]) for row in event["records"])


def _attention_metrics(example: dict[str, Any]) -> dict[str, float]:
    target = [_event_target_mass(event) for event in example["events"]]
    totals = [_event_needle_total(event) for event in example["events"]]
    top1 = []
    relative = []
    for event, target_mass, total in zip(example["events"], target, totals):
        wrong = [float(row["mass"]) for row in event["records"] if not row["is_target"]]
        top1.append(float(target_mass >= max(wrong, default=0.0)))
        relative.append(target_mass / total if total > 0 else 0.0)
    return {
        "mean_target_mass": sum(target) / len(target),
        "mean_target_share": sum(relative) / len(relative),
        "target_top1_rate": sum(top1) / len(top1),
    }


def _attention_svg(example: dict[str, Any]) -> str:
    events = example["events"]
    first_records = events[0]["records"]
    labels = [f"N{row['source_index']} · {row['city']}" for row in first_records]
    labels.append("non-needle / trace context")
    rows_n = len(labels)
    columns_n = len(events)
    cell_width = 74
    cell_height = 31
    left = 176
    top = 58
    right = 92
    bottom = 80
    plot_width = columns_n * cell_width
    plot_height = rows_n * cell_height
    width = left + plot_width + right
    height = top + plot_height + bottom
    is_bank = "bank_size" in example
    bank_size = int(example.get("bank_size", 1))
    identity = (
        f"Top-{bank_size} bank-summed"
        if is_bank
        else f"L{example['layer']}H{example['head']}"
    )
    # The aggregate non-needle row often holds most of the K units of total
    # mass.  Scaling a bank map to 0..K would therefore collapse every
    # individual city cell into the darkest colors.  Keep the encoded quantity
    # as the raw sum, but cap the color scale at the largest observed *needle*
    # cell; context values above that cap are intentionally saturated.
    legend_maximum = (
        max(
            float(record["mass"])
            for event in events
            for record in event["records"]
        )
        if is_bank
        else 1.0
    )
    legend_maximum = max(legend_maximum, 1e-12)
    legend_label = (
        f"Σ attention mass over Top-{bank_size} heads (needle max)"
        if is_bank
        else "raw attention mass"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="attention-map" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(example["model_label"])} {_escape(identity)} P0 per-needle attention distribution">',
        f'<title>{_escape(example["model_label"])} · {_escape(identity)} · exact-P0 city attention map</title>',
        '<rect width="100%" height="100%" rx="14" fill="#101927"/>',
    ]
    for column, event in enumerate(events):
        city_rows = {
            str(row["city"]): row for row in event["records"]
        }
        ordered = [city_rows[str(row["city"])] for row in first_records]
        masses = [float(row["mass"]) for row in ordered]
        masses.append(float(event["non_needle_context_mass"]))
        for row_index, mass in enumerate(masses):
            x = left + column * cell_width
            y = top + row_index * cell_height
            is_target = row_index < len(ordered) and bool(ordered[row_index]["is_target"])
            title = (
                f"event {event['from_occurrence']}→{event['to_occurrence']} · "
                f"{labels[row_index]} · mass={mass:.6f}"
                + (" · target" if is_target else "")
            )
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="{_color(mass, legend_maximum)}" stroke="{"#ff6b57" if is_target else "#243348"}" '
                f'stroke-width="{3 if is_target else 0.7}"><title>{_escape(title)}</title></rect>'
            )
            if is_target:
                parts.append(
                    f'<circle cx="{x + cell_width - 8}" cy="{y + 8}" r="3.2" fill="#ff6b57"/>'
                )
        x = left + (column + 0.5) * cell_width
        parts.append(
            f'<text x="{x}" y="{top - 23}" text-anchor="middle" font-size="11" '
            f'font-weight="800" fill="#e6eef8">{event["from_occurrence"]}→{event["to_occurrence"]}</text>'
        )
        parts.append(
            f'<text x="{x}" y="{top - 8}" text-anchor="middle" font-size="9" '
            f'fill="#90a5bd">q={event["query_output_token_index"]}</text>'
        )
    for row_index, label in enumerate(labels):
        y = top + (row_index + 0.66) * cell_height
        parts.append(
            f'<text x="{left - 10}" y="{y:.2f}" text-anchor="end" font-size="10" '
            f'fill="#c9d5e4">{_escape(label)}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" '
            'font-size="12" font-weight="700" fill="#d8e2ee">P0 transition query (k→k+1)</text>',
            f'<text transform="translate(15 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-size="12" font-weight="700" fill="#d8e2ee">prompt region</text>',
        ]
    )
    legend_x = left + plot_width + 27
    legend_y = top
    legend_height = min(220, plot_height)
    for index in range(60):
        fraction = index / 59
        y = legend_y + (1 - fraction) * legend_height
        parts.append(
            f'<rect x="{legend_x}" y="{y:.2f}" width="14" height="{legend_height / 60 + 0.8:.2f}" '
            f'fill="{_color(fraction * legend_maximum, legend_maximum)}"/>'
        )
    for fraction in (0.0, 0.5, 1.0):
        y = legend_y + (1 - fraction) * legend_height + 4
        parts.append(
            f'<text x="{legend_x + 20}" y="{y:.2f}" font-size="9" fill="#b7c5d6">{fraction * legend_maximum:.1f}</text>'
        )
    parts.append(
        f'<text transform="translate({legend_x + 53} {legend_y + legend_height / 2}) rotate(-90)" '
        f'text-anchor="middle" font-size="10" fill="#d8e2ee">{_escape(legend_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _top_head_rows(rankings: dict[str, Any]) -> str:
    rows: list[str] = []
    for grammar, bundle in rankings.items():
        n_seeds = int(bundle["n_seeds"])
        top = bundle["rows"][:5]
        heads = ", ".join(
            f"L{row['layer']}H{row['head']} ({float(row['score']):.3f})"
            for row in top
        )
        status = "claim-grade" if n_seeds >= ROBUST_SEED_THRESHOLD else "exploratory"
        rows.append(
            "<tr>"
            f"<td><code>{_escape(grammar)}</code></td>"
            f"<td>{n_seeds}</td>"
            f"<td><span class=\"status {status}\">{status}</span></td>"
            f"<td>{_escape(heads)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _write_csvs(
    bundles: dict[str, dict[str, Any]], assets: Path
) -> tuple[Path, Path, Path]:
    head_path = assets / "p0_targeted_retrieval_head_scores.csv"
    with head_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_label", "grammar", "plan_k", "layer", "head", "score", "rank", "n_seeds"],
        )
        writer.writeheader()
        for model in MODEL_ORDER:
            for grammar, bundle in bundles[model]["rankings"].items():
                for row in bundle["rows"]:
                    writer.writerow(
                        {
                            "model_label": model,
                            "grammar": grammar,
                            "plan_k": bundle["plan_k"],
                            "layer": row["layer"],
                            "head": row["head"],
                            "score": row["score"],
                            "rank": row["rank"],
                            "n_seeds": row["n_seeds"],
                        }
                    )
    ordinal_path = assets / "p0_needle_ordinal_by_head.csv"
    with ordinal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_label",
                "scope",
                "target_ordinal",
                "rank",
                "layer",
                "head",
                "raw_attention_mass",
                "n_seeds",
                "n_events",
            ],
        )
        writer.writeheader()
        for model in MODEL_ORDER:
            for scope, scope_bundle in bundles[model]["ordinal"]["scopes"].items():
                for row in scope_bundle["ordinal_rows"]:
                    writer.writerow(
                        {
                            "model_label": model,
                            "scope": scope,
                            "target_ordinal": row["target_ordinal"],
                            "rank": row["rank"],
                            "layer": row["layer"],
                            "head": row["head"],
                            "raw_attention_mass": row["value"],
                            "n_seeds": row["n_seeds"],
                            "n_events": row["n_events"],
                        }
                    )
    attention_path = assets / "p0_significant_head_attention_masses.csv"
    with attention_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_label",
                "grammar",
                "layer",
                "head",
                "seed",
                "request_id",
                "from_occurrence",
                "to_occurrence",
                "query_output_token_index",
                "source_index",
                "source_city",
                "region",
                "is_target",
                "raw_attention_mass",
            ],
        )
        writer.writeheader()
        for model in MODEL_ORDER:
            for example in bundles[model]["examples"]:
                for event in example["events"]:
                    for row in event["records"]:
                        writer.writerow(
                            {
                                "model_label": model,
                                "grammar": example["grammar"],
                                "layer": example["layer"],
                                "head": example["head"],
                                "seed": example["seed"],
                                "request_id": example["request_id"],
                                "from_occurrence": event["from_occurrence"],
                                "to_occurrence": event["to_occurrence"],
                                "query_output_token_index": event["query_output_token_index"],
                                "source_index": row["source_index"],
                                "source_city": row["city"],
                                "region": "needle_record",
                                "is_target": row["is_target"],
                                "raw_attention_mass": row["mass"],
                            }
                        )
                    writer.writerow(
                        {
                            "model_label": model,
                            "grammar": example["grammar"],
                            "layer": example["layer"],
                            "head": example["head"],
                            "seed": example["seed"],
                            "request_id": example["request_id"],
                            "from_occurrence": event["from_occurrence"],
                            "to_occurrence": event["to_occurrence"],
                            "query_output_token_index": event["query_output_token_index"],
                            "source_index": "",
                            "source_city": "",
                            "region": "non_needle_context",
                            "is_target": False,
                            "raw_attention_mass": event["non_needle_context_mass"],
                        }
                    )
    return head_path, ordinal_path, attention_path










def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=DEFAULT_DATA / "p0_head_atlas_qwen.json")
    parser.add_argument("--gemma", type=Path, default=DEFAULT_DATA / "p0_head_atlas_gemma.json")
    parser.add_argument(
        "--qwen-ordinal",
        type=Path,
        default=DEFAULT_DATA / "p0_head_ordinal_qwen.json",
    )
    parser.add_argument(
        "--gemma-ordinal",
        type=Path,
        default=DEFAULT_DATA / "p0_head_ordinal_gemma.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    paths = {"Qwen3-8B": args.qwen, "Gemma4-E4B": args.gemma}
    ordinal_paths = {
        "Qwen3-8B": args.qwen_ordinal,
        "Gemma4-E4B": args.gemma_ordinal,
    }
    bundles = {
        model: json.loads(path.read_text(encoding="utf-8"))
        for model, path in paths.items()
    }
    ordinals = {
        model: json.loads(path.read_text(encoding="utf-8"))
        for model, path in ordinal_paths.items()
    }
    for model in MODEL_ORDER:
        if bundles[model].get("model_label") != model:
            raise ValueError(f"Expected {model} data in {paths[model]}")
        if bundles[model].get("query_site") != "p0_item_end":
            raise ValueError(f"{model} atlas is not exact P0 data")
        if ordinals[model].get("model_label") != model:
            raise ValueError(f"Expected {model} ordinal data in {ordinal_paths[model]}")
        if ordinals[model].get("query_site") != "p0_item_end":
            raise ValueError(f"{model} ordinal data is not exact P0 data")
        if ordinals[model].get("selection_aggregation") != EXPECTED_AGGREGATION:
            raise ValueError(f"{model} ordinal data does not use registered seed weighting")
        if int(ordinals[model].get("top_k", -1)) != MODEL_K[model]:
            raise ValueError(f"{model} ordinal Top-K does not match the registered display K")
        if "all" not in ordinals[model].get("scopes", {}):
            raise ValueError(f"{model} ordinal data has no all-scope aggregation")
        grammar_scopes = set(bundles[model]["rankings"])
        ordinal_grammar_scopes = set(ordinals[model]["scopes"]) - {"all"}
        if grammar_scopes != ordinal_grammar_scopes:
            raise ValueError(
                f"{model} grammar scopes differ between atlas and ordinal data: "
                f"{grammar_scopes ^ ordinal_grammar_scopes}"
            )
        all_scope = ordinals[model]["scopes"]["all"]
        bundles[model]["rankings"] = {
            "all": {
                "plan_k": MODEL_K[model],
                "n_seeds": all_scope["n_seeds"],
                "rows": all_scope["ranking"],
            },
            **bundles[model]["rankings"],
        }
        bundles[model]["ordinal"] = ordinals[model]
    args.assets.mkdir(parents=True, exist_ok=True)
    head_csv, ordinal_csv, attention_csv = _write_csvs(bundles, args.assets)
    manifest = {
        "schema_version": "realistic_niah_v5_p0_head_atlas_exports_v1",
        "query_site": "p0_item_end",
        "selection_split": "discovery",
        "selection_metric": "seed_event_mean_target_source_attention_mass",
        "exploratory_seed_threshold": ROBUST_SEED_THRESHOLD,
        "inputs": {
            model: {
                "atlas": {"path": str(paths[model]), "sha256": _sha256(paths[model])},
                "ordinal": {
                    "path": str(ordinal_paths[model]),
                    "sha256": _sha256(ordinal_paths[model]),
                },
            }
            for model in MODEL_ORDER
        },
        "outputs": {
            "head_scores_csv": str(head_csv),
            "needle_ordinal_by_head_csv": str(ordinal_csv),
            "attention_masses_csv": str(attention_csv),
        },
    }
    manifest_path = args.assets / "p0_head_atlas_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
