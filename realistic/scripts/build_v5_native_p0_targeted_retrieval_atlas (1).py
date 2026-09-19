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
        f'<svg class="head-map" viewBox="0 0 {width} {height}" role="img" '
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
        f'<svg class="ordinal-map" style="{explicit_width}" viewBox="0 0 {width} {height}" '
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
        f'<svg class="attention-map" viewBox="0 0 {width} {height}" role="img" '
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


def _model_section(model: str, bundle: dict[str, Any], assets: Path) -> str:
    rankings = bundle["rankings"]
    default_grammar = "all"
    shared_vmax = max(
        float(row["score"])
        for value in rankings.values()
        for row in value["rows"]
    )
    maps: list[str] = []
    options: list[str] = []
    for grammar, value in rankings.items():
        n_seeds = int(value["n_seeds"])
        exploratory = n_seeds < ROBUST_SEED_THRESHOLD
        status = " · exploratory" if exploratory else ""
        options.append(
            f'<option value="{_escape(grammar)}"{' selected' if grammar == default_grammar else ''}>'
            f'{_escape(_scope_label(grammar))} · {n_seeds} seeds{status}</option>'
        )
        svg = _head_map_svg(model, grammar, value, shared_vmax=shared_vmax)
        asset_name = f"{model}_{grammar}_p0_head_map.svg"
        (assets / asset_name).write_text(svg, encoding="utf-8")
        maps.append(
            f'<div class="map-panel" data-grammar="{_escape(grammar)}" '
            f'style="display:{"block" if grammar == default_grammar else "none"}">'
            f'<div class="map-meta"><code>{_escape(grammar)}</code><span>{n_seeds} discovery seeds</span>'
            f'<span>{"descriptive global Top-K" if grammar == "all" else "frozen grammar bank"}</span>'
            f'<span class="status {"exploratory" if exploratory else "claim-grade"}">'
            f'{"exploratory" if exploratory else "claim-grade"}</span></div>{svg}</div>'
        )
    return f"""
    <section class="model-block" id="map-{_escape(model)}">
      <div class="section-kicker">{_escape(model)} · all + grammar-specific P0</div>
      <div class="model-heading">
        <div>
          <h3>{_escape(model)} head map</h3>
          <p>共享色标范围 0–{shared_vmax:.3f}；grammar 视图的白色细框为冻结 Top-{MODEL_K[model]} bank，<code>all</code> 视图的白框仅表示按全体 discovery 聚合得到的描述性 global Top-{MODEL_K[model]}。格内数字标出 Top-8 rank。</p>
        </div>
        <label class="selector">trace scope
          <select data-map-selector="{_escape(model)}">{''.join(options)}</select>
        </label>
      </div>
      <div data-map-container="{_escape(model)}">{''.join(maps)}</div>
      <details>
        <summary>查看各 grammar 的 Top-5 heads</summary>
        <div class="table-wrap"><table><thead><tr><th>Grammar</th><th>Discovery seeds</th><th>Status</th><th>Top-5 (score)</th></tr></thead>
        <tbody>{_top_head_rows(rankings)}</tbody></table></div>
      </details>
    </section>
    """


def _ordinal_model_section(model: str, bundle: dict[str, Any], assets: Path) -> str:
    scopes = bundle["ordinal"]["scopes"]
    default_scope = "all"
    shared_vmax = max(
        float(row["value"])
        for scope_bundle in scopes.values()
        for row in scope_bundle["ordinal_rows"]
        if row.get("value") is not None
    )
    panels: list[str] = []
    options: list[str] = []
    for scope, scope_bundle in scopes.items():
        n_seeds = int(scope_bundle["n_seeds"])
        exploratory = n_seeds < ROBUST_SEED_THRESHOLD
        status = " · exploratory" if exploratory else ""
        options.append(
            f'<option value="{_escape(scope)}"{' selected' if scope == default_scope else ''}>'
            f'{_escape(_scope_label(scope))} · {n_seeds} seeds{status}</option>'
        )
        svg = _ordinal_head_svg(
            model,
            scope,
            scope_bundle,
            shared_vmax=shared_vmax,
        )
        asset_name = f"{model}_{scope}_p0_needle_ordinal_by_head.svg"
        (assets / asset_name).write_text(svg, encoding="utf-8")
        panels.append(
            f'<div class="ordinal-panel" data-scope="{_escape(scope)}" '
            f'style="display:{"block" if scope == default_scope else "none"}">'
            f'<div class="map-meta"><code>{_escape(scope)}</code>'
            f'<span>{n_seeds} discovery seeds</span>'
            f'<span>Top-{MODEL_K[model]} heads ranked within this scope</span>'
            f'{"<span>横向滚动查看全部 128 heads</span>" if model == "Qwen3-8B" else ""}'
            f'<span class="status {"exploratory" if exploratory else "claim-grade"}">'
            f'{"exploratory" if exploratory else "claim-grade"}</span></div>'
            f'<div class="ordinal-scroll">{svg}</div></div>'
        )
    return f"""
    <section class="model-block" id="ordinal-{_escape(model)}">
      <div class="section-kicker">{_escape(model)} · retrieval progression</div>
      <div class="model-heading">
        <div>
          <h3>{_escape(model)} needle ordinal × head</h3>
          <p>横轴按当前 scope 的 P0 targeted-retrieval score 从高到低排列 heads；纵轴是正在检索的下一条 needle 序号 #2–#10。模型内所有 scope 共享色标 0–{shared_vmax:.3f}。</p>
        </div>
        <label class="selector">trace scope
          <select data-ordinal-selector="{_escape(model)}">{''.join(options)}</select>
        </label>
      </div>
      <div data-ordinal-container="{_escape(model)}">{''.join(panels)}</div>
    </section>
    """


def _example_cards(bundles: dict[str, dict[str, Any]], assets: Path) -> str:
    cards: list[str] = []
    for model in MODEL_ORDER:
        ranking = bundles[model]["rankings"]
        for example in bundles[model]["examples"]:
            svg = _attention_svg({"model_label": model, **example})
            name = f"{model}_{example['grammar']}_L{example['layer']}H{example['head']}_p0_attention.svg"
            (assets / name).write_text(svg, encoding="utf-8")
            rank_row = next(
                row
                for row in ranking[example["grammar"]]["rows"]
                if int(row["layer"]) == int(example["layer"])
                and int(row["head"]) == int(example["head"])
            )
            metrics = _attention_metrics(example)
            cards.append(
                f"""
                <article class="attention-card">
                  <div class="attention-title">
                    <div><span class="model-pill">{_escape(model)}</span>
                    <h3>L{example['layer']}H{example['head']} · rank {rank_row['rank']}</h3></div>
                    <div class="score-chip">P0 score <strong>{float(rank_row['score']):.3f}</strong></div>
                  </div>
                  <p class="mono-line"><code>{_escape(example['grammar'])}</code> · seed {example['seed']} · N={example['gold_count']} · {len(example['events'])} P0 queries</p>
                  {svg}
                  <div class="metric-row">
                    <span>mean target mass <strong>{metrics['mean_target_mass']:.3f}</strong></span>
                    <span>target / all-needle <strong>{metrics['mean_target_share']:.1%}</strong></span>
                    <span>target top-1 <strong>{metrics['target_top1_rate']:.1%}</strong></span>
                  </div>
                </article>
                """.strip()
            )
        for example in bundles[model].get("bank_examples", []):
            svg = _attention_svg({"model_label": model, **example})
            name = (
                f"{model}_{example['grammar']}_Top{example['bank_size']}"
                "_p0_attention_sum.svg"
            )
            (assets / name).write_text(svg, encoding="utf-8")
            metrics = _attention_metrics(example)
            cards.append(
                f"""
                <article class="attention-card">
                  <div class="attention-title">
                    <div><span class="model-pill">{_escape(model)}</span>
                    <h3>Top-{example['bank_size']} · bank-summed city map</h3></div>
                    <div class="score-chip">Frozen bank <strong>K={example['bank_size']}</strong></div>
                  </div>
                  <p class="mono-line"><code>{_escape(example['grammar'])}</code> · seed {example['seed']} · N={example['gold_count']} · {len(example['events'])} exact-P0 queries</p>
                  {svg}
                  <div class="metric-row">
                    <span>mean Σ target mass <strong>{metrics['mean_target_mass']:.3f}</strong></span>
                    <span>target / all-needle <strong>{metrics['mean_target_share']:.1%}</strong></span>
                    <span>target top-1 <strong>{metrics['target_top1_rate']:.1%}</strong></span>
                  </div>
                </article>
                """.strip()
            )
    return "".join(cards)


def _build_html(bundles: dict[str, dict[str, Any]], assets: Path) -> str:
    model_sections = "".join(
        _model_section(model, bundles[model], assets) for model in MODEL_ORDER
    )
    ordinal_sections = "".join(
        _ordinal_model_section(model, bundles[model], assets) for model in MODEL_ORDER
    )
    example_cards = _example_cards(bundles, assets)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native-thinking P0 Targeted Retrieval Atlas</title>
<style>
:root{{--ink:#162235;--muted:#607087;--paper:#f6f2ea;--card:#fffdf9;--line:#d8d0c3;--navy:#132238;--teal:#0f8b8d;--amber:#c56f12;--coral:#e35d4f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Segoe UI","Noto Sans SC",Arial,sans-serif;line-height:1.62}}
.hero{{background:linear-gradient(135deg,#101b2b,#17384a 68%,#0f7b78);color:#fff;padding:64px max(28px,calc((100vw - 1180px)/2)) 54px}}
.eyebrow,.section-kicker{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:850;color:#8bd4cf}} .hero h1{{font-size:clamp(32px,5vw,58px);line-height:1.05;margin:12px 0 18px;max-width:980px}}
.hero p{{max-width:930px;margin:0;color:#d6e4eb;font-size:18px}} .hero-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:30px}}
.hero-stat{{border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.07);border-radius:14px;padding:14px}} .hero-stat strong{{display:block;font-size:24px}}
main{{max-width:1180px;margin:0 auto;padding:34px 24px 72px}} section{{margin:0 0 30px}} .card,.model-block,.attention-card{{background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 30px rgba(31,41,55,.055);padding:24px}}
h2{{font-size:30px;margin:4px 0 12px}} h3{{margin:4px 0 8px;font-size:22px}} p{{margin:8px 0 14px}} code{{background:#eef2f1;border:1px solid #d8e2df;border-radius:5px;padding:1px 5px;font-family:"SFMono-Regular",Consolas,monospace;font-size:.92em}}
.formula{{background:#122034;color:#e9f2f6;border-radius:14px;padding:19px 22px;font-family:Georgia,serif;font-size:18px;overflow:auto}} .formula small{{display:block;color:#9eb3c6;font-family:Inter,"Segoe UI",sans-serif;font-size:13px;margin-top:8px}}
.callout{{border-left:5px solid var(--teal);background:#e9f4f1;border-radius:0 12px 12px 0;padding:14px 17px;margin:18px 0}} .conclusion{{border-left-color:var(--amber);background:#fff0d8}}
.model-block{{padding:26px;margin-top:20px}} .model-heading,.attention-title{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}} .model-heading p{{color:var(--muted);margin:0;max-width:760px}}
.selector{{font-size:12px;text-transform:uppercase;letter-spacing:.08em;font-weight:800;color:var(--muted);min-width:300px}} select{{display:block;width:100%;margin-top:6px;border:1px solid #bac5d0;border-radius:9px;background:#fff;padding:9px 11px;color:var(--ink)}}
.map-panel,.ordinal-panel{{margin-top:20px}} .map-meta{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px;color:var(--muted);font-size:13px}} svg{{display:block;width:100%;height:auto}} .head-map{{max-height:690px}} .ordinal-scroll{{overflow-x:auto;overflow-y:hidden;padding-bottom:8px}} .ordinal-map{{height:auto}}
.status{{display:inline-flex;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:850;text-transform:uppercase;letter-spacing:.04em}} .status.claim-grade{{background:#dff2e8;color:#17633a}} .status.exploratory{{background:#fff0d8;color:#9a5212}}
details{{margin-top:16px;border-top:1px solid var(--line);padding-top:13px}} summary{{cursor:pointer;font-weight:800}} .table-wrap{{overflow:auto;margin-top:12px}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid #e3ddd3;text-align:left;vertical-align:top}} th{{background:#f0ece4}}
.example-grid{{display:grid;grid-template-columns:1fr;gap:20px}} .attention-card{{padding:24px}} .model-pill{{color:var(--teal);font-weight:900;font-size:12px;letter-spacing:.08em;text-transform:uppercase}} .score-chip{{background:#fff0d8;border:1px solid #e8c99f;border-radius:10px;padding:8px 12px;color:#7d480e}} .score-chip strong{{font-size:22px;margin-left:6px}} .mono-line{{color:var(--muted);font-size:13px}}
.metric-row{{display:flex;gap:12px;flex-wrap:wrap;margin-top:13px}} .metric-row span{{background:#eef3f5;border-radius:9px;padding:7px 10px;font-size:13px}} .metric-row strong{{margin-left:4px}}
.downloads{{display:flex;gap:12px;flex-wrap:wrap}} .downloads a{{display:inline-flex;padding:9px 13px;border-radius:9px;background:#132238;color:#fff;text-decoration:none;font-weight:750;font-size:13px}}
.footnote{{font-size:13px;color:var(--muted)}}
@media(max-width:800px){{.hero-grid{{grid-template-columns:1fr 1fr}} .model-heading,.attention-title{{display:block}} .selector{{min-width:0;margin-top:14px}} .model-block,.attention-card,.card{{padding:17px}} main{{padding-left:14px;padding-right:14px}}}}
</style>
</head>
<body>
<header class="hero">
  <div class="eyebrow">Native-thinking · descriptive attention atlas</div>
  <h1>P0 Targeted Retrieval<br>Head Map & Attention Distributions</h1>
  <p>严格使用当前 grammar-specific P0 discovery ranking：同一个精确 <code>p0_item_end</code> token 同时用于 attention ranking 与后续 ablation。这里先回答“哪些 heads 在 P0 对正确 needle 有高 attention mass，以及单头具体看向哪里”。</p>
  <div class="hero-grid">
    <div class="hero-stat"><strong>2</strong><span>models</span></div>
    <div class="hero-stat"><strong>14</strong><span>all + grammar map scopes</span></div>
    <div class="hero-stat"><strong>14</strong><span>ordinal × head maps</span></div>
    <div class="hero-stat"><strong>4</strong><span>single-head examples</span></div>
  </div>
</header>
<main>
  <section class="card">
    <div class="section-kicker">Definition</div>
    <h2>图 1：Targeted retrieval score 的 layer × head 地图</h2>
    <p>对 grammar <em>g</em>，先在每个 discovery seed 内对该 grammar 的所有 P0 transition events 求均值，再对 seed 等权平均。每个 event 的基础量是该 head 从 P0 query 指向“下一条正确 city 所在 prompt record span”的 raw attention mass。<code>all</code> 使用完全相同的两级聚合，只是 seed 内纳入该 seed 的全部 eligible grammar events；它不是各 grammar 均值的再平均。</p>
    <div class="formula">S<sub>g</sub><sup>P0</sup>(ℓ,h) = (1 / |D<sub>g</sub>|) Σ<sub>s∈Dg</sub> (1 / |E<sub>s,g</sub>|) Σ<sub>e∈Es,g</sub> Σ<sub>t∈R(target(e))</sub> A<sub>ℓ,h</sub>(q<sub>e</sub><sup>P0</sup>, t)
      <small>q<sup>P0</sup>: 完整 item k 的 endpoint token；R(target(e)): needle k+1 在 prompt 中的完整 record token span。</small>
    </div>
    <div class="formula">S<sub>all</sub><sup>P0</sup>(ℓ,h) = (1 / |D|) Σ<sub>s∈D</sub> (1 / |E<sub>s,all</sub>|) Σ<sub>e∈Es,all</sub> Σ<sub>t∈R(target(e))</sub> A<sub>ℓ,h</sub>(q<sub>e</sub><sup>P0</sup>, t)
      <small>这保证每个 discovery seed 的总权重相同；event 较多的 grammar 不会跨 seed 重复加权。</small>
    </div>
    <div class="callout"><strong>读图规则。</strong> 横轴是 head，纵轴是 decoder layer；颜色越亮表示 discovery P0 targeted-retrieval score 越大。每个模型内 <code>all</code> 与所有 grammar 共享同一色标，因此可以比较绝对强度。grammar 视图的白框是实际冻结的 selected bank；<code>all</code> 视图的白框只是描述性的 global Top-K，不对应新增 intervention。数字仅标出 Top-8。</div>
  </section>

  {model_sections}

  <section class="card">
    <div class="section-kicker">Ordinal decomposition</div>
    <h2>图 2：Needle 序号 × ranked head</h2>
    <p>这一图把同一个 P0 targeted-retrieval quantity 按“下一条 needle 是第几个”拆开。对固定 scope 和 ordinal <em>j</em>，仍然先在每个 seed 内平均所有检索 needle #<em>j</em> 的 eligible events，再对 seed 等权平均；head 的横向次序则由该 scope 的整体 P0 score 决定。</p>
    <div class="callout"><strong>坐标与色标。</strong> 横轴的每一列是一个完整 layer–head identity（<code>LxHy</code>），不是只看 head index；纵轴是 transition k→k+1 中被读取的 target needle ordinal #2–#10。颜色是该精确 P0 query 指向正确 target record span 的 raw attention mass。Qwen 展示每个 scope 的 Top-128，Gemma 展示 Top-8；灰色表示该 scope 在该 ordinal 没有 eligible event。</div>
  </section>

  {ordinal_sections}

  <section class="card">
    <div class="section-kicker">Single-head routing</div>
    <h2>图 3：显著 retrieval heads 的逐 needle attention 分布</h2>
    <p>为避免低样本 grammar 的视觉偶然性，四个例子都来自两模型 seed 覆盖最完整的 <code>adjacent_rank_after_city</code>。每个模型选其稳定 Top heads，并在一个确定性代表 trace 上重算 exact P0 attention：优先含最多 eligible events、再优先更大 N、最后取更小 seed；没有按图形“好看程度”挑样本。</p>
    <div class="callout"><strong>坐标与标记。</strong> 横轴是 transition k→k+1 的 P0 query；纵轴按 prompt token 位置排列 needle record spans，最后一行合并所有非-needle key（instruction、filler 与已经生成的 trace）。颜色是 raw attention mass，红框标出该列真正应该读取的 target needle。</div>
  </section>
  <section class="example-grid">{example_cards}</section>

  <section class="card">
    <div class="section-kicker">Interpretation</div>
    <h2>目前能得到的结论</h2>
    <ol>
      <li><strong>P0 retrieval 不是均匀分散在所有层。</strong> Qwen 的高分 heads 明显集中在中后段（尤其 L20–L24），Gemma 则集中在少数离散层（尤其 L17、L23、L29）。</li>
      <li><strong>grammar-specific ranking 仍共享核心 heads。</strong> 切换 grammar 后强度和次序会变化，但 Qwen 的 L20H30/L24H29 一组与 Gemma 的 L29H4/L17H2 一组反复出现在前列；这解释了为什么统一 bank 有一定可迁移性，同时又不等同于每类的最优 bank。</li>
      <li><strong><code>all</code> 视图给出不依赖单一 surface grammar 的共同排序。</strong> 它严格按 discovery seed 等权，因此可以作为全体 trace 的描述性总览；但因果实验仍以各 grammar 的冻结 bank 为准。</li>
      <li><strong>ordinal 图检验 retrieval 是否随 count 进程换头。</strong> 同一列跨 #2–#10 的颜色变化表示同一 head 在不同检索步的强度变化；跨列的亮带则显示不同 heads 对 ordinal 的分工或稳定复用。</li>
      <li><strong>单头图直接显示“在 P0 看向哪个 prompt record”。</strong> 红框单元格对应下一条正确 needle；target mass、target/all-needle share 与 target top-1 是三个互补描述量。</li>
    </ol>
    <div class="callout conclusion"><strong>结论边界。</strong> 这些图证明的是 attention routing / localization，不单独证明某一个 head 对输出具有因果必要性。因果结论仍由同位点、持续关闭的 selected-vs-random ablation 给出。</div>
  </section>

  <section class="card">
    <div class="section-kicker">Artifacts</div>
    <h2>可复核数据</h2>
    <div class="downloads">
      <a href="v5_native_p0_head_atlas/p0_targeted_retrieval_head_scores.csv">Head scores CSV</a>
      <a href="v5_native_p0_head_atlas/p0_needle_ordinal_by_head.csv">Ordinal × head CSV</a>
      <a href="v5_native_p0_head_atlas/p0_significant_head_attention_masses.csv">Attention masses CSV</a>
      <a href="v5_native_p0_head_atlas/p0_head_atlas_manifest.json">Manifest</a>
    </div>
    <p class="footnote">所有数值均来自 discovery split；confirmation 没有参与 head ranking 或示例选择。n&lt;10 discovery seeds 的 grammar 在页面中统一标为 exploratory。</p>
  </section>
</main>
<script>
document.querySelectorAll('[data-map-selector]').forEach(select => {{
  select.addEventListener('change', () => {{
    const model = select.dataset.mapSelector;
    document.querySelectorAll(`[data-map-container="${{model}}"] .map-panel`).forEach(panel => {{
      panel.style.display = panel.dataset.grammar === select.value ? 'block' : 'none';
    }});
  }});
}});
document.querySelectorAll('[data-ordinal-selector]').forEach(select => {{
  select.addEventListener('change', () => {{
    const model = select.dataset.ordinalSelector;
    document.querySelectorAll(`[data-ordinal-container="${{model}}"] .ordinal-panel`).forEach(panel => {{
      panel.style.display = panel.dataset.scope === select.value ? 'block' : 'none';
    }});
  }});
}});
</script>
</body>
</html>
"""


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
    document = _build_html(bundles, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    manifest = {
        "schema_version": "realistic_niah_v5_p0_head_atlas_report_v2",
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
            "html": str(args.output),
            "head_scores_csv": str(head_csv),
            "needle_ordinal_by_head_csv": str(ordinal_csv),
            "attention_masses_csv": str(attention_csv),
        },
    }
    manifest_path = args.assets / "p0_head_atlas_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
