#!/usr/bin/env python3
"""Build the final Native-thinking report in the Non-thinking report grammar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_v5_native_thinking_report_v4 import (
    alignment_svg,
    load_atlas,
    load_duplicates,
    load_representation,
    representation_svg,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
GRAMMAR_LABELS = {
    "adjacent_rank_after_city": "city → rank（相邻单元）",
    "adjacent_rank_before_city": "rank → city（相邻单元）",
    "same_unit_rank_before_city": "rank → city（同一单元）",
    "structural_unmarked": "无显式 marker",
    "structural_invariant_bullet": "固定 bullet",
    "evidence_sequence_unranked": "无 rank evidence sequence",
    "structural_explicit_rank_before_city": "structural explicit rank → city",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def f(value: Any) -> float:
    return float(value)


def i(value: Any) -> int:
    return int(value)


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * f(value):.{digits}f}%"


def pp(value: Any, digits: int = 1) -> str:
    return f"{100.0 * f(value):+.{digits}f} pp"


def effect_ci(row: Mapping[str, Any], digits: int = 3) -> str:
    return (
        f"{f(row['mean_effect']):+.{digits}f} "
        f"[{f(row['ci_low']):+.{digits}f}, {f(row['ci_high']):+.{digits}f}]"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{esc(cls)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def primer(title: str, what: str, how: str, example: str) -> str:
    return (
        f'<div class="figure-primer" data-for="{esc(title)}">'
        '<div class="figure-primer-header">先看懂这张图</div>'
        '<div class="figure-primer-grid">'
        f'<p><strong>这张图画什么。</strong>{what}</p>'
        f'<p><strong>怎么读。</strong>{how}</p>'
        f'<p class="primer-example"><strong>一个例子。</strong>{example}</p>'
        "</div></div>"
    )


def conclusion(label: str, text: str, *, boundary: bool = False) -> str:
    cls = "conclusion-line boundary" if boundary else "conclusion-line"
    return f'<div class="{cls}"><strong>{esc(label)}。</strong>{text}</div>'


def extract_reference_css(path: Path) -> str:
    match = re.search(r"<style>(.*?)</style>", path.read_text(encoding="utf-8"), flags=re.S)
    if not match:
        raise ValueError(f"Could not extract style from {path}")
    return match.group(1)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def blend(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def head_map_svg(
    rows: list[dict[str, str]],
    title: str,
    top_outline: int = 128,
    *,
    value_field: str = "discovery_selection_value",
    value_label: str = "display value",
) -> str:
    layers = max(i(row["layer"]) for row in rows) + 1
    heads = max(i(row["head"]) for row in rows) + 1
    by_head = {(i(row["layer"]), i(row["head"])): row for row in rows}
    scores = [max(0.0, f(row[value_field])) for row in rows]
    cap = max(quantile(scores, 0.99), 1e-12)
    x0, y0, cw, ch = 74, 48, 25, 13
    width, height = x0 + heads * cw + 116, y0 + layers * ch + 74
    parts = [
        f'<svg class="head-map" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        f'<text x="{x0}" y="22" class="heat-title">{esc(title)}</text>',
    ]
    for head in range(heads):
        if head % 4 == 0:
            parts.append(f'<text x="{x0 + (head + .5) * cw:.1f}" y="40" text-anchor="middle" class="heat-x">H{head}</text>')
    for layer in range(layers):
        if layer % 2 == 0:
            parts.append(f'<text x="{x0-8}" y="{y0 + layer*ch + 10}" text-anchor="end" class="heat-row">L{layer}</text>')
        for head in range(heads):
            row = by_head[(layer, head)]
            score = max(0.0, f(row[value_field]))
            normalized = math.sqrt(min(score / cap, 1.0))
            color = blend((241, 245, 249), (15, 118, 110), normalized)
            rank = i(row["discovery_rank"])
            stroke = "#b42318" if rank <= top_outline else "#ffffff"
            sw = 0.85 if rank <= top_outline else 0.25
            x, y = x0 + head * cw, y0 + layer * ch
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{color}" stroke="{stroke}" stroke-width="{sw}">'
                f'<title>L{layer}H{head}; {esc(value_label)}={score:.6g}; frozen selection rank={rank}</title></rect>'
            )
    legend_x = x0 + heads * cw + 24
    for step in range(40):
        t = step / 39
        parts.append(f'<rect x="{legend_x}" y="{y0 + (39-step)*5}" width="16" height="5" fill="{blend((241,245,249),(15,118,110),math.sqrt(t))}"/>')
    parts.append(f'<text x="{legend_x+24}" y="{y0+8}" class="heat-x">≥99%</text>')
    parts.append(f'<text x="{legend_x+24}" y="{y0+198}" class="heat-x">0</text>')
    parts.append(f'<text x="{x0 + heads*cw/2}" y="{height-18}" text-anchor="middle" class="axis-label">Attention head h</text>')
    parts.append(f'<text transform="translate(18 {y0+layers*ch/2}) rotate(-90)" text-anchor="middle" class="axis-label">Transformer layer ℓ</text>')
    parts.append("</svg>")
    return "".join(parts)


def split_gemma_head_map_svg(rows: list[dict[str, str]]) -> str:
    """Compact two-column layer×head map for Gemma's 42×8 attention grid."""
    layers = max(i(row["layer"]) for row in rows) + 1
    heads = max(i(row["head"]) for row in rows) + 1
    require((layers, heads) == (42, 8), "Unexpected Gemma attention grid")
    by_head = {(i(row["layer"]), i(row["head"])): row for row in rows}
    scores = [max(0.0, f(row["score"])) for row in rows]
    cap = max(quantile(scores, 0.99), 1e-12)
    x_starts = (70, 404)
    y0, cw, ch = 62, 30, 13
    panel_layers = ((0, 21), (21, 42))
    width, height = 760, 410
    parts = [
        f'<svg class="head-map gemma-split-map" viewBox="0 0 {width} {height}" role="img" aria-label="Gemma4-E4B adjacent_rank_after_city exact-P0 targeted retrieval head map split by layer range">',
        '<title>Gemma4-E4B adjacent_rank_after_city exact-P0 target attention mass</title>',
    ]
    for panel, ((layer_start, layer_end), x0) in enumerate(zip(panel_layers, x_starts)):
        parts.append(f'<text x="{x0}" y="24" class="heat-title">L{layer_start}–L{layer_end-1}</text>')
        for head in range(heads):
            parts.append(f'<text x="{x0 + (head + .5) * cw:.1f}" y="52" text-anchor="middle" class="heat-x">H{head}</text>')
        for local_layer, layer in enumerate(range(layer_start, layer_end)):
            if layer % 5 == 0 or layer in (layer_start, layer_end - 1):
                parts.append(f'<text x="{x0-8}" y="{y0 + local_layer*ch + 10}" text-anchor="end" class="heat-row">L{layer}</text>')
            for head in range(heads):
                row = by_head[(layer, head)]
                score = max(0.0, f(row["score"]))
                normalized = math.sqrt(min(score / cap, 1.0))
                color = blend((27, 16, 61), (253, 231, 37), normalized)
                rank = i(row["rank"])
                stroke = "#ff6b57" if rank <= 8 else "#243348"
                sw = 1.35 if rank <= 8 else 0.55
                x, y = x0 + head * cw, y0 + local_layer * ch
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{color}" stroke="{stroke}" stroke-width="{sw}">'
                    f'<title>L{layer}H{head}; P0 target-mass score={score:.6f}; rank={rank}</title></rect>'
                )
        parts.append(f'<text x="{x0 + heads*cw/2}" y="{height-42}" text-anchor="middle" class="axis-label">Attention head h</text>')
    legend_x, legend_y = 694, 66
    for step in range(40):
        t = step / 39
        parts.append(f'<rect x="{legend_x}" y="{legend_y + (39-step)*5}" width="15" height="5" fill="{blend((27,16,61),(253,231,37),math.sqrt(t))}"/>')
    parts.append(f'<text x="{legend_x+21}" y="{legend_y+8}" class="heat-x">≥99%</text>')
    parts.append(f'<text x="{legend_x+21}" y="{legend_y+198}" class="heat-x">0</text>')
    parts.append(f'<text transform="translate(20 {y0+21*ch/2}) rotate(-90)" text-anchor="middle" class="axis-label">Transformer layer ℓ</text>')
    parts.append('</svg>')
    return ''.join(parts)


def hypergeom_sf(universe: int, left_size: int, right_size: int, observed: int) -> float:
    """Exact one-sided P[X >= observed] under uniform set overlap."""
    denominator = math.comb(universe, right_size)
    upper = min(left_size, right_size)
    numerator = sum(
        math.comb(left_size, overlap) * math.comb(universe - left_size, right_size - overlap)
        for overlap in range(observed, upper + 1)
        if 0 <= right_size - overlap <= universe - left_size
    )
    return numerator / denominator


def shared_head_spearman(
    shared: set[tuple[int, int]],
    broad_ranks: dict[tuple[int, int], int],
    targeted_ranks: dict[tuple[int, int], int],
) -> float | None:
    """Spearman rho for the relative ordering of heads shared by two Top-K banks."""
    if len(shared) < 2:
        return None
    broad_order = sorted(shared, key=lambda head: broad_ranks[head])
    targeted_order = sorted(shared, key=lambda head: targeted_ranks[head])
    broad_relative = {head: rank for rank, head in enumerate(broad_order, 1)}
    targeted_relative = {head: rank for rank, head in enumerate(targeted_order, 1)}
    squared_rank_distance = sum(
        (broad_relative[head] - targeted_relative[head]) ** 2 for head in shared
    )
    n_shared = len(shared)
    return 1.0 - 6.0 * squared_rank_distance / (n_shared * (n_shared**2 - 1))


def bank_overlap_metrics(
    label: str,
    broad: set[tuple[int, int]],
    targeted: set[tuple[int, int]],
    universe: int,
    broad_ranks: dict[tuple[int, int], int] | None = None,
    targeted_ranks: dict[tuple[int, int], int] | None = None,
) -> dict[str, Any]:
    intersection = broad & targeted
    expected = len(broad) * len(targeted) / universe
    spearman = None
    if broad_ranks is not None and targeted_ranks is not None:
        require(intersection <= broad_ranks.keys(), f"Missing broad ranks for {label}")
        require(intersection <= targeted_ranks.keys(), f"Missing targeted ranks for {label}")
        spearman = shared_head_spearman(intersection, broad_ranks, targeted_ranks)
    return {
        "label": label,
        "broad_size": len(broad),
        "targeted_size": len(targeted),
        "intersection": len(intersection),
        "expected": expected,
        "broad_retention": len(intersection) / len(broad),
        "chance_retention": len(targeted) / universe,
        "jaccard": len(intersection) / len(broad | targeted),
        "enrichment": len(intersection) / expected,
        "p_value": hypergeom_sf(universe, len(broad), len(targeted), len(intersection)),
        "shared_spearman": spearman,
        "heads": sorted(intersection),
        "universe": universe,
    }


def overlap_svg(rows: list[dict[str, Any]]) -> str:
    width, left, right, top, row_gap = 840, 245, 34, 54, 43
    plot_width = width - left - right
    height = top + row_gap * len(rows) + 50
    parts = [
        f'<svg class="chart bank-overlap" viewBox="0 0 {width} {height}" role="img" aria-label="Broad retrieval and targeted retrieval head-bank overlap">',
        '<title>Non-thinking broad bank retained inside Native-thinking targeted bank</title>',
        '<rect x="0" y="0" width="840" height="100%" rx="12" fill="#fbfaf6"/>',
        '<rect x="510" y="17" width="18" height="8" fill="#0f766e"/><text x="534" y="25" class="chart-axis">observed broad-bank retention</text>',
        '<rect x="685" y="17" width="18" height="8" fill="#cbd5e1"/><text x="709" y="25" class="chart-axis">random expectation</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-34}" stroke="#d8d5ca" stroke-width="0.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-17}" text-anchor="middle" class="chart-axis">{tick:.0%}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_gap
        observed = f(row["broad_retention"])
        expected = f(row["chance_retention"])
        parts.append(f'<text x="{left-12}" y="{y+15}" text-anchor="end" class="chart-model">{esc(row["label"])}</text>')
        parts.append(f'<rect x="{left}" y="{y+15}" width="{expected*plot_width:.1f}" height="9" rx="4" fill="#cbd5e1"><title>random expectation={expected:.3%}</title></rect>')
        parts.append(f'<rect x="{left}" y="{y}" width="{observed*plot_width:.1f}" height="12" rx="5" fill="#0f766e"><title>observed={observed:.3%}; overlap={row["intersection"]}</title></rect>')
        parts.append(f'<text x="{min(left+observed*plot_width+7,width-46):.1f}" y="{y+10}" class="chart-value">{observed:.1%}</text>')
    parts.append(f'<text x="{left+plot_width/2:.1f}" y="{height-2}" text-anchor="middle" class="axis-label">Fraction of the Non-thinking broad bank also present in the Native-thinking bank</text>')
    parts.append('</svg>')
    return ''.join(parts)


def effective_needle_support(masses: Sequence[float]) -> float:
    """Inverse-Simpson effective number of attended needle spans."""

    total = sum(masses)
    squared = sum(value * value for value in masses)
    return total * total / squared if squared > 0 else 0.0


def shared_broad_attention_svg(
    gallery: Mapping[str, Any],
    shared_heads: set[tuple[int, int]],
) -> str:
    """Render pre-registered Non-thinking answer-query maps for shared heads."""

    records = [
        record for record in gallery["records"]
        if (
            i(record["selection"]["layer"]),
            i(record["selection"]["head"]),
        ) in shared_heads
    ]
    records.sort(key=lambda record: (
        i(record["selection"]["frozen_head_rank"]),
        i(record["selection"]["gold_count"]),
    ))
    require(records, "No Non-thinking attention-gallery records for shared heads")
    max_needle = max(
        f(span["attention_mass"])
        for record in records
        for span in record["attention"]["needle_rows"]
    )
    max_ordinal = max(i(record["selection"]["gold_count"]) for record in records)
    left, top, cw, ch, right = 184, 76, 67, 36, 210
    plot_width = max_ordinal * cw
    width = left + plot_width + right
    height = top + len(records) * ch + 76
    parts = [
        f'<svg class="chart shared-broad-map" viewBox="0 0 {width} {height}" role="img" aria-labelledby="shared-broad-title shared-broad-desc">',
        '<title id="shared-broad-title">Shared Qwen heads at the Non-thinking answer query</title>',
        '<desc id="shared-broad-desc">Rows are two Non-thinking broad-bank heads that also enter both Native-thinking formal banks, crossed with fixed confirmation prompts N equals 3, 6, and 9. Columns are active needle spans. Color and text show raw attention mass.</desc>',
        '<text x="22" y="27" class="chart-title">Non-thinking answer query · shared-head needle attention</text>',
        '<text x="22" y="48" class="chart-axis">fixed seed 1254; prompts N=3/6/9 were chosen before inspecting these attention rows</text>',
    ]
    for ordinal in range(1, max_ordinal + 1):
        x = left + (ordinal - .5) * cw
        parts.append(f'<text x="{x:.1f}" y="68" text-anchor="middle" class="heat-x">N{ordinal}</text>')
    for row_index, record in enumerate(records):
        selection = record["selection"]
        spans = {i(span["slot_index"]): span for span in record["attention"]["needle_rows"]}
        masses = [f(spans[ordinal]["attention_mass"]) for ordinal in sorted(spans)]
        neff = effective_needle_support(masses)
        active_mass = f(record["attention"]["category_mass"]["active_needle"])
        y = top + row_index * ch
        label = (
            f'L{i(selection["layer"])}H{i(selection["head"])} · '
            f'N={i(selection["gold_count"])}'
        )
        parts.append(f'<text x="{left-10}" y="{y+23}" text-anchor="end" class="chart-model">{esc(label)}</text>')
        for ordinal in range(1, max_ordinal + 1):
            x = left + (ordinal - 1) * cw
            if ordinal not in spans:
                parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="#eef1f3" stroke="#ffffff" stroke-width="1"/>')
                continue
            span = spans[ordinal]
            value = f(span["attention_mass"])
            normalized = math.sqrt(min(value / max(max_needle, 1e-12), 1.0))
            fill = blend((242, 246, 246), (15, 118, 110), normalized)
            text_fill = "#ffffff" if normalized > .58 else "#172033"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="#ffffff" stroke-width="1">'
                f'<title>{esc(label)}; N{ordinal} {esc(span["city"])}; raw mass={value:.6f}</title></rect>'
            )
            parts.append(f'<text x="{x+cw/2:.1f}" y="{y+23}" text-anchor="middle" style="fill:{text_fill};font-size:11px;font-weight:650">{100*value:.1f}%</text>')
        parts.append(
            f'<text x="{left+plot_width+15}" y="{y+16}" class="chart-value">'
            f'needle total {100*active_mass:.1f}%</text>'
        )
        parts.append(
            f'<text x="{left+plot_width+15}" y="{y+31}" class="chart-axis">'
            f'effective support {neff:.2f} / {len(masses)}</text>'
        )
    parts.append(f'<text x="{left+plot_width/2:.1f}" y="{height-26}" text-anchor="middle" class="axis-label">Active needle span ordinal</text>')
    parts.append('<text x="22" y="{0}" class="chart-axis">Cell = raw share of the complete answer-query attention row; effective support = (Σm)²/Σm².</text>'.format(height-7))
    parts.append('</svg>')
    return ''.join(parts)


def shared_native_concentration_svg(
    broad_ranks: Mapping[tuple[int, int], int],
    adjacent_rows: Sequence[Mapping[str, str]],
    same_rows: Sequence[Mapping[str, str]],
    adjacent_bank: set[tuple[int, int]],
    same_bank: set[tuple[int, int]],
) -> str:
    """Compare city-pre concentration for every formal-bank overlap head."""

    adjacent = {
        (i(row["layer"]), i(row["head"])): row for row in adjacent_rows
    }
    same = {(i(row["layer"]), i(row["head"])): row for row in same_rows}
    heads = sorted(
        (set(broad_ranks) & (adjacent_bank | same_bank)),
        key=lambda head: broad_ranks[head],
    )
    require(heads, "No broad/formal causal overlap heads")
    columns = (
        ("adjacent-after", "target / all needles", adjacent, adjacent_bank, "discovery_target_source_relative_attention_mass"),
        ("adjacent-after", "target top-1", adjacent, adjacent_bank, "discovery_target_source_attention_top1"),
        ("same-unit-before", "target / all needles", same, same_bank, "discovery_target_source_relative_attention_mass"),
        ("same-unit-before", "target top-1", same, same_bank, "discovery_target_source_attention_top1"),
    )
    left, top, cw, ch, right = 185, 100, 166, 25, 56
    width = left + len(columns) * cw + right
    height = top + len(heads) * ch + 65
    parts = [
        f'<svg class="chart shared-native-map" viewBox="0 0 {width} {height}" role="img" aria-labelledby="shared-native-title shared-native-desc">',
        '<title id="shared-native-title">Native city-pre attention concentration of Qwen heads shared with the Non-thinking broad bank</title>',
        '<desc id="shared-native-desc">Rows are Non-thinking broad Top-32 heads that also enter at least one formal Native Top-128 bank. Columns report target share among all needle attention and the probability that the correct target is the most attended needle. Gray cells mean that head is not a member of that grammar bank.</desc>',
        '<text x="22" y="27" class="chart-title">Native city-pre · attention concentration of shared formal-bank heads</text>',
        '<text x="22" y="48" class="chart-axis">colored cells are formal membership; gray = not selected into that grammar bank</text>',
    ]
    for column_index, (grammar, metric, _, _, _) in enumerate(columns):
        x = left + (column_index + .5) * cw
        parts.append(f'<text x="{x:.1f}" y="72" text-anchor="middle" class="chart-model">{esc(grammar)}</text>')
        parts.append(f'<text x="{x:.1f}" y="89" text-anchor="middle" class="heat-x">{esc(metric)}</text>')
    for row_index, head in enumerate(heads):
        y = top + row_index * ch
        label = f'#{broad_ranks[head]} · L{head[0]}H{head[1]}'
        parts.append(f'<text x="{left-10}" y="{y+17}" text-anchor="end" class="chart-model">{label}</text>')
        for column_index, (_, metric, lookup, membership, field) in enumerate(columns):
            x = left + column_index * cw
            if head not in membership:
                parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="#eef1f3" stroke="#ffffff" stroke-width="1"><title>{label}; not in this formal bank</title></rect>')
                parts.append(f'<text x="{x+cw/2:.1f}" y="{y+17}" text-anchor="middle" class="chart-axis">—</text>')
                continue
            row = lookup[head]
            value = f(row[field])
            normalized = math.sqrt(min(max(value, 0.0), 1.0))
            fill = blend((241, 245, 249), (15, 118, 110), normalized)
            text_fill = "#ffffff" if normalized > .62 else "#172033"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="#ffffff" stroke-width="1">'
                f'<title>{label}; {esc(metric)}={value:.6f}; Native rank={i(row["discovery_rank"])}</title></rect>'
            )
            parts.append(f'<text x="{x+cw/2:.1f}" y="{y+17}" text-anchor="middle" style="fill:{text_fill};font-size:11px;font-weight:650">{100*value:.1f}%</text>')
    parts.append(f'<text x="{left+len(columns)*cw/2:.1f}" y="{height-19}" text-anchor="middle" class="axis-label">Attention concentration at the registered city_pre_d1 query</text>')
    parts.append('</svg>')
    return ''.join(parts)


def dose_svg(qwen_rows: list[dict[str, Any]], gemma_rows: list[dict[str, Any]]) -> str:
    panels = (("Qwen3-8B", qwen_rows), ("Gemma4-E4B", gemma_rows))
    parts = [
        '<svg class="chart" viewBox="0 0 1060 400" role="img" aria-labelledby="dose-title dose-desc">',
        '<title id="dose-title">Native-thinking targeted-retrieval confirmation dose response</title>',
        '<desc id="dose-desc">Selected-bank failure, pooled random-control failure, and selected-minus-random difference across nested Top-K.</desc>',
    ]
    for panel_index, (model, rows) in enumerate(panels):
        ox = 24 + panel_index * 520
        parts.append(f'<rect x="{ox}" y="14" width="496" height="352" rx="16" fill="#fbfaf6" stroke="#d9d6cd"/>')
        parts.append(f'<text x="{ox+24}" y="46" class="chart-title">{esc(model)} · frozen confirmation</text>')
        x_start, x_end, y_top, y_bottom = ox + 70, ox + 456, 72, 310
        for tick in range(0, 101, 25):
            y = y_bottom - (y_bottom-y_top) * tick/100
            parts.append(f'<line x1="{x_start}" y1="{y:.1f}" x2="{x_end}" y2="{y:.1f}" stroke="#e2e0d9"/>')
            parts.append(f'<text x="{x_start-9}" y="{y+4:.1f}" text-anchor="end" class="chart-axis">{tick}%</text>')
        xs = [x_start + (x_end-x_start)*idx/max(1,len(rows)-1) for idx in range(len(rows))]
        series = (
            ("selected", "selected_failure_rate", "#0f766e", ""),
            ("random", "random_failure_rate", "#98a2b3", ""),
            ("selected−random", "selected_minus_random_failure_rate", "#b54708", "6 4"),
        )
        for label, key, color, dash in series:
            points = []
            for x, row in zip(xs, rows):
                y = y_bottom - (y_bottom-y_top)*f(row[key])
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
            for x, row in zip(xs, rows):
                y = y_bottom - (y_bottom-y_top)*f(row[key])
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"><title>{label}; K={row["bank_size"]}; {pct(row[key])}</title></circle>')
        for x, row in zip(xs, rows):
            parts.append(f'<text x="{x:.1f}" y="330" text-anchor="middle" class="chart-axis">K{row["bank_size"]}</text>')
        for idx, (label, _, color, dash) in enumerate(series):
            lx = ox + 68 + idx*138
            parts.append(f'<line x1="{lx}" y1="352" x2="{lx+24}" y2="352" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
            parts.append(f'<text x="{lx+30}" y="356" class="chart-axis">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denom = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denom
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/denom
    return max(0.0,center-half), min(1.0,center+half)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def seed_equal_mean(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        raw = row.get(metric)
        if raw in {None, ""}:
            continue
        if str(raw).strip().lower() in {"true", "false"}:
            value = float(truth(raw))
        else:
            value = f(raw)
        by_seed.setdefault(i(row["seed"]), []).append(value)
    require(by_seed, f"No finite rows for token-ablation metric {metric}")
    seed_means = [sum(values) / len(values) for values in by_seed.values()]
    return sum(seed_means) / len(seed_means)


def load_token_ablation_evidence(root: Path, model: str) -> dict[str, Any]:
    """Load the registered 20-discovery/10-confirmation source-blank evidence."""

    layouts = {
        "Qwen3-8B": {
            "targeting": "targeting_adj_citypre_k128_confirmation_v1",
            "trace": "answer_tracebank_top32_confirmation_all20_v2",
            "prompt": "answer_promptbank_top32_confirmation_all20_v2",
            "trace_plan": "answer_trace_items_top32_plan_all20_v2",
            "prompt_plan": "answer_prompt_records_top32_plan_all20_v2",
            "target_seed_count": 10,
        },
        "Gemma4-E4B": {
            "targeting": "targeting_adj_p0_k8_confirmation_v1",
            "trace": "answer_tracebank_top32_confirmation_all20_v1",
            "prompt": "answer_promptbank_top32_confirmation_all20_v1",
            "trace_plan": "answer_trace_items_top32_plan_all20_v1",
            "prompt_plan": "answer_prompt_records_top32_plan_all20_v1",
            "target_seed_count": 9,
        },
    }
    require(model in layouts, f"Unknown token-ablation model: {model}")
    layout = layouts[model]
    targeting_dir = root / str(layout["targeting"])
    trace_dir = root / str(layout["trace"])
    prompt_dir = root / str(layout["prompt"])
    trace_plan_dir = root / str(layout["trace_plan"])
    prompt_plan_dir = root / str(layout["prompt_plan"])
    analysis_rel = Path("analysis_registered_v1")

    target_audit = read_json(targeting_dir / analysis_rel / "analysis_audit.json")
    trace_audit = read_json(trace_dir / analysis_rel / "analysis_audit.json")
    prompt_audit = read_json(prompt_dir / analysis_rel / "analysis_audit.json")
    require(target_audit.get("status") == "PASS", "Target-token analysis is not PASS")
    require(trace_audit.get("status") == "PASS", "Trace-bank answer analysis is not PASS")
    require(prompt_audit.get("status") == "PASS", "Prompt-bank answer analysis is not PASS")
    require(
        i(target_audit["seed_count"]) == i(layout["target_seed_count"]),
        f"{model} target-token registered seed count is unexpected",
    )
    for label, audit in (("trace", trace_audit), ("prompt", prompt_audit)):
        require(i(audit["seed_count"]) == 10, f"{label} answer confirmation is not 10-seed")
        require(i(audit["request_count"]) == 100, f"{label} answer confirmation is not 100-row")

    targeting_rows = read_csv(targeting_dir / analysis_rel / "token_level_detail.csv")
    targeting_rows = [
        row for row in targeting_rows
        if row.get("status") == "ok" and not row.get("matched_control_for")
    ]
    target_conditions = (
        "clean",
        "early_half_trace_blank",
        "cumulative_trace_blank",
        "recent_transition_blank",
        "full_trace_blank",
    )
    target_summary: dict[str, dict[str, Any]] = {}
    for condition in target_conditions:
        rows = [row for row in targeting_rows if row["condition"] == condition]
        require(rows, f"Missing target-token condition {condition}")
        target_summary[condition] = {
            "rows": len(rows),
            "retrieved": sum(truth(row["target_city_retrieved"]) for row in rows),
            "retrieval_rate": seed_equal_mean(rows, "target_city_retrieved"),
            "target_share": seed_equal_mean(
                rows, "bank_target_attention_share_of_gold_mass"
            ),
            "target_top1": seed_equal_mean(rows, "bank_target_top1_fraction"),
            "target_city_logp": seed_equal_mean(rows, "target_city_log_probability"),
        }

    matched_rows = read_csv(
        targeting_dir / analysis_rel / "matched_control_specificity.csv"
    )
    specificity: dict[str, dict[str, float]] = {}
    for condition in target_conditions[1:]:
        rows = [row for row in matched_rows if row["treatment"] == condition]
        require(rows, f"Missing matched-control rows for {condition}")
        specificity[condition] = {
            "retrieval": seed_equal_mean(rows, "specificity__target_city_retrieved"),
            "target_share": seed_equal_mean(
                rows,
                "specificity__bank_target_attention_share_of_gold_mass",
            ),
            "target_city_logp": seed_equal_mean(
                rows, "specificity__target_city_log_probability"
            ),
        }

    answer_rows_by_bank = {
        "trace_items": read_csv(trace_dir / analysis_rel / "token_level_detail.csv"),
        "prompt_records": read_csv(prompt_dir / analysis_rel / "token_level_detail.csv"),
    }
    answer_conditions = (
        "clean",
        "prompt_records_blank",
        "trace_all_blank",
        "prompt_and_trace_blank",
        "prompt_all_blank",
    )
    answer_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for bank, all_rows in answer_rows_by_bank.items():
        bank_summary: dict[str, dict[str, Any]] = {}
        for condition in answer_conditions:
            rows = [row for row in all_rows if row["condition"] == condition]
            require(len(rows) == 100, f"{bank}/{condition} is not 100-row")
            bank_summary[condition] = {
                "rows": len(rows),
                "exact": sum(truth(row["exact_count"]) for row in rows),
                "parsed": sum(bool(row.get("prediction")) for row in rows),
                "exact_rate": seed_equal_mean(rows, "exact_count"),
                "gold_first_logp": seed_equal_mean(
                    rows, "gold_first_answer_token_log_probability"
                ),
                "prompt_mass": seed_equal_mean(rows, "bank_prompt_records_mass_sum"),
                "trace_item_mass": seed_equal_mean(rows, "bank_trace_items_mass_sum"),
                "trace_context_mass": seed_equal_mean(
                    rows, "bank_trace_context_mass_sum"
                ),
            }
        answer_summary[bank] = bank_summary
    require(
        [answer_summary["trace_items"][c]["exact"] for c in answer_conditions]
        == [answer_summary["prompt_records"][c]["exact"] for c in answer_conditions],
        "The two frozen answer banks disagree on generated exact-count outcomes",
    )

    plan_info: dict[str, dict[str, Any]] = {}
    for bank, directory in (
        ("trace_items", trace_plan_dir),
        ("prompt_records", prompt_plan_dir),
    ):
        manifest = read_json(directory / "manifest.json")
        plan_rows = read_csv(directory / "answer_broad_head_plan.csv")
        selected = next(row for row in plan_rows if row["condition"] == "selected_bank")
        ranking_seeds = [i(seed) for seed in manifest["ranking_seeds"]]
        require(ranking_seeds == list(range(1234, 1254)), f"{bank} bank is not 20-seed")
        require(i(selected["selection_seed_count"]) == 20, f"{bank} plan records the wrong seed count")
        heads = {tuple(map(int, head)) for head in json.loads(selected["heads"])}
        require(len(heads) == 32, f"{bank} selected bank is not Top-32")
        plan_info[bank] = {
            "bank_sha256": selected["bank_sha256"],
            "heads": heads,
            "ranking_seeds": ranking_seeds,
        }
    overlap = plan_info["trace_items"]["heads"] & plan_info["prompt_records"]["heads"]

    return {
        "model": model,
        "formal_confirmation_seed_count": 10,
        "target_registered_seed_count": i(target_audit["seed_count"]),
        "target_request_count": i(target_audit["request_count"]),
        "targeting": target_summary,
        "specificity": specificity,
        "answer": answer_summary,
        "plans": plan_info,
        "answer_bank_overlap": len(overlap),
        "answer_bank_jaccard": len(overlap)
        / len(plan_info["trace_items"]["heads"] | plan_info["prompt_records"]["heads"]),
        "input_files": [
            targeting_dir / analysis_rel / "analysis_audit.json",
            targeting_dir / analysis_rel / "token_level_detail.csv",
            targeting_dir / analysis_rel / "matched_control_specificity.csv",
            trace_dir / analysis_rel / "analysis_audit.json",
            trace_dir / analysis_rel / "token_level_detail.csv",
            prompt_dir / analysis_rel / "analysis_audit.json",
            prompt_dir / analysis_rel / "token_level_detail.csv",
            trace_plan_dir / "manifest.json",
            trace_plan_dir / "answer_broad_head_plan.csv",
            prompt_plan_dir / "manifest.json",
            prompt_plan_dir / "answer_broad_head_plan.csv",
        ],
    }


def token_source_ablation_svg(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    """Two-panel, model-paired success-rate chart for source-token ablation."""

    width, height = 1120, 420
    panel_width = 465
    panel_x = (78, 635)
    plot_y, plot_h = 62, 250
    models = (
        ("Qwen3-8B", "Qwen", "#0f766e"),
        ("Gemma4-E4B", "Gemma", "#b45309"),
    )
    target_conditions = [
        ("clean", "clean"),
        ("early_half_trace_blank", "early half"),
        ("cumulative_trace_blank", "cumulative"),
        ("recent_transition_blank", "recent"),
        ("full_trace_blank", "full trace"),
    ]
    answer_conditions = [
        ("clean", "clean"),
        ("prompt_records_blank", "prompt records"),
        ("trace_all_blank", "full trace"),
        ("prompt_and_trace_blank", "full prompt + trace"),
    ]
    parts = [
        f'<svg class="chart token-source-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Token-source ablation confirmation success rates">',
        '<title>Which token states trigger targeted retrieval and support the final count?</title>',
    ]
    retrieval_panel_title = (
        "A · Teacher-forced next-city exact"
        if evidence
        and all(
            row.get("target_exact_endpoint")
            == "target_city_teacher_forced_exact"
            for row in evidence.values()
        )
        else "A · Next-city retrieval success"
    )
    for x0, title in zip(
        panel_x,
        (retrieval_panel_title, "B · Final exact-count accuracy"),
    ):
        parts.append(f'<text x="{x0}" y="25" class="chart-title">{title}</text>')
        for tick in range(0, 101, 25):
            y = plot_y + plot_h * (1 - tick / 100)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_width}" y2="{y:.1f}" stroke="#e4e7ec"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="chart-axis">{tick}%</text>')
    bar_w = 28
    step = panel_width / len(target_conditions)
    for index, (condition, label) in enumerate(target_conditions):
        group_center = panel_x[0] + step * (index + .5)
        for model_index, (model, _, color) in enumerate(models):
            center = group_center + (model_index - .5) * (bar_w + 6)
            selected = evidence[model]["targeting"][condition]["retrieval_rate"]
            value_h = plot_h * selected
            parts.append(f'<rect x="{center-bar_w/2:.1f}" y="{plot_y+plot_h-value_h:.1f}" width="{bar_w}" height="{value_h:.1f}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{center:.1f}" y="{plot_y+plot_h-value_h-6:.1f}" text-anchor="middle" class="chart-value">{100*selected:.0f}</text>')
            if condition != "clean":
                specificity = evidence[model]["specificity"][condition]["retrieval"]
                control = min(1.0, max(0.0, selected - specificity))
                control_y = plot_y + plot_h * (1 - control)
                parts.append(f'<line x1="{center-bar_w/2-1:.1f}" y1="{control_y:.1f}" x2="{center+bar_w/2+1:.1f}" y2="{control_y:.1f}" stroke="#475467" stroke-width="3"/>')
        parts.append(f'<text x="{group_center:.1f}" y="{plot_y+plot_h+21}" text-anchor="middle" class="chart-axis">{label}</text>')
    step = panel_width / len(answer_conditions)
    for index, (condition, label) in enumerate(answer_conditions):
        group_center = panel_x[1] + step * (index + .5)
        for model_index, (model, _, color) in enumerate(models):
            center = group_center + (model_index - .5) * (bar_w + 8)
            value = evidence[model]["answer"]["trace_items"][condition]["exact_rate"]
            value_h = plot_h * value
            parts.append(f'<rect x="{center-bar_w/2:.1f}" y="{plot_y+plot_h-value_h:.1f}" width="{bar_w}" height="{value_h:.1f}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{center:.1f}" y="{plot_y+plot_h-value_h-6:.1f}" text-anchor="middle" class="chart-value">{100*value:.0f}</text>')
        parts.append(f'<text x="{group_center:.1f}" y="{plot_y+plot_h+21}" text-anchor="middle" class="chart-axis">{label}</text>')
    parts.extend([
        '<rect x="80" y="370" width="14" height="9" fill="#0f766e"/><text x="101" y="379" class="chart-axis">Qwen</text>',
        '<rect x="158" y="370" width="14" height="9" fill="#b45309"/><text x="179" y="379" class="chart-axis">Gemma</text>',
        '<line x1="260" y1="374" x2="284" y2="374" stroke="#475467" stroke-width="3"/><text x="292" y="379" class="chart-axis">equal-token matched control</text>',
        '<text x="815" y="379" text-anchor="middle" class="axis-label">Blanked source group (sequence length and query position preserved)</text>',
        '</svg>',
    ])
    return "".join(parts)


def final_readout_accuracy_svg(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    """Direct behavioral effect of prompt-record and trace-state blanking."""

    width, height = 980, 390
    left, right, top, bottom = 92, 42, 55, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    models = (
        ("Qwen3-8B", "Qwen", "#0f766e"),
        ("Gemma4-E4B", "Gemma", "#b45309"),
    )
    conditions = (
        ("clean", "clean"),
        ("prompt_records_blank", "prompt records\nblank"),
        ("trace_all_blank", "full trace\nblank"),
    )
    parts = [
        f'<svg class="chart token-source-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Final exact-count accuracy after position-preserving source-token blanking">',
        '<title>Final exact-count accuracy after prompt-record or trace-state blanking</title>',
        '<text x="24" y="27" class="chart-title">Direct behavioral endpoint · greedy exact-count accuracy</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h * (1 - tick / 100)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e4e7ec"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="chart-axis">{tick}%</text>')
    group_step = plot_w / len(conditions)
    bar_w = 56
    for condition_index, (condition, label) in enumerate(conditions):
        group_center = left + group_step * (condition_index + .5)
        for model_index, (model, model_label, color) in enumerate(models):
            center = group_center + (model_index - .5) * (bar_w + 16)
            row = evidence[model]["answer"]["trace_items"][condition]
            value = f(row["exact_rate"])
            exact = i(row["exact"])
            bar_h = plot_h * value
            parts.append(
                f'<rect x="{center-bar_w/2:.1f}" y="{top+plot_h-bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="3" fill="{color}">'
                f'<title>{model_label}; {condition}; exact={exact}/100 ({100*value:.1f}%)</title></rect>'
            )
            parts.append(f'<text x="{center:.1f}" y="{top+plot_h-bar_h-8:.1f}" text-anchor="middle" class="chart-value">{exact}/100</text>')
        label_lines = label.split("\n")
        parts.append(f'<text x="{group_center:.1f}" y="{top+plot_h+28}" text-anchor="middle" class="chart-axis">{esc(label_lines[0])}</text>')
        if len(label_lines) > 1:
            parts.append(f'<text x="{group_center:.1f}" y="{top+plot_h+45}" text-anchor="middle" class="chart-axis">{esc(label_lines[1])}</text>')
    q_clean = f(evidence["Qwen3-8B"]["answer"]["trace_items"]["clean"]["exact_rate"])
    q_trace = f(evidence["Qwen3-8B"]["answer"]["trace_items"]["trace_all_blank"]["exact_rate"])
    g_clean = f(evidence["Gemma4-E4B"]["answer"]["trace_items"]["clean"]["exact_rate"])
    g_trace = f(evidence["Gemma4-E4B"]["answer"]["trace_items"]["trace_all_blank"]["exact_rate"])
    parts.extend([
        '<rect x="100" y="352" width="15" height="10" fill="#0f766e"/><text x="123" y="362" class="chart-axis">Qwen</text>',
        '<rect x="190" y="352" width="15" height="10" fill="#b45309"/><text x="213" y="362" class="chart-axis">Gemma</text>',
        f'<text x="390" y="362" class="chart-axis">full-trace effect from clean: Qwen {100*(q_trace-q_clean):+.0f} pp · Gemma {100*(g_trace-g_clean):+.0f} pp</text>',
        f'<text x="24" y="382" class="chart-note">n=100 confirmation prompts/model; sequence length, answer query and absolute positions fixed to clean values</text>',
        '</svg>',
    ])
    return "".join(parts)


def answer_source_rerouting_svg(evidence: Mapping[str, Mapping[str, Any]]) -> str:
    """Visualize prompt-vs-trace attention composition for both answer banks."""

    width, height = 1120, 580
    left, bar_x, bar_w = 30, 405, 560
    top, row_h, group_step = 58, 30, 120
    models = (("Qwen3-8B", "Qwen"), ("Gemma4-E4B", "Gemma"))
    banks = (("trace_items", "trace-items Top-32"), ("prompt_records", "prompt-records Top-32"))
    conditions = (
        ("clean", "clean"),
        ("prompt_records_blank", "prompt records blank"),
        ("trace_all_blank", "full trace blank"),
    )
    parts = [
        f'<svg class="chart token-source-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Prompt-record and trace-context attention composition after source blanking">',
        '<title>Answer-bank attention reroutes between prompt records and trace context</title>',
        '<text x="30" y="25" class="chart-title">Bank-summed source composition at the answer query</text>',
        f'<text x="{bar_x}" y="46" class="chart-axis">0%</text>',
        f'<text x="{bar_x+bar_w/2}" y="46" text-anchor="middle" class="chart-axis">source-local share</text>',
        f'<text x="{bar_x+bar_w}" y="46" text-anchor="end" class="chart-axis">100%</text>',
    ]
    row_index = 0
    for model, model_label in models:
        for bank, bank_label in banks:
            group_y = top + row_index * group_step
            parts.append(f'<text x="{left}" y="{group_y+21}" class="chart-label">{model_label} · {bank_label}</text>')
            for condition_index, (condition, condition_label) in enumerate(conditions):
                y = group_y + condition_index * row_h
                row = evidence[model]["answer"][bank][condition]
                prompt_mass = max(0.0, f(row["prompt_mass"]))
                trace_mass = max(0.0, f(row["trace_context_mass"]))
                source_total = prompt_mass + trace_mass
                prompt_share = prompt_mass / source_total if source_total else 0.0
                prompt_w = bar_w * prompt_share
                trace_w = bar_w - prompt_w
                parts.append(f'<text x="{bar_x-12}" y="{y+21}" text-anchor="end" class="chart-axis">{condition_label}</text>')
                parts.append(f'<rect x="{bar_x}" y="{y+7}" width="{prompt_w:.1f}" height="18" fill="#0f766e"/>')
                parts.append(f'<rect x="{bar_x+prompt_w:.1f}" y="{y+7}" width="{trace_w:.1f}" height="18" fill="#b45309"/>')
                parts.append(f'<text x="{bar_x+bar_w+10}" y="{y+21}" class="chart-axis">Σ={source_total:.2f}</text>')
            row_index += 1
    parts.extend([
        '<rect x="405" y="548" width="14" height="9" fill="#0f766e"/><text x="427" y="557" class="chart-axis">prompt-record mass</text>',
        '<rect x="565" y="548" width="14" height="9" fill="#b45309"/><text x="587" y="557" class="chart-axis">trace-context mass</text>',
        '<text x="755" y="557" class="chart-axis">Σ = raw mass over both source groups</text>',
        '</svg>',
    ])
    return "".join(parts)


def main_rows_at_k(rows: list[dict[str, Any]], bank_size: int, *, min_anchors: int = 10) -> list[list[str]]:
    output = []
    for row in sorted((r for r in rows if i(r["bank_size"]) == bank_size and i(r["confirmation_anchors"]) >= min_anchors), key=lambda r: -i(r["confirmation_anchors"])):
        anchors = i(row["confirmation_anchors"])
        selected = i(row["selected_failures"])/anchors
        random_rate = i(row["random_failures"])/(3*anchors)
        output.append([
            f'<code>{esc(row["grammar"])}</code><br><span class="muted">{esc(GRAMMAR_LABELS.get(str(row["grammar"]),str(row["grammar"])))}</span>',
            f'<code>{esc(row["selection_anchor_role"])}</code>',
            str(anchors),
            f'{i(row["selected_failures"])}/{anchors} · {pct(selected)}',
            f'{i(row["random_failures"])}/{3*anchors} · {pct(random_rate)}',
            pp(selected-random_rate),
        ])
    return output


def top_head_rows(ranking: list[dict[str, str]], n: int = 5) -> list[list[str]]:
    rows = sorted(ranking, key=lambda row: i(row["discovery_rank"]))[:n]
    return [[
        f'L{i(row["layer"])}H{i(row["head"])}',
        f'{f(row["discovery_selection_value"]):.4g}',
        f'{f(row["discovery_target_source_attention_mass"]):.3f}',
        pct(row["discovery_target_source_relative_attention_mass"]),
        pct(row["discovery_target_source_attention_top1"]),
        f'{f(row["discovery_source_specific_ov_write_norm"]):.3f}',
    ] for row in rows]


def load_attention_examples(asset_root: Path) -> list[dict[str, str]]:
    specs = (
        ("qwen_l24h29", "Qwen3-8B · L24H29", "exact P0 · single trace", "Qwen3-8B_adjacent_rank_after_city_L24H29_p0_attention.svg", True),
        ("qwen_l20h30", "Qwen3-8B · L20H30", "exact P0 · single trace", "Qwen3-8B_adjacent_rank_after_city_L20H30_p0_attention.svg", True),
        ("qwen_top128_city_sum", "Qwen3-8B · Top-128 Σ mass", "exact P0 · bank-summed single trace", "Qwen3-8B_adjacent_rank_after_city_Top128_p0_attention_sum.svg", False),
        ("qwen_top128_aggregate", "Qwen3-8B · Top-128 aggregate", "exact P0 · 20 discovery seeds", "Qwen3-8B_adjacent_rank_after_city_p0_needle_ordinal_by_head.svg", True),
        ("gemma_l29h4", "Gemma4-E4B · L29H4", "exact P0 · strongest single-head trace", "Gemma4-E4B_adjacent_rank_after_city_L29H4_p0_attention.svg", True),
        ("gemma_l17h2", "Gemma4-E4B · L17H2", "exact P0 · complementary single trace", "Gemma4-E4B_adjacent_rank_after_city_L17H2_p0_attention.svg", True),
        ("gemma_top6_city_sum", "Gemma4-E4B · Top-6 Σ mass", "exact P0 · bank-summed single trace", "Gemma4-E4B_adjacent_rank_after_city_Top6_p0_attention_sum.svg", False),
        ("gemma_top8_aggregate", "Gemma4-E4B · Top-8 aggregate", "exact P0 · 20 discovery seeds", "Gemma4-E4B_adjacent_rank_after_city_p0_needle_ordinal_by_head.svg", True),
    )
    examples: list[dict[str, str]] = []
    for key, label, evidence, filename, required in specs:
        path = asset_root / filename
        if not path.exists() and not required:
            continue
        svg = path.read_text(encoding="utf-8").strip()
        require(svg.startswith("<svg") and svg.endswith("</svg>"), f"Invalid SVG: {path}")
        examples.append({
            "key": key,
            "label": label,
            "grammar": "adjacent_rank_after_city",
            "evidence": evidence,
            "path": str(path),
            "svg": svg,
        })
    return examples


def attention_example_switcher(examples: list[dict[str, str]]) -> str:
    options = "".join(
        f'<option value="{esc(example["key"])}"{" selected" if index == 0 else ""}>{esc(example["label"])}</option>'
        for index, example in enumerate(examples)
    )
    panels = "".join(
        f'<div class="attention-example-panel" data-attention-example="{esc(example["key"])}" style="display:{"block" if index == 0 else "none"}">'
        f'<div class="map-meta"><strong>{esc(example["label"])}</strong><span><code>{esc(example.get("grammar", "adjacent_rank_after_city"))}</code></span><span>{esc(example["evidence"])}</span></div>'
        f'<div class="attention-example-svg">{example["svg"]}</div></div>'
        for index, example in enumerate(examples)
    )
    return (
        '<div class="attention-switcher">'
        '<label class="attention-select">切换模型 / 证据视图'
        f'<select data-attention-selector>{options}</select></label>'
        f'<div data-attention-container>{panels}</div></div>'
    )


SPAN_GEOMETRY_LABELS = {
    "full_item": "Full item",
    "marker_core": "Marker core",
    "grammar_terminal_update": "Grammar-timed tail",
    "boundary_commit": "Boundary commit",
    "retrieved_city": "Retrieved city",
}


def grammar_span_estimand(
    model_span: Mapping[str, Any],
    phase: str,
    geometry: str,
    contrast: str,
    *,
    stratum: str | None = None,
) -> Mapping[str, Any]:
    source = (
        model_span[phase]["primary_estimands"]
        if stratum is None
        else model_span[phase]["grammar_stratum_estimands"]
    )
    matches = [
        row
        for row in source
        if row["geometry"] == geometry
        and row["contrast"] == contrast
        and (stratum is None or row["stratum"] == stratum)
    ]
    require(
        len(matches) == 1,
        f"Missing grammar-span estimand {phase}/{geometry}/{contrast}/{stratum}",
    )
    return matches[0]


def grammar_span_effect_svg(
    grammar_span: Mapping[str, Mapping[str, Any]],
) -> str:
    """Discovery-to-confirmation forest plot for terminal-span localization."""
    geometries = tuple(SPAN_GEOMETRY_LABELS)
    plotted: list[Mapping[str, Any]] = []
    for model_span in grammar_span.values():
        for phase in ("discovery", "confirmation"):
            for geometry in geometries:
                plotted.append(
                    grammar_span_estimand(
                        model_span, phase, geometry, "restoration"
                    )
                )
        for geometry in geometries:
            plotted.append(
                grammar_span_estimand(
                    model_span,
                    "confirmation",
                    geometry,
                    "matched_random_specificity",
                )
            )
    low = min(-1.0, math.floor(min(f(row["ci_low"]) for row in plotted)))
    high = max(5.0, math.ceil(max(f(row["ci_high"]) for row in plotted)))
    width, height = 1120, 670
    left, right = 248, 44
    plot_width = width - left - right
    panel_top = (100, 390)
    row_gap = 38

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    def interval(
        row: Mapping[str, Any], y: float, color: str, opacity: float
    ) -> list[str]:
        x1 = sx(f(row["ci_low"]))
        x2 = sx(f(row["ci_high"]))
        xm = sx(f(row["mean_effect"]))
        tooltip = (
            f'{f(row["mean_effect"]):+.3f} '
            f'[{f(row["ci_low"]):+.3f}, {f(row["ci_high"]):+.3f}]'
        )
        return [
            f'<line x1="{x1:.1f}" x2="{x2:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2" opacity="{opacity}"/>',
            f'<line x1="{x1:.1f}" x2="{x1:.1f}" y1="{y-4:.1f}" y2="{y+4:.1f}" stroke="{color}" opacity="{opacity}"/>',
            f'<line x1="{x2:.1f}" x2="{x2:.1f}" y1="{y-4:.1f}" y2="{y+4:.1f}" stroke="{color}" opacity="{opacity}"/>',
            f'<circle cx="{xm:.1f}" cy="{y:.1f}" r="4.8" fill="{color}" opacity="{opacity}"><title>{esc(tooltip)}</title></circle>',
        ]

    parts = [
        f'<svg class="chart span-effect-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Discovery and confirmation terminal-span restoration effects for Qwen and Gemma">',
        '<title>Terminal counter carrier: discovery and confirmation effects</title>',
        '<rect x="258" y="24" width="14" height="3" fill="#98a2b3"/><circle cx="265" cy="25.5" r="4" fill="#98a2b3"/>',
        '<text x="280" y="30" class="chart-axis">Discovery restoration</text>',
        '<rect x="432" y="24" width="14" height="3" fill="#0f766e"/><circle cx="439" cy="25.5" r="4" fill="#0f766e"/>',
        '<text x="454" y="30" class="chart-axis">Confirmation restoration</text>',
        '<rect x="642" y="24" width="14" height="3" fill="#b54708"/><circle cx="649" cy="25.5" r="4" fill="#b54708"/>',
        '<text x="664" y="30" class="chart-axis">Confirmation selected − matched random</text>',
    ]
    for tick in range(int(low), int(high) + 1):
        x = sx(float(tick))
        grid_color = "#98a2b3" if tick == 0 else "#e4e7ec"
        grid_width = 1.5 if tick == 0 else 1
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="58" y2="612" stroke="{grid_color}" stroke-width="{grid_width}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="640" text-anchor="middle" class="chart-axis">{tick:+d}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="663" text-anchor="middle" class="axis-label">Effect on correct-count margin (positive = restoration)</text>'
    )

    for panel_index, model in enumerate(MODELS):
        model_span = grammar_span[model]
        top = panel_top[panel_index]
        parts.append(
            f'<text x="20" y="{top-28}" class="chart-model">{esc(model)}</text>'
        )
        parts.append(
            f'<text x="{left}" y="{top-28}" class="chart-axis">20-seed discovery → frozen 10-seed confirmation</text>'
        )
        for row_index, geometry in enumerate(geometries):
            y = top + row_index * row_gap
            if geometry == "marker_core":
                parts.append(
                    f'<rect x="8" y="{y-17}" width="{width-16}" height="34" rx="5" fill="#edf8f6"/>'
                )
            label = SPAN_GEOMETRY_LABELS[geometry]
            parts.append(
                f'<text x="{left-16}" y="{y+4}" text-anchor="end" class="chart-axis">{esc(label)}</text>'
            )
            if geometry == "marker_core":
                parts.append(
                    f'<text x="16" y="{y+4}" class="chart-axis" fill="#0f766e">FROZEN</text>'
                )
            discovery = grammar_span_estimand(
                model_span, "discovery", geometry, "restoration"
            )
            confirmation = grammar_span_estimand(
                model_span, "confirmation", geometry, "restoration"
            )
            specificity = grammar_span_estimand(
                model_span,
                "confirmation",
                geometry,
                "matched_random_specificity",
            )
            parts.extend(interval(discovery, y - 8, "#98a2b3", 0.75))
            parts.extend(interval(confirmation, y, "#0f766e", 1.0))
            parts.extend(interval(specificity, y + 8, "#b54708", 0.92))
    parts.append("</svg>")
    return "".join(parts)


def grammar_timing_svg(
    grammar_span: Mapping[str, Mapping[str, Any]],
) -> str:
    """Grammar-stratified diagnostic for marker versus terminal-tail carriers."""
    rows = (
        ("rank_after_city", "city → rank", "marker_core", "marker"),
        ("rank_after_city", "city → rank", "grammar_terminal_update", "tail"),
        ("rank_before_city", "rank → city", "marker_core", "marker"),
        ("rank_before_city", "rank → city", "grammar_terminal_update", "tail"),
    )
    values = []
    for model_span in grammar_span.values():
        for stratum, _, geometry, _ in rows:
            for contrast in ("restoration", "matched_random_specificity"):
                values.append(
                    f(
                        grammar_span_estimand(
                            model_span,
                            "confirmation",
                            geometry,
                            contrast,
                            stratum=stratum,
                        )["mean_effect"]
                    )
                )
    low = min(-1.0, math.floor(min(values)))
    high = max(4.0, math.ceil(max(values)))
    width, height = 1120, 470
    left, right = 250, 42
    plot_width = width - left - right
    panel_top = (92, 282)
    row_gap = 36

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = [
        f'<svg class="chart span-timing-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Grammar-stratified confirmation diagnostic for marker and terminal-tail restoration">',
        '<title>Grammar-timed marker and tail diagnostic</title>',
        '<rect x="310" y="20" width="18" height="8" rx="2" fill="#0f766e"/><text x="338" y="29" class="chart-axis">Restoration</text>',
        '<rect x="446" y="20" width="18" height="8" rx="2" fill="#d99058"/><text x="474" y="29" class="chart-axis">Selected − matched random</text>',
        '<text x="690" y="29" class="chart-axis">Each grammar stratum: n=5 confirmation seeds · diagnostic only</text>',
    ]
    for tick in range(int(low), int(high) + 1):
        x = sx(float(tick))
        grid_color = "#98a2b3" if tick == 0 else "#e4e7ec"
        grid_width = 1.5 if tick == 0 else 1
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="52" y2="422" stroke="{grid_color}" stroke-width="{grid_width}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="449" text-anchor="middle" class="chart-axis">{tick:+d}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="468" text-anchor="middle" class="axis-label">Mean effect on correct-count margin</text>'
    )
    for panel_index, model in enumerate(MODELS):
        model_span = grammar_span[model]
        top = panel_top[panel_index]
        parts.append(
            f'<text x="18" y="{top-26}" class="chart-model">{esc(model)}</text>'
        )
        for row_index, (stratum, grammar_label, geometry, short_label) in enumerate(rows):
            y = top + row_index * row_gap
            if row_index == 2:
                parts.append(
                    f'<line x1="18" x2="{width-18}" y1="{y-18}" y2="{y-18}" stroke="#d0d5dd"/>'
                )
            parts.append(
                f'<text x="{left-16}" y="{y+4}" text-anchor="end" class="chart-axis">{esc(grammar_label)} · {esc(short_label)}</text>'
            )
            restoration = f(
                grammar_span_estimand(
                    model_span,
                    "confirmation",
                    geometry,
                    "restoration",
                    stratum=stratum,
                )["mean_effect"]
            )
            specificity = f(
                grammar_span_estimand(
                    model_span,
                    "confirmation",
                    geometry,
                    "matched_random_specificity",
                    stratum=stratum,
                )["mean_effect"]
            )
            for value, dy, color in (
                (restoration, -6, "#0f766e"),
                (specificity, 6, "#d99058"),
            ):
                x0, xv = sx(0.0), sx(value)
                parts.append(
                    f'<rect x="{min(x0,xv):.1f}" y="{y+dy-4:.1f}" width="{abs(xv-x0):.1f}" height="8" rx="2" fill="{color}"/>'
                )
                anchor = "start" if value >= 0 else "end"
                dx = 6 if value >= 0 else -6
                parts.append(
                    f'<text x="{xv+dx:.1f}" y="{y+dy+4:.1f}" text-anchor="{anchor}" class="chart-value">{value:+.2f}</text>'
                )
    parts.append("</svg>")
    return "".join(parts)


def build_report(
    *,
    css: str,
    rep: Mapping[str, Any],
    qwen: Mapping[str, Any],
    gemma: Mapping[str, Any],
    q_adj_rank: list[dict[str, str]],
    q_same_rank: list[dict[str, str]],
    gemma_p0_rank: list[dict[str, str]],
    nonthinking_membership: list[dict[str, str]],
    nonthinking_attention_gallery: Mapping[str, Any],
    attention_examples: list[dict[str, str]],
    screen_rows: list[dict[str, str]],
    duplicates: Mapping[str, int],
    token_evidence: Mapping[str, Mapping[str, Any]],
    terminal_chain: Mapping[str, Any],
    grammar_span: Mapping[str, Mapping[str, Any]],
    generated: str,
    hashes: Mapping[str, str],
) -> str:
    q_overall = [row for row in qwen["overall"] if row["scope"] == "all_registered_grammars"]
    g_overall = [row for row in gemma["overall"] if row["scope"] == "all_registered_grammars"]
    q_primary = next(row for row in q_overall if i(row["bank_size"]) == 128)
    g_primary = next(row for row in g_overall if i(row["bank_size"]) == 8)
    g_peak = max(g_overall, key=lambda row: f(row["selected_minus_random_failure_rate"]))
    q_major = {(row["grammar"],i(row["bank_size"])):row for row in qwen["major_grammar_rows"]}
    q_adj128 = q_major[("adjacent_rank_after_city",128)]
    q_same128 = q_major[("same_unit_rank_before_city",128)]
    require(terminal_chain.get("status") == "PASS", "Terminal-chain synthesis audit failed")
    q_chain = terminal_chain["qwen"]
    g_chain = terminal_chain["gemma"]
    q_span = grammar_span["Qwen3-8B"]
    g_span = grammar_span["Gemma4-E4B"]
    for model, span in grammar_span.items():
        require(span.get("status") == "PASS", f"{model} grammar-span run is not PASS")
        require(
            span.get("discovery_seed_count") == 20
            and span.get("confirmation_seed_count") == 10,
            f"{model} grammar-span seed contract changed",
        )
        require(
            span.get("outcome_blind") is True
            and span.get("selection_rank_used") is False,
            f"{model} grammar-span plan is not outcome-blind",
        )
        require(
            span.get("discovery_selected_split_geometry") == "marker_core",
            f"{model} grammar-span discovery selection changed",
        )
        require(
            span.get(
                "discovery_selected_geometry_confirmation_descriptive_signal"
            )
            is True,
            f"{model} grammar-span confirmation signal is absent",
        )
    q_geometries = q_chain["balanced_count2_6_geometries"]
    q_legacy = q_chain["legacy_teacher_forced_bridge"]
    q_terminal = q_geometries["terminal_span"]["confirmation"]
    q_generated = q_geometries["generated_suffix_span"]["confirmation"]
    q_prefix = q_geometries["terminal_prefix_span"]["confirmation"]
    q_count9 = q_chain["count9_confirmation"]["means"]
    q_count10 = q_chain["count10_confirmation"]["means"]
    g_terminal = g_chain["confirmation"]

    def span_estimand(
        model_span: Mapping[str, Any],
        phase: str,
        geometry: str,
        contrast: str,
    ) -> Mapping[str, Any]:
        matches = [
            row
            for row in model_span[phase]["primary_estimands"]
            if row["geometry"] == geometry and row["contrast"] == contrast
        ]
        require(
            len(matches) == 1,
            f"Missing grammar-span estimand {phase}/{geometry}/{contrast}",
        )
        return matches[0]

    span_geometry_order = (
        "full_item",
        "marker_core",
        "grammar_terminal_update",
        "boundary_commit",
        "retrieved_city",
    )
    grammar_span_rows = []
    for model, model_span in (("Qwen3-8B", q_span), ("Gemma4-E4B", g_span)):
        for phase in ("discovery", "confirmation"):
            values = []
            for geometry in span_geometry_order:
                restore = span_estimand(model_span, phase, geometry, "restoration")
                specificity = span_estimand(
                    model_span, phase, geometry, "matched_random_specificity"
                )
                values.append(
                    f'{f(restore["mean_effect"]):+.2f} / '
                    f'{f(specificity["mean_effect"]):+.2f}'
                )
            grammar_span_rows.append(
                [model, phase, *values, model_span[phase]["largest_split_geometry"]]
            )
    grammar_span_effect_figure = grammar_span_effect_svg(grammar_span)
    grammar_timing_figure = grammar_timing_svg(grammar_span)
    q_geometry_rows = []
    geometry_labels = {
        "terminal_span": "terminal causal suffix",
        "generated_suffix_span": "query→terminal generated suffix",
        "terminal_prefix_span": "terminal→answer prefix",
    }
    for geometry in ("terminal_span", "generated_suffix_span", "terminal_prefix_span"):
        row = q_geometries[geometry]["confirmation"]
        q_geometry_rows.append([
            f'<code>{esc(geometry)}</code><br><span class="muted">{esc(geometry_labels[geometry])}</span>',
            effect_ci(row["targeted_terminal_nonmarker_damage"]),
            effect_ci(row["targeted_answer_damage_margin"]),
            effect_ci(row["clean_state_restoration_margin"]),
            effect_ci(row["restoration_specificity_margin"]),
            effect_ci(row["selected_state_occlusion_margin"]),
            "PASS" if row["registered_probability_utility_gate_pass"] else "FAIL",
        ])
    q_count_rows = []
    for stratum in q_generated["count_strata"]:
        means = stratum["means"]
        q_count_rows.append([
            str(i(stratum["gold_count"])),
            str(i(stratum["n_seeds"])),
            "generated suffix",
            f'{f(means["targeted_terminal_nonmarker_damage"]):+.3f}',
            f'{f(means["targeted_answer_damage__correct_count_margin"]):+.3f}',
            f'{f(means["selected_clean_state_restoration__correct_count_margin"]):+.3f}',
            f'{f(means["restoration_specificity__correct_count_margin"]):+.3f}',
        ])
    for count, means in ((9, q_count9), (10, q_count10)):
        q_count_rows.append([
            str(count),
            "5",
            "terminal suffix",
            f'{f(means["targeted_terminal_nonmarker_damage"]):+.3f}',
            f'{f(means["targeted_answer_damage__correct_count_margin"]):+.3f}',
            f'{f(means["selected_clean_state_restoration__correct_count_margin"]):+.3f}',
            f'{f(means["restoration_specificity__correct_count_margin"]):+.3f}',
        ])

    trigger_labels = {
        "clean": "clean",
        "early_half_trace_blank": "早期一半 trace blank",
        "cumulative_trace_blank": "更早累计 trace blank",
        "recent_transition_blank": "最近 transition blank",
        "full_trace_blank": "完整 trace blank",
    }
    q_token = token_evidence["Qwen3-8B"]
    g_token = token_evidence["Gemma4-E4B"]
    trigger_rows_by_model: dict[str, list[list[str]]] = {}
    for model, model_evidence in token_evidence.items():
        trigger_rows = []
        for condition in (
            "clean",
            "early_half_trace_blank",
            "cumulative_trace_blank",
            "recent_transition_blank",
            "full_trace_blank",
        ):
            row = model_evidence["targeting"][condition]
            specificity = model_evidence["specificity"].get(condition)
            trigger_rows.append([
                esc(trigger_labels[condition]),
                f'{i(row["retrieved"])}/{i(row["rows"])}',
                pct(row["retrieval_rate"]),
                pct(row["target_share"]),
                pct(row["target_top1"]),
                f'{f(row["target_city_logp"]):.3f}',
                "—" if specificity is None else pp(specificity["retrieval"]),
            ])
        trigger_rows_by_model[model] = trigger_rows
    answer_labels = {
        "clean": "clean",
        "prompt_records_blank": "prompt records blank",
        "trace_all_blank": "完整 trace blank",
        "prompt_and_trace_blank": "entire prompt + trace blank",
        "prompt_all_blank": "整个 prompt blank",
    }
    answer_primary_by_model = {
        model: model_evidence["answer"]["trace_items"]
        for model, model_evidence in token_evidence.items()
    }
    answer_rows_by_model: dict[str, list[list[str]]] = {}
    for model, answer_primary in answer_primary_by_model.items():
        rows = []
        for condition in (
            "clean",
            "prompt_records_blank",
            "trace_all_blank",
            "prompt_and_trace_blank",
            "prompt_all_blank",
        ):
            row = answer_primary[condition]
            rows.append([
                esc(answer_labels[condition]),
                f'{i(row["exact"])}/{i(row["rows"])}',
                f'{i(row["parsed"])}/{i(row["rows"])}',
                f'{f(row["gold_first_logp"]):.3f}',
            ])
        answer_rows_by_model[model] = rows
    source_mass_rows = []
    for model, model_evidence in token_evidence.items():
        for bank, label in (
            ("trace_items", "trace-items Top-32"),
            ("prompt_records", "prompt-records Top-32"),
        ):
            for condition in ("clean", "prompt_records_blank", "trace_all_blank"):
                row = model_evidence["answer"][bank][condition]
                source_mass_rows.append([
                    esc(model),
                    esc(label),
                    esc(answer_labels[condition]),
                    f'{f(row["prompt_mass"]):.2f}',
                    f'{f(row["trace_item_mass"]):.2f}',
                    f'{f(row["trace_context_mass"]):.2f}',
                ])

    rep_rows = []
    for model in MODELS:
        for endpoint, label in (("running","running index"),("final","final count")):
            row = rep[model][endpoint]
            rep_rows.append([
                esc(model), label, f'<code>{esc(row.get("site_kind",row.get("token_site")))}</code>',
                f'L{i(row["layer"])}', pct(row["confirmation_logistic_balanced_accuracy"]),
                pct(row["confirmation_ncc_balanced_accuracy"]), f'{f(row["confirmation_class_balanced_snr_db"]):+.2f} dB', str(i(row["confirmation_rows"])),
            ])

    q_dose_rows = [[
        f'K{i(row["bank_size"])}', str(i(row["confirmation_anchors"])),
        f'{i(row["selected_failures"])}/{i(row["selected_trials"])} · {pct(row["selected_failure_rate"])}',
        f'{i(row["random_failures"])}/{i(row["random_trials"])} · {pct(row["random_failure_rate"])}',
        pp(row["selected_minus_random_failure_rate"]),
    ] for row in q_overall]
    g_dose_rows = [[
        f'K{i(row["bank_size"])}', str(i(row["confirmation_anchors"])),
        pct(row["selected_failure_rate"]), pct(row["random_failure_rate"]), pp(row["selected_minus_random_failure_rate"]),
    ] for row in g_overall]
    appendix_screen = []
    for label in ("adj_p0_abs","adj_citypre_ovnorm","same_p0_abs","same_citypre_abs","same_premarker_abs"):
        row = next(row for row in screen_rows if row["candidate"] == label)
        appendix_screen.append([f'<code>{esc(label)}</code>', f'<code>{esc(row["behavior_grammar"])}</code>', f'<code>{esc(row["selection_anchor_role"])}</code>', f'{row["failures"]}/{row["n"]} · {pct(row["failure_rate"])}'])

    q_high = {
        row["grammar"]: row for row in qwen["rows"]
        if i(row["bank_size"]) == 128 and i(row["confirmation_anchors"]) >= 10
    }
    g_high = {
        row["grammar"]: row for row in gemma["rows"]
        if i(row["bank_size"]) == 8 and i(row["confirmation_anchors"]) >= 10
    }
    localizer_rows = [
        ["Qwen3-8B", '<code>adjacent_rank_after_city</code>', '<code>city_pre_d1</code>', "target-specific OV-write norm", "Top-128", str(i(q_high["adjacent_rank_after_city"]["confirmation_anchors"]))],
        ["Qwen3-8B", '<code>adjacent_rank_before_city</code>', '<code>post_marker</code>', "target attention mass", "Top-128", str(i(q_high["adjacent_rank_before_city"]["confirmation_anchors"]))],
        ["Qwen3-8B", '<code>same_unit_rank_before_city</code>', '<code>city_pre_d1</code>', "target attention mass", "Top-128", str(i(q_high["same_unit_rank_before_city"]["confirmation_anchors"]))],
        ["Gemma4-E4B", '<code>adjacent_rank_after_city</code>', '<code>p0_item_end</code>', "target attention mass", "Top-8", str(i(g_high["adjacent_rank_after_city"]["confirmation_anchors"]))],
        ["Gemma4-E4B", '<code>same_unit_rank_before_city</code>', '<code>post_marker</code>', "target attention mass", "Top-8", str(i(g_high["same_unit_rank_before_city"]["confirmation_anchors"]))],
    ]

    gemma_adj_rank = [
        row for row in gemma_p0_rank
        if row["model_label"] == "Gemma4-E4B" and row["grammar"] == "adjacent_rank_after_city"
    ]
    gemma_top_rows = [[
        f'L{i(row["layer"])}H{i(row["head"])}', f'{f(row["score"]):.3f}', str(i(row["n_seeds"]))
    ] for row in sorted(gemma_adj_rank, key=lambda row: i(row["rank"]))[:5]]

    q_broad = {
        (i(row["layer"]), i(row["head"])) for row in nonthinking_membership
        if row["model_label"] == "Qwen3-8B" and i(row["top_n"]) == 32
    }
    g_broad = {
        (i(row["layer"]), i(row["head"])) for row in nonthinking_membership
        if row["model_label"] == "Gemma4-E4B" and i(row["top_n"]) == 8
    }
    q_adj_bank = {
        (i(row["layer"]), i(row["head"])) for row in q_adj_rank if i(row["discovery_rank"]) <= 128
    }
    q_same_bank = {
        (i(row["layer"]), i(row["head"])) for row in q_same_rank if i(row["discovery_rank"]) <= 128
    }
    g_adj_bank = {
        (i(row["layer"]), i(row["head"])) for row in gemma_adj_rank if i(row["rank"]) <= 8
    }
    q_broad_ranks = {
        (i(row["layer"]), i(row["head"])): i(row["rank"])
        for row in nonthinking_membership
        if row["model_label"] == "Qwen3-8B" and i(row["top_n"]) == 32
    }
    g_broad_ranks = {
        (i(row["layer"]), i(row["head"])): i(row["rank"])
        for row in nonthinking_membership
        if row["model_label"] == "Gemma4-E4B" and i(row["top_n"]) == 8
    }
    q_adj_ranks = {
        (i(row["layer"]), i(row["head"])): i(row["discovery_rank"]) for row in q_adj_rank
    }
    q_same_ranks = {
        (i(row["layer"]), i(row["head"])): i(row["discovery_rank"]) for row in q_same_rank
    }
    g_adj_ranks = {
        (i(row["layer"]), i(row["head"])): i(row["rank"]) for row in gemma_adj_rank
    }
    q_universe = len({(i(row["layer"]), i(row["head"])) for row in q_adj_rank})
    g_universe = len({(i(row["layer"]), i(row["head"])) for row in gemma_adj_rank})
    require((len(q_broad), len(g_broad)) == (32, 8), "Unexpected Non-thinking broad-bank sizes")
    require((len(q_adj_bank), len(q_same_bank), len(g_adj_bank)) == (128, 128, 8), "Unexpected Native-thinking bank sizes")
    require((q_universe, g_universe) == (1152, 336), "Unexpected model head universes")
    overlap_rows = [
        bank_overlap_metrics(
            "Qwen · adjacent-after", q_broad, q_adj_bank, q_universe,
            q_broad_ranks, q_adj_ranks,
        ),
        bank_overlap_metrics(
            "Qwen · same-unit-before", q_broad, q_same_bank, q_universe,
            q_broad_ranks, q_same_ranks,
        ),
        bank_overlap_metrics("Qwen · either major grammar", q_broad, q_adj_bank | q_same_bank, q_universe),
        bank_overlap_metrics(
            "Gemma · adjacent-after", g_broad, g_adj_bank, g_universe,
            g_broad_ranks, g_adj_ranks,
        ),
    ]
    overlap_table_rows = [[
        esc(row["label"]), f'{i(row["broad_size"])} / {i(row["targeted_size"])}',
        str(i(row["intersection"])), f'{f(row["expected"]):.2f}', pct(row["broad_retention"]),
        pct(row["jaccard"]), f'{f(row["enrichment"]):.2f}×',
        "—" if row["shared_spearman"] is None else f'{f(row["shared_spearman"]):.3f}',
        f'{f(row["p_value"]):.2e}' if f(row["p_value"]) < .001 else f'{f(row["p_value"]):.3f}',
    ] for row in overlap_rows]
    overlap_head_lists = ''.join(
        f'<p><strong>{esc(row["label"])}：</strong> '
        + ', '.join(f'<code>L{layer}H{head}</code>' for layer, head in row["heads"])
        + '</p>'
        for row in overlap_rows
    )
    q_adj_overlap = q_broad & q_adj_bank
    q_same_overlap = q_broad & q_same_bank
    gallery_heads = {
        (i(row["layer"]), i(row["head"]))
        for row in nonthinking_attention_gallery["selection"]["heads"]
    }
    gallery_shared_heads = gallery_heads & q_adj_overlap & q_same_overlap
    require(
        gallery_shared_heads == {(27, 18), (23, 13)},
        f"Unexpected pre-registered gallery overlap heads: {sorted(gallery_shared_heads)}",
    )
    shared_broad_map = shared_broad_attention_svg(
        nonthinking_attention_gallery,
        gallery_shared_heads,
    )
    shared_native_map = shared_native_concentration_svg(
        q_broad_ranks,
        q_adj_rank,
        q_same_rank,
        q_adj_bank,
        q_same_bank,
    )
    q_adj_by_head = {
        (i(row["layer"]), i(row["head"])): row for row in q_adj_rank
    }
    q_same_by_head = {
        (i(row["layer"]), i(row["head"])): row for row in q_same_rank
    }
    adj_shared_target_share = [
        f(q_adj_by_head[head]["discovery_target_source_relative_attention_mass"])
        for head in q_adj_overlap
    ]
    adj_shared_top1 = [
        f(q_adj_by_head[head]["discovery_target_source_attention_top1"])
        for head in q_adj_overlap
    ]
    same_shared_target_share = [
        f(q_same_by_head[head]["discovery_target_source_relative_attention_mass"])
        for head in q_same_overlap
    ]
    same_shared_top1 = [
        f(q_same_by_head[head]["discovery_target_source_attention_top1"])
        for head in q_same_overlap
    ]
    broad_example_support = {}
    for record in nonthinking_attention_gallery["records"]:
        head = (
            i(record["selection"]["layer"]),
            i(record["selection"]["head"]),
        )
        if head not in gallery_shared_heads:
            continue
        masses = [f(row["attention_mass"]) for row in record["attention"]["needle_rows"]]
        broad_example_support[(head, i(record["selection"]["gold_count"]))] = effective_needle_support(masses)
    shared_example_head = (24, 29)
    require(
        shared_example_head in q_broad & q_adj_bank & q_same_bank,
        "L24H29 is no longer shared by all three Qwen banks",
    )
    shared_native_example = next(
        example["svg"] for example in attention_examples
        if example["key"] == "qwen_l24h29"
    )
    attention_switcher_examples = list(attention_examples)
    attention_switcher_examples.extend([
        {
            "key": "qwen_formal_causal_adjacent_after",
            "label": "Qwen3-8B · formal causal Top-128 · adjacent-after",
            "grammar": "adjacent_rank_after_city",
            "evidence": "city_pre_d1 · OV-write-ranked membership",
            "svg": head_map_svg(
                q_adj_rank,
                "Qwen adjacent_rank_after_city · formal causal Top-128 at city-pre",
                value_field="discovery_target_source_attention_mass",
                value_label="raw target attention mass",
            ),
        },
        {
            "key": "qwen_formal_causal_same_before",
            "label": "Qwen3-8B · formal causal Top-128 · same-unit-before",
            "grammar": "same_unit_rank_before_city",
            "evidence": "city_pre_d1 · target-attention-ranked membership",
            "svg": head_map_svg(
                q_same_rank,
                "Qwen same_unit_rank_before_city · formal causal Top-128 at city-pre",
                value_field="discovery_target_source_attention_mass",
                value_label="raw target attention mass",
            ),
        },
    ])
    has_city_bank_sums = {
        "qwen_top128_city_sum", "gemma_top6_city_sum"
    }.issubset({example["key"] for example in attention_switcher_examples})
    if has_city_bank_sums:
        figure2_primer = primer(
            '图 2 · 可切换的 targeted-retrieval attention heatmap',
            '下拉框包含三种互补证据：单头/整 bank 在同一条真实 N=10 trace 上的 transition×city map、跨 discovery seeds 的 ordinal×head 汇总，以及 Qwen 两个进入正式 ablation 的 city-pre Top-128 causal-bank head maps。P0 routing bank 与正式 causal bank 分开标注。',
            'City map 的横轴是 P0 transition k→k+1，纵轴是该真实样本的 N1…N10 city records 与 non-needle context；红框/红点是真正的 k+1 target。Ordinal×head 图用于看 routing 是否跨 seed 重现。Formal causal-bank map 则改用横轴=head、纵轴=layer；颜色为同一 city-pre query 的 raw target mass，红框为实际冻结的 Top-128 membership。Adjacent-after 的红框按 OV-write norm 选取，same-unit-before 的红框按 raw target mass 选取。',
            '先切换 Qwen L24H29 与 Qwen Top-128 Σ，查看单头与 P0 atlas bank 的 city routing；再切到两个 “formal causal Top-128” panel，直接核查进入 confirmation ablation 的 heads 在 layer×head 空间中的位置与 grammar-specific 重排。',
        )
        figure2_title = '图 2 · P0 routing views 与 Qwen formal causal-bank maps'
        figure2_caption = '单头与 bank-summed city panels：横轴=P0 transition k→k+1，纵轴=带真实 city 名的 prompt record region，最后一行=所有非-needle key；红框/红点标出正确 k+1 target。Top-K Σ 是每格跨 frozen P0 atlas heads 的 raw mass 之和，色标上限取 needle cells 的 observed maximum。Ordinal×head panels：横轴=discovery-ranked head，纵轴=正确 next-needle ordinal，颜色=seed-equal raw target mass。两个 Qwen formal causal panels：横轴=head h，纵轴=layer ℓ，颜色=city-pre raw target-record mass，红框=实际冻结并进入 confirmation ablation 的 Top-128；adjacent-after 的 membership 来自 OV-write ranking，same-unit-before 来自 target-mass ranking。因而颜色与红框在 adjacent-after panel 中表示两个互补量。P0 atlas bank 用于描述 routing，formal causal bank 用于 confirmation intervention；后文 selected-vs-random ablation给出 causal necessity。'
    else:
        figure2_primer = primer(
            '图 2 · 可切换的 targeted-retrieval attention heatmap',
            '下拉框包含 Qwen/Gemma 各两个真实 N=10 trace 的 single-head city map，以及两个模型各自的跨 discovery-seed ordinal×head 汇总。',
            '单 trace 图的横轴是 P0 transition k→k+1，纵轴是带真实 city 名的 prompt regions，红框/红点标出正确 k+1 target；ordinal×head 图的横轴是 discovery-ranked head，纵轴是正确 next-needle ordinal。',
            '先选 Qwen L24H29，沿着红框从 N2 移动到 N10；再选 Qwen Top-128 aggregate，检查这种 target preference 是否跨 discovery seeds 重现。',
        )
        figure2_title = '图 2 · Exact P0 query 的单头与跨-seed attention views'
        figure2_caption = '单轨迹 panel：横轴=P0 transition k→k+1，纵轴=带真实 city 名的 prompt region，红框/红点=正确 k+1 target。Ordinal×head panel：横轴=discovery-ranked head，纵轴=正确 next-needle ordinal，颜色=seed-equal raw target mass。这组图给出 routing/localization 证据；selected-vs-random causal ablation 在图 5 中独立估计行为必要性。'

    extra_css = r"""
.chart,.head-map { display:block; width:100%; height:auto; }
.chart-title,.heat-title { fill:#172033; font-size:15px; font-weight:750; }
.chart-model { fill:#172033; font-size:13px; font-weight:700; }
.chart-axis,.heat-x,.heat-row { fill:#667085; font-size:10px; }
.chart-value { fill:#344054; font-size:11px; font-weight:650; }
.axis-label { fill:#475467; font-size:11px; font-weight:650; }
.native-formula { margin:14px 0; padding:13px 16px; border:1px solid #d8d5ca; border-left:4px solid #0f766e; background:#fbfaf6; font-family:Cambria Math,Georgia,serif; overflow-x:auto; }
.native-two { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.native-card { padding:17px; border:1px solid var(--line); border-radius:8px; background:#fbfcfc; }
.native-card h3 { margin-top:0; }
.native-metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:18px 0; }
.native-metric { padding:14px; border:1px solid var(--line); border-radius:8px; background:#fff; }
.native-metric strong { display:block; color:#0f766e; font-size:24px; }
.native-metric span { color:#667085; font-size:12px; }
.muted { color:#667085; font-size:12px; }
.attention-switcher { margin:14px 0; padding:16px; border:1px solid var(--line); border-radius:10px; background:#fbfaf6; }
.attention-select { display:block; max-width:420px; color:#475467; font-size:12px; font-weight:750; letter-spacing:.02em; }
.attention-select select { display:block; width:100%; margin-top:7px; padding:9px 11px; border:1px solid #b8b5ab; border-radius:7px; background:#fff; color:#172033; font:inherit; }
.attention-example-panel { margin-top:16px; }
.attention-example-svg { overflow-x:auto; }
.attention-example-svg svg { display:block; width:100%; min-width:760px; height:auto; margin:0 auto; }
.map-meta { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:0 0 9px; color:#667085; font-size:12px; }
.map-meta strong { color:#172033; font-size:14px; }
.paper-flow { grid-template-columns:repeat(5,1fr); }
.retrieval-flow { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0 22px; }
.retrieval-flow > div { padding:15px; border:1px solid var(--line); border-radius:8px; background:#fbfcfc; }
.retrieval-flow strong { display:block; margin-bottom:6px; color:#0f766e; }
.evidence-ladder { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:20px 0 24px; }
.evidence-step { position:relative; padding:17px; border-top:3px solid #0f766e; background:#f7faf9; }
.evidence-step::after { content:"→"; position:absolute; right:-17px; top:42%; z-index:2; color:#98a2b3; font-size:24px; }
.evidence-step:last-child::after { display:none; }
.evidence-step strong { display:block; margin:5px 0 8px; color:#172033; }
.evidence-step p { margin:0; color:#475467; font-size:13px; }
.evidence-step .step-kicker { color:#0f766e; font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.11em; }
.chain-evidence { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; align-items:stretch; margin:18px 0 12px; }
.chain-node { position:relative; padding:15px 14px; border:1px solid var(--line); border-top:4px solid #0f766e; background:#fff; }
.chain-node::after { content:"→"; position:absolute; right:-17px; top:41%; z-index:3; color:#98a2b3; font-size:24px; }
.chain-node:last-child::after { display:none; }
.chain-node.partial { border-top-color:#b54708; background:#fffaf5; }
.chain-node strong { display:block; margin-bottom:6px; color:#172033; }
.chain-node span { display:block; color:#475467; font-size:12px; line-height:1.45; }
.chain-node .status { margin-top:9px; color:#0f766e; font-size:10px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
.chain-node.partial .status { color:#b54708; }
.parallel-paths { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:0 0 20px; padding:11px; border:1px dashed #b9c0ca; background:#fafbfc; }
.parallel-paths div { padding:7px 10px; color:#475467; font-size:12px; }
.parallel-paths strong { color:#344054; }
.confound-high { color:#b42318; font-weight:750; }
.confound-medium { color:#b54708; font-weight:750; }
.confound-low { color:#0f766e; font-weight:750; }
@media(max-width:900px){.native-two,.native-metrics,.retrieval-flow,.paper-flow,.evidence-ladder,.chain-evidence,.parallel-paths{grid-template-columns:1fr}.head-map,.token-source-chart{min-width:760px}.figure-scroll{overflow-x:auto}.evidence-step::after,.chain-node::after{display:none}}
"""
    ledger = "<br>".join(f"{esc(path)} · <code>{digest}</code>" for path,digest in sorted(hashes.items()))

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>Native-thinking 模型如何计数：逐步提交、定向检索、分布式状态与 count readout</title><style>{css}\n{extra_css}</style></head>
<body><article class="page"><header>
<p class="eyebrow">Realistic NIAH · Native-thinking V5 · Mechanistic analysis</p>
<h1>Native-thinking 模型如何计数：逐步提交、定向检索、分布式状态与 count readout</h1>
<p class="dek">正文沿用 Non-thinking 报告的阶段化判定顺序：先定位并因果验证 grammar-specific targeted retrieval，再检验 retrieval 后生成内容如何写入多-token trace state，最后测试该状态能否修复或破坏 answer count。Qwen 与 Gemma 使用相同的 20-seed discovery / 10-seed confirmation 合同，但结论强度分别标记：Gemma 完整通过 free-running serial bridge，Qwen 只支持 count-dependent 的 margin-level partial pathway。</p>
<div class="meta"><span>模型：Qwen3-8B / Gemma4-E4B</span><span>mother panel：300 prompts / model</span><span>counts：N=1…10</span><span>discovery：seed1234–1253</span><span>confirmation：seed1254–1263</span><span>更新：2026-08-21</span></div>
</header><nav aria-label="report sections"><a href="#summary">机制链</a><a href="#setup">实验口径</a><a href="#representation">Step 1 · commit</a><a href="#retrieval">Step 2 · targeted retrieval</a><a href="#readout">Step 3 · trace readout</a><a href="#state-write">Step 4 · state write</a><a href="#serial-chain">Step 5 · output</a><a href="#ledger">证据取舍</a><a href="#appendix">Appendix</a></nav><main>

<section id="summary"><h2>先说机制：trace 同时承载推理过程、下一次 retrieval context 与最终 count readout state</h2>
<div class="claim"><strong>中心 claim。</strong>正式实验支持一条分阶段功能链：item <em>k</em> 完成时，trace endpoint 携带可跨 seed 解码的 running-progress state；最近一次 transition 的 token states 为下一次 target selection 提供关键 context，grammar-specific heads 定向检索第 <em>k+1</em> 条 prompt record；retrieved content 随后被写入多-token generated-suffix state，answer query 再利用 trace state 产生 count。Old-HTML-aligned span decomposition 进一步把两模型的 terminal counter carrier 定位到 progress/count marker 及其到 commit 的 grammar-timed tail，而不是 city lexical span。前三阶段与 terminal counter localization 在两模型均有独立 confirmation；把 targeted bank 与 counter state 连成同一中介链时，Gemma 完整通过，Qwen 只得到 count-dependent、margin-level 的部分支持。</div>
<figure>{primer('机制图 M0 · Native-thinking 的循环、state write 与最终 readout','这是整篇报告的机制示意图。前三级构成逐 item 循环，第四级检验 retrieval 后生成内容是否写入 trace state，第五级检验该状态能否影响 final count。','从左到右读；每格下方写明主要证据。COMMIT 来自 held-out decoding，TARGET/RETRIEVE 来自 attention、source blank 与 head ablation，WRITE/OUTPUT 来自自由生成 suffix 的 multi-token residual-state restore、occlude 与 matched-random controls。','若 trace 刚写完 “4. Riga, 95.”，最近 transition context 使下一次 query 定向到第 5 条 record；retrieved Riga 内容写入生成 suffix，循环终止后 answer query 从累积 trace state 读出 count。')}<h4 class="figure-title">机制图 M0 · Commit → target → retrieve → write → output</h4>
<div class="paper-flow"><div class="paper-step"><strong>I · COMMIT</strong><span>完整 item k 的 endpoint 携带可 held-out 解码的 running index。</span><span class="operation">h(P0) → k</span></div><div class="paper-step"><strong>II · TARGET</strong><span>最近 transition context 与 surface grammar 共同约束 next-record query。</span><span class="operation">recent context + q<sub>g</sub></span></div><div class="paper-step"><strong>III · RETRIEVE</strong><span>冻结 selected bank 对生成正确第 k+1 个 city 具有集合级必要性。</span><span class="operation">selected vs matched random</span></div><div class="paper-step"><strong>IV · WRITE</strong><span>retrieved content 写入 query 后的多-token generated-suffix residual state。</span><span class="operation">full-span state restore</span></div><div class="paper-step"><strong>V · OUTPUT</strong><span>answer query 从累积 trace state 产生 final count。</span><span class="operation">restore / occlude / blank</span></div></div>
<figcaption>上图表达五组实验约束的功能顺序。P0 是统一 head-ablation start；Qwen 两个主要 grammar 的 localizer 在 <code>city_pre_d1</code> 计算。State-write bridge 只 patch targeted query 之后、answer query 之前的因果 suffix positions；source-token blank 固定序列长度和绝对位置。前三格分别对应 representation、targeting 与 bank-level necessity；后两格对应 state sufficiency / specificity 与 final count readout。最后两格的证据强度必须按模型分别报告。</figcaption></figure>
<div class="native-metrics"><div class="native-metric"><strong>{pct(rep['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy'])}</strong><span>Qwen running-index Logistic BA</span></div><div class="native-metric"><strong>{pct(rep['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy'])}</strong><span>Gemma running-index Logistic BA</span></div><div class="native-metric"><strong>{pct(q_primary['selected_minus_random_failure_rate'])}</strong><span>Qwen K128 selected−random</span></div><div class="native-metric"><strong>{pct(g_primary['selected_minus_random_failure_rate'])}</strong><span>Gemma frozen K8 selected−random</span></div></div>
{conclusion('机制总览结论','Native-thinking 的候选机制链分为五级：commit、target、retrieve、write、output。两模型的 terminal marker/tail state 都能显著恢复 final-count margin，确认 trace stream 中存在可干预 counter carrier。Gemma Top-6 的 free-running generated-suffix bridge在 discovery 与 confirmation 均闭合 targeted→state→answer；Qwen Top-128 在 query 后 multi-token state restoration 上有可重复的 count-margin 效应，但 registered probability-utility gate 与唯一性要求未通过，因此 integrated claim 仍保持较弱。')}</section>

<section id="setup"><h2>1. 任务、样本与统一实验口径</h2>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>在分析 representation 与 causal effect 前，预先注册样本、transition、query、endpoint，并分别定义 parser site、head-ranking site 与 intervention site。<span class="mini-example"><strong>直观例子：</strong><code>(Record 2: Riga, 60)</code> 中，item 1 的末 token 是 P0；Riga 首 token 前的位置是 city-pre。head 可在 city-pre 排名，同时从更早的 P0 开始持续关闭。</span></p></div>
{table(['Model','Mother panel','N=1 base cases','N≥2 duplicate exclusions','Final causal registry','Frozen confirmation'],[['Qwen3-8B','300','30',str(duplicates['qwen_extra']),'256','87'],['Gemma4-E4B','300','30',str(duplicates['gemma_extra']),'270','90']])}
<p>Discovery seeds 1234–1253 专门负责选层、选 token-site 与排 head；confirmation seeds 1254–1263 专门估计冻结方案的泛化效应。Qwen 有 14 条 N≥2 traces 重复列举同一 city，使 event→prompt-needle 映射成为一对多；最终 causal registry 保留 256 条唯一映射 anchors。Gemma 的 N=2…10 traces 均具有唯一映射，270 条 anchors 全部进入 causal registry。</p>
<h3>1.1 三个精确位置</h3><div class="native-two"><div class="native-card"><h3>P0 · <code>p0_item_end</code></h3><p>完整 item k 的最后一个真实 output token。所有正式 selected/random ablation 都从这里开始，并覆盖后续每个 decode token。</p></div><div class="native-card"><h3>city-pre · <code>city_pre_d1</code></h3><p>next city 首 token 前一个真实 output token。Qwen 两个主要 grammar 在此 ranking；正式 intervention start 统一注册为更早的 P0。</p></div></div>
<h3>1.2 因果操作与 endpoint</h3><div class="native-formula">对 t ≥ P0，在每个被选 head 上： z<sub>ℓ,h</sub>(t) ← 0（attention output projection 的 pre-O slice）；持续到 decode 结束。<br>Failure = 1[first semantic city after intervention ≠ registered next needle city].</div>
<p>清零发生在每个 head 送入 W<sub>O</sub> 之前；QK attention 权重与同层其余 heads 保持原值。持续关闭覆盖从注册起点到 decode 结束的全部后续 query，使各样本具有一致的 intervention exposure。</p>
<h3>1.3 Source-token blank：保持位置，只移除指定 token states</h3>
<div class="native-formula">对注册 source positions S：H<sup>(0)</sup>[S]←0，并在每个 decoder block ℓ 后令 H<sup>(ℓ)</sup>[S]←0。<br>Sequence length、absolute query index 与 query token 本身保持原值；干预形式为 fixed-position hidden-state zeroing。</div>
<div class="native-two"><div class="native-card"><h3>Target-trigger confirmation</h3><p><strong>Frozen bank/query：</strong>Qwen <code>city_pre_d1</code> Top-128；Gemma exact <code>p0_item_end</code> Top-8，均为正式 <code>adjacent_rank_after_city</code> bank。</p><p><strong>Cohort：</strong>confirmation seeds 1254–1263；Qwen 45 events/10 seeds，Gemma 30 events/9 eligible seeds。每个 event 运行 clean、early、cumulative、recent、full-trace treatments，并为每个 treatment 配置 3 个等 token 数 matched controls。</p><p><strong>Primary endpoint：</strong>greedy continuation 是否首先检索 registered next city；attention share、target top-1 与 gold-city log probability 为 secondary endpoints。</p></div><div class="native-card"><h3>Final-readout confirmation</h3><p><strong>Frozen banks：</strong>每个模型分别在 20 个 discovery seeds 上冻结 trace-items Top-32 与 prompt-records Top-32；这些 banks 测量 answer-query source composition。</p><p><strong>Cohort：</strong>10 个 confirmation seeds × N=1…10，共 100 prompts/model；每个 prompt 运行 clean、prompt-record blank、full-trace blank、prompt-all blank 与 full-prompt+trace blank。</p><p><strong>Primary endpoint：</strong>actual greedy exact-count accuracy；gold first-answer-token log probability 为 secondary endpoint。</p></div></div>
<div class="native-formula"><strong>Aggregation。</strong>Target-trigger 先在每个 seed 内平均 eligible events，再对 registered seeds 等权；Early-half 使用含更早 trace 的 Qwen 38 / Gemma 27 events，并与同一 event cohort 的 matched controls 配对。Final-readout 每个 seed 含十个 N 条件，先算 seed 内均值，再对 10 个 confirmation seeds 等权。</div>
{conclusion('本节结论','Head ablation 估计冻结 head bank 的集合级必要性；source-token blank 估计指定 prompt/trace positions 对 targeting 或 final readout 的状态贡献。两种实验共享 20-seed discovery / 10-seed confirmation 划分，并分别注册 query、treatment、control、aggregation 与 primary endpoint。')}</section>

<section id="representation"><h2>2. Step 1 · Commit：item endpoint 是否携带进度，answer query 是否携带最终 count</h2>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>检验 trace endpoint 是否携带可跨 seed 解码的 running index k，以及 answer query 是否携带 final count N；这一实验建立 representation-level claim。<span class="mini-example"><strong>直观例子：</strong>在 seed1258 的第 4 个 item 末端取 hidden vector；若 discovery-frozen classifier 预测 k=4，记一次 held-out 正确。</span></p></div>
<div class="native-formula">BA = (1/C) Σ<sub>c=1…C</sub> TP<sub>c</sub> /(TP<sub>c</sub>+FN<sub>c</sub>)；chance = 1/C = 10%。<br>SNR<sub>dB</sub> = 10 log<sub>10</sub>(between-centroid signal power / within-class noise power).</div>
<p>Discovery 内冻结 StandardScaler、whitened PCA-16、层与分类器；confirmation 只做一次外推。Balanced accuracy 对十个 index/count 类等权，避免高频 k 类支配结果。</p>
<figure>{primer('图 1 · Running index 与 final count 的 confirmation decoding','两块 panel 分别汇总 exact item endpoint 的 running index 与 answer query 的 final count；每个模型显示 Logistic 与 nearest-centroid classifier。','横轴是 balanced accuracy 0–100%；十类 chance 是 10%。条更长表示 held-out seed 上类别结构更容易读出；本图的 estimand 是 representation decodability。','Qwen final-count Logistic 为 100%，表示 100 个 confirmation rows 全部被正确分到十个 count 类；running-index 69.8% 表示进度结构可跨 seed 泛化，同时保留 trace 内容与位置造成的类内变化。')}<h4 class="figure-title">图 1 · Confirmation balanced accuracy</h4>{representation_svg(rep)}<figcaption>左 panel：横轴为 exact P0/item-end 上 running-index balanced accuracy；右 panel：answer-query 上 final-count balanced accuracy。绿色为 multinomial Logistic，灰色为 nearest-centroid（NCC）；每个 endpoint 的层、PCA basis 与 classifier 在 discovery 上冻结后进入 confirmation。纵向分组表示模型。</figcaption></figure>
{table(['Model','Target','Token site','Layer','Logistic BA','NCC BA','SNR','Rows'],rep_rows)}
<p>Qwen running-index Logistic/NCC 为 {pct(rep['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Qwen3-8B']['running']['confirmation_ncc_balanced_accuracy'])}，Gemma 为 {pct(rep['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Gemma4-E4B']['running']['confirmation_ncc_balanced_accuracy'])}。回答前的 final count 更清晰：Qwen 为 100.0%/99.0%，Gemma 为 71.0%/70.0%。Running SNR 为负，表示十个 centroid 的间距小于 class 内总噪声能量；结合高于 chance 的 held-out BA，支持“多维、带显著类内变化且可跨 seed 解码”的 progress representation。</p>
{conclusion('Step 1 claim','两模型都在 trace commit 上携带可跨 seed 泛化的 running-progress representation，并在 answer query 上携带 final-count representation。依据是两类 frozen classifiers 在 held-out confirmation 上均高于 10% chance，同时 running-index 的负 SNR 表明该 representation 是多维且保留显著类内变化的状态。')}</section>

<section id="retrieval"><h2>3. Step 2 · Targeted retrieval：从 target attention mass 到冻结因果 bank</h2>
<p class="lead">这一节完全对齐 Non-thinking 的 Broad retrieval 判定顺序：<strong>先看 routing，再冻结 selection，最后做 matched ablation</strong>。Native-thinking 的 estimand 位于每次 transition k→k+1，并衡量对下一条 record 的定向读取；因此基础量采用正确 next-record 的 target mass。</p>
<div class="claim"><strong>Stage-II claim。</strong>当 trace 已完成 item k，一组 grammar-conditioned heads 会优先指向 prompt 中第 k+1 条 needle record，并参与下一条 city 的生成。Target-mass heatmap刻画 routing 方向；frozen selected bank 相对 matched random bank 的 confirmation failure contrast刻画该 bank 的行为必要性。</div>
<div class="retrieval-flow"><div><strong>01 · ROUTING</strong>从 exact query 到正确 next-record span 计算 raw attention mass。</div><div><strong>02 · FREEZE</strong>由 discovery seeds 选择并冻结 grammar-specific query、metric 与 nested Top-K。</div><div><strong>03 · CAUSAL TEST</strong>从统一 P0 起持续关闭 selected/random bank，比较首次 city failure。</div></div>

<h3>3.1 先从最直观的 target attention mass 开始</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>直接测量单头在第 k 次 transition 对 prompt 第 k+1 条 record 的 attention mass，并观察 target 是否随 k 移动。<span class="mini-example"><strong>直观例子：</strong>若当前完成第 4 项，图中 4→5 这一列的红框是 N5 city；该格子的 mass 越大，表示这个 head 从当前 query 越强地回看第 5 条 prompt record。</span></p></div>
<div class="native-formula"><strong>Target attention mass。</strong> 对 event e 的正确下一条 record span R(k+1)：<br>m<sub>ℓh</sub>(e)=Σ<sub>t∈R(k+1)</sub>A<sub>ℓh</sub>(q<sub>e</sub>,t).<br><br><strong>Seed-equal ranking。</strong> S<sub>g</sub>(ℓ,h)=mean<sub>s</sub> mean<sub>e∈E(s,g)</sub>m<sub>ℓh</sub>(e).</div>
<p>先在每个 discovery seed 内平均该 grammar 的 eligible transitions，再对 seeds 等权；因此每个 discovery seed 对 ranking 的总权重相同。下面的 exact-P0 单头图是 routing 的可解释入口：例子按“eligible events 最多→N 最大→seed 最小”的预定规则确定。</p>
<div class="native-formula"><strong>Bank-summed target mass。</strong>对 frozen nested bank B<sub>K</sub> 与 target ordinal n，T<sub>K</sub>(n)=Σ<sub>h∈B<sub>K</sub></sub>S<sub>n</sub>(h)，其中 S<sub>n</sub>(h) 仍是在 seed 内平均 eligible events、再对支持 ordinal n 的 discovery seeds 等权得到的正确-record mass。图中同时给出 T<sub>K</sub>(n) 与 T<sub>K</sub>(n)/K。</div>
<figure>{figure2_primer}<h4 class="figure-title">{figure2_title}</h4>{attention_example_switcher(attention_switcher_examples)}<figcaption>{figure2_caption}</figcaption></figure>
{conclusion('3.1 claim','在 coverage 充分的 adjacent city→rank traces 中，Qwen 与 Gemma 都存在随 k 移动、对角追踪正确 next-city record 的 P0 heads。单头对角结构给出 event-level routing 证据；bank-summed mass 随 ordinal 稳定落在正确 record 上，给出跨 discovery seeds 的集合级复现。Intervention bank 的最终规格由后续 discovery behavior screen 冻结。')}

<h3>3.2 如何选择 query 位置、ranking metric 与 head bank</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>把 P0 routing 结构推进成可冻结、可外推的 causal candidate，并保持 confirmation 作为纯估计集。<span class="mini-example"><strong>直观例子：</strong><code>Riga, 95. Fifth.</code> 的 marker 在 city 后；P0 raw mass 描述下一次 routing，Qwen discovery behavior screen 则选择 next city 首 token 前的 <code>city_pre_d1</code> 与 target-specific OV write 来确定正式 bank。</span></p></div>
<p>对每个 grammar，在 discovery seeds 上比较预注册的精确 query sites；每个 candidate 的 ranking 与 attention measurement 使用同一个 exact token。Discovery behavior screen 选择 query/metric/K 后冻结配置，confirmation 独立估计该配置的效应。主文的 grammar-level claim 使用至少 10 个 confirmation anchors 的注册单元。</p>
{table(['Model','Grammar','Frozen ranking query','Ranking quantity','Bank','Confirmation anchors'],localizer_rows)}
<div class="native-formula"><strong>Target-specific OV write。</strong> w<sub>ℓh</sub>(e)=W<sub>O,ℓh</sub>[Σ<sub>t∈R(k+1)</sub>A<sub>ℓh</sub>(q<sub>e</sub>,t)V<sub>ℓh</sub>(t)]，ranking score=||w<sub>ℓh</sub>(e)||<sub>2</sub>.<br><br>Raw target mass 只问“看向哪里”；OV-write norm 还要求该 target-gated value contribution 经 W<sub>O</sub> 后实际写入 residual 的幅度较大。</div>
<p>Qwen <code>adjacent_rank_after_city</code> 的 raw mass 能显示 routing，但 target-specific OV write 对 causal head priority 更有判别力；Qwen <code>same_unit_rank_before_city</code> 则由 raw target mass 排名。Rank-before traces 在 marker 已出现后可用 <code>post_marker</code>/<code>city_pre_d1</code> 作为 marker-conditioned query；Gemma 的 adjacent city→rank 保留 P0。所有 bank 从统一 P0 起干预，因此 ranking query 与 intervention start 作为两个量分别注册。</p>

<figure>{primer('图 3a · Qwen adjacent-rank-after-city 的 frozen attention map','每个格子是一枚 attention head；颜色是 city-pre query 指向正确 next record 的 raw target mass，红框保留由 OV-write norm 选出的 Top-128。','横轴是 head index，纵轴是 transformer layer；深绿色表示 target attention mass 更高。颜色在本图内部按 99th percentile 截断；红框与颜色分别表示 OV ranking 与 raw attention。','L33H29 的 OV-write norm 排名第一，因此进入红框；它的颜色同时显示该 head 的 raw target mass，两个量共同描述 routing 与 residual write。')}<h4 class="figure-title">图 3a · Qwen adjacent_rank_after_city · city-pre target attention mass</h4><div class="figure-scroll">{head_map_svg(q_adj_rank,'Qwen adjacent_rank_after_city · city-pre target attention mass',value_field='discovery_target_source_attention_mass',value_label='raw target attention mass')}</div><figcaption>横轴=head h（0–31），纵轴=decoder layer ℓ（0–35）；颜色为 seed-equal raw target-record attention mass，深绿=更高；红框=frozen OV-ranked Top-128 membership。颜色回答“看向 target 多少”，红框回答“哪些 heads 的 target-gated post-O write 被 discovery 选入”；二者并列呈现 routing 与 write 两个互补统计量。</figcaption></figure>
<figure>{primer('图 3b · Qwen same-unit-rank-before-city 的 frozen attention-mass head map','坐标与上一图相同；颜色仍是 city-pre query 指向正确 next record 完整 span 的 raw attention mass。这里红框也由同一个 raw-mass ranking 产生。','横轴=head，纵轴=layer，深绿色=target mass 更大，红框=Top-128。两个 panel 的色标分别在各自 grammar 内缩放，颜色用于 panel 内排序。','L24H29 的 target mass=0.347：平均约 34.7% attention 直接落到正确 record span，且在全部 needle mass 中 target 占 76.4%。')}<h4 class="figure-title">图 3b · Qwen same_unit_rank_before_city · city-pre target mass</h4><div class="figure-scroll">{head_map_svg(q_same_rank,'Qwen same_unit_rank_before_city · city-pre target attention mass',value_label='raw target attention mass')}</div><figcaption>横轴=head h，纵轴=decoder layer ℓ；颜色为 seed-equal raw target-record attention mass，红框=frozen Top-128。与图 3a 的 membership 差异说明 surface grammar 与“routing-only / attention-gated write”指标都会重排 head priority。</figcaption></figure>
<figure>{primer('图 3c · Gemma adjacent-rank-after-city 的 frozen P0 head map','这张图给出 Gemma 的正式 localizer：每个格子是一枚 layer×head，颜色是 exact P0 query 指向正确 next record span 的 seed-equal raw target attention mass，红框是冻结 Top-8。','横轴是 Gemma 的 head index H0–H7，纵轴是 decoder layer；L0–20 与 L21–41 以两个版式 panel 并排。颜色越亮表示 target mass 越大；Gemma 与 Qwen 的色标分别归一，用于各自模型内排序。','L29H4 是 Gemma rank-1：它在 20 个 discovery seeds 上的 grammar-specific target-mass score 为 0.336，并与单轨迹图中随 k 移动的亮 target 相互印证。')}<h4 class="figure-title">图 3c · Gemma adjacent_rank_after_city · exact-P0 target mass</h4><div class="attention-example-svg">{split_gemma_head_map_svg(gemma_adj_rank)}</div><figcaption>横轴=head h（0–7），纵轴=decoder layer ℓ；左、右 panel 分别对应 L0–20 与 L21–41。颜色为 discovery seed-equal raw target-record attention mass，红框=frozen Top-8。该图与图 3a/3b 使用相同的“先 seed 内平均 eligible events、再等权平均 seeds”原则；Gemma 的最终 query 保留 exact P0，selection quantity 即颜色所示 target mass。</figcaption></figure>
<div class="native-two"><div><h3>Qwen adjacent-after Top-5</h3>{table(['Head','Selection score','Target mass','Target / all needles','Target top-1','OV norm'],top_head_rows(q_adj_rank))}</div><div><h3>Qwen same-unit-before Top-5</h3>{table(['Head','Selection score','Target mass','Target / all needles','Target top-1','OV norm'],top_head_rows(q_same_rank))}</div></div>
<h3>Gemma adjacent-after Top-5</h3>{table(['Head','P0 target-mass score','Discovery seeds'],gemma_top_rows)}
<p>三张 map 都由 discovery seeds 内先平均 events、再等权平均 seeds 得到；各 grammar 独立排名并冻结 bank，因此相同 head 的跨 grammar 重现是数据结果。图 3 给出 frozen localization，后续 confirmation ablation 独立估计这些 banks 的行为必要性。</p>
{conclusion('3.2 目前结论','P0 raw attention 提供统一、可解释的起点；正式 bank 则需要 grammar-specific query 与 metric。Qwen 的两个主要新增 grammar 都在 city-pre 附近局部化，但 adjacent-after 更适合 OV-write score、same-unit-before 更适合 raw mass；Gemma adjacent-after 则在 exact P0 上由 raw target mass 冻结成窄 Top-8 bank。')}

<h3 id="bank-overlap">3.3 Non-thinking broad bank 与 Native-thinking 正式 causal bank 是否复用相同 heads</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>量化两种推理模式在同一模型坐标中复用 layer-head 的程度，以及共同 heads 的优先级是否保持一致。Native 集合采用进入正式 selected-vs-random ablation 的 causal banks；图 2 P0 atlas bank 作为独立 descriptive collection。<span class="mini-example"><strong>直观例子：</strong>若 Non-thinking broad Top-32 中有 20 枚也进入某个 Native Top-128，而随机 Top-128 预期覆盖约 3.56 枚 broad heads，则存在明显的跨模式 head preference；共同 heads 的 Top-K Spearman 再量化两套 ranking 的相对次序。</span></p></div>
<div class="native-formula"><strong>Membership 与 rank agreement 分开计算。</strong>令 A 为 Non-thinking frozen broad bank，B 为进入正式 ablation 的 Native grammar-specific causal bank，U 为该模型全部 layer-head universe。Intersection=|A∩B|；Jaccard=|A∩B|/|A∪B|；random E[|A∩B|]=|A||B|/|U|；enrichment=|A∩B|/E。Top-K Spearman 在共同 heads H=A∩B 上计算：分别按每枚 head 在完整 Non-thinking 与正式 Native discovery ranking 中的顺序，对 H 内部重新编号，再计算 ρ<sub>shared</sub>=corr(rank<sub>A</sub>,rank<sub>B</sub>)。因此 overlap 衡量共同 membership，ρ<sub>shared</sub> 衡量交集内部的排序一致性；跨模式比较采用共同的 rank 尺度。Bank 并集按 membership 汇总，单侧 overlap p 值来自 exact hypergeometric test。</div>
<figure>{primer('图 4 · Broad-bank heads 在正式 causal targeted bank 中的保留率','每一行比较一个 Non-thinking broad bank 与一个进入正式 ablation 的 Native causal bank。绿色条是 broad heads 实际有多少比例也进入 causal bank；灰条是同规模 causal bank 在全部 heads 中随机抽取时的期望比例。','横轴是 broad-bank retention |A∩B|/|A|，0–100%；纵轴列出模型与 Native grammar。绿色明显长于灰色表示重合超过仅由 bank size 造成的随机期望；共同 heads 的 rank agreement 另见下表 Spearman。','Qwen same-unit-before 的绿色为 62.5%，即 broad Top-32 中 20 枚进入正式 Native Top-128；灰色期望为 128/1152=11.1%，对应 20 枚 observed 对 3.56 枚 expected。')}<h4 class="figure-title">图 4 · Non-thinking broad vs Native-thinking formal causal head-bank overlap</h4>{overlap_svg(overlap_rows)}<figcaption>横轴为 Non-thinking broad bank 被正式 Native causal bank 覆盖的比例；绿色=observed，灰色=random expectation |B|/|U|。Qwen broad=Top-32、每个主要 Native grammar=Top-128、universe=36×32=1152；“either major grammar”使用两个 Top-128 的并集（152 unique heads）汇总 membership，Spearman 则按各 grammar 的单一 ranking 分别报告。Gemma broad=Top-8、正式 Native adjacent-after=Top-8、universe=42×8=336。图 2 的 Qwen P0 Top-128 与 Gemma Top-6 属于 descriptive atlas；图 4 使用进入正式 confirmation ablation 的 causal banks。</figcaption></figure>
{table(['Comparison','|Broad| / |Targeted|','Observed overlap','Random expected','Broad retained','Jaccard','Enrichment','Top-K Spearman ρshared','Hypergeom p'],overlap_table_rows)}
<details><summary>查看逐 head 交集</summary>{overlap_head_lists}</details>
<p>Qwen adjacent-after 与 same-unit-before 分别重合 18/32 和 20/32 枚 broad heads，远高于 3.56 枚随机期望；两个 major causal banks 的并集覆盖 21/32 枚 broad heads。共同 heads 的 ρ<sub>shared</sub> 分别为 −0.172 与 0.044：两种模式复用同一 retrieval-capable pool，同时按 Native query 重新排列该 pool 内部的优先级。Gemma 重合 3/8：<code>L29H0</code>、<code>L29H2</code>、<code>L29H4</code>，随机期望为 0.19 枚；三枚共同 heads 的相对次序一致（ρ<sub>shared</sub>=1.000），该 n=3 结果按描述性证据解释。Qwen 的集合规模为 32 对 128，因此 membership 同时报告 broad retention 与 enrichment，Spearman 则解释交集内部排序。</p>
{conclusion('3.3 claim','正式 causal banks 与 Non-thinking broad bank 的 membership overlap 显著高于同规模随机期望，支持两种推理模式复用一个 retrieval-capable head pool。Qwen 共同 heads 的 ρ 接近 0，表明 Native grammar/metric 会重排该 pool 的优先级；Gemma 的 3 枚共同 heads 具有一致相对次序，该结果作为小交集的描述性证据报告。',boundary=True)}

<div id="causal"><h3>3.4 Causal ablation：持续关闭 selected bank 是否破坏下一条 city</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>检验 attention/OV ranking 所定位的 heads 是否比同规模 random heads 具有更强的行为必要性。<span class="mini-example"><strong>直观例子：</strong>K96 保留排名 1–96 的 heads；若 selected 造成 53% failure 而三个 random K96 造成 2%，则 51 pp 的 selected−random contrast 是 ranking 的 causal specificity。</span></p></div>
<div class="native-formula">对 t≥P0，在 selected/random bank 的每枚 head 上持续令 z<sub>ℓ,h</sub>(t)←0，直到 decode 结束。<br><br>Δ<sub>failure</sub>(K)=failures(selected Top-K)/n−failures(3 matched random Top-K)/(3n).</div>
<p>Selected 与 random 使用相同 K、相同 P0 起点、相同持续时间；random 首选匹配 layer composition，容量触发预注册 fallback 的 cells 使用 global same-K。Qwen 两个新 major grammar 对 confirmation seed 做 10,000 次 block bootstrap；同一 seed 的全部 anchors一起重采样。</p>
<figure>{primer('图 5 · Frozen confirmation dose response','两个 panel 分别显示 Qwen 与 Gemma 在嵌套 Top-K 下的 selected failure、pooled random failure 与二者差值。','横轴是 bank size K；纵轴是首次 semantic city failure rate。绿色=selected，灰色=三个 random controls pooled，橙色虚线=selected−random。K 的剂量比较在各模型 panel 内解释。','Qwen K128 关闭 128 个被定位 heads 后，81/87 anchors 首 city 出错；三个 random K128 合计 9/261 出错，差值 89.7 pp。')}<h4 class="figure-title">图 5 · Targeted-retrieval head-bank dose response</h4>{dose_svg(q_overall,g_overall)}<figcaption>横轴为每个 grammar 自己 ranking 的 nested Top-K，纵轴为 confirmation 首次 semantic-city failure rate。Qwen overall 按 87 个 registered anchors 加权；Gemma 按 90 个 anchors 加权。绿色 selected 与灰色 random 使用相同 K、相同 P0 起点、相同持续时长；橙线是逐 K 的 selected−random。Qwen 的 K80/96/112 adjacent-after controls 触发预注册 layer-capacity fallback，采用 global same-K；其余已注册 cells 按各自 manifest 的 layer-matched/global policy。</figcaption></figure>
<div class="chart-pair"><div><h3>Qwen · overall confirmation</h3>{table(['K','Anchors','Selected','Random pooled','Difference'],q_dose_rows)}</div><div><h3>Gemma · overall confirmation</h3>{table(['K','Anchors','Selected','Random pooled','Difference'],g_dose_rows)}</div></div>
<h4>Qwen 最终 K128：三个高样本 grammar 均出现大 contrast</h4>{table(['Grammar','Ranking query','Anchors','Selected failure','Random failure','Difference'],main_rows_at_k(qwen['rows'],128))}
<p>新补的 <code>adjacent_rank_after_city</code> 为 42/45 selected、0/135 random，seed-block bootstrap 的差值 95% CI 为 [{pct(q_adj128['seed_bootstrap_lo'])}, {pct(q_adj128['seed_bootstrap_hi'])}]；<code>same_unit_rank_before_city</code> 为 12/15、0/45，CI=[{pct(q_same128['seed_bootstrap_lo'])}, {pct(q_same128['seed_bootstrap_hi'])}]。合并其他冻结 grammar 后，Qwen overall 为 81/87={pct(q_primary['selected_failure_rate'])}，random 为 9/261={pct(q_primary['random_failure_rate'])}，差值 {pct(q_primary['selected_minus_random_failure_rate'])}；保守的独立 Wilson difference 下界仍为 {pct(q_primary['conservative_unpaired_difference_lo'])}。</p>
<h4>Gemma 冻结 K8 与 dose peak</h4>{table(['Grammar','Ranking query','Anchors','Selected failure','Random failure','Difference'],main_rows_at_k(gemma['rows'],8))}
<p>Gemma frozen K8 的 overall selected/random 为 {pct(g_primary['selected_failure_rate'])}/{pct(g_primary['random_failure_rate'])}，差值 {pct(g_primary['selected_minus_random_failure_rate'])}。完整 dose 中 K6 contrast 最大：selected {pct(g_peak['selected_failure_rate'])}、random {pct(g_peak['random_failure_rate'])}、差值 {pct(g_peak['selected_minus_random_failure_rate'])}。K6→K8 时 selected failure 与 random damage 共同变化，呈现集合级非单调 dose response；因此注册 claim 以完整 bank 的 necessity 为分析单位。</p>
{conclusion('3.4 claim','冻结 confirmation 支持 grammar-specific bank 的集合级 causal necessity：Qwen K128 的 selected−random failure contrast 为 '+pct(q_primary['selected_minus_random_failure_rate'])+'，Gemma K8 为 '+pct(g_primary['selected_minus_random_failure_rate'])+'。Selected 与 random 共享 K、P0 起点和持续时长，因此差值归因于 discovery ranking 对必要 head 集合的富集。')}</div>

<h3>3.5 什么 trace context 使 retrieval 变得 targeted</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>在已经确认 head bank 必要之后，定位它依赖的 trace source：是整个累积历史、最近一次 transition，还是只要清空等量任意 tokens 都会失败。<span class="mini-example"><strong>直观例子：</strong>模型刚写完第 4 项并准备检索第 5 条 record；清空最近一次 item/transition 的 token states 后，仍在同一 city-pre query 测量 Top-128 bank 是否指向 N5。</span></p></div>
<div class="native-formula"><strong>Position-preserving treatment。</strong>在 embedding 后与每个 decoder block 后持续令指定 source-token states 为 0；query token 与全部绝对位置固定为 clean 值。<br><strong>Specificity。</strong>Δ<sub>specific</sub>=[Y(treatment)−Y(clean)]−mean<sub>r=1…3</sub>[Y(equal-token control<sub>r</sub>)−Y(clean)].</div>
<figure>{primer('图 6 · Source-token ablation：targeting trigger 与 final readout','左 panel 并列检验 Qwen city-pre Top-128 与 Gemma exact-P0 Top-8 adjacent-after bank 依赖哪段 trace；右 panel 比较两模型最终 exact count 对 prompt records 与完整 trace 的依赖。','两个 panel 的纵轴都是 held-out confirmation success 0–100%。绿色/橙色柱分别是 Qwen/Gemma；左 panel 的灰色横线是三个等 token 数 matched controls。柱顶数字是百分比，所有结果按 seed 等权。','Source treatment 与同 event cohort 的等 token 数 control 之差是注册 specificity estimand；较大的负差值表示该 trace segment 为 targeting 提供特异状态。')}<h4 class="figure-title">图 6 · Qwen 与 Gemma：哪些 token states 支持 next retrieval 与 final count</h4><div class="figure-scroll">{token_source_ablation_svg(token_evidence)}</div><figcaption>Panel A：横轴为 trace source treatment，纵轴为 seed-equal target-city retrieval success；灰线=三个等 token 数 ordinary-position matched controls。Qwen 使用 45 个 registered events/10 seeds；Gemma 使用 {i(g_token['target_request_count'])} 个 events，来自冻结 10 个 confirmation seeds 中含合格 adjacent-after transition 的 {i(g_token['target_registered_seed_count'])} 个。Early-half 的注册 cohort 为 Qwen {i(q_token['targeting']['early_half_trace_blank']['rows'])}、Gemma {i(g_token['targeting']['early_half_trace_blank']['rows'])} events；该柱与同一位置的灰色 matched control 构成配对比较。Panel B：横轴为 answer-query source treatment，纵轴为每模型 100 个 confirmation prompts 的 exact-count accuracy。两类干预都将 sequence length、query token 与绝对 query position 固定为 clean 值。</figcaption></figure>
<details><summary>查看两模型 targeting 的完整数值表</summary><h4>Qwen3-8B</h4>{table(['Condition','Raw retrieval','Seed-equal success','Target / all-needle mass','Target top-1','Gold city log p','Treatment−control success'],trigger_rows_by_model['Qwen3-8B'])}<h4>Gemma4-E4B</h4>{table(['Condition','Raw retrieval','Seed-equal success','Target / all-needle mass','Target top-1','Gold city log p','Treatment−control success'],trigger_rows_by_model['Gemma4-E4B'])}</details>
<p>Qwen clean 为 {i(q_token['targeting']['clean']['retrieved'])}/{i(q_token['targeting']['clean']['rows'])}；recent/full treatments 分别为 {i(q_token['targeting']['recent_transition_blank']['retrieved'])}/{i(q_token['targeting']['recent_transition_blank']['rows'])} 与 {i(q_token['targeting']['full_trace_blank']['retrieved'])}/{i(q_token['targeting']['full_trace_blank']['rows'])}。Gemma clean 为 {i(g_token['targeting']['clean']['retrieved'])}/{i(g_token['targeting']['clean']['rows'])}；相应 recent/full 为 {i(g_token['targeting']['recent_transition_blank']['retrieved'])}/{i(g_token['targeting']['recent_transition_blank']['rows'])} 与 {i(g_token['targeting']['full_trace_blank']['retrieved'])}/{i(g_token['targeting']['full_trace_blank']['rows'])}。灰色 matched-control mark 给出同 event、同 token 数 ordinary positions 的配对基线；正文 claim 使用 treatment−control specificity。</p>
{conclusion('3.5 claim',f"在高样本 adjacent-after grammar 中，最近 transition 是两模型 next-record targeting 的主要局部状态来源：Qwen recent blank/control success 为 {pct(q_token['targeting']['recent_transition_blank']['retrieval_rate'])}/{pct(q_token['targeting']['recent_transition_blank']['retrieval_rate']-q_token['specificity']['recent_transition_blank']['retrieval'])}，Gemma 为 {pct(g_token['targeting']['recent_transition_blank']['retrieval_rate'])}/{pct(g_token['targeting']['recent_transition_blank']['retrieval_rate']-g_token['specificity']['recent_transition_blank']['retrieval'])}。Gemma cumulative blank/control 为 {pct(g_token['targeting']['cumulative_trace_blank']['retrieval_rate'])}/{pct(g_token['targeting']['cumulative_trace_blank']['retrieval_rate']-g_token['specificity']['cumulative_trace_blank']['retrieval'])}，进一步显示更早累计 trace 对 Gemma targeting 提供支持。",boundary=True)}
{conclusion('Step 2 总结','Target mass 先给出可解释的 next-city routing；discovery-only site/metric search 将其冻结成 grammar-specific bank；persistent selected-vs-random ablation 确认 bank-level necessity；并列 source-token blank 再定位两模型主 grammar 的 targeting context。')}</section>

<section id="readout"><h2>4. Step 3 · Final readout：source-token intervention 如何改变最终 count accuracy</h2>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>直接比较最终 count 对两个候选信息源的行为依赖：原始 prompt records 与此前循环保留的 trace-token states。<span class="mini-example"><strong>直观例子：</strong>对一个 clean 输出为 10 的样本，分别清空 prompt 中十条 city/score records 与完整 trace，再让模型自由生成答案；输出 10 的比例就是直接行为 endpoint。</span></p></div>

<h3>4.1 Primary causal test：自由生成 exact-count accuracy</h3>
<p><strong>Primary intervention 直接作用于 source-token hidden states。</strong>对注册 source positions，在 embedding 后和每个 decoder block 后持续清零 hidden states；随后保留原 answer query 并进行 greedy generation。<code>prompt_records_blank</code> 清零 prompt 中 city/score records；<code>trace_all_blank</code> 清零完整生成 trace。Sequence length、answer-query token 与 absolute positions 固定为对应 clean prompt 的值。Answer-head attention mass 在 4.2 中作为 secondary mechanism diagnostic。</p>
<div class="native-formula"><strong>Registered cohort。</strong>10 confirmation seeds（1254–1263）× N=1…10 = 100 prompts/model；每个 prompt 运行 clean、prompt-records blank 与 full-trace blank。<br><strong>Primary endpoint。</strong>Exact=1[greedy generated count = registered N]；每个 seed 先平均十个 N，再对 10 个 seeds 等权。<br><strong>Primary effects。</strong>Δ<sub>prompt</sub>=Acc(prompt-records blank)−Acc(clean)；Δ<sub>trace</sub>=Acc(full-trace blank)−Acc(clean)。</div>
<figure>{primer('图 7 · Source blank 对最终 count accuracy 的直接影响','每组柱对应一个 position-preserving source treatment；柱高是模型在 100 个 held-out confirmation prompts 上自由生成正确 count 的比例。','横轴是 clean、prompt-records blank 与 full-trace blank；纵轴是 greedy exact-count accuracy。柱顶同时给出正确样本数/100。','Qwen clean 与 prompt-record blank 都是 97/100，而 full-trace blank 是 1/100，对应 Δtrace=−96 pp；Gemma 为 70/100、70/100、12/100，对应 Δtrace=−58 pp。')}<h4 class="figure-title">图 7 · Final-readout causal effect on exact-count accuracy</h4><div class="figure-scroll">{final_readout_accuracy_svg(token_evidence)}</div><figcaption>横轴=source-token treatment；纵轴=100 个 confirmation prompts/model 上的 greedy exact-count accuracy。Qwen clean/prompt-record/full-trace 为 {i(answer_primary_by_model['Qwen3-8B']['clean']['exact'])}/{i(answer_primary_by_model['Qwen3-8B']['prompt_records_blank']['exact'])}/{i(answer_primary_by_model['Qwen3-8B']['trace_all_blank']['exact'])}；Gemma 为 {i(answer_primary_by_model['Gemma4-E4B']['clean']['exact'])}/{i(answer_primary_by_model['Gemma4-E4B']['prompt_records_blank']['exact'])}/{i(answer_primary_by_model['Gemma4-E4B']['trace_all_blank']['exact'])}。这是自由生成行为结果；attention mass 在下一节作为次级机制诊断。</figcaption></figure>
<details><summary>查看两模型 answer-query 的完整行为数值表</summary><h4>Qwen3-8B</h4>{table(['Condition','Exact count','Parsed output','Gold first-token log p'],answer_rows_by_model['Qwen3-8B'])}<h4>Gemma4-E4B</h4>{table(['Condition','Exact count','Parsed output','Gold first-token log p'],answer_rows_by_model['Gemma4-E4B'])}</details>
{conclusion('4.1 causal claim',f"在 held-out confirmation 上，full-trace hidden-state blank 对最终 count 产生直接行为效应：Qwen exact accuracy 从 {pct(answer_primary_by_model['Qwen3-8B']['clean']['exact_rate'])} 降至 {pct(answer_primary_by_model['Qwen3-8B']['trace_all_blank']['exact_rate'])}（−96 pp），Gemma 从 {pct(answer_primary_by_model['Gemma4-E4B']['clean']['exact_rate'])} 降至 {pct(answer_primary_by_model['Gemma4-E4B']['trace_all_blank']['exact_rate'])}（−58 pp）。Prompt-record blank 在两模型中均保持 clean accuracy，因此注册对比支持 trace-token states 是 final count readout 的主要行为信息源。")}

<h3>4.2 Secondary mechanism：answer heads 的 source attention composition</h3>
<p>Attention analysis用于解释上述行为效应。每个模型独立使用全部 20 个 discovery seeds（1234–1253）冻结两套 Top-32 readout banks：一套按 answer query 对 <code>trace_items</code> 的 broad score 排名，另一套按 <code>prompt_records</code> 的 broad score 排名。Qwen 两套 bank 重合 {i(q_token['answer_bank_overlap'])}/32（Jaccard={f(q_token['answer_bank_jaccard']):.3f}）；Gemma 重合 {i(g_token['answer_bank_overlap'])}/32（Jaccard={f(g_token['answer_bank_jaccard']):.3f}）。这些 banks 描述 source composition；图 7 的 count-accuracy treatment 直接作用于 source-token states。</p>
<figure>{primer('图 8 · Answer heads 的 source composition','每行显示一个模型的一套冻结 answer bank；每个小条把 prompt-record 与 trace-context 两个互斥 source group 内的 attention mass 归一到 100%。','横轴是这两个 source groups 内的相对占比；绿色=prompt records，橙色=完整 trace context。右侧 Σ 是两组 raw bank-summed attention mass 之和，因此相对 composition 与 absolute mass 同时可见。','Prompt-record blank 后 exact count 保持且 trace share 占优，表示 trace states 提供足够 readout source；trace blank 后准确率下降则定位 trace content 的行为贡献。')}<h4 class="figure-title">图 8 · Qwen / Gemma answer-bank source composition</h4><div class="figure-scroll">{answer_source_rerouting_svg(token_evidence)}</div><figcaption>每个模型各有 trace-items Top-32 与 prompt-records Top-32 两套 discovery-frozen bank；每套 bank 显示 clean、prompt-record blank、full-trace blank。横向堆叠是在 prompt-record 与 trace-context 两组内归一的 source-local composition；右侧 Σ 给出对应 raw mass 总量。图 7 给出对应的自由生成 exact-count causal effect。</figcaption></figure>
<details><summary>查看两模型 raw source-mass 明细</summary>{table(['Model','Frozen 20-seed bank','Condition','Σ prompt-record mass','Σ trace-item mass','Σ trace-context mass'],source_mass_rows)}</details>
<p>Clean 时，prompt-records bank 对 prompt records 分配的 mass 高于 trace-items bank；prompt records 被 blank 后，两套 bank 的 composition 都转向 trace，同时 exact behavior 保持。Trace blank 后，raw attention 仍可落在这些占位 positions，而 exact behavior 显著下降；在 token 数量与绝对位置固定的条件下，这一行为差异定位到 trace-token content。</p>
<p><code>prompt_all_blank</code> 与 <code>prompt_and_trace_blank</code> 是辅助 formatting-disruption conditions；gold first-answer-token log probability 与解析率是 secondary diagnostics。正文 causal claim 由 registered records/trace source treatments 与 greedy exact-count endpoint 给出。</p>
{conclusion('Step 3 claim',f"Final readout 的主要证据是 source-token intervention 对自由生成 count accuracy 的影响：Qwen prompt-record/trace blank exact rate 为 {pct(answer_primary_by_model['Qwen3-8B']['prompt_records_blank']['exact_rate'])}/{pct(answer_primary_by_model['Qwen3-8B']['trace_all_blank']['exact_rate'])}，Gemma 为 {pct(answer_primary_by_model['Gemma4-E4B']['prompt_records_blank']['exact_rate'])}/{pct(answer_primary_by_model['Gemma4-E4B']['trace_all_blank']['exact_rate'])}。Answer-head attention composition 与这一行为结果一致，并作为 source routing 的次级机制证据。",boundary=True)}</section>

<section id="state-write"><h2>5. Step 4 · State write：targeted retrieval 后的生成内容如何写入 count state</h2>
<p class="lead">前面已证明 targeted bank 对下一条 city 的生成具有必要性，且 final answer 依赖 trace content。本阶段把两者放进同一次干预：在最后一次 targeted query 精确关闭 frozen bank，不再 teacher-force 后续 terminal content，而是用固定 token budget 自由生成并在原位置 replay；随后从 clean 或 ablated trajectory 捕获多层 full residual state，patch 到 receiver，但<strong>始终排除 answer query 本身</strong>。</p>
<div class="chain-purpose"><span class="step-kicker">核心判据</span><p><strong>若生成后的 trace state 是中介，</strong>selected bank 应先损伤 terminal nonmarker tokens 与 answer margin；把同样本 clean state 写回 selected receiver 应修复 margin；把 selected state 写入 clean receiver应造成 occlusion；selected restoration 还应强于三组 layer-matched random-bank restoration。<span class="mini-example"><strong>位置边界：</strong>若 targeted query 位于 terminal item 内，只 patch query 之后真正可能被它因果影响的 terminal suffix，不把 query 前 tokens 伪装成 downstream mediator。</span></p></div>
<div class="native-formula"><strong>Free-running replay。</strong>从 frozen targeted query 后 greedy 生成固定数量 tokens，并替换原序列同一位置；early EOS 只记录，不改变长度。<br><strong>State restoration。</strong>Qwen 从 L19 起、Gemma 从 L16 起，在注册 token span 上逐层 cumulative clamp clean residual；readout 是 answer count utility 与 gold-count margin。<br><strong>Specificity。</strong>selected clean-state restoration − mean(layer-matched random clean-state restoration)。</div>

<h3>5.1 Qwen：多-token generated suffix 比单 terminal span 更能恢复 count margin</h3>
{table(['State geometry','Terminal token damage','Answer margin damage','Clean-state restore','Restore specificity','Selected-state occlusion','Registered utility gate'],q_geometry_rows)}
<p>表中前三个 geometry 在同一 outcome-blind count 2–6 balanced panel 上各自完成 20 discovery / 10 confirmation；discovery 每个 count 4 条、confirmation 每个 count 2 条。三种几何共享相同 Top-128 receiver generation，因此 token damage 与 answer damage相同；区别只来自 mediator span。单 terminal suffix 的 confirmation restore/specificity 为 {effect_ci(q_terminal['clean_state_restoration_margin'])}/{effect_ci(q_terminal['restoration_specificity_margin'])}；扩展到 targeted query 后的完整 generated suffix 后提高到 {effect_ci(q_generated['clean_state_restoration_margin'])}/{effect_ci(q_generated['restoration_specificity_margin'])}；继续加入 post-terminal teacher-forced grammar prefix 后回落到 {effect_ci(q_prefix['clean_state_restoration_margin'])}/{effect_ci(q_prefix['restoration_specificity_margin'])}。</p>
<p>因此 Qwen 的最佳描述不是“一枚 terminal token 保存 count”，而是<strong>targeted query 后的一段自由生成 suffix 承载分布式 count-relevant state</strong>。但该 geometry 是三种注册诊断中事后数值最大的一个，且 registered probability-utility gate 仍为 FAIL；正文只把它写成 margin-level partial pathway。此前 teacher-forced bridge 同样保留：Top-128 receiver damage={f(q_legacy['targeted_receiver_damage_margin']['estimate']):+.3f} margins，clean restore={f(q_legacy['clean_state_restoration_margin']['estimate']):+.3f}，但 selected−random restoration specificity 仅 {f(q_legacy['restoration_specificity_margin']['estimate']):+.3f}，所以不能据此声称独占 terminal counter。</p>
{conclusion('Qwen Step 4 claim','Top-128 targeted retrieval 会改变后续生成；在不 patch answer query 的前提下，恢复 query 后 multi-token generated-suffix state 对 answer count margin 产生中等、matched-control-adjusted 的修复。更宽的 post-terminal prefix 没有继续增强，说明主要 carrier 位于自由生成 suffix，而非随后 teacher-forced grammar tokens。该结论是 partial distributed-state pathway，不是完整或唯一串行中介。',boundary=True)}

<h3>5.2 Gemma：Top-6 → generated terminal suffix → L16:41 state 完整通过 confirmation</h3>
{table(['Confirmation estimand','Effect [95% CI]','Endpoint'],[
['Clean replay adequacy',effect_ci(g_terminal['clean_replay_exact']),'exact suffix'],
['Targeted terminal nonmarker damage',effect_ci(g_terminal['targeted_terminal_nonmarker_damage']),'token accuracy'],
['Targeted answer damage',effect_ci(g_terminal['targeted_answer_damage_utility']),'expected-count utility'],
['Clean-state restoration',effect_ci(g_terminal['clean_state_restoration_utility']),'expected-count utility'],
['Selected-state occlusion',effect_ci(g_terminal['selected_state_occlusion_utility']),'expected-count utility'],
['Restoration specificity',effect_ci(g_terminal['restoration_specificity_utility']),'expected-count utility'],
['Clean-state restoration',effect_ci(g_terminal['clean_state_restoration_margin']),'gold-count margin'],
])}
<p>Gemma 的 fixed-budget clean suffix 在 confirmation 上 10/10 exact replay。关闭 Top-6 后 terminal nonmarker token accuracy 相对 random 下降 {f(g_terminal['targeted_terminal_nonmarker_damage']['mean_effect']):.3f}；answer utility damage={f(g_terminal['targeted_answer_damage_utility']['mean_effect']):.3f}；把 clean L16:41 full-span state 写回 selected receiver 修复 {f(g_terminal['clean_state_restoration_utility']['mean_effect']):.3f} utility / {f(g_terminal['clean_state_restoration_margin']['mean_effect']):.3f} margin；反向把 selected state 写入 clean receiver造成 {f(g_terminal['selected_state_occlusion_utility']['mean_effect']):.3f} utility damage。Discovery 与 confirmation 的完整 registered gate 均 PASS。</p>
{conclusion('Gemma Step 4 claim','Gemma Top-6 retrieval heads causally影响自由生成的 terminal content；该 content 的 L16:41 多-token residual state 中介其 answer-count effect。Clean restore、reverse occlusion 与 layer-matched random specificity 同时成立，因此这条模型内串行 bridge 得到 held-out confirmation。',boundary=True)}

<h3>5.3 Old-HTML-aligned terminal counter：拆开 full span 后，count state 在哪里</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>前两节检验 targeted intervention 后的 mediator；这里改用更接近旧 <code>NIAH-counting.html</code> 的 same-trajectory sufficiency 操作，专门定位 terminal counter carrier。先把所有 trace items 替换为等长 ordinary prompt-background tokens，再从 frozen layer 起逐层恢复 clean terminal hidden states；answer query 始终不 patch。<span class="mini-example"><strong>拆分方式：</strong><code>marker core</code> 是 “total count should be 8” 一类显式进度证据，<code>retrieved city</code> 只含 city span，<code>grammar terminal update</code> 则从该 grammar 的最后一个语义成分延伸到 item commit。</span></p></div>
<div class="native-formula"><strong>Formal panel。</strong>每个模型 20 discovery / 10 confirmation；每 phase 对 <code>rank-after-city</code> 与 <code>rank-before-city</code> 严格平衡为 10:10 / 5:5。Qwen 固定 L19:35，Gemma 固定 L16:41 cumulative clamp。<br><strong>表中每格。</strong>clean semantic-position restoration / equal-token depth-matched non-item specificity，endpoint 均为 correct-count margin。</div>
<div class="evidence-ladder" aria-label="Three-stage terminal counter localization protocol"><div class="evidence-step"><span class="step-kicker">01 · REPRODUCE</span><strong>先复现 full-span rescue</strong><p>在不 patch answer query 的前提下，把 clean item-span hidden states 写回等长 uninformative receiver，检验旧 HTML 的 full-state sufficiency 是否跨模型复现。</p></div><div class="evidence-step"><span class="step-kicker">02 · DISCOVER</span><strong>再拆成五种语义 geometry</strong><p>只用 20 discovery seeds 比较 full item、marker、grammar-timed tail、boundary 与 city；两模型均预先冻结 <code>marker_core</code>。</p></div><div class="evidence-step"><span class="step-kicker">03 · CONFIRM</span><strong>最后做独立 confirmation</strong><p>在 10 个新 seeds 上同时报告 restoration 与等 token 数、同深度 non-item specificity；不按 confirmation 重新选择 split。</p></div></div>
<figure>{primer('图 8a · Terminal counter carrier 的 discovery→confirmation 效应','每个模型的五行对应五种 patch geometry；同一行同时显示 discovery restoration、confirmation restoration，以及 confirmation 中相对等 token 数 matched-random patch 的 specificity。','横轴是 correct-count margin 的变化，0 表示没有修复；点是 seed-equal mean，横线是 95% CI。浅灰用于 discovery，绿色用于 confirmation restoration，橙色用于 selected−random specificity。淡绿色行是 discovery 冻结的 marker geometry。','Qwen marker 在 discovery 中胜出，随后在独立 confirmation 上仍恢复 +2.37 margin，且比同预算 random patch 高 +2.00；city-only 接近 0，因此 carrier 不是 city 字面 span。')}<h4 class="figure-title">图 8a · Full span 拆分后的 restoration 与 specificity</h4>{grammar_span_effect_figure}<figcaption>Qwen 与 Gemma 均固定 20 discovery / 10 confirmation、outcome-blind 且不使用 selection rank。绿色与橙色 point/CI 是独立 confirmation；灰色 discovery 仅用于冻结 geometry。Full-item 给出可恢复上界，marker 与 grammar-tail 定位主要 carrier，boundary/city 则检验局部提交位点与词汇内容是否足够。横轴单位是 correct-count logit margin，不等同于 exact-count accuracy 的百分点。</figcaption></figure>
{table(['Model','Phase','Full item','Marker core','Grammar-timed tail','Boundary commit','Retrieved city','Largest split'],grammar_span_rows)}
<p><strong>主结果。</strong>Discovery 在两个模型均选择 <code>marker_core</code>。Qwen confirmation 的 marker restoration 为 {effect_ci(q_span['confirmation_selected_geometry_restoration'])}，相对 matched-random specificity 为 {effect_ci(q_span['confirmation_selected_geometry_matched_random_specificity'])}；Gemma 相应为 {effect_ci(g_span['confirmation_selected_geometry_restoration'])} 与 {effect_ci(g_span['confirmation_selected_geometry_matched_random_specificity'])}。作为范围参照，full-item confirmation restoration 为 Qwen {f(span_estimand(q_span,'confirmation','full_item','restoration')['mean_effect']):+.2f}、Gemma {f(span_estimand(g_span,'confirmation','full_item','restoration')['mean_effect']):+.2f}；city-only 仅为 {f(span_estimand(q_span,'confirmation','retrieved_city','restoration')['mean_effect']):+.2f}/{f(span_estimand(g_span,'confirmation','retrieved_city','restoration')['mean_effect']):+.2f}。因此效应量排序本身已经排除“最后 city token 就是 counter”的简单解释。</p>
<figure>{primer('图 8b · Grammar-specific marker 与 terminal tail 诊断','把 confirmation 按 trace surface order 分成 city→rank 与 rank→city，并分别画 marker 与 grammar-timed tail 的 restoration/specificity。','每行只有 5 个 confirmation seeds，所以读方向与效应量，不把 CI/p 当作独立 formal gate。绿色是 restoration，浅橙是相对 matched-random specificity；横轴仍是 correct-count margin。','Qwen 的 rank-before marker restoration 为 +3.77，而 rank-after grammar-tail 为 +1.95；这与 marker 出现时间不同导致 carrier 沿 grammar tail 传递的解释一致。')}<h4 class="figure-title">图 8b · Surface grammar 改变有效 carrier 的时间范围</h4>{grammar_timing_figure}<figcaption>每个 grammar stratum 的 confirmation 只有 n=5，故本图是预注册 diagnostic，而不是第二次 geometry selection。City→rank 时 marker 在 item 尾部出现，grammar-tail patch直接覆盖最后更新；rank→city 时 marker 先出现，后续 city/commit 可继续携带该 state。两模型方向大体一致，但 Gemma rank-before 的 tail specificity较弱，不能声称所有 grammar 都由同一固定 token span 承载。</figcaption></figure>
<p><strong>Grammar diagnostic。</strong>Qwen <code>rank-after-city</code> 的 grammar-tail restoration 为 +1.95，而 <code>rank-before-city</code> 的 marker restoration 为 +3.77；Gemma相应为 +3.26 与 +1.46。该方向符合时序解释：city→rank traces 的最后更新段以 marker→commit 为中心，rank→city traces 则保留上游 marker state 并把它传到后续 commit。但该解释仍是 n=5 分层诊断，不能替代整体 20/10 panel。</p>
{conclusion('Terminal counter localization claim','两模型的 old-HTML-aligned full-span restoration 均可在 held-out confirmation 修复 correct-count margin；拆分后，主要 carrier 是显式 progress/count marker 的 hidden states及其到 commit 的 grammar-timed多-token尾部，而不是 retrieved-city lexical span。该实验确认“counter state 在 trace stream 中存在且可干预”，但它本身不证明 frozen targeted head bank 写入了这段 state；targeted→counter 的中介边仍需由前述 integrated bridge 单独判定。',boundary=True)}</section>

<section id="serial-chain"><h2>6. Step 5 · Integrated chain：从 targeted retrieval 到 final count，哪里闭合、哪里仍断裂</h2>
<div class="paper-flow"><div class="paper-step"><strong>I · COMMIT</strong><span>item k endpoint 携带 running-progress representation。</span><span class="operation">h(P0) → k</span></div><div class="paper-step"><strong>II · TARGET</strong><span>recent transition 与 grammar 形成 next-record query。</span><span class="operation">q<sub>g,k→k+1</sub></span></div><div class="paper-step"><strong>III · RETRIEVE</strong><span>Top-128 / Top-6 bank 对下一 city 具有 causal necessity。</span><span class="operation">selected vs random</span></div><div class="paper-step"><strong>IV · WRITE</strong><span>自由生成 suffix 把 retrieved content 写成分布式 trace state。</span><span class="operation">multi-token residual clamp</span></div><div class="paper-step"><strong>V · OUTPUT</strong><span>answer query 从 trace state 产生 final count。</span><span class="operation">restore / occlude / blank</span></div></div>
<figure>{primer('机制图 M1 · 已闭合主路径与仍然开放的并行路径','主行把最新正式证据压成 targeted bank→generated suffix→marker/tail counter→final count；下方三格列出尚未被共同阻断的替代路径。','绿色节点表示两模型都有直接 causal evidence；橙色节点表示模型间证据等级不同。主行箭头只表示实验支持的时序约束，不自动表示唯一中介；下方虚线路径解释为什么目前不能宣称严格串行、独占 circuit。','即使 marker-state patch 能恢复答案，answer query 仍可能同时直接重读整段 trace；只有在同一 forward 中阻断这条并行路径并让 restoration 特异消失，才能升级成唯一串行中介。')}<h4 class="figure-title">机制图 M1 · Targeted retrieval 到 final count 的证据状态</h4><div class="chain-evidence" role="img" aria-label="Evidence chain from targeted retrieval through generated suffix and marker-tail counter state to final count"><div class="chain-node"><strong>Grammar-specific targeted bank</strong><span>Qwen Top-128；Gemma latest bridge Top-6。Selected−random intervention 改变 next-city / terminal generation。</span><span class="status">集合级因果支持</span></div><div class="chain-node partial"><strong>Generated multi-token suffix</strong><span>Gemma terminal nonmarker damage 与 reverse occlusion 完整通过；Qwen 为 count-dependent margin signal。</span><span class="status">Gemma confirmed · Qwen partial</span></div><div class="chain-node"><strong>Marker / grammar-tail counter</strong><span>两模型 full-span rescue；discovery 冻结 marker，confirmation restoration 与 specificity 均为正。</span><span class="status">跨模型定位</span></div><div class="chain-node partial"><strong>Final count readout</strong><span>Trace blank 显著损伤答案；state restore 可救 margin，但 prompt/trace rere读与自我纠错仍可并行。</span><span class="status">存在通路 · 非唯一通路</span></div></div><div class="parallel-paths"><div><strong>并行 A · Prompt broad readout</strong><br>Answer query 可直接从 prompt records 广域读取，无需完全经过 terminal marker。</div><div><strong>并行 B · Full-trace reread</strong><br>Final query 可利用多个既有 trace items，而非只依赖最后一个局部 counter carrier。</div><div><strong>并行 C · Self-correction / grammar cue</strong><br>后续生成 token 可根据表面数字、列表长度或不一致性修正初始 count state。</div></div><figcaption>主路径中，targeted→generated-suffix 在 Gemma 得到同一 intervention family 的完整 confirmation，在 Qwen 只得到 margin-level partial support；marker/tail→answer 来自独立 same-trajectory span decomposition。两组实验的组合支持“至少存在一条此类路径”，但因为 receiver、干预族与自然轨迹不同，尚不足以证明所有效应都由同一个唯一 mediator 串联。</figcaption></figure>
<div class="claim"><strong>模型间结论必须分开。</strong>Gemma 的 III→IV→V 在同一 free-running intervention family 内通过 discovery 与 confirmation；Qwen 的 III 很强，IV→V 在 count margin 上可恢复且多-token 优于单-span，但 registered utility gate 不通过、效应随 count 明显变化。因此不能写成“两模型都完整闭环”，只能写成“共享功能阶段，闭环强度不同”。</div>

<h3>6.1 Qwen count 2–6 与 count 9/10 诊断</h3>
{table(['Count','Confirmation n','Mediator geometry','Terminal token damage','Answer margin damage','Clean-state restore','Restore specificity'],q_count_rows)}
<p>count 2–6 的每格 confirmation 只有 2 seeds，因此单 count 行只作诊断；pooled 20/10 panel 才是正式分析单位。生成 suffix 的 confirmation 信号主要来自 count 3 与 5，count 4/6 在这一小 confirmation stratum 为 0，count 2 虽有 restoration但 targeted damage 为负，不能组成同一串行边。更准确的表述是：Qwen state pathway具有明显的 count/trajectory heterogeneity，而不是从 count 2 到 10 由同一均匀局部寄存器承担。</p>
<p>count 10 的旧负结果也不能再解释成“没有 state”。在 5-seed confirmation 的 balanced 9/10 diagnostic 中，count 10 的 probability utility 近 0，但 gold-count margin 上 targeted damage={f(q_count10['targeted_answer_damage__correct_count_margin']):+.3f}、clean restore={f(q_count10['selected_clean_state_restoration__correct_count_margin']):+.3f}、specificity={f(q_count10['restoration_specificity__correct_count_margin']):+.3f}；count 9 相应为 {f(q_count9['targeted_answer_damage__correct_count_margin']):+.3f}/{f(q_count9['selected_clean_state_restoration__correct_count_margin']):+.3f}/{f(q_count9['restoration_specificity__correct_count_margin']):+.3f}。这表明 count 10 的主要问题是 bounded probability endpoint 饱和与小样本 trajectory heterogeneity；margin diagnostic 支持 state 存在，但 n=5 不足以升级成独立强 claim。</p>

<h3>6.2 最终 claim boundary</h3>
{table(['Model','Targeted retrieval','Generated suffix','State→answer','整链状态'],[
['Qwen3-8B','Top-128：强 selected−random necessity','token damage 中等且 count-dependent','multi-token margin restore +1.025；specificity +0.921','PARTIAL；registered utility gate FAIL'],
['Gemma4-E4B','Top-6：冻结 targeted bank','terminal nonmarker damage +0.600','utility restore +0.077；occlusion +0.394','CONFIRMED；20d/10c complete PASS'],
])}
<h3>6.3 哪些地方仍然 confounded</h3>
<div class="table-wrap"><table><thead><tr><th>候选边 / claim</th><th>最强现有证据</th><th>仍然开放的混淆</th><th>当前允许写法</th></tr></thead><tbody>
<tr><td>Targeted bank → next-city generation</td><td>冻结 selected bank 相对同大小 random bank 增加 failure；Gemma Top-6 bridge 内 terminal token damage=+0.600</td><td><span class="confound-medium">Bank-level granularity。</span>Random bank只匹配 head 数/大致层分布，不保证功能、输出范数和 redundancy完全匹配；早期 target-trigger主表用 Gemma K8，最新 integrated bridge 用 K6。</td><td>Grammar-specific selected bank 对检索/生成具有集合级必要性；不声称每个 head 单独必要，也不把历史 K8 与最新 K6 当成完全相同实验。</td></tr>
<tr><td>Targeted retrieval → marker/tail counter state</td><td>Gemma 同一 free-running Top-6 family 中存在 token damage、state restore 与 reverse occlusion；Qwen generated-suffix state 有 margin repair</td><td><span class="confound-high">当前最关键断点。</span>Span decomposition 在 clean/uninformative same-trajectory receiver 上定位 marker；integrated bridge在 targeted-ablation receiver 上 patch整个 generated suffix。尚未直接证明 targeted bank 特异写入的正是 marker/tail carrier。</td><td>Gemma 支持 targeted→distributed generated-state→answer；Qwen 支持 partial pathway。两模型都不能仅凭拼接实验宣称“targeted heads 写入一个已唯一识别的 marker counter”。</td></tr>
<tr><td>Marker/tail state → final count</td><td>Marker confirmation restoration/specificity：Qwen +2.37/+2.00，Gemma +1.92/+0.99；city-only近 0</td><td><span class="confound-medium">抽象 state 与表面证据未完全分离。</span>Marker span本身含 “count 8” 等词法数字；full-vector cumulative clamp 是高维干预。Equal-token random control未匹配 hidden-state norm、semantic content或局部 manifold。</td><td>Marker/tail hidden states 是充分且位置特异的 counter carrier；暂不声称已分离出纯抽象整数 subspace或最小 token set。</td></tr>
<tr><td>严格串行 / 唯一中介</td><td>State restoration、reverse occlusion、trace blank 与 targeted ablation方向一致</td><td><span class="confound-high">并行路径未共同阻断。</span>Answer query 可直接重读 prompt/全 trace，后续 token也可能自我纠错；broad retrieval 与 residual-stream carrier可互补。</td><td>至少一条显著 causal pathway，且 Gemma bridge闭合；不声称该路径是唯一、穷尽或所有 counts 上同质。</td></tr>
<tr><td>跨 grammar / count 泛化</td><td>Formal overall panel严格平衡两类 grammar；Qwen count 2–6 与 9/10 diagnostics保留</td><td><span class="confound-medium">小分层与异质性。</span>Grammar-stratum confirmation每格 n=5；Qwen单 count常只有 n=2，且高 count probability endpoint饱和。</td><td>整体 20/10 panel 可作正式 claim；grammar与单 count只用于解释 carrier时序和异质性，不单独升级。</td></tr>
</tbody></table></div>
{conclusion('Integrated-chain 结论','两模型都支持“targeted retrieval 之后存在分布式 trace-state pathway并参与 final count”，但证据等级不同：Gemma 的 Top-6→generated terminal suffix→L16:41 state→answer count 在 held-out confirmation 完整闭合；Qwen 的 Top-128→generated suffix→answer-margin 只构成一条 count-dependent partial pathway。报告保留 Qwen 弱 claim，并明确不声称唯一 counter、独占 head bank 或跨全部 counts 的同质串行机制。',boundary=True)}</section>

<section id="ledger"><h2>Claim ledger：每一层结论由哪一个注册实验支持</h2>
<div class="table-wrap"><table><thead><tr><th>Stage</th><th>注册实验</th><th>Claim</th><th>理由与适用范围</th></tr></thead><tbody>
<tr><td>Commit</td><td>Seed-held-out Logistic/NCC</td><td>Item endpoint 与 answer query 携带可泛化的 k/N representation</td><td>Frozen classifier 在 10 个 confirmation seeds 上高于 10% chance；estimand 为 decodability</td></tr>
<tr><td>Localize</td><td>Exact-query target mass / OV-write ranking</td><td>Discovery-only score 可定位 next-record candidate heads</td><td>正确 record 的 mass 随 ordinal 对角移动，并跨 discovery seeds 复现</td></tr>
<tr><td>Retrieve</td><td>Persistent selected vs matched-random ablation</td><td>Frozen grammar-specific bank 对正确 next city 具有集合级必要性</td><td>Selected−random failure contrast；query、K、start 与持续时长匹配</td></tr>
<tr><td>Trigger source</td><td>Recent/cumulative/full trace blanks vs equal-token controls</td><td>Adjacent-after targeting 主要依赖最近 transition，Gemma 同时利用累计 trace</td><td>Position-preserving treatment−control specificity；Qwen 45 events，Gemma 30 events</td></tr>
<tr><td>Final readout</td><td>Prompt-record vs trace token-state blanks</td><td>两模型 final count 主要利用 trace-token content</td><td>100 prompts/model；prompt-record blank 保持 accuracy，trace blank 显著降低 accuracy</td></tr>
<tr><td>State write · Qwen</td><td>Top-128 free-running suffix + three cumulative state geometries</td><td>query 后 multi-token generated-suffix state 可部分恢复 answer margin</td><td>固定 20d/10c；generated-suffix confirmation restore +1.025、specificity +0.921；registered utility gate FAIL</td></tr>
<tr><td>State write · Gemma</td><td>Top-6 free-running suffix → L16:41 full-state restore/occlude</td><td>generated terminal content 的 residual state 中介 answer-count effect</td><td>固定 20d/10c；confirmation utility restore +0.077、occlusion +0.394；完整 gate PASS</td></tr>
<tr><td>Terminal counter carrier</td><td>Old-HTML-aligned full-state span decomposition</td><td>progress marker 与 grammar-timed tail 是主要 count-state carrier；city span 不是</td><td>固定 20d/10c；discovery 选 marker，confirmation Qwen restore/specificity +2.37/+2.00，Gemma +1.92/+0.99</td></tr>
<tr><td>Integrated chain</td><td>Targeted bank → generated suffix → residual state → answer</td><td>Gemma 完整闭环；Qwen 为 count-dependent partial pathway</td><td>模型内分别判定，不用 Gemma PASS 替 Qwen 升级 claim</td></tr>
</tbody></table></div>
<div class="claim boundary"><strong>当前总论。</strong>Native-thinking 模型把逐项显式 trace 组织成一个可重复的检索与状态积累循环：item endpoint 携带 running progress；最近 transition 与 surface grammar 共同约束 next-record targeting；grammar-specific retrieval bank 支持正确 next city；retrieval 后生成的多-token suffix 写入分布式 state；循环结束后，final answer 主要利用 trace-token content。Gemma 的后半链得到完整 confirmation，Qwen 的后半链停留在 margin-level partial support。</div>
{conclusion('Claim ledger 结论','主文 claim 分别停留在 representation decodability、routing localization、bank-level necessity、source specificity、state mediation 与 final source dependence六个层级；每个结论均附注册 cohort、对照组、聚合方式和 endpoint，并对 Qwen/Gemma 使用不同强度的 claim。')}</section>

<section id="appendix"><h2>Appendix · Discovery 选择、稳健性与 exploratory scope</h2>
<details class="paper-appendix"><summary>Appendix A · Discovery screen 如何确定正式 localizer</summary><h3>Discovery-only localizer screen</h3>
{table(['Candidate','Behavior grammar','Selection site','Selected failure'],appendix_screen)}
<p>这些数字是 discovery selection statistics。City-pre adjacent-after OV score 与 same-unit-before raw mass 产生最高 discovery disruption，因此被注册为正式 localizer；冻结配置随后直接进入 confirmation effect estimation。</p>
{conclusion('Appendix A 结论','Token site 与 ranking metric 会改变 bank membership；正式方案完全由 discovery screen 冻结，confirmation 承担独立效应估计。')}</details>

<details class="paper-appendix"><summary>Appendix B · Exact causal-site cohort 的 representation geometry</summary><h3>Legacy item ends vs exact causal commits</h3>
<figure>{primer('图 B1 · Representation cohort alignment','把所有 parser-observed item ends 收窄到 exact causal commits，比较同类 classifier 的 confirmation BA。','横轴两端是旧/新 cohort；纵向位置是 BA。线向上表示 decoding 改善；SNR 同时报告 centroid 间距与类内变化的比例。','Qwen Logistic 约增加 1.7 pp，SNR 从 −5.25 dB 变为 −5.34 dB：exact-site cohort 提升分类边界，同时保留相近的类内几何复杂度。')}<h4 class="figure-title">图 B1 · All item ends → exact causal P0 commits</h4>{alignment_svg(rep)}<figcaption>每个 panel 横轴为 cohort definition，纵轴为 confirmation balanced accuracy；绿线=Logistic，灰线=NCC。Qwen/Gemma 的 BA 小幅增加，两模型 SNR 保持负值，因此 exact causal-site representation 仍呈多维、带显著类内变化的 geometry。</figcaption></figure>
{conclusion('Appendix B 结论','Exact causal-site restriction 带来小幅 classification improvement，并保留负 SNR 所刻画的多维类内结构；正文据此采用“可跨 seed 解码的多维 progress representation”这一 claim。')}</details>

<details class="paper-appendix"><summary>Appendix C · Exploratory grammar 与完整 P0 descriptive atlas</summary><p>Confirmation anchors 少于 10 的 grammar 按 exploratory estimand 报告：Qwen <code>structural_unmarked</code> n=6、bullet/evidence 各 n=1；Gemma adjacent-before/bullet 各 n=1。对应比例的解释单位是这些注册 anchors。</p>
<p>完整的 P0 descriptive head maps 与逐 grammar needle-order profiles 保留在独立的 <a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 Targeted-Retrieval Atlas</a> 中。该 atlas 的 head map 横轴为 head、纵轴为 layer、颜色为 seed-equal target-mass score；它专门刻画 P0 descriptive attention structure，图 3 的 city-pre/OV bank 则对应正式 causal selection。</p>
{conclusion('Appendix C 结论','P0 atlas 提供可复查的 descriptive localization；主文 causal claim 使用冻结后的正式 bank 与至少 10 个 confirmation anchors，低样本 grammar 以 exploratory 范围单列。')}</details>

<details class="paper-appendix"><summary>Appendix D · 重叠 heads 仍是 broad，还是会变成 targeted</summary>
<h3>D.1 先看同一批共享 heads 在 Non-thinking answer query 上做什么</h3>
<div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>Membership overlap 标识两套正式 bank 共同选中的 layer–head；attention concentration 进一步刻画这些共享 heads 在不同 query 下的空间分布。<span class="mini-example"><strong>直观例子：</strong>如果一个 head 对 N1…N6 都分配相近 mass，它是 broad；如果大部分 needle mass 落在单一 next target，则是 targeted。</span></p></div>
<div class="native-formula"><strong>Effective needle support。</strong> 对同一 query 指向各 active needle span 的 mass m<sub>i</sub>，N<sub>eff</sub>=(Σ<sub>i</sub>m<sub>i</sub>)²/Σ<sub>i</sub>m<sub>i</sub>²。N<sub>eff</sub>=1 表示集中于一条 needle；若在 N 条 needles 上完全均匀，则 N<sub>eff</sub>=N。</div>
<p>Non-thinking 的展示 gallery 在本次 overlap 分析之前已经冻结：取 formal broad ranking Top-4，并固定 confirmation seed 1254 的 N=3/6/9；样本选择完全由该预定规则决定。其中 <code>L27H18</code>（broad rank 1）与 <code>L23H13</code>（broad rank 4）同时进入两个 Native formal Top-128，因而构成预先冻结的共享-head 样本。</p>
<figure>{primer('图 D1 · Non-thinking answer query 的共享-head attention map','每行是一个预先冻结 broad head 与一个固定自然 prompt；每列是该 prompt 的一条 active needle span。','格内数字与颜色都是该完整 answer-query attention row 落在整条 needle span 上的 raw mass；灰格对应超出该 prompt N 的 ordinal。右侧 needle total 是所有 active needles 的总 mass，effective support 衡量这些 mass 分散到多少条 needles。','N=6 时，L27H18 的 effective support 为 '+f'{broad_example_support[((27,18),6)]:.2f}'+'/6，L23H13 为 '+f'{broad_example_support[((23,13),6)]:.2f}'+'/6；mass 分布覆盖多条 records。')}<h4 class="figure-title">图 D1 · Broad-bank shared heads at the Non-thinking answer query</h4>{shared_broad_map}<figcaption>横轴=active needle ordinal；纵轴=共享 head × 固定 prompt N；格内=raw attention mass，占完整 answer-query attention row 的比例。右侧 N<sub>eff</sub> 使用上式，衡量 active-needle mass 内部的分散程度。L27H18 在 N=9 时 N<sub>eff</sub>={broad_example_support[((27,18),9)]:.2f}/9，L23H13 为 {broad_example_support[((23,13),9)]:.2f}/9，支持它们在 Non-thinking answer query 上具有 broad coverage。</figcaption></figure>

<h3>D.2 再看全部正式 overlap heads 在 Native city-pre query 上是否集中</h3>
<p>对 Qwen broad Top-32 与每个 Native formal Top-128 的交集，直接读取 discovery seed-equal city-pre statistics。<code>target / all needles</code> 把正确 next record 的 mass 除以所有 needle records 的 mass；<code>target top-1</code> 是正确 record 在 event 内成为 attention 最大 needle 的比例。灰格标记该 head 位于交集之外，着色格对应正式 membership。</p>
<figure>{primer('图 D2 · 全部共享 heads 的 Native city-pre concentration','行是 broad Top-32 中至少进入一个 Native formal bank 的 head；列分别给两个 major grammar 的 target share 与 target top-1。','颜色和格内数字统一使用 0–100%；越深表示 query 越集中到正确 next record。正式 membership 使用着色格，灰格标记交集之外。','Adjacent-after 的 18 枚共享 heads 中，target/all-needle share 中位数为 '+pct(quantile(adj_shared_target_share,.5))+'；same-unit-before 的 20 枚中位数为 '+pct(quantile(same_shared_target_share,.5))+'.')}<h4 class="figure-title">图 D2 · Native city-pre attention concentration among formal-bank overlaps</h4>{shared_native_map}<figcaption>横轴=grammar-specific concentration statistic；纵轴按 Non-thinking broad rank 排列的共享 layer–head。Adjacent-after overlap 的 target share / target-top1 中位数分别为 {pct(quantile(adj_shared_target_share,.5))} / {pct(quantile(adj_shared_top1,.5))}，其中 {sum(value >= .5 for value in adj_shared_target_share)}/{len(adj_shared_target_share)} 枚 target share≥50%；same-unit-before 分别为 {pct(quantile(same_shared_target_share,.5))} / {pct(quantile(same_shared_top1,.5))}，target share≥50% 为 {sum(value >= .5 for value in same_shared_target_share)}/{len(same_shared_target_share)}。共享 pool 的 concentration 随 grammar 与 ranking route 系统变化。</figcaption></figure>

<h3>D.3 一个同时进入三套 bank 的具体 head</h3>
<p><code>L24H29</code> 是 Non-thinking broad rank {q_broad_ranks[shared_example_head]}，同时是 Native adjacent-after formal rank {q_adj_ranks[shared_example_head]} 与 same-unit-before formal rank {q_same_ranks[shared_example_head]}。下面复用图 2 的预先固定 exact-P0 N=10 trace：红框沿 N2→N10 对角移动，展示同一共享 head 在 Native transition 上形成 next-record-specific routing。该 P0 query 描述 attention 形状；正式 membership 由 city-pre ranking 给出，因果效应由 confirmation ablation 给出。</p>
<figure>{primer('图 D3 · 共享 head L24H29 的 Native targeted pattern','横轴是 P0 transition k→k+1，纵轴是带真实 city 名的 prompt records；红框/红点是正确 next record。','颜色是 L24H29 的 raw attention mass。亮格随红框逐列移动表示该 query 下形成 targeted routing。','同一个 layer–head 坐标在 Non-thinking answer query 参与 broad aggregation，并在 Native transition query 参与 next-record routing，展示 query-dependent functional reuse。')}<h4 class="figure-title">图 D3 · Qwen L24H29 · shared-bank exact-P0 attention map</h4><div class="attention-example-svg">{shared_native_example}</div><figcaption>横轴=P0 transition；纵轴=prompt record；红框/红点=正确 k+1 target；颜色=raw attention mass。L24H29 同时属于 Non-thinking broad Top-32 与两个 Native formal Top-128，其 Native attention 呈现随 transition 移动的对角结构。该预先冻结单例展示共享 head 的 query-dependent reuse。</figcaption></figure>
{conclusion('Appendix D 结论','正式 bank 的重叠 heads 表现出 query-与 grammar-dependent reuse：在 Non-thinking answer query 上，它们广泛覆盖多条 needles；在 Native city-pre/P0 transitions 上，同一候选 pool 提高对正确 next record 的 concentration。模型因此能够复用 retrieval-capable heads，同时按当前 query 重设目标与优先级。',boundary=True)}</details>

<details class="paper-appendix"><summary>Appendix E · 复现账本与底层文件</summary><div class="source-list"><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><br><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><br><a href="v5_native_final_localizers/analysis/qwen_final_merged_dose_grid.json">Final Qwen merged analysis</a><br><a href="v5_native_final_localizers/analysis/qwen_final_major_grammar_dose.csv">Major-grammar bootstrap table</a><br><a href="v5_native_final_localizers/screening/screen_results.csv">Discovery screen</a><br><a href="v5_native_token_level_ablation/Qwen3-8B/targeting_adj_citypre_k128_confirmation_v1/analysis_registered_v1/analysis_audit.json">Qwen target-trigger audit</a><br><a href="v5_native_token_level_ablation/Qwen3-8B/answer_tracebank_top32_confirmation_all20_v2/analysis_registered_v1/analysis_audit.json">Qwen trace-bank readout audit</a><br><a href="v5_native_token_level_ablation/Qwen3-8B/answer_promptbank_top32_confirmation_all20_v2/analysis_registered_v1/analysis_audit.json">Qwen prompt-bank readout audit</a><br><a href="v5_native_token_level_ablation/Gemma4-E4B/targeting_adj_p0_k8_confirmation_v1/analysis_registered_v1/analysis_audit.json">Gemma target-trigger audit</a><br><a href="v5_native_token_level_ablation/Gemma4-E4B/answer_tracebank_top32_confirmation_all20_v1/analysis_registered_v1/analysis_audit.json">Gemma trace-bank readout audit</a><br><a href="v5_native_token_level_ablation/Gemma4-E4B/answer_promptbank_top32_confirmation_all20_v1/analysis_registered_v1/analysis_audit.json">Gemma prompt-bank readout audit</a><br><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">Detailed P0 atlas</a><br><a href="../work_remote_snapshots/native_terminal_chain_evidence_20d10c_20260821.json">Terminal-chain synthesized evidence</a><br><a href="../work_remote_snapshots/qwen_grammar_span_decomposition_complete.json">Qwen grammar-span decomposition</a><br><a href="../work_remote_snapshots/gemma_grammar_span_decomposition_complete.json">Gemma grammar-span decomposition</a></div><p class="audit">Generated UTC: {esc(generated)}<br>{ledger}<br>Schema: realistic_niah_v5_native_thinking_final_v4</p>
{conclusion('Appendix E 结论','Builder 在输出前验证 Qwen/Gemma final causal analyses、两模型 token-source registered audits、representation selected rows、head rankings、Non-thinking frozen attention gallery、terminal-chain evidence，以及 grammar-span decomposition 的 frozen 20/10 seed contract 与 sealed discovery/confirmation summaries；正文全部数值与图形都由冻结文件生成。')}</details></section>

</main></article><script>
document.querySelectorAll('[data-attention-selector]').forEach(select => {{
  const update = () => {{
    const value = select.value;
    document.querySelectorAll('[data-attention-example]').forEach(panel => {{
      panel.style.display = panel.dataset.attentionExample === value ? 'block' : 'none';
    }});
  }};
  select.addEventListener('change', update);
  update();
}});
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, default=Path("reports/NiaH_Non-thinking_report.html"))
    parser.add_argument("--qwen-analysis", type=Path, default=Path("reports/v5_native_final_localizers/analysis/qwen_final_merged_dose_grid.json"))
    parser.add_argument("--gemma-analysis", type=Path, default=Path("reports/v5_native_hybrid_supplement/Gemma4-E4B/analysis_hybrid_supplement_registered_v1/hybrid_dose_grid_complete.json"))
    parser.add_argument("--representation-root", type=Path, default=Path("reports/v5_native_causal_aligned_geometry"))
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--trajectory-registry", type=Path, default=Path("reports/v5_native_causal_site_review/trajectory_registry.jsonl"))
    parser.add_argument("--qwen-adj-ranking", type=Path, default=Path("reports/v5_native_final_localizers/Qwen3-8B/plans/adj_citypre_ovnorm/k128/crossfit_source_specific_head_ranking.csv"))
    parser.add_argument("--qwen-same-ranking", type=Path, default=Path("reports/v5_native_final_localizers/Qwen3-8B/plans/same_citypre_abs/k128/crossfit_source_specific_head_ranking.csv"))
    parser.add_argument("--gemma-p0-ranking", type=Path, default=Path("reports/v5_native_p0_head_atlas/p0_targeted_retrieval_head_scores.csv"))
    parser.add_argument("--nonthinking-broad-membership", type=Path, default=Path("reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv"))
    parser.add_argument("--nonthinking-attention-gallery", type=Path, default=Path("reports/v4_non-thinking_causal/v4_4_report_additions/qwen_attention_gallery.json"))
    parser.add_argument("--p0-atlas-assets", type=Path, default=Path("reports/v5_native_p0_head_atlas"))
    parser.add_argument("--screen-results", type=Path, default=Path("reports/v5_native_final_localizers/screening/screen_results.csv"))
    parser.add_argument(
        "--token-ablation-root",
        "--qwen-token-ablation-root",
        dest="qwen_token_ablation_root",
        type=Path,
        default=Path("reports/v5_native_token_level_ablation/Qwen3-8B"),
    )
    parser.add_argument(
        "--gemma-token-ablation-root",
        type=Path,
        default=Path("reports/v5_native_token_level_ablation/Gemma4-E4B"),
    )
    parser.add_argument(
        "--terminal-chain-evidence",
        type=Path,
        default=Path(
            "work_remote_snapshots/native_terminal_chain_evidence_20d10c_20260821.json"
        ),
    )
    parser.add_argument(
        "--qwen-grammar-span-evidence",
        type=Path,
        default=Path(
            "work_remote_snapshots/qwen_grammar_span_decomposition_complete.json"
        ),
    )
    parser.add_argument(
        "--gemma-grammar-span-evidence",
        type=Path,
        default=Path(
            "work_remote_snapshots/gemma_grammar_span_decomposition_complete.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/NiaH_Native-Thinking_report.html"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/v5_native_final_localizers/report_manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qwen = read_json(args.qwen_analysis)
    gemma = read_json(args.gemma_analysis)
    require(qwen.get("status") == "PASS", "Qwen final analysis is not PASS")
    require(gemma.get("status") == "PASS", "Gemma hybrid analysis is not PASS")
    require(qwen.get("persistent_ablation") is True, "Qwen ablation is not persistent")
    require(qwen.get("intervention_start_anchor_role") == "p0_item_end", "Qwen intervention start is not P0")
    rep = load_representation(args.representation_root, args.dual_endpoint_root)
    duplicates = load_duplicates(args.trajectory_registry)
    q_adj_rank = read_csv(args.qwen_adj_ranking)
    q_same_rank = read_csv(args.qwen_same_ranking)
    gemma_p0_rank = read_csv(args.gemma_p0_ranking)
    nonthinking_membership = read_csv(args.nonthinking_broad_membership)
    nonthinking_attention_gallery = read_json(args.nonthinking_attention_gallery)
    attention_examples = load_attention_examples(args.p0_atlas_assets)
    screen_rows = read_csv(args.screen_results)
    token_evidence = {
        "Qwen3-8B": load_token_ablation_evidence(
            args.qwen_token_ablation_root, "Qwen3-8B"
        ),
        "Gemma4-E4B": load_token_ablation_evidence(
            args.gemma_token_ablation_root, "Gemma4-E4B"
        ),
    }
    terminal_chain = read_json(args.terminal_chain_evidence)
    grammar_span = {
        "Qwen3-8B": read_json(args.qwen_grammar_span_evidence),
        "Gemma4-E4B": read_json(args.gemma_grammar_span_evidence),
    }
    inputs = [
        args.reference_report, args.qwen_analysis, args.gemma_analysis,
        args.representation_root / "site_selected.csv",
        args.representation_root / "legacy_vs_causal_item_end.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        args.trajectory_registry, args.qwen_adj_ranking, args.qwen_same_ranking,
        args.gemma_p0_ranking, args.nonthinking_broad_membership,
        args.nonthinking_attention_gallery,
        args.screen_results,
        args.terminal_chain_evidence,
        args.qwen_grammar_span_evidence,
        args.gemma_grammar_span_evidence,
        *(path for evidence in token_evidence.values() for path in evidence["input_files"]),
        *(Path(example["path"]) for example in attention_examples),
    ]
    hashes = {str(path): sha256(path) for path in inputs}
    generated = datetime.now(timezone.utc).isoformat()
    report = build_report(
        css=extract_reference_css(args.reference_report), rep=rep, qwen=qwen, gemma=gemma,
        q_adj_rank=q_adj_rank, q_same_rank=q_same_rank, gemma_p0_rank=gemma_p0_rank,
        nonthinking_membership=nonthinking_membership,
        nonthinking_attention_gallery=nonthinking_attention_gallery,
        attention_examples=attention_examples,
        screen_rows=screen_rows,
        duplicates=duplicates, token_evidence=token_evidence,
        terminal_chain=terminal_chain,
        grammar_span=grammar_span,
        generated=generated, hashes=hashes,
    )
    args.output.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version":"realistic_niah_v5_native_thinking_final_v4",
        "status":"PASS",
        "generated_at":generated,
        "output":str(args.output),
        "output_sha256":sha256(args.output),
        "qwen_primary":next(row for row in qwen["overall"] if row["scope"]=="all_registered_grammars" and i(row["bank_size"])==128),
        "gemma_primary":next(row for row in gemma["overall"] if row["scope"]=="all_registered_grammars" and i(row["bank_size"])==8),
        "token_source_ablation": {
            model: {
                "targeting_frozen_confirmation_seeds": evidence["formal_confirmation_seed_count"],
                "targeting_registered_seeds": evidence["target_registered_seed_count"],
                "targeting_requests": evidence["target_request_count"],
                "answer_discovery_seeds": 20,
                "answer_confirmation_seeds": 10,
                "answer_bank_overlap": evidence["answer_bank_overlap"],
                "answer_bank_jaccard": evidence["answer_bank_jaccard"],
            }
            for model, evidence in token_evidence.items()
        },
        "terminal_chain": {
            "qwen_status": terminal_chain["qwen"]["overall_status"],
            "gemma_status": terminal_chain["gemma"]["overall_status"],
            "qwen_descriptive_best_geometry": terminal_chain["qwen"][
                "descriptive_best_geometry"
            ],
            "gemma_complete_bridge_pass": terminal_chain["gemma"][
                "complete_bridge_pass"
            ],
        },
        "grammar_span_decomposition": {
            model: {
                "status": evidence["status"],
                "discovery_seed_count": evidence["discovery_seed_count"],
                "confirmation_seed_count": evidence["confirmation_seed_count"],
                "discovery_selected_split_geometry": evidence[
                    "discovery_selected_split_geometry"
                ],
                "confirmation_descriptive_signal": evidence[
                    "discovery_selected_geometry_confirmation_descriptive_signal"
                ],
            }
            for model, evidence in grammar_span.items()
        },
        "inputs_sha256":hashes,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(args.output),"sha256":manifest["output_sha256"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
