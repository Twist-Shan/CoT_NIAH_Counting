#!/usr/bin/env python3
"""Build the concise Non-thinking V4.4 mechanism report.

The rendered report is deliberately organized by the proposed computation:

1. mechanism at a glance, followed by the behavioral target and measurements;
2. distributed prompt-side evidence formation;
3. broad, partial answer-query retrieval;
4. late consolidation plus architecture-specific OV/residual write;
5. evidence synthesis, extension audit, and explicitly prioritized open work.

All displayed numbers are loaded from frozen report artifacts.  The only
embedded observations are four pre-computed, frozen-PCA projections used by
the two interactive 3-D figures.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import html
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "v4_non-thinking_causal"
QWEN_COLOR = "#00C2FF"
GEMMA_COLOR = "#F29E4C"
QWEN_LIGHT = "#A8EBFF"
GEMMA_LIGHT = "#F8CCA4"
MODEL_COLORS = {"Qwen3-8B": QWEN_COLOR, "Gemma4-E4B": GEMMA_COLOR}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick(rows: Iterable[dict[str, str]], **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    rendered = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
    raise KeyError(f"No row matched {rendered}")


def extract_embedded_json(document: str, variable_name: str) -> dict[str, Any]:
    marker = f"const {variable_name}="
    start = document.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not locate embedded {variable_name}")
    start += len(marker)
    end = document.find(";\nconst ", start)
    if end < 0:
        end = document.find(";</script>", start)
    if end < 0:
        raise RuntimeError(f"Could not locate end of embedded {variable_name}")
    return json.loads(document[start:end])


def f(value: object, digits: int = 3) -> str:
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def p_text(value: object) -> str:
    number = float(value)
    return f"{number:.2e}" if number < 0.001 else f"{number:.3f}"


def effect_interval_excludes_zero(row: dict[str, object]) -> bool:
    if str(row.get("exact_sign_flip_p", "")).strip():
        return float(row["exact_sign_flip_p"]) < 0.05
    return str(row.get("primary_ci95_excludes_zero_positive", "")).lower() == "true"


def seed_cluster_bootstrap_interval(
    rows: Iterable[dict[str, object]],
    metric: str,
    *,
    draws: int = 10_000,
    random_seed: int = 4_456,
) -> dict[str, float]:
    """Deterministic percentile interval over equal-weight seed means."""

    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    seed_means = [
        sum(values) / len(values)
        for _, values in sorted(by_seed.items())
    ]
    if not seed_means:
        raise RuntimeError(f"No seed clusters found for {metric}")
    rng = random.Random(random_seed)
    bootstrap = sorted(
        sum(rng.choice(seed_means) for _ in seed_means) / len(seed_means)
        for _ in range(draws)
    )
    return {
        "mean": sum(seed_means) / len(seed_means),
        "ci95_low": bootstrap[int(0.025 * draws) - 1],
        "ci95_high": bootstrap[int(0.975 * draws) - 1],
        "seed_clusters": float(len(seed_means)),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Pointwise Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def ci(row: dict[str, str], low: str, high: str, digits: int = 3) -> str:
    return f"{f(row[low], digits)}–{f(row[high], digits)}"


def svg_line_chart(
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str, str]],
    *,
    x_label: str = "Transformer layer ℓ",
    x_value_prefix: str = "L",
    x_ticks: list[float] | None = None,
    x_domain: tuple[float, float] | None = None,
    y_domain: tuple[float, float] | None = None,
    reference: tuple[float, str] | None = None,
    vertical_references: list[tuple[float, str]] | None = None,
    intervals: dict[str, list[tuple[float, float, float, bool]]] | None = None,
    interval_kind: str = "effect",
    interval_kinds: dict[str, str] | None = None,
    x_as_percent: bool = False,
    width: int = 620,
    height: int = 320,
) -> str:
    """Return a compact responsive SVG line chart.

    Each series is (label, points, color, dash).  The data domain is derived
    from the observations unless an interpretable fixed domain is supplied.
    """

    all_points = [point for _, points, _, _ in series for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    if intervals:
        for interval_rows in intervals.values():
            for _, low, high, _ in interval_rows:
                y_values.extend((low, high))
    if reference:
        y_values.append(reference[0])
    x_min, x_max = x_domain or (min(x_values), max(x_values))
    if y_domain is None:
        y_min, y_max = min(y_values), max(y_values)
        pad = max((y_max - y_min) * 0.12, 0.02)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        y_min, y_max = y_domain
    left, right, top, bottom = 68, 24, 34, 58
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / max(y_max - y_min, 1e-9) * plot_h

    tick_values = x_ticks or [x_min + i * (x_max - x_min) / 5 for i in range(6)]
    y_ticks = [y_min + i * (y_max - y_min) / 4 for i in range(5)]
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}。横轴为{html.escape(x_label)}，纵轴为{html.escape(y_label)}。">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>')
    for tick in tick_values:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        tick_label = f"{100 * tick:.1f}%" if x_as_percent else f"{tick:.0f}"
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick_label}</text>')
    if reference:
        y = sy(reference[0])
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-7:.2f}" text-anchor="end">{html.escape(reference[1])}</text>')
    for value, label in vertical_references or []:
        if value < x_min or value > x_max:
            continue
        x = sx(value)
        anchor = "end" if value > x_min + 0.72 * (x_max - x_min) else "start"
        label_x = x - 5 if anchor == "end" else x + 5
        parts.append(
            f'<line class="reference" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<text class="reference-label" x="{label_x:.2f}" y="{top+15}" '
            f'text-anchor="{anchor}">{html.escape(label)}</text>'
        )
    for label, points, color, dash in series:
        series_interval_kind = (interval_kinds or {}).get(label, interval_kind)
        interval_lookup = {
            x: (low, high, significant)
            for x, low, high, significant in (intervals or {}).get(label, [])
        }
        for x, y in points:
            if x not in interval_lookup:
                continue
            low, high, _ = interval_lookup[x]
            x_pos = sx(x)
            low_pos, high_pos = sy(low), sy(high)
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos:.2f}" y1="{high_pos:.2f}" '
                f'x2="{x_pos:.2f}" y2="{low_pos:.2f}" stroke="{color}"/>'
            )
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos-4:.2f}" y1="{high_pos:.2f}" '
                f'x2="{x_pos+4:.2f}" y2="{high_pos:.2f}" stroke="{color}"/>'
            )
            parts.append(
                f'<line class="ci-whisker" x1="{x_pos-4:.2f}" y1="{low_pos:.2f}" '
                f'x2="{x_pos+4:.2f}" y2="{low_pos:.2f}" stroke="{color}"/>'
            )
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline class="series-line" points="{coords}" stroke="{color}"{dash_attr}/>')
        for x, y in points:
            interval = interval_lookup.get(x)
            if interval is None:
                marker_class = "series-dot"
                marker_fill = color
                marker_stroke = "#fff"
                marker_radius = 2.6
                title_suffix = ""
            else:
                low, high, significant = interval
                marker_class = "ci-dot"
                descriptive_interval = series_interval_kind in {"binomial", "bootstrap"}
                marker_fill = color if (significant or descriptive_interval) else "#fff"
                marker_stroke = "#fff" if (significant or descriptive_interval) else color
                marker_radius = 3.2
                if series_interval_kind == "binomial":
                    title_suffix = f" · pointwise 95% Wilson CI [{low:.3f}, {high:.3f}]"
                elif series_interval_kind == "bootstrap":
                    title_suffix = f" · 95% seed-bootstrap CI [{low:.3f}, {high:.3f}]"
                else:
                    title_suffix = (
                        f" · 95% CI [{low:.3f}, {high:.3f}]"
                        f" · exact sign-flip p {'< 0.05' if significant else '≥ 0.05'}"
                    )
            x_tooltip = (
                f"{100 * x:.2f}% eligible heads"
                if x_as_percent
                else f"{html.escape(x_value_prefix)}{int(x)}"
            )
            parts.append(
                f'<circle class="{marker_class}" cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{marker_radius:.1f}" '
                f'fill="{marker_fill}" stroke="{marker_stroke}"><title>{html.escape(label)} · '
                f'{x_tooltip} · {y:.3f}{html.escape(title_suffix)}</title></circle>'
            )
    legend_x = left
    for label, _, color, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line class="legend-line" x1="{legend_x}" y1="18" x2="{legend_x+24}" y2="18" stroke="{color}"{dash_attr}/>')
        parts.append(f'<text class="legend-label" x="{legend_x+30}" y="22">{html.escape(label)}</text>')
        legend_x += 28 + max(84, len(label) * 7)
    parts.extend(
        [
            f'<text class="axis-label" x="{left + plot_w/2:.2f}" y="{height-10}" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="axis-label" transform="translate(17 {top + plot_h/2:.2f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_bar_chart(
    title: str,
    x_label: str,
    rows: list[tuple[str, float, str]],
    *,
    domain: tuple[float, float] | None = None,
    references: list[tuple[float, str]] | None = None,
    width: int = 780,
    row_height: int = 40,
) -> str:
    height = 74 + row_height * len(rows)
    values = [value for _, value, _ in rows]
    lo, hi = domain or (min(0.0, min(values)), max(values) * 1.12)
    left, right, top, bottom = 210, 48, 30, 44
    plot_w = width - left - right

    def sx(value: float) -> float:
        return left + (value - lo) / max(hi - lo, 1e-9) * plot_w

    zero = sx(0.0)
    parts = [
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{height-top-bottom}"/>',
        f'<line class="zero" x1="{zero:.2f}" y1="{top}" x2="{zero:.2f}" y2="{height-bottom}"/>',
    ]
    ticks = [lo + i * (hi - lo) / 4 for i in range(5)]
    for tick in ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle">{tick:.2f}</text>')
    for value, label in references or []:
        x = sx(value)
        parts.append(f'<line class="reference" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="reference-label" x="{x+5:.2f}" y="{top+13}">{html.escape(label)}</text>')
    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_height + 8
        x = min(zero, sx(value))
        bar_w = abs(sx(value) - zero)
        parts.append(f'<text class="bar-label" x="{left-12}" y="{y+17}" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y}" width="{bar_w:.2f}" height="22" fill="{color}"><title>{value:.4f}</title></rect>')
        anchor = "start" if value >= 0 else "end"
        dx = 7 if value >= 0 else -7
        parts.append(f'<text class="bar-value" x="{sx(value)+dx:.2f}" y="{y+17}" text-anchor="{anchor}">{value:.3f}</text>')
    parts.append(f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-7}" text-anchor="middle">{html.escape(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_interval_chart(
    title: str,
    x_label: str,
    rows: list[tuple[str, float, float, float, str]],
    *,
    domain: tuple[float, float] | None = None,
    reference: tuple[float, str] = (0.0, "no matched-control advantage"),
    width: int = 820,
    row_height: int = 44,
) -> str:
    """Return a horizontal point-and-interval chart for paired contrasts."""

    height = 78 + row_height * len(rows)
    observed = [value for _, mean, low, high, _ in rows for value in (mean, low, high)]
    observed.append(reference[0])
    if domain is None:
        lo, hi = min(observed), max(observed)
        pad = max((hi - lo) * 0.12, 0.002)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = domain
    left, right, top, bottom = 260, 62, 34, 46
    plot_w = width - left - right

    def sx(value: float) -> float:
        return left + (value - lo) / max(hi - lo, 1e-9) * plot_w

    parts = [
        f'<svg class="interval-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{height-top-bottom}"/>',
    ]
    for tick in [lo + i * (hi - lo) / 4 for i in range(5)]:
        x = sx(tick)
        parts.append(
            f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" '
            f'text-anchor="middle">{tick:.3f}</text>'
        )
    reference_x = sx(reference[0])
    parts.append(
        f'<line class="reference" x1="{reference_x:.2f}" y1="{top}" '
        f'x2="{reference_x:.2f}" y2="{height-bottom}"/>'
    )
    parts.append(
        f'<text class="reference-label" x="{reference_x+5:.2f}" y="{top+14}">'
        f'{html.escape(reference[1])}</text>'
    )
    for index, (label, mean, low, high, color) in enumerate(rows):
        y = top + index * row_height + 24
        parts.append(
            f'<text class="bar-label" x="{left-12}" y="{y+5:.2f}" '
            f'text-anchor="end">{html.escape(label)}</text>'
        )
        parts.append(
            f'<line class="ci-whisker" x1="{sx(low):.2f}" y1="{y:.2f}" '
            f'x2="{sx(high):.2f}" y2="{y:.2f}" stroke="{color}"/>'
        )
        for bound in (low, high):
            x = sx(bound)
            parts.append(
                f'<line class="ci-whisker" x1="{x:.2f}" y1="{y-6:.2f}" '
                f'x2="{x:.2f}" y2="{y+6:.2f}" stroke="{color}"/>'
            )
        parts.append(
            f'<circle cx="{sx(mean):.2f}" cy="{y:.2f}" r="5.5" fill="{color}">'
            f'<title>mean={mean:.6f}; 95% CI [{low:.6f}, {high:.6f}]</title></circle>'
        )
        anchor = "start" if sx(mean) < width - right - 66 else "end"
        dx = 9 if anchor == "start" else -9
        parts.append(
            f'<text class="bar-value" x="{sx(mean)+dx:.2f}" y="{y+5:.2f}" '
            f'text-anchor="{anchor}">{mean:+.3f}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-7}" '
        f'text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_map_error_chart(
    model: str,
    rows: list[dict[str, str]],
    *,
    width: int = 660,
    height: int = 390,
) -> str:
    """Plot predictive and refit errors for adjacent-layer rank-3 maps."""

    rows = sorted(rows, key=lambda row: int(row["target_layer"]))
    x_values = [int(row["target_layer"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.003, 10.0
    left, right, top, bottom = 78, 24, 44, 60
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sy(value: float) -> float:
        value = min(max(float(value), y_min), y_max)
        return top + (log_max - math.log10(value)) / (log_max - log_min) * plot_h

    stable = {
        int(row["target_layer"]): (
            float(row["cv_centroid_r2"]) >= 0.9
            and float(row["bootstrap_map_relative_frobenius_median"]) <= 0.1
        )
        for row in rows
    }
    series: list[tuple[str, str, str, list[tuple[int, float, int]]]] = [
        (
            "CV centroid NRMSE",
            "#0f766e",
            "",
            [
                (
                    int(row["target_layer"]),
                    float(row["cv_centroid_normalized_rmse"]),
                    int(row["source_layer"]),
                )
                for row in rows
            ],
        ),
        (
            "bootstrap map error",
            "#7c3aed",
            "6 4",
            [
                (
                    int(row["target_layer"]),
                    float(row["bootstrap_map_relative_frobenius_median"]),
                    int(row["source_layer"]),
                )
                for row in rows
            ],
        ),
    ]
    parts = [
        f'<svg class="line-chart map-error-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} adjacent-layer rank-3 map errors across depth">',
        f'<title>{html.escape(model)} adjacent-layer rank-3 map errors</title>',
        '<desc>Two log-scale errors across answer-query layer boundaries. '
        'Lower green values indicate better held-out centroid prediction; lower purple values indicate '
        'better reproducibility after seed resampling and PCA-gauge alignment.</desc>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    half_step = plot_w / max(x_max - x_min, 1) / 2
    for target_layer, is_stable in stable.items():
        if not is_stable:
            continue
        x = max(left, sx(target_layer) - half_step)
        x2 = min(width - right, sx(target_layer) + half_step)
        parts.append(
            f'<rect class="stable-band" x="{x:.2f}" y="{top}" width="{x2-x:.2f}" height="{plot_h}">'
            f'<title>L{target_layer-1}→L{target_layer}: both local-error cutoffs pass</title></rect>'
        )
    y_ticks = (0.003, 0.01, 0.03, 0.1, math.sqrt(0.1), 1.0, 3.0, 10.0)
    for tick in y_ticks:
        y = sy(tick)
        label = f"{tick:.3f}" if tick < 0.01 else (f"{tick:.2f}" if tick < 1 else f"{tick:g}")
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{label}</text>')
    x_ticks = sorted(
        {
            x_min,
            x_max,
            *[round(x_min + i * (x_max - x_min) / 5) for i in range(1, 5)],
        }
    )
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick}</text>')
    for value, label in ((math.sqrt(0.1), "CV R²=.90"), (0.1, "bootstrap=.10")):
        y = sy(value)
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-6:.2f}" text-anchor="end">{label}</text>')
    for label, color, dash, points in series:
        coords = " ".join(f"{sx(target):.2f},{sy(value):.2f}" for target, value, _ in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline class="series-line" points="{coords}" stroke="{color}"{dash_attr}/>')
        for target, value, source in points:
            local = "pass" if stable[target] else "fail"
            parts.append(
                f'<circle class="series-dot" cx="{sx(target):.2f}" cy="{sy(value):.2f}" r="2.6" '
                f'fill="{color}" stroke="#fff"><title>{html.escape(model)} L{source}→L{target}; '
                f'{html.escape(label)}={value:.4f}; local rule={local}</title></circle>'
            )
    legend_x = left
    for label, color, dash, _ in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line class="legend-line" x1="{legend_x}" y1="18" x2="{legend_x+24}" y2="18" stroke="{color}"{dash_attr}/>')
        parts.append(f'<text class="legend-label" x="{legend_x+30}" y="22">{html.escape(label)}</text>')
        legend_x += 32 + len(label) * 7
    parts.extend(
        [
            f'<rect class="stable-band legend-band" x="{legend_x}" y="11" width="14" height="12"/>',
            f'<text class="legend-label" x="{legend_x+20}" y="22">both cutoffs pass</text>',
            f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-10}" text-anchor="middle">Adjacent answer-query boundary (target layer shown)</text>',
            f'<text class="axis-label" transform="translate(17 {top+plot_h/2:.2f}) rotate(-90)" text-anchor="middle">Relative error (log; lower is better)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_map_cosine_chart(
    model: str,
    rows: list[dict[str, str]],
    *,
    width: int = 660,
    height: int = 390,
) -> str:
    """Plot gauge-invariant orientation continuity of consecutive map operators."""

    rows = sorted(rows, key=lambda row: int(row["target_layer"]))
    points = [
        (
            int(row["target_layer"]),
            float(row["full_operator_cosine_to_next"]),
            int(row["source_layer"]),
        )
        for row in rows
        if row["full_operator_cosine_to_next"]
    ]
    x_values = [int(row["target_layer"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = -0.10, 1.05
    left, right, top, bottom = 78, 24, 44, 60
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg class="line-chart map-cosine-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} consecutive full-operator cosine across depth">',
        f'<title>{html.escape(model)} consecutive full-operator cosine</title>',
        '<desc>Frobenius cosine between each gauge-invariant ambient rank-3 map operator and the '
        'operator at the next answer-query boundary. Higher values mean more similar operator orientation; '
        'one means the same direction up to a positive scalar.</desc>',
        f'<rect class="plot-bg" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>',
    ]
    for tick in (-0.10, 0.0, 0.25, 0.50, 0.75, 1.0):
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>')
    x_ticks = sorted(
        {
            x_min,
            x_max,
            *[round(x_min + i * (x_max - x_min) / 5) for i in range(1, 5)],
        }
    )
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<line class="grid vertical" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick}</text>')
    for value, label in ((1.0, "same orientation up to scale"), (0.0, "Frobenius-orthogonal")):
        y = sy(value)
        parts.append(f'<line class="reference" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="reference-label" x="{width-right-4}" y="{y-6:.2f}" text-anchor="end">{label}</text>')
    coords = " ".join(f"{sx(target):.2f},{sy(value):.2f}" for target, value, _ in points)
    parts.append(f'<polyline class="series-line" points="{coords}" stroke="#d97706"/>')
    for target, value, source in points:
        parts.append(
            f'<circle class="series-dot" cx="{sx(target):.2f}" cy="{sy(value):.2f}" r="2.8" '
            f'fill="#d97706" stroke="#fff"><title>{html.escape(model)}: T(L{source}→L{target}) '
            f'vs T(L{target}→L{target+1}); full-operator cosine={value:.4f}</title></circle>'
        )
    parts.extend(
        [
            '<line class="legend-line" x1="78" y1="18" x2="102" y2="18" stroke="#d97706"/>',
            '<text class="legend-label" x="108" y="22">full-operator cosine to next boundary</text>',
            f'<text class="axis-label" x="{left+plot_w/2:.2f}" y="{height-10}" text-anchor="middle">Adjacent answer-query boundary (target layer shown)</text>',
            f'<text class="axis-label" transform="translate(17 {top+plot_h/2:.2f}) rotate(-90)" text-anchor="middle">Cosine to next boundary (higher is closer)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _blend_rgb(start: tuple[int, int, int], end: tuple[int, int, int], value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    channels = [round(left + (right - left) * value) for left, right in zip(start, end)]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def svg_accuracy_heatmap(model: str, rows: list[dict[str, str]]) -> str:
    """Render per-count exact accuracy and mean absolute error as two heat rows."""

    rows = sorted(rows, key=lambda row: int(row["gold_count"]))
    width, height = 820, 210
    left, right, top = 132, 22, 42
    cell_w = (width - left - right) / len(rows)
    cell_h = 46
    maximum_error = max(float(row["mean_absolute_error"]) for row in rows) or 1.0
    parts = [
        f'<svg class="baseline-heatmap" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} baseline exact accuracy and mean absolute error by gold needle count">',
        f"<title>{html.escape(model)} baseline by gold needle count</title>",
        f'<text class="heat-title" x="{left}" y="22">{html.escape(model)}</text>',
    ]
    for index, row in enumerate(rows):
        x = left + index * cell_w
        count = int(row["gold_count"])
        accuracy = float(row["accuracy"])
        error = float(row["mean_absolute_error"])
        prediction = float(row["mean_prediction"])
        acc_fill = _blend_rgb((241, 245, 249), (15, 118, 110), accuracy)
        err_fill = _blend_rgb((255, 247, 237), (180, 35, 24), error / maximum_error)
        parts.extend(
            [
                f'<text class="heat-x" x="{x + cell_w / 2:.2f}" y="37" text-anchor="middle">{count}</text>',
                f'<rect class="heat-cell" x="{x:.2f}" y="{top}" width="{cell_w:.2f}" height="{cell_h}" fill="{acc_fill}">'
                f'<title>N={count}; exact accuracy={accuracy:.3f}; examples={row["examples"]}</title></rect>',
                f'<text class="heat-value {"inverse" if accuracy > 0.53 else ""}" x="{x + cell_w / 2:.2f}" y="{top + 29}" text-anchor="middle">{100 * accuracy:.0f}%</text>',
                f'<rect class="heat-cell" x="{x:.2f}" y="{top + cell_h + 8}" width="{cell_w:.2f}" height="{cell_h}" fill="{err_fill}">'
                f'<title>N={count}; mean absolute error={error:.3f} counts; mean prediction={prediction:.3f}</title></rect>',
                f'<text class="heat-value {"inverse" if error / maximum_error > 0.55 else ""}" x="{x + cell_w / 2:.2f}" y="{top + cell_h + 37}" text-anchor="middle">{error:.2f}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="heat-row" x="{left - 10}" y="{top + 29}" text-anchor="end">Exact accuracy</text>',
            f'<text class="heat-row" x="{left - 10}" y="{top + cell_h + 37}" text-anchor="end">Mean |ŷ−N|</text>',
            f'<text class="axis-label" x="{left + (width-left-right)/2:.2f}" y="{height-8}" text-anchor="middle">Gold needle count N</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_head_score_map(
    model: str,
    atlas_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
) -> str:
    """Render a layer-by-head map of answer-query broad-retrieval score."""

    values = {
        (int(row["layer"]), int(row["head"])): float(row["pool_primary"])
        for row in atlas_rows
    }
    selected = {
        (int(row["layer"]), int(row["head"])): int(row["rank"])
        for row in selected_rows
    }
    layers = list(range(max(layer for layer, _ in values) + 1))
    heads = list(range(max(head for _, head in values) + 1))
    maximum = max(values.values()) or 1.0
    width = 1010
    left, right, top, bottom = 66, 28, 40, 66
    cell_w = (width - left - right) / len(heads)
    cell_h = 15 if len(layers) >= 36 else 18
    height = top + bottom + cell_h * len(layers)
    parts = [
        f'<svg class="attention-map" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(model)} layer by head broad retrieval score map at the answer query">',
        f"<title>{html.escape(model)} answer-query broad retrieval score map</title>",
    ]
    for layer in layers:
        y = top + layer * cell_h
        for head in heads:
            x = left + head * cell_w
            score = values.get((layer, head), 0.0)
            fill = _blend_rgb((247, 250, 252), (15, 118, 110), score / maximum)
            rank = selected.get((layer, head))
            class_name = "attention-cell selected" if rank else "attention-cell"
            parts.append(
                f'<rect class="{class_name}" x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}">'
                f'<title>L{layer}H{head}; broad score={score:.4f}'
                f'{f"; frozen rank={rank}" if rank else ""}</title></rect>'
            )
            if rank and cell_w >= 20:
                parts.append(
                    f'<text class="heat-rank" x="{x + cell_w/2:.2f}" y="{y + cell_h - 3:.2f}" text-anchor="middle">{rank}</text>'
                )
    head_step = 2 if len(heads) > 16 else 1
    for head in heads[::head_step]:
        x = left + (head + 0.5) * cell_w
        parts.append(f'<text class="heat-x" x="{x:.2f}" y="{height-bottom+19}" text-anchor="middle">{head}</text>')
    layer_step = 4 if len(layers) > 38 else 3
    for layer in layers[::layer_step]:
        y = top + (layer + 0.5) * cell_h + 4
        parts.append(f'<text class="heat-y" x="{left-9}" y="{y:.2f}" text-anchor="end">{layer}</text>')
    legend_x = left
    legend_y = height - 23
    for index in range(6):
        value = maximum * index / 5
        fill = _blend_rgb((247, 250, 252), (15, 118, 110), index / 5)
        parts.append(f'<rect x="{legend_x + index*44}" y="{legend_y-11}" width="44" height="10" fill="{fill}"/>')
    parts.extend(
        [
            f'<text class="heat-legend" x="{legend_x}" y="{legend_y+13}">0</text>',
            f'<text class="heat-legend" x="{legend_x+264}" y="{legend_y+13}" text-anchor="end">{maximum:.2f}</text>',
            f'<text class="heat-legend" x="{legend_x+282}" y="{legend_y+5}">broad score B</text>',
            f'<text class="axis-label" x="{left + (width-left-right)/2:.2f}" y="{height-bottom+42}" text-anchor="middle">Attention head H (zero-based)</text>',
            f'<text class="axis-label" transform="translate(17 {top + cell_h*len(layers)/2:.2f}) rotate(-90)" text-anchor="middle">Transformer layer L (zero-based)</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def render_attention_token_example(payload: dict[str, Any]) -> str:
    """Render one audited natural answer-query attention row at token resolution."""

    selection = payload["selection"]
    attention = payload["attention"]
    category_mass = attention["category_mass"]
    maximum = max(float(attention["max_token_attention"]), 1e-12)
    category_labels = {
        "active_needle": "active needles",
        "ordinary_passage": "ordinary passage",
        "hard_negative": "nearby controls",
        "prompt_wrapper_or_instruction": "instruction / wrapper",
        "answer_query": "answer query",
    }

    summary = "".join(
        '<div class="attention-summary-item">'
        f'<strong>{html.escape(category_labels[key])}</strong>'
        f'<span>{100 * float(category_mass[key]):.2f}%</span>'
        '<div class="attention-mass-track"><i '
        f'style="width:{min(100.0, 100 * float(category_mass[key])):.2f}%"></i></div>'
        '</div>'
        for key in (
            "active_needle",
            "ordinary_passage",
            "hard_negative",
            "prompt_wrapper_or_instruction",
            "answer_query",
        )
    )

    needle_rows: list[str] = []
    for row in attention["needle_rows"]:
        token_html: list[str] = []
        for token in row["tokens"]:
            weight = float(token["attention"])
            intensity = 0.06 + 0.88 * math.sqrt(max(weight, 0.0) / maximum)
            display = str(token["display"]).replace("\u2028", "↵").replace("\u2029", "↵")
            title = (
                f"position {int(token['position'])}; attention={weight:.6f}; "
                f"token={display!r}"
            )
            foreground = "#ffffff" if intensity >= 0.58 else "#5b1b16"
            token_html.append(
                '<span class="attention-token" '
                f'style="background:rgba(180,35,24,{intensity:.3f});color:{foreground}" '
                f'title="{html.escape(title, quote=True)}">{html.escape(display)}</span>'
            )
        needle_rows.append(
            '<div class="needle-attention-row">'
            '<div class="needle-attention-meta">'
            f'<strong>N{int(row["slot_index"])} · {html.escape(str(row["city"]))}</strong>'
            f'<span>score {int(row["score"])} · span mass {100 * float(row["attention_mass"]):.2f}%</span>'
            '</div>'
            f'<div class="needle-token-line">{"".join(token_html)}</div>'
            '</div>'
        )

    top_non_needle = "".join(
        "<li>"
        f'<code>pos {int(token["position"])}</code> '
        f'<span class="token-region">{html.escape(category_labels.get(str(token["region"]), str(token["region"])))}</span> '
        f'<strong>{html.escape(str(token["display"]).replace(chr(0x2028), "↵").replace(chr(0x2029), "↵"))}</strong> '
        f'({100 * float(token["attention"]):.2f}%)'
        "</li>"
        for token in attention["top_non_needle_tokens"][:12]
    )
    return (
        '<div class="attention-example">'
        '<div class="attention-example-head">'
        f'<strong>Qwen3-8B · L{int(selection["layer"])}H{int(selection["head"])} · '
        f'seed {int(selection["seed"])} · gold N={int(selection["gold_count"])}</strong>'
        '<span>自然 forward 的最后一个 answer-query attention row</span>'
        '</div>'
        f'<div class="attention-summary">{summary}</div>'
        '<div class="attention-token-legend"><span></span>颜色越深，单个 token 获得的 attention weight 越大；'
        '悬停可看 token position 与精确权重。</div>'
        f'<div class="attention-needle-list">{"".join(needle_rows)}</div>'
        '<details class="collapsible-list"><summary>展开：attention 最高的 non-needle tokens</summary>'
        f'<ol class="top-token-list">{top_non_needle}</ol></details>'
        '</div>'
    )


def render_attention_gallery_controls(
    payload: dict[str, Any],
    gallery_id: str,
) -> str:
    heads = payload["selection"]["heads"]
    prompts = payload["selection"]["prompts"]
    head_options = "".join(
        f'<option value="L{int(row["layer"])}H{int(row["head"])}">'
        f'rank {int(row["frozen_rank"])} · L{int(row["layer"])}H{int(row["head"])}</option>'
        for row in heads
    )
    prompt_options = "".join(
        f'<option value="V4_4_T10000_N{int(row["gold_count"])}_seed{int(row["seed"])}"'
        f'{" selected" if int(row["gold_count"]) == 6 else ""}>'
        f'seed {int(row["seed"])} · gold N={int(row["gold_count"])}</option>'
        for row in prompts
    )
    return (
        '<div class="attention-gallery-controls">'
        f'<label for="{gallery_id}-head">Frozen head<select id="{gallery_id}-head" '
        f'data-gallery-head>{head_options}</select></label>'
        f'<label for="{gallery_id}-prompt">Natural prompt<select id="{gallery_id}-prompt" '
        f'data-gallery-prompt>{prompt_options}</select></label>'
        '<span>选择只改变展示，不重新拟合或重跑统计检验。</span>'
        '</div>'
    )


def render_attention_span_overview(
    record: dict[str, Any],
    *,
    global_span_max: float,
    global_bin_max: float,
    bin_width_tokens: int,
) -> str:
    """Render an audited full-document routing map plus per-needle summaries."""

    selection = record["selection"]
    prompt = record["prompt"]
    attention = record["attention"]
    sequence_length = int(prompt["sequence_length"])
    needle_rows = attention["needle_rows"]
    hard_negative_rows = attention["hard_negative_rows"]
    bins = attention["bins"]
    category_mass = attention["category_mass"]

    category_specs = (
        ("active_needle", "active needles", "#d84a3f"),
        ("ordinary_passage", "ordinary haystack", "#2f8a83"),
        ("hard_negative", "nearby controls", "#dca33a"),
        ("prompt_wrapper_or_instruction", "instruction / wrapper", "#98a2b3"),
        ("answer_query", "answer query", "#587da5"),
    )
    category_total = sum(float(category_mass[key]) for key, _, _ in category_specs)
    category_segments = "".join(
        '<span class="attention-category-segment" '
        f'style="width:{100 * float(category_mass[key]) / max(category_total, 1e-12):.6f}%;'
        f'background:{colour}" title="{html.escape(label)}: '
        f'{100 * float(category_mass[key]):.4f}%"></span>'
        for key, label, colour in category_specs
    )
    category_legend = "".join(
        '<span class="attention-category-key">'
        f'<i style="background:{colour}"></i><strong>{html.escape(label)}</strong>'
        f'<b>{100 * float(category_mass[key]):.2f}%</b></span>'
        for key, label, colour in category_specs
    )

    # The source artifact stores exact total attention for every 64-token bin.
    # Bins fully inside the passage and containing no registered special span
    # are exact ordinary-haystack bins; mixed bins are explicitly marked.
    def bin_colour(value: float) -> str:
        ratio = min(1.0, max(0.0, value / max(global_bin_max, 1e-12)))
        # A monotone log transform makes the low-mass haystack visible while
        # retaining a much stronger pale-rose -> deep-red visual separation.
        display_ratio = math.log1p(28.0 * ratio) / math.log1p(28.0)
        return _blend_rgb((255, 248, 246), (139, 36, 28), 0.03 + 0.97 * display_ratio)

    bin_cells: list[str] = []
    for bin_index, row in enumerate(bins):
        token_start = int(row["token_start"])
        token_end = int(row["token_end"])
        value = float(row["attention_mass"])
        needle_hits = [
            span
            for span in needle_rows
            if token_start < int(span["token_end"]) and token_end > int(span["token_start"])
        ]
        negative_hits = [
            span
            for span in hard_negative_rows
            if token_start < int(span["token_end"]) and token_end > int(span["token_start"])
        ]
        fully_in_passage = (
            token_start >= int(prompt["passage_token_start"])
            and token_end <= int(prompt["passage_token_end"])
        )
        clean_haystack = fully_in_passage and not needle_hits and not negative_hits
        contains_query = token_start <= int(prompt["query_position"]) < token_end
        fill = bin_colour(value)
        classes = ["attention-bin"]
        if clean_haystack:
            classes.append("is-clean-haystack")
        if needle_hits:
            classes.append("has-needle")
        if negative_hits:
            classes.append("has-negative")
        if not fully_in_passage:
            classes.append("is-boundary")
        if contains_query:
            classes.append("has-query")
        hit_labels = ", ".join(
            f'N{int(span["slot_index"])} {span["city"]}' for span in needle_hits
        )
        region_note = (
            "clean ordinary-haystack bin"
            if clean_haystack
            else "mixed/special bin; total mass, not ordinary-only mass"
        )
        title = (
            f'bin {bin_index + 1}; tokens {token_start}–{token_end-1}; '
            f'total attention={100 * value:.4f}%; {region_note}'
            f'{"; " + hit_labels if hit_labels else ""}'
        )
        city_labels = "".join(
            '<span class="attention-bin-city">'
            f'{html.escape(str(span["city"]))}</span>'
            for span in needle_hits
            if token_start <= int(span["token_start"]) < token_end
        )
        bin_cells.append(
            f'<div class="{" ".join(classes)}" title="{html.escape(title, quote=True)}">'
            f'{city_labels}<i class="attention-bin-tile" style="background:{fill}"></i></div>'
        )

    legend_swatches = "".join(
        f'<i style="background:{bin_colour(level / 4 * global_bin_max)}"></i>'
        for level in range(5)
    )

    width = 1080
    bar_left, bar_right = 164, 72
    bar_plot_width = width - bar_left - bar_right
    row_height = 31
    bar_top = 58
    bar_height = bar_top + row_height * len(needle_rows) + 52

    def mass_x(value: float) -> float:
        return bar_left + bar_plot_width * value / max(global_span_max, 1e-12)

    bar_parts = [
        f'<svg class="attention-document-map attention-span-summary" '
        f'viewBox="0 0 {width} {bar_height}" role="img" '
        'aria-label="Attention mass assigned to each complete needle span">',
        '<text class="attention-panel-label" x="24" y="25">B · 每条完整 needle span 的 attention mass</text>',
    ]
    for fraction in (0.0, 0.5, 1.0):
        value = global_span_max * fraction
        xx = mass_x(value)
        bar_parts.extend(
            [
                f'<line class="grid vertical" x1="{xx:.2f}" x2="{xx:.2f}" y1="42" '
                f'y2="{bar_top + row_height * len(needle_rows) - 6}"/>',
                f'<text class="tick" x="{xx:.2f}" y="49" text-anchor="middle">{100*value:.1f}%</text>',
            ]
        )
    for index, span in enumerate(needle_rows):
        value = float(span["attention_mass"])
        yy = bar_top + index * row_height
        bar_width = max(1.0, mass_x(value) - bar_left)
        label = f'N{int(span["slot_index"])} · {html.escape(str(span["city"]))}'
        bar_parts.extend(
            [
                f'<text class="attention-span-row-label" x="{bar_left-12}" y="{yy+16}" '
                f'text-anchor="end">{label}</text>',
                f'<rect class="attention-span-bar-bg" x="{bar_left}" y="{yy+3}" '
                f'width="{bar_plot_width}" height="18"/>',
                f'<rect class="attention-span-bar" x="{bar_left}" y="{yy+3}" '
                f'width="{bar_width:.2f}" height="18"><title>{label}; '
                f'tokens {int(span["token_start"])}–{int(span["token_end"])-1}; '
                f'{100*value:.3f}% of the complete attention row</title></rect>',
                f'<text class="attention-bar-value" '
                f'x="{min(width-bar_right+8, bar_left+bar_width+8):.2f}" '
                f'y="{yy+17}">{100*value:.2f}%</text>',
            ]
        )
    bar_parts.extend(
        [
            f'<text class="axis-label" x="{bar_left+bar_plot_width/2:.2f}" '
            f'y="{bar_height-12}" text-anchor="middle">Span attention mass（占完整 answer-query attention row 的百分比）</text>',
            "</svg>",
        ]
    )

    return (
        '<div class="attention-overview">'
        '<div class="attention-example-head">'
        f'<strong>Qwen3-8B · rank {int(selection["frozen_head_rank"])} · '
        f'L{int(selection["layer"])}H{int(selection["head"])} · seed {int(selection["seed"])} · '
        f'gold N={int(selection["gold_count"])}</strong>'
        f'<span>needles {100*float(category_mass["active_needle"]):.1f}% · '
        f'ordinary haystack {100*float(category_mass["ordinary_passage"]):.1f}% · '
        f'nearby controls {100*float(category_mass["hard_negative"]):.1f}%</span>'
        "</div>"
        '<div class="attention-category-block">'
        '<div class="attention-section-heading"><strong>整条 attention row 的区域组成</strong>'
        '<span>同一分母下的精确 mass；五类之和约为 100%</span></div>'
        f'<div class="attention-category-bar">{category_segments}</div>'
        f'<div class="attention-category-legend">{category_legend}</div>'
        "</div>"
        '<div class="attention-raster-block">'
        '<div class="attention-section-heading"><strong>A · 全文分箱 attention landscape</strong>'
        f'<span>全文 {sequence_length:,} tokens；{bin_width_tokens} tokens / 格；从左到右，换行后继续</span></div>'
        f'<div class="attention-bin-scroll"><div class="attention-bin-grid">'
        f'{"".join(bin_cells)}</div></div>'
        '<div class="attention-bin-legend">'
        f'<span class="attention-bin-scale">低 {legend_swatches} 高（全 panel passage-bin 最大 '
        f'{100 * global_bin_max:.2f}% / 格）</span>'
        '<span>格子颜色 = 格内<strong>总 attention mass</strong>（对数色阶；悬停看精确值）</span>'
        '<span class="legend-needle"></span><span>红上边与城市名 = active needle</span>'
        '<span class="legend-negative"></span><span>黄下边 = nearby length-matched control</span>'
        '<span class="legend-clean"></span><span>无标记且完整位于 passage 内 = ordinary-haystack bin</span>'
        "</div></div>"
        f'{"".join(bar_parts)}'
        "</div>"
    )


def render_attention_gallery(
    payload: dict[str, Any],
    *,
    gallery_id: str,
    mode: str,
) -> str:
    """Render fixed gallery records with client-side head/prompt selectors."""

    if mode not in {"overview", "tokens"}:
        raise ValueError(f"Unsupported attention gallery mode: {mode}")
    default_prompt = "V4_4_T10000_N6_seed1254"
    default_head = "L27H18"
    raw_span_max = max(
        float(span["attention_mass"])
        for record in payload["records"]
        for span in record["attention"]["needle_rows"]
    )
    global_span_max = max(0.05, math.ceil(raw_span_max / 0.05) * 0.05)
    global_passage_bin_max = max(
        float(bin_row["attention_mass"])
        for record in payload["records"]
        for bin_row in record["attention"]["bins"]
        if int(bin_row["token_start"]) >= int(record["prompt"]["passage_token_start"])
        and int(bin_row["token_end"]) <= int(record["prompt"]["passage_token_end"])
    )
    panels: list[str] = []
    for record in payload["records"]:
        selection = record["selection"]
        prompt_key = str(selection["stimulus_id"])
        head_key = f'L{int(selection["layer"])}H{int(selection["head"])}'
        active = prompt_key == default_prompt and head_key == default_head
        content = (
            render_attention_span_overview(
                record,
                global_span_max=global_span_max,
                global_bin_max=global_passage_bin_max,
                bin_width_tokens=int(payload["display"]["bin_width_tokens"]),
            )
            if mode == "overview"
            else render_attention_token_example(record)
        )
        panels.append(
            f'<div class="attention-gallery-panel" data-gallery-panel '
            f'data-head="{head_key}" data-prompt="{prompt_key}"'
            f'{"" if active else " hidden"}>{content}</div>'
        )
    return (
        f'<div class="attention-gallery" data-attention-gallery id="{gallery_id}">'
        f'{render_attention_gallery_controls(payload, gallery_id)}'
        f'{"".join(panels)}'
        '</div>'
    )


def classifier_data() -> dict[str, list[dict[str, str]]]:
    root = REPORTS / "v4_4_extension" / "classification"
    paths = {
        "Qwen3-8B": root / "classification_all_qwen" / "answer_classifier_metrics.csv",
        "Gemma4-E4B": root / "classification_all_gemma" / "answer_classifier_metrics.csv",
    }
    reported_algorithms = {"logistic_l2", "nearest_centroid"}
    return {
        key: [
            row for row in read_csv(path)
            if row["algorithm"] in reported_algorithms
        ]
        for key, path in paths.items()
    }


def build(output: Path) -> None:
    extension = REPORTS / "v4_4_extension"
    base_html = (
        REPORTS
        / "v4_4_3"
        / "realistic_niah_v4_4_mechanism_report.html"
    ).read_text(encoding="utf-8")
    prompt_all = extract_embedded_json(base_html, "PROMPT_DATA")
    answer_all = extract_embedded_json(base_html, "ANSWER_DATA")
    # Keep all layers selectable while stripping unused PC4--PC6 columns from
    # the embedded rows.  The first nine columns contain metadata plus PC1--3.
    display_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "prompt": {"Qwen3-8B": {}, "Gemma4-E4B": {}},
        "answer": {"Qwen3-8B": {}, "Gemma4-E4B": {}},
    }
    for key, payload in prompt_all.items():
        model, layer = key.split("|")
        display_data["prompt"][model][layer] = {
            "layer": int(layer),
            "rows": [row[:9] for row in payload["rows"]],
        }
    for key, payload in answer_all.items():
        model, layer, cohort = key.split("|")
        if cohort != "all":
            continue
        display_data["answer"][model][layer] = {
            "layer": int(layer),
            "rows": [row[:9] for row in payload["rows"]],
        }

    classifiers = classifier_data()
    classifier_oof_paths = {
        "Qwen3-8B": (
            extension
            / "classification"
            / "classification_all_qwen"
            / "answer_classifier_oof_predictions.csv.gz"
        ),
        "Gemma4-E4B": (
            extension
            / "classification"
            / "classification_all_gemma"
            / "answer_classifier_oof_predictions.csv.gz"
        ),
    }
    classifier_display_layers = {"Qwen3-8B": 29, "Gemma4-E4B": 37}
    classifier_by_count: dict[str, list[dict[str, float | int]]] = {}
    for model, path in classifier_oof_paths.items():
        rows = [
            row
            for row in read_csv_gz(path)
            if row["algorithm"] == "logistic_l2"
            and int(row["layer"]) == classifier_display_layers[model]
        ]
        if len(rows) != 200:
            raise RuntimeError(
                f"Expected 200 grouped-OOF classifier rows for {model}; got {len(rows)}"
            )
        per_count: list[dict[str, float | int]] = []
        for count in range(1, 11):
            group = [row for row in rows if int(row["gold_count"]) == count]
            if len(group) != 20:
                raise RuntimeError(
                    f"Expected 20 held-out classifier rows for {model} N={count}; got {len(group)}"
                )
            correct = sum(int(row["predicted_count"]) == count for row in group)
            mae = sum(
                abs(int(row["predicted_count"]) - count) for row in group
            ) / len(group)
            low, high = wilson_interval(correct, len(group))
            per_count.append(
                {
                    "count": count,
                    "correct": correct,
                    "total": len(group),
                    "accuracy": correct / len(group),
                    "mae": mae,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        classifier_by_count[model] = per_count
    classifier_count_chart = svg_line_chart(
        "Answer-query classifier accuracy by gold count",
        "Grouped-OOF exact-count accuracy",
        [
            (
                model,
                [
                    (float(row["count"]), float(row["accuracy"]))
                    for row in classifier_by_count[model]
                ],
                color,
                dash,
            )
            for model, color, dash in (
                ("Qwen3-8B", QWEN_COLOR, ""),
                ("Gemma4-E4B", GEMMA_COLOR, "6 5"),
            )
        ],
        x_label="Gold count N",
        x_value_prefix="N=",
        x_ticks=list(range(1, 11)),
        y_domain=(0.0, 1.05),
        reference=(0.10, "ten-class chance"),
        intervals={
            model: [
                (
                    float(row["count"]),
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                    True,
                )
                for row in classifier_by_count[model]
            ]
            for model in classifier_by_count
        },
        interval_kind="binomial",
        width=780,
        height=350,
    )
    classifier_count_table_rows = "".join(
        "<tr>"
        f"<td>N={count}</td>"
        + "".join(
            f"<td>{int(next(row for row in classifier_by_count[model] if row['count'] == count)['correct'])}/20 "
            f"({pct(next(row for row in classifier_by_count[model] if row['count'] == count)['accuracy'])})</td>"
            f"<td>{f(next(row for row in classifier_by_count[model] if row['count'] == count)['mae'])}</td>"
            for model in ("Qwen3-8B", "Gemma4-E4B")
        )
        + "</tr>"
        for count in range(1, 11)
    )
    regression = read_csv(extension / "geometry" / "count_regression_summary.csv")
    ranks = read_csv(extension / "geometry" / "rank_and_compression_by_layer.csv")
    clusters = read_csv(extension / "geometry" / "clustering_summary.csv")
    counter_rows = read_csv(
        extension / "counter_properties" / "counter_property_metrics_by_layer.csv"
    )
    counter_selected = read_csv(
        extension / "counter_properties" / "selected_layer_counter_properties.csv"
    )
    gated_formula_rows = read_csv(
        extension / "all_token" / "gated_curve_formula_tests.csv"
    )
    followup_root = REPORTS / "v4_4_5_followup"
    followup = json.loads(
        (followup_root / "campaign_summary.json").read_text(encoding="utf-8")
    )
    top6_followup_root = REPORTS / "v4_4_5_top6_followup"
    top6_config_diff_audit = json.loads(
        (top6_followup_root / "config_diff_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        top6_config_diff_audit.get("status") != "PASS"
        or len(top6_config_diff_audit.get("comparisons", [])) != 4
        or any(
            row.get("only_allowed_changes") is not True
            or sha256_file(ROOT / str(row["original"]))
            != str(row["original_sha256"])
            or sha256_file(ROOT / str(row["extension"]))
            != str(row["extension_sha256"])
            for row in top6_config_diff_audit.get("comparisons", [])
        )
    ):
        raise RuntimeError("Gemma Top-6 config-difference audit failed")
    top6_audit = json.loads(
        (top6_followup_root / "FINAL_AUDIT.json").read_text(encoding="utf-8")
    )
    expected_gemma_top6 = [
        [29, 4],
        [35, 2],
        [35, 7],
        [35, 1],
        [35, 3],
        [29, 2],
    ]
    expected_top6_followups = {
        "config_diff_audit",
        "attention_response",
        "retrieval_geometry",
        "retrieval_subspace_L29",
        "serial_mediation",
        "retrieval_subspace_L35",
        "induction_broad_readout",
        "noise_outside_context_broad_readout",
        "behavior_invariance_audit",
        "original_preservation_audit",
    }
    if (
        top6_audit.get("status") != "PASS"
        or top6_audit.get("model") != "Gemma4-E4B"
        or top6_audit.get("global_frozen_top6") != expected_gemma_top6
        or set(top6_audit.get("completed", [])) != expected_top6_followups
        or top6_audit.get("original_outputs_overwritten") is not False
    ):
        raise RuntimeError("Gemma Top-6 follow-up campaign failed its final audit")
    top6_invariance_audit = json.loads(
        (top6_followup_root / "behavior_invariance_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        top6_invariance_audit.get("status") != "PASS"
        or len(top6_invariance_audit.get("comparisons", [])) != 4
        or {
            row.get("label")
            for row in top6_invariance_audit.get("comparisons", [])
        }
        != {
            "induction_canonical",
            "induction_synthetic",
            "noise_factorial",
            "outside_context",
        }
        or any(
            row.get("non_bank_behavior_exactly_equal") is not True
            or row.get("allowed_fields_present_both") is not True
            or row.get("expected_bank_change_observed") is not True
            for row in top6_invariance_audit.get("comparisons", [])
        )
    ):
        raise RuntimeError("Top-6 auxiliary-readout invariance audit failed")
    top6_preservation_audit = json.loads(
        (top6_followup_root / "original_preservation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        top6_preservation_audit.get("status") != "PASS"
        or len(top6_preservation_audit.get("comparisons", [])) != 14
        or any(
            row.get("unchanged") is not True
            for row in top6_preservation_audit.get("comparisons", [])
        )
    ):
        raise RuntimeError("Historical Top-8 artifact preservation audit failed")

    top6_geometry_audit = json.loads(
        (
            top6_followup_root
            / "retrieval_geometry"
            / "geometry_audit.json"
        ).read_text(encoding="utf-8")
    )
    if (
        top6_geometry_audit.get("status") != "PASS"
        or top6_geometry_audit.get("bank_construction")
        != "sum_registered_per_head_post_o_writes"
        or top6_geometry_audit.get("registered_heads", {}).get("Gemma4-E4B")
        != expected_gemma_top6
    ):
        raise RuntimeError("Gemma Top-6 retrieval geometry failed its audit")
    top6_geometry_rows = read_csv(
        top6_followup_root
        / "retrieval_geometry"
        / "retrieval_geometry_metrics.csv"
    )
    if len(top6_geometry_rows) != 2:
        raise RuntimeError("Gemma Top-6 retrieval geometry must contain two layers")
    followup["retrieval_geometry"]["rows"] = [
        row
        for row in followup["retrieval_geometry"]["rows"]
        if row["model_label"] == "Qwen3-8B"
    ] + [
        {
            "model_label": row["model_label"],
            "layer": int(row["layer"]),
            "exact_classifier_accuracy": float(row["exact_classifier_accuracy"]),
            "nearest_centroid_accuracy": float(row["nearest_centroid_accuracy"]),
            "classifier_mad": float(row["exact_classifier_mad"]),
        }
        for row in top6_geometry_rows
    ]

    top6_subspace_rows: list[dict[str, Any]] = []
    top6_subspace_clean: list[dict[str, Any]] = []
    for layer in (29, 35):
        layer_root = top6_followup_root / "retrieval_subspace" / f"Gemma_L{layer}"
        layer_audit = json.loads(
            (layer_root / "analysis_audit.json").read_text(encoding="utf-8")
        )
        if (
            layer_audit.get("status") != "PASS"
            or int(layer_audit.get("rows", -1)) != 400
            or int(layer_audit.get("paired_units", -1)) != 100
        ):
            raise RuntimeError(f"Gemma Top-6 retrieval L{layer} failed its audit")
        layer_raw = read_csv(layer_root / "raw_detail.csv")
        expected_layer_heads = [4, 2] if layer == 29 else [2, 7, 1, 3]
        if (
            len(layer_raw) != 400
            or any(
                json.loads(row["retrieval_heads"]) != expected_layer_heads
                or int(row["retrieval_layer"]) != layer
                or int(row["source_patch_layer"]) != 8
                for row in layer_raw
            )
        ):
            raise RuntimeError(
                f"Gemma Top-6 retrieval L{layer} raw membership failed audit"
            )
        layer_summary = read_csv(layer_root / "effect_summary.csv")
        all_row = pick(layer_summary, model_label="Gemma4-E4B", population="all")
        top6_subspace_rows.append(
            {
                "model_label": "Gemma4-E4B",
                "layer": layer,
                "natural_specificity_mean": float(
                    all_row["mean_natural_expected_error_specificity"]
                ),
                "restoration_mediation_mean": float(
                    all_row["mean_restoration_mediation_expected_error"]
                ),
                "mediated_fraction_mean": float(
                    all_row["mean_estimated_mediated_fraction"]
                ),
                "natural_specificity_median": float(
                    all_row["median_natural_expected_error_specificity"]
                ),
                "restoration_mediation_median": float(
                    all_row["median_restoration_mediation_expected_error"]
                ),
                "mediated_fraction_median": float(
                    all_row["median_estimated_mediated_fraction"]
                ),
                "natural_strict_error_specificity": float(
                    all_row["mean_natural_strict_error_specificity"]
                ),
                "natural_accuracy_damage_specificity": float(
                    all_row["mean_natural_accuracy_damage_specificity"]
                ),
                "restoration_strict_error_specificity": float(
                    all_row["mean_restoration_mediation_strict_error"]
                ),
                "restoration_accuracy_damage_specificity": float(
                    all_row["mean_restoration_mediation_accuracy_damage"]
                ),
            }
        )
        if layer == 29:
            clean_row = pick(
                layer_summary,
                model_label="Gemma4-E4B",
                population="clean_correct",
            )
            top6_subspace_clean.append(
                {
                    "model_label": "Gemma4-E4B",
                    "layer": layer,
                    "paired_units": int(clean_row["rows"]),
                    "restoration_mediation_mean": float(
                        clean_row["mean_restoration_mediation_expected_error"]
                    ),
                    "mediated_fraction_mean": float(
                        clean_row["mean_estimated_mediated_fraction"]
                    ),
                }
            )
    followup["retrieval_subspace"]["rows"] = [
        row
        for row in followup["retrieval_subspace"]["rows"]
        if row["model_label"] == "Qwen3-8B"
    ] + top6_subspace_rows
    followup["retrieval_subspace"]["clean_correct_robustness"] = [
        row
        for row in followup["retrieval_subspace"]["clean_correct_robustness"]
        if row["model_label"] == "Qwen3-8B"
    ] + top6_subspace_clean
    model_order = ("Qwen3-8B", "Gemma4-E4B")
    exp19: dict[str, dict[str, Any]] = {}
    exp22: dict[str, dict[str, Any]] = {}
    exp22_registration: dict[str, dict[str, Any]] = {}
    exp22_synthetic: dict[str, dict[str, Any]] = {}
    exp23: dict[str, dict[str, Any]] = {}
    exp23_registration: dict[str, dict[str, Any]] = {}
    for model in model_order:
        model_followup_root = (
            top6_followup_root if model == "Gemma4-E4B" else followup_root
        )
        exp19_path = (
            model_followup_root
            / ("serial_mediation" if model == "Gemma4-E4B" else "exp19")
            / model
            / "serial_summary.json"
        )
        exp19_payload = json.loads(
            exp19_path.read_text(encoding="utf-8")
        )
        exp19_audit = json.loads(
            exp19_path.with_name("analysis_audit.json").read_text(encoding="utf-8")
        )
        expected_broad_rows = 2200 if model == "Gemma4-E4B" else 7700
        if (
            exp19_payload.get("status") != "PASS"
            or set(exp19_payload.get("models", {})) != {model}
            or exp19_audit.get("status") != "PASS"
            or int(exp19_audit.get("detail_rows", -1)) != 1100
            or int(exp19_audit.get("unique_detail_keys", -1)) != 1100
            or int(exp19_audit.get("paired_units", -1)) != 100
            or int(exp19_audit.get("broad_rows", -1)) != expected_broad_rows
            or int(exp19_audit.get("expected_broad_rows", -1))
            != expected_broad_rows
            or int(exp19_audit.get("hook_failures", -1)) != 0
        ):
            raise RuntimeError(f"Experiment 19 summary failed audit for {model}")
        cluster_schema = (
            exp19_payload.get("schema_version")
            == "realistic_niah_v4_4_5_serial_mediation_summary_v2"
            and exp19_audit.get("schema_version")
            == "realistic_niah_v4_4_5_serial_mediation_analysis_audit_v2"
        )
        if cluster_schema:
            inference = exp19_payload.get("inference", {})
            if (
                inference.get("independent_unit") != "seed"
                or inference.get("bootstrap_method")
                != "seed-cluster percentile bootstrap"
                or int(inference.get("bootstrap_draws", -1)) != 10000
                or exp19_audit.get("inference_unit") != "seed"
                or int(exp19_audit.get("seed_clusters_per_model", -1)) != 10
                or int(exp19_audit.get("counts_per_seed_cluster", -1)) != 10
                or int(exp19_audit.get("independent_seed_clusters", -1)) != 10
                or int(exp19_audit.get("paired_seed_count_units", -1)) != 100
                or int(exp19_audit.get("bootstrap_draws", -1)) != 10000
                or exp19_audit.get("bootstrap_method")
                != "seed-cluster percentile bootstrap"
                or exp19_audit.get("exact_sign_flip_unit") != "seed"
                or not exp19_path.with_name("paired_serial_effects.csv").is_file()
            ):
                raise RuntimeError(
                    f"Experiment 19 seed-cluster contract failed for {model}"
                )
            for metric_name, metric in exp19_payload["models"][model].items():
                if metric_name == "ordered_criterion_diagnostics":
                    continue
                if (
                    not isinstance(metric, dict)
                    or int(metric.get("units", -1)) != 10
                    or int(metric.get("independent_seed_clusters", -1)) != 10
                    or int(metric.get("paired_seed_count_units", -1)) != 100
                    or int(metric.get("bootstrap_draws", -1)) != 10000
                    or metric.get("bootstrap_method")
                    != "seed-cluster percentile bootstrap"
                    or metric.get("cluster_column") != "seed"
                    or int(metric.get("exact_sign_flip_permutations", -1)) != 1024
                    or "exact_sign_flip_p_two_sided" not in metric
                ):
                    raise RuntimeError(
                        f"Experiment 19 clustered metric failed for "
                        f"{model}/{metric_name}"
                    )
        else:
            raise RuntimeError(
                f"Experiment 19 requires retained paired effects and valid "
                f"seed-cluster inference for {model}"
            )
        ordered = exp19_payload["models"][model][
            "ordered_criterion_diagnostics"
        ]
        if not (
            ordered["i_source_changes_retrieval"]["source_repair_positive"]
            and ordered["ii_retrieval_precedes_late"][
                "retrieval_mediation_positive"
            ]
            and ordered["iii_late_affects_output_not_earlier_readout"][
                "late_mediation_positive"
            ]
            and float(
                ordered["iii_late_affects_output_not_earlier_readout"][
                    "retrieval_invariance_max_over_units"
                ]
            )
            == 0.0
        ):
            raise RuntimeError(
                f"Experiment 19 ordered criteria no longer pass for {model}"
            )
        if model == "Gemma4-E4B":
            exp19_provenance = json.loads(
                exp19_path.with_name("run_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                exp19_provenance.get("retrieval_heads") != [4, 2]
                or int(exp19_provenance.get("retrieval_layer", -1)) != 29
                or int(exp19_provenance.get("source_layer", -1)) != 9
                or int(exp19_provenance.get("late_answer_layer", -1)) != 37
                or exp19_provenance.get("stimuli_sha256")
                != "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
                or exp19_provenance.get("experiment_config_sha256")
                != sha256_file(
                    ROOT
                    / "configs"
                    / "realistic_niah_v4_4_5_serial_mediation_top6_extension.json"
                )
                or exp19_provenance.get("retrieval_basis_sha256")
                != sha256_file(
                    top6_followup_root
                    / "retrieval_geometry"
                    / "retrieval_bases.pt"
                )
            ):
                raise RuntimeError("Gemma Top-6 experiment 19 provenance failed")
        exp19[model] = exp19_payload["models"][model]

        exp22_dir = (
            model_followup_root
            / ("induction_v3" if model == "Gemma4-E4B" else "exp22_v3")
            / model
        )
        exp22[model] = json.loads(
            (exp22_dir / "analysis_summary.json").read_text(encoding="utf-8")
        )
        exp22_registration[model] = json.loads(
            (exp22_dir / "canonical_registration.json").read_text(encoding="utf-8")
        )
        exp22_synthetic[model] = json.loads(
            (exp22_dir / "synthetic_audit.json").read_text(encoding="utf-8")
        )
        if (
            exp22[model].get("status") != "PASS"
            or exp22[model].get("model") != model
            or exp22[model].get("scientific_decision") != "not_supported"
            or not exp22[model].get("synthetic_relation_gate")
            or exp22[model].get("canonical_matched_block_gate")
        ):
            raise RuntimeError(f"Unexpected experiment 22 verdict for {model}")
        if model == "Gemma4-E4B":
            exp22_complete = json.loads(
                (exp22_dir / "complete.json").read_text(encoding="utf-8")
            )
            exp22_provenance = json.loads(
                (exp22_dir / "run_provenance.json").read_text(encoding="utf-8")
            )
            if (
                exp22_complete.get("status") != "complete"
                or int(exp22_complete.get("canonical_rows", -1)) != 300
                or int(exp22_complete.get("synthetic_rows", -1)) != 1200
                or exp22_provenance.get("stimuli_sha256")
                != "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
                or exp22_provenance.get("experiment_config_sha256")
                != sha256_file(
                    ROOT
                    / "configs"
                    / "realistic_niah_v4_4_5_induction_circuit_top6_extension.json"
                )
            ):
                raise RuntimeError("Gemma Top-6 experiment 22 provenance failed")

        exp23_dir = (
            model_followup_root
            / ("noise_factorial_v2" if model == "Gemma4-E4B" else "exp23_v2")
            / model
        )
        exp23[model] = json.loads(
            (exp23_dir / "analysis_summary.json").read_text(encoding="utf-8")
        )
        exp23_registration[model] = json.loads(
            (exp23_dir / "outside_context_registration.json").read_text(
                encoding="utf-8"
            )
        )
        exp23_audit = json.loads(
            (exp23_dir / "analysis_audit.json").read_text(encoding="utf-8")
        )
        if (
            exp23[model].get("status") != "PASS"
            or exp23[model].get("model") != model
            or exp23[model]["outside_context"].get("candidate_exceeds_both_controls")
            or exp23_audit != {
                "status": "PASS",
                "factorial_rows": 240,
                "outside_context_rows": 400,
            }
        ):
            raise RuntimeError(f"Unexpected experiment 23 audit or verdict for {model}")
        if model == "Gemma4-E4B":
            exp23_complete = json.loads(
                (exp23_dir / "complete.json").read_text(encoding="utf-8")
            )
            exp23_provenance = json.loads(
                (exp23_dir / "run_provenance.json").read_text(encoding="utf-8")
            )
            if (
                exp23_complete.get("status") != "complete"
                or int(exp23_complete.get("factorial_rows", -1)) != 240
                or int(exp23_complete.get("outside_context_rows", -1)) != 400
                or exp23_provenance.get("stimuli_sha256")
                != "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
                or exp23_provenance.get("experiment_config_sha256")
                != sha256_file(
                    ROOT
                    / "configs"
                    / "realistic_niah_v4_4_5_noise_factorial_top6_extension.json"
                )
            ):
                raise RuntimeError("Gemma Top-6 experiment 23 provenance failed")
    span_layerwise = json.loads(
        (
            REPORTS
            / "v4_4_5_followup"
            / "span_restoration"
            / "layerwise_seed_statistics.json"
        ).read_text(encoding="utf-8")
    )
    if span_layerwise.get("status") != "PASS":
        raise RuntimeError("Dense span layerwise seed statistics did not pass audit")
    state_effect_root = (
        REPORTS / "v4_4_5_followup" / "span_state_deformation"
    )
    state_effect_layerwise = read_csv(
        state_effect_root / "layerwise_state_deformation.csv"
    )
    state_effect_summary = json.loads(
        (state_effect_root / "summary.json").read_text(encoding="utf-8")
    )
    state_effect_audit = json.loads(
        (state_effect_root / "analysis_audit.json").read_text(encoding="utf-8")
    )
    if (
        state_effect_audit.get("status") != "PASS"
        or int(state_effect_audit.get("layerwise_rows", -1)) != 78
        or int(state_effect_audit.get("expected_layerwise_rows", -1)) != 78
        or state_effect_summary.get("schema_version")
        != "realistic_niah_v4_4_5_span_state_deformation_analysis_v1"
    ):
        raise RuntimeError("Span-state deformation analysis did not pass audit")
    state_effect_counts = {
        model: sum(row["model_label"] == model for row in state_effect_layerwise)
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    if state_effect_counts != {"Qwen3-8B": 36, "Gemma4-E4B": 42}:
        raise RuntimeError(
            f"Unexpected span-state deformation layer coverage: {state_effect_counts}"
        )
    state_retention_root = REPORTS / "v4_4_5_followup" / "state_retention"
    state_retention_layerwise = read_csv(
        state_retention_root / "analysis" / "layerwise_state_retention.csv"
    )
    state_retention_window = read_csv(
        state_retention_root / "analysis" / "window_state_retention.csv"
    )
    state_retention_levels = read_csv(
        state_retention_root / "analysis" / "window_condition_levels.csv"
    )
    state_retention_summary = json.loads(
        (state_retention_root / "analysis" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    state_retention_selection = json.loads(
        (state_retention_root / "analysis" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    state_retention_audit = json.loads(
        (state_retention_root / "analysis" / "analysis_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        state_retention_audit.get("status") != "PASS"
        or int(state_retention_audit.get("layerwise_rows", -1)) != 780
        or int(state_retention_audit.get("expected_layerwise_rows", -1)) != 780
        or int(state_retention_audit.get("window_rows", -1)) != 20
        or int(state_retention_audit.get("expected_window_rows", -1)) != 20
        or len(state_retention_layerwise) != 780
        or len(state_retention_window) != 20
        or len(state_retention_levels) != 64
        or state_retention_summary.get("schema_version")
        != "realistic_niah_v4_4_5_state_retention_analysis_v1"
        or state_retention_selection.get("selected_candidate")
        != "span_end_correct_distance_specificity"
        or not state_retention_selection.get("cross_model_supported_candidate_found")
    ):
        raise RuntimeError("State-retention analysis did not pass frozen audit")
    for model, expected_rows in (("Qwen3-8B", 720), ("Gemma4-E4B", 840)):
        run_audit = json.loads(
            (state_retention_root / "formal" / model / "complete.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            run_audit.get("status") != "PASS"
            or int(run_audit.get("layer_rows", -1)) != expected_rows
            or int(run_audit.get("unique_layer_keys", -1)) != expected_rows
            or not run_audit.get("sequence_and_query_preserved")
            or not run_audit.get("token_budget_matched")
        ):
            raise RuntimeError(f"State-retention run audit failed for {model}")
    original_span_attention_response = read_csv(
        REPORTS
        / "v4_4_5_followup"
        / "span_restoration"
        / "attention_response_canonical.csv"
    )
    top6_attention_audit = json.loads(
        (
            top6_followup_root
            / "attention_response"
            / "attention_response_audit.json"
        ).read_text(encoding="utf-8")
    )
    if (
        top6_attention_audit.get("status") != "PASS"
        or top6_attention_audit.get("model") != "Gemma4-E4B"
        or int(top6_attention_audit.get("head_count", -1)) != 6
        or int(top6_attention_audit.get("layers", -1)) != 42
        or int(top6_attention_audit.get("seeds", -1)) != 30
        or sorted(
            [int(row["layer"]), int(row["head"])]
            for row in top6_attention_audit.get("frozen_heads", [])
        )
        != sorted(expected_gemma_top6)
        or top6_attention_audit.get("experiment_config_sha256")
        != sha256_file(
            ROOT
            / "configs"
            / "realistic_niah_v4_4_5_span_restoration_canonical_top6_extension.json"
        )
        or top6_attention_audit.get("source_broad_summary_sha256")
        != "bd4c958f6248621f8eac5242b25011b2d57d9aae02b533a3617d70b9b04d608a"
    ):
        raise RuntimeError("Gemma Top-6 attention response failed its audit")
    top6_span_attention_response = read_csv(
        top6_followup_root
        / "attention_response"
        / "attention_response_topk.csv"
    )
    if top6_attention_audit.get("output_sha256") != sha256_file(
        top6_followup_root
        / "attention_response"
        / "attention_response_topk.csv"
    ):
        raise RuntimeError("Gemma Top-6 attention response hash mismatch")
    span_attention_response = [
        row
        for row in original_span_attention_response
        if row["model_label"] == "Qwen3-8B"
    ] + top6_span_attention_response
    attention_response_counts = {
        model: sum(row["model_label"] == model for row in span_attention_response)
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    if attention_response_counts != {"Qwen3-8B": 36, "Gemma4-E4B": 42}:
        raise RuntimeError(
            "Unexpected canonical attention-response coverage: "
            f"{attention_response_counts}"
        )
    for model, expected_layers, expected_heads in (
        ("Qwen3-8B", set(range(36)), 32),
        ("Gemma4-E4B", set(range(42)), 6),
    ):
        rows = [row for row in span_attention_response if row["model_label"] == model]
        if {int(row["patch_layer"]) for row in rows} != expected_layers:
            raise RuntimeError(f"Incomplete canonical attention layers for {model}")
        if any(
            int(row["frozen_heads"]) != expected_heads
            or int(row["seeds"]) != 30
            or row["counts"] != "1-10"
            for row in rows
        ):
            raise RuntimeError(f"Unexpected canonical attention provenance for {model}")

    # Question 21 is answered by the earlier V4.4.2 cue-removal grid.  The
    # intervention deletes only the two opening definition sentences; it keeps
    # the passage, counting question, numeric-output instruction, and assistant
    # formatting fixed.  Parse the frozen embedded analysis so the appendix
    # cannot drift from the original paired result.
    cue_report_path = (
        REPORTS
        / "v4_4_2"
        / "realistic_niah_v4_4_2_mode_geometry_attention_report.html"
    )
    cue_report = cue_report_path.read_text(encoding="utf-8")
    cue_prompt_geometry = extract_embedded_json(cue_report, "PROMPT_GEOM")
    if cue_prompt_geometry.get("schema_version") != "realistic_niah_v4_4_2_prompt_counter_geometry_v1":
        raise RuntimeError("Unexpected V4.4.2 prompt-cue geometry schema")
    cue_expected_coverage = {
        "Qwen3-8B": {"paired_prompts": 10, "paired_endpoint_states": 100, "layers": 36, "seeds": 10},
        "Gemma4-E4B": {"paired_prompts": 10, "paired_endpoint_states": 100, "layers": 42, "seeds": 10},
    }
    for model, expected in cue_expected_coverage.items():
        observed = cue_prompt_geometry["coverage"][model]
        if any(int(observed[key]) != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected V4.4.2 cue-removal coverage for {model}: {observed}")
        if int(observed["prompt_gold_count"]) != 10:
            raise RuntimeError(f"Cue-removal prompt counter must use final N=10 for {model}")
    cue_statistics = list(cue_prompt_geometry["statistics"].values())
    cue_display_layers = {"Qwen3-8B": 8, "Gemma4-E4B": 9}
    cue_display_rows = {
        model: cue_prompt_geometry["statistics"][f"{model}|prompt_counter|{layer}"]
        for model, layer in cue_display_layers.items()
    }
    cue_cka_chart = svg_line_chart(
        "Opening-cue removal: running-index centroid topology across depth",
        "Linear CKA between cue-present and cue-absent centroids",
        [
            (
                model,
                [
                    (float(row["layer"]), float(row["centroid_cka"]))
                    for row in sorted(
                        [item for item in cue_statistics if item["model"] == model],
                        key=lambda item: int(item["layer"]),
                    )
                ],
                color,
                dash,
            )
            for model, color, dash in (
                ("Qwen3-8B", QWEN_COLOR, ""),
                ("Gemma4-E4B", GEMMA_COLOR, "6 5"),
            )
        ],
        x_ticks=[0, 8, 16, 24, 32, 40],
        y_domain=(0.94, 1.005),
        reference=(1.0, "identical centroid relations"),
        width=760,
        height=350,
    )
    cue_ridge_chart = svg_bar_chart(
        "Opening-cue removal: shallow running-index readout",
        "Seed-held-out ridge R²",
        [
            (
                f"Qwen L8 · cue present",
                float(cue_display_rows["Qwen3-8B"]["r2_present"]),
                QWEN_COLOR,
            ),
            (
                f"Qwen L8 · cue absent",
                float(cue_display_rows["Qwen3-8B"]["r2_absent"]),
                QWEN_LIGHT,
            ),
            (
                f"Gemma L9 · cue present",
                float(cue_display_rows["Gemma4-E4B"]["r2_present"]),
                GEMMA_COLOR,
            ),
            (
                f"Gemma L9 · cue absent",
                float(cue_display_rows["Gemma4-E4B"]["r2_absent"]),
                GEMMA_LIGHT,
            ),
        ],
        domain=(-0.05, 0.90),
        references=[(0.0, "R² = 0")],
        width=760,
    )
    cue_appendix_rows = "".join(
        f"""<tr><td>{model}</td><td>L{cue_display_layers[model]}</td>
        <td>{f(row['centroid_cka'], 4)}</td>
        <td>{f(row['r2_present'])} / {f(row['r2_absent'])}</td>
        <td>{f(row['count_eta_present'])} / {f(row['count_eta_absent'])}</td>
        <td>{f(row['interaction_eta_sq'])}</td></tr>"""
        for model, row in cue_display_rows.items()
    )

    display_layers = {
        ("Qwen3-8B", "prompt_running"): 8,
        ("Gemma4-E4B", "prompt_running"): 9,
        ("Qwen3-8B", "answer_query"): 29,
        ("Gemma4-E4B", "answer_query"): 37,
    }
    geometry_rows: list[dict[str, object]] = []
    for (model, role), layer in display_layers.items():
        reg = pick(regression, model_label=model, role=role, layer=layer, algorithm="ridge")
        rank_population = "discovery" if role == "prompt_running" else "all_available"
        rank = pick(
            ranks,
            model_label=model,
            role=role,
            layer=layer,
            fit_population=rank_population,
        )
        cluster = pick(clusters, model_label=model, role=role, layer=layer)
        logistic = (
            pick(classifiers[model], layer=layer, algorithm="logistic_l2")
            if role == "answer_query"
            else None
        )
        nearest_centroid = (
            pick(classifiers[model], layer=layer, algorithm="nearest_centroid")
            if role == "answer_query"
            else None
        )
        geometry_rows.append(
            {
                "model": model,
                "role": role,
                "layer": layer,
                "accuracy": (
                    float(logistic["accuracy"])
                    if logistic is not None
                    else None
                ),
                "class_mad": (
                    float(logistic["count_mae"])
                    if logistic is not None
                    else None
                ),
                "centroid_accuracy": (
                    float(nearest_centroid["accuracy"])
                    if nearest_centroid is not None
                    else None
                ),
                "centroid_mad": (
                    float(nearest_centroid["count_mae"])
                    if nearest_centroid is not None
                    else None
                ),
                "r2": float(reg["r2_mean"]),
                "reg_mad": float(reg["mae_mean"]),
                "stable_rank": float(rank["stable_rank"]),
                "rank3_all": float(rank["total_variance_capture_k3"]),
                "rank3_centroid": float(rank["centroid_curve_capture_k3"]),
                "eta2": float(rank["count_eta_squared"]),
                "silhouette": float(cluster["silhouette_cosine_mean"]),
            }
        )

    behavior: dict[str, dict[str, float]] = {}
    for model, model_layers in display_data["answer"].items():
        payload = model_layers[str(display_layers[(model, "answer_query")])]
        rows = payload["rows"]
        gold = [int(row[5]) for row in rows]
        prediction = [int(row[3]) for row in rows]
        errors = [pred - target for pred, target in zip(prediction, gold)]
        wrong = [abs(error) for error in errors if error != 0]
        behavior[model] = {
            "accuracy": sum(error == 0 for error in errors) / len(errors),
            "mad": sum(abs(error) for error in errors) / len(errors),
            "signed": sum(errors) / len(errors),
            "wrong_mad": sum(wrong) / len(wrong),
            "rows": float(len(rows)),
        }

    # Layerwise representation plots deliberately separate supervised
    # decodability from unsupervised rank-3 geometry.  They answer different
    # questions and should not be used as substitutes for one another.
    representation_charts: list[str] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        prompt_rows = sorted(
            [
                row for row in regression
                if row["model_label"] == model
                and row["role"] == "prompt_running"
                and row["algorithm"] == "ridge"
            ],
            key=lambda row: int(row["layer"]),
        )
        answer_logistic_rows = sorted(
            [row for row in classifiers[model] if row["algorithm"] == "logistic_l2"],
            key=lambda row: int(row["layer"]),
        )
        answer_centroid_rows = sorted(
            [row for row in classifiers[model] if row["algorithm"] == "nearest_centroid"],
            key=lambda row: int(row["layer"]),
        )
        prompt_rank_rows = sorted(
            [
                row for row in ranks
                if row["model_label"] == model
                and row["role"] == "prompt_running"
                and row["fit_population"] == "discovery"
            ],
            key=lambda row: int(row["layer"]),
        )
        answer_rank_rows = sorted(
            [
                row for row in ranks
                if row["model_label"] == model
                and row["role"] == "answer_query"
                and row["fit_population"] == "all_available"
            ],
            key=lambda row: int(row["layer"]),
        )
        representation_charts.append(
            '<div class="chart-pair">'
            + svg_line_chart(
                f"{model}: prompt running-index ridge",
                "Seed-held-out ridge R²",
                [("needle-end ridge", [(float(row["layer"]), float(row["r2_mean"])) for row in prompt_rows], "#d97706", "")],
                y_domain=(-0.1, 1.0),
                reference=(0.0, "R² = 0"),
            )
            + svg_line_chart(
                f"{model}: answer-query exact-count classification",
                "Seed-held-out accuracy",
                [
                    ("L2 logistic", [(float(row["layer"]), float(row["accuracy"])) for row in answer_logistic_rows], "#2563eb", ""),
                    ("nearest centroid", [(float(row["layer"]), float(row["accuracy"])) for row in answer_centroid_rows], "#0f766e", "5 4"),
                ],
                y_domain=(0.0, 0.75),
                reference=(0.1, "10-class chance = 0.10"),
            )
            + "</div>"
            + '<div class="chart-pair geometry-pair">'
            + svg_line_chart(
                f"{model}: prompt rank-3 geometry",
                "PCA rank-3 variance capture",
                [
                    ("all endpoint states", [(float(row["layer"]), float(row["total_variance_capture_k3"])) for row in prompt_rank_rows], "#d97706", ""),
                    ("count centroids", [(float(row["layer"]), float(row["centroid_curve_capture_k3"])) for row in prompt_rank_rows], "#7c3aed", "5 4"),
                ],
                y_domain=(0.0, 1.03),
            )
            + svg_line_chart(
                f"{model}: answer-query rank-3 geometry",
                "PCA rank-3 variance capture",
                [
                    ("all answer states", [(float(row["layer"]), float(row["total_variance_capture_k3"])) for row in answer_rank_rows], "#2563eb", ""),
                    ("count centroids", [(float(row["layer"]), float(row["centroid_curve_capture_k3"])) for row in answer_rank_rows], "#0f766e", "5 4"),
                ],
                y_domain=(0.0, 1.03),
            )
            + "</div>"
        )

    baseline_by_count = read_csv(REPORTS / "v4_4_causal_v2" / "baseline_by_count.csv")
    baseline_heatmaps = {
        model: svg_accuracy_heatmap(
            model,
            [row for row in baseline_by_count if row["model_label"] == model],
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    baseline_summary: dict[str, dict[str, float]] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        rows = [row for row in baseline_by_count if row["model_label"] == model]
        examples = sum(int(row["examples"]) for row in rows)
        baseline_summary[model] = {
            "examples": float(examples),
            "accuracy": sum(float(row["accuracy"]) * int(row["examples"]) for row in rows) / examples,
            "mad": sum(float(row["mean_absolute_error"]) * int(row["examples"]) for row in rows) / examples,
            "signed": sum(float(row["mean_signed_error"]) * int(row["examples"]) for row in rows) / examples,
        }

    token_stats = read_csv(extension / "token_corruption" / "token_corruption_statistics.csv")
    prompt_remove = read_csv(extension / "prompt_subspace_ablation" / "subspace_ablation_statistics.csv")
    formation_rows: list[tuple[str, float, str]] = []
    for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR)):
        token = pick(
            token_stats,
            population="all",
            model_label=model,
            endpoint="absolute_error_increase",
            estimand="specificity",
        )
        remove = pick(
            prompt_remove,
            population="all",
            model_label=model,
            condition="actual_rank3_remove",
            endpoint="absolute_error_increase",
        )
        formation_rows.extend(
            [
                (f"{model} · active-needle corruption", float(token["mean"]), color),
                (f"{model} · endpoint rank-3 removal", float(remove["mean"]), color),
            ]
        )
    formation_chart = svg_bar_chart(
        "Prompt evidence versus decoded endpoint subspace",
        "Control-adjusted increase in absolute count error",
        formation_rows,
        domain=(-0.5, 10.0),
    )

    earlier_qwen = sorted(
        read_csv(extension / "endpoint_attention_mask" / "earlier_span_head_confirmation.csv"),
        key=lambda row: float(row["confirmation_preference_mean"]),
        reverse=True,
    )[0]
    earlier_gemma = sorted(
        read_csv(extension / "endpoint_attention_mask" / "gemma_earlier_span_head_confirmation.csv"),
        key=lambda row: float(row["confirmation_preference_mean"]),
        reverse=True,
    )[0]

    topk_root = REPORTS / "v4_4_causal_v2" / "full_span_topk"
    topk = read_csv(topk_root / "full_span_topk_primary_statistics.csv")
    topk_membership = read_csv(topk_root / "full_span_topk_membership.csv")
    top6_ablation_root = REPORTS / "v4_4_causal_v2" / "full_span_topk_k6_extension"
    top6_ablation_audit = json.loads(
        (top6_ablation_root / "report_extension_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        top6_ablation_audit.get("status") != "PASS"
        or int(top6_ablation_audit.get("rows", -1)) != 400
        or top6_ablation_audit.get("global_frozen_top6")
        != ["L29H4", "L35H2", "L35H7", "L35H1", "L35H3", "L29H2"]
    ):
        raise RuntimeError("Gemma Top-6 ablation report extension failed its audit")
    topk.extend(read_csv(top6_ablation_root / "top6_primary_statistics.csv"))
    additions_root = REPORTS / "v4_4_report_additions"
    additions_audit = json.loads(
        (additions_root / "report_additions_audit.json").read_text(encoding="utf-8")
    )
    if additions_audit.get("status") != "PASS":
        raise RuntimeError("Report additions failed their source/coverage audit")
    raw_topk = read_csv(additions_root / "full_span_topk_raw_arms.csv")
    raw_topk.extend(read_csv(top6_ablation_root / "top6_raw_arms.csv"))
    attention_gallery_payload = json.loads(
        (additions_root / "qwen_attention_gallery.json").read_text(
            encoding="utf-8"
        )
    )
    attention_span_gallery = render_attention_gallery(
        attention_gallery_payload,
        gallery_id="attention-span-gallery",
        mode="overview",
    )
    attention_token_gallery = render_attention_gallery(
        attention_gallery_payload,
        gallery_id="attention-token-gallery",
        mode="tokens",
    )
    selected_heads = {
        "Qwen3-8B": [
            row for row in topk_membership
            if row["model_label"] == "Qwen3-8B" and row["top_n"] == "32"
        ],
        "Gemma4-E4B": [
            row for row in topk_membership
            if row["model_label"] == "Gemma4-E4B"
            and row["top_n"] == "8"
            and int(row["rank"]) <= 6
        ],
    }
    if {
        model: len(rows) for model, rows in selected_heads.items()
    } != {"Qwen3-8B": 32, "Gemma4-E4B": 6}:
        raise RuntimeError("Frozen broad-head display registry is not Qwen-32/Gemma-6")
    head_atlas = read_csv(REPORTS / "v4_4" / "realistic_niah_v4_head_atlas.csv")
    attention_maps: dict[str, str] = {}
    selected_head_tables: dict[str, str] = {}
    eligible_head_counts: dict[str, int] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        atlas_rows = [
            row for row in head_atlas
            if row["model"] == model
            and row["variant"] == "v4.4"
            and row["pooling"] == "span_sum"
        ]
        eligible_head_counts[model] = len(
            {(int(row["layer"]), int(row["head"])) for row in atlas_rows}
        )
        if eligible_head_counts[model] <= 0:
            raise RuntimeError(f"No discovery-eligible span-sum heads for {model}")
        attention_maps[model] = svg_head_score_map(model, atlas_rows, selected_heads[model])
        atlas_lookup = {
            (int(row["layer"]), int(row["head"])): row for row in atlas_rows
        }
        selected_head_tables[model] = "".join(
            f"<tr><td>{row['rank']}</td><td>{row['head_label']}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_sum'])}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_coverage'])}</td>"
            f"<td>{f(atlas_lookup[(int(row['layer']), int(row['head']))]['pool_primary'])}</td></tr>"
            for row in selected_heads[model]
        )
    retrieval_charts: dict[str, str] = {}
    retrieval_damage_charts: dict[str, str] = {}
    for model, color in (
        ("Qwen3-8B", QWEN_COLOR),
        ("Gemma4-E4B", GEMMA_COLOR),
    ):
        denominator = eligible_head_counts[model]
        rows = [
            row
            for row in topk
            if row["model_label"] == model and row["analysis_population"] == "all_examples_signed"
        ]
        rows.sort(key=lambda row: int(row["top_n"]))
        intervals = [
            (
                float(row["top_n"]) / denominator,
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                effect_interval_excludes_zero(row),
            )
            for row in rows
        ]
        raw_shift = {
            condition: sorted(
                [
                    row
                    for row in raw_topk
                    if row["model_label"] == model
                    and row["metric"] == "absolute_count_shift"
                    and row["condition"] == condition
                ],
                key=lambda row: int(row["top_n"]),
            )
            for condition in ("ranked", "layer_matched_random")
        }
        expected_doses = 7 if model == "Gemma4-E4B" else 6
        if any(
            len(condition_rows) != expected_doses
            for condition_rows in raw_shift.values()
        ):
            raise RuntimeError(f"Incomplete raw top-K shift arms for {model}")
        raw_shift_intervals = {
            label: [
                (
                    float(row["head_proportion"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                    True,
                )
                for row in raw_shift[condition]
            ]
            for label, condition in (
                ("ranked top-K (raw)", "ranked"),
                ("layer-matched random (raw)", "layer_matched_random"),
            )
        }
        max_prop = 32 / denominator
        ticks = [max_prop * index / 4 for index in range(5)]
        retrieval_charts[model] = svg_line_chart(
            f"{model} broad-head raw arms and ranked-minus-random contrast",
            "absolute count shift / contrast (counts)",
            [
                (
                    "ranked top-K (raw)",
                    [
                        (float(row["head_proportion"]), float(row["mean"]))
                        for row in raw_shift["ranked"]
                    ],
                    color,
                    "",
                ),
                (
                    "layer-matched random (raw)",
                    [
                        (float(row["head_proportion"]), float(row["mean"]))
                        for row in raw_shift["layer_matched_random"]
                    ],
                    "#64748b",
                    "7 4",
                ),
                (
                    "ranked − random",
                    [
                        (
                            float(row["top_n"]) / denominator,
                            float(row["primary_effect"]),
                        )
                        for row in rows
                    ],
                    "#8B5CF6",
                    "2 4",
                )
            ],
            x_label=f"Ablated share of discovery-eligible heads (H={denominator})",
            x_ticks=ticks,
            x_domain=(0.0, max_prop),
            y_domain=(-0.15, 2.45),
            reference=(0.0, "zero shift / zero contrast"),
            intervals={**raw_shift_intervals, "ranked − random": intervals},
            interval_kinds={
                "ranked top-K (raw)": "bootstrap",
                "layer-matched random (raw)": "bootstrap",
                "ranked − random": "effect",
            },
            x_as_percent=True,
            width=760,
            height=320,
        )

        damage_rows = [
            row
            for row in topk
            if row["model_label"] == model
            and row["analysis_population"] == "clean_correct_only"
        ]
        damage_rows.sort(key=lambda row: int(row["top_n"]))
        damage_intervals = [
            (
                float(row["top_n"]) / denominator,
                float(row["ci95_low"]),
                float(row["ci95_high"]),
                effect_interval_excludes_zero(row),
            )
            for row in damage_rows
        ]
        raw_damage = {
            condition: sorted(
                [
                    row
                    for row in raw_topk
                    if row["model_label"] == model
                    and row["metric"] == "clean_correct_to_wrong_rate"
                    and row["condition"] == condition
                ],
                key=lambda row: int(row["top_n"]),
            )
            for condition in ("ranked", "layer_matched_random")
        }
        if any(
            len(condition_rows) != expected_doses
            for condition_rows in raw_damage.values()
        ):
            raise RuntimeError(f"Incomplete raw top-K damage arms for {model}")
        raw_damage_intervals = {
            label: [
                (
                    float(row["head_proportion"]),
                    float(row["ci95_low"]),
                    float(row["ci95_high"]),
                    True,
                )
                for row in raw_damage[condition]
            ]
            for label, condition in (
                ("ranked top-K (raw)", "ranked"),
                ("layer-matched random (raw)", "layer_matched_random"),
            )
        }
        retrieval_damage_charts[model] = svg_line_chart(
            f"{model} clean-correct raw damage arms and contrast",
            "correct→wrong rate / contrast",
            [
                (
                    "ranked top-K (raw)",
                    [
                        (float(row["head_proportion"]), float(row["mean"]))
                        for row in raw_damage["ranked"]
                    ],
                    color,
                    "",
                ),
                (
                    "layer-matched random (raw)",
                    [
                        (float(row["head_proportion"]), float(row["mean"]))
                        for row in raw_damage["layer_matched_random"]
                    ],
                    "#64748b",
                    "7 4",
                ),
                (
                    "ranked − random",
                    [
                        (
                            float(row["top_n"]) / denominator,
                            float(row["primary_effect"]),
                        )
                        for row in damage_rows
                    ],
                    "#8B5CF6",
                    "2 4",
                )
            ],
            x_label=f"Ablated share of discovery-eligible heads (H={denominator})",
            x_ticks=ticks,
            x_domain=(0.0, max_prop),
            y_domain=(-0.12, 0.85),
            reference=(0.0, "zero damage / zero contrast"),
            intervals={**raw_damage_intervals, "ranked − random": damage_intervals},
            interval_kinds={
                "ranked top-K (raw)": "bootstrap",
                "layer-matched random (raw)": "bootstrap",
                "ranked − random": "effect",
            },
            x_as_percent=True,
            width=760,
            height=320,
        )

    topk_result_rows: list[str] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        doses = (1, 2, 4, 6, 8, 16, 32) if model == "Gemma4-E4B" else (1, 2, 4, 8, 16, 32)
        for top_n in doses:
            shift = pick(
                topk,
                model_label=model,
                top_n=top_n,
                analysis_population="all_examples_signed",
            )
            damage = pick(
                topk,
                model_label=model,
                top_n=top_n,
                analysis_population="clean_correct_only",
            )
            post_hoc = str(shift.get("post_hoc_extension", "")).lower() == "true"
            positive_seed_text = (
                "—"
                if post_hoc
                else str(
                    round(
                        float(shift["positive_seed_fraction"])
                        * int(shift["seed_clusters"])
                    )
                )
                + f"/{shift['seed_clusters']}"
            )
            shift_sig = effect_interval_excludes_zero(shift)
            damage_sig = effect_interval_excludes_zero(damage)
            shift_inference = (
                "是（post-hoc CI）"
                if post_hoc and shift_sig
                else (
                    f"{'是' if shift_sig else '否'} (p={p_text(shift['exact_sign_flip_p'])})"
                )
            )
            damage_inference = (
                "是（post-hoc CI）"
                if post_hoc and damage_sig
                else (
                    f"{'是' if damage_sig else '否'} (p={p_text(damage['exact_sign_flip_p'])})"
                )
            )
            topk_result_rows.append(
                "<tr>"
                f"<td>{model}</td><td>{top_n}{'†' if post_hoc else ''}</td>"
                f"<td>{float(shift['primary_effect']):+.3f} "
                f"[{f(shift['ci95_low'])}, {f(shift['ci95_high'])}]</td>"
                f"<td>{positive_seed_text}</td>"
                f"<td>{shift_inference}</td>"
                f"<td>{float(damage['primary_effect']):+.3f} "
                f"[{f(damage['ci95_low'])}, {f(damage['ci95_high'])}]</td>"
                f"<td>{damage_inference}</td>"
                "</tr>"
            )
    if len(topk_result_rows) != 13:
        raise RuntimeError("Top-K result table must contain 13 model-by-K rows")
    topk_result_table = "".join(topk_result_rows)

    answer_remove = read_csv(
        extension / "layerwise_subspace" / "answer_query_removal" / "layerwise_answer_query_removal_statistics.csv"
    )
    removal_series = []
    for model, color, dash in (
        ("Qwen3-8B", QWEN_COLOR, ""),
        ("Gemma4-E4B", GEMMA_COLOR, "6 5"),
    ):
        rows = [
            row
            for row in answer_remove
            if row["model_label"] == model
            and row["population"] == "all"
            and row["endpoint"] == "absolute_error_specificity"
        ]
        rows.sort(key=lambda row: int(row["layer"]))
        removal_series.append(
            (
                model,
                [(float(row["layer"]), float(row["mean_effect"])) for row in rows],
                color,
                dash,
            )
        )
    removal_chart = svg_line_chart(
        "Answer-query rank-3 removal across layers",
        "Δ absolute error vs orthogonal removal",
        removal_series,
        y_domain=(-0.25, 1.45),
        reference=(0.0, "no specificity"),
        width=760,
        height=350,
    )

    patch_rows = read_csv(REPORTS / "v4_4" / "v4_4_answer_query_patching.csv")
    patch_series = []
    for model, color, dash in (
        ("Qwen3-8B", QWEN_COLOR, ""),
        ("Gemma4-E4B", GEMMA_COLOR, "6 5"),
    ):
        rows = sorted(
            [row for row in patch_rows if row["model"] == model],
            key=lambda row: int(row["layer"]),
        )
        patch_series.append(
            (
                model,
                [(float(row["layer"]), float(row["eligible_donor_adoption_rate"])) for row in rows],
                color,
                dash,
            )
        )
    patch_chart = svg_line_chart(
        "Answer-state patching onset",
        "Donor-prediction adoption rate",
        patch_series,
        y_domain=(0.0, 1.05),
        width=760,
        height=350,
    )

    layer_maps = read_csv(
        extension / "layerwise_subspace" / "layer_maps" / "layerwise_linear_map_summary.csv"
    )
    answer_rank3_maps = {
        model: [
            row
            for row in layer_maps
            if row["model_label"] == model
            and row["role"] == "answer_query"
            and row["rank"] == "3"
        ]
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    map_error_charts = {
        model: svg_map_error_chart(model, rows)
        for model, rows in answer_rank3_maps.items()
    }
    map_cosine_charts = {
        model: svg_map_cosine_chart(model, rows)
        for model, rows in answer_rank3_maps.items()
    }
    map_stable_counts = {
        model: sum(
            float(row["cv_centroid_r2"]) >= 0.9
            and float(row["bootstrap_map_relative_frobenius_median"]) <= 0.1
            for row in rows
        )
        for model, rows in answer_rank3_maps.items()
    }
    selected_map_rows = {
        "Qwen3-8B": pick(
            answer_rank3_maps["Qwen3-8B"], source_layer=28, target_layer=29
        ),
        "Gemma4-E4B": pick(
            answer_rank3_maps["Gemma4-E4B"], source_layer=36, target_layer=37
        ),
    }

    transport = read_csv(
        extension / "layerwise_subspace" / "transport" / "layerwise_transport_condition_summary.csv"
    )
    transport_statistics = read_csv(
        extension / "layerwise_subspace" / "transport" / "layerwise_transport_statistics.csv"
    )
    transport_contrasts = {
        (model, contrast): pick(
            transport_statistics,
            model_label=model,
            source_layer=source,
            target_layer=target,
            contrast=contrast,
            metric="target_donor_fraction",
        )
        for model, source, target in (
            ("Qwen3-8B", 28, 29),
            ("Gemma4-E4B", 36, 37),
        )
        for contrast in ("aligned_dose_1_minus_orthogonal",)
    }
    transport_rows: list[tuple[str, float, str]] = []
    for model, source, target, color in (
        ("Qwen3-8B", 28, 29, QWEN_COLOR),
        ("Gemma4-E4B", 36, 37, GEMMA_COLOR),
    ):
        short_model = "Qwen" if model == "Qwen3-8B" else "Gemma"
        for condition, label in (
            ("matched_orthogonal", "orthogonal 1×"),
            ("aligned_dose_1", "aligned 1×"),
        ):
            row = pick(
                transport,
                model_label=model,
                source_layer=source,
                target_layer=target,
                condition=condition,
            )
            transport_rows.append((f"{short_model} L{source}→{target} · {label}", float(row["mean_target_donor_fraction"]), color))
    transport_chart = svg_bar_chart(
        "Adjacent-layer aligned transport",
        "Propagated target-chord coefficient F (1 = one R→D chord unit)",
        transport_rows,
        domain=(-0.08, 1.12),
        references=[(1.0, "one target chord")],
        width=860,
    )

    pooled = read_csv(REPORTS / "v4_4_causal_v2" / "correct_patching_pooled.csv")
    answer_patching = {
        model: pick(pooled, model_label=model, family="answer_patching")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    prompt_patching = {
        model: pick(pooled, model_label=model, family="prompt_patching")
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }

    qwen_natural = json.loads(
        (REPORTS / "v4_4_4" / "realistic_niah_v4_4_4_analysis.json").read_text(encoding="utf-8")
    )
    qwen_rw = json.loads(
        (REPORTS / "v4_4_4" / "read_write" / "realistic_niah_v4_4_4_read_write_analysis.json").read_text(encoding="utf-8")
    )
    qwen_upstream = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "upstream_confirmation"
            / "realistic_niah_v4_4_4_upstream_confirmation_analysis.json"
        ).read_text(encoding="utf-8")
    )
    gemma_residual = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "gemma"
            / "residual"
            / "k2"
            / "realistic_niah_v4_4_4_residual_analysis.json"
        ).read_text(encoding="utf-8")
    )
    gemma_l29h4_analysis = json.loads(
        (
            REPORTS
            / "v4_4_4"
            / "gemma"
            / "search"
            / "l29h4"
            / "realistic_niah_v4_4_4_analysis.json"
        ).read_text(encoding="utf-8")
    )

    gemma_candidate = {
        row["endpoint"]: row
        for row in gemma_residual["summary"]
        if row["set_role"] == "candidate_core"
    }
    gemma_l29h4 = {
        endpoint: pick(
            gemma_l29h4_analysis["summary"],
            endpoint=endpoint,
            set_role="candidate_core",
        )
        for endpoint in (
            "natural_carrier_count_slope",
            "injection_dose_slope",
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
            "donor_patch_transport",
        )
    }
    gemma_l29h4_specificity = {
        endpoint: pick(
            gemma_l29h4_analysis["summary"],
            endpoint=endpoint,
            set_role="candidate_specificity",
        )
        for endpoint in (
            "natural_carrier_count_slope__candidate_minus_control_mean",
            "injection_dose_slope__candidate_minus_control_mean",
            "removal_error_axis_minus_control__candidate_minus_control_mean",
            "removal_margin_axis_minus_control__candidate_minus_control_mean",
            "donor_patch_transport__candidate_minus_control_mean",
            "mediation_control_minus_axis_block__candidate_minus_control_mean",
        )
    }

    # These values are frozen primary estimands in the corresponding analysis
    # artifacts.  Pulling them through the JSON schema where possible keeps the
    # prose synchronized with the experiment outputs.
    qwen_natural_summary = qwen_natural["summary"]
    qwen_rw_summary = qwen_rw["summary"]
    qwen_upstream_primary = qwen_upstream["primary_decision"]
    qwen_nat = {
        endpoint: pick(
            qwen_natural_summary,
            endpoint=endpoint,
            set_role="candidate_core",
        )
        for endpoint in (
            "natural_carrier_count_slope",
            "injection_dose_slope",
            "removal_error_axis_minus_control",
            "removal_margin_axis_minus_control",
            "donor_patch_transport",
            "mediation_control_minus_axis_block",
        )
    }
    qwen_read_write = {
        metric: pick(qwen_rw_summary, metric=metric, stratum="all")
        for metric in (
            "read_full_behavior_transport",
            "read_routing_behavior_transport",
            "read_value_behavior_transport",
            "write_behavior_specificity",
        )
    }
    qwen_h19_loo = next(
        row for row in qwen_upstream["leave_one_out"] if row["removed_head"] == "H19"
    )
    qwen_axis_mediated_fraction = (
        float(qwen_nat["mediation_control_minus_axis_block"]["mean"])
        / float(qwen_nat["donor_patch_transport"]["mean"])
    )

    model_order = {"Qwen3-8B": 0, "Gemma4-E4B": 1}
    prompt_geometry_rows = sorted(
        [row for row in geometry_rows if row["role"] == "prompt_running"],
        key=lambda row: model_order[str(row["model"])],
    )
    answer_geometry_rows = sorted(
        [row for row in geometry_rows if row["role"] == "answer_query"],
        key=lambda row: model_order[str(row["model"])],
    )
    expected_models = ["Qwen3-8B", "Gemma4-E4B"]
    if [str(row["model"]) for row in prompt_geometry_rows] != expected_models:
        raise RuntimeError("Prompt geometry table must contain Qwen and Gemma exactly once")
    if [str(row["model"]) for row in answer_geometry_rows] != expected_models:
        raise RuntimeError("Answer geometry table must contain Qwen and Gemma exactly once")

    prompt_geometry_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row['model']))}</td>
          <td>needle-end · L{row['layer']}</td>
          <td>{f(row['r2'])}</td><td>{f(row['reg_mad'])}</td>
          <td>{f(row['rank3_all'])}</td><td>{f(row['rank3_centroid'])}</td>
          <td>{f(row['stable_rank'])}</td><td>{f(row['eta2'])}</td><td>{f(row['silhouette'])}</td>
        </tr>"""
        for row in prompt_geometry_rows
    )
    answer_geometry_table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(row['model']))}</td>
          <td>answer query · L{row['layer']}</td>
          <td>{pct(row['accuracy'])}</td><td>{f(row['class_mad'])}</td>
          <td>{pct(row['centroid_accuracy'])}</td><td>{f(row['centroid_mad'])}</td>
          <td>{f(row['r2'])}</td><td>{f(row['reg_mad'])}</td>
          <td>{f(row['rank3_all'])}</td><td>{f(row['rank3_centroid'])}</td>
          <td>{f(row['stable_rank'])}</td><td>{f(row['eta2'])}</td><td>{f(row['silhouette'])}</td>
        </tr>"""
        for row in answer_geometry_rows
    )

    behavior_rows = "".join(
        f"""<tr><td>{model}</td><td>{int(values['rows'])}</td><td>{pct(values['accuracy'])}</td>
        <td>{f(values['mad'])}</td><td>{f(values['signed'])}</td><td>{f(values['wrong_mad'])}</td></tr>"""
        for model, values in behavior.items()
    )

    best_rows = []
    algorithm_labels = {
        "logistic_l2": "L2 logistic",
        "nearest_centroid": "nearest centroid",
    }
    for model, rows in classifiers.items():
        for algorithm, label in algorithm_labels.items():
            algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
            # Accuracy is the primary metric; use lower count MAD only to make
            # tied maxima deterministic (Qwen logistic ties at L24 and L29).
            best = max(
                algorithm_rows,
                key=lambda row: (float(row["accuracy"]), -float(row["count_mae"])),
            )
            best_rows.append(
                f"{model} {label}: "
                f"L{best['layer']} / {pct(best['accuracy'])} / MAD {f(best['count_mae'])}"
            )

    counter_plot_path = extension / "counter_properties" / "counter_properties_by_layer.png"
    counter_plot_uri = (
        "data:image/png;base64,"
        + base64.b64encode(counter_plot_path.read_bytes()).decode("ascii")
    )
    counter_by_model = {row["model_label"]: row for row in counter_selected}
    counter_table_rows = "".join(
        f"""<tr><td>{model}</td><td>L{row['layer']}</td>
        <td>{f(row['trajectory_line_r2'])}</td>
        <td>{f(row['centroid_distance_vs_count_gap_spearman'])}</td>
        <td>{f(row['adjacent_step_pairwise_cosine_mean'])}</td>
        <td>{f(row['adjacent_step_length_cv'])}</td>
        <td>{f(row['discovery_confirmation_same_step_cosine_mean'])}</td>
        <td>{f(row['frozen_pc3_ridge_r2'])} / {f(row['frozen_pc3_ridge_mad'])}</td>
        <td>{f(row['confirmation_seed_projection_spearman_mean'])} / {pct(row['confirmation_adjacent_increment_positive_fraction'])}</td>
        <td>{f(row['position_count_spearman'])}</td>
        <td>{f(row['position_residual_pc3_grouped_ridge_r2'])} / {f(row['position_residual_pc3_grouped_ridge_mad'])}</td></tr>"""
        for model, row in counter_by_model.items()
    )
    gated_table_rows = []
    for model, layer in (("Qwen3-8B", 8), ("Gemma4-E4B", 9)):
        values = {
            (category, formula): float(
                pick(
                    gated_formula_rows,
                    model_label=model,
                    layer=layer,
                    category=category,
                    model=formula,
                )["incremental_r2_vs_category_baseline"]
            )
            for category, formula in (
                ("needle_endpoint", "endpoint_gated_curve"),
                ("needle_interior", "needle_span_gated_curve"),
                ("ordinary_passage", "ungated_prefix_curve"),
                ("hard_negative", "ungated_prefix_curve"),
            )
        }
        gated_table_rows.append(
            f"<tr><td>{model}</td><td>L{layer}</td>"
            f"<td>{f(values[('needle_endpoint', 'endpoint_gated_curve')])}</td>"
            f"<td>{f(values[('needle_interior', 'needle_span_gated_curve')])}</td>"
            f"<td>{f(values[('ordinary_passage', 'ungated_prefix_curve')])}</td>"
            f"<td>{f(values[('hard_negative', 'ungated_prefix_curve')])}</td></tr>"
        )
    gated_table_rows_html = "".join(gated_table_rows)

    span = followup["dense_span_restoration"]
    span_layer_rows_by_model = {
        model: sorted(
            [
                row
                for row in span_layerwise["layer_rows"]
                if row["model_label"] == model
            ],
            key=lambda row: int(row["layer"]),
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    span_summary_by_model = {
        row["model_label"]: row for row in span_layerwise["model_summaries"]
    }

    def render_layer_segments(segments: list[list[int]]) -> str:
        rendered: list[str] = []
        for segment in segments:
            if len(segment) == 1:
                rendered.append(f"L{segment[0]}")
            else:
                rendered.append(f"L{segment[0]}–L{segment[-1]}")
        return ", ".join(rendered) if rendered else "none"

    def consecutive_layer_segments(layers: list[int]) -> list[list[int]]:
        segments: list[list[int]] = []
        for layer in sorted(set(int(value) for value in layers)):
            if not segments or layer != segments[-1][-1] + 1:
                segments.append([layer])
            else:
                segments[-1].append(layer)
        return segments

    span_layerwise_charts: dict[str, str] = {}
    span_transition_rows: list[str] = []
    for model, color, ticks in (
        ("Qwen3-8B", QWEN_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]),
        ("Gemma4-E4B", GEMMA_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41]),
    ):
        rows = span_layer_rows_by_model[model]
        summary = span_summary_by_model[model]
        drop = summary["largest_adjacent_drop"]
        primary_label = "full span − ordinary"
        span_layerwise_charts[model] = svg_line_chart(
            f"{model} dense span restoration on confirmation seeds",
            "Needle-specific expected-error repair (counts)",
            [
                (
                    primary_label,
                    [(float(row["layer"]), float(row["mean"])) for row in rows],
                    color,
                    "",
                ),
                (
                    "endpoint − ordinary",
                    [
                        (
                            float(row["layer"]),
                            float(row["endpoint_minus_ordinary_mean"]),
                        )
                        for row in rows
                    ],
                    "#d97706",
                    "6 4",
                ),
            ],
            x_label="One-time post-block restoration layer ℓ (zero-based)",
            x_ticks=[float(value) for value in ticks],
            y_domain=(-0.5, 3.4),
            reference=(0.0, "no needle-specific repair"),
            vertical_references=[
                (
                    float(drop["from_layer"]) + 0.5,
                    f"max drop L{drop['from_layer']}→L{drop['to_layer']}: "
                    f"{float(drop['mean']):.3f}",
                )
            ],
            intervals={
                primary_label: [
                    (
                        float(row["layer"]),
                        float(row["ci95_low"]),
                        float(row["ci95_high"]),
                        bool(row["nominal_p_lt_0_05"]),
                    )
                    for row in rows
                ]
            },
            width=860,
            height=350,
        )
        span_transition_rows.append(
            f"<tr><td>{model}</td>"
            f"<td>{render_layer_segments(summary['positive_nominal_segments'])} "
            f"({int(summary['positive_nominal_layer_count'])} layers)</td>"
            f"<td>L{drop['from_layer']}→L{drop['to_layer']}: "
            f"{f(drop['mean'])} [{f(drop['ci95_low'])}, {f(drop['ci95_high'])}], "
            f"p={p_text(drop['exact_signflip_p'])}</td>"
            f"<td>{render_layer_segments(summary['negative_nominal_segments'])}</td>"
            f"<td>{f(summary['endpoint_minus_ordinary_min'])}…"
            f"{f(summary['endpoint_minus_ordinary_max'])}</td></tr>"
        )
    span_transition_rows_html = "".join(span_transition_rows)
    state_effect_rows_by_model = {
        model: sorted(
            [row for row in state_effect_layerwise if row["model_label"] == model],
            key=lambda row: int(row["layer"]),
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    state_effect_raw_charts: dict[str, str] = {}
    state_effect_specificity_charts: dict[str, str] = {}
    state_effect_window_text: dict[str, str] = {}
    state_effect_nominal_text: dict[str, str] = {}
    state_effect_relative_nominal_text: dict[str, str] = {}
    state_effect_cosine_nominal_text: dict[str, str] = {}
    state_effect_window_rows: list[str] = []
    for model, color, ticks, usable_through in (
        ("Qwen3-8B", QWEN_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 35], 20),
        ("Gemma4-E4B", GEMMA_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41], 16),
    ):
        rows = state_effect_rows_by_model[model]
        model_summary = state_effect_summary["models"][model]
        raw_window = model_summary["window_raw_specificity"]
        nominal_layers = [
            int(value) for value in model_summary["pointwise_nominal_positive_layers"]
        ]
        state_effect_window_text[model] = (
            f"{f(raw_window['mean'], 4)} "
            f"[{f(raw_window['ci95_low'], 4)}, {f(raw_window['ci95_high'], 4)}], "
            f"p={p_text(raw_window['exact_sign_flip_p'])}"
        )
        state_effect_nominal_text[model] = render_layer_segments(
            consecutive_layer_segments(nominal_layers)
        )
        relative_nominal_layers = [
            int(row["layer"])
            for row in rows
            if float(row["relative_specificity_mean"]) > 0
            and float(row["relative_specificity_exact_sign_flip_p"]) < 0.05
        ]
        cosine_nominal_layers = [
            int(row["layer"])
            for row in rows
            if float(row["cosine_specificity_mean"]) > 0
            and float(row["cosine_specificity_exact_sign_flip_p"]) < 0.05
        ]
        state_effect_relative_nominal_text[model] = render_layer_segments(
            consecutive_layer_segments(relative_nominal_layers)
        )
        state_effect_cosine_nominal_text[model] = render_layer_segments(
            consecutive_layer_segments(cosine_nominal_layers)
        )
        for metric_label, metric_key in (
            ("Raw RMS specificity · primary", "window_raw_specificity"),
            ("Relative RMS specificity · secondary", "window_relative_specificity"),
            ("Cosine-distance specificity · secondary", "window_cosine_specificity"),
        ):
            stats = model_summary[metric_key]
            state_effect_window_rows.append(
                f"<tr><td>{model}</td><td>{metric_label}</td>"
                f"<td>{f(stats['mean'], 4)} "
                f"[{f(stats['ci95_low'], 4)}, {f(stats['ci95_high'], 4)}]</td>"
                f"<td>{p_text(stats['exact_sign_flip_p'])}</td></tr>"
            )
        state_effect_raw_charts[model] = svg_line_chart(
            f"{model} full-vector span-state deformation",
            "Full-vector RMS change (residual units)",
            [
                (
                    "needle-target deformation",
                    [
                        (
                            float(row["layer"]),
                            float(row["needle_raw_rms_change_mean"]),
                        )
                        for row in rows
                    ],
                    color,
                    "",
                ),
                (
                    "ordinary-target control",
                    [
                        (
                            float(row["layer"]),
                            float(row["ordinary_raw_rms_change_mean"]),
                        )
                        for row in rows
                    ],
                    "#64748b",
                    "6 4",
                ),
            ],
            x_label="Post-block layer ℓ (zero-based)",
            x_ticks=[float(value) for value in ticks],
            width=860,
            height=330,
        )
        state_effect_specificity_charts[model] = svg_line_chart(
            f"{model} needle-specific span-state deformation",
            "Needle − ordinary RMS deformation (residual units)",
            [
                (
                    "needle-specific state effect",
                    [
                        (
                            float(row["layer"]),
                            float(row["raw_specificity_mean"]),
                        )
                        for row in rows
                    ],
                    color,
                    "",
                )
            ],
            x_label="Post-block layer ℓ (zero-based)",
            x_ticks=[float(value) for value in ticks],
            reference=(0.0, "needle deformation = ordinary control"),
            vertical_references=[
                (
                    float(usable_through) + 0.5,
                    f"independently frozen behavior-reuse boundary ≈L{usable_through}",
                )
            ],
            intervals={
                "needle-specific state effect": [
                    (
                        float(row["layer"]),
                        float(row["raw_specificity_ci95_low"]),
                        float(row["raw_specificity_ci95_high"]),
                        float(row["raw_specificity_mean"]) > 0
                        and float(row["raw_specificity_exact_sign_flip_p"]) < 0.05,
                    )
                    for row in rows
                ]
            },
            width=860,
            height=330,
        )
    state_effect_window_rows_html = "".join(state_effect_window_rows)
    state_retention_charts: dict[str, str] = {}
    state_retention_level_lookup = {
        (
            row["model_label"],
            row["pooling"],
            row["measure"],
            row["condition"],
        ): row
        for row in state_retention_levels
    }
    state_retention_window_lookup = {
        (row["model_label"], row["candidate"]): row
        for row in state_retention_window
    }
    for model, color, ticks, usable_through in (
        ("Qwen3-8B", QWEN_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 35], 20),
        ("Gemma4-E4B", GEMMA_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41], 16),
    ):
        endpoint_rows = sorted(
            [
                row
                for row in state_retention_layerwise
                if row["model_label"] == model
                and row["pooling"] == "span_end"
                and row["metric"] == "correct_distance_specificity"
            ],
            key=lambda row: int(row["layer"]),
        )
        mean_rows = sorted(
            [
                row
                for row in state_retention_layerwise
                if row["model_label"] == model
                and row["pooling"] == "span_mean"
                and row["metric"] == "correct_distance_specificity"
            ],
            key=lambda row: int(row["layer"]),
        )
        state_retention_charts[model] = svg_line_chart(
            f"{model} running-index state retention after matched corruption",
            "Needle − ordinary correct-centroid distance (class radii)",
            [
                (
                    "span endpoint",
                    [
                        (float(row["layer"]), float(row["mean"]))
                        for row in endpoint_rows
                    ],
                    color,
                    "",
                ),
                (
                    "span-token mean",
                    [
                        (float(row["layer"]), float(row["mean"]))
                        for row in mean_rows
                    ],
                    "#2563eb",
                    "6 4",
                ),
            ],
            x_label="Post-block layer ℓ (zero-based)",
            x_ticks=[float(value) for value in ticks],
            reference=(0.0, "needle corruption = ordinary corruption"),
            vertical_references=[
                (
                    float(usable_through) + 0.5,
                    f"independently frozen reusable window through L{usable_through}",
                )
            ],
            intervals={
                "span endpoint": [
                    (
                        float(row["layer"]),
                        float(row["ci95_low"]),
                        float(row["ci95_high"]),
                        float(row["exact_sign_flip_p"]) < 0.05,
                    )
                    for row in endpoint_rows
                ],
                "span-token mean": [
                    (
                        float(row["layer"]),
                        float(row["ci95_low"]),
                        float(row["ci95_high"]),
                        float(row["exact_sign_flip_p"]) < 0.05,
                    )
                    for row in mean_rows
                ],
            },
            width=860,
            height=350,
        )

    state_retention_level_bars = svg_bar_chart(
        "Frozen-window correct-centroid distance under clean and matched corruptions",
        "Distance to the correct running-index centroid (discovery-class radii)",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {condition_label}",
                float(
                    state_retention_level_lookup[
                        (model, "span_end", "correct_distance", condition)
                    ]["mean"]
                ),
                color,
            )
            for model, needle_color in (
                ("Qwen3-8B", QWEN_COLOR),
                ("Gemma4-E4B", GEMMA_COLOR),
            )
            for condition, condition_label, color in (
                ("clean", "clean", "#cbd5e1"),
                ("ordinary_corrupt", "ordinary-corrupt", "#64748b"),
                ("needle_corrupt", "needle-corrupt", needle_color),
            )
        ],
        domain=(0.0, 9.0),
        references=[(1.0, "typical clean held-out radius")],
        width=860,
        row_height=38,
    )

    retention_metric_labels = {
        "span_end_correct_distance_specificity": "Endpoint · correct-centroid distance",
        "span_mean_correct_distance_specificity": "Span mean · correct-centroid distance",
        "span_end_margin_damage_specificity": "Endpoint · nearest-centroid margin damage",
        "span_mean_margin_damage_specificity": "Span mean · nearest-centroid margin damage",
        "span_end_accuracy_damage_specificity": "Endpoint · nearest-centroid accuracy damage",
        "span_mean_accuracy_damage_specificity": "Span mean · nearest-centroid accuracy damage",
        "span_end_relative_rms_specificity": "Endpoint · relative RMS change",
        "span_mean_relative_rms_specificity": "Span mean · relative RMS change",
        "span_end_cosine_specificity": "Endpoint · cosine-distance change",
        "span_mean_cosine_specificity": "Span mean · cosine-distance change",
    }
    retention_candidate_order = list(state_retention_selection["priority"]) + [
        "span_end_relative_rms_specificity",
        "span_mean_relative_rms_specificity",
        "span_end_cosine_specificity",
        "span_mean_cosine_specificity",
    ]
    state_retention_all_metric_rows: list[str] = []
    for candidate in retention_candidate_order:
        cells: list[str] = []
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            row = state_retention_window_lookup[(model, candidate)]
            cells.append(
                f"{f(row['mean'], 3)} [{f(row['ci95_low'], 3)}, "
                f"{f(row['ci95_high'], 3)}]; Holm p="
                f"{p_text(float(row['holm_p_within_model_family']))}"
            )
        state_retention_all_metric_rows.append(
            f"<tr><td>{retention_metric_labels[candidate]}</td>"
            f"<td>{cells[0]}</td><td>{cells[1]}</td>"
            f"<td>{'Retention (6 tests/model)' if 'correct_distance' in candidate or 'margin' in candidate or 'accuracy' in candidate else 'Deformation (4 tests/model)'}</td></tr>"
        )
    state_retention_all_metric_rows_html = "".join(state_retention_all_metric_rows)

    state_retention_condition_rows: list[str] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for pooling, pooling_label in (
            ("span_end", "Endpoint"),
            ("span_mean", "Span-token mean"),
        ):
            clean = state_retention_level_lookup[
                (model, pooling, "correct_distance", "clean")
            ]
            ordinary = state_retention_level_lookup[
                (model, pooling, "correct_distance", "ordinary_corrupt")
            ]
            needle = state_retention_level_lookup[
                (model, pooling, "correct_distance", "needle_corrupt")
            ]
            distance_row = state_retention_window_lookup[
                (model, f"{pooling}_correct_distance_specificity")
            ]
            margin_row = state_retention_window_lookup[
                (model, f"{pooling}_margin_damage_specificity")
            ]
            accuracy_row = state_retention_window_lookup[
                (model, f"{pooling}_accuracy_damage_specificity")
            ]
            state_retention_condition_rows.append(
                f"<tr><td>{model}</td><td>{pooling_label}</td>"
                f"<td>{f(clean['mean'], 3)}</td><td>{f(ordinary['mean'], 3)}</td>"
                f"<td>{f(needle['mean'], 3)}</td>"
                f"<td>+{f(distance_row['mean'], 3)}</td>"
                f"<td>+{f(margin_row['mean'], 3)}</td>"
                f"<td>+{f(accuracy_row['mean'], 3)}</td></tr>"
            )
    state_retention_condition_rows_html = "".join(state_retention_condition_rows)
    state_retention_primary = {
        model: state_retention_window_lookup[
            (model, "span_end_correct_distance_specificity")
        ]
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    state_retention_endpoint_levels = {
        model: {
            condition: state_retention_level_lookup[
                (model, "span_end", "correct_distance", condition)
            ]
            for condition in ("clean", "ordinary_corrupt", "needle_corrupt")
        }
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    span_attention_charts: dict[str, str] = {}
    for model, color, ticks, usable_through in (
        ("Qwen3-8B", QWEN_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 35], 20),
        ("Gemma4-E4B", GEMMA_COLOR, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41], 16),
    ):
        rows = sorted(
            [row for row in span_attention_response if row["model_label"] == model],
            key=lambda row: int(row["patch_layer"]),
        )
        span_attention_charts[model] = svg_line_chart(
            f"{model} answer-query attention response to full-span restoration",
            "Needle restoration response minus ordinary restoration response",
            [
                (
                    "needle-mass specificity",
                    [
                        (
                            float(row["patch_layer"]),
                            float(row["mass_specificity"]),
                        )
                        for row in rows
                    ],
                    "#64748b",
                    "6 4",
                ),
                (
                    "broad-score specificity",
                    [
                        (
                            float(row["patch_layer"]),
                            float(row["broad_specificity"]),
                        )
                        for row in rows
                    ],
                    color,
                    "",
                ),
            ],
            x_label="One-time post-block full-span restoration layer ℓ",
            x_ticks=[float(value) for value in ticks],
            y_domain=(-0.02, 0.44),
            reference=(0.0, "needle response = ordinary response"),
            vertical_references=[
                (float(usable_through) + 0.5, f"canonical reuse through ≈L{usable_through}")
            ],
            width=860,
            height=350,
        )
    span_layer_lookup = {
        (row["model_label"], int(row["layer"])): row
        for row in span_layerwise["layer_rows"]
    }
    qwen_l20 = span_layer_lookup[("Qwen3-8B", 20)]
    qwen_l21 = span_layer_lookup[("Qwen3-8B", 21)]
    gemma_l16 = span_layer_lookup[("Gemma4-E4B", 16)]
    gemma_l17 = span_layer_lookup[("Gemma4-E4B", 17)]
    qwen_span_drop = span_summary_by_model["Qwen3-8B"]["largest_adjacent_drop"]
    gemma_span_drop = span_summary_by_model["Gemma4-E4B"]["largest_adjacent_drop"]
    span_landmarks = {row["model_label"]: row for row in span["formal_landmarks"]}
    span_landmark_chart = svg_bar_chart(
        "Dense full-span restoration landmarks",
        "full-needle minus ordinary expected-error repair (counts)",
        [
            ("Qwen discovery early plateau", span_landmarks["Qwen3-8B"]["discovery_early_plateau"], QWEN_COLOR),
            ("Qwen confirmation L19", span_landmarks["Qwen3-8B"]["half_boundary_confirmation_specificity"], QWEN_COLOR),
            ("Qwen confirmation L23", span_landmarks["Qwen3-8B"]["near_zero_confirmation_specificity"], QWEN_COLOR),
            ("Gemma discovery early plateau", span_landmarks["Gemma4-E4B"]["discovery_early_plateau"], GEMMA_COLOR),
            ("Gemma confirmation L17", span_landmarks["Gemma4-E4B"]["half_boundary_confirmation_specificity"], GEMMA_COLOR),
            ("Gemma confirmation L18", span_landmarks["Gemma4-E4B"]["near_zero_confirmation_specificity"], GEMMA_COLOR),
        ],
        domain=(-0.3, 3.2),
        references=[(0.0, "no direction-specific repair")],
        width=860,
    )
    span_landmark_rows = "".join(
        f"""<tr><td>{model}</td><td>{f(row['discovery_early_plateau'])}</td>
        <td>L{row['half_boundary_layer']}: {f(row['half_boundary_confirmation_specificity'])}</td>
        <td>L{row['near_zero_boundary_layer']}: {f(row['near_zero_confirmation_specificity'])}</td></tr>"""
        for model, row in span_landmarks.items()
    )

    answer_reuse_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{pct(row['nearest_centroid_accuracy'])}</td><td>{f(row['integer_mad'])}</td>
        <td>{f(row['discovery_rank3_all_state_capture'])}</td></tr>"""
        for row in followup["answer_geometry_reuse"]["rows"]
    )
    retrieval_geometry_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{pct(row['exact_classifier_accuracy'])}</td>
        <td>{pct(row['nearest_centroid_accuracy'])}</td>
        <td>{f(row['classifier_mad'])}</td></tr>"""
        for row in followup["retrieval_geometry"]["rows"]
    )
    retrieval_geometry_charts: list[str] = []
    for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR)):
        rows = [
            row
            for row in followup["retrieval_geometry"]["rows"]
            if row["model_label"] == model
        ]
        layers = [float(row["layer"]) for row in rows]
        retrieval_geometry_charts.append(
            svg_line_chart(
                f"{model}: count readout from the summed broad-bank write",
                "held-out accuracy (fraction)",
                [
                    (
                        "exact classifier",
                        [
                            (float(row["layer"]), float(row["exact_classifier_accuracy"]))
                            for row in rows
                        ],
                        color,
                        "",
                    ),
                    (
                        "nearest centroid",
                        [
                            (float(row["layer"]), float(row["nearest_centroid_accuracy"]))
                            for row in rows
                        ],
                        "#64748b",
                        "6 4",
                    ),
                ],
                x_label="Frozen broad-bank layer (zero-based)",
                x_ticks=layers,
                y_domain=(0.0, 0.60),
                reference=(0.10, "10-class chance = 0.10"),
                width=720,
                height=340,
            )
        )

    retrieval_subspace = followup["retrieval_subspace"]
    retrieval_subspace_rows = "".join(
        f"""<tr><td>{row['model_label']}</td><td>L{row['layer']}</td>
        <td>{f(row['natural_specificity_mean'])}</td>
        <td>{f(row['restoration_mediation_mean'])}</td>
        <td>{f(row['mediated_fraction_mean'])}</td></tr>"""
        for row in retrieval_subspace["rows"]
    )
    retrieval_subspace_charts: list[str] = []
    for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR)):
        rows = [row for row in retrieval_subspace["rows"] if row["model_label"] == model]
        retrieval_subspace_charts.append(
            svg_line_chart(
                f"{model}: frozen retrieval-subspace removal",
                "direction-specific damage (counts)",
                [
                    (
                        "natural specificity",
                        [(float(row["layer"]), float(row["natural_specificity_mean"])) for row in rows],
                        color,
                        "",
                    ),
                    (
                        "restoration mediation",
                        [(float(row["layer"]), float(row["restoration_mediation_mean"])) for row in rows],
                        "#8B5CF6",
                        "6 4",
                    ),
                ],
                x_label="Frozen intervention layer",
                y_domain=(-0.1, 0.6),
                reference=(0.0, "aligned = orthogonal"),
                x_ticks=[float(row["layer"]) for row in rows],
            )
        )

    exp19_chart = svg_bar_chart(
        "Same-forward ordered partial serial mediation",
        "direction-specific expected-count effect (counts)",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {label}",
                float(exp19[model][metric]["mean"]),
                color,
            )
            for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR))
            for label, metric in (
                ("source repair", "source_repair"),
                ("retrieval mediation", "retrieval_mediation"),
                ("late mediation", "late_mediation"),
            )
        ],
        domain=(0.0, 3.2),
        references=[(0.0, "matched-control difference = 0")],
        width=840,
    )
    def exp19_metric_cell(values: dict[str, Any], metric: str) -> str:
        result = values[metric]
        return (
            f"{f(result['mean'])} "
            f"[{f(result['ci95_low'])}, {f(result['ci95_high'])}]; "
            f"exact p={p_text(result['exact_sign_flip_p_two_sided'])}"
        )

    exp19_rows = "".join(
        f"""<tr><td>{model}</td>
        <td>{exp19_metric_cell(values, 'source_repair')}</td>
        <td>{exp19_metric_cell(values, 'retrieval_mediation')}</td>
        <td>{exp19_metric_cell(values, 'late_mediation')}</td>
        <td>{exp19_metric_cell(values, 'joint_interaction')}</td>
        <td>{exp19_metric_cell(values, 'remaining_repair')}</td></tr>"""
        for model, values in exp19.items()
    )

    exp22_selected: dict[str, dict[str, Any]] = {}
    for model in model_order:
        registration = exp22_registration[model]
        exp22_selected[model] = next(
            row
            for row in exp22_synthetic[model]["head_summaries"]
            if int(row["layer"]) == int(registration["source_layer"])
            and int(row["head"]) == int(registration["source_head"])
        )
    exp22_chart = svg_interval_chart(
        "Canonical test of the frozen induction-like edge registry",
        "candidate minus attention/distance-matched control: expected absolute error (counts)",
        [
            (
                model,
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["mean"]),
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["ci95_low"]),
                float(exp22[model]["metrics"]["expected_absolute_error_candidate_minus_control"]["ci95_high"]),
                color,
            )
            for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR))
        ],
        domain=(-0.04, 0.012),
        reference=(0.0, "registered specificity requires > 0"),
        width=840,
    )
    exp22_rows = "".join(
        f"""<tr><td>{model}</td><td>L{int(exp22_registration[model]['source_layer'])}H{int(exp22_registration[model]['source_head'])}</td>
        <td>{f(exp22_selected[model]['repeated_relation_mean'], 5)}</td>
        <td>{f(exp22_selected[model]['reassignment_follow_mean'], 5)}</td>
        <td>{f(exp22_selected[model]['unique_anchor_abs_mean'], 5)} / {f(exp22_selected[model]['ordinary_repeat_abs_mean'], 5)}</td>
        <td>{f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['mean'], 5)} [{f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['ci95_low'], 5)}, {f(exp22[model]['metrics']['expected_absolute_error_candidate_minus_control']['ci95_high'], 5)}]</td>
        <td>not supported</td></tr>"""
        for model in model_order
    )

    exp23_factor_chart = svg_bar_chart(
        "Controlled identity/context/position deformation in the frozen rank-3 basis",
        "confirmation held-out incremental ΔR²",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {factor}",
                float(exp23[model]["factorial"]["confirmation_incremental_delta_r2"][factor]),
                color,
            )
            for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR))
            for factor in ("identity", "context", "position")
        ],
        domain=(-0.005, 0.020),
        references=[(0.0, "no incremental held-out prediction")],
        width=840,
    )
    exp23_specificity_chart = svg_interval_chart(
        "Outside-context halo-edge specificity",
        "candidate removal minus matched control: expected absolute error (counts)",
        [
            (
                f"{model.replace('3-8B', '').replace('4-E4B', '')} · {label}",
                float(exp23[model]["outside_context"]["metrics"][metric]["mean"]),
                float(exp23[model]["outside_context"]["metrics"][metric]["ci95_low"]),
                float(exp23[model]["outside_context"]["metrics"][metric]["ci95_high"]),
                color,
            )
            for model, color in (("Qwen3-8B", QWEN_COLOR), ("Gemma4-E4B", GEMMA_COLOR))
            for label, metric in (
                ("vs distance-random", "expected_error_candidate_minus_distance_random"),
                ("vs attention-mass", "expected_error_candidate_minus_attention_mass"),
            )
        ],
        domain=(-0.014, 0.030),
        reference=(0.0, "registered specificity requires CI > 0"),
        width=840,
    )
    exp23_rows = "".join(
        f"""<tr><td>{model}</td>
        <td>{f(exp23[model]['factorial']['confirmation_full_model_r2'], 4)}</td>
        <td>{f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['identity'], 4)} / {f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['context'], 4)} / {f(exp23[model]['factorial']['confirmation_incremental_delta_r2']['position'], 4)}</td>
        <td>L{int(exp23_registration[model]['source_layer'])}H{int(exp23_registration[model]['source_head'])}</td>
        <td>{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['mean'], 4)} [{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['ci95_low'], 4)}, {f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_distance_random']['ci95_high'], 4)}]</td>
        <td>{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['mean'], 4)} [{f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['ci95_low'], 4)}, {f(exp23[model]['outside_context']['metrics']['expected_error_candidate_minus_attention_mass']['ci95_high'], 4)}]</td>
        <td>not supported</td></tr>"""
        for model in model_order
    )

    retrieval_geometry_by_key = {
        (str(row["model_label"]), int(row["layer"])): row
        for row in followup["retrieval_geometry"]["rows"]
    }
    retrieval_subspace_by_key = {
        (str(row["model_label"]), int(row["layer"])): row
        for row in retrieval_subspace["rows"]
    }
    retrieval_clean_by_key = {
        (str(row["model_label"]), int(row["layer"])): row
        for row in retrieval_subspace["clean_correct_robustness"]
    }
    qwen_retrieval_l23 = retrieval_subspace_by_key[("Qwen3-8B", 23)]
    gemma_retrieval_l29 = retrieval_subspace_by_key[("Gemma4-E4B", 29)]
    gemma_retrieval_l35 = retrieval_subspace_by_key[("Gemma4-E4B", 35)]
    qwen_retrieval_l23_clean = retrieval_clean_by_key[("Qwen3-8B", 23)]
    gemma_retrieval_l29_clean = retrieval_clean_by_key[("Gemma4-E4B", 29)]
    qwen_serial = exp19["Qwen3-8B"]
    gemma_serial = exp19["Gemma4-E4B"]
    retrieval_exact_values = [
        float(row["exact_classifier_accuracy"])
        for row in followup["retrieval_geometry"]["rows"]
    ]
    retrieval_nearest_values = [
        float(row["nearest_centroid_accuracy"])
        for row in followup["retrieval_geometry"]["rows"]
    ]
    retrieval_mad_values = [
        float(row["classifier_mad"])
        for row in followup["retrieval_geometry"]["rows"]
    ]

    # Claim-level Top-6/Top-8 consistency audit.  This is deliberately not
    # presented as a formal equivalence test: it asks whether each conclusion
    # that depended on the historical Gemma Top-8 bank is retained when the
    # exact discovery-ranked Top-6 prefix is used instead.
    comparison_root = top6_followup_root / "comparison_sources"
    top8_geometry_rows = read_csv(
        comparison_root
        / "top8_retrieval_geometry"
        / "retrieval_geometry"
        / "retrieval_geometry_metrics.csv"
    )
    top8_geometry_by_layer = {
        int(row["layer"]): row
        for row in top8_geometry_rows
        if row["model_label"] == "Gemma4-E4B"
    }
    top6_geometry_by_layer = {
        int(row["layer"]): row for row in top6_geometry_rows
    }
    if set(top8_geometry_by_layer) != {29, 35} or set(top6_geometry_by_layer) != {
        29,
        35,
    }:
        raise RuntimeError("Top-6/Top-8 geometry comparison is incomplete")

    top8_subspace_by_layer = {
        layer: pick(
            read_csv(
                comparison_root
                / "top8_retrieval_subspace"
                / f"Gemma_L{layer}"
                / "effect_summary.csv"
            ),
            model_label="Gemma4-E4B",
            population="all",
        )
        for layer in (29, 35)
    }
    top6_subspace_by_layer = {
        29: gemma_retrieval_l29,
        35: gemma_retrieval_l35,
    }
    top8_subspace_paired_by_layer = {
        layer: read_csv(
            comparison_root
            / "top8_retrieval_subspace"
            / f"Gemma_L{layer}"
            / "paired_effects.csv"
        )
        for layer in (29, 35)
    }
    top6_subspace_paired_by_layer = {
        layer: read_csv(
            top6_followup_root
            / "retrieval_subspace"
            / f"Gemma_L{layer}"
            / "paired_effects.csv"
        )
        for layer in (29, 35)
    }
    expected_subspace_keys = {
        (seed, count) for seed in range(1254, 1264) for count in range(1, 11)
    }
    for label, layer_rows in (
        ("Top-8", top8_subspace_paired_by_layer),
        ("Top-6", top6_subspace_paired_by_layer),
    ):
        for layer, rows in layer_rows.items():
            observed_keys = {(int(row["seed"]), int(row["gold_count"])) for row in rows}
            if len(rows) != 100 or observed_keys != expected_subspace_keys:
                raise RuntimeError(
                    f"{label} L{layer} retrieval comparison panel is incomplete"
                )
    top8_l29_natural_interval = seed_cluster_bootstrap_interval(
        top8_subspace_paired_by_layer[29],
        "natural_expected_error_specificity",
        random_seed=4_456,
    )
    top8_l29_mediation_interval = seed_cluster_bootstrap_interval(
        top8_subspace_paired_by_layer[29],
        "restoration_mediation_expected_error",
        random_seed=4_457,
    )
    top6_l29_natural_interval = seed_cluster_bootstrap_interval(
        top6_subspace_paired_by_layer[29],
        "natural_expected_error_specificity",
        random_seed=4_456,
    )
    top6_l29_mediation_interval = seed_cluster_bootstrap_interval(
        top6_subspace_paired_by_layer[29],
        "restoration_mediation_expected_error",
        random_seed=4_457,
    )
    top8_serial = json.loads(
        (
            followup_root
            / "exp19"
            / "Gemma4-E4B"
            / "serial_summary.json"
        ).read_text(encoding="utf-8")
    )["models"]["Gemma4-E4B"]

    top8_attention_by_layer = {
        int(row["patch_layer"]): row
        for row in original_span_attention_response
        if row["model_label"] == "Gemma4-E4B"
    }
    top6_attention_by_layer = {
        int(row["patch_layer"]): row for row in top6_span_attention_response
    }
    if set(top8_attention_by_layer) != set(range(42)) or set(
        top6_attention_by_layer
    ) != set(range(42)):
        raise RuntimeError("Top-6/Top-8 attention comparison is incomplete")
    top8_broad_curve = [
        float(top8_attention_by_layer[layer]["broad_specificity"])
        for layer in range(42)
    ]
    top6_broad_curve = [
        float(top6_attention_by_layer[layer]["broad_specificity"])
        for layer in range(42)
    ]
    top8_broad_mean = sum(top8_broad_curve) / len(top8_broad_curve)
    top6_broad_mean = sum(top6_broad_curve) / len(top6_broad_curve)
    top6_top8_attention_corr = sum(
        (old - top8_broad_mean) * (new - top6_broad_mean)
        for old, new in zip(top8_broad_curve, top6_broad_curve)
    ) / math.sqrt(
        sum((value - top8_broad_mean) ** 2 for value in top8_broad_curve)
        * sum((value - top6_broad_mean) ** 2 for value in top6_broad_curve)
    )
    top6_top8_attention_max_delta = max(
        abs(old - new)
        for old, new in zip(top8_broad_curve, top6_broad_curve)
    )

    gemma_k8_shift = pick(
        topk,
        model_label="Gemma4-E4B",
        top_n="8",
        analysis_population="all_examples_signed",
    )
    gemma_k6_shift = pick(
        topk,
        model_label="Gemma4-E4B",
        top_n="6",
        analysis_population="all_examples_signed",
    )
    gemma_k8_damage = pick(
        topk,
        model_label="Gemma4-E4B",
        top_n="8",
        analysis_population="clean_correct_only",
    )
    gemma_k6_damage = pick(
        topk,
        model_label="Gemma4-E4B",
        top_n="6",
        analysis_population="clean_correct_only",
    )
    if not (
        float(gemma_k8_shift["ci95_low"]) > 0
        and float(gemma_k6_shift["ci95_low"]) > 0
        and float(gemma_k8_damage["ci95_low"]) > 0
        and float(gemma_k6_damage["ci95_low"]) > 0
        and top6_top8_attention_corr > 0.99
        and all(
            float(rows[layer]["exact_classifier_accuracy"]) > 0.10
            and float(rows[layer]["rank3_centroids"]) > 0.95
            and abs(float(rows[layer]["cosine_silhouette"])) < 0.10
            for rows in (top8_geometry_by_layer, top6_geometry_by_layer)
            for layer in (29, 35)
        )
        and float(top8_subspace_by_layer[29]["mean_natural_expected_error_specificity"])
        > 0
        and float(top6_subspace_by_layer[29]["natural_specificity_mean"]) > 0
        and top8_l29_natural_interval["ci95_low"] > 0
        and top6_l29_natural_interval["ci95_low"] > 0
        and float(top8_subspace_by_layer[29]["mean_restoration_mediation_expected_error"])
        > 0
        and float(top6_subspace_by_layer[29]["restoration_mediation_mean"]) > 0
        and top8_l29_mediation_interval["ci95_low"] > 0
        and top6_l29_mediation_interval["ci95_low"] > 0
        and abs(
            float(top8_subspace_by_layer[35]["mean_natural_expected_error_specificity"])
        )
        < 0.10
        and abs(
            float(top8_subspace_by_layer[35]["mean_restoration_mediation_expected_error"])
        )
        < 0.10
        and abs(float(top6_subspace_by_layer[35]["natural_specificity_mean"]))
        < 0.10
        and abs(float(top6_subspace_by_layer[35]["restoration_mediation_mean"]))
        < 0.10
        and top8_serial["ordered_criterion_diagnostics"][
            "i_source_changes_retrieval"
        ]["source_repair_positive"]
        and gemma_serial["ordered_criterion_diagnostics"][
            "i_source_changes_retrieval"
        ]["source_repair_positive"]
        and top8_serial["ordered_criterion_diagnostics"][
            "ii_retrieval_precedes_late"
        ]["retrieval_mediation_positive"]
        and gemma_serial["ordered_criterion_diagnostics"][
            "ii_retrieval_precedes_late"
        ]["retrieval_mediation_positive"]
        and top8_serial["ordered_criterion_diagnostics"][
            "iii_late_affects_output_not_earlier_readout"
        ]["late_mediation_positive"]
        and gemma_serial["ordered_criterion_diagnostics"][
            "iii_late_affects_output_not_earlier_readout"
        ]["late_mediation_positive"]
    ):
        raise RuntimeError("Top-6 does not preserve a reported Top-8 claim")

    top6_vs_top8_rows = "".join(
        f"<tr><td>{analysis}</td><td>{top8_value}</td><td>{top6_value}</td>"
        f"<td>{interpretation}</td></tr>"
        for analysis, top8_value, top6_value, interpretation in (
            (
                "Matched-control ablation",
                f"K8 shift +{f(gemma_k8_shift['primary_effect'])} "
                f"[{ci(gemma_k8_shift, 'ci95_low', 'ci95_high')}]; "
                f"correct→wrong +{f(gemma_k8_damage['primary_effect'])} "
                f"[{ci(gemma_k8_damage, 'ci95_low', 'ci95_high')}]",
                f"K6 shift +{f(gemma_k6_shift['primary_effect'])} "
                f"[{ci(gemma_k6_shift, 'ci95_low', 'ci95_high')}]; "
                f"correct→wrong +{f(gemma_k6_damage['primary_effect'])} "
                f"[{ci(gemma_k6_damage, 'ci95_low', 'ci95_high')}]",
                "两项方向与区间判定一致，K6 效应不小于 K8；正文选 K6。",
            ),
            (
                "Restoration→broad attention",
                "42-layer Top-8 response curve",
                f"42-layer Top-6 curve；Pearson r={f(top6_top8_attention_corr, 5)}，"
                f"max |ΔB|={f(top6_top8_attention_max_delta, 4)}",
                "曲线形状与 reusable-source boundary 不变；正文选 K6。",
            ),
            (
                "Broad-bank geometry",
                "exact acc L29/L35 "
                f"{pct(top8_geometry_by_layer[29]['exact_classifier_accuracy'])}/"
                f"{pct(top8_geometry_by_layer[35]['exact_classifier_accuracy'])}; "
                "rank-3 "
                f"{f(top8_geometry_by_layer[29]['rank3_centroids'])}/"
                f"{f(top8_geometry_by_layer[35]['rank3_centroids'])}",
                "exact acc L29/L35 "
                f"{pct(top6_geometry_by_layer[29]['exact_classifier_accuracy'])}/"
                f"{pct(top6_geometry_by_layer[35]['exact_classifier_accuracy'])}; "
                "rank-3 "
                f"{f(top6_geometry_by_layer[29]['rank3_centroids'])}/"
                f"{f(top6_geometry_by_layer[35]['rank3_centroids'])}",
                "低秩、可读但单样本 noisy 的结论不变；正文选 K6。",
            ),
            (
                "Retrieval-subspace removal",
                "L29 natural "
                f"{f(top8_l29_natural_interval['mean'])} "
                f"[{ci(top8_l29_natural_interval, 'ci95_low', 'ci95_high')}]; "
                "mediation "
                f"{f(top8_l29_mediation_interval['mean'])} "
                f"[{ci(top8_l29_mediation_interval, 'ci95_low', 'ci95_high')}]; "
                "L35 "
                f"{f(top8_subspace_by_layer[35]['mean_natural_expected_error_specificity'])}/"
                f"{f(top8_subspace_by_layer[35]['mean_restoration_mediation_expected_error'])}",
                "L29 natural "
                f"{f(top6_l29_natural_interval['mean'])} "
                f"[{ci(top6_l29_natural_interval, 'ci95_low', 'ci95_high')}]; "
                "mediation "
                f"{f(top6_l29_mediation_interval['mean'])} "
                f"[{ci(top6_l29_mediation_interval, 'ci95_low', 'ci95_high')}]; "
                "L35 "
                f"{f(top6_subspace_by_layer[35]['natural_specificity_mean'])}/"
                f"{f(top6_subspace_by_layer[35]['restoration_mediation_mean'])}",
                "两者都定位 L29 正效应、L35 约 0；正文选 K6。",
            ),
            (
                "Same-forward serial mediation",
                "retrieval "
                f"{f(top8_serial['retrieval_mediation']['mean'])} "
                f"[{ci(top8_serial['retrieval_mediation'], 'ci95_low', 'ci95_high')}]; "
                "late "
                f"{f(top8_serial['late_mediation']['mean'])} "
                f"[{ci(top8_serial['late_mediation'], 'ci95_low', 'ci95_high')}]",
                "retrieval "
                f"{f(gemma_serial['retrieval_mediation']['mean'])} "
                f"[{ci(gemma_serial['retrieval_mediation'], 'ci95_low', 'ci95_high')}]; "
                "late "
                f"{f(gemma_serial['late_mediation']['mean'])} "
                f"[{ci(gemma_serial['late_mediation'], 'ci95_low', 'ci95_high')}]",
                "两列均按 seed-cluster inference 重算，三项 ordered criteria 均保留；"
                "正文/附录统一采用该口径。",
            ),
            (
                "Induction/noise auxiliary readouts",
                "Top-8-derived broad readout；两项 scientific verdict 均 not_supported",
                "Top-6 broad readout；4 个 row panels 的全部非-bank 字段 exact equal",
                "行为干预与否定性判定完全不变；当前 report 选 K6。",
            ),
        )
    )

    status_labels = {
        "verified": "已验证",
        "falsified": "已证伪",
        "partial": "部分回答",
        "open": "未完成",
        "closed": "已关闭",
    }
    extension_claims = [
        (1, "Prompt centroid trajectory 是否低维但 noisy", "verified", "30 seeds×10 running indices；discovery-fitted PCA/rank metrics", "centroid rank-3 94.7%–97.9%，all-state 59.8%–72.9%，silhouette 接近 0", "平均 count trajectory 低维；单样本不是十个紧密、彼此分离的 clusters。"),
        (2, "Needle-end running index 是否可预测", "verified", "seed-held-out PCA-32 ridge；另审计 frozen PC1–PC3 ordinal geometry", "Qwen L8 full probe R²=0.945/MAD=0.561；Gemma L9 0.719/1.249；3-PC held-out R²={:.3f}/{:.3f}".format(float(counter_by_model['Qwen3-8B']['frozen_pc3_ridge_r2']), float(counter_by_model['Gemma4-E4B']['frozen_pc3_ridge_r2'])), "running index 可线性读取；这里不需要、也不汇报 prompt exact-count classifier。"),
        (3, "Active needle evidence 是否因果必要", "verified", "全部 active spans 替换；同长度、同 token-budget ordinary replacement control", "absolute-error specificity Qwen +8.930、Gemma +8.780 counts", "模型确实依赖 active needle evidence；这不是把 candidate 与无干预 baseline 混比。"),
        (4, "Prompt 的可复用信息是 endpoint 还是 whole span", "verified", "canonical seeds 1234–1263×counts 1–10；endpoint/full-span/ordinary restoration 在 Qwen L0–35、Gemma L0–41 逐层扫描", "endpoint−ordinary 全层接近 0；whole-span 正向 nominal window 为 Qwen L0–20、Gemma L0–16", "可复用 source 分散在完整 needle span，不是 endpoint 单点寄存器；Qwen 后段渐降，Gemma L16→17 cliff。"),
        (5, "Broad score、attention map 与 frozen head sets 是否定义清楚", "verified", "discovery 上按 B=M×C 排名；Qwen top-32、Gemma top-6 冻结后进入干预", "报告给出 layer×head map、完整 membership、mass M、coverage C 与 score B", "broad 同时要求较大 needle mass 与多-span coverage；attention map 只描述 routing。"),
        (6, "Broad-ranked heads 是否有 matched-control 因果效应", "verified", "top-K answer-query ablation vs layer-distribution-matched random heads", "Qwen K32 +1.623 counts；Gemma K6 +0.825 [0.690,0.975]（post-hoc dose）", "这些 head banks 被自然行为使用；Gemma K6 复用冻结 ranking/stimuli/controls，但不被追溯性写成原注册网格。"),
        (7, "Broad-bank output 是否含低秩 count geometry", "verified", "3,000 broad-bank answer-query states；discovery fit、confirmation readout；Gemma 两层按 exact Top-6 post-O write 重构", f"rank-3 centroid capture 0.968–0.995；exact accuracy {pct(min(retrieval_exact_values))}–{pct(max(retrieval_exact_values))}；silhouette −0.098–0.014", "存在低秩可读 geometry，但单样本仍 noisy，且 fitted basis 不是跨层固定 counter。"),
        (8, "Broad-bank rank-3 是否自然参与 aggregation", "verified", "7 个 frozen layers；每层 100 paired units；aligned removal vs equal-norm orthogonal；natural/restored 两种状态；Gemma 按 exact Top-6 分层 bank 重跑", f"Qwen L21/L23 restoration mediation +0.166/+{f(qwen_retrieval_l23['restoration_mediation_mean'])}；Gemma L29 +{f(gemma_retrieval_l29['restoration_mediation_mean'])}；其余 frozen late layers约 0", "count-aligned retrieval subspace 在局部 aggregation window 被自然计算使用。"),
        (9, "Answer query 是否形成可比较的 exact-count manifold", "verified", "seed-held-out PCA-32 nearest-centroid 为主要紧致度/可分性读数；L2 logistic 为辅助 robustness；3D 层只服务展示", "代表层 accuracy 约 53%–56%，nearest-centroid MAD 0.615–0.640；post-hoc 3D display 的 61%/63% 不作正式比较", "建立 non-thinking counter-manifold baseline，供后续按同一协议与 native thinking 比较；不把 classifier 当作机制证据。"),
        (10, "Late answer state 是否可执行并影响输出", "verified", "full donor answer-state patch + answer rank-3 vs equal-norm orthogonal removal", "correct-only donor hit Qwen 96.6%、Gemma 96.0%；peak removal +0.878/+1.222 counts", "晚层完整 answer state 具有充分性；其 count-aligned component 具有方向特异必要性。"),
        (11, "Late write/transport 是否能定位", "verified", "Qwen natural OV injection/removal/mediation；Gemma exact residual mediation；adjacent 1× matched control", "Qwen OV slope +0.0640、removal +0.0732；Gemma L37 residual mediation +0.0864；1× transport约 0.95/0.98", "Qwen 有较局部的 OV writer，Gemma 有较分布式的 residual path；功能阶段相同，实现粒度不同。"),
        (12, "同一 γ(n) 是否适用于 needle span 内所有 token", "falsified", "all-token frozen-PCA formula test：endpoint-gated、span-gated、ungated prefix curves vs category baseline", "endpoint ΔR² Qwen/Gemma +0.551/+0.326；interior span-gated −0.057/−0.036", "支持 endpoint-restricted descriptive curve；否定整个 span 每个 token 共用同一 γ(n) 的强版本。"),
        (13, "Decoded endpoint rank-3 是否是强局部必要 counter", "falsified", "所有 endpoints 同层 rank-3 removal vs realized-norm-matched orthogonal residual basis", "Qwen 层范围 −0.022…+0.056；Gemma −0.011…+0.022", "当前线性 endpoint rank-3 不能解释主要行为效应，不能称为强局部寄存器。"),
        (14, "同一 broad rank-3 counter 是否持续到更晚层", "falsified", "frozen retrieval-subspace 在 aggregation 与 late layers 逐层干预；Gemma exact Top-6 分层重跑", f"Qwen L23 +{f(qwen_retrieval_l23['restoration_mediation_mean'])} 后 L24/L26/L27=+0.002/+0.006/+0.008；Gemma L29 +{f(gemma_retrieval_l29['restoration_mediation_mean'])}、L35 {f(gemma_retrieval_l35['restoration_mediation_mean'])}", "因果作用集中在局部 retrieval window，不是跨后层不变的 persistent counter。"),
        (15, "Residual 是否逐层保持严格 identity map", "falsified", "adjacent rank-3 maps + ambient-operator cosine + one-block intervention", "selected map cosine 0.767/0.796，而非 1；aligned transport高、orthogonal近 0", "后续 block 选择性接收 count-aligned change，但不会原样复制完整三维 operator。"),
        (16, "Gemma 是否与 Qwen 一样有局部 OV head set", "falsified", "L35H2 与 {L29H4,L35H2} natural carrier/injection/removal package", "完整 localized-OV 判据未通过；L37 exact residual mediation +0.0864", "Gemma 的支持结论是分布式 residual mediator，而不是与 Qwen 同构的局部 OV writer。"),
        (17, "Prompt noise 的主要来源是什么", "partial", "count/seed-context/interaction decomposition + grouped cubic absolute-position control；与原 23 的 attention controls 联合解释", "代表层 count/seed/interaction 方差占比：Qwen 0.599/0.161/0.241，Gemma 0.385/0.228/0.386；position-count ρ≈0.965；去 position 后 3-PC R² Qwen {:.3f}、Gemma {:.3f}".format(float(counter_by_model['Qwen3-8B']['position_residual_pc3_grouped_ridge_r2']), float(counter_by_model['Gemma4-E4B']['position_residual_pc3_grouped_ridge_r2'])), "绝对位置解释 frozen 前三维的大部分 ordering；seed/context 与交互仍贡献噪声，但 token identity、context、attention 的独立因果份额尚未完全识别。当前论文无需继续细分。"),
        (18, "为什么 prompt manifold 浅层出现、answer manifold 深层出现", "partial", "跨层 prompt probe、answer classifier、dense restoration、restoration→attention response、answer patch/removal timing", "prompt 浅层已可读；source reuse 在 Qwen 约到 L20、Gemma约到 L16；answer 可执行性在中后层上升", "局部 occurrence 可早期记录；全局 answer query 需等待 retrieval 与 consolidation。时序与因果边界成立，但“架构为何必然如此”不是单一干预可证明的命题。"),
        (19, "是否建立完整 distributed prompt evidence→retrieval→late answer→output 因果链", "verified", "canonical confirmation seeds 1254–1263×counts 1–10；同一 forward 的 11-arm source restoration、retrieval/late aligned-vs-orthogonal removal 与 2×2 joint block；每模型 1,100 rows、100 paired seed–count units；正式推断先在 seed 内平均 10 counts，再以 10 seeds 为独立单位；两模型均使用 10,000-draw cluster bootstrap + exact seed sign-flip", f"source repair Qwen/Gemma +{f(qwen_serial['source_repair']['mean'])}/+{f(gemma_serial['source_repair']['mean'])} counts；retrieval +{f(qwen_serial['retrieval_mediation']['mean'])}/+{f(gemma_serial['retrieval_mediation']['mean'])}；late +{f(qwen_serial['late_mediation']['mean'])}/+{f(gemma_serial['late_mediation']['mean'])}。Retrieval cluster CI Qwen [{f(qwen_serial['retrieval_mediation']['ci95_low'])}, {f(qwen_serial['retrieval_mediation']['ci95_high'])}]、exact p={p_text(qwen_serial['retrieval_mediation']['exact_sign_flip_p_two_sided'])}；Gemma [{f(gemma_serial['retrieval_mediation']['ci95_low'])}, {f(gemma_serial['retrieval_mediation']['ci95_high'])}]、exact p={p_text(gemma_serial['retrieval_mediation']['exact_sign_flip_p_two_sided'])}", "两模型的 seed-level inference 均支持同一试次内的 ordered partial serial mediation。负 interaction 与剩余 repair 继续反对唯一、穷尽通路。"),
        (20, "是否需要对所有 non-needle token 做 frozen-PCA census", "partial", "all-token capture 已含 endpoint、interior、hard-negative 与确定性 ordinary-passage samples", "ordinary/hard-negative 的 ungated prefix curve ΔR² 为负，未显示与 endpoint 相同 trajectory", "已有足够多类负对照支持当前限定结论；逐 token 无遗漏 census 成本高且不会改变 span-level mechanism，故不再扩展。"),
        (22, "经典 induction-head micro-circuit 是否是 canonical running-index update 的特异机制", "falsified", "独立 30×4 synthetic relation-following assay 冻结一个 head/model；随后在 seeds 1254–1263×counts 1–10 对 previous-successor natural edges 做 pre-O αV subtraction，并与 layer/head/distance/edge-count/attention-mass matched ordinary edges 比较；counts 2–10 为主分析", "synthetic gate 保留 Qwen L5H13 与 Gemma L5H0；但 canonical candidate-minus-control expected-error 为 −0.02193 [−0.03311,−0.01076] 与 −0.01207 [−0.02499,0.00127]，两模型决策均 not_supported", "存在 induction-like relation-following head，但预注册的 canonical edge-specific necessity 不成立；因此不能把 earlier-span routing 定名为已验证的 classical induction-head mechanism。该否定不排除分布式 span evidence、其他 registry 或 fully renormalized QK counterfactual。详见 Appendix B。"),
        (23, "预注册的 identity/context/position nuisance model 与 selected outside-halo edge specificity 是否成立", "falsified", "冻结 Qwen L8/Gemma L9 rank-3 basis 做 30 seeds×8 cells factorial（160 discovery、80 confirmation、2,400 endpoint states/model）；另在 100 confirmation units 上阻断 natural-attention-ranked ordinary halo edges，并分别匹配 exact-distance random 与 attention-mass controls", "factorial held-out full R² 为 −0.0221/−0.0893；最大 factor ΔR² 仅 Qwen position +0.0175、Gemma identity +0.0031。candidate removal 对两个 controls 的 expected-error CI 在两模型均跨 0，candidate_exceeds_both_controls=false", "强解释包被否定：三类受控操作未形成稳定的 held-out nuisance model，选定 halo edges 也没有超出两个 matched controls 的特异必要性。该结果不把自然 prompt noise 唯一分解，也不否定广泛 outside context 与 needle span 的分布式协同。详见 Appendix C。"),
        (25, "固定 prompt endpoint rank-3 是否直接变成 answer rank-3", "falsified", "prompt endpoint aligned removal、full-span restoration、retrieval-subspace mediation 与 late answer interventions 联合判定", "endpoint rank-3 removal 近 0；full-span restoration 强正；retrieval basis 只在局部窗口有效且跨层 operator 非 identity", "否定“一枚固定三维 prompt counter 直接搬到 answer”的强版本。它不是原 19 所缺的必要箭头；原 19 需要的是允许各阶段重参数化的串联中介。"),
        (21, "Opening counting-definition cue 是否为 running geometry 的必要条件", "falsified", "V4.4.2 paired opening-definition removal；完整删除边界、coverage、计算定义与数值见 Appendix A", "两个预选浅层均保留近乎相同的 centroid topology 与相近的 seed-held-out readout；同时 full-state cue displacement 仍随 running index 系统变化", "“opening definition cue 是必要条件”的强命题被证伪：没有它仍形成有序、可读的 running geometry；但 cue 会调制完整 state，因此非必要不等于无作用。该结果不能外推为删除全部 task/query instructions。"),
        (24, "是否继续做跨 final-count N 的 prefix invariance", "closed", "不再运行；当前全部 running-index capture 的 final N=10，报告始终限定为 position-confounded counter-like record", "无新增实验；现有证据不用于声称跨 final-N 的抽象 counter", "该实验只有在升级为 abstract-counter 主张时才必要；当前 mechanism 不作此主张，因此关闭并保留为范围边界。"),
    ]
    status_counts = {key: 0 for key in status_labels}
    for _, _, status, _, _, _ in extension_claims:
        status_counts[status] += 1
    if len(extension_claims) != 25 or status_counts != {
        "verified": 12,
        "falsified": 9,
        "partial": 3,
        "open": 0,
        "closed": 1,
    }:
        raise RuntimeError(f"Unexpected extension audit partition: {status_counts}")
    extension_audit_rows = "".join(
        f"""<tr><td>{source_id}</td><td>{question}</td>
        <td><span class="status status-{status}">{status_labels[status]}</span></td>
        <td>{setting}</td><td>{result}</td><td>{conclusion}</td></tr>"""
        for source_id, question, status, setting, result, conclusion in extension_claims
    )

    payload_json = json.dumps(display_data, separators=(",", ":"), ensure_ascii=False)

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Non-thinking V4.4/V4.4.5：从双位置表征到 broad retrieval 与 late write</title>
<style>
:root {{ --ink:#172033; --muted:#5f6b7a; --line:#d8dee9; --paper:#ffffff; --wash:#f4f6f8;
  --teal:#0f766e; --violet:#7c3aed; --amber:#d97706; --blue:#2563eb; --red:#b42318; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; color:var(--ink); background:#eef1f5; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.68; }}
a {{ color:#155eef; text-decoration:none; }}
.page {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 72px; background:var(--paper); box-shadow:0 18px 50px rgba(23,32,51,.10); }}
header {{ padding:54px 64px 36px; border-bottom:1px solid var(--line); background:linear-gradient(135deg,#f8fafc 0%,#eef8f7 52%,#f6f1ff 100%); }}
.eyebrow {{ margin:0 0 8px; color:var(--teal); letter-spacing:.12em; text-transform:uppercase; font-size:12px; font-weight:700; }}
h1 {{ max-width:900px; margin:0; font-family:Georgia,"Noto Serif SC",serif; font-size:clamp(32px,5vw,55px); line-height:1.12; letter-spacing:-.02em; font-weight:600; }}
.dek {{ max-width:880px; margin:20px 0 0; color:#3c4858; font-size:18px; }}
.meta {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:24px; color:var(--muted); font-size:13px; }}
nav {{ position:sticky; top:0; z-index:20; display:flex; gap:18px; padding:11px 64px; overflow-x:auto; white-space:nowrap; background:rgba(255,255,255,.95); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); font-size:13px; }}
main {{ padding:0 64px 64px; }}
section {{ padding:54px 0 20px; border-bottom:1px solid var(--line); }}
section:last-child {{ border-bottom:0; }}
h2 {{ margin:0 0 16px; font-family:Georgia,"Noto Serif SC",serif; font-size:32px; line-height:1.25; font-weight:600; }}
h3 {{ margin:34px 0 10px; font-size:20px; line-height:1.35; }}
h4 {{ margin:24px 0 8px; font-size:16px; }}
p {{ margin:10px 0; }}
.lead {{ max-width:920px; color:#344054; font-size:17px; }}
.claim {{ padding:18px 20px; margin:22px 0; border-left:4px solid var(--teal); background:#f0fdfa; }}
.claim strong {{ color:#075e58; }}
.boundary {{ border-left-color:var(--amber); background:#fff8eb; }}
.boundary strong {{ color:#9a4b00; }}
.mechanism {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:28px 0; }}
.stage {{ position:relative; min-height:202px; padding:19px; border:1px solid var(--line); background:#fff; }}
.stage::after {{ content:"→"; position:absolute; right:-20px; top:43%; z-index:2; color:#98a2b3; font-size:28px; }}
.stage:last-child::after {{ display:none; }}
.stage-no {{ color:var(--muted); font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.12em; }}
.stage h3 {{ margin:8px 0; }}
.stage p {{ color:#475467; font-size:14px; }}
.paper-mechanism {{ padding:22px; border:1px solid var(--line); background:#fbfcfe; }}
.paper-prompt {{ display:grid; grid-template-columns:1.05fr .45fr 1.05fr .45fr 1.05fr .75fr; align-items:center; gap:9px; margin-bottom:18px; }}
.paper-token {{ padding:10px 8px; border:1px solid #cfd6e2; background:#fff; color:#344054; text-align:center; font-size:12px; }}
.paper-token.context {{ border-style:dashed; color:#667085; background:#f8fafc; }}
.paper-token.query {{ border-color:#8ac8bc; background:#effaf7; color:#075e58; font-weight:700; }}
.paper-flow {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:10px 0 20px; }}
.paper-step {{ position:relative; min-height:132px; padding:13px 12px; border-top:4px solid var(--teal); background:#fff; box-shadow:inset 0 0 0 1px #d8dee8; }}
.paper-step:nth-child(2) {{ border-top-color:#2563eb; }}
.paper-step:nth-child(3) {{ border-top-color:#0891b2; }}
.paper-step:nth-child(4) {{ border-top-color:#7c3aed; }}
.paper-step:nth-child(5) {{ border-top-color:#d97706; }}
.paper-step::after {{ content:"→"; position:absolute; right:-20px; top:42%; z-index:2; color:#8791a3; font-size:24px; }}
.paper-step:last-child::after {{ display:none; }}
.paper-step strong {{ display:block; margin-bottom:5px; color:#172033; font-size:13px; }}
.paper-step span {{ display:block; color:#5e6a7d; font-size:12px; line-height:1.55; }}
.paper-step .operation {{ margin-top:7px; color:#344054; font-weight:650; }}
.layer-lanes {{ display:grid; gap:10px; margin-top:16px; }}
.layer-lane {{ display:grid; grid-template-columns:115px repeat(3,minmax(0,1fr)); border:1px solid #d8dee8; background:#fff; }}
.lane-model {{ display:flex; align-items:center; padding:12px; background:#f2f4f7; color:#172033; font-weight:750; }}
.lane-phase {{ padding:10px 11px; border-left:1px solid #e1e6ee; color:#475467; font-size:11px; line-height:1.5; }}
.lane-phase strong {{ display:block; margin-bottom:3px; color:#172033; font-size:12px; }}
.lane-note {{ margin-top:10px; color:#667085; font-size:11px; }}
.evidence {{ display:inline-block; margin-top:9px; padding:5px 8px; background:#edf7f6; color:#08675f; font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.reading-protocol {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0; margin:28px 0 34px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.protocol-step {{ padding:18px 20px 20px; border-right:1px solid var(--line); background:#fff; }}
.protocol-step:last-child {{ border-right:0; }}
.protocol-no,.step-kicker {{ display:block; margin-bottom:7px; color:var(--teal); font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; }}
.protocol-step h3 {{ margin:0 0 7px; font-size:17px; }}
.protocol-step p {{ margin:0; color:#566176; font-size:13px; }}
.chain-map {{ margin:28px 0 34px; border-top:1px solid #cfd6e2; }}
.chain-row {{ display:grid; grid-template-columns:110px 1fr 1fr 1fr 120px; gap:0; border-bottom:1px solid #e5e9f0; }}
.chain-row > div {{ padding:13px 14px; border-right:1px solid #edf0f4; font-size:13px; }}
.chain-row > div:last-child {{ border-right:0; }}
.chain-row.header > div {{ color:#667085; background:#f8fafc; font-size:11px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }}
.chain-name {{ font-weight:750; color:#172033; }}
.chain-status {{ color:#075e58; font-weight:700; }}
.chain-example {{ margin:20px 0 28px; padding:16px 18px; border-left:3px solid #8ac8bc; background:#f7fbfa; color:#344054; }}
.chain-blueprint {{ margin:24px 0 30px; border-top:2px solid var(--teal); border-bottom:1px solid var(--line); }}
.chain-purpose {{ display:grid; grid-template-columns:120px 1fr; gap:18px; padding:15px 0; border-bottom:1px solid var(--line); }}
.chain-purpose strong {{ color:#075e58; }}
.evidence-triad {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }}
.triad-step {{ padding:17px 18px 19px; border-right:1px solid var(--line); }}
.triad-step:last-child {{ border-right:0; }}
.triad-step h3 {{ margin:0 0 7px; font-size:16px; }}
.triad-step p {{ margin:0; color:#4c596d; font-size:13px; }}
.step-heading {{ display:flex; align-items:baseline; gap:12px; margin:34px 0 12px; padding-top:14px; border-top:1px solid #d9dfe8; }}
.step-heading .step-kicker {{ flex:0 0 auto; margin:0; }}
.step-heading h3 {{ margin:0; }}
.purpose {{ color:#344054; }}
.purpose strong,.setting strong,.result-analysis strong {{ color:#172033; }}
.mini-example {{ display:block; margin-top:8px; color:#596579; font-size:13px; }}
.conclusion-line {{ margin:18px 0 28px; padding:14px 16px; border-left:4px solid var(--teal); background:#f0fdfa; color:#23413f; }}
.conclusion-line strong {{ color:#075e58; }}
.table-wrap {{ overflow-x:auto; margin:18px 0; }}
.collapsible-list {{ margin:12px 0 22px; border-top:1px solid #e5e9f0; }}
.collapsible-list > summary {{ padding:10px 2px; cursor:pointer; color:#475467; font-size:13px; font-weight:650; }}
.collapsible-list[open] > summary {{ margin-bottom:4px; }}
.collapsible-list > .table-wrap {{ margin:0 0 12px; }}
.collapsible-list > ul,.collapsible-list > ol {{ margin-top:4px; margin-bottom:14px; }}
.paper-appendix {{ margin:22px 0 30px; border:1px solid #d8dee8; background:#fbfcfe; }}
.paper-appendix > summary {{ padding:15px 18px; cursor:pointer; color:#344054; font-size:14px; font-weight:750; }}
.paper-appendix[open] > summary {{ border-bottom:1px solid #d8dee8; background:#f5f7fa; }}
.paper-appendix > :not(summary) {{ margin-left:18px; margin-right:18px; }}
.placement-core {{ color:#075e58; font-weight:750; }}
.placement-appendix {{ color:#8a4b08; font-weight:750; }}
.head-table td,.head-table th {{ padding:5px 7px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; line-height:1.45; }}
th,td {{ padding:10px 9px; border-bottom:1px solid #e5e9f0; text-align:left; vertical-align:top; }}
th {{ background:#f8fafc; color:#475467; font-weight:650; white-space:nowrap; }}
td:not(:first-child) {{ font-variant-numeric:tabular-nums; }}
code,.math {{ font-family:"Iowan Old Style",Cambria,Georgia,serif; color:#203251; }}
.formula {{ margin:18px 0; padding:16px 18px; overflow-x:auto; background:#f8fafc; border:1px solid var(--line); font-family:"Iowan Old Style",Cambria,Georgia,serif; font-size:16px; }}
.example {{ display:block; margin-top:8px; color:#475467; font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; font-size:13px; }}
figure {{ margin:30px 0 38px; }}
figcaption {{ max-width:940px; margin:10px auto 0; color:#586579; font-size:13px; line-height:1.55; }}
.figure-title {{ margin:0 0 12px; font-size:17px; }}
.figure-primer {{ margin:0 0 16px; padding:15px 17px; border:1px solid #cfe2df; border-left:4px solid var(--teal); background:#f5fbfa; }}
.figure-primer-header {{ margin:0 0 9px; color:#075e58; font-size:12px; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }}
.figure-primer-grid {{ display:grid; grid-template-columns:1.15fr 1.15fr 1fr; gap:16px; }}
.figure-primer p {{ margin:0; color:#344054; font-size:13px; line-height:1.58; }}
.figure-primer strong {{ color:#172033; }}
.figure-primer .primer-example {{ padding-left:14px; border-left:1px solid #b8d9d5; }}
.figure-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.figure-stack {{ display:grid; grid-template-columns:1fr; gap:18px; }}
.report-image {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
.three-d {{ border:1px solid var(--line); background:linear-gradient(#fbfcfe,#f5f8fb); }}
.three-d-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); }}
.three-d-head label {{ color:#475467; font-size:13px; }}
.three-d-head select {{ margin-left:8px; padding:6px 8px; border:1px solid #b8c1cf; background:#fff; color:var(--ink); }}
.three-d canvas {{ display:block; width:100%; height:460px; cursor:grab; touch-action:none; }}
.three-d canvas:active {{ cursor:grabbing; }}
.chart-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:18px; }}
.line-chart,.bar-chart,.baseline-heatmap,.attention-map {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
.heat-cell {{ stroke:#fff; stroke-width:1.2; }}
.heat-title {{ fill:#172033; font-size:15px; font-weight:650; }}
.heat-row,.heat-x,.heat-y,.heat-legend {{ fill:#536074; font-size:11px; }}
.heat-value {{ fill:#172033; font-size:11px; font-weight:650; }}
.heat-value.inverse,.heat-rank {{ fill:#fff; }}
.attention-cell {{ stroke:#fff; stroke-width:.55; }}
.attention-cell.selected {{ stroke:#172033; stroke-width:1.5; }}
.heat-rank {{ font-size:7px; font-weight:650; pointer-events:none; }}
.plot-bg {{ fill:#fbfcfe; stroke:#cfd6e2; }}
.grid {{ stroke:#e7ebf0; stroke-width:1; }}
.grid.vertical {{ stroke:#eef1f4; }}
.zero {{ stroke:#768196; stroke-width:1.2; }}
.reference {{ stroke:#98a2b3; stroke-width:1.2; stroke-dasharray:4 4; }}
.reference-label,.tick,.legend-label,.bar-label,.bar-value {{ fill:#536074; font-size:12px; }}
.stable-band {{ fill:#dff4ee; opacity:.72; }}
.legend-band {{ stroke:#a8d9cc; stroke-width:1; opacity:1; }}
.axis-label {{ fill:#2f3b4d; font-size:13px; font-weight:600; }}
.series-line {{ fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round; }}
.series-dot {{ stroke:#fff; stroke-width:1; }}
.ci-whisker {{ stroke-width:1.25; opacity:.82; }}
.ci-dot {{ stroke-width:1.5; }}
.legend-line {{ stroke-width:2.4; }}
.bar {{ opacity:.88; }}
.attention-example {{ border:1px solid var(--line); background:#fff; }}
.attention-example-head {{ display:flex; justify-content:space-between; gap:18px; padding:14px 16px; border-bottom:1px solid var(--line); background:#f8fafc; }}
.attention-example-head span {{ color:#667085; font-size:12px; }}
.attention-summary {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; padding:14px 16px; }}
.attention-summary-item {{ padding:9px 10px; border:1px solid #e4e8ef; background:#fbfcfe; }}
.attention-summary-item strong,.attention-summary-item span {{ display:block; }}
.attention-summary-item strong {{ font-size:11px; }}
.attention-summary-item span {{ color:#475467; font-size:14px; font-variant-numeric:tabular-nums; }}
.attention-mass-track {{ height:5px; margin-top:6px; overflow:hidden; background:#e8edf3; }}
.attention-mass-track i {{ display:block; height:100%; background:#b42318; }}
.attention-token-legend {{ padding:0 16px 12px; color:#667085; font-size:12px; }}
.attention-token-legend > span {{ display:inline-block; width:35px; height:11px; margin-right:7px; vertical-align:-1px; background:linear-gradient(90deg,rgba(180,35,24,.08),rgba(180,35,24,.94)); }}
.attention-needle-list {{ border-top:1px solid #e7ebf0; }}
.needle-attention-row {{ display:grid; grid-template-columns:150px 1fr; gap:12px; padding:10px 16px; border-bottom:1px solid #edf0f4; }}
.needle-attention-meta strong,.needle-attention-meta span {{ display:block; }}
.needle-attention-meta strong {{ font-size:12px; }}
.needle-attention-meta span {{ color:#667085; font-size:11px; font-variant-numeric:tabular-nums; }}
.needle-token-line {{ display:flex; flex-wrap:wrap; align-content:flex-start; gap:2px; }}
.attention-token {{ display:inline-block; min-width:8px; padding:2px 3px; border-radius:2px; font:600 11px ui-monospace,SFMono-Regular,Consolas,monospace; cursor:help; }}
.top-token-list {{ columns:2; padding:0 38px 15px; color:#475467; font-size:12px; }}
.token-region {{ color:#667085; }}
.attention-gallery-controls {{ display:flex; align-items:end; gap:14px; flex-wrap:wrap; padding:14px 16px; border:1px solid var(--line); border-bottom:0; background:#f8fafc; }}
.attention-gallery-controls label {{ color:#344054; font-size:12px; font-weight:700; }}
.attention-gallery-controls select {{ display:block; min-width:205px; margin-top:5px; padding:7px 9px; border:1px solid #b8c1cf; background:#fff; color:var(--ink); }}
.attention-gallery-controls > span {{ margin-left:auto; color:#667085; font-size:12px; }}
.attention-gallery-panel[hidden] {{ display:none !important; }}
.control-definition {{ margin:16px 0 18px; padding:15px 17px; border-top:3px solid #b23a30; border-bottom:1px solid #e3c5c1; background:#fffaf9; }}
.control-definition > h4 {{ margin:0 0 5px; color:#6f201a; }}
.control-definition > p {{ margin:0 0 12px; color:#536074; font-size:12px; }}
.control-definition-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:0; border-top:1px solid #ead8d5; border-left:1px solid #ead8d5; }}
.control-term {{ padding:11px 12px; border-right:1px solid #ead8d5; border-bottom:1px solid #ead8d5; background:rgba(255,255,255,.62); font-size:12px; }}
.control-term strong {{ display:block; margin-bottom:4px; color:#302322; }}
.control-term code {{ color:#7a271a; }}
.control-contrast {{ margin-top:11px; padding-top:10px; border-top:1px solid #ead8d5; color:#3e3332; font-size:12px; }}
.control-contrast .math {{ white-space:nowrap; }}
.attention-overview {{ border:1px solid var(--line); background:#fff; }}
.attention-category-block,.attention-raster-block {{ padding:11px 14px; border-bottom:1px solid var(--line); }}
.attention-section-heading {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:10px; color:#172033; font-size:13px; }}
.attention-section-heading span {{ color:#667085; font-size:11px; }}
.attention-category-bar {{ display:flex; width:100%; height:18px; overflow:hidden; border:1px solid #d0d7e2; background:#f3f5f8; }}
.attention-category-segment {{ display:block; min-width:1px; height:100%; }}
.attention-category-legend {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin-top:10px; }}
.attention-category-key {{ display:grid; grid-template-columns:10px 1fr auto; align-items:center; gap:6px; min-width:0; color:#5e6a7d; font-size:10px; }}
.attention-category-key i {{ width:10px; height:10px; }}
.attention-category-key strong {{ overflow:hidden; color:#344054; font-weight:650; text-overflow:ellipsis; white-space:nowrap; }}
.attention-category-key b {{ color:#172033; font-variant-numeric:tabular-nums; }}
.attention-bin-scroll {{ overflow-x:auto; padding-bottom:3px; }}
.attention-bin-grid {{ display:grid; grid-template-columns:repeat(24,minmax(25px,1fr)); gap:3px; min-width:780px; padding:2px 0; }}
.attention-bin {{ position:relative; height:20px; cursor:help; overflow:visible; }}
.attention-bin-tile {{ display:block; width:100%; height:20px; border:1px solid rgba(103,42,36,.20); }}
.attention-bin-city {{ position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); z-index:1; padding:1px 2px; background:rgba(255,248,246,.82); color:#6f201a; font:750 7.5px ui-monospace,SFMono-Regular,Consolas,monospace; line-height:1.1; white-space:nowrap; pointer-events:none; }}
.attention-bin.has-needle .attention-bin-tile {{ border-top:4px solid #a92f26; }}
.attention-bin.has-negative .attention-bin-tile {{ border-bottom:4px solid #dca33a; }}
.attention-bin.has-query .attention-bin-tile {{ outline:2px solid #587da5; outline-offset:-2px; }}
.attention-bin.is-boundary .attention-bin-tile {{ border-left-color:#667085; border-right-color:#667085; }}
.attention-bin-legend {{ display:flex; align-items:center; flex-wrap:wrap; gap:7px; margin-top:10px; color:#667085; font-size:10px; }}
.attention-bin-scale {{ display:inline-flex; align-items:center; gap:2px; margin-right:8px; color:#475467; }}
.attention-bin-scale i {{ width:14px; height:11px; border:1px solid rgba(44,75,78,.15); }}
.attention-bin-legend > span.legend-needle,.attention-bin-legend > span.legend-negative,.attention-bin-legend > span.legend-clean {{ width:18px; height:10px; margin-left:9px; background:#eef2f4; }}
.attention-bin-legend > span.legend-needle {{ border-top:4px solid #d84a3f; }}
.attention-bin-legend > span.legend-negative {{ border-bottom:4px solid #dca33a; }}
.attention-bin-legend > span.legend-clean {{ border:1px solid #2f8a83; }}
.attention-document-map {{ display:block; width:100%; height:auto; background:#fff; }}
.attention-panel-label {{ fill:#172033; font-size:13px; font-weight:700; }}
.attention-position-track {{ fill:#edf1f6; stroke:#cfd6e2; }}
.attention-position-tick {{ stroke:#98a2b3; stroke-width:1; }}
.attention-needle-block {{ fill:#d92d20; opacity:.86; }}
.attention-hard-negative-tick {{ fill:#fdb022; opacity:.65; }}
.attention-span-row-label {{ fill:#344054; font-size:11px; font-weight:650; }}
.attention-span-bar-bg {{ fill:#f1f3f6; }}
.attention-span-bar {{ fill:#d92d20; opacity:.86; }}
.attention-bar-value {{ fill:#7a271a; font-size:11px; font-weight:700; font-variant-numeric:tabular-nums; }}
.attention-document-legend {{ display:flex; gap:9px; align-items:center; flex-wrap:wrap; padding:0 16px 12px; color:#667085; font-size:11px; }}
.attention-document-legend span {{ display:inline-block; width:18px; height:9px; margin-left:8px; }}
.attention-document-legend .legend-needle {{ margin-left:0; background:#d92d20; opacity:.86; }}
.attention-document-legend .legend-negative {{ background:#fdb022; opacity:.55; }}
.hidden-state-chain {{ display:grid; grid-template-columns:minmax(0,1fr) 190px minmax(0,1fr) 190px minmax(0,1fr); gap:10px; align-items:stretch; margin:20px 0; }}
.hidden-state-node,.hidden-state-arrow {{ padding:14px; border:1px solid var(--line); background:#fbfcfe; }}
.hidden-state-node {{ border-top:4px solid var(--teal); }}
.hidden-state-node strong,.hidden-state-arrow strong {{ display:block; margin-bottom:5px; }}
.hidden-state-node small,.hidden-state-arrow small {{ color:#667085; }}
.hidden-state-arrow {{ position:relative; background:#f8fafc; color:#344054; font-size:12px; }}
.hidden-state-arrow::after {{ content:"→"; position:absolute; right:-9px; top:42%; z-index:1; color:#98a2b3; font-size:19px; }}
.hidden-state-arrow:last-of-type::after {{ display:block; }}
.experiment {{ display:grid; grid-template-columns:150px 1fr; gap:18px; margin:18px 0; padding:18px 0; border-top:1px solid #e7ebf0; }}
.experiment-label {{ color:var(--muted); font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace; text-transform:uppercase; letter-spacing:.08em; }}
.experiment h4 {{ margin:0 0 6px; }}
.result-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:18px 0; }}
.result {{ padding:15px 16px; border:1px solid var(--line); }}
.result .value {{ display:block; color:var(--teal); font:650 23px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.result .label {{ color:#596579; font-size:12px; }}
.path {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); align-items:stretch; gap:12px; margin:22px 0; }}
.node {{ position:relative; padding:14px 12px; border:1px solid var(--line); background:#f9fafb; font-size:13px; }}
.node::after {{ content:"→"; position:absolute; right:-18px; top:35%; color:#98a2b3; font-size:20px; }}
.node:last-child::after {{ display:none; }}
.node strong {{ display:block; margin-bottom:5px; }}
.node small {{ color:#667085; }}
.contrast-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:18px 0; }}
.contrast-lane {{ padding:16px; border:1px solid var(--line); border-top:4px solid var(--teal); background:#fbfcfe; }}
.contrast-lane.restored {{ border-top-color:var(--amber); }}
.contrast-source {{ margin-bottom:12px; font-weight:700; }}
.contrast-source small {{ display:block; margin-top:3px; color:#667085; font-weight:400; }}
.contrast-branches {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.contrast-arm {{ min-height:108px; padding:12px; border:1px solid #cfd6e2; background:#fff; font-size:13px; }}
.contrast-arm strong {{ display:block; margin-bottom:5px; }}
.contrast-arm small {{ color:#667085; }}
.contrast-result {{ margin-top:11px; padding-top:10px; border-top:1px solid var(--line); color:#344054; font-size:13px; }}
.contrast-result strong {{ color:#172033; }}
.evidence-map td:nth-child(1) {{ white-space:nowrap; font-weight:650; }}
.source-list {{ columns:2; column-gap:32px; font-size:12px; color:#596579; }}
.source-list li {{ break-inside:avoid; margin:0 0 8px; }}
.pill {{ display:inline-block; padding:2px 7px; border:1px solid #b8d9d5; color:#08675f; background:#f0fdfa; font-size:11px; white-space:nowrap; }}
.audit-summary {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:22px 0; }}
.audit-card {{ padding:15px 16px; border:1px solid var(--line); background:#fff; }}
.audit-card strong {{ display:block; font:700 26px ui-monospace,SFMono-Regular,Consolas,monospace; }}
.audit-card span {{ color:#667085; font-size:12px; }}
.status {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; font-weight:700; white-space:nowrap; }}
.status-verified {{ color:#075e58; background:#dff7f2; border:1px solid #a8d9cc; }}
.status-falsified {{ color:#9f1f18; background:#fff0ee; border:1px solid #f5b7b1; }}
.status-partial {{ color:#8a4b08; background:#fff6df; border:1px solid #f0cf83; }}
.status-open {{ color:#475467; background:#f2f4f7; border:1px solid #d0d5dd; }}
.status-closed {{ color:#344054; background:#eaecf0; border:1px solid #b9c0cc; }}
.extension-audit td:nth-child(1) {{ width:38px; color:#667085; }}
.extension-audit td:nth-child(2) {{ min-width:190px; font-weight:650; }}
.extension-audit td:nth-child(4) {{ min-width:260px; }}
.extension-audit td:nth-child(5),.extension-audit td:nth-child(6) {{ min-width:230px; }}
@media(max-width:900px) {{ header,main {{ padding-left:28px; padding-right:28px; }} nav {{ padding-left:28px; }} .mechanism,.figure-grid,.chart-pair,.result-grid,.audit-summary,.contrast-grid,.reading-protocol,.evidence-triad,.figure-primer-grid,.paper-flow,.attention-summary,.attention-span-cards,.hidden-state-chain {{ grid-template-columns:1fr; }} .paper-prompt {{ grid-template-columns:1fr 1fr; }} .paper-step::after,.stage::after,.hidden-state-arrow::after {{ display:none; }} .layer-lane {{ grid-template-columns:1fr 1fr; }} .lane-model {{ grid-column:1/-1; }} .figure-primer .primer-example {{ padding-left:0; padding-top:10px; border-left:0; border-top:1px solid #b8d9d5; }} .protocol-step,.triad-step {{ border-right:0; border-bottom:1px solid var(--line); }} .protocol-step:last-child,.triad-step:last-child {{ border-bottom:0; }} .chain-row {{ grid-template-columns:1fr; }} .chain-row.header {{ display:none; }} .chain-row > div {{ border-right:0; padding:9px 14px; }} .chain-row > div::before {{ display:block; color:#8791a3; font-size:10px; font-weight:700; text-transform:uppercase; }} .chain-row > div:nth-child(2)::before {{ content:"Mechanism"; }} .chain-row > div:nth-child(3)::before {{ content:"Representation"; }} .chain-row > div:nth-child(4)::before {{ content:"Causal test"; }} .chain-row > div:nth-child(5)::before {{ content:"Status"; }} .path {{ grid-template-columns:1fr; }} .node::after {{ display:none; }} .source-list {{ columns:1; }} .needle-attention-row {{ grid-template-columns:1fr; }} .top-token-list {{ columns:1; }} .attention-gallery-controls > span {{ margin-left:0; width:100%; }} }}
@media(max-width:560px) {{ .page {{ width:100%; margin:0; }} header,main {{ padding-left:18px; padding-right:18px; }} nav {{ padding-left:18px; }} h1 {{ font-size:34px; }} h2 {{ font-size:27px; }} .experiment,.chain-purpose {{ grid-template-columns:1fr; gap:5px; }} .three-d canvas {{ height:380px; }} .contrast-branches {{ grid-template-columns:1fr; }} .contrast-arm {{ min-height:0; }} }}
@media(max-width:900px) {{ .attention-category-legend {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .attention-section-heading {{ align-items:flex-start; flex-direction:column; gap:3px; }} .control-definition-grid {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:#fff; }} .page {{ width:100%; margin:0; box-shadow:none; }} nav {{ display:none; }} section,figure {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<article class="page">
<header>
  <p class="eyebrow">Realistic NIAH · Non-thinking V4.4/V4.4.5 · Mechanistic analysis</p>
  <h1>Non-thinking 模型如何计数：分布式证据、广域检索与晚层写入</h1>
  <p class="dek">正文按三步机制组织：浅层形成 noisy running-index evidence，中层由 answer query 广域取回，晚层将其整合为可执行 count state。每一步只保留最清楚的一项 representation 与一项 matched causal test；描述性诊断、失败实验、模型特定 writer 与完整注册审计均保留在 Appendix。</p>
  <div class="meta"><span>模型：Qwen3-8B / Gemma4-E4B</span><span>计数范围：1–10</span><span>canonical seeds：1234–1263</span><span>位置：needle end / <code>Total:</code> 后首数字前</span><span>更新：2026-08-29</span></div>
</header>
<nav aria-label="report sections">
  <a href="#summary">三步机制</a><a href="#representation">核心表征图</a><a href="#formation">Step 1 · form</a>
  <a href="#retrieval">Step 2 · retrieve</a><a href="#write">Step 3 · consolidate</a><a href="#integrated-chain">Q19 · 整链闭环</a><a href="#ledger">证据取舍</a><a href="#appendix">Appendix</a>
</nav>
<main>

<section id="baseline">
  <h2>1. 任务与行为基线：模型需要解释什么？</h2>
  <p class="lead">在进入 hidden-state geometry 前，先固定模型要解释的外部行为。每个模型、每个 gold needle count <span class="math">N∈{{0,…,10}}</span> 有 30 个样本；下图将 Qwen 与 Gemma 分开显示。</p>
  <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>确认机制实验面对的行为误差类型：模型是在所有 count 上随机失败，还是主要在较大 count 上系统性低估。<span class="mini-example"><strong>直观例子：</strong>若 gold=8 而模型常答 5，那么机制干预除了看 exact accuracy，还必须看 expected count 是否向 8 移动，以及 absolute error 减少了多少 counts。</span></p></div>
  <details class="paper-appendix"><summary>Behavior baseline · 展开按 count 热图与误差定义</summary>
  <div class="formula"><strong>单例误差与按 count 准确率。</strong>生成计数为 <span class="math">ŷ</span> 时，signed error 为 <span class="math">e_{{signed}}=ŷ−N</span>，absolute error 为 <span class="math">e_{{abs}}=|ŷ−N|</span>；<span class="math">Acc(N)=S_N^{{−1}}Σ_s 𝟙[ŷ_{{s,N}}=N]</span>。行为 MAD 是所有单例 absolute error 的平均：<span class="math">MAD_{{behavior}}=M^{{−1}}Σ_i e_{{abs,i}}</span>；wrong-only MAD 只在 <span class="math">e_{{abs}}>0</span> 的样本上取平均。单位均为 counts，而不是 median absolute deviation。<span class="example">例：gold <em>N</em>=8、输出 ŷ=6，则 signed error=−2、absolute error=2；若 30 个 N=8 样本中 7 个答对，则 Acc(8)=7/30=23.3%；若三例 absolute errors 为 [0,1,2]，behavior MAD=1，而 wrong-only MAD=(1+2)/2=1.5 counts。</span></div>
  <div class="figure-stack">
    <figure><h4 class="figure-title">图 1a · Qwen3-8B baseline</h4>{baseline_heatmaps['Qwen3-8B']}<figcaption>每一列是 gold needle count N；上排色块与数字是 exact accuracy（深绿=更高），下排是 mean absolute error（深红=更大，单位 counts）。每格聚合 30 个样本。全计数加权汇总：accuracy={pct(baseline_summary['Qwen3-8B']['accuracy'])}，MAD={f(baseline_summary['Qwen3-8B']['mad'])}，mean signed error={f(baseline_summary['Qwen3-8B']['signed'])}。</figcaption></figure>
    <figure><h4 class="figure-title">图 1b · Gemma4-E4B baseline</h4>{baseline_heatmaps['Gemma4-E4B']}<figcaption>坐标、颜色和每格样本数与左图相同。全计数加权汇总：accuracy={pct(baseline_summary['Gemma4-E4B']['accuracy'])}，MAD={f(baseline_summary['Gemma4-E4B']['mad'])}，mean signed error={f(baseline_summary['Gemma4-E4B']['signed'])}。两模型在较大 N 上主要低估，因此后文同时报告 accuracy、absolute error 与 count shift。</figcaption></figure>
  </div>
  </details>
</section>

<section id="summary">
  <h2>先说机制：non-thinking 计数是一条分阶段、会重参数化的计算链</h2>
  <div class="claim"><strong>中心机制。</strong>最符合现有全部 matched-control 结果的解释不是“模型在某个 endpoint 写下整数，再把同一枚三维寄存器原样传到输出”，而是：<strong>完整 needle spans 先形成分布式、可复用的上游证据；answer query 再从多个 spans 做 broad retrieval；聚合结果随后被重写成晚层可执行的 count-aligned answer state；最后由模型特定的 writer/residual path 推向数字 logits。</strong>“重参数化”意味着每个阶段可以携带同一个任务变量，但不必使用相同层、相同 token site 或同一个固定 basis。</div>
  <figure><h4 class="figure-title">机制图 M0 · 从 needle-span evidence 到数字输出的分层计算链</h4>
    <div class="paper-mechanism" role="img" aria-label="Non-thinking counting mechanism from distributed needle spans through answer-query retrieval, multi-head aggregation, late consolidation, and output writing, with approximate layer landmarks for Qwen and Gemma">
      <div class="paper-prompt" aria-hidden="true">
        <div class="paper-token">Needle span 1</div><div class="paper-token context">ordinary context</div>
        <div class="paper-token">Needle span 2</div><div class="paper-token context">…</div>
        <div class="paper-token">Needle span N</div><div class="paper-token query">Answer query<br><code>Total:</code></div>
      </div>
      <div class="paper-flow">
        <div class="paper-step"><strong>I · FORM</strong><span>Needle 文本改变完整 span 内多个 token 的 residual states，形成带 occurrence-order 信息但受上下文噪声影响的分布式 evidence；endpoint 只是可观察的 readout。</span><span class="operation">needle input → H<sub>span</sub><sup>(ℓ)</sup></span></div>
        <div class="paper-step"><strong>II · RETRIEVE</strong><span>末尾 answer query 通过多个 broad heads 回看多条 active spans；各 head 取得的 V-content 经 W<sub>O</sub> 写入并在 query residual 中聚合。</span><span class="operation">Σ<sub>h</sub> W<sub>O,h</sub>(α<sub>h</sub>V<sub>h</sub>)</span></div>
        <div class="paper-step"><strong>III · CONSOLIDATE</strong><span>后续 blocks 将 noisy retrieved content 重参数化为可 donor-transfer、对 count-aligned removal 敏感的 answer state，并最终写向数字 logits。</span><span class="operation">answer state → unembedding → digit</span></div>
      </div>
      <div class="layer-lanes">
        <div class="layer-lane">
          <div class="lane-model">Qwen3-8B<br><small>L0–35</small></div>
          <div class="lane-phase"><strong>FORM · L0–20</strong>Endpoint ordering 在约 L8–13 可读；whole-span restoration 到 L20 仍能救回行为，L15–22 逐步交接。</div>
          <div class="lane-phase"><strong>RETRIEVE · 约 L15–23</strong>恢复 source 对 broad routing 的主要影响约到 L20；冻结 count-aligned aggregation 的方向特异窗口在 L21–23。</div>
          <div class="lane-phase"><strong>CONSOLIDATE / OUTPUT · 约 L24–35</strong>Answer-state donor adoption 在 L26 为 53.3%、L29 为 98.3%；L28 H16/H19 提供一条较局部的写入路径。</div>
        </div>
        <div class="layer-lane">
          <div class="lane-model">Gemma4-E4B<br><small>L0–41</small></div>
          <div class="lane-phase"><strong>FORM · L0–16</strong>Endpoint ordering 在约 L9–11 可读；whole-span restoration 在 L16 仍强，L16→17 出现陡峭 source-reuse cliff。</div>
          <div class="lane-phase"><strong>RETRIEVE · L17–29</strong>L17 后恢复旧 prompt positions 已太晚；现有实验只把 count-aligned aggregation 的方向特异窗口明确定位到 L29，不逐层唯一命名 L17–28。</div>
          <div class="lane-phase"><strong>CONSOLIDATE / OUTPUT · 约 L30–41</strong>Answer-state donor adoption 在 L31 为 87.5%、L35 为 98.8%；L37 后由较分布式 residual path 推向 terminal state。</div>
        </div>
      </div>
      <p class="lane-note">层号均为 zero-based empirical landmarks。色块表示当前实验能支持的大致功能窗口，允许重叠；它们不是硬边界，也不声称每个区间内所有 heads 都执行同一功能。</p>
    </div>
    <figcaption>上半部分把机制压成论文正文使用的三步：形成分布式 running-index evidence、由 answer query 广域取回并聚合、在晚层整合为可执行 count state 后输出。下半部分给出两个模型由 matched interventions 约束的大致层段；范围允许重叠，不是按 classifier 峰值切出的硬边界。Qwen 的 source→retrieval 交接较渐进，Gemma 在 L16→17 出现更陡的 source-reuse cliff；模型特定 writer 只作为第三步的实现细节放入 Appendix。</figcaption>
  </figure>
  <div class="mechanism" role="img" aria-label="三阶段 non-thinking 计数机制：形成、检索和整合输出">
    <div class="stage"><span class="stage-no">STAGE I · FORM</span><h3>Needle → 分布式 span state</h3><p>等长替换 needle 输入会让 occurrence states 离开正确 running-index clean cloud；在保持 corrupt 输入不变时，把 clean full-span states 写回又能救答案。局部 endpoint 很早显示 ordering 与干预响应，但它只是 readout，不是已确认的唯一 mediator。</p><span class="evidence">held-out state retention · full-span state→count patch</span></div>
    <div class="stage"><span class="stage-no">STAGE II · RETRIEVE</span><h3>Answer query 广域取回</h3><p>多个 frozen broad heads 同时回看多条 active spans，并把 retrieved V-content 聚合写入 answer-query residual。</p><span class="evidence">broad routing · matched head ablation</span></div>
    <div class="stage"><span class="stage-no">STAGE III · CONSOLIDATE</span><h3>形成可执行 answer state</h3><p>中后层将聚合内容重参数化为可 donor-transfer、对 aligned removal 敏感的 state，再经模型特定路径写向数字输出。</p><span class="evidence">sufficiency · necessity · partial mediation</span></div>
  </div>
  <div class="chain-example"><strong>一个最简单的直观例子。</strong>假设长 passage 中依次出现三条有效 record。模型读到第 1/2/3 条时，相关 span states 会带有“这是目前第几个”的 noisy ordering；到末尾 <code>Total:</code> 处，answer query 再从三条远端 evidence 取回内容，合成一个更接近“总数=3”的 answer-side state，最后把该 state 写向数字 <code>3</code>。实验要分别证明：上游 evidence 存在、聚合 state 可读、以及删除候选路径真的比 matched control 更伤输出。</div>

  <h3>每个机制步骤统一使用的三层证据标准</h3>
  <div class="reading-protocol" aria-label="Mechanism representation causal-test validation protocol">
    <div class="protocol-step"><span class="protocol-no">01 · Mechanism</span><h3>先写可证伪的计算主张</h3><p>明确 source、receiver、预期时序与边界；同时写出什么结果会否定强版本。这里不使用 classifier 或 attention 图替代机制。</p></div>
    <div class="protocol-step"><span class="protocol-no">02 · Representation</span><h3>再问候选状态应呈现什么结构</h3><p>用 held-out probe、centroid geometry、attention routing 或 carrier slope 检查候选变量是否可读。它负责定位和描述，不单独证明自然因果使用。</p></div>
    <div class="protocol-step"><span class="protocol-no">03 · Causal test</span><h3>最后用 matched intervention 判定</h3><p>用 corruption、patch、removal、ablation 或 mediation，并匹配位置、层、head 数、token budget 或 realized norm。只有候选操作超过 control 才升级机制结论。</p></div>
  </div>

  <h3>三步机制及其正文判定门槛</h3>
  <div class="chain-map" role="table" aria-label="Three mechanism stages and their evidence gates">
    <div class="chain-row header" role="row"><div>Chain</div><div>Mechanism</div><div>Representation prediction</div><div>Decisive causal test</div><div>Status</div></div>
    <div class="chain-row" role="row"><div class="chain-name">A · FORM</div><div>Needle 输入先改变完整 span 内的多-token hidden states；其中一部分 states 在早层仍可被后续计算使用。</div><div>Clean/needle-corrupt/ordinary-corrupt 三臂逐层量化 full-vector deformation，并用独立 clean seeds 冻结 running-index centroids，检验扰动后 state 是否离开正确 count cloud。</div><div>等长 active-span input corruption 直接检验 needle→state；固定 corrupt 输入的 clean full-span state patch 直接检验 state→count。</div><div class="chain-status">Input→state 的 count-specific organization 与 state→count use 均获跨模型支持；单 endpoint rank-3 作为必要 mediator 仍不受支持</div></div>
    <div class="chain-row" role="row"><div class="chain-name">B · RETRIEVE</div><div>Answer query 经 broad head bank 聚合多个 spans。</div><div>Broad score 与合计 post-O write 携带 noisy final-count geometry。</div><div>Top-K vs layer-matched random；aligned vs equal-norm orthogonal；source mediation。</div><div class="chain-status">支持一条部分路径</div></div>
    <div class="chain-row" role="row"><div class="chain-name">C · CONSOLIDATE</div><div>Retrieval 结果被重写成晚层可执行 answer state，再由模型特定路径推向数字输出。</div><div>Final-count geometry 在晚层更紧致、可 donor-transfer，且自然 computation 对 count-aligned component 敏感。</div><div>Full-state donor patch、rank-3 removal；Q19 在同一 forward 中检验 retrieval→late→output 的有序中介。</div><div class="chain-status">跨模型支持可执行 late state；writer 粒度放 Appendix</div></div>
  </div>

  <h3>正文与 Appendix 的证据取舍</h3>
  <div class="table-wrap"><table><thead><tr><th>Stage</th><th>论文正文核心</th><th>Appendix / report audit</th><th>取舍理由</th></tr></thead><tbody>
    <tr><td>Form</td><td class="placement-core">Prompt PCA 示意；held-out correct-centroid retention；dense full-span restoration</td><td class="placement-appendix">Raw RMS deformation、counter-property、position control、endpoint rank-3 null、完整注册表</td><td>正文保留一项直观 representation 与两条直接 causal arrows；纯描述或不跨模型的量放附录。</td></tr>
    <tr><td>Retrieve</td><td class="placement-core">自然 attention span 示例；top-K vs layer-matched random 剂量曲线</td><td class="placement-appendix">完整 head atlas、token 级 gallery、broad-bank classifier、逐层 retrieval-subspace 技术表</td><td>正文先证明“看向多处”与“删掉这些 heads 更伤”；低维写入细节留作机制限定。</td></tr>
    <tr><td>Consolidate</td><td class="placement-core">Answer-state PCA；full-state donor patch；late rank-3 removal；Q19 ordered mediation</td><td class="placement-appendix">相邻层 map、one-block chord transport、Qwen/Gemma writer 分解、Q21–Q23 与全部复现账本</td><td>正文聚焦晚层 state 的紧致化、充分性和必要性；局部实现与失败归因不承担中心主张。</td></tr>
  </tbody></table></div>

  <div class="conclusion-line"><strong>机制总览目前结论。</strong>两模型共享“span evidence → broad retrieval / aggregation → late answer state / output”的三步功能顺序，但不共享同样尖锐的层边界或同一组局部 writer heads。正文先逐段建立每个箭头，随后由 Q19 在同一个 forward 内做最终串联检验。</div>
  <div class="claim boundary"><strong>论文级边界。</strong>Prompt endpoint geometry 是 descriptive readout；span-level evidence 是被因果支持的 source unit；broad retrieval 是一条被自然使用的 aggregation pathway；late answer state 是充分且方向特异必要的执行状态。Joint retrieval×late interaction 为负，fully aligned block 后仍有正 source repair，明确反对“两个冻结 mediator 就是完整唯一 circuit”。现有证据也不要求解析 span 内哪个 token 是地址或内容，不要求把 induction-like head 命名为 canonical induction mechanism，更不要求 Qwen 与 Gemma 共享同一组 writer heads。</div>
</section>

<section id="integrated-chain">
  <h2>整链闭环 · Q19：修复上游证据后，收益是否依次经过 retrieval 和 late state？</h2>
  <p class="lead">前面各节分别证明了三件事：完整 needle spans 能救回答案；<strong>retrieval</strong>（末尾 answer query 从前文取回信息）存在；<strong>late state</strong>（生成数字前的晚层答案状态）能控制输出。Q19 要排除一种更弱的解释——这三件事可能彼此无关，只是事后看起来可以串起来。做法是：对<strong>同一批 prompts</strong>，在一次 intervened forward 内按深度依次修复或阻断这些阶段，并同时读取中间状态和最终答案。</p>
  <div class="chain-purpose"><span class="step-kicker">核心问题</span><p><strong>如果信息真的按 source → retrieval → late state → output 流动，</strong>那么恢复早层 spans 应先改变后续 retrieval；移除 retrieval 中与 count 对齐的变化，应削弱更晚的 count state；再移除 late count state，应进一步损伤答案。<span class="mini-example"><strong>直观例子：</strong>正确答案是 8。破坏 needles 后模型只倾向答 3；恢复早层完整 spans 后倾向答 7。若随后阻断 retrieval，结果退到 6；再阻断晚层 count state，结果退到 4，就说明修复得到的信息确实依次经过了这两个阶段。</span></p></div>

  <div class="path" aria-label="Experiment 19 intervention sequence">
    <div class="node"><strong>1 · 破坏 needles</strong><small>先制造一个缺少有效 evidence 的 receiver。</small></div>
    <div class="node"><strong>2 · 恢复完整 spans</strong><small>在早层一次性写回同一 prompt 的 clean span states。</small></div>
    <div class="node"><strong>3 · 检查 retrieval</strong><small>比较删除 count-aligned retrieval component 与删除等强度无关方向。</small></div>
    <div class="node"><strong>4 · 检查 late state</strong><small>在更晚层做同样的 aligned-vs-control 删除。</small></div>
    <div class="node"><strong>5 · 读取结果</strong><small>同时看 retrieval、late coordinate 和最终 expected count。</small></div>
  </div>

  <div class="experiment"><div class="experiment-label">Experiment 19 · setting</div><div><h4>同一批 prompts、固定三处层位、逐段阻断</h4>
    <p><strong>数据与独立单位。</strong>每个模型使用 confirmation seeds 1254–1263 × gold counts 1–10，即 <strong>100 个配对 seed–count prompts</strong>。每个 prompt 运行 11 个固定 intervention conditions，因此是 <strong>1,100 rows/model</strong>；报告合并 Qwen 原始确认实验与 Gemma Top-6 consistency rerun，共 2,200 rows。Counts 1–10 是同一 seed 下的固定重复条件，不当作 100 个彼此独立样本；正式推断应先在 seed 内对 10 个 counts 等权平均，再以 <strong>10 个 seeds</strong> 为独立单位。</p>
    <p><strong>层位。</strong>Qwen 在 L8 恢复 spans、L23 检查 retrieval、L29 检查 late state；Gemma 对应 L9、L29、L37。Layers、rank-3 拟合协议、norm matching 与方向符号均沿用原实验；Gemma 只把 retrieval bank 换成 frozen global Top-6 在 L29 的精确子集 H4/H2，且不根据本次结果重新选 head。</p>
    <div class="formula"><strong>“一个预注册层上的 rank-3 retrieval subspace”具体指什么？</strong>
      <p><strong>冻结位置与 head bank。</strong>在查看 Q19 的 confirmation prompts 之前，retrieval 位置已固定为 Qwen L23 与 Gemma L29。Qwen 使用原冻结的 7 heads（H29/H13/H28/H12/H31/H30/H10）；Gemma consistency rerun 使用跨实验统一的 global Top-6 前缀落在 L29 的两个 heads（H4/H2）。Top-6 是论文统一口径下追加的 post-hoc extension，选择规则来自既有 discovery ranking，不使用本次 Q19 outcomes。</p>
      <p><strong>rank-3。</strong>对每个模型，在该层的 answer-query 位置，把这些 frozen heads 经 output projection 后写入 residual stream 的向量相加，得到一个 <span class="math">d</span> 维 broad-bank write <span class="math">b</span>。只用 discovery seeds 1234–1253 × counts 1–10 的 200 个 clean natural forwards，先减 discovery 均值 <span class="math">μ</span>，再做 PCA，冻结方差最大的三个正交方向 <span class="math">U∈R<sup>d×3</sup></span>。rank-3 因此表示 hidden space 中的一个三维子空间，<strong>不是三个神经元、三个 heads，也不是只研究三个 count</strong>。</p>
      <p><strong>为什么简称 count-aligned？</strong>PCA 拟合本身不使用 Q19 confirmation 结果，也不把 PC1、PC2、PC3 分别指定成某个 count。这个名称只表示：更早的 held-out geometry 检查发现，counts 1–10 的 centroid 变化和高于 chance 的 count readout 确实有相当部分落在这个三维 retrieval subspace 中。因此更精确的名称是“<strong>含有 count-related variation 的 frozen rank-3 retrieval subspace</strong>”，而不是“一根每移动一单位就加一的整数轴”。</p>
      <p><strong>Q19 实际删除什么。</strong>对某个 confirmation prompt 的 broad-bank write <span class="math">b</span>，候选干预删除 <span class="math">a=(b−μ)UU<sup>⊤</sup></span>；matched control 则删除一个与 <span class="math">U</span> 正交、但实际删除范数与 <span class="math">a</span> 相同的向量。两者的答案误差差值才是 retrieval mediation。因此 Qwen 的 {f(qwen_serial['retrieval_mediation']['mean'])} counts 问的是“删掉这一个冻结子空间，比删掉同样大的无关方向平均多造成多少误差”，<strong>不是 retrieval 总贡献、不是准确率，也不是百分比</strong>。</p>
      <p class="example"><strong>简单例子。</strong>若正确答案为 8，删除等强度无关方向后 expected count=6.3（误差 1.7），删除 frozen rank-3 retrieval component 后 expected count=6.0（误差 2.0），则该 prompt 的方向特异 retrieval effect 为 <span class="math">2.0−1.7=0.3</span> count。图中的 Qwen {f(qwen_serial['retrieval_mediation']['mean'])} 先对每个 seed 的 counts 1–10 求均值，再对 10 个 seeds 等权；由于面板完全平衡，点估计与直接平均 100 个配对 prompts 数值相同，但不确定性不能按 100 个独立样本计算。</p>
    </div>
    <p><strong>matched control。</strong>每次删除 count-aligned component，都与同层、同位置、实际删除范数相同但方向与 count 无关的 orthogonal component 比较。因此效应不能简单归因于“在这一层随便删掉了一段同样大的 hidden state”。</p>
    <p><strong>推断与留存审计。</strong>两模型均保留 100-row paired-effects 文件，并按 10 个 seed means 做 10,000 次 percentile cluster bootstrap，枚举全部 <span class="math">2<sup>10</sup>=1,024</span> 个 two-sided exact seed sign flips。Qwen 的原始 1,100-row detail、7,700-row broad metrics 与 paired-effects 已从通过 persistence audit 的 Filestream 副本恢复；远端与本地 9 个 Qwen 文件逐文件 SHA-256 一致，重新生成的 paired CSV 也与原文件 byte-identical。旧 seed–count bootstrap 区间不再使用。</p>
  </div></div>

  <div class="claim"><strong>怎样才算形成有序链？</strong>我们要求三个观察同时成立：第一，恢复早层 spans 后，后续 broad retrieval 确实改变；第二，特异删除 retrieval component 后，晚层 count state 和答案修复都减弱；第三，删除更晚的 count component 会伤答案，却不能反向改变已经计算完成的早期 retrieval readout。两个模型的同一-forward诊断均满足三条，且按 seed 为独立单位的 cluster interval 与 exact test 均可复算。</div>

  <details class="collapsible-list"><summary>展开技术定义：11 个 arms 与效应公式</summary>
    <p><span class="math">C</span> 是 needle-corrupt reference，<span class="math">O</span> 是等 token-budget ordinary-span restoration，<span class="math">S</span> 是 clean full-needle-span restoration。其余 arms 是 <span class="math">S+R⊥/S+R∥</span>、<span class="math">S+T⊥/S+T∥</span>，以及四种 <span class="math">S+R<sub>a</sub>+T<sub>b</sub></span> 联合条件。每个 arm 都保存 counts 1–10 candidate logits、strict generation、broad attention mass/score、retrieval coordinate 与 late-answer coordinate。</p>
    <div class="formula"><strong>效应定义。</strong>令 <span class="math">e(X)=|E[c]<sub>X</sub>−N|</span>。Source repair 为 <span class="math">e(O)−e(S)</span>；retrieval mediation 为 <span class="math">e(S+R∥)−e(S+R⊥)</span>；late mediation 为 <span class="math">e(S+T∥)−e(S+T⊥)</span>。Joint interaction 比较存在与不存在 late block 时，retrieval block 的附加损伤是否改变；负值表示两次阻断作用有重叠，不能把两根柱直接相加。</div>
  </details>
  <figure><h4 class="figure-title">机制图 M1 · 同一 forward 中的有序 partial serial mediation</h4>{exp19_chart}<figcaption>每根柱的点估计先在同一 seed 内平均 counts 1–10，再对 10 seeds 等权；横轴单位为 counts。Source repair 问“恢复真实 needle spans 比恢复同样多的无关文本多救回多少”；retrieval 与 late mediation 分别问“删除 count-aligned component 比删除同样强度的无关方向多损失多少”。两模型均给出 seed-cluster CI 与 exact sign-flip；三种效应回答不同问题，不能把柱长直接相加。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q19 精确结果（seed-equal 均值、95% cluster CI 与 exact p，单位 counts）</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Source repair</th><th>Retrieval mediation</th><th>Late mediation</th><th>Joint interaction</th><th>Remaining repair</th></tr></thead><tbody>{exp19_rows}</tbody></table></div></details>
  <div class="conclusion-line"><strong>如何解释结果。</strong>完整 span restoration 相对 ordinary control 的 Qwen/Gemma 点估计为 <strong>{f(qwen_serial['source_repair']['mean'])}/{f(gemma_serial['source_repair']['mean'])} counts</strong>。在已经恢复 source 的条件下，特异阻断 retrieval 的点估计为 <strong>{f(qwen_serial['retrieval_mediation']['mean'])}/{f(gemma_serial['retrieval_mediation']['mean'])} count</strong>，阻断 late state 为 <strong>{f(qwen_serial['late_mediation']['mean'])}/{f(gemma_serial['late_mediation']['mean'])} counts</strong>。Retrieval 的 seed-cluster 95% CI 为 Qwen <strong>[{f(qwen_serial['retrieval_mediation']['ci95_low'])}, {f(qwen_serial['retrieval_mediation']['ci95_high'])}]</strong>、Gemma <strong>[{f(gemma_serial['retrieval_mediation']['ci95_low'])}, {f(gemma_serial['retrieval_mediation']['ci95_high'])}]</strong>，two-sided exact seed sign-flip 分别为 <strong>p={p_text(qwen_serial['retrieval_mediation']['exact_sign_flip_p_two_sided'])}</strong> 与 <strong>p={p_text(gemma_serial['retrieval_mediation']['exact_sign_flip_p_two_sided'])}</strong>；两模型的 source 与 late 指标也通过同一 seed-level 推断。因此双模型均支持较小但可靠的 retrieval 部分中介。</div>
  <div class="claim boundary"><strong>Q19 能说明什么、不能说明什么。</strong>两模型的完整 seed-level 推断共同支持“上游 span evidence 的一部分先经过 retrieval，再进入 late answer state，最后影响输出”。Retrieval 与 late 两次阻断存在重叠，联合阻断后仍剩 Qwen/Gemma <strong>{f(qwen_serial['remaining_repair']['mean'])}/{f(gemma_serial['remaining_repair']['mean'])} counts</strong> repair；因此它们<strong>不是彼此独立的可加模块，也没有穷尽所有通路</strong>。最准确的结论仍是：存在一条有序、自然使用、但带有冗余和 bypass 的<strong>部分串联中介链</strong>。</div>
</section>

<section id="representation">
  <h2>2. 通用测量框架：Representation 负责定位，Causal test 负责判定</h2>
  <p class="lead"><strong>Prompt site</strong> 定义为每个 active needle 的最后一个 token，状态记为 <span class="math">h<sup>P</sup><sub>s,n,ℓ</sub></span>；其中 <span class="math">n∈{{1,…,10}}</span> 是当前 running index。<strong>Answer site</strong> 定义为生成首个数字前 <code>Total:</code> 的 query token，状态记为 <span class="math">h<sup>A</sup><sub>s,ℓ</sub></span>，标签是 prompt 的最终 gold count。本节只建立 representation facts 与统一 readouts；真正的机制判断由后续 matched interventions 给出。</p>
  <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>给三个机制步骤提供同一套术语和计算标尺，特别是把“状态中能读出 count”与“模型自然依赖该状态”分开。<span class="mini-example"><strong>直观例子：</strong>页面页码可以完美预测书读到哪里，但删掉页码通常不影响故事内容；同理，hidden state 的 classifier 很准，也可能只是伴随信号。只有 matched removal 或 patch 改变答案，才进入因果结论。</span></p></div>
  <div class="formula"><strong>证据层级。</strong><em>Decodability</em> 问某个 held-out readout 能否从状态预测 count；<em>sufficiency</em> 问把 donor state 写入 receiver 后能否驱动 donor-directed output；<em>direction-specific necessity</em> 问删除候选方向是否比删除同层、同位置、同 realized norm 的正交方向更伤行为；<em>mediation</em> 问阻断候选中介是否会特异地削弱一个已产生的上游效应。<span class="example">例：分类准确率高只说明 count 可读；只有 donor patch 能转移答案、或候选方向 removal 比等范数 orthogonal removal 更伤，才说明该状态具备可执行性或方向特异必要性。</span></div>

  <h3>2.1 Frozen-PCA 三维流形</h3>
  <div class="figure-grid">
    <figure>
      <h4 class="figure-title">图 2a · Prompt needle-end running-index manifold</h4>
      <div class="three-d"><div class="three-d-head"><strong>Needle-end state</strong><div><label>模型<select id="prompt-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><label>层<select id="prompt-layer" aria-label="Prompt manifold layer"></select></label></div></div><canvas id="prompt-canvas" aria-label="Prompt needle-end hidden states projected onto frozen PC1 PC2 PC3. Drag to rotate.">Your browser needs Canvas support.</canvas></div>
      <figcaption>每个半透明点是一条 seed 在第 <em>n</em> 个 needle end 的 hidden state；颜色与数字表示 running index <em>n</em>。较大的编号圆点是 30 seeds 的 count centroid，连线只连接相邻 <em>n</em>。PC1/PC2/PC3 在 discovery rows 上按所选层独立冻结，数值无物理单位；因此可比较同层 geometry，不能把跨层坐标值当作同一基底。默认 Qwen L8、Gemma L9 是冻结的早期代表层，用于展示 running geometry 已经形成，而不是按最高 R² 重新选出的层；可切换任意层，拖动旋转，双击复位。</figcaption>
    </figure>
    <figure>
      <h4 class="figure-title">图 2b · Answer-query consolidated-count manifold</h4>
      <div class="three-d"><div class="three-d-head"><strong>Answer-query state</strong><div><label>模型<select id="answer-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><label>层<select id="answer-layer" aria-label="Answer manifold layer"></select></label></div></div><canvas id="answer-canvas" aria-label="Answer query hidden states projected onto frozen PC1 PC2 PC3. Drag to rotate.">Your browser needs Canvas support.</canvas></div>
      <figcaption>每个点是一条 seed 的 answer-query state；颜色与编号表示最终 gold count。PC1/PC2/PC3 同样按所选层冻结；图展示 3D 投影 geometry，而 exact-count classifier 使用 PCA-32。默认 Qwen L28、Gemma L37 是在 V4.4.5 reused-state analysis 中按 confirmation 三维 nearest-centroid accuracy 选出的 display-only 层；这种 post-hoc 选择只服务可视化，不进入因果或跨层推断。层选择器可检查 formation、aggregation 与 consolidation 的完整跨层变化。</figcaption>
    </figure>
  </div>

  <div class="claim"><strong>为什么默认展示浅层 prompt 与深层 answer？</strong><p>两张图对应不同计算阶段，而不是 layer-matched representation comparison。Prompt needle end 在阅读中刚完成局部 phrase，prefix-local running index 因而可在浅层形成；answer query 位于全文之后，必须经过远距离 retrieval、aggregation 与 residual write，行为上可执行的 final-count state 才在中后层出现。</p><p>数据支持这种时序，但不支持“prompt representation 只在浅层、answer representation 只在深层”的排他说法。Prompt ridge 的实际最高点是 Qwen L13（R²=0.960，MAD=0.428）和 Gemma L11（R²=0.857，MAD=0.859）；answer classifier 也会逐层上升。交互图的 answer 默认层改为 reused-state 3D 最易分辨的 Qwen L28 / Gemma L37，而 causal timing 仍由独立的 dense full-state patching、rank-3 removal 与 retrieval-subspace experiments 判断。</p><p><strong>解释边界：</strong>直接比较 L8/L9 与 L28/L37 同时改变 token site 与 layer depth，只能示意“formation → retrieval → consolidation”，不能单独证明差异来自位置而非深度；层选择器与下方 layerwise curves 用于审计这一点。<strong>目前结论</strong>是 shallow/deep timing 与机制顺序一致，但其架构原因仍是解释而非专门干预的结果。</p></div>

  <details class="paper-appendix"><summary>Appendix diagnostics · classifier、counter-property、逐层低维与位置控制</summary>
  <h3>2.2 Counter-manifold 的紧致程度：classifier 只是统一的描述性标尺</h3>
  <p class="lead"><strong>本小节只有一个目的：</strong>用完全相同的 seed-held-out protocol，量化不同 count 对应的 hidden-state clouds 是否紧致、是否容易彼此区分，从而给后续 non-thinking 与 native thinking 提供可直接比较的 representation baseline。Classifier 不参与 mechanism discovery，不用于证明 causal use，也不意味着模型内部存在一个离散整数寄存器。</p>
  <div class="formula"><strong>Prompt continuous probe。</strong>每个 seed fold 只用训练 rows 拟合 PCA-32 与 ridge regression，再在 held-out seeds 上预测 running index <span class="math">n</span>。<span class="math">R²=1−Σ_i(n_i−n̂_i)²/Σ_i(n_i−n̄)²</span>；ridge MAD=<span class="math">M⁻¹Σ_i|n̂_i−n_i|</span> counts。Prompt needle ends 不做 exact-count classifier：每个 prompt 内是连续 occurrence positions，而不是十个独立的最终计数样本；这里要检验的是 running-index curve 的连续可解码性。<span class="example">例：held-out predictions 解释了 gold running index 方差的 90%，则 R²=0.90；若三例 |预测−gold| 为 [0.2,0.5,1.1]，ridge MAD=0.6 count。</span></div>
  <div class="formula"><strong>Answer exact-count classifier：如何读。</strong>按 seed 做五折 GroupKFold；每折只在训练 seeds 上拟合 StandardScaler 与 PCA-32，再在 held-out seeds 上预测 count。<strong>Nearest centroid</strong> 直接问测试 state 是否靠近正确 count 的训练 centroid，因此是本文最接近“manifold 紧致度”的 operational readout；<strong>L2 logistic</strong> 只作为 learned linear-boundary robustness check。Accuracy 越高表示 count clouds 越容易区分；classifier MAD=<span class="math">M⁻¹Σ_i|n̂_i−n_i|</span> 越低表示即使分错也更接近正确 count。严格说 classifier 同时受 within-count dispersion 与 between-count separation 影响，所以这里称为<strong>紧致度/可分性诊断</strong>，而不是纯粹的几何半径。<span class="example">例：四个 gold counts [2,4,7,9] 被预测为 [2,5,7,6]，accuracy=2/4=50%，MAD=(0+1+0+3)/4=1 count；若 native thinking 在完全相同 split 下达到 80%/0.3，就说明它的 count-conditioned manifold 更紧致且更可分。</span></div>
  <details class="collapsible-list"><summary>展开：辅助 manifold quantities 的定义</summary>
    <div class="formula"><strong>Manifold quantities。</strong>令中心化 state matrix 为 <span class="math">X</span>。stable rank=<span class="math">‖X‖²_F/‖X‖²_2</span>；all-state rank-3 capture=<span class="math">Σ_{{k=1}}^3σ_k²/Σ_kσ_k²</span>，centroid rank-3 capture 对十个 count centroids 做同一计算；<span class="math">η²_{{count}}=Σ_nN_n‖μ_n−μ‖²/Σ_i‖h_i−μ‖²</span>；silhouette=<span class="math">(b-a)/max(a,b)</span>，其中 <em>a</em> 是同 count 平均 cosine distance，<em>b</em> 是最近异 count cluster distance。<span class="example">例：奇异值 [2,1] 时 stable rank=(4+1)/4=1.25；rank-3 capture=0.80 表示前三轴承载 80% 方差；η²=0.60 表示 60% 总方差来自 count centroids 之间；若 a=1、b=2，则 silhouette=0.5。</span></div>
    <div class="formula"><strong>为什么还保留 PCA quantities。</strong>Classifier 回答“count clouds 是否容易区分”；centroid rank-3 capture 回答“十个平均 state 是否沿低维轨迹排列”；all-state rank-3 capture 与 silhouette 则补充单样本 cloud 是否真的紧致。高 centroid rank-3 不保证高 classifier accuracy，也不保证形成干净 clusters。</div>
  </details>

  <h4>Prompt running-index geometry（manifold quantities；不含 exact classifier）</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Site / layer</th><th>Ridge R²</th><th>Ridge MAD</th><th>PCA rank-3: all states</th><th>PCA rank-3: centroids</th><th>Stable rank</th><th>η² count</th><th>Cosine silhouette</th></tr></thead><tbody>{prompt_geometry_table_rows}</tbody></table></div>
  <h4>Answer-query final-count geometry（exact classifiers + manifold quantities）</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Site / layer</th><th>L2-logistic acc.</th><th>L2-logistic MAD</th><th>Nearest-centroid acc.</th><th>Nearest-centroid MAD</th><th>Ridge R²</th><th>Ridge MAD</th><th>PCA rank-3: all states</th><th>PCA rank-3: centroids</th><th>Stable rank</th><th>η² count</th><th>Cosine silhouette</th></tr></thead><tbody>{answer_geometry_table_rows}</tbody></table></div>
  <figure><h4 class="figure-title">图 2c · Answer-query classifier accuracy 随 gold count 的变化</h4>{classifier_count_chart}<figcaption>横轴是 gold count N=1,…,10；纵轴是预注册代表层 Qwen L29 / Gemma L37 的 L2-logistic grouped-OOF exact-count accuracy。每个点恰有 20 个 held-out seed predictions；竖线为该 20-trial proportion 的 pointwise 95% Wilson interval，虚线 0.10 是十类 chance。Classifier 在每个 fold 内只用训练 seeds 拟合 scaler、PCA-32 与 logistic weights，held-out seed 不参与拟合。该图按 gold count 拆开整体 56%/53% accuracy，仍然只衡量 hidden-state clouds 的可分性，不是模型最终行为 accuracy。</figcaption></figure>
  <details class="collapsible-list"><summary>展开：每个 count 的命中数与 classifier MAD</summary><div class="table-wrap"><table><thead><tr><th>Gold count</th><th>Qwen correct / 20</th><th>Qwen MAD</th><th>Gemma correct / 20</th><th>Gemma MAD</th></tr></thead><tbody>{classifier_count_table_rows}</tbody></table></div><p><strong>结果与分析。</strong>两模型在 N=1–3 最容易区分：Qwen 平均 91.7%，Gemma 90.0%；N=4–9 分别降至 36.7%/31.7%。但曲线<strong>不是单调下降</strong>：N=10 回升到 Qwen 60% 与 Gemma 70%。这类边界反弹可能来自最高类别没有更大的邻近 count、以及训练决策边界的端点效应；当前数据没有把这两个解释单独拆开。每个 count 只有 20 个 held-out predictions，因此应同时看 Wilson interval，而不能把相邻 5–10 percentage points 的差异当成稳定层级。</p><p><strong>目前结论。</strong>“N 越大越难”作为粗略趋势在中间与高 count 上成立，但严格的单调命题不成立。更准确的说法是：non-thinking answer manifold 对小 counts 最可分，中高 counts 的 clouds 重叠明显增大，而最高边界 N=10 出现可复现的分类回升。后续与 native thinking 比较时应逐 count 复用同一 split，而不只比较一个 overall accuracy。</p></details>
  <h4>V4.4.5 reused-state display-only 三维层</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Display layer</th><th>Confirmation 3-PC nearest-centroid accuracy</th><th>Integer MAD</th><th>Discovery rank-3 all-state capture</th></tr></thead><tbody>{answer_reuse_rows}</tbody></table></div>
  <p>这张补充表复用 canonical dense run 的每模型 300 个 clean natural forwards（共 23,400 个 layer-state rows）；basis 只在 discovery seeds 1234–1253 拟合，数值在 confirmation seeds 1254–1263 计算。层是按 confirmation 三维 nearest-centroid accuracy 最大化后选出的，所以 61%/63% <strong>只用于选择最好看的三维展示层</strong>。后续与 native thinking 做正式比较时，应预先冻结相同的 layer-selection 与 held-out-seed protocol，而不使用这两个 post-hoc display 数值。</p>
  <div class="claim"><strong>本小节的唯一结论。</strong>Non-thinking answer-query manifold 的 exact-count accuracy 约为 53%–56%（chance 10%），nearest-centroid MAD 为 0.615–0.640 count：final count 明显可读，但 count-conditioned clouds 还不是非常紧致。Centroid rank-3 capture 很高而 silhouette 接近 0，进一步说明“平均 count 轨迹低维”不等于“每个 count 形成干净、分离的 cluster”。这些数值只建立 non-thinking 的 representation baseline，留给后续用同一 classifier protocol 与 native thinking 比较；它们不参与 retrieval、patching 或 OV-write 的机制结论。</div>

  <figure>
    <h4 class="figure-title">图 3 · Counter manifold 的可读性与低维结构如何跨层变化（descriptive baseline）</h4>
    {''.join(representation_charts)}
    <figcaption>每个模型有两排 panel，横轴均为 zero-based transformer layer。第一排是统一的 held-out compactness/separability diagnostic：左图为 prompt running-index ridge R²；右图为 answer-query L2-logistic accuracy（蓝色实线）与 nearest-centroid accuracy（绿色虚线），水平虚线为十分类 chance=0.10。第二排是辅助的无监督 geometry：实线为全部单样本 cloud 的 PCA rank-3 capture，虚线为十个 count centroids 的 rank-3 capture。未来与 native thinking 比较时应复用完全相同的 split、PCA、readout 与 layer-selection rule；这些曲线不用于确定 causal mechanism。当前 non-thinking 的代表层：{'；'.join(best_rows)}。</figcaption>
  </figure>

  <h3>2.3 Running-index manifold 是否具有 counter 性质？</h3>
  <p class="lead">仅有高 ridge <span class="math">R²</span> 只能说明 running index 可读，不能说明十个状态构成稳定的逐步累加器。这里在原报告已经冻结的 PC1–PC3 中，追加检验 centroid 的有序距离、相邻更新方向、跨 discovery/confirmation 的稳定性，并单独做绝对位置对照。该分析不重新拟合 PCA，也不是因果干预。</p>
  <div class="formula"><strong>Ordered 与 additive counter quantities。</strong>令 discovery seeds 上第 <span class="math">n</span> 个 endpoint 的三维 centroid 为 <span class="math">μ_n</span>。<strong>Line R²</strong> 将 <span class="math">μ_n</span> 拟合为 <span class="math">a+n,v</span> 后计算 explained variance；<strong>distance-gap ρ</strong> 是 45 对 centroids 的欧氏距离 <span class="math">‖μ_i−μ_j‖</span> 与 count gap <span class="math">|i−j|</span> 的 Spearman correlation。令相邻 step <span class="math">Δ_n=μ_{{n+1}}−μ_n</span>，报告所有 step-pair cosine 的平均、step length 的 coefficient of variation <span class="math">SD(‖Δ_n‖)/mean(‖Δ_n‖)</span>，以及 discovery step 与同一 confirmation step 的 cosine。理想等步长直线 counter 会同时具有 line R²=1、distance-gap ρ=1、step cosine=1、length CV=0。<span class="example">例：centroids [0,1,2,3] 在一维中等距排列，则相邻 steps 都为 +1，四个指标分别为 1、1、1、0；若 steps 变为 [+3,−1,+3]，count 仍可能被某个 regression 读取，但 step cosine 与 distance ordering 会明显下降。</span></div>
  <div class="formula"><strong>Position control。</strong>all-token confirmation capture 中，每个 endpoint 有绝对 token position <span class="math">p</span>。我们以 seed 为 group 做 5-fold GroupKFold：每折只在训练 seeds 上拟合 cubic position baseline <span class="math">n∼1+p+p²+p³</span>；同时把 PC1–PC3 对同一 cubic design 的训练折拟合值减掉，再用 residual PC1–PC3 ridge-predict held-out running index。表中的 residual <span class="math">R²</span> 因而问“冻结前三维中还有多少不能由平滑绝对位置解释的 count signal”。<span class="example">例：若原始三维 probe R²=0.80，而每个 PC 都几乎是 position 的平滑函数，residual probe 可降到 0；这不会证明完整 hidden state 没有 counter，只说明当前前三维没有 position-independent evidence。</span></div>
  <figure><h4 class="figure-title">图 3b · Frozen PC1–PC3 的 counter-property 跨层审计</h4><img class="report-image" src="{counter_plot_uri}" alt="Prompt needle-end counter properties across all layers for Qwen and Gemma"><figcaption>四个 panel 的横轴均为 zero-based transformer layer；绿色为 Qwen、紫色为 Gemma，大圆点标出报告代表层 Qwen L8 / Gemma L9。左上纵轴是十个 discovery centroids 的直线拟合 R²；右上是 centroid distance 与 count gap 的 Spearman ρ；左下是九个相邻 centroid steps 两两 cosine 的平均；右下是同一 step 在 discovery 与 confirmation centroids 之间的 cosine 平均。四者越接近 1，越接近有序、同向、跨 seed split 稳定的增量轨迹；它们不衡量行为因果效应。</figcaption></figure>
  <h4>代表层 counter-property 与 position robustness</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Line R²</th><th>Distance-gap ρ</th><th>Mean step cosine</th><th>Step-length CV</th><th>Split step cosine</th><th>Frozen 3-PC ridge R² / MAD</th><th>Per-seed projection ρ / positive steps</th><th>Position-count ρ</th><th>Position-residual 3-PC R² / MAD</th></tr></thead><tbody>{counter_table_rows}</tbody></table></div>
  <p>Qwen L8 的 line R²=0.633、distance-gap ρ=0.648、mean step cosine=0.761、cross-split step cosine=0.817，说明冻结三维中确有中等偏强的有序、同向 trajectory；Gemma L9 对应为 0.403、0.488、0.084、0.412，明显更弯曲且 step directions 不稳定。两模型 step-length CV 分别为 1.438 与 1.729，也远离理想等步长 counter。更重要的是 position-count ρ 均约 0.965，cubic position baseline 的 grouped R² 约 0.91；去除 cubic position 后三维 R² 只剩 Qwen 0.043、Gemma 0.004。<strong>目前结论：</strong>Qwen 的 endpoint 3D 轨迹具有 counter-like ordering，Gemma 较弱；但两者都不能与 ordinal/absolute-position code 区分，因此不能称为已识别的抽象 counter。</p>
  <div class="formula"><strong>Endpoint-gated formula 的 incremental R²。</strong>对每类 token 先以 category mean 为 baseline，再把在 endpoint 学到的 frozen curve <span class="math">γ(n)</span> 按不同 gate 应用于 endpoint、needle interior 或所有 prefix-count tokens。定义 <span class="math">ΔR²_{{inc}}=(SSE_{{baseline}}−SSE_{{curve}})/SSE_{{baseline}}</span>；正值表示 curve 比 category mean 多解释方差，负值表示强行套用后反而更差。<span class="example">例：baseline SSE=100、curve SSE=40，则 ΔR²=0.60；若 curve SSE=110，则 ΔR²=−0.10。</span></div>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Endpoint-gated ΔR² on endpoints</th><th>Span-gated ΔR² on interiors</th><th>Ungated ΔR² on ordinary tokens</th><th>Ungated ΔR² on hard negatives</th></tr></thead><tbody>{gated_table_rows_html}</tbody></table></div>
  <div class="claim boundary"><strong>Counter-property 结论。</strong>在 representational level，endpoint centroids 确实按 running index 排列，尤其 Qwen 在浅层具有同向、跨 split 可复现的三维 step；但它不是等步长，Gemma 的 step alignment 很弱，而且 position residual 几乎不再可解码。all-token 公式进一步支持“curve 只在 needle endpoint gate 上出现”，证伪“needle span 内每个 token 共享同一 γ(n)”的强版本。full-hidden-space 的位置去混淆尚未完成；跨 final-N prefix invariance 未运行且已按当前论文的主张范围关闭。因此报告只写 <em>position-confounded counter-like record</em>，不升级为独立、跨任务不变的 abstract counter。</div>

  <h3>2.4 原始行为是 representation 的外部标尺</h3>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Rows</th><th>Exact accuracy</th><th>Behavior MAD</th><th>Signed error</th><th>Wrong-only MAD</th></tr></thead><tbody>{behavior_rows}</tbody></table></div>
  <p>本表对应 Frozen-PCA payload 使用的 20 seeds × 10 counts（N=1,…,10）geometry cohort；开头热图对应 30 seeds × 11 counts（N=0,…,10）的完整 baseline grid。两者用途与分母不同，不混合汇总。geometry cohort 上 Qwen 与 Gemma 都主要低估：signed error 分别为 −0.815 与 −1.320。因而后续干预既看 exact accuracy，也看 absolute count shift / error；只看命中率会丢掉“向正确 count 移动了多少”的信息。<strong>目前结论：</strong>representation 图回答“信息是否存在以及以何种 geometry 存在”，不能替代后续因果实验；Qwen 较强、Gemma 较弱的 prompt ordering 与两者行为差异一致，但尚未被证明是 performance gap 的因果来源。</p>
  </details>
</section>

<section id="formation">
  <h2>Step 1 · Form — 浅层形成分布式 running-index evidence</h2>
  <p class="lead">这一阶段区分两个经常被混淆的问题：needle evidence 是否进入网络，以及 endpoint probe 解出的三维曲线是否就是模型自然使用的存储格式。前者有强因果证据；后者仅是描述性 readout。Qwen 的 running-index ordering 明显强于 Gemma，这与 Qwen 更好的计数行为一致，但不是单独的因果解释。</p>
  <div class="claim"><strong>Stage-I hypothesis。</strong>浅层首先形成与 needle occurrences 相关的分布式 span-level evidence；needle endpoint 可以作为观察该过程的 readout site，但不是已经确认的独立 counter register。本文把“完整 active needle span”作为上游 causal source unit，不进一步要求区分 endpoint 是检索地址、内容载体还是与 interior 协同。</div>
  <div class="chain-blueprint" aria-label="Chain A mechanism representation causal test">
    <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>确定模型读过 needle 后，后续 retrieval 真正可使用的上游信息单元是 endpoint、完整 span，还是仅仅普通 passage 被扰动的副作用。<span class="mini-example"><strong>直观例子：</strong>一条 record 像一张写有城市和分数的卡片；endpoint 像卡片右下角。若只还原右下角救不回计数、还原整张卡片却能救回，就应把整张卡片视为 causal evidence unit。</span></p></div>
    <div class="evidence-triad">
      <div class="triad-step"><span class="protocol-no">01 · Mechanism</span><h3>完整 span 保存可复用 evidence</h3><p>预测：早层 full-span states 应能在 corrupt run 中恢复答案；单个 endpoint 不必局部充分。</p></div>
      <div class="triad-step"><span class="protocol-no">02 · Representation</span><h3>Endpoint 是 noisy readout</h3><p>预测：running index 可由 endpoint 解码，Qwen ordering 更强；但 geometry 会受绝对位置和上下文影响。</p></div>
      <div class="triad-step"><span class="protocol-no">03 · Causal test</span><h3>Span 与 endpoint 做 matched 对照</h3><p>用 token-budget-matched corruption、equal-norm rank-3 removal 与逐层 full-span restoration 判定真正被使用的 source unit。</p></div>
    </div>
  </div>

  <details class="paper-appendix"><summary>Appendix A1 · 设计注册表、earlier-span attention 与完整样本范围</summary>
  <h3>3.1 设计注册表：每个候选操作改变什么、控制什么</h3>
  <div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Candidate intervention / measurement</th><th>Control 的具体构造</th><th>保持匹配的量</th><th>唯一希望改变的量</th></tr></thead><tbody>
    <tr><td>1A · earlier-span attention</td><td>在 clean forward 中，从第 <em>n</em> 个 needle endpoint query 计算指向前 <em>n</em>−1 个完整 active-needle spans 的 attention mass。</td><td>对每个 needle span 构造一个同 token 长度的 ordinary non-needle segment：从该 span 之前 8–511 tokens 中选择最近可用区间；排除全部 slot/hard-negative spans，且各 control segments 互不重叠。只比较前 <em>n</em>−1 对 spans。Heads 由 discovery seeds 的 earlier-needle mass 冻结排序；control 与 confirmation 估计不重新选 head。</td><td>同一模型、seed、prompt、query、head、layer、span 数量、每个 span 的 token 长度与近似相对深度。</td><td>被 query 指向的是 active needle evidence，还是附近的 ordinary passage tokens。它不是 random-head control。</td></tr>
    <tr><td>1A-state · layerwise state response</td><td>分别执行 clean 与 active-needle-corrupt forward；在每个 post-block layer 读取全部 registered needle-token positions 的完整 residual vectors，并计算每个 hidden coordinate 的 RMS change。</td><td>第三个 pure forward 只替换 ordinary passage targets。Targets 与 active needles 逐段一一匹配 token 长度，段间不重叠，总 token budget 完全相同；在这些 ordinary positions 计算同一 full-vector RMS change。三臂均不 patch hidden state。</td><td>同一 seed/count prompt、替换段数、逐段 token-length vector、总 token budget、sequence length、answer-query token/position、forward 与 capture 时点。</td><td>输入替换发生在 active needle spans，还是等规模 ordinary passage；主统计量是同层 needle RMS deformation 减 ordinary RMS deformation。</td></tr>
    <tr><td>1A-retention · held-out centroid assay</td><td>在 final N=10 prompt 中读取第 1–10 条 span 的 endpoint 与 span-token mean；用 20 个 discovery clean seeds 冻结每层、每个 running index 的 centroid/radius，再在 10 个 confirmation seeds 比较三臂到正确 centroid 的标准化距离。</td><td>Needle-corrupt 与 ordinary-corrupt 沿用同 prompt、逐段长度向量与总 token-budget matching；centroids、窗口、十项指标、优先级与 multiplicity family 都在 confirmation outcome 前冻结。</td><td>同一 final-count context、sequence/query、layer、running index、pooling rule、corruption budget；discovery 与 confirmation seeds 不重叠。</td><td>普通 state 位移是否特异地破坏“第 k 条 occurrence”对应的 clean count geometry，而非只造成任意方向的大变化。</td></tr>
    <tr><td>1B · token corruption</td><td>把每条完整 active-needle span 从第一个到最后一个 token，逐 span 替换成同一 prompt 中抽取的等长 ordinary token sequence；随后从头 greedy generation。</td><td>第三个 paired forward 在 ordinary passage 中分配与各 needle span 一一等长的 target segments，再用另一组、不重叠的 ordinary sequences 替换。所有 source/target segments 均避开 slot 与 hard-negative spans，并限制在 slot 区域前后 64 tokens 的 passage window 内。另有完全不改输入的 clean forward。</td><td>同一 seed/count prompt、总替换 token budget、每段长度、总序列长度、后续绝对位置、answer query 位置与 generation/parser。匹配的是预算，不强求替换前后碰巧相同的 token IDs 数也完全相等。</td><td>机械上同规模的文本替换发生在 active-needle evidence，还是 ordinary passage。</td></tr>
    <tr><td>1C · prompt rank-3 removal</td><td>在每个测试层，同时从同一 prompt 的全部 active needle-end states 删除 discovery-fitted count-centroid rank-3 basis 上的 within-prompt centered projection。</td><td>先从 discovery rows 减去各自 count centroid，再删除 count-basis 分量；对剩余 within-count residuals 做 PCA，取前三轴并再次正交化到 count basis。测试时从这个数据驱动的 orthogonal rank-3 basis 删除分量，并按每个 prompt 缩放到与 candidate 实际删除量相同的 Frobenius norm。另有 clean forward 作为共同基线。</td><td>同一 prompt、layer、全部 endpoint positions、rank=3、hook 时点、实际删除 Frobenius norm、后续 forward 与 greedy generation。</td><td>删除方向是否与 count-centroid geometry 对齐。Control 不是随机方向，而是高方差的 within-count nuisance subspace。</td></tr>
  </tbody></table></div>
  <p class="lead">样本范围：1A 使用 confirmation seeds 1254–1263、N=10 prompts、occurrences 2/4/6/8/10；1A-retention 使用 discovery seeds 1234–1253 的 20 个 clean N=10 prompts 冻结 centroid/radius，再在 confirmation seeds 1254–1263 的 10 个三臂 N=10 prompts 上检验 running indices 1–10；1B 使用同十个 confirmation seeds、counts 1–10（每模型 100 个 paired prompts）；1C 使用同十个 seeds、counts 2–10，并按报告列出的 Qwen 10 层与 Gemma 13 层逐层配对。1A 在同一 seed/occurrence/head 内做 needle-minus-ordinary 后再平均 occurrences；1A-retention 先在 seed 内平均十个 running indices 与独立冻结窗口；1B/1C 在同一 seed/count prompt 内做 candidate-minus-control 后再跨 counts 与 seeds 汇总，避免把不同 prompts 的自然难度当作干预效应。</p>

  <div class="step-heading"><span class="step-kicker">02 · Representation</span><h3>3.2 Endpoint 上能观察到什么？</h3></div>
  <p>图 2a、图 3 与图 3b 已给出 endpoint running-index geometry：Qwen 的有序、同向 centroid trajectory 强于 Gemma，但两者前三维都与 absolute position 高度混淆。下面的 attention readout 再问，形成当前 occurrence state 时是否会指向更早的 active evidence。</p>
  <div class="formula"><strong>Earlier-span preference。</strong>在当前 needle endpoint query 上，将全部较早 active-needle spans 的 attention mass 记为 <span class="math">A_{{needle}}</span>，将长度和相对位置匹配的 ordinary spans mass 记为 <span class="math">A_{{control}}</span>；preference=<span class="math">A_{{needle}}−A_{{control}}</span>，再先在 seed 内平均 occurrence rows、后跨 seeds 平均。<span class="example">例：某 head 对 earlier needles 的总 mass=0.80，对 matched ordinary spans 的 mass=0.10，则 preference=0.70。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1A · representation</div><div><h4>Earlier-span attention：endpoint 在形成时会回看此前 evidence</h4><p><strong>目的。</strong>检查当前 needle endpoint 的候选 heads 是否优先读取此前 active needles。<strong>设置。</strong>在同一 query/head/layer 内，用长度和相对位置匹配的 ordinary spans 作 control；heads 只在 discovery seeds 冻结。<strong>结果。</strong>confirmation 中最强 Qwen head 为 L{earlier_qwen['layer']}H{earlier_qwen['head']}，preference={f(earlier_qwen['confirmation_preference_mean'])}（区间 {ci(earlier_qwen,'ci95_low','ci95_high')}）；Gemma 为 L{earlier_gemma['layer']}H{earlier_gemma['head']}，preference={f(earlier_gemma['confirmation_preference_mean'])}（{ci(earlier_gemma,'ci95_low','ci95_high')}）。<strong>分析与目前结论。</strong>这支持“当前 occurrence state 的形成会参考此前 needles”，但 attention preference 仍是 routing representation，不能证明该单头保存完整整数或对输出必要。</p></div></div>
  </details>

  <div class="step-heading"><span class="step-kicker">03 · Causal test</span><h3 id="hidden-state-causal-status">3.3 Needle → span hidden state → count：两个箭头分别如何因果检验？</h3></div>
  <p class="lead">这里把 hidden state 明确放在逻辑链中间，并把两个箭头分开做 intervention。<strong>箭头 A（needle input→span state）</strong>现在由新增的 clean/needle-corrupt/ordinary-corrupt 三-forward 实验逐层量化；它问“改掉输入中的 needles 后，完整 span residual tensor 改变多少，以及这种改变是否超过等 token-budget 的普通文本替换”。<strong>箭头 B（span state→count）</strong>则固定 corrupt 输入，只在某层把同一 prompt 的 clean span states 写回；图 4b 问这次内部 state patch 能否救回 count。两条箭头的 estimand 不同：state 仍然不同，不代表该层 state 仍能被后续答案计算使用。</p>
  <div class="hidden-state-chain" aria-label="Two-step causal chain from needle input through span hidden states to predicted count">
    <div class="hidden-state-node"><strong>输入：needle spans</strong><small>长文本中真正应计数的 records</small></div>
    <div class="hidden-state-arrow"><strong>箭头 A · input → state</strong><small>把 active needles 等长替换为 ordinary text；其余 prompt、长度与 query 位置不变。图 3c 逐层比较完整向量 RMS deformation 与 ordinary-region matched control。</small></div>
    <div class="hidden-state-node"><strong>中间变量：H<sup>(ℓ)</sup><sub>span</sub></strong><small>该层所有 active-span token 的完整 residual vectors；不是单个 endpoint，也不是仅 PC1–PC3。</small></div>
    <div class="hidden-state-arrow"><strong>箭头 B · state → count</strong><small>固定 corrupt 输入，只在 Lℓ 把 H<sup>clean</sup><sub>span</sub> 写回一次；后续自由运行，并与等 token-budget ordinary-state patch 比较。</small></div>
    <div class="hidden-state-node"><strong>结果：predicted count</strong><small>十个数字候选的 expected count 与严格生成答案</small></div>
  </div>
  <div class="table-wrap"><table><thead><tr><th>逻辑箭头</th><th>目前证据</th><th>因果状态</th><th>严谨结论</th></tr></thead><tbody>
    <tr><td>Needle content → 完整 span hidden states</td><td>第一组用 100 confirmation prompts/model 做逐层 full-tensor deformation；第二组在独立 clean seeds 上冻结 running-index centroids，再用 10 个 held-out N=10 prompts 比较 clean、needle-corrupt 与 token-budget-matched ordinary-corrupt 的 endpoint/span-mean states</td><td><strong>Input→state 的干预响应与 count-specific state organization 均已直接量化</strong></td><td>Raw full-tensor RMS 的强全窗口 needle-minus-ordinary 版本并未跨模型成立；但方向更明确的 held-out centroid assay 在两模型均显示，删掉 needles 会把 occurrence states 显著推离其正确 running-index cloud，而等预算 ordinary corruption 几乎不动它们。因而支持的是 count-specific state organization，不是“任意 state 距离都更大”。</td></tr>
    <tr><td>完整 needle-span hidden states → prediction</td><td>固定 corrupt 输入；逐层 full-span clean-state restoration；endpoint 与 token-budget-matched ordinary restoration controls</td><td><strong>Span-state → count 已直接因果验证</strong></td><td>仅改变内部 H<sup>(ℓ)</sup><sub>span</sub> 就能在早层显著救回答案；因此完整多-token span state 是可用的 causal source。尚未定位其中哪一个 token、方向或非线性 feature 承担作用。</td></tr>
    <tr><td>Span hidden states → downstream retrieval</td><td>同一次 restored forward 中重新读取 frozen broad bank 的 active-needle mass 与多-span coverage</td><td><strong>State patch 的下游因果响应已验证</strong></td><td>早层 clean span-state patch 会重新配置后续 answer-query retrieval；主要行为耦合窗口约止于 Qwen L20、Gemma L16。</td></tr>
    <tr><td>Needles → endpoint PC1–PC3 curve → prediction</td><td>Endpoint geometry 与 earlier-span attention；随后对全部 endpoints 做 count rank-3 vs actual-norm-matched orthogonal removal</td><td><strong>曲线可读；作为局部 mediator 的效应很弱/近零</strong></td><td>PC/ridge 图能描述 running order，但现有结果反对“该单点线性三维曲线就是必要 counter register”的强版本。</td></tr>
  </tbody></table></div>
  <details class="paper-appendix"><summary>Appendix A2 · Raw full-vector deformation：状态改了多少，但方向不稳定</summary>
  <div class="experiment"><div class="experiment-label">Experiment 1A-state · causal response</div><div><h4>逐层 full-vector state deformation：输入被改掉后，span state 如何变化？</h4><p><strong>目的。</strong>直接补齐 Arrow A，而不依赖 endpoint PCA 或事后选方向。<strong>数据。</strong>每模型使用 confirmation seeds 1254–1263 × gold counts 1–10=100 prompts；每个 prompt 运行 3 次纯 forward，共 300 forwards/model。Qwen 记录 L0–35、Gemma 记录 L0–41，因此得到 3,600/4,200=7,800 个 seed–count–layer detail rows。<strong>操作。</strong>Needle arm 将全部 active needle spans 替换成 ordinary donor tokens；control arm 在同一 prompt 中选择不重叠 ordinary target spans，严格匹配 needle spans 的逐段长度向量与总 token budget，再用 ordinary donor tokens 替换。两臂都保持 sequence length、answer query position 与其余 prompt 不变。Clean forward 在两组 registered target positions 保存每个 block 后的完整 residual vectors；两个 corrupt forwards 只在各自 target positions 取 state。<strong>推断。</strong>每层先在同一 seed 内平均 counts 1–10，再以十个 seeds 等权；whisker 是 50,000 次 seed bootstrap，实心点只表示预注册 two-sided exact sign-flip nominal p&lt;0.05。没有根据本实验结果选择层或 subspace。</p></div></div>
  <div class="formula"><strong>State-deformation 与 matched specificity。</strong>令 <span class="math">S_N</span> 为全部 registered needle-token positions，<span class="math">S_O</span> 为长度向量完全相同的 ordinary targets，hidden width 为 <span class="math">d</span>：<br><span class="math">D_N(ℓ)=||H_{{S_N}}^{{clean,ℓ}}−H_{{S_N}}^{{needle-corrupt,ℓ}}||_F/√(|S_N|d)</span>，<br><span class="math">D_O(ℓ)=||H_{{S_O}}^{{clean,ℓ}}−H_{{S_O}}^{{ordinary-corrupt,ℓ}}||_F/√(|S_O|d)</span>，<br><span class="math">S_{{state}}(ℓ)=D_N(ℓ)−D_O(ℓ)</span>。<span class="math">D</span> 是每个 hidden coordinate 的 RMS change，单位是该模型该层 residual 数值；<span class="math">S_{{state}}&gt;0</span> 才表示 needle replacement 的 state deformation 比等预算 ordinary replacement 更大。Secondary relative-RMS 再各自除以 clean-state RMS；cosine specificity 则比较平均 token-wise cosine distance。<span class="example">例：L8 的 needle/ordinary RMS changes 为 0.42/0.40，则 state 确实因输入操作而改变，但 specificity 只有 0.02；不能把 0.42 单独解释成 needle-specific mechanism。若为 0.42/0.10，specificity 才是 0.32。</span></div>
  <span id="state-effect-evidence"></span><figure><h4 class="figure-title">图 3c · Active-needle input intervention 引起的逐层 full-span state effect</h4><div class="figure-stack"><div><h4>Qwen3-8B · raw needle/control deformation</h4>{state_effect_raw_charts['Qwen3-8B']}</div><div><h4>Qwen3-8B · raw needle-minus-ordinary specificity</h4>{state_effect_specificity_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B · raw needle/control deformation</h4>{state_effect_raw_charts['Gemma4-E4B']}</div><div><h4>Gemma4-E4B · raw needle-minus-ordinary specificity</h4>{state_effect_specificity_charts['Gemma4-E4B']}</div></div><figcaption>每个模型的第一张图显示两条原始 full-vector RMS deformation：实色为替换 active needles 后在原 needle positions 的 state change，灰色虚线为等 span-length/总 token-budget ordinary replacement 在 ordinary targets 的 state change。第二张图显示预注册 primary difference <span class="math">S_{{state}}(ℓ)=D_N−D_O</span>；横轴是 zero-based post-block layer，纵轴是 residual RMS units，不是 counts。每层先在 seed 内平均十个 counts，再跨十个 seeds；whisker 为 seed-bootstrap 95% interval，实心/空心点分别表示 exact sign-flip nominal p&lt;0.05/未达到。两种推断摘要在 seed effects 偏斜时可能不同，注册判据以点形的 exact sign-flip 为准。竖虚线来自独立 dense-restoration 实验预先冻结的 behavior-reuse boundary，不是从当前曲线选择。不同模型和深度的 residual scale 不同，因此 raw 曲线只作同模型、同层的 paired comparison；跨层解释还需参考 relative/cosine secondary metrics 与图 4b 的行为 patch。</figcaption></figure>
  </details>

  <div class="experiment"><div class="experiment-label">Experiment 1A-retention · confirmatory geometry</div><div><h4>删掉 needles 后，occurrence state 是否还留在正确 running-index cloud 附近？</h4><p><strong>目的。</strong>图 3c 的 RMS/cosine 只告诉我们 state 改了多少，不能判断它往哪里改。这里追加一个方向明确、held-out 的检验：第 <span class="math">k</span> 条 needle 被删后，其 state 是否离开由 clean 数据定义的“第 <span class="math">k</span> 条 occurrence”状态云。<strong>数据。</strong>固定 final gold count=10，使每个 prompt 都在同一上下文中提供 running indices 1–10。Discovery seeds 1234–1253 的 20 个 clean prompts 只用于冻结每层、每个 <span class="math">k</span> 的 centroid 与 radius；confirmation seeds 1254–1263 的 10 个 prompts 才用于推断。每个 confirmation prompt 各运行 clean、needle-corrupt、ordinary-corrupt 三臂；Qwen/Gemma 分别审计 720/840 个 seed–layer–pooling keys 与 7,200/8,400 个 occurrence rows。<strong>读取位置。</strong><em>Span endpoint</em> 是每条完整 needle span 的最后一个 token；<em>span-token mean</em> 是该 span 全部 token residual vectors 的算术平均。后者只是一个不挑方向的简单摘要，不等于完整多-token tensor。<strong>窗口。</strong>Qwen L0–20、Gemma L0–16 来自独立 dense-restoration behavior curve，当前结果不能重选层。<strong>目前结论。</strong>该设计专门检验 needle-dependent running-index organization，而不是一般文本扰动或事后选择的 PCA 方向。</p></div></div>

  <div class="formula"><strong>Correct-centroid distance。</strong>在 discovery clean seeds 上，对层 <span class="math">ℓ</span>、running index <span class="math">k</span> 定义 centroid <span class="math">μ_{{ℓ,k}}=20^{{−1}}Σ_s h^{{clean}}_{{s,ℓ,k}}</span>，并定义 class radius <span class="math">r_{{ℓ,k}}=[Σ_s||h^{{clean}}_{{s,ℓ,k}}−μ_{{ℓ,k}}||²/(20−1)]^{{1/2}}</span>。Confirmation state 到其正确 centroid 的标准化距离为 <span class="math">z_{{ℓ,k}}(h)=||h−μ_{{ℓ,k}}||/r_{{ℓ,k}}</span>；约 1 表示与普通 held-out clean state 的离心距离同量级。主效应 <span class="math">S_z=z_{{needle-corrupt}}−z_{{ordinary-corrupt}}</span>：正值表示删掉真正 needles 比删掉同样 token budget 的 ordinary spans 更强地破坏正确 running-index geometry。<span class="example">例：第 5 条 occurrence 的 clean/ordinary/needle distances 为 0.8/1.0/2.6，则 needle-specific effect=2.6−1.0=+1.6 个 discovery-radius。若 RMS change 很大但 <span class="math">z</span> 仍约 1，说明 state 虽移动，却仍在正确 cloud 附近；反之 <span class="math">z</span> 大幅上升才表示 count-specific organization 被破坏。</span></div>

  <div class="formula"><strong>为什么还报告 margin、accuracy、relative RMS 与 cosine？</strong><em>Nearest-centroid margin</em> 是“最近错误 centroid 的标准化距离减正确 centroid 距离”；正值表示正确类更近，margin damage 越大越坏。<em>Nearest-centroid accuracy</em> 是十个 centroids 中最近者是否等于真实 running index；它是 hidden-state 分类率，不是最终答题 accuracy。Relative RMS 与 cosine distance 只量化改变量，不能单独判断是否离开正确 count cloud。我们在看结果前冻结 6 个 retention tests/model 与 4 个 deformation tests/model，并分别做 Holm 校正；正文按固定优先级选择第一个在两模型均为正且校正后 <span class="math">p&lt;0.05</span> 的指标。入选的是事先排第一的 endpoint correct-distance，而非事后寻找最小 <span class="math">p</span>。Hidden width 远大于 20 个 discovery seeds，因此不拟合不稳定的 full-covariance Mahalanobis distance。</div>

  <figure><h4 class="figure-title">图 3d · Needle corruption 是否把 span state 推离正确 running-index manifold</h4><div class="figure-stack"><div><h4>Frozen reusable window · endpoint absolute levels</h4>{state_retention_level_bars}</div><div><h4>Qwen3-8B · needle-minus-ordinary correct-centroid distance</h4>{state_retention_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B · needle-minus-ordinary correct-centroid distance</h4>{state_retention_charts['Gemma4-E4B']}</div></div><figcaption>第一张图在独立冻结的 behavior-reuse window 内汇总 endpoint 到正确 running-index centroid 的标准化距离；横轴单位是对应 discovery class 的 RMS radius。Clean 与 ordinary-corrupt 都约为 1，而 needle-corrupt 升至 Qwen {f(state_retention_endpoint_levels['Qwen3-8B']['needle_corrupt']['mean'], 3)}、Gemma {f(state_retention_endpoint_levels['Gemma4-E4B']['needle_corrupt']['mean'], 3)}。后两张图的横轴是 zero-based post-block layer，纵轴是 <span class="math">z_{{needle}}−z_{{ordinary}}</span>；实线读取 span endpoint，蓝色虚线读取 span-token mean。正值表示 needle corruption 更强地把 state 推离正确 count cloud。Whisker 为 50,000 次 seed-bootstrap 95% interval；点形的 nominal sign-flip 仅作逐层定位，正式判据是预冻结窗口内先平均 layers 与十个 running indices、再以 10 个 confirmation seeds 做 exact <span class="math">2^{{10}}</span> sign-flip 并在检验族内 Holm 校正。竖线来自独立 dense-restoration 实验，不由本图选择。Layer 0 的 endpoint effect 较大，但结论不依赖单层峰值：Qwen 到 L20 仍为 +2.24，Gemma 到 L16 仍为 +1.31；完整窗口主效应分别为 +{f(state_retention_primary['Qwen3-8B']['mean'], 3)} 与 +{f(state_retention_primary['Gemma4-E4B']['mean'], 3)} class radii。</figcaption></figure>

  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Pooling</th><th>Clean correct distance</th><th>Ordinary-corrupt</th><th>Needle-corrupt</th><th>Needle-specific distance</th><th>Margin damage</th><th>Accuracy damage</th></tr></thead><tbody>{state_retention_condition_rows_html}</tbody></table></div>
  <details><summary>展开：所有预先冻结的 retention 与 deformation 指标（没有隐藏不显著项）</summary><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Qwen window mean [95% CI]; adjusted p</th><th>Gemma window mean [95% CI]; adjusted p</th><th>Multiplicity family</th></tr></thead><tbody>{state_retention_all_metric_rows_html}</tbody></table></div><p>Retention family 每模型 6 项，deformation family 每模型 4 项，分别做 Holm 校正。十项在两个模型均为正且校正后 <span class="math">p&lt;0.05</span>；因此主图的选择不改变科学结论。Nearest-centroid accuracy 的 clean window level 本身仅约 Qwen 0.40、Gemma 0.22，说明十类 clouds 有重叠；这也是正文把连续 correct-distance 置于离散 accuracy 之前的原因。</p></details>
  <div class="claim"><strong>图 3d 的直接结论。</strong>在 held-out prompts 上，ordinary corruption 后的 occurrence state 仍与 clean state 一样位于正确 running-index centroid 的约一个 class radius 内；needle corruption 则将 endpoint 推至 Qwen 7.97、Gemma 4.78 radii。主效应为 Qwen +{f(state_retention_primary['Qwen3-8B']['mean'], 3)} [{f(state_retention_primary['Qwen3-8B']['ci95_low'], 3)}, {f(state_retention_primary['Qwen3-8B']['ci95_high'], 3)}]、Gemma +{f(state_retention_primary['Gemma4-E4B']['mean'], 3)} [{f(state_retention_primary['Gemma4-E4B']['ci95_low'], 3)}, {f(state_retention_primary['Gemma4-E4B']['ci95_high'], 3)}] class radii；两者 exact p=0.00195，retention-family Holm p=0.0117。Span-token mean、margin、accuracy、relative RMS 与 cosine 全部同方向通过校正。<strong>这直接支持 needle input 因果维持 count-specific span-state organization；它仍不单独证明这些几何变化就是输出 mediator，第二个箭头必须由图 4b 的内部 state restoration 建立。</strong></div>
  <div class="result-grid">
    <div class="result"><span class="value">Arrow A · count-specific state organization 已跨模型支持</span><span class="label">正确-centroid 距离的 needle-minus-ordinary frozen-window effect：Qwen +{f(state_retention_primary['Qwen3-8B']['mean'], 3)}、Gemma +{f(state_retention_primary['Gemma4-E4B']['mean'], 3)} class radii，Holm p=0.0117/0.0117。Raw full-tensor RMS 的更强“任何尺度都应更大”版本仍不成立；方向明确的 centroid retention 才是主结论。</span></div>
    <div class="result"><span class="value">Arrow B · 已直接验证</span><span class="label">固定同一个 corrupt prompt，只执行 <span class="math">do(H<sub>span</sub><sup>(ℓ)</sup>:=H<sub>span,clean</sub><sup>(ℓ)</sup>)</span>；图 4b 显示它相对等 token-budget ordinary-state patch 可修复约 2–3 counts，图 4c 显示同一 patch 还会重配后续 retrieval。</span></div>
  </div>
  <details><summary>展开：primary 与两项 scale-normalized secondary 的 frozen-window 统计</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Metric</th><th>Window mean [95% seed-bootstrap CI]</th><th>Exact sign-flip p</th></tr></thead><tbody>{state_effect_window_rows_html}</tbody></table></div><p>Frozen windows 由更早、独立的 dense-restoration behavior 结果定义：Qwen L0–20、Gemma L0–16。Relative-RMS pointwise nominal-positive layers：Qwen {state_effect_relative_nominal_text['Qwen3-8B']}；Gemma {state_effect_relative_nominal_text['Gemma4-E4B']}。Cosine pointwise nominal-positive layers：Qwen {state_effect_cosine_nominal_text['Qwen3-8B']}；Gemma {state_effect_cosine_nominal_text['Gemma4-E4B']}。这些 pointwise p 未做跨层 multiplicity correction，只用于定位，不用于重新选择窗口。</p></details>
  <div class="claim"><strong>图 3c、3d 与 4b 为什么必须并列？</strong>图 3c 只问 clean/corrupt hidden tensors 改了多少；图 3d 进一步问它们是否离开正确 running-index cloud；图 4b 才问把 clean state 写回是否还能改变答案。Raw RMS 的 matched window null 与 centroid-retention 的强正效应不矛盾：两种等预算 corruption 可以造成相近的总体位移量，但只有 needle corruption 系统地破坏 count-specific organization。Gemma 的几何差异在 L17–24 仍可见，而 behavior restoration 在 L16→17 已跌到近零；Qwen 晚层也保留几何差异，却不再获得同等行为修复。这说明 perturbation 痕迹可继续传播，但 source evidence 已被 retrieval 转移或错过使用时机。<strong>目前结论：</strong>Arrow A 的 count-specific state response 已由输入干预直接测量，Arrow B 的 causal use 已由内部 patch 直接测量；“可观察到 state difference”仍不能替代 mediation test。</div>
  <div class="formula"><strong>一个具体例子。</strong>Final gold count=10；看第 6 条 occurrence。Clean endpoint 距离“第 6 条”centroid 为 0.9 radius；等预算 ordinary corruption 后为 1.0；删掉 needles 后升至 4.0。图 3d 的 needle-specific effect 是 4.0−1.0=3.0 radii，说明该 state 已离开正确 running-index cloud。随后仍不把输入文字改回来，只在 L8 将全部 span positions 的 residual vectors 换成 clean values；若模型从倾向答 2 回到 5，而同层恢复同样多 ordinary-position vectors 后仍答 2，图 4b 才说明 clean span-state content 对 count 可用。即使到 L24 clean/corrupt states 仍不同、且仍离开 centroid，若两种 restoration 都不救答案，也只能说 perturbation 痕迹还在，不能说 L24 仍保存可复用 source evidence。Endpoint PC1–PC3 即使排列成 1→6，也仍只是高维 states 的 readout。</div>
  <div class="formula"><strong>Absolute-error increase 与 specificity。</strong>对 clean、needle-corrupt、matched-control 输出分别记为 <span class="math">ŷ_0,ŷ_N,ŷ_C</span>，条件 <span class="math">c</span> 的 error increase 为 <span class="math">Δe_{{abs}}(c)=|ŷ_c−N|−|ŷ_0−N|</span>；token specificity=<span class="math">Δe_{{abs}}(needle)−Δe_{{abs}}(control)=|ŷ_N−N|−|ŷ_C−N|</span>。Accuracy damage 是 <span class="math">d_{{acc}}(c)=𝟙[ŷ_0=N]−𝟙[ŷ_c=N]</span>，accuracy-damage specificity 同样为 needle damage 减 control damage。<span class="example">例：gold N=8，clean 输出 8、needle corruption 输出 1、control 输出 7；两种 absolute-error damage 分别为 7 与 1，所以 specificity=6 counts。此例 needle/control 都从 correct 变 wrong，accuracy damages 都为 1，故 accuracy specificity=0；若 control 仍输出 8，则为 1−0=1。</span></div>
  <details class="paper-appendix"><summary>Appendix A3 · Token corruption 与 endpoint rank-3 null 对照</summary>
  <div class="experiment"><div class="experiment-label">Experiment 1B · causal</div><div><h4>Token corruption：active needle 本身是强因果输入</h4><p><strong>目的。</strong>确认 needle 文本而非等规模的普通 passage 扰动决定计数。<strong>设置。</strong>Candidate 与 control 都用同 prompt 的 ordinary token sequences 做等长度替换，并保持总 token budget、序列长度和 query position；只改变被替换区域是否为 active needle。<strong>结果。</strong>Needle-minus-control absolute-error specificity 为 Qwen +8.930 counts（8.700–9.180）、Gemma +8.780（8.590–8.950）；accuracy-damage specificity 为 +0.450/+0.360，ordinary control 自身误差变化接近 0。<strong>分析与目前结论。</strong>Active needle evidence 是强因果输入；效应不能归因于任意等长文本替换。</p></div></div>
  <div class="formula"><strong>Prompt rank-3 removal。</strong>对同一 prompt 的所有 active needle-end states <span class="math">H</span>，删除 discovery-fitted count basis <span class="math">U_3</span> 上的 centered component：<span class="math">H′=H−(H−H̄)U_3U_3^⊤</span>。Orthogonal control basis 不是任取随机方向：它由 discovery rows 的 within-count residuals 拟合，先减各 count centroid、再移除 <span class="math">U_3</span>、取 residual PCA 前三轴并正交化到 <span class="math">U_3</span>。在每个测试 prompt 上，将 control projection 缩放到与 candidate 实际移除量相同的 Frobenius norm。报告的 absolute-error specificity 为 <span class="math">[|ŷ_{{rank3}}−N|−|ŷ_0−N|]−[|ŷ_{{orth}}−N|−|ŷ_0−N|]=|ŷ_{{rank3}}−N|−|ŷ_{{orth}}−N|</span>；正值才表示 count-aligned removal 比同位置、同 rank、等删除量的 nuisance-direction removal 更伤。<span class="example">例：gold N=8，clean 输出 8，rank-3 removal 输出 6（error increase 2），orthogonal removal 输出 7（increase 1），则 specificity=(2−0)−(1−0)=1 count。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1C · causal</div><div><h4>Prompt endpoint rank-3 removal：可解码曲线没有显示局部必要性</h4><p><strong>目的。</strong>检验 endpoint 上可解码的 rank-3 curve 是否就是模型自然依赖的局部 counter。<strong>设置。</strong>在所有 active endpoints 删除 discovery-fitted count component，并与同层、同位置、同 rank、actual-norm-matched 的 within-count orthogonal component 配对；Qwen 扫 10 层，Gemma 扫 13 层。<strong>结果。</strong>逐层 specificity 很小：Qwen −0.022…+0.056，Gemma −0.011…+0.022；独立代表层结果为 +0.056（−0.011–0.133）与 −0.022（−0.089–0.033）。<strong>分析与目前结论。</strong>当前线性 rank-3 endpoint curve 没有显示足以解释行为的局部必要性；这否定的是“强局部 register”，不是否定 prompt states 或完整 spans 的作用。</p></div></div>

  <figure><h4 class="figure-title">图 4 · “输入 evidence 必要”与“decoded endpoint subspace 必要”不是同一命题</h4>{formation_chart}<figcaption>横轴均为 candidate 相对其 paired control 多增加的 absolute count error，单位是 counts。长条的 candidate 是全 active-needle token replacement，control 是同一 prompt 内、相同 span-length vector 与总 token budget 的 ordinary-passage replacement；短条的 candidate 是全部 needle endpoints 上的 count rank-3 removal，control 是同层、同位置、同 rank、每个 prompt 实际删除 Frobenius norm 相同的 orthogonal within-count-residual removal。两者相差约两个数量级。该图不能推出 prompt states 完全无因果作用；它只排除了“当前线性 rank-3 endpoint subspace 是一个强、局部、必要 counter”这一较窄主张。</figcaption></figure>
  </details>
  <div class="claim boundary"><strong>证据边界。</strong>我们将早层表述为 <em>noisy counter-like record</em>，而不是“无 causal effect 的 counter”。更准确的结论是：其 geometry 可解码，active evidence 强因果，但当前 endpoint rank-3 ablation 未检测到相称的局部必要效应。可能原因包括信息分散在整个 span/多个 token、非线性编码、跨位置冗余，或 broad heads 后续重新从原始 evidence 聚合。</div>

  <div class="formula"><strong>完整 hidden-state patch 的主统计量。</strong>在同一 seed–count prompt 和同一层 <span class="math">ℓ</span>，先算恢复全部 clean needle-span residual vectors 后减少的 expected-count absolute error，再减去恢复同样数量、同样 span-length vector 的 ordinary-position residuals 所带来的改善。记为 <span class="math">S<sub>restore</sub>(ℓ)</span>。<span class="math">S<sub>restore</sub>(ℓ)&gt;0</span> 表示“写回真正的 needle-dependent hidden state”比“写回一块同样大的普通 hidden region”更能救回 count；这里 patch 的对象是完整 residual tensor，不要求先投影到任何 subspace。</div>
  <span id="hidden-state-restoration-evidence"></span><figure><h4 class="figure-title">图 4b · Canonical dense span restoration 的完整逐层曲线</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{span_layerwise_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{span_layerwise_charts['Gemma4-E4B']}</div></div><figcaption>每个 panel 的横轴是只执行一次 clean-state restoration 的 zero-based post-block layer；纵轴是 <span class="math">S_{{restore}}(ℓ)</span>，即 full-needle restoration 相对等 token-budget ordinary restoration 多减少的 expected-count absolute error，单位 counts。实线是完整 span hidden-state 主分数；橙色虚线是 endpoint−ordinary control。每层先在同一 confirmation seed 内平均 counts 1–10，再对十个 seeds 等权平均；whisker 是 50,000 次 seed-cluster bootstrap 95% 区间。实心主曲线点表示 two-sided exact <span class="math">2^{{10}}</span> seed sign-flip nominal <span class="math">p&lt;0.05</span>，空心点表示未达到；未做跨层 multiplicity correction，因此“显著”只指逐层 nominal evidence。竖虚线标出 paired seed-level 最大相邻层下降。该图检验的是完整 span state 是否能修复行为，而不是某个预选三维方向是否必要。</figcaption></figure>
  <div class="result-grid">
    <div class="result"><span class="value">Qwen L0–L20</span><span class="label">21 个连续层具有正向 nominal evidence；早层约修复 2.5–2.8 counts，L20 仍为 {f(qwen_l20['mean'])} [{f(qwen_l20['ci95_low'])}, {f(qwen_l20['ci95_high'])}]，L21 降至 {f(qwen_l21['mean'])}。</span></div>
    <div class="result"><span class="value">Gemma L0–L16</span><span class="label">17 个连续层具有正向 nominal evidence；L16 仍为 {f(gemma_l16['mean'])} [{f(gemma_l16['ci95_low'])}, {f(gemma_l16['ci95_high'])}]，L17 降至 {f(gemma_l17['mean'])}，形成更陡的 cliff。</span></div>
  </div>
  <div class="claim"><strong>3.3 的两条因果箭头。</strong><strong>Arrow A（needle input→span state）</strong> 现在有两级直接干预证据：图 3c 量化总体 full-vector response；图 3d 在 held-out clean centroids 上证明 needle corruption 特异地破坏 running-index state organization。<strong>Arrow B（span state→count）</strong> 由图 4b 直接验证：固定 corrupt 输入，只恢复完整 clean span states，可相对 ordinary-state patch 修复约 2–3 counts。另一个不受支持的窄主张仍是“单 endpoint、线性 rank-3 component 必须被自然使用”。这并不与 endpoint centroid response 冲突：endpoint 可以是对上游 count organization 敏感的 readout，却不是独立充分或必要的 mediator；真正获 patch 支持的 causal object 是完整多-token span tensor。</div>
  <details class="collapsible-list"><summary>展开：本次 centroid assay 的边界与仍可追加的检验</summary><div class="experiment"><div class="experiment-label">Scope and next test</div><div><h4>已解决“方向是否正确”，尚未穷尽编码形式</h4><p><strong>已经解决。</strong>指标、split、窗口、优先级与多重比较均在结果前冻结；所有十项候选都报告，主指标不是按最小 p 值挑选。<strong>边界一。</strong>为让每个 prompt 同时含 1–10 的 running indices，本次固定 final N=10；结论尚未直接证明同一 centroid structure 跨 final-count context 不变。<strong>边界二。</strong>Endpoint 与 span-token mean 都是摘要；完整 tensor 的行为作用仍由 restoration 而非 centroid classifier 提供。<strong>边界三。</strong>最近-centroid accuracy 的 clean baseline 有重叠，因此不主张十个离散、紧致 clusters。若论文还需强化，可预注册 state→count 剂量曲线（恢复 25%/50%/75%/100% registered span positions）或在新 seeds 上做跨 final-N centroid transfer；不需要再换 metric 追求更小 p 值。</p></div></div></details>

  <h3>3.4 Dense span restoration：逐层追踪“prompt evidence 还来得及被使用吗？”</h3>
  <div class="experiment"><div class="experiment-label">Experiment 1D · causal</div><div><h4>在内部恢复 clean span state，定位 reusable-source window</h4><p><strong>目的。</strong>判定完整 prompt evidence 在多深的层仍可被后续 computation 使用，并直接比较 endpoint 与 whole-span sufficiency。<strong>设置。</strong>先保存 clean states，再破坏输入 needles；每个 patched forward 只在一个 post-block layer 恢复一次 endpoint、full span 或 token-budget-matched ordinary states，随后完全自由运行。<strong>结果。</strong>72,000-row canonical sweep 显示 Qwen whole-span repair 在 L0–L20 为正向 nominal window，Gemma 在 L0–L16；endpoint−ordinary 全层接近 0。<strong>分析与目前结论。</strong>因果 source 是分布式 span state；Qwen 的可用性逐步衰减，Gemma 在 L16→L17 出现更陡的边界。</p></div></div>
  <p class="lead">可以把这个实验理解为<strong>“先把 needles 擦掉，再在网络内部把 clean 记忆还回去”</strong>。我们先运行 clean prompt，保存每一层 needle positions 的 hidden states；再把输入中的全部 active needles 替换成等长 ordinary text，使模型失去计数证据。随后只在一个指定层 <span class="math">ℓ</span>，把 clean hidden states 写回 corrupt run，并让后面的层照常运行。如果答案因此变好，就说明在 L<span class="math">ℓ</span> 时，把这些上游 states 还给模型仍然来得及影响最终计数。</p>
  <div class="path" aria-label="Dense span restoration workflow"><div class="node"><strong>Clean run</strong><small>保存每层 needle states</small></div><div class="node"><strong>Corrupt run</strong><small>用等长 ordinary text 擦掉 needles</small></div><div class="node"><strong>只在 Lℓ 恢复一次</strong><small>endpoint、whole span 或 ordinary control</small></div><div class="node"><strong>继续正常 forward</strong><small>后续不再 clamp 或重复 patch</small></div><div class="node"><strong>比较答案误差</strong><small>看恢复是否真的救回计数</small></div></div>
  <div class="formula"><strong>一条 layerwise 曲线在问什么。</strong>横轴是“在哪一层把 clean states 还回去”，纵轴是相对 ordinary control 多修复了多少 expected-count error。早层为正，表示 retrieval 尚未结束，恢复后的 prompt evidence 还能被后续 answer query 读取；晚层趋近 0，表示此时再修复 prompt positions 已经太晚，相关信息要么已经被取走并写到 answer state，要么后续层已不再读取这些位置。<strong>因此下降位置定位的是 reusable-source window 的结束，而不是声称该层把信息物理删除了。</strong></div>
  <details class="collapsible-list"><summary>展开：四组 forward 与 matched control 的完整设置</summary><div class="table-wrap"><table><thead><tr><th>Condition</th><th>具体设置</th><th>Matched control / 保持不变</th><th>目的</th></tr></thead><tbody>
    <tr><td>clean / needle-corrupt</td><td>clean prompt；或将所有 active needle spans 用同 prompt 的等长 ordinary source tokens 替换。</td><td>序列长度、每段 token 数、answer-query 绝对位置、generation/parser 固定。</td><td>建立未破坏与破坏后的行为端点。</td></tr>
    <tr><td>endpoint restoration</td><td>在 corrupt run 的单个 post-block layer，只把每个 active needle 的最后一个 token state 换成 clean run 同层、同位置 state。</td><td>与 full-span restoration 使用同一 corrupt prompt、layer 与后续 forward；只改变被恢复的位置集合。</td><td>检验 endpoint 是否局部充分。</td></tr>
    <tr><td>full-span restoration</td><td>同一层把每个 active needle span 的全部 token states 从 clean run 写回 corrupt run，然后继续后续 layers 与严格生成。</td><td>single-layer intervention，不在之后各层重复 clamp。</td><td>检验分布式 span state 是否可复用。</td></tr>
    <tr><td>ordinary-corrupt / ordinary restoration</td><td>在 ordinary passage 中分配两组互不重叠、与 needle span-length vector 完全相同的 target/source banks；破坏后恢复 clean ordinary target states。</td><td>同一 seed/count、总 token budget、span 数、每段长度、layer 与 hook 时点。</td><td>排除“修复任意同样大的 hidden region”本身造成的收益。</td></tr>
  </tbody></table></div></details>
  <p>每个 seed-count 单元包含三个 baseline conditions，以及每层的 endpoint/full-span/ordinary 三种 restoration：Qwen 每单元 3+36×3=111 行、共 33,300 行；Gemma 3+42×3=129 行、共 38,700 行。总计 72,000 unique intervention rows；70,200 行含 patch，51,617 行使用优化后的两次 hook application，18,583 行为兼容保留的三次 application，审计 mismatch=0。<strong>目前结论：</strong>数据覆盖完整逐层扫描，不是从少数“好看层”外推的 coarse sweep。</p>
  <div class="formula"><strong>主分数只做两次减法。</strong>第一步，计算恢复 clean needle states 后少了多少误差：<span class="math">A^N_{{repair}}(ℓ)=|E_{{N-corrupt}}−N|−|E_{{N-restored,ℓ}}−N|</span>。第二步，减去恢复同样多 ordinary-position states 带来的机械收益：<span class="math">S_{{restore}}(ℓ)=A^N_{{repair}}(ℓ)−A^O_{{repair}}(ℓ)</span>。这里 <span class="math">E[c]=Σ_{{c=1}}^{{10}}c·p(c)</span> 是十个数字候选的 softmax expected count。<span class="example">例：gold N=8。擦掉 needles 后 E[c]=3，误差为 5；在 L8 恢复 whole-span states 后 E[c]=6，误差降为 2，因此 needle repair=3 counts。若同层 ordinary control 只修复 0.2 count，则图上的 L8 点为 3−0.2=2.8 counts。正值表示恢复真正的 needle evidence 有额外收益，0 表示它并不比恢复一块同样大的 ordinary hidden region 更有用。</span><details class="collapsible-list"><summary>展开：辅助 normalized recovery</summary><p><span class="math">R_{{restore}}=(E_{{restored}}−E_{{corrupt}})/(E_{{clean}}−E_{{corrupt}})</span>。0.6 表示恢复 clean–corrupt displacement 的 60%；它可因 overshoot 大于 1，也可因方向错误小于 0，所以正文以有 count 单位且更稳定的 <span class="math">S_{{restore}}</span> 为主。</p></details></div>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>正向 nominally detectable layers</th><th>最大相邻层下降 [95% CI], exact p</th><th>微小负向 nominal layers</th><th>Endpoint−ordinary 全层范围</th></tr></thead><tbody>{span_transition_rows_html}</tbody></table></div>
  <p><strong>Qwen 是多层斜坡，而不是单层开关。</strong>Whole-span specificity 从 L0–L14 大致保持在 2.5–2.7 counts，随后连续下降；L20 仍为 {f(qwen_l20['mean'])} [{f(qwen_l20['ci95_low'])}, {f(qwen_l20['ci95_high'])}]，L21 只剩 {f(qwen_l21['mean'])} [{f(qwen_l21['ci95_low'])}, {f(qwen_l21['ci95_high'])}] 且不再 nominally detectable。最大一步是 L{qwen_span_drop['from_layer']}→L{qwen_span_drop['to_layer']} 的 {f(qwen_span_drop['mean'])} counts [{f(qwen_span_drop['ci95_low'])}, {f(qwen_span_drop['ci95_high'])}]，exact p={p_text(qwen_span_drop['exact_signflip_p'])}。因此 Qwen 有效窗口是 L0–L20 共 21 层，转换主要铺在约 L15–L22。</p>
  <p><strong>Gemma 更接近真正的 cliff。</strong>L16 仍有 {f(gemma_l16['mean'])} [{f(gemma_l16['ci95_low'])}, {f(gemma_l16['ci95_high'])}] counts；到 L17 立刻变成 {f(gemma_l17['mean'])} [{f(gemma_l17['ci95_low'])}, {f(gemma_l17['ci95_high'])}]。L{gemma_span_drop['from_layer']}→L{gemma_span_drop['to_layer']} 的 paired drop 为 {f(gemma_span_drop['mean'])} [{f(gemma_span_drop['ci95_low'])}, {f(gemma_span_drop['ci95_high'])}]，exact p={p_text(gemma_span_drop['exact_signflip_p'])}，十个 seed 的差值全部为负。Gemma 的正向有效窗口因此是 L0–L16 共 17 层；discovery 选出的 L17 虽未复现“恰好减半”的数值，却准确落在 confirmation cliff 上。</p>
  <details class="paper-appendix"><summary>Appendix A4 · Restoration→attention response 与 discovery landmarks</summary>
  <h4>同一次 restoration 是否会改变后续 answer-query attention？</h4>
  <div class="formula"><strong>Attention-response specificity。</strong>对同一 frozen head <span class="math">h</span>，先算 true-needle restoration 相对 needle-corrupt baseline 的变化 <span class="math">δM_h^N(ℓ)=M_h^{{N-restored(ℓ)}}−M_h^{{N-corrupt}}</span>；再算等 token-budget ordinary restoration 相对 ordinary-corrupt baseline 的机械变化 <span class="math">δM_h^O(ℓ)=M_h^{{O-restored(ℓ)}}−M_h^{{O-corrupt}}</span>。图中蓝线为 <span class="math">ΔM(ℓ)=|ℋ|^{{-1}}Σ_{{h∈ℋ}}𝔼_{{s,N}}[δM_h^N(ℓ)−δM_h^O(ℓ)]</span>；对 broad score <span class="math">B=M×C</span> 做同样两次减法得到 <span class="math">ΔB(ℓ)</span>。正值表示修复 true needle evidence 比修复同样大的 ordinary hidden region 更能改变后续 answer-query routing。<span class="example">例：needle baseline/restored 的 mass 为 0.20/0.48，ordinary baseline/restored 为 0.21/0.22，则 ΔM=(0.48−0.20)−(0.22−0.21)=0.27。若 broad score 两组为 0.15/0.39 与 0.16/0.165，则 ΔB=0.235。它们是 attention-derived units，不是修复了 0.27 或 0.235 个 count。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 1E · causal response</div><div><h4>Restore span states once, then re-measure the downstream frozen broad bank</h4><p><strong>目的。</strong>检验早层 full-span restoration 是否不仅修复答案，还会重新配置后续 answer-query retrieval。<strong>设置。</strong>复用 canonical 30 seeds×10 counts；Qwen/Gemma 固定最终 top-32/top-6 head registry，不按 response curve 重选 head。每个 forward 只在指定层恢复一次，随后重建 downstream cache 并读取同一 forward 的 final broad bank。<strong>结果。</strong>Behavior-coupled 主窗口约止于 Qwen L20、Gemma L16；attention-only 弱尾部延至 Qwen L26、Gemma L22。Top-6 重算后 Gemma 的 broad-score specificity 为 L16 0.340、L17 0.142、L22 0.108、L23 0，边界不变。<strong>分析与目前结论。</strong>早层 span evidence 会因果改变后续 routing，但曲线定位的是“何时还可影响 retrieval”，不是 retrieval head 直接跨层读取某个历史 layer。</p></div></div>
  <figure><h4 class="figure-title">图 4c · Canonical full-span restoration 对后续 broad retrieval attention 的逐层影响</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{span_attention_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{span_attention_charts['Gemma4-E4B']}</div></div><figcaption>横轴是只恢复一次 clean full-needle-span states 的 zero-based post-block layer；纵轴是 true-needle response 再减 ordinary-restoration response 的 matched specificity。灰色虚线为 active needle 总 attention mass specificity <span class="math">ΔM</span>；模型色实线为同时考虑 mass 与多-span coverage 的 broad-score specificity <span class="math">ΔB</span>。Qwen 的 ΔB 在 L0/L16/L20/L21/L24/L26 为 0.310/0.302/0.196/0.061/0.023/0.010，L27 后为 0；Gemma Top-6 在 L0/L16/L17/L20/L22/L23 为 0.218/0.340/0.142/0.108/0.108/0。ordinary-control response 全程接近 0，因此主要变化来自 true needle evidence。竖虚线标出独立 behavior restoration curve 的主要 reusable-source 边界；attention-only 的衰减尾部不等于仍能显著修复最终答案。</figcaption></figure>
  <div class="claim"><strong>Restoration→retrieval 结论。</strong>在较早层恢复完整 needle-span evidence，会重新配置后续 answer-query 的 broad retrieval；这一影响在 Qwen 约持续到 L20、Gemma 约持续到 L16。这里的“持续”专指同时伴随 behavior repair 的主要可用窗口；canonical attention readout 还显示较弱尾部，分别延至 Qwen L26 与 Gemma L22，但此时已不足以恢复最终 count。它定位的是“prompt evidence 在深度上何时仍可被后续 retrieval 使用”，而不是 retrieval head 直接读取了哪个历史层。Transformer 的某个 retrieval head 只读取其自身输入深度上的 token states；layerwise restoration 改变的是这些 token states 经后续 blocks 演化后，是否还能影响该 head 的 routing。</div>
  <div class="claim boundary"><strong>如何读晚层的负值。</strong>Qwen L24–L27/L29–L30 与 Gemma L20–L22 有约 −0.04 至 −0.12 count 的 nominal negative specificity：此时恢复 needle positions 反而比 ordinary control 略差。它们比早层 +2–3 counts 小一个数量级，只说明 late patch 有轻微 overshoot/control imbalance；不能解释成模型在这些层使用“反向 counter”。Endpoint−ordinary 在全层仅 Qwen {f(span_summary_by_model['Qwen3-8B']['endpoint_minus_ordinary_min'])}…+{f(span_summary_by_model['Qwen3-8B']['endpoint_minus_ordinary_max'])}、Gemma {f(span_summary_by_model['Gemma4-E4B']['endpoint_minus_ordinary_min'])}…+{f(span_summary_by_model['Gemma4-E4B']['endpoint_minus_ordinary_max'])} counts，继续支持 endpoint 单点不充分。</div>
  <details class="collapsible-list"><summary>展开：discovery pilot 与预冻结 transition landmarks</summary><div class="experiment"><div class="experiment-label">Discovery-only contrast</div><div><h4>Endpoint 几乎不能修复，whole span 在早层能修复 2–3 counts</h4><p><strong>目的。</strong>在 canonical confirmation 之前冻结 candidate layers、span-vs-endpoint contrast 与 transition landmarks。<strong>设置。</strong>Fresh pilot 只用 seeds 2000–2003、counts 3/6/9，不与 confirmation 合并。<strong>结果。</strong>Qwen endpoint normalized recovery 全层仅 −0.047…+0.032；whole-span recovery 在 L0/4/8/12 为 1.018/0.997/0.976/0.982，L16=0.849、L20=0.198。Gemma endpoint repair −0.042…+0.062；whole-span specificity 在 L0/4/8/12/16/20/24 为 3.327/3.497/3.350/2.618/2.297/−0.032/0。Broad-score change 在相同窗口衰减。<strong>分析与目前结论。</strong>Pilot 预先支持“full span 强、endpoint 弱”并冻结边界；正式科学结论只使用后续 canonical confirmation，而不把 pilot 与 confirmation 合并增大样本。</p></div></div>
  <figure><h4 class="figure-title">图 4d · Discovery 预冻结 landmarks 在 confirmation 上的 readout</h4>{span_landmark_chart}<figcaption>横轴是 full-needle restoration 相对等 token-budget ordinary restoration 多减少的 expected-count absolute error，单位 counts；竖向每条 bar 是一个预定义 landmark。Early plateau 是 discovery seeds 1234–1253 前四分之一层的 median specificity；half-boundary 与 near-zero boundary 的层号只由 discovery curve 冻结，图中 bar value 则在 confirmation seeds 1254–1263 计算。Qwen 在 L19 仍有 +1.294，L23 已为 −0.074；Gemma L17 的 confirmation 值为 −0.088。完整曲线表明，Gemma 的 literal half-height 没有复现，但 L17 确实是 +2.018→−0.088 的 cliff 位置。不同 bar 来自不同阶段/层，不把 discovery 与 confirmation 混成一个总体均值。</figcaption></figure>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Discovery early plateau</th><th>Frozen half-boundary: confirmation specificity</th><th>Frozen near-zero boundary: confirmation specificity</th></tr></thead><tbody>{span_landmark_rows}</tbody></table></div></details>
  </details>
  <div class="claim"><strong>Stage-I conclusion。</strong>旧 endpoint rank-3 null 不是“prompt states 没有 causal effect”。新的 held-out centroid assay 表明，needle input 对 early occurrence states 的 running-index organization 是必要的；dense restoration 又表明，完整 needle span 在早层含有强、可复用的因果信息。单个 endpoint 虽能显示 count-specific response，却既不充分，其线性 rank-3 component 也未显示必要性，因此不能升级为独立 counter register。逐层行为曲线把两个模型区分得更清楚：Qwen 的 reusable-source effect 在 L15–L22 逐步衰减，Gemma 在 L16→L17 突降。两者都表明在该边界之后才恢复 prompt positions 已经太晚，符合后续 retrieval 已开始/完成并把信息转入 answer-side state 的机制；现有实验不定位是哪一个 span token 或哪一个 head 完成转换。我们因此将上游状态概括为 <em>distributed, count-organized span-level evidence</em>。</div>
</section>

<section id="retrieval">
  <h2>Step 2 · Retrieve — Answer query 广域取回并聚合多条 needle evidence</h2>
  <p class="lead">在 answer query <span class="math">q</span> 上，head 是否“broad”不能只由总 attention mass 决定：只盯住一个 needle 的 head 不是 aggregation head。我们同时要求 mass 高且覆盖多个 needles。该阶段的目标不是证明唯一 counting channel，而是证明一组预先冻结的 broad heads 及其 count-aligned output subspace 在自然 computation 中具有 matched-control causal effect。</p>
  <div class="claim"><strong>Stage-II hypothesis。</strong>Answer query 在中层通过多个 broad heads 并行读取 active needle spans，并把其中一部分聚合结果写入 count-aligned answer residual。Attention map 只用于定义和可视化 routing；最终机制主张由 head ablation、donor source patch 和 subspace mediation 共同支持。</div>
  <div class="chain-blueprint" aria-label="Chain B mechanism representation causal test">
    <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>区分“某些 heads 的 attention 看起来覆盖多个 needles”与“模型作答时确实依赖这些 heads 写出的 count-aligned content”。<span class="mini-example"><strong>直观例子：</strong>一个调查组可以同时翻阅三份证据（routing），但只有它把三份内容汇总到报告中（post-O write），而且删掉该汇总比删掉同样大小的无关段落更伤结论，才能说这条 aggregation path 被自然使用。</span></p></div>
    <div class="evidence-triad">
      <div class="triad-step"><span class="protocol-no">01 · Mechanism</span><h3>Answer query 广域聚合</h3><p>多个 heads 从不同 active spans 读取 evidence，并把一部分合计写入 answer residual。</p></div>
      <div class="triad-step"><span class="protocol-no">02 · Representation</span><h3>Routing 与 content 分开测</h3><p>Broad score <span class="math">B=M×C</span> 描述“看哪里”；broad-bank post-O state 的 held-out geometry 描述“写出了什么”。</p></div>
      <div class="triad-step"><span class="protocol-no">03 · Causal test</span><h3>集合与方向都做 matched control</h3><p>Top-K 对 layer-matched random；count-aligned removal 对 equal-realized-norm orthogonal；再测试 source→late mediation。</p></div>
    </div>
  </div>
  <details class="paper-appendix"><summary>Appendix B1 · 完整 head atlas、broad-score 定义与 frozen membership</summary>
  <div class="step-heading"><span class="step-kicker">02 · Representation · routing</span><h3>4.1 哪些 heads 同时覆盖多个 active spans？</h3></div>
  <div class="formula"><strong>Broad retrieval score。</strong>对第 <span class="math">i</span> 个完整 active-needle span <span class="math">S_i</span>，<span class="math">m_{{i,h}}=Σ_{{j∈S_i}}α_h(q,j)</span>；总 needle mass <span class="math">M_h=Σ_i m_{{i,h}}</span>；<span class="math">p_{{i,h}}=m_{{i,h}}/M_h</span>；coverage <span class="math">C_h=exp(−Σ_ip_{{i,h}} log p_{{i,h}})/N</span>；最终 <span class="math">B_h=M_hC_h</span>。<span class="math">C=1</span> 表示均匀覆盖全部 N 个 needles，接近 <span class="math">1/N</span> 表示只覆盖一个。排名使用 discovery 数据上 <span class="math">B_h</span> 的平均值。<span class="example">例：N=4，四个 span masses 都为 0.10，则 M=0.40、C=1、B=0.40；若全部 0.40 只落在一个 span，则 M 仍为 0.40，但 C=1/4，B=0.10。</span></div>
  <div class="path" aria-label="Broad aggregation computation">
    <div class="node"><strong>Needle spans S₁…Sₙ</strong><small>分散在约 10k-token prompt</small></div>
    <div class="node"><strong>Answer query q</strong><small>每个 head 产生 α(q,j)</small></div>
    <div class="node"><strong>Broad head bank</strong><small>Qwen L23/L27；Gemma full L29/L35</small></div>
    <div class="node"><strong>pre-O z<sub>h</sub></strong><small>Σ αVh，先聚合再经各自 W<sub>O</sub></small></div>
    <div class="node"><strong>Answer residual</strong><small>写入 donor/count-related state</small></div>
  </div>
  <figure><h4 class="figure-title">图 5 · Answer-query broad-score attention maps</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{attention_maps['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{attention_maps['Gemma4-E4B']}</div></div><figcaption>每个 panel 的横轴是 attention head H，纵轴是 transformer layer L，均为 zero-based；每个 cell 是该 head 从最后一个 answer query（<code>Total:</code> 后首数字前）指向所有完整 active needle spans 的 discovery-mean broad score B。颜色越深 B 越大；黑框标出后续因果分析统一使用的 frozen Qwen top-32 / Gemma top-6，cell 内数字是 frozen rank。这里是 layer×head 的聚合 attention-score map，不是单个样本的 token×token raw attention matrix。</figcaption></figure>

  <h4>Frozen broad-head membership</h4>
  <div class="figure-grid">
    <div><h4>Qwen3-8B top-32</h4><div class="table-wrap"><table class="head-table"><thead><tr><th>Rank</th><th>Head</th><th>M</th><th>C</th><th>B</th></tr></thead><tbody>{selected_head_tables['Qwen3-8B']}</tbody></table></div></div>
    <div><h4>Gemma4-E4B top-6</h4><div class="table-wrap"><table class="head-table"><thead><tr><th>Rank</th><th>Head</th><th>M</th><th>C</th><th>B</th></tr></thead><tbody>{selected_head_tables['Gemma4-E4B']}</tbody></table></div><p class="lead">M 是完整 needle spans 的 attention mass，C 是 occurrence coverage，B=M×C。表内顺序严格来自 causal run 使用的 frozen membership；由于源 registry 与汇总 atlas 的数值聚合版本略有差异，B 列用于量级解释，rank/membership 以 frozen registry 为准。</p></div>
  </div>
  </details>

  <div class="experiment"><div class="experiment-label">Display panel · fixed before viewing</div><div><h4>从聚合 heatmap 展开到多个 head × 多条自然文本</h4><p><strong>目的。</strong>让读者检查“broad”是否只由一个特殊 head 或一条特殊文本造成，并同时看清全文位置分布与 span 内 token 分布。<strong>Head 选择。</strong>固定使用图 5 discovery ranking 的 Qwen top-4：L27H18、L28H19、L23H29、L23H13；confirmation attention 不参与排序。<strong>文本选择。</strong>固定最小 confirmation seed 1254，并在查看 raw attention 前选低/中/高三个 counts N=3/6/9；没有按图形显著程度挑文本。页面下拉框只切换已冻结的 4×3=12 条自然 attention rows，不重跑模型、不改变任何统计检验。<strong>分析与目前结论。</strong>该 panel 用于解释 routing 的空间与 token 形态；因果必要性仍由后面的 top-K matched ablation 与 rank-3 removal 判定。</p></div></div>

  <div class="control-definition">
    <h4>术语与对照边界：协议中的 <code>hard_negative</code> 是什么？</h4>
    <p><strong>一句话定义。</strong>它不是另一条“错误城市记录”，而是同一条自然 forward 中、从原始 haystack 选出的 <strong>nearby length-matched ordinary-span control</strong>。报告正文使用后一个名称；<code>hard_negative</code> 仅保留为冻结数据字段名。</p>
    <div class="control-definition-grid">
      <div class="control-term"><strong>Active needle span</strong>第 <span class="math">i</span> 条真实城市—分数记录，token 区间为 <span class="math">S_i=[s_i,e_i)</span>，长度 <span class="math">w_i=e_i−s_i</span>；它计入 gold count。</div>
      <div class="control-term"><strong>Nearby length-matched control</strong>首选区间为 <span class="math">H_i=[s_i−8−w_i,s_i−8)</span>：与对应 needle 等长，通常在其前方留 8-token gap。若该窗口不可用，则按冻结搜索顺序选择其他不重叠窗口。</div>
      <div class="control-term"><strong>候选必须满足</strong>不与任何 slot 或已选 control 重叠，并且文本中不得匹配城市计分模板 <code>In the 2024 city score audit, CITY received a score of SCORE.</code>。因此它是普通 haystack prose，而非合成 distractor record。</div>
      <div class="control-term"><strong>匹配了什么、没有匹配什么</strong>它匹配 token 长度、slot 配对身份与近似局部深度；<strong>不匹配绝对 token position，也不严格匹配句法或语义</strong>。“同 slot”表示配对索引相同，不表示两个 spans 坐标重合。</div>
      <div class="control-term"><strong>为什么不能与 needle 重合</strong>Figure 5b 只展示一条 natural forward；同一 token position 不可能同时放置 needle 与 ordinary text。黄色 control 因而通常出现在红色 needle 左侧，而不是盖在红格上。</div>
      <div class="control-term"><strong>真正的 same-position control</strong>需要第二条 paired forward：保持其余 prompt 不变，在原区间 <span class="math">S_i</span> 用等 token-budget ordinary text 替换 needle，再比较两次 forward 的 attention、state 或 output。Figure 5b 没有执行这一 counterfactual。</div>
    </div>
    <div class="control-contrast"><strong>正式 attention contrast。</strong>令 <span class="math">m_i^+=Σ_{{t∈S_i}}α(q,t)</span>、<span class="math">m_i^−=Σ_{{t∈H_i}}α(q,t)</span>，则 matched density contrast 为 <span class="math">N^{{−1}}Σ_i[m_i^+/w_i−m_i^−/w_i]</span>。它检验 needle 是否比附近等长普通文本获得更多 per-token attention，但不能单独证明语义识别或因果必要性。<span class="example">例：默认 N=6 panel 中，Harbin needle 为 tokens 1059–1089；其 control 为 1020–1050，两者均长 31 tokens，中间相隔 8 tokens。L27H18 给它们的 span mass 分别约为 6.45% 与 0.058%。</span><br><strong>为什么 N=6 仍显示 10 个黄色 spans。</strong>每条 frozen stimulus 对最多 10 个 slots 都注册 control，便于所有 N=1…10 共享同一位置骨架；正式 N=6 contrast 只配对 active slots 1–6，而 Figure 5b 的区域总 mass 为保持全文记账完整，会显示并统计全部 10 个 registered controls。</div>
  </div>

  <figure><h4 class="figure-title">图 5b · 自然 forward 的全文 attention routing：needle 城市、haystack 分箱与 span mass</h4>{attention_span_gallery}<figcaption>每个可切换 panel 都来自一条<strong>未干预自然 forward</strong>，显示最后 answer query 在一个 discovery-frozen head 中的完整 attention row。顶部组成条精确给出 active needles、ordinary haystack、nearby length-matched controls（冻结字段名 <code>hard_negative</code>）、instruction/wrapper 与 query 的总 mass。A 图把约 10k-token 全文按<strong>每格 64 tokens</strong>切分，按原始 token 顺序从左到右、换行后继续；由浅粉到深红表示该格所有 token 的总 attention mass。颜色使用单调对数映射，并以全部 panels 的 passage-bin 最大值定标，使低-mass haystack 与高-mass bins 都可见；悬停显示精确 token 范围与百分比。普通格内不放任何文字；城市名只写在对应 needle 起始格内部，needle 覆盖格另有深红上边，nearby-control 覆盖格有黄色下边。无红/黄边且完整位于 passage 内的格子才是精确 ordinary-haystack bin；mixed/special 格不能误读成 ordinary-only mass。B 图将每条 needle 内所有 token weights 求和，12 个 panels 共用同一百分比上限。<span class="example">例：若 Vilnius 所在的 64-token 格颜色较深，说明该局部窗口得到较多总 attention；下方 Vilnius 的 span bar=13.72% 才表示完整 attention row 中有 13.72% 落在整条城市记录。两者都是 routing 描述，不是最终答案的 causal contribution。</span></figcaption></figure>

  <details class="paper-appendix"><summary>Appendix B2 · Token-level attention gallery</summary>
  <figure><h4 class="figure-title">图 5c · 选定 head–prompt 的 needle-token attention 细图</h4>{attention_token_gallery}<figcaption>该图与 5b 使用完全相同的 12 条冻结 rows，只把当前选择展开到 active needle 内部。每一行是一条真实 needle span；行首 span mass 是该 span 全部 token 权重之和，token 红色深浅按当前 row 的最大单-token weight 做平方根尺度显示。顶部五项把全序列 mass 分为 active needles、ordinary passage、nearby length-matched controls（冻结字段名 <code>hard_negative</code>）、instruction/wrapper 与 query。可切换不同 heads 与 N=3/6/9 文本，检查 head 是较均匀覆盖 record、偏向固定标点/模板，还是集中在少数 spans。图形语法参考 <a href="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/">Olsson et al. (2022)</a>；attention weight 只表示 routing，不等于该 token 的 OV/logit attribution，也不证明删掉它会改变答案。</figcaption></figure>
  </details>

  <div class="step-heading"><span class="step-kicker">03 · Causal test · routing</span><h3>4.2 Broad-head 集合是否比 layer-matched random 更重要？</h3></div>
  <div class="formula"><strong>Absolute count shift。</strong>对同一个样本，clean 生成数为 <span class="math">ŷ_0</span>，消融后为 <span class="math">ŷ_a</span>，定义 <span class="math">shift_{{abs}}=|ŷ_a−ŷ_0|</span>；它衡量输出被移动多少，不以 gold N 为参照，也不是 absolute error。Top-K 主效应在每个 seed 内计算 <span class="math">mean(shift_{{ranked}})−mean(shift_{{layer-matched random}})</span>，再对 20 seeds 等权平均。Clean-correct correct→wrong=<span class="math">𝟙[ŷ_0=N∧ŷ_a≠N]</span>。<span class="example">例：clean 输出 8；ranked-head ablation 输出 5，shift=3；matched random ablation 输出 7，shift=1；ranked-minus-random absolute count shift=3−1=2 counts。若 gold=8，该 ranked trial 的 correct→wrong=1；若 gold 不是 8，则它不进入 clean-correct 指标。</span></div>
  <div class="formula"><strong>逐 K 的区间与“统计可检出”判据。</strong>原注册网格令第 <span class="math">s</span> 个 seed 的 ranked-minus-random 效应为 <span class="math">δ_s(K)</span>，报告 20 seeds 的等权均值与 10,000 次 seed-cluster bootstrap 区间；表中的 <span class="math">p&lt;0.05</span> 专指 two-sided exact seed sign-flip。Gemma K6 是后加的 post-hoc dose，沿用正式 analyzer 的 cluster bootstrap（clean-correct 时按 seed 内有效样本数加权），但不追加 exact sign-flip 声称。这个区分防止把事后加入的 dose 伪装成原注册检验族的一部分。效应的 count 单位回答“移动有多大”，而 p 或 CI 回答“这套推断下是否排除 0”，两者不能互换。</div>
  <div class="experiment"><div class="experiment-label">Experiment 2A · causal</div><div><h4>Ranked top-K ablation with layer-matched random-head controls</h4><p><strong>目的。</strong>检验 high-broad-score head set 是否比同层、同数量的普通 heads 更影响输出。<strong>正式数据。</strong>Qwen 与 Gemma 分开使用此前未参与 head ranking 的 seeds 1316–1335、counts 1–5，即每模型 20×5=100 个 prompt 单元；两个模型从不合并。<strong>K 网格。</strong>原始预冻结网格为 K∈{{1,2,4,8,16,32}}。为与 native-thinking 最终采用的 Gemma K=6 对齐，我们随后在完全相同的 100 个 Gemma prompts、同一 discovery ranking 与三个 layer-matched random controls 上追加一次 K=6 post-hoc dose；它不被追溯性写成原注册网格的一部分。<strong>干预。</strong>只在原始完整 prompt 的一次 prefill 中，将所选 head 在 answer-query token 上、进入该层 W<sub>O</sub> 之前的 pre-O slice 置零；其他 token positions、未选 heads 与后续 decoding steps 不消融。<strong>Matched control。</strong>对每个 ranked prefix，逐层保留完全相同的 head 数，再从这些层的全部 heads 中无放回抽取；每个 K 使用 3 个固定随机 replicate，ranked/random 之间允许偶然重叠。<strong>结果。</strong>Qwen K32 的原始 ranked/random shift 为 1.750/0.127 counts，差值 +1.623 [1.117,2.137]；Gemma K6 为 0.990/0.165，差值 +0.825 [0.690,0.975]。Gemma K6 的 clean-correct correct→wrong contrast 为 +0.275 [0.179,0.348]。<strong>分析与目前结论。</strong>冻结 broad bank 具有集合级 matched-control 行为作用，但 heads 不是可简单相加的独立 counters；K=6 是同一冻结前缀上的透明 post-hoc dose extension，不是结果驱动的新 head selection。</p></div></div>
  <details class="collapsible-list"><summary>展开：Experiment 2A 的可复现数据、K 与 ablation 算子</summary><div class="table-wrap"><table><thead><tr><th>项目</th><th>冻结设置</th><th>为什么这样控制</th></tr></thead><tbody>
    <tr><td>Ranking discovery</td><td>Broad ranking 来自独立 discovery seeds 1274–1283；formal panel 的 1316–1335 在任何 outcome 被查看前已冻结 registry 与 K 网格。</td><td>避免在测试数据上选 heads 或挑最好看的 K。</td></tr>
    <tr><td>Formal evaluation</td><td>每模型 seeds 1316–1335 × counts 1–5=100 prompt 单元；原网格 K=1/2/4/8/16/32，Gemma 另加 post-hoc K=6。</td><td>每个 seed 是 cluster；K=6 复用原 stimuli、ranking 与 controls，并单独标记其 post-hoc 状态。</td></tr>
    <tr><td>Ranked arm</td><td>每个 prompt/K 运行一次 frozen top-K ablation，共 100 rows/K/model。</td><td>直接测量该 ranked prefix 的行为损伤。</td></tr>
    <tr><td>Random arms</td><td>同一批 100 prompts ×3 个 layer-matched random sets，共 300 rows/K/model；每组内无放回，同层 head 数与 ranked set 完全一致。</td><td>控制“在这些层删掉 K 个任意 heads”的非特异损伤。</td></tr>
    <tr><td>总 intervention coverage</td><td>每 K 每模型 400 rows；原网格 2,400 rows/model，Gemma K6 extension 另有 400 rows。</td><td>所有 doses 使用同一 stimulus panel；extension 不覆盖原始文件。</td></tr>
    <tr><td>精确置零位置</td><td>Full-prompt prefill 的 answer-query position、selected-head pre-O z slice；该 slice 置零后仍经过 head 自己的 W<sub>O</sub>。Prompt 其他位置与 generation token rows 保持 natural。</td><td>把结论限定为 answer-query retrieval/write 的局部必要性，而非全序列全时段 head knockout。</td></tr>
  </tbody></table></div><p><span class="example"><strong>例：</strong>若 Qwen 的 K=4 prefix 分布为 L27 一个 head、L28 一个、L23 两个，则每个 random replicate 也必须从 L27/L28/L23 分别抽 1/1/2 个 heads；不能从另一个更脆弱的层随便抽四个。对同一 seed-count prompt，ranked arm 和三个 random arms 都只在 answer query 的 pre-O slices 上置零。</span></p></details>
  <figure><h4 class="figure-title">图 6a · Broad-head ablation 的 absolute-shift 剂量曲线</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{retrieval_charts['Gemma4-E4B']}</div></div><figcaption>横轴是被消融 heads 占 discovery-eligible broad-atlas heads 的比例：Qwen 分母 1,152，Gemma 分母 56。Qwen 显示原网格 K=1/2/4/8/16/32；Gemma 在相同网格中加入 post-hoc K=6。模型色实线、灰色虚线与紫色点线分别为 ranked raw arm、三个 layer-matched random replicates 的 raw mean，以及 ranked−random contrast；whisker 为 10,000 次 seed-cluster bootstrap 95% CI。原网格 contrast 的点形沿用 exact sign-flip 判据；Gemma K6 只作 post-hoc bootstrap-CI 推断，详见表中 † 标记。模型色按全文约定固定为 Qwen Aurora blue、Gemma orange。</figcaption></figure>
  <div class="conclusion-line"><strong>Step 2 正文结论。</strong>自然 attention 示例显示 answer query 会同时覆盖多条远端 needle spans；冻结 broad ranking 的 matched-control ablation 进一步造成 Qwen K32 +1.623 counts、Gemma K6 +0.825 counts 的额外输出移动。Gemma K6 是透明标注的 post-hoc dose extension，但其 head membership 仍是 discovery ranking 的固定前缀。因此 broad routing 不是仅供观看的 attention pattern，而是两模型作答时使用的一组集合级 retrieval/aggregation components；它们并非可独立相加的 counters。</div>
  <details class="paper-appendix"><summary>Appendix B3 · Correct→wrong、source patch、broad-bank geometry 与 retrieval-subspace mediation</summary>
  <figure><h4 class="figure-title">图 6b · Clean-correct correct→wrong damage 的剂量曲线</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_damage_charts['Qwen3-8B']}</div><div><h4>Gemma4-E4B</h4>{retrieval_damage_charts['Gemma4-E4B']}</div></div><figcaption>只保留 clean baseline 答对且格式有效、并在 ranked/random 条件间具有相同 stimulus ID 的样本。纵轴为 <span class="math">P(wrong|ranked, clean correct)−P(wrong|random, clean correct)</span> 及其两条 raw arms；0.20 即额外 20 percentage points。Gemma K6 的 ranked/random raw rates 为 0.365/0.090，contrast +0.275 [0.179,0.348]。原注册网格保留 exact sign-flip 点形；K6 仅报告 post-hoc seed-cluster bootstrap CI。</figcaption></figure>
  <h4>表 2 · 每个 K 的完整 matched-control ablation 结果</h4>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>K</th><th>Δ absolute shift [95% CI]</th><th>Positive seeds</th><th>Shift nominal p&lt;.05?</th><th>Δ correct→wrong [95% CI]</th><th>Damage nominal p&lt;.05?</th></tr></thead><tbody>{topk_result_table}</tbody></table></div>
  <p class="lead">两列效应都是 ranked top-K 减去 layer-matched random-head control；正值表示 broad-ranked heads 被消融后，输出移动得更多或更容易由 clean-correct 变为错误。† Gemma K6 是同一 frozen ranking 与 stimuli 上追加的 post-hoc dose，因此表中不补写 exact sign-flip p，只报告预先固定分析代码得到的 seed-cluster bootstrap CI。</p>
  <div class="claim boundary"><strong>如何解读大小。</strong>效应量与推断标签必须分开：Qwen K32 +1.623 来自原注册网格；Gemma K6 +0.825 是后来为跨模式一致性追加、但没有重新挑 heads 的 dose。两者都支持集合级 retrieval 作用；K6 的 post-hoc 身份要求在论文中保留，不能改写成预注册最优 K。后续 retrieval-subspace、serial mediation 和 broad-score readout 已统一按这六个 Gemma heads 重算。</div>
  <details class="collapsible-list"><summary>展开：Gemma Top-6 与历史 Top-8 的逐项 claim-preservation 对照</summary>
    <div class="table-wrap"><table><thead><tr><th>依赖 broad bank 的分析</th><th>历史 Top-8</th><th>统一 Top-6</th><th>报告级判定</th></tr></thead><tbody>{top6_vs_top8_rows}</tbody></table></div>
    <p class="lead">这张表是<strong>结论保持性审计</strong>，不是预先设定界值的统计等效性检验。判断标准是：同一 discovery ranking、stimuli、controls 与 intervention protocol 下，Top-6 是否保留 Top-8 所支撑的效应方向、关键 gate 与科学判定。Retrieval-subspace 对照中的区间是在同一个 10-seed×10-count frozen paired panel 上追加的 10,000-draw equal-weight seed-cluster bootstrap，仅用于审计方向与不确定性，不追溯性升格为新的 confirmatory test。所有列出的结论均保留，因此当前 report 统一使用 global Top-6；Top-8 只保留为历史 provenance，以及原始 K 网格中的 K=8 剂量。</p>
  </details>

  <div class="formula"><strong>Source transport、terminal adoption 与 mediation。</strong>令 <span class="math">E_R[c]</span> 与 <span class="math">E_{{patch}}[c]</span> 为 counts 1–10 的 candidate-sequence softmax expected count；continuous normalized transport=<span class="math">(E_{{patch}}[c]−E_R[c])/(N_D−N_R)</span>。若改用实际生成数，则 strict generated transport=<span class="math">(ŷ_{{patch}}−ŷ_R)/(N_D−N_R)</span>，invalid generation 记 0；两者都不裁剪，1 表示完成一个 receiver→donor displacement。Terminal adoption 把 L41 residual change 投到该层 frozen one-count step <span class="math">s_T</span> 后再除以 count gap：<span class="math">A_T=⟨h'_T−h_T,s_T⟩/[‖s_T‖²(N_D−N_R)]</span>。Qwen sequence readout 定义 <span class="math">g(x)=log p(a_D|x)−log p(a_R|x)</span>，source gain=<span class="math">g(x_{{patch}})−g(x_R)</span>。Exact-component mediation specificity 是 source intervention 后“正交 control block 保留的 gain/transport − exact induced-component block 保留的 gain/transport”。<span class="example">例：receiver N=3、donor N=8，E<sub>R</sub>[c]=3.2、E<sub>patch</sub>[c]=6.2，则 continuous transport=(6.2−3.2)/5=0.6；若 L41 投影得到 2.5 count-axis units，则 terminal adoption=2.5/5=0.5。若 source gain=0.40，orthogonal block 后为 0.35、exact block 后为 0.10，则 mediation specificity=0.25。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 2B · causal mediation</div><div><h4>Donor source-state patch and downstream mediation</h4><p><strong>目的。</strong>把“消融一组 heads 会伤行为”推进为有方向的 source→downstream mediator 链。这里的 donor 与 receiver 是<strong>同一 seed、token 坐标完全对齐、但 gold count 不同</strong>的两个 prompts；patch 问的是“把 donor 内部 source state 写进 receiver 后，receiver 的输出是否朝 donor count 移动”。<strong>结果。</strong>Qwen early top-4 产生 +{f(qwen_upstream_primary['early_effect']['mean'],4)} donor log-odds gain，L28 H16–H19 mediation specificity +{f(qwen_upstream_primary['mediation']['mean'],4)}；Gemma K2 source transport +{f(gemma_candidate['source_donor_transport']['mean'],4)}，L37 exact residual mediation +{f(gemma_candidate['exact_residual_mediation']['mean'],4)}。<strong>分析与目前结论。</strong>Broad source state 可传 donor-directed information，且其中一部分经注册的晚层 mediator 传递；仍不证明它是唯一通路。</p></div></div>
  <details class="collapsible-list" open><summary>展开：Experiment 2B 的两种 model-specific patch 实现</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Formal data / frozen source</th><th>Source patch 的精确位置</th><th>Downstream mediation test</th></tr></thead><tbody>
    <tr><td><strong>Qwen3-8B</strong></td><td>Independent seeds 1294–1313；六个有向 pairs 1↔6、3↔8、5↔10，共 20×6=120 primary paired units。Early top-4 固定为 L27H18、L23H28、L23H29、L26H20。</td><td>在 receiver 的 full-prompt prefill 中，对每个 early head，只把所有 registered slot-token positions 的 pre-O z slices 替换为同 seed donor 的对应 slices；answer-query slice 与 non-slot positions 不 patch。各层只改一次，随后自然前传。</td><td>在 L28 answer query 读取 H16–H19 因 source patch 诱发的完整 pre-O Δz。Exact block 加 −Δz，使该 set 回到 receiver 值；control 在相同 H16–H19 W<sub>O</sub> span 内加入 post-O 等范数且与 induced output 正交的向量。比较 control 保留的 donor gain 与 exact block 保留的 donor gain。</td></tr>
    <tr><td><strong>Gemma4-E4B</strong></td><td>Discovery seeds 1630–1639 只在 L36–L40 中选择 mediator；formal seeds 1640–1659 使用三个有向 pairs (1→6,3→8,5→10)，每个 source set 20×3=60 units。Candidate K=2 固定为 L29H4+L35H2，并配三个同层 frozen control sets。</td><td>在 receiver 的 answer-query token 上，分别将 L29H4 与 L35H2 的 pre-O z slice 替换为同 seed donor 值；其他 heads/positions 不 patch。Source layers 均早于候选 residual mediator。</td><td>Formal mediator 固定为 post-block L37 answer-query residual；令 Δh=h<sub>source-patch</sub>−h<sub>receiver</sub>。Exact block 加 −Δh；matched arm 加与 Δh 等范数且正交的 residual vector；另有 count-axis block/control。L41 读取 terminal adoption。四个 source sets×五个 downstream conditions×60 units=1,200 formal rows。</td></tr>
  </tbody></table></div><p><span class="example"><strong>例：</strong>receiver count=3、donor count=8。Qwen patch 的不是输入文字，也不是最终答案，而是四个 early heads 在各 slot token 上、进入 W<sub>O</sub> 前的 donor z slices；如果 receiver 更偏向“8”，再在 L28 精确撤销这次 patch 诱发的 H16–H19 变化后该偏移明显消失，而等范数正交扰动没有同样效果，才把部分效应归到这段 mediator。Gemma 做同一逻辑，但 source 写在 answer-query 的 L29H4/L35H2，mediator 是 L37 的 distributed residual。</span></p></details>
  <div class="claim"><strong>Attention 图的解释边界。</strong>Broad heads 更像“对多个已出现 records/evidence 进行并行读取并写入 answer query”，而不是从 prompt 某一个位置读取一枚稳定整数。图 5 的亮区只说明某层某头对完整 active spans 具有较高 broad score；它不区分 span 内哪个 token 是地址或内容。当前机制不依赖这一细分：行为因果性由 matched head ablation 与 source→mediator experiments 提供。</div>

  <div class="claim"><strong>下面进入第二条子链：从 routing 集合转向 retrieved content。</strong><strong>4.3 只问“这些 broad heads 合起来写出的向量里，能不能看出 count？”</strong>这是 representation/decodability。<strong>4.4 再问“模型自然运行时是否真的依赖其中的 count-aligned 方向？”</strong>这是 matched intervention。前者高于 chance 不能替代后者。</div>

  <div class="step-heading"><span class="step-kicker">02 · Representation · content</span><h3>4.3 Broad-bank representation：这些 heads 合起来写出了什么？</h3></div>
  <p class="lead">把每个 broad head 想成一个并行“取证员”：attention 决定它从哪些 needle spans 取信息，value/output projection 把取回的内容写到 answer-query residual。我们在每个冻结层把这一组 heads 的<strong>实际 post-O writes 相加</strong>，得到一个 broad-bank state；随后只问能否从这个合计向量预测最终 count。这里没有删除、patch 或改变模型输出，所以本节仍是描述性 representation analysis。</p>
  <div class="experiment"><div class="experiment-label">Experiment 2C · representation</div><div><h4>Broad-bank post-O state 的 held-out count geometry</h4><p><strong>目的。</strong>确认 broad heads 不只是“看向多个 spans”，其合计写入也确实携带 final-count information。<strong>设置。</strong>在每个 frozen layer 对注册 heads 的实际 post-O writes 求和；basis/classifier 仅用 discovery seeds 1234–1253，confirmation seeds 1254–1263 只评估。Qwen 使用 Top-32，Gemma 按 global Top-6 的分层子集重构。<strong>结果。</strong>Exact accuracy 为 Qwen 39%–54%、Gemma {pct(min(float(retrieval_geometry_by_key[('Gemma4-E4B', layer)]['exact_classifier_accuracy']) for layer in (29, 35)))}–{pct(max(float(retrieval_geometry_by_key[('Gemma4-E4B', layer)]['exact_classifier_accuracy']) for layer in (29, 35)))}，高于 10% chance；centroid rank-3 capture 为 0.968–0.995，但 silhouette 约 −0.098…+0.014。<strong>分析与目前结论。</strong>平均 count trajectory 低维且可读，单样本 clouds 却高度重叠；这是候选 retrieved representation，不是因果证明。<span class="mini-example"><strong>直观例子：</strong>十个班级的平均身高可沿一条线有序排列，但班内学生身高仍大量重叠；“平均点低维”不等于“每个类别形成干净 cluster”。</span></p></div></div>
  <figure><h4 class="figure-title">图 6c · 4.3 实际测量的对象：从多头读取到一个合计写入向量</h4>
    <div class="path" aria-label="How the broad-bank representation is constructed">
      <div class="node"><strong>完整 needle spans</strong><small>计数证据分散在多个远端位置</small></div>
      <div class="node"><strong>Frozen broad heads</strong><small>每个 head 用 attention 读取多个 spans</small></div>
      <div class="node"><strong>每头 post-O write</strong><small>每个 head 真正加进 answer residual 的向量</small></div>
      <div class="node"><strong>合计 state w<sub>ℓ</sub>(q)</strong><small>把 frozen bank 的 writes 在同层相加</small></div>
      <div class="node"><strong>Held-out count readout</strong><small>discovery 拟合，confirmation 评估</small></div>
    </div>
    <figcaption>这是计算流程示意图，不是额外实验结果。被分类的对象不是 raw attention map，也不是完整 answer hidden state，而是指定层、指定 frozen broad-head bank 在 answer query 上实际写入 residual 的向量之和。若该向量能预测 count，只说明聚合输出携带 count information。</figcaption>
  </figure>
  <figure><h4 class="figure-title">图 6d · Broad-bank 合计输出的 held-out exact-count readout</h4>
    <div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_geometry_charts[0]}</div><div><h4>Gemma4-E4B</h4>{retrieval_geometry_charts[1]}</div></div>
    <figcaption>横轴是冻结的 broad-bank layer；纵轴是 confirmation seeds 1254–1263 上的十类 count accuracy，0.50 即 50%。模型色实线为 regularized exact-count classifier，灰色虚线为 nearest-centroid；水平虚线 0.10 是十类 chance。Qwen 沿用 frozen Top-32，Gemma L29/L35 由 exact Top-6 的分层 post-O writes 重构。折线只连接实际测试层，不能据此推断未测层趋势。</figcaption>
  </figure>
  <p><strong>怎么读图 6d。</strong>Qwen 在 L21/23/24/26/27 的 exact accuracy 为 49%/54%/39%/44%/39%；Gemma Top-6 在 L29/L35 为 {pct(retrieval_geometry_by_key[('Gemma4-E4B', 29)]['exact_classifier_accuracy'])}/{pct(retrieval_geometry_by_key[('Gemma4-E4B', 35)]['exact_classifier_accuracy'])}。全体层的 exact/nearest-centroid accuracy 范围分别为 {pct(min(retrieval_exact_values))}–{pct(max(retrieval_exact_values))} 与 {pct(min(retrieval_nearest_values))}–{pct(max(retrieval_nearest_values))}，classifier MAD 为 {f(min(retrieval_mad_values))}–{f(max(retrieval_mad_values))} counts。十个 count centroids 的前三维方差占比很高（0.968–0.995），但单样本 cosine silhouette 接近 0（−0.098…+0.014），所以更准确的图像是“十个平均点沿低维轨迹移动，但每个 count 周围的样本云彼此重叠”，而不是十个干净分开的 clusters。</p>
  <details class="collapsible-list"><summary>展开：4.3 的完整公式、稳定性指标与逐层数值</summary>
    <div class="formula"><strong>Broad-bank state。</strong>对 layer <span class="math">ℓ</span> 的 frozen head bank <span class="math">𝒮_ℓ</span>，定义 <span class="math">w_ℓ(q)=Σ_{{h∈𝒮_ℓ}}W_O^hΣ_jα_h(q,j)W_V^hh_j</span>。Canonical dense run 保存了 3,000 个 clean broad-bank states；basis 与 classifier 只用 discovery seeds 1234–1253 拟合，所有报告 prediction 都在 confirmation seeds 1254–1263 上计算。</div>
    <div class="formula"><strong>Retrieval-geometry readouts。</strong>Exact classifier 是训练折内 StandardScaler/PCA 后的十类线性分类器；nearest centroid 将 held-out state 分配给欧氏距离最近的 discovery count centroid；classifier MAD=<span class="math">mean|ĉ−c|</span>。Rank-3 centroid capture 衡量十个 count centroids 的前三奇异方向方差占比；cosine silhouette 衡量单样本同 count 是否比异 count 更近；bootstrap maximum principal angle 比较 seed-resampled rank-3 basis 与 full-discovery basis，角度越大表示估计的 subspace 越不稳定。<span class="example">例：counts [4,8] 被预测为 [5,6] 时，accuracy=0、MAD=(1+2)/2=1.5；centroid rank-3 capture=0.99 但 silhouette=0，表示 mean curve 近三维，却没有干净的 sample clusters。</span></div>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th>Frozen layer</th><th>Exact classifier acc.</th><th>Nearest-centroid acc.</th><th>Classifier MAD</th></tr></thead><tbody>{retrieval_geometry_rows}</tbody></table></div>
    <p>Seed-bootstrap 95th-percentile maximum principal angle 约 60°–87°，进一步说明 fitted rank-3 basis 会随 discovery seeds 明显变化；它不是一枚跨样本完全固定的三维寄存器。</p>
  </details>
  <div class="conclusion-line"><strong>4.3 Representation 目前结论。</strong>Broad-head bank 的合计 post-O write 中确实存在 noisy、低秩、可解码的 final-count geometry；classifier 只衡量“含不含、紧不紧”，不能证明模型自然生成答案时使用了这些方向。这个因果问题交给 4.4。</div>

  <div class="step-heading"><span class="step-kicker">03 · Causal test · content</span><h3>4.4 Retrieval-subspace matched removal：正常运行依赖与上游修复中介</h3></div>
  <div class="claim"><strong>核心判定逻辑。</strong>4.3 只证明 broad heads 的合计写入中可解码最终 count；4.4 进一步删除其中与 count 对齐的三维分量，检验输出误差是否增加。为排除“删除任意同等大小的 hidden-state 分量都会损伤模型”这一解释，matched control 在<strong>同一层、同一 answer-query output span 删除实际范数完全相同、但与 count basis 正交的分量</strong>。只有 count-aligned removal 相对该 control 产生额外损伤，才构成模型自然使用该 subspace 的证据。</div>
  <div class="experiment"><div class="experiment-label">Restored upstream state · exact construction</div><div><h4>“恢复完整 needle spans”具体做了什么</h4><p><strong>目的。</strong>构造一个“上游 needle evidence 已被救回”的 receiver，再检验该收益是否会经过后续 frozen retrieval component。<strong>第一步：clean donor。</strong>对同一个 seed–count 的原始 prompt 做 clean forward，保存 post-block L8 上所有 active needle spans 内、每一个 token position 的 residual state。<strong>第二步：corrupt receiver。</strong>将输入中每段 active needle 从首 token 到末 token，替换为同一 prompt 中一段 token 数完全相同的 ordinary passage；ordinary source 与所有 slot/hard-negative spans 不重叠，因此总序列长度、各 span 的绝对位置和 answer-query 位置均不改变。<strong>第三步：单次内部恢复。</strong>corrupt forward 运行完 block L8 后，只把 needle-position states 换成 clean donor 在同层、同位置的 states；answer-query state 与全部 non-needle positions 保留 corrupt run 自己的值。L8 的 attention 此时已经计算完，所以恢复后的 evidence 只能从 L9 开始影响后续计算。之后不再 clamp 或重复 patch，模型自由运行到预冻结 retrieval layer，再执行 aligned/orthogonal removal。</p></div></div>
  <div class="formula"><strong>一次性 full-span restoration operator。</strong>令 <span class="math">S<sub>needle</sub></span> 为当前 prompt 中所有 active needle spans 的 token-position 并集。对 post-block L8 state，定义 <span class="math">H<sup>restored</sup><sub>8,p</sub>=H<sup>clean</sup><sub>8,p</sub></span>（若 <span class="math">p∈S<sub>needle</sub></span>），否则 <span class="math">H<sup>restored</sup><sub>8,p</sub>=H<sup>corrupt</sup><sub>8,p</sub></span>；再令 layers L9 以后从这组混合 states 正常计算。这里恢复的是模型对原 span 已形成的<strong>联合内部表征</strong>，不是把输入 token 文本改回去，也不是直接写入 answer query。<span class="example"><strong>例：</strong>gold count=3，三段 active needles 分别占 6、5、7 个 tokens。实验先用 ordinary text 覆盖这 18 个输入位置；到 post-block L8 时，仅把这 18 个 residual vectors 换回 clean values。其余数千个 prompt positions 和 answer-query vector 都不动，随后从 L9 继续运行。若后续 retrieval/output 得到修复，说明恢复的 span evidence 仍能被下游重新利用。</span></div>
  <div class="path" aria-label="Full-span restoration used in retrieval-subspace mediation"><div class="node"><strong>Clean donor</strong><small>保存 post-block L8 的全 span states</small></div><div class="node"><strong>Corrupt receiver</strong><small>等长 ordinary text 覆盖输入 needles</small></div><div class="node"><strong>只 patch span positions</strong><small>不 patch answer query；只执行一次</small></div><div class="node"><strong>L9+ 自由运行</strong><small>不持续 clamp</small></div><div class="node"><strong>Frozen retrieval layer</strong><small>aligned vs orthogonal removal</small></div></div>
  <div class="claim boundary"><strong>两个术语的准确含义。</strong><strong>Natural-use</strong> 中的 “natural” 只表示<strong>未经 corruption 或上游 restoration 的正常模型 forward</strong>，不是“自然语言”的意思；这里译为<strong>正常运行依赖效应</strong>。<strong>Restoration-mediation</strong> 表示先在早层恢复 clean needle-span evidence 后，候选 retrieval component 是否承接了部分修复收益；这里译为<strong>上游修复中介效应</strong>。“中介”不表示唯一通路，也不自动等于被解释比例。</div>
  <div class="result-grid">
    <div class="result"><span class="value">正常运行依赖</span><span class="label"><strong>Natural-use / natural specificity</strong><br>在正常 forward 中比较 aligned 与 orthogonal removal。正值表示模型平时确实依赖 count-aligned component。</span></div>
    <div class="result"><span class="value">上游修复中介</span><span class="label"><strong>Restoration-mediation</strong><br>先恢复 clean needle spans，再做同一 removal contrast。正值表示恢复收益有一部分经该 component 传递。</span></div>
  </div>
  <div class="table-wrap"><table><thead><tr><th>中文名称（英文）</th><th>从什么状态开始</th><th>比较哪两个条件</th><th>正值说明什么</th></tr></thead><tbody>
    <tr><td><strong>正常运行依赖效应</strong><br><small>Natural-use / natural specificity</small></td><td>未经上游修复的正常 clean forward</td><td>正常状态下：count-aligned removal 的误差 − equal-norm orthogonal removal 的误差</td><td>模型正常作答时，对该 count-aligned component 存在方向特异依赖。</td></tr>
    <tr><td><strong>上游修复中介效应</strong><br><small>Restoration-mediation</small></td><td>Corrupt run 中先恢复 clean full-needle-span states</td><td>Restored 状态下：count-aligned removal 的误差 − equal-norm orthogonal removal 的误差</td><td>早层 span restoration 产生的收益，有一部分经该 retrieval component 传递。</td></tr>
  </tbody></table></div>
  <div class="formula"><strong>两个量只差“从什么状态开始”。</strong>令 <span class="math">e(X)=|E[c]_X−N|</span>。正常运行依赖效应为 <span class="math">S_{{normal}}=e(normal+aligned)−e(normal+orthogonal)</span>；上游修复中介效应为 <span class="math">M_{{restore}}=e(restored+aligned)−e(restored+orthogonal)</span>。两者都先控制删除位置与实际 norm，再问 aligned removal 是否额外增加误差。<span class="example"><strong>同一个 gold=8 例子：</strong>正常状态下 aligned/orthogonal 两臂的 <span class="math">E[c]=6.5/7.5</span>，误差 1.5/0.5，所以正常运行依赖效应=1.0 count。若先恢复 spans 后，两臂误差为 1.2/0.8，则上游修复中介效应=0.4 count：恢复收益中至少有一部分会经过该 component。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 2D · causal</div><div><h4>Count-aligned removal versus equal-norm orthogonal control</h4><p><strong>目的。</strong>同时估计正常运行依赖效应与上游修复中介效应。<strong>Count-aligned component 的定义。</strong>它是仅在 discovery seeds 1234–1253、counts 1–10 上，从同层 frozen broad-head 合计 post-O write 中拟合的 final-count centroid rank-3 basis；confirmation seeds 1254–1263 不参与 basis、layer 或 head selection。Qwen 使用 Top-32 的分层 head bank；Gemma 使用 global Top-6 在该层的精确子集（L29: H4/H2；L35: H2/H7/H1/H3）。<strong>设置。</strong>每个 confirmation seed–count–layer 包含 clean/restored × aligned/orthogonal 四个配对条件；其中 restored 固定使用上文定义的 post-block L8、full-span、single-patch receiver。Aligned arm 从该层 answer-query attention output 中减去 broad-bank centered write 在 frozen rank-3 basis 上的投影；orthogonal arm 在同一 output position 减去与 rank-3 basis 正交、且按该样本 aligned projection 的实际范数缩放的 deterministic control vector。<strong>结果。</strong>Qwen L23 的正常运行依赖/上游修复中介效应为 +{f(qwen_retrieval_l23['natural_specificity_mean'])}/+{f(qwen_retrieval_l23['restoration_mediation_mean'])} counts；Gemma L29 为 +{f(gemma_retrieval_l29['natural_specificity_mean'])}/+{f(gemma_retrieval_l29['restoration_mediation_mean'])}；更晚 frozen layers 约为 0。<strong>分析与目前结论。</strong>两个模型都在一个较窄的 retrieval window 自然使用该 component，且 Stage I 恢复的 span evidence 有一部分经此传递；实验支持一条部分路径，不建立通路唯一性。</p></div></div>
  <figure><h4 class="figure-title">图 6e · 正常运行依赖与上游修复中介的四条件配对设计</h4>
    <div class="contrast-grid" aria-label="Four-condition retrieval-subspace intervention design">
      <div class="contrast-lane">
        <div class="contrast-source">正常运行依赖效应<small>Natural-use / natural specificity；未经上游修复的正常 forward</small></div>
        <div class="contrast-branches">
          <div class="contrast-arm"><strong>A · Count-aligned removal</strong><small>删除 frozen final-count rank-3 component</small></div>
          <div class="contrast-arm"><strong>B · Orthogonal control removal</strong><small>同层、同位置、同实际 norm；方向与 rank-3 basis 正交</small></div>
        </div>
        <div class="contrast-result"><strong>正常运行依赖效应</strong> = error(A) − error(B)<br>正值：正常计算对 count-aligned component 具有方向特异依赖。</div>
      </div>
      <div class="contrast-lane restored">
        <div class="contrast-source">上游修复中介效应<small>Restoration-mediation；先在 corrupt run 中恢复 clean full-span states</small></div>
        <div class="contrast-branches">
          <div class="contrast-arm"><strong>C · Count-aligned removal</strong><small>删除 restored state 中的 frozen rank-3 component</small></div>
          <div class="contrast-arm"><strong>D · Orthogonal control removal</strong><small>其余设置与 C 完全匹配</small></div>
        </div>
        <div class="contrast-result"><strong>上游修复中介效应</strong> = error(C) − error(D)<br>正值：source-restoration gain 有一部分经该 component 传递。</div>
      </div>
    </div>
    <figcaption>四个条件在同一个 seed、gold count 和 layer 内配对，因此比较的是删除方向，而不是不同 prompt 的难度。左半边估计正常运行依赖效应；右半边估计上游修复中介效应。两条差值的单位都是 expected-count absolute error 的 counts；0.5 表示 aligned removal 比 control 平均多造成半个 count 的误差，不是 50% mediation，也不是 classifier accuracy。</figcaption>
  </figure>
  <p><strong>图 6f 的判读规则。</strong>每个点均为 count-aligned removal 相对 equal-norm orthogonal removal 的 paired error contrast。模型色实线报告<strong>正常运行依赖效应</strong>；紫色虚线报告<strong>上游修复中介效应</strong>。纵轴大于 0 表示存在注册方向上的特异作用；等于 0 表示当前层未检测到该 frozen basis 的特异作用。效应量的单位是 count error，不是百分比；0 也不等于 count information 消失。</p>
  <figure><h4 class="figure-title">图 6f · Frozen retrieval rank-3 subspace 的逐层依赖与中介效应</h4>
    <div class="figure-stack"><div><h4>Qwen3-8B</h4>{retrieval_subspace_charts[0]}</div><div><h4>Gemma4-E4B</h4>{retrieval_subspace_charts[1]}</div></div>
    <figcaption>横轴是预先冻结的 zero-based intervention layer，纵轴是 aligned removal 相对 equal-realized-norm orthogonal removal 多造成的 expected-count absolute error，单位 counts。Qwen 在 L21–L23 为正、从 L24 起约为 0；Gemma 在 L29 明显为正、L35 回到约 0。曲线定位的是 retrieval-basis 的自然使用窗口，不是完整 count information 的寿命。只在图示 frozen layers 做了 intervention；Gemma L29 与 L35 之间的连线仅连接两个观测点，不代表测过 L30–L34 或证明线性下降。</figcaption>
  </figure>
  <p><strong>逐层结果。</strong>Qwen L21 已出现正向效应，L23 最强：正常运行依赖效应为 {f(qwen_retrieval_l23['natural_specificity_mean'])} count，上游修复中介效应为 {f(qwen_retrieval_l23['restoration_mediation_mean'])} count；L24/L26/L27 约为 0。Gemma Top-6 的正向证据集中在 L29：两个效应分别为 {f(gemma_retrieval_l29['natural_specificity_mean'])}/{f(gemma_retrieval_l29['restoration_mediation_mean'])}；L35 为 {f(gemma_retrieval_l35['natural_specificity_mean'])}/{f(gemma_retrieval_l35['restoration_mediation_mean'])}。因此结果支持：<strong>Qwen L21–L23、Gemma L29 附近存在该 frozen retrieval subspace 的方向特异使用与部分中介；更晚层未复现同一 basis 的作用。</strong></p>
  <details class="collapsible-list"><summary>展开：4.4 的层、样本量、完整公式、逐层表与 clean-correct robustness</summary>
    <div class="table-wrap"><table><thead><tr><th>Design item</th><th>Frozen setting</th></tr></thead><tbody>
      <tr><td>Basis fit / held-out data</td><td>Basis 只用 discovery seeds 1234–1253 × counts 1–10=200 clean states/layer 拟合；因果检验只用 confirmation seeds 1254–1263 × counts 1–10=100 paired units/layer。模型间不 pooled。</td></tr>
      <tr><td>Frozen layers</td><td>Qwen L21/L23/L24/L26/L27；Gemma L29/L35。层在 confirmation outcome 之前冻结。</td></tr>
      <tr><td>Frozen head membership per tested layer</td><td>Qwen：L21={{H11,H16,H18,H19,H20,H23,H25,H27,H31}}；L23={{H10,H12,H13,H28,H29,H30,H31}}；L24={{H13,H14,H16,H29,H31}}；L26={{H20,H21,H24,H26}}；L27={{H16,H18,H29}}。Gemma Top-6：L29={{H4,H2}}；L35={{H2,H7,H1,H3}}。</td></tr>
      <tr><td>Source restoration</td><td>Restored arms 先在 post-block L8 只 patch 全部 active needle-token residual states；clean arms 不做 source patch。两种状态之后均自由前传至 retrieval layer。</td></tr>
      <tr><td>Aligned removal</td><td>在 retrieval layer 的 full-prompt prefill，先由上述 heads 的实际 pre-O z 经各自 W<sub>O</sub> 求和得到 bank write；减去 discovery mean 后投影到 rank-3 basis，并从该层 answer-query attention post-O output 中减去该投影。</td></tr>
      <tr><td>Orthogonal control</td><td>在相同 layer、相同 answer-query post-O output position，减去位于该 bank W<sub>O</sub> span、与 rank-3 basis 正交的固定方向；每个样本把它缩放到与 aligned projection 完全相同的 realized L2 norm。它不是“另一个 rank-3 basis”。</td></tr>
      <tr><td>Four conditions</td><td>clean+aligned、clean+orthogonal、restored+aligned、restored+orthogonal。Candidate-sequence expected count 与最多 8-token strict generation 分别执行一个 prefill，因此每行 source/removal hook 的审计 application count 均为 2。</td></tr>
      <tr><td>Coverage audit</td><td>100 units×4 conditions=400 unique rows/layer；7 layers 共 2,800。每层 paired keys、hook applications、removed-norm equality 与 orthogonal leakage audits 均 PASS。</td></tr>
    </tbody></table></div>
    <div class="formula"><strong>正常运行依赖效应与上游修复中介效应。</strong>Gold count 为 <span class="math">N</span> 时，正常运行依赖效应 <span class="math">S_{{normal}}=|E_{{normal+aligned}}−N|−|E_{{normal+orth}}−N|</span>；在 full-span restoration 后，上游修复中介效应 <span class="math">M_{{restore}}=|E_{{restored+aligned}}−N|−|E_{{restored+orth}}−N|</span>。<span class="example">例：gold N=8；aligned removal 后 E[c]=6.5、orthogonal removal 后 7.5，则正常运行依赖效应=|6.5−8|−|7.5−8|=1 count。</span></div>
    <div class="formula"><strong>Mediated fraction。</strong>对同一 seed-count 单元，以未 block 的 full-span expected-error repair <span class="math">A_{{repair}}</span> 为分母，定义 <span class="math">F_{{med}}=M_{{restore}}/A_{{repair}}</span>。它不裁剪、不是概率；分母很小或为负时可超出 [0,1]，所以 mean 与 median 只作量级描述。<span class="example">例：full-span restoration 原本修复 2 counts，aligned-specific block 额外损失 0.5 count，则 fraction=0.5/2=0.25；不能把它读成“25% 的 heads”。</span></div>
    <div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>正常运行依赖效应 mean</th><th>上游修复中介效应 mean</th><th>Mediated fraction mean</th></tr></thead><tbody>{retrieval_subspace_rows}</tbody></table></div>
    <p>Qwen L23 的 natural/restored medians 为 {f(qwen_retrieval_l23['natural_specificity_median'])}/{f(qwen_retrieval_l23['restoration_mediation_median'])}，fraction median={f(qwen_retrieval_l23['mediated_fraction_median'])}；Gemma Top-6 L29 对应为 {f(gemma_retrieval_l29['natural_specificity_median'])}/{f(gemma_retrieval_l29['restoration_mediation_median'])}、{f(gemma_retrieval_l29['mediated_fraction_median'])}。Clean-correct robustness 子集有 Qwen {int(qwen_retrieval_l23_clean['paired_units'])}、Gemma {int(gemma_retrieval_l29_clean['paired_units'])} units：Qwen L23 restoration mediation +{f(qwen_retrieval_l23_clean['restoration_mediation_mean'])}、mean fraction {f(qwen_retrieval_l23_clean['mediated_fraction_mean'])}；Gemma L29 +{f(gemma_retrieval_l29_clean['restoration_mediation_mean'])}、{f(gemma_retrieval_l29_clean['mediated_fraction_mean'])}。Clean-correct 是条件化 robustness，不替代 100-unit primary population。</p>
  </details>
  <div class="claim"><strong>4.4 与 Stage-II 的因果结论。</strong>
    <p><strong>正常运行依赖：</strong>正常 forward 中，count-aligned removal 比 equal-norm orthogonal removal 产生更大的行为损伤，因此 4.3 的可解码 geometry 不只是 classifier 所捕获的伴随相关。</p>
    <p><strong>上游修复的部分中介：</strong>在 full-span restoration 后，同一 aligned-versus-orthogonal contrast 特异地削弱 restoration gain，支持 span evidence→broad-bank output 的部分中介关系。</p>
    <p><strong>Scope：</strong>效应只中介一部分 source repair，并在更晚层对同一 frozen basis 回到约 0；结合 Stage III 的晚层 patch/removal 结果，更合理的解释是 representation 被重新参数化，而不是 count information 消失。故本节支持<strong>Qwen L21–L23、Gemma L29 附近的一条自然使用、但非唯一且非穷尽的 aggregation pathway</strong>。</p>
  </div>
  </details>
</section>

<section id="write">
  <h2>Step 3 · Consolidate — 晚层形成可执行的 answer-side count state</h2>
  <p class="lead">Section 2 已经表明 final count 在晚层 answer query 可由 exact-count classifiers 读取，但 decodability 本身不证明模型依赖该状态。本阶段因此组合四类互补证据：完整 donor-state patch 检验充分性，rank-3 removal 检验方向特异的必要性，相邻层三维 map 描述 centroid geometry 的可靠性，aligned 1× intervention 检验单个 block 对 count-aligned change 的选择性传播。</p>
  <div class="claim"><strong>Stage-III hypothesis。</strong>Broad retrieval 之后，模型在 answer-query residual 中逐渐形成一个可直接控制输出的 consolidated count state。若该假说成立，完整 donor state 应在中后层诱发 donor-answer adoption；删除冻结的 count-aligned component 应比等范数正交删除更伤答案；一个沿 count chord 的局部扰动应被下一 block 选择性接收。</div>
  <div class="chain-blueprint" aria-label="Chain C mechanism representation causal test">
    <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>判断 retrieval 之后的 answer-query state 何时从“可以读出 count”升级为“足以决定输出、且模型自然依赖”的执行状态。<span class="mini-example"><strong>直观例子：</strong>草稿纸上写着“8”只说明信息存在；把这张草稿交给另一道题能让它答 8，且擦掉“8”对应方向比擦掉同样多的无关笔画更伤，才说明它是可执行状态。</span></p></div>
    <div class="evidence-triad">
      <div class="triad-step"><span class="protocol-no">01 · Mechanism</span><h3>聚合结果被整合为 answer state</h3><p>预测：中后层 answer residual 会获得直接控制首数字生成的能力。</p></div>
      <div class="triad-step"><span class="protocol-no">02 · Representation</span><h3>Late geometry 变得稳定可读</h3><p>Exact-count readout 上升；相邻层 centroid map 在晚层更可预测，但 basis 不必固定。</p></div>
      <div class="triad-step"><span class="protocol-no">03 · Causal test</span><h3>充分性、必要性与传递分开检验</h3><p>Full donor patch 测 sufficiency；aligned-vs-orthogonal removal 测 natural use；one-block transport 测局部接收。</p></div>
    </div>
  </div>

  <div class="step-heading"><span class="step-kicker">02 · Representation · final state</span><h3>5.1 Answer query 的 final-count geometry 在中后层增强</h3></div>
  <p>图 2b 与图 3 已显示 answer-query exact-count accuracy 随层上升；默认 Qwen L28 / Gemma L37 只是 display-only 三维层。这里的 representation prediction 是“晚层更容易读取 final count”，而不是“classifier 选出的层就是 causal peak”。</p>
  <div class="step-heading"><span class="step-kicker">03 · Causal test · final state</span><h3>5.2 完整状态充分性与 count-aligned 方向必要性</h3></div>

  <div class="formula"><strong>Full-state donor patch。</strong>对 receiver prompt R 与 donor prompt D，在层 <span class="math">ℓ</span> 的 answer-query 位置 <span class="math">q</span> 执行 <span class="math">h^R_ℓ(q)←h^D_ℓ(q)</span>，其余 receiver states 和 tokens 不变，再从 receiver 的下一步计算继续生成。Dense layerwise 指标只在 clean donor prediction 与 clean receiver prediction 不同的 eligible pairs 上定义 adoption=<span class="math">𝟙[ŷ_{{patch}}=ŷ_D]</span>。Correct-only pooled patching accuracy 则用更严格的 <span class="math">𝟙[ŷ_{{patch}}=N_D]</span>，并要求 receiver/donor clean 均答对。<span class="example">例：receiver clean 输出 5，donor clean/gold 都为 8；patched 输出 8，则 donor-prediction adoption=1 且 strict donor-gold hit=1；若 patched 输出 7，两者均为 0。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3A · causal sufficiency</div><div><h4>Full answer-state donor patch：何时能接管 receiver 的答案？</h4><p><strong>目的。</strong>检验完整 answer-query residual 是否足以把 receiver 输出推向 donor count。<strong>设置。</strong>只替换一个 layer 的 answer-query post-block residual，其他 receiver tokens/states 不变，然后继续 receiver forward；同时保留 self-patch、same-count-seed 与 correct-only controls。<strong>结果。</strong>Dense adoption 在早层近 0，中后层陡升：Qwen L26 53.3%、L29 98.3%；Gemma L31 87.5%、L35 98.8%。Correct-only strict donor-gold hit 为 Qwen {pct(answer_patching['Qwen3-8B']['pooled_average_patching_acc'])}、Gemma {pct(answer_patching['Gemma4-E4B']['pooled_average_patching_acc'])}。<strong>分析与目前结论。</strong>完整 answer state 的充分性在中后层形成并接近饱和；这不等于其中任意一个线性方向都必要。</p><div class="table-wrap"><table><thead><tr><th>Patch object</th><th>Donor → receiver</th><th>Position / layer protocol</th><th>Readout</th></tr></thead><tbody>
    <tr><td>Prompt endpoint / full span residual</td><td>Gold count <em>N</em><sub>D</sub> prompt → different-count <em>N</em><sub>R</sub> prompt</td><td>改变的 k∈{{1,3,5}} 个 nested slots；needle-end 或整个 span；single layer 或从 start layer 累积 clamp 到最后层</td><td>strict patched count=donor gold；pooled Qwen {prompt_patching['Qwen3-8B']['patching_acc_successes']}/{prompt_patching['Qwen3-8B']['patching_acc_denominator']}={pct(prompt_patching['Qwen3-8B']['pooled_average_patching_acc'])}，Gemma {prompt_patching['Gemma4-E4B']['patching_acc_successes']}/{prompt_patching['Gemma4-E4B']['patching_acc_denominator']}={pct(prompt_patching['Gemma4-E4B']['pooled_average_patching_acc'])}。这是上游信息充分性，不与 rank-3 removal 的局部必要性矛盾。</td></tr>
    <tr><td>Full answer-query residual</td><td>Donor answer state → receiver 同层 <code>Total:</code> query</td><td>single-layer dense sweep；另有 frozen single/cumulative protocols 与 self-patch、same-count-seed controls</td><td>dense donor-prediction adoption；correct-only strict donor-gold hit。</td></tr>
    <tr><td>Broad-bank source / pre-O state</td><td>Donor slot-query state or donor pre-O z → receiver registered slice</td><td>Qwen early top-4 → L28 H16–H19；Gemma L29H4/L35H2 → L37 residual</td><td>donor candidate log-odds / normalized transport；再用 exact induced-component block 做 mediation。</td></tr>
    <tr><td>Count-aligned component</td><td>Receiver state + one frozen receiver→donor displacement</td><td>Qwen L28→29；Gemma L36→37；主图比较 aligned 1× 与 actual-norm-matched orthogonal 1×，完整 panel 另含 aligned 2× dose check</td><td>下一层相对 clean state 的 target-chord propagation coefficient F。</td></tr>
  </tbody></table></div></div></div>
  <figure><h4 class="figure-title">图 7 · Answer state 的因果可执行性在中后层出现</h4>{patch_chart}<figcaption>横轴是被替换的 single layer，纵轴是 eligible donor-prediction adoption rate。每点将 donor 的完整 post-block answer-query residual 写入 receiver 同层 query，随后继续 receiver forward pass。早层近 0，中后层陡升并接近 1；这证明完整 state 的充分性，不等同于某一个线性 count direction 的必要性。</figcaption></figure>

  <div class="formula"><strong>Answer-query absolute-error specificity。</strong>在每层只对 answer query state 删除相对全局 center 的 frozen count rank-3 projection；control 在正交 rank-3 basis 上删除相同实际 norm。定义 <span class="math">S<sub>abs</sub>(ℓ)=|ŷ<sub>count-remove,ℓ</sub>−N|−|ŷ<sub>orth-remove,ℓ</sub>−N|</span>，等价于两者相对 clean 的 absolute-error increase 之差。<span class="example">例：gold N=8，count removal 输出 5（error 3），orthogonal removal 输出 7（error 1），则 S<sub>abs</sub>=3−1=2 counts。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3B · causal necessity</div><div><h4>Layerwise answer-query rank-3 removal</h4><p><strong>目的。</strong>检验可解码的 late count directions 是否被自然生成过程依赖。<strong>设置。</strong>每层只在 answer query 删除 frozen count rank-3；control 删除同位置、同 rank、actual-norm-matched 的 orthogonal component。<strong>结果。</strong>峰值 absolute-error specificity 为 Qwen L28 +0.878 counts（0.556–1.244）和 Gemma L32 +1.222（1.044–1.411），且中后层整体强于浅层。<strong>分析与目前结论。</strong>晚层 count-aligned geometry 不只是 classifier 可读的 trace；matched control 下它具有局部方向特异必要性。</p></div></div>
  <figure><h4 class="figure-title">图 8 · Answer-query count subspace 的必要性随深度增加</h4>{removal_chart}<figcaption>横轴是层，纵轴是 count rank-3 removal 相对 actual-norm-matched orthogonal removal 增加的 absolute error，单位 counts；0 表示无方向特异性。共同模式是中后层强于浅层，也强于 prompt endpoint 的逐层 rank-3 removal。</figcaption></figure>
  <div class="conclusion-line"><strong>Step 3 正文结论。</strong>晚层 answer-query state 同时满足两种更强的因果标准：把 donor 的完整 state 写给 receiver，可在中后层几乎完全接管答案；删除其中 frozen count-aligned component，又比删除等范数正交 component 更伤输出。因此这里不只是“classifier 看得出 count”，而是一个既具充分性、又具有方向特异必要性的可执行 state。相邻层坐标变换和局部 transport 作为实现诊断保留在 Appendix。</div>

  <details class="paper-appendix"><summary>Appendix C1 · 相邻层 geometry maps 与 one-block transport</summary>
  <div class="chain-blueprint" aria-label="Chain C cross-layer transformation subchain">
    <div class="chain-purpose"><span class="step-kicker">Subchain C2</span><p><strong>目的。</strong>回答“晚层不是原样复制同一个 counter，那 count-aligned change 如何跨 block 延续？”先描述相邻层 centroid relation，再用方向注入测试下一 block 是否选择性接收。<span class="mini-example"><strong>直观例子：</strong>摄氏与华氏都表示温度，但坐标不同；先拟合两套坐标之间的 map，再把一个摄氏方向扰动送过转换器，看它是否在华氏方向出现。</span></p></div>
  </div>
  <div class="step-heading"><span class="step-kicker">02 · Representation · transformation</span><h3>5.3 相邻层 geometry 是否可预测、可复现、方向连续？</h3></div>
  <div class="formula"><strong>相邻层的三维 centroid-coordinate map。</strong>在每个 answer-query boundary <span class="math">ℓ→ℓ+1</span>，只用 20 个 discovery seeds 分别计算 counts 1–10 的 centroids <span class="math">C_ℓ,C_{{ℓ+1}}</span>，并分别拟合 rank-3 centroid-PCA bases <span class="math">U_ℓ,U_{{ℓ+1}}</span>。中心化三维坐标为 <span class="math">Z_ℓ=(C_ℓ−C̄_ℓ)U_ℓ</span>，再拟合 ridge map <span class="math">Â_ℓ=argmin_A ‖Z_{{ℓ+1}}−Z_ℓA‖²_F+λ‖A‖²_F</span>。这只是对相邻层 centroid geometry 的局部三维坐标映射，不假设整个 manifold 全局线性，也不要求所有层共享同一个 <span class="math">A</span>。</div>
  <div class="formula"><strong>图 9 左列：两个 error 的定义与意义。</strong>第一，5-fold seed-held-out centroid normalized RMSE 为 <span class="math">E^{{CV}}_ℓ=[Σ_f‖Z^{{test}}_{{ℓ+1,f}}−Z^{{test}}_{{ℓ,f}}Â_{{ℓ,f}}‖²_F/Σ_f‖Z^{{test}}_{{ℓ+1,f}}−Z̄^{{test}}_{{ℓ+1,f}}‖²_F]^{{1/2}}</span>，并满足 <span class="math">R²_{{CV}}=1−(E^{{CV}}_ℓ)²</span>。它问的是：只在训练 seeds 上拟合的局部三维 map，能否预测未见 seeds 的下一层 count-centroid coordinates；<span class="math">E^{{CV}}=0</span> 为完美预测，<span class="math">E^{{CV}}=1</span> 表示 residual energy 与“只预测该 test fold 的 target mean”相同。第二，对 20 个 discovery seeds 有放回重采样 500 次，每次重拟合两端 PCA bases 与 map；用 orthogonal Procrustes 把两端 PCA gauges 对齐到 full-discovery gauges 后，定义 <span class="math">E^{{boot}}_ℓ=median_b ‖Ã^{{(b)}}_ℓ−Â_ℓ‖_F/‖Â_ℓ‖_F</span>。它问的是：换一批 discovery seeds 后，估计出的 map 参数是否可复现；0 表示对齐后的重拟合 map 完全相同。两者都是无量纲的 geometry-estimation error，<strong>都不是</strong>生成数字的 absolute count error、分类错误率或 intervention 的行为效应。浅绿色 boundary 同时满足 <span class="math">R²_{{CV}}≥0.90</span>（等价于 <span class="math">E^{{CV}}≤√0.1≈0.316</span>）与 <span class="math">E^{{boot}}≤0.10</span>。<span class="example">例：若 held-out target centered energy=100、map residual energy=4，则 E<sup>CV</sup>=√(4/100)=0.20、R²=0.96；若 ‖Â‖<sub>F</sub>=10，而一次 gauge-aligned bootstrap map 与它的 Frobenius distance 为 0.5，则该次 relative map error=0.5/10=0.05。</span></div>
  <div class="formula"><strong>图 9 右列：跨 boundary 的 full-operator cosine。</strong>raw PCA axes 在不同层可任意翻转或旋转，因此先把三维坐标 map 重建为 ambient low-rank operator <span class="math">T_ℓ=U_ℓÂ_ℓU^⊤_{{ℓ+1}}</span>，再定义相邻两个 maps 的 Frobenius cosine：<span class="math">C^{{next}}_ℓ=⟨T_ℓ,T_{{ℓ+1}}⟩_F/(‖T_ℓ‖_F‖T_{{ℓ+1}}‖_F)</span>。它衡量连续两个 boundary 的映射<strong>方向与结构</strong>是否一致：1 表示同向且只允许整体正比例缩放，0 表示 Frobenius-orthogonal，−1 表示完全反向。这里用 cosine 而不用 drift，是因为问题关心三维映射方向能否跨层延续，不希望把整体增益变化混进来；它同样不衡量最终计数行为。<span class="example">例：若 <span class="math">T_{{ℓ+1}}=2T_ℓ</span>，两者 scale 不同但方向完全相同，因此 <span class="math">C^{{next}}_ℓ=1</span>；若 <span class="math">T_{{ℓ+1}}=−T_ℓ</span>，则 cosine=−1。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3C · representation</div><div><h4>三维 map 的 held-out predictability、seed reproducibility 与方向连续性</h4><p><strong>目的。</strong>描述 answer count geometry 在相邻 blocks 间如何换坐标，并为随后 intervention 冻结 target direction。<strong>设置。</strong>只用 discovery seeds 对 Qwen 35、Gemma 41 个 boundaries 拟合 rank-3 centroid maps；用 seed-held-out error、500 次 bootstrap map error 与 gauge-invariant operator cosine 审计。<strong>结果。</strong>Qwen {map_stable_counts['Qwen3-8B']}/35、Gemma {map_stable_counts['Gemma4-E4B']}/41 个 boundaries 通过两个 local-error cutoffs；最后一个未通过边界均为 L18→19。选定 late boundaries 的 cosine 较高但低于 1。<strong>分析与目前结论。</strong>晚层局部 map 可预测、可复现且方向连续，但模型不是逐层复制同一固定 operator；本实验仍是 representation analysis。</p></div></div>
  <figure><h4 class="figure-title">图 9 · Answer-query 三维相邻层映射：error 与跨层 cosine</h4><div class="figure-stack"><div><h4>Qwen3-8B</h4><div class="chart-pair"><div><h4>Local-map errors（越低越好）</h4>{map_error_charts['Qwen3-8B']}</div><div><h4>Full-operator cosine（越高越连续）</h4>{map_cosine_charts['Qwen3-8B']}</div></div></div><div><h4>Gemma4-E4B</h4><div class="chart-pair"><div><h4>Local-map errors（越低越好）</h4>{map_error_charts['Gemma4-E4B']}</div><div><h4>Full-operator cosine（越高越连续）</h4>{map_cosine_charts['Gemma4-E4B']}</div></div></div></div><figcaption>四幅图的横轴均显示 current map 的 target layer，因此点 <span class="math">29</span> 表示 current map 是 L28→29；右图该点比较 L28→29 与下一张 L29→30 map。最后一张 map 没有可比较的 next boundary，所以每个右图都比对应左图少一个点。左列使用 log 纵轴：绿色是未见 seed 上的 centroid-coordinate prediction error，紫色是 500 次 seed-bootstrap 重拟合后的 gauge-aligned relative map error；浅绿色只标记两个 error cutoffs 同时通过。右列使用 linear 纵轴，橙线是 gauge-invariant ambient operators 的 Frobenius cosine，虚线 1 表示方向相同但允许整体 scale 不同。晚层 error 降低而 cosine 总体升高，支持局部 map 变得可预测、可复现且方向更连续；cosine 仍低于 1，因此不支持所有晚层共享完全相同的固定 operator。</figcaption></figure>

  <div class="claim boundary"><strong>Map 与 causal basis 的关系。</strong>图 9 的 <span class="math">A_ℓ</span> 是三维 centroid coordinates 之间的描述性 map。下面的 1× intervention 使用相关但不相同的 source transport basis <span class="math">B_ℓ</span>：把 source ambient centroids ridge-regress 到 target 三维 coordinates，再对 regression weights 做 QR。两者共享 discovery centroids 与 target rank-3 geometry，但不能把“<span class="math">A_ℓ</span> 稳定”直接当作“模型自然使用 <span class="math">B_ℓ</span>”的因果证据。</div>

  <div class="step-heading"><span class="step-kicker">03 · Causal test · transformation</span><h3>5.4 下一 block 是否选择性接收 count-aligned change？</h3></div>
  <div class="formula"><strong>1× count-aligned intervention 与 control。</strong>令 source-layer centroid chord 为 <span class="math">c_ℓ(R→D)=μ^ℓ_D−μ^ℓ_R</span>，frozen transport basis 为 <span class="math">B_ℓ∈ℝ^{{d×3}}</span>，则唯一报告的 aligned displacement 是 <span class="math">δ^{{align}}_ℓ=B_ℓB^⊤_ℓc_ℓ(R→D)</span>，并在 answer query 执行 <span class="math">h^{{int}}_ℓ(q)=h^{{clean}}_ℓ(q)+δ^{{align}}_ℓ</span>。Control axis 是先从 discovery within-count residuals 删除 <span class="math">B_ℓ</span> 分量后得到的 top residual PC，因此与 <span class="math">B_ℓ</span> 正交；将它缩放到与 BF16 实际写入的 aligned displacement 完全相同的 norm，再加到同一 clean receiver state。两组只改变一个 source layer 的同一个 answer-query position，随后只继续运行一个 block。</div>
  <div class="claim"><strong>这个量在机制链中回答什么。</strong>假设第 <span class="math">ℓ</span> 层与第 <span class="math">ℓ+1</span> 层都存在 count geometry，但两个层的坐标轴可以旋转。我们在第 <span class="math">ℓ</span> 层沿冻结的 receiver→donor transport direction 推一下，然后只运行下一 block；<span class="math">F</span> 问“这次变化在下一层还剩多少个 target centroid-chord 单位”。因此它是一个<strong>局部、方向性的跨层传递系数</strong>，直觉上近似在探测该 block 的局部 Jacobian 是否把 source count coordinate 映到 target count coordinate。只有 aligned 方向的 <span class="math">F</span> 高、而等范数 orthogonal control 接近 0，才说明这不是“任意大扰动都会留下来”。</div>
  <div class="formula"><strong>Target-chord propagation coefficient。</strong>在 target layer 令 <span class="math">d_{{ℓ+1}}=μ^{{ℓ+1}}_D−μ^{{ℓ+1}}_R</span>，并令实际下一层变化为 <span class="math">Δh=h^{{int}}_{{ℓ+1}}(q)−h^{{clean}}_{{ℓ+1}}(q)</span>；定义 <span class="math">F_{{ℓ→ℓ+1}}=⟨Δh,d_{{ℓ+1}}⟩/‖d_{{ℓ+1}}‖²</span>。等价地，<span class="math">F</span> 是 <span class="math">Δh</span> 在 target chord 上的有符号系数：若 <span class="math">Δh=0.6d_{{ℓ+1}}+r_⊥</span>，则 <span class="math">F=0.6</span>，无论正交变化 <span class="math">r_⊥</span> 多大。<span class="math">F=0/1</span> 分别表示沿该方向没有净移动/移动了一个完整 chord unit；负值表示反向，超过 1 表示沿该方向 overshoot。它不表示最终 state 等于 donor centroid，也不等于最终生成数移动了一 count。<span class="example">例：target chord d=[10,0]，同一样本的 clean state=[3,4]、干预后=[12.5,6]；则 Δh=[9.5,2]，F=⟨[9.5,2],[10,0]⟩/100=0.95。第二维变化被投影忽略，而且基线是该样本的 clean state，不是 receiver centroid，所以只能读成“沿目标方向移动了 0.95 个 chord”，不能读成“形成了 95% donor state”。</span></div>
  <div class="experiment"><div class="experiment-label">Experiment 3D · causal transport</div><div><h4>Aligned source-chord injection 与等范数 orthogonal control</h4><p><strong>目的。</strong>检验一个 block 是否选择性把 source count coordinate 转成 target count coordinate，而不只是响应任意同 norm 扰动。<strong>Discovery construction。</strong>只用 seeds 1234–1253、counts 1–10 的 answer-query states；对每个相邻 layer boundary，把 source ambient centroids ridge-regress 到 target rank-3 count coordinates，再对回归权重的 row space 做 QR，冻结 source transport basis <span class="math">B_ℓ</span>。<strong>Confirmation pairs。</strong>在 seeds 1254–1263 上预注册四个有向 one-count pairs：receiver→donor 为 1→2、2→1、5→6、6→5。它们在看 confirmation result 前固定，用两个相邻区间、双向配对来保持 count gap 恒为 1 并同时检验正/反方向；不是从结果中挑出的最佳 donor pairs，也不覆盖 counts 1–10 的全部相邻边。<strong>Intervention。</strong>对某个 receiver prompt，在 source-layer answer query 加 <span class="math">Proj_{{B_ℓ}}(μ_D^ℓ−μ_R^ℓ)</span> 的 realized 1× 或 2×；control 位于 <span class="math">B_ℓ</span> 正交补，并逐样本匹配 1× 的实际 norm。只继续一个 block，再计算 target-layer <span class="math">F</span>。<strong>结果。</strong>Qwen L28→29 的 orthogonal/aligned 1×/aligned 2× raw means 为 0.0069/0.9486/1.8096；1× specificity contrast={f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['mean_contrast'],4)} [{f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_low'],4)}, {f(transport_contrasts[('Qwen3-8B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_high'],4)}]。Gemma L36→37 为 0.0020/0.9779/1.7990；1× contrast={f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['mean_contrast'],4)} [{f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_low'],4)}, {f(transport_contrasts[('Gemma4-E4B', 'aligned_dose_1_minus_orthogonal')]['bootstrap_95ci_high'],4)}]。<strong>分析与目前结论。</strong>下一 block 对冻结 count-aligned direction 具有选择性接收能力；2× 结果提供剂量一致性，但只有两档非零剂量，仍不足以建立一般 scaling law。</p></div></div>
  <details class="collapsible-list" open><summary>展开：donor/receiver 到底有多少、如何选取</summary><div class="table-wrap"><table><thead><tr><th>项目</th><th>精确设置</th><th>解释边界</th></tr></thead><tbody>
    <tr><td>Discovery states</td><td>每模型 seeds 1234–1253 × counts 1–10；用于拟合 source/target centroids、target rank-3 coordinates 与每个 boundary 的 transport basis。</td><td>这些 rows 只定义方向，不进入 confirmation 推断。</td></tr>
    <tr><td>Confirmation receiver prompts</td><td>每模型 10 seeds（1254–1263）×4 receiver counts（1、2、5、6）=40 actual receiver prompt units / boundary。</td><td>模型真正运行的是 receiver prompt；同一 seed 内四个方向最后先平均。</td></tr>
    <tr><td>“Donor” 的含义</td><td>四个 frozen directed labels：1→2、2→1、5→6、6→5。Donor direction 来自 discovery centroid difference <span class="math">μ_D−μ_R</span>，不是在 confirmation 中逐个复制另一条 donor prompt 的完整 hidden state。</td><td>因此这里是 centroid-chord injection，不是 full-state donor patch；后者由 Experiment 3A 单独检验。</td></tr>
    <tr><td>Pair-selection rule</td><td>在 confirmation 前写入 config；只选 gap=1 的两个区间，并把每个区间两个方向都纳入。</td><td>控制位移长度与符号，避免大 count gap 自动产生更大 perturbation；不能外推为所有相邻 count 都已验证。</td></tr>
    <tr><td>Rows per boundary</td><td>40 receiver-pair units×3 conditions（aligned 1×、aligned 2×、matched orthogonal）=120 rows/model/boundary。主图只画 1× 与 orthogonal；2× 是 secondary dose check。</td><td>推断单位是每个 seed 对四个 directed pairs 的均值，共 n=10 seeds；区间为 50,000 次 seed bootstrap。</td></tr>
    <tr><td>Full layerwise coverage</td><td>Qwen 11 个、Gemma 14 个预注册 boundaries，共 25×120=3,000 audited rows；图 10 突出显示机制链对应的 Qwen L28→29 与 Gemma L36→37。</td><td>图中两条 boundary 不是整个 layerwise sweep 的全部观测。</td></tr>
  </tbody></table></div></details>
  <div class="claim"><strong>支持的结论与限制。</strong>等范数 orthogonal control 接近 0，而 aligned 1× 接近一个 target chord，说明下一 block 对 frozen count-aligned direction 具有选择性，而不是只对干预 norm 敏感；aligned 2× 约产生 1.8 个 target-chord units，提供同方向的 dose consistency。它仍只是一个预定义方向跨一个 block 的局部因果传播测试。由于 Transformer 的 residual connection 本身会保留部分扰动，<span class="math">F&gt;0</span> 不能单独区分“被 block 主动计算”与“沿 residual stream 保留下来”；方向特异的 orthogonal contrast 缩小了这一解释空间，但自然 forward 是否依赖该方向，仍需结合图 8 removal、broad-head ablation、source mediation 与 full-state answer patching。</div>
  <figure><h4 class="figure-title">图 10 · Aligned 1× 跨一个 block 的方向选择性传播</h4>{transport_chart}<figcaption>每条横条是 10 confirmation seeds×4 frozen directed pairs 的 condition mean <span class="math">F</span>；横轴单位是 target-layer <span class="math">R→D</span> centroid chord，虚线 1 表示一个 chord unit，而不是“到达 donor centroid”。主图比较 actual-norm-matched orthogonal 1× 与 aligned 1×；完整 panel 还运行了 aligned 2×，其结果在正文与展开表中报告。该图读取的是相对于同一样本 clean target state 的下一层 hidden-state change，并只保留其沿 target chord 的分量。</figcaption></figure>
  <div class="claim"><strong>Stage-III conclusion。</strong>四类证据共同把晚层 answer representation 从“可解码”提升为“可执行”：full-state patch 证明充分性，rank-3 removal 证明自然 computation 对 count-aligned component 的方向特异依赖，aligned 1× 证明相邻 block 具有选择性接收能力，而跨层 map 表明 centroid relation 在晚层可预测且可复现。Map 是描述性证据，不承担因果证明；cosine 低于 1 也说明模型并非逐层复制一枚固定三维 counter，而是在连续重参数化一个可执行的 answer-side state。</div>
  </details>
</section>

<section id="ov-write">
  <h2>Appendix D · 模型特定的最终写入路径：Qwen 局部 OV 与 Gemma 分布式 residual</h2>
  <details class="paper-appendix"><summary>展开 Appendix D 的 representation、steering、removal 与 mediation 证据</summary>
  <div class="claim"><strong>先把 OV 讲清楚。</strong>一个 attention head 做两件事：attention weights 决定“去哪些 token 取信息”，这是 <em>where to read</em>；V projection 把那些 token 变成可传递的内容，再由 <span class="math">W_O</span> 加进 answer-query residual，这是 <em>what to write</em>。所谓 OV write，就是后半步 <span class="math">W_OV</span>：它不再问头看了哪里，而是问“这个头最终给答案位置增加了什么向量”。只有 attention map 还不能证明该向量包含计数，也不能证明模型使用了它。</div>
  <div class="path"><div class="node"><strong>Attention routing</strong><small>选中 prompt 中的证据</small></div><div class="node"><strong>Value content</strong><small>把证据变成 head 内部向量 z</small></div><div class="node"><strong>Output projection W<sub>O</sub></strong><small>把 z 写入共享 residual</small></div><div class="node"><strong>Later blocks</strong><small>继续整合、维持或修改</small></div><div class="node"><strong>Count logits</strong><small>最终数字分布</small></div></div>
  <p class="lead">这一节进一步问：Stage III 的可执行 state 由哪些组件写入？我们按强度递增检查四件事：自然 head output 是否随真实 count 有序变化；沿自然方向增减是否按符号移动 expected count；删除自然方向是否比等范数无关方向更伤答案；上游 patch 的效应是否经该组件传到后层。模型间不必共享同一个微观电路：Qwen 满足局部 OV-writer 证据链，Gemma 则更符合“若干 heads 参与、后续 residual 分布式承接”的实现。</p>
  <div class="claim"><strong>Stage-IIIb hypothesis。</strong>Broad retrieval 不会直接等同于最终 count logits；retrieved content 还要经 attention value/output path 或后续 residual blocks 写成 answer-side state。我们检验的是一条自然使用的写入路径，而非穷尽所有并行或冗余通道。</div>
  <div class="chain-blueprint" aria-label="Chain D mechanism representation causal test">
    <div class="chain-purpose"><span class="step-kicker">Purpose</span><p><strong>目的。</strong>解释 broad retrieval 得到的内容如何真正进入共享 residual 并影响数字输出，同时允许两个模型使用不同粒度的写入电路。<span class="mini-example"><strong>直观例子：</strong>Attention 像选择要复制哪几行文字，V→W<sub>O</sub> 像把选中的内容改写进最终答案草稿。只看到选择框落在哪里，不足以证明复制进去的内容是什么；还要检查草稿中的自然方向、注入、删除和中介。</span></p></div>
    <div class="evidence-triad">
      <div class="triad-step"><span class="protocol-no">01 · Mechanism</span><h3>Retrieved content 需要被写向输出</h3><p>预测：候选 head/residual carrier 不仅随 count 有序，还应传递上游效应。</p></div>
      <div class="triad-step"><span class="protocol-no">02 · Representation</span><h3>Natural carrier 先定位写入方向</h3><p>把自然 post-O output 投影到 frozen one-count direction，检查其 slope 与 ordering。</p></div>
      <div class="triad-step"><span class="protocol-no">03 · Causal test</span><h3>Steering、removal 与 mediation 闭环</h3><p>Signed injection 测 capacity；matched removal 测 natural use；exact component block 测路径中介。</p></div>
    </div>
  </div>

  <h3>6.1 Qwen3-8B：可以定位到 L28 的局部 OV writer</h3>
  <div class="path"><div class="node"><strong>Prompt records</strong><small>有序但 noisy</small></div><div class="node"><strong>L23/L27 broad heads</strong><small>从多个位置取回证据</small></div><div class="node"><strong>L28 H16/H19 core</strong><small>把证据写进 count-relevant residual</small></div><div class="node"><strong>L29–L35 residual</strong><small>这部分信号继续存在</small></div><div class="node"><strong>Answer</strong><small>晚层状态决定数字</small></div></div>
  <div class="step-heading"><span class="step-kicker">02 · Representation</span><h3>6.1a Natural carrier：候选 heads 自然写入什么？</h3></div>
  <div class="experiment"><div class="experiment-label">OV-1 · representation</div><div><h4>自然状态中确实带着 count ordering</h4><p><strong>目的。</strong>定位 Qwen L28 core set {{H16,H19}} 的自然写入方向。<strong>设置。</strong>不干预 forward，把实际 post-O output 投影到 discovery-frozen one-count direction，并在 seed 内回归 gold count。<strong>结果。</strong>Gold count 每增加 1，投影坐标平均增加 <strong>{f(qwen_nat['natural_carrier_count_slope']['mean'],4)}</strong> hidden-coordinate units。<strong>分析与目前结论。</strong>自然 head output 的排列与 count 一致，构成候选 writer representation；该 slope 不是输出数字变化，也不证明自然使用。<span class="mini-example"><strong>例：</strong>counts 1/2/3 的投影为 0.2/0.4/0.6，carrier slope=0.2 hidden units/count；不能读成输出增加 0.2。</span></p></div></div>
  <div class="step-heading"><span class="step-kicker">03 · Causal test</span><h3>6.1b Signed capacity、natural use 与 upstream mediation</h3></div>
  <div class="experiment"><div class="experiment-label">OV-2 · causal capacity</div><div><h4>沿自然写入方向增减，会按符号推动答案</h4><p><strong>目的。</strong>检验 candidate direction 是否有 signed steering capacity。<strong>设置。</strong>在 pre-O state 施加 <span class="math">±β</span>，再经过 heads 自身的 <span class="math">W_O</span>。<strong>结果。</strong>每增加一个 β，softmax expected count 平均改变 <strong>{f(qwen_nat['injection_dose_slope']['mean'],4)}</strong>。<strong>分析与目前结论。</strong>该通道能按符号推动数字分布；β 是干预剂量，不等于一 count，且 steering capacity 仍不等同于 natural use。<span class="mini-example"><strong>例：</strong>若 +β/−β 令 E[c] 相对 clean 改变 +0.08/−0.04，则中心差分 slope=(0.08−(−0.04))/2=0.06 expected counts/β。</span></p></div></div>
  <div class="experiment"><div class="experiment-label">OV-3 · causal necessity</div><div><h4>模型自然运行时也在使用这条方向</h4><p><strong>目的。</strong>区分“可人工 steer”与“自然 computation 实际依赖”。<strong>设置。</strong>从自然 output 删除 count-aligned component；control 在同一 W<sub>O</sub> output span 删除相同 realized norm 的正交 component。<strong>结果。</strong>自然轴 removal 比 control 多增加 <strong>{f(qwen_nat['removal_error_axis_minus_control']['mean'],4)}</strong> expected-count error，并使 correct margin 多下降 <strong>{abs(float(qwen_nat['removal_margin_axis_minus_control']['mean'])):.4f}</strong>。<strong>分析与目前结论。</strong>效应量不大，但方向与 matched control 一致，支持局部 natural use。<span class="mini-example"><strong>例：</strong>Gold=8，axis/orth removal 后 E[c]=6.5/7.5，则 error specificity=1.5−0.5=1 count。</span></p></div></div>
  <div class="experiment"><div class="experiment-label">OV-4 · causal mediation</div><div><h4>上游取回的证据有一部分经这里传到后层</h4><p><strong>目的。</strong>验证 L28 writer 位于 broad retrieval 与后续输出之间。<strong>设置。</strong>先做 early broad-head donor patch，再精确阻断该 patch 在 L28 H16–H19 诱发的 component，并与等范数正交 block 配对。<strong>结果。</strong>Source patch 产生 <strong>{f(qwen_upstream_primary['early_effect']['mean'],4)}</strong> donor log-odds gain；exact block 相对 control 消去 <strong>{f(qwen_upstream_primary['mediation']['mean'],4)}</strong>，另一 donor-z assay 的 frozen count axis 解释 <strong>{pct(qwen_axis_mediated_fraction)}</strong> transport。<strong>分析与目前结论。</strong>L28 OV path 中介上游 retrieval 的一部分，而不是完整或唯一通路。<span class="mini-example"><strong>例：</strong>Source gain=0.40，orthogonal block 后保留 0.35、exact block 后只剩 0.10，则 mediation specificity=0.25。</span></p></div></div>
  <div class="claim"><strong>Qwen 的简化结论。</strong>Broad heads 先从多个 prompt positions 收集证据；L28 core set {{H16,H19}} 再通过自己的 V→W<sub>O</sub> 通道，把其中一部分变成 answer residual 中可影响 count 的有符号变化；更宽的 H16–H19 set 用于上游 mediation 检验。Full/routing/value normalized transports 分别为 {f(qwen_read_write['read_full_behavior_transport']['mean'],4)} / {f(qwen_read_write['read_routing_behavior_transport']['mean'],4)} / {f(qwen_read_write['read_value_behavior_transport']['mean'],4)}，说明“看哪里”和“取到什么内容”都重要。H19 leave-one-out decrement={f(qwen_h19_loo['decrement']['mean'],4)}，说明 H19 在该集合中不可完全替代，但不能据此称它为单头计数器。</div>

  <h3>6.2 Gemma4-E4B：候选 heads 参与写入，但可确认的主要对象是分布式 residual path</h3>
  <div class="path"><div class="node"><strong>L29H4 / L35H2</strong><small>full-attention layers 可直接读取远端</small></div><div class="node"><strong>L37 answer residual</strong><small>出现可中介的分布式变化</small></div><div class="node"><strong>L38–L40</strong><small>同一 query 位置继续传递</small></div><div class="node"><strong>L41</strong><small>进入终端 count representation</small></div></div>
  <div class="step-heading"><span class="step-kicker">02 · Representation</span><h3>6.2a Natural carrier：Gemma 也有 head-level count ordering</h3></div>
  <p>L29H4 的自然 carrier slope 为 {f(gemma_l29h4['natural_carrier_count_slope']['mean'],4)}，且 candidate-minus-matched-head 为 +{f(gemma_l29h4_specificity['natural_carrier_count_slope__candidate_minus_control_mean']['mean'],4)}。因此不能把 Gemma 概括为“没有 write representation”；问题是它是否像 Qwen 一样能被定位为相对 matched heads 明显更特异的局部 writer。</p>
  <div class="step-heading"><span class="step-kicker">03 · Causal test</span><h3>6.2b Head participation 与 distributed residual mediation</h3></div>
  <div class="experiment"><div class="experiment-label">Gemma head-level · representation + causal</div><div><h4>L29H4 的自然信号、steering 与 removal 均为正，但并非全部具有 matched-head specificity</h4><p><strong>目的。</strong>检验 Gemma 是否存在与 Qwen 同等粒度的 localized OV writer。<strong>设置。</strong>对 L29H4 同时执行 natural carrier、signed injection、aligned-vs-orthogonal removal、donor-z transport，并与 matched heads 比较。<span class="mini-example"><strong>直观例子：</strong>某个候选头可以推动答案，不代表它比同层相似 heads 更特殊；只有 candidate-minus-control 也为正，才支持 localized specificity。</span></p><div class="table-wrap"><table><thead><tr><th>Test</th><th>L29H4 raw candidate effect [95% CI]</th><th>Candidate-minus-control interpretation</th></tr></thead><tbody>
    <tr><td>Natural carrier slope</td><td>{f(gemma_l29h4['natural_carrier_count_slope']['mean'],4)} [{f(gemma_l29h4['natural_carrier_count_slope']['ci95_low'],4)}, {f(gemma_l29h4['natural_carrier_count_slope']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['natural_carrier_count_slope__candidate_minus_control_mean']['mean'],4)}；自然 count ordering 强于 matched heads。</td></tr>
    <tr><td>Signed pre-O injection slope</td><td>{f(gemma_l29h4['injection_dose_slope']['mean'],4)} [{f(gemma_l29h4['injection_dose_slope']['ci95_low'],4)}, {f(gemma_l29h4['injection_dose_slope']['ci95_high'],4)}]</td><td>{f(gemma_l29h4_specificity['injection_dose_slope__candidate_minus_control_mean']['mean'],4)}；能 steer，但不优于 matched heads。</td></tr>
    <tr><td>Removal error specificity</td><td>{f(gemma_l29h4['removal_error_axis_minus_control']['mean'],4)} [{f(gemma_l29h4['removal_error_axis_minus_control']['ci95_low'],4)}, {f(gemma_l29h4['removal_error_axis_minus_control']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['removal_error_axis_minus_control__candidate_minus_control_mean']['mean'],4)}；自然方向删除更伤 expected-count error。</td></tr>
    <tr><td>Removal margin specificity</td><td>{f(gemma_l29h4['removal_margin_axis_minus_control']['mean'],4)} [{f(gemma_l29h4['removal_margin_axis_minus_control']['ci95_low'],4)}, {f(gemma_l29h4['removal_margin_axis_minus_control']['ci95_high'],4)}]</td><td>{f(gemma_l29h4_specificity['removal_margin_axis_minus_control__candidate_minus_control_mean']['mean'],4)}；负值表示正确答案 margin 下降更多。</td></tr>
    <tr><td>Donor-z transport</td><td>{f(gemma_l29h4['donor_patch_transport']['mean'],4)} [{f(gemma_l29h4['donor_patch_transport']['ci95_low'],4)}, {f(gemma_l29h4['donor_patch_transport']['ci95_high'],4)}]</td><td>+{f(gemma_l29h4_specificity['donor_patch_transport__candidate_minus_control_mean']['mean'],4)}；候选头确实可传 donor-directed content。</td></tr>
  </tbody></table></div><p><strong>结果与分析。</strong>L29H4 自然携带 count、可 signed steering、aligned removal 会伤输出，并能传 donor-directed content；但 injection candidate-minus-control={f(gemma_l29h4_specificity['injection_dose_slope__candidate_minus_control_mean']['mean'],4)}，path-mediation specificity={f(gemma_l29h4_specificity['mediation_control_minus_axis_block__candidate_minus_control_mean']['mean'],4)}。<strong>目前结论。</strong>Gemma 有 head-level participation，却没有 Qwen 那样清楚的 localized-head exclusivity；不能声称 L29H4 是唯一或显著更强的局部 writer。</p></div></div>

  <div class="experiment"><div class="experiment-label">Gemma residual path · causal mediation</div><div><h4>L37 以后承接 source effect，并把一部分 count-aligned change 传到终端</h4><p><strong>目的。</strong>在单头 specificity 不充分时，检验更下游的 distributed residual 是否承接 source effect。<strong>设置。</strong>把 {{L29H4,L35H2}} donor source 写给 receiver；在 L37 分别阻断完整 induced residual、预定义线性 count axis 或 matched control，并读取 L41 terminal adoption。<strong>结果。</strong>Source transport=<strong>{f(gemma_candidate['source_donor_transport']['mean'],4)}</strong>；L37 exact residual mediation=<strong>{f(gemma_candidate['exact_residual_mediation']['mean'],4)}</strong>，count-axis mediation=<strong>{f(gemma_candidate['count_axis_mediation']['mean'],4)}</strong>；L41 adoption=<strong>{f(gemma_candidate['terminal_count_adoption']['mean'],4)}</strong>。<strong>分析与目前结论。</strong>L37 distributed residual 承接主要被测 source effect，其中只有一部分落入预定义线性 axis；后续层继续把它转为终端 count state。<span class="mini-example"><strong>例：</strong>若 source patch 将 normalized transport 提到 0.30，control block 后仍为 0.28、exact residual block 后降到 0.08，则该 residual 中介约 0.20；剩余 0.08 仍可来自 bypass。</span></p></div></div>
  <div class="claim boundary"><strong>Gemma 的证据边界。</strong>不能写成“已经定位到与 Qwen 同等强度的单一 OV head set”，也不能写成“Gemma 没有 OV/write mechanism”。当前数据支持的是：周期性 full-attention heads 参与远端 retrieval 与写入，随后由 L37 及更晚层 answer-query residual 分布式承接。该模型差异与 Gemma 较弱的 prompt ordering 和较差的行为表现相容，但现有实验没有证明前者导致后者。</div>

  <details class="collapsible-list"><summary>展开：OV 各分数的精确定义与例子</summary>
    <div class="formula"><strong>1. Head output 与 natural carrier slope。</strong>Head <span class="math">h</span> 在 answer query 聚合出的 pre-O content 为 <span class="math">z_h(q)=Σ_jα_h(q,j)W_V^hh_j</span>，实际写入 residual 的是 <span class="math">W_O^hz_h(q)</span>。若候选 head set 的冻结一-count写入方向为 <span class="math">m_S</span>，则 carrier coefficient <span class="math">κ</span> 是自然 set output 在单位方向 <span class="math">m̂_S</span> 上的投影；natural carrier slope 是每个 seed 内 <span class="math">κ</span> 对 gold count 的 OLS slope。<span class="example">例：counts [1,2,3] 的 κ=[0.2,0.4,0.6]，则 slope=0.2 hidden-coordinate units/count；它不是 0.2 output counts。</span></div>
    <div class="formula"><strong>2. Injection slope 与 write behavior specificity。</strong>对 counts 1–10 的 candidate scores 做 softmax，定义 <span class="math">E[c]=Σ_{{c=1}}^{{10}}cp(c)</span>。在自然写入方向施加 <span class="math">±β</span>，中心差分 <span class="math">g=(ΔE[c]_{{+β}}−ΔE[c]_{{−β}})/(2β)</span>；write specificity 是自然方向的 <span class="math">g</span> 减去等 post-O norm 正交方向的 <span class="math">g</span>。<span class="example">例：β=1 时自然方向的 ΔE 为 +0.08/−0.04，正交方向为 +0.01/−0.01；两条 slope 为 0.06 与 0.01，specificity=0.05 expected counts/β。</span></div>
    <div class="formula"><strong>3. Removal error、margin 与 mediation。</strong>Expected-count error 为 <span class="math">e_E=|E[c]−N|</span>；removal error specificity=<span class="math">e_E(axis-remove)−e_E(orth-remove)</span>。Correct margin=<span class="math">log p(a_N)−max_{{c≠N}}log p(a_c)</span>；margin specificity 也取 axis-remove 减 orth-remove，负值表示自然轴 removal 更伤正确答案。Path mediation=<span class="math">T_{{orth-block}}−T_{{exact/axis-block}}</span>，mediated fraction 再除以未阻断的 donor-patch transport。<span class="example">例：gold N=8，axis/orth removal 的 E[c] 为 6.5/7.5，则 error specificity=|6.5−8|−|7.5−8|=1；若 donor transport=0.20、orth-block 后=0.18、axis-block 后=0.14，则 mediation=0.04、fraction=20%。</span></div>
    <div class="formula"><strong>4. Routing/value decomposition。</strong>RR/DD 分别是 receiver/donor 的真实 pre-O endpoints；RD 保留 receiver attention weights 但换成 donor values，DR 则保留 donor attention weights但使用 receiver values。定义 <span class="math">Δz_{{value}}=½[(z_{{RD}}−z_{{RR}})+(z_{{DD}}−z_{{DR}})]</span>、<span class="math">Δz_{{route}}=½[(z_{{DR}}−z_{{RR}})+(z_{{DD}}−z_{{RD}})]</span>，两者相加等于 full donor movement。Behavior transport 是各 component 引起的 expected-count change 除以 donor–receiver count gap。<span class="example">例：receiver count=3、donor count=8，某 component 使 E[c] 从 3.2 变为 4.2，则 normalized transport=(4.2−3.2)/(8−3)=0.20。</span></div>
  </details>
  <div class="claim"><strong>Stage-IIIb conclusion。</strong>两模型共享功能阶段，但不共享已解析到同一粒度的电路。Qwen 给出“broad retrieval → L28 H16/H19 OV write → late residual”的局部路径；Gemma 给出“full-attention-head participation → L37 distributed residual mediator → terminal state”的路径。报告因此把 OV/residual write 作为一条经验证的部分通道，而不声称它是唯一实现。</div>
  </details>
</section>

<section id="ledger">
  <h2>证据取舍：正文主张、限定条件与复现账本</h2>
  <details class="paper-appendix"><summary>展开完整实验—证据—结论表</summary>
  <div class="table-wrap"><table class="evidence-map"><thead><tr><th>Stage</th><th>Experiment / operation</th><th>Readout</th><th>Observed value</th><th>Supported conclusion</th></tr></thead><tbody>
    <tr><td>Representation</td><td>Seed-held-out ridge at needle ends</td><td>R² / MAD</td><td>Qwen L8 0.945 / 0.561；Gemma L9 0.719 / 1.249</td><td>Running index follows a continuous, noisy, linearly decodable geometry.</td></tr>
    <tr><td>Representation</td><td>Frozen 3-PC counter-property + grouped cubic position control</td><td>line R² / step cosine / position-residual R²</td><td>Qwen L8 0.633 / 0.761 / 0.043；Gemma L9 0.403 / 0.084 / 0.004</td><td>Qwen has a stronger counter-like endpoint trajectory, but neither model's frozen 3D curve is separable from absolute-position structure.</td></tr>
    <tr><td>Representation baseline</td><td>Seed-held-out exact classifiers at answer query</td><td>L2 logistic and nearest-centroid accuracy / MAD</td><td>Qwen L29: logistic 56.0% / 0.635, centroid 54.5% / 0.640；Gemma L37: logistic 53.0% / 0.720, centroid 55.0% / 0.615</td><td>Descriptive compactness/separability baseline for later native-thinking comparison; not causal evidence or a discrete-counter claim.</td></tr>
    <tr><td>Representation</td><td>Centroid vs all-state rank-3 capture</td><td>explained variance</td><td>centroid 94.7%–98.8%；all state 59.8%–79.9%</td><td>Low-dimensional mean trajectory sits inside a higher-dimensional contextual cloud.</td></tr>
    <tr><td>Representation</td><td>V4.4.5 reused-state display-only 3-PC nearest centroid</td><td>confirmation accuracy / integer MAD</td><td>Qwen L28 61% / 0.72；Gemma L37 63% / 0.43</td><td>These layers provide clearer 3D answer visualization; post-hoc display selection is excluded from causal inference.</td></tr>
    <tr><td>Formation</td><td>Clean vs needle-corrupt vs length/token-budget-matched ordinary-corrupt full-vector capture；100 confirmation prompts/model、all post-block layers</td><td>needle-minus-ordinary RMS state deformation；pre-frozen behavior-reuse-window seed test</td><td>Raw primary Qwen +0.5125 [−0.0208,1.1509], p=0.1484；Gemma +0.0102 [−0.0060,0.0375], p=0.8359。Gemma relative-RMS/cosine secondaries +0.0240/+0.0079, both p=0.0020；Qwen normalized effects localized but whole-window tests fail.</td><td>Needle input causally changes span states, but a stronger-than-ordinary raw deformation is not stable across the whole frozen reuse window in either model. Geometry can persist after behavioral reusability ends, so state difference is not itself mediation.</td></tr>
    <tr><td>Formation</td><td>20 discovery clean N=10 prompts freeze running-index centroids/radii；10 held-out three-arm N=10 prompts test endpoint/span-mean retention across all layers</td><td>needle-minus-ordinary correct-centroid distance；margin/accuracy and relative-RMS/cosine sensitivity analyses；frozen behavior windows + Holm families</td><td>Endpoint correct-distance Qwen +{f(state_retention_primary['Qwen3-8B']['mean'], 3)} [{f(state_retention_primary['Qwen3-8B']['ci95_low'], 3)}, {f(state_retention_primary['Qwen3-8B']['ci95_high'], 3)}]；Gemma +{f(state_retention_primary['Gemma4-E4B']['mean'], 3)} [{f(state_retention_primary['Gemma4-E4B']['ci95_low'], 3)}, {f(state_retention_primary['Gemma4-E4B']['ci95_high'], 3)}] class radii；exact p=0.00195、Holm p=0.0117 in both</td><td>Needle input specifically maintains count-organized occurrence states: ordinary corruption leaves states near the correct clean cloud, whereas needle corruption pushes them far outside. This is input→state evidence, not by itself state→output mediation.</td></tr>
    <tr><td>Formation</td><td>Active-needle token corruption vs ordinary-token control</td><td>Δ absolute error</td><td>Qwen +8.930；Gemma +8.780</td><td>Needle evidence is causally required.</td></tr>
    <tr><td>Formation</td><td>All-endpoint prompt rank-3 removal vs orthogonal</td><td>Δ absolute error</td><td>Qwen +0.056；Gemma −0.022</td><td>No strong localized necessity for the decoded endpoint curve.</td></tr>
    <tr><td>Formation</td><td>72,000-row dense endpoint/full-span/ordinary restoration</td><td>full-minus-ordinary expected-error repair；seed-cluster CI / exact sign flip</td><td>positive nominal window Qwen L0–20（21 layers）、Gemma L0–16（17）；largest drop Qwen L20→21 −0.796、Gemma L16→17 −2.106 counts；endpoint−ordinary near 0</td><td>Prompt evidence is causally reusable as a distributed whole-span state; Qwen loses reuse gradually over L15–22, whereas Gemma has a sharp L16→17 boundary.</td></tr>
    <tr><td>Formation → retrieval</td><td>Canonical 一次性 full-span restoration 后重建 cache 并读取 frozen answer-query broad bank</td><td>(needle restored−needle corrupt)−(ordinary restored−ordinary corrupt) 的 mass ΔM / broad score ΔB</td><td>Qwen Top-32 ΔB：L0 0.310、L16 0.302、L20 0.196、L21 0.061、L26 0.010；Gemma Top-6：L0 0.218、L16 0.340、L17 0.142、L22 0.108、L23 0</td><td>较早层修复 prompt evidence 会特异地重新配置后续 broad retrieval；behavior-coupled 主窗口约止于 Qwen L20 / Gemma L16，另有不再修复答案的 attention-only tail。该曲线不表示 retrieval head 直接读取某个历史层。</td></tr>
    <tr><td>Retrieval</td><td>Broad top-K head ablation vs layer-matched random</td><td>absolute count shift</td><td>Qwen K32 +1.623 [1.117, 2.137]；Gemma K6 +0.825 [0.690, 0.975]（post-hoc dose）</td><td>Both frozen primary head sets have matched-control effects; Gemma K6 retains an explicit post-hoc label.</td></tr>
    <tr><td>Retrieval</td><td>Broad-bank state geometry on confirmation seeds</td><td>exact/nearest-centroid accuracy; rank-3 capture; silhouette</td><td>accuracy {pct(min(retrieval_exact_values))}–{pct(max(retrieval_exact_values))} / {pct(min(retrieval_nearest_values))}–{pct(max(retrieval_nearest_values))}；centroid rank-3 0.968–0.995；silhouette −0.098…0.014；Gemma uses exact Top-6</td><td>The retrieved aggregate has a low-rank mean count trajectory but noisy and unstable individual-state geometry.</td></tr>
    <tr><td>Retrieval</td><td>Aligned retrieval rank-3 removal vs equal-norm orthogonal, natural and restored</td><td>natural specificity / restoration mediation</td><td>Qwen L23 +{f(qwen_retrieval_l23['natural_specificity_mean'])} / +{f(qwen_retrieval_l23['restoration_mediation_mean'])}；Gemma Top-6 L29 +{f(gemma_retrieval_l29['natural_specificity_mean'])} / +{f(gemma_retrieval_l29['restoration_mediation_mean'])}；later frozen layers approximately zero</td><td>The fitted retrieval subspace is causally used in a localized aggregation window, not as a persistent late counter.</td></tr>
    <tr><td>Retrieval</td><td>Donor source patch + downstream exact block</td><td>source gain / mediation</td><td>Qwen +{f(qwen_upstream_primary['early_effect']['mean'],4)} / +{f(qwen_upstream_primary['mediation']['mean'],4)}；Gemma +{f(gemma_candidate['source_donor_transport']['mean'],4)} / +{f(gemma_candidate['exact_residual_mediation']['mean'],4)}</td><td>Retrieval writes a donor-directed state into a downstream path.</td></tr>
    <tr><td>Patch</td><td>Prompt full-span donor→receiver patch</td><td>strict donor-gold hit</td><td>Qwen {pct(prompt_patching['Qwen3-8B']['pooled_average_patching_acc'])}；Gemma {pct(prompt_patching['Gemma4-E4B']['pooled_average_patching_acc'])}</td><td>Distributed prompt-span states contain sufficient upstream information, despite weak rank-3 necessity.</td></tr>
    <tr><td>Write</td><td>Full answer-state donor→receiver patch</td><td>strict donor-gold hit</td><td>Qwen 96.6%；Gemma 96.0%</td><td>Late answer-query state is sufficient to determine the count output.</td></tr>
    <tr><td>Write</td><td>Answer-query rank-3 removal</td><td>peak Δ absolute error</td><td>Qwen L28 +0.878；Gemma L32 +1.222</td><td>Late count-aligned state is direction-specifically necessary.</td></tr>
    <tr><td>Map</td><td>Discovery-only adjacent-layer rank-3 centroid maps</td><td>seed-CV centroid NRMSE / gauge-aligned bootstrap map error / full-operator cosine to next</td><td>Qwen L28→29 {f(selected_map_rows['Qwen3-8B']['cv_centroid_normalized_rmse'],4)} / {f(selected_map_rows['Qwen3-8B']['bootstrap_map_relative_frobenius_median'],4)} / {f(selected_map_rows['Qwen3-8B']['full_operator_cosine_to_next'],4)}；Gemma L36→37 {f(selected_map_rows['Gemma4-E4B']['cv_centroid_normalized_rmse'],4)} / {f(selected_map_rows['Gemma4-E4B']['bootstrap_map_relative_frobenius_median'],4)} / {f(selected_map_rows['Gemma4-E4B']['full_operator_cosine_to_next'],4)}</td><td>The selected late boundaries have predictive and reproducible local 3D centroid relations, with substantial but imperfect orientation continuity to the next boundary.</td></tr>
    <tr><td>Transport</td><td>Adjacent-layer aligned 1× vs actual-norm-matched orthogonal</td><td>target-chord propagation coefficient F</td><td>Qwen contrast +0.9417 [0.9127, 0.9670]；Gemma +0.9759 [0.9639, 0.9884]</td><td>An injected count-aligned change is selectively relayed across one block; this is local transport capacity, not by itself proof of natural-axis use.</td></tr>
    <tr><td>OV write</td><td>Qwen L28 H16/H19 natural pre-O injection/removal</td><td>expected-count slope / error specificity</td><td>+0.0640 / +0.0732</td><td>A localized natural OV transporter writes signed count content in Qwen.</td></tr>
    <tr><td>OV / residual write</td><td>Gemma L29H4 natural-OV tests + L37 residual mediation</td><td>carrier / injection / removal error / exact residual mediation</td><td>+0.1360 / +0.0612 / +0.0628 / +0.0864</td><td>L29H4 participates in count-relevant writing, while matched-head controls prevent a unique localized-writer claim; L37 is the confirmed distributed mediator.</td></tr>
    <tr><td>Integrated chain</td><td>Same-forward 11-arm source restoration × retrieval/late directional blocks；counts averaged within seed</td><td>source / retrieval / late expected-count effects；10-seed cluster CI + exact sign-flip</td><td>Qwen +{f(qwen_serial['source_repair']['mean'])} / +{f(qwen_serial['retrieval_mediation']['mean'])} / +{f(qwen_serial['late_mediation']['mean'])} counts，retrieval CI [{f(qwen_serial['retrieval_mediation']['ci95_low'])}, {f(qwen_serial['retrieval_mediation']['ci95_high'])}]；Gemma Top-6 +{f(gemma_serial['source_repair']['mean'])} / +{f(gemma_serial['retrieval_mediation']['mean'])} / +{f(gemma_serial['late_mediation']['mean'])} counts，retrieval CI [{f(gemma_serial['retrieval_mediation']['ci95_low'])}, {f(gemma_serial['retrieval_mediation']['ci95_high'])}]</td><td>Both models support ordered partial serial mediation at the seed level. Negative interaction and remaining repair rule out an exhaustive unique path.</td></tr>
    <tr><td>Formation micro-circuit</td><td>Independent induction assay + canonical previous-successor edge removal vs matched ordinary edge</td><td>candidate-minus-control expected-error damage</td><td>Qwen −0.02193 [−0.03311,−0.01076]；Gemma −0.01207 [−0.02499,0.00127]</td><td>Induction-like heads exist in the synthetic assay, but the registered classical-induction edge specificity is not supported in canonical counting.</td></tr>
    <tr><td>Prompt-noise attribution</td><td>I×C×P factorial + selected outside-halo edge removal vs two matched controls</td><td>held-out full R² / candidate specificity gate</td><td>full R² −0.0221/−0.0893；candidate_exceeds_both_controls=false in both models</td><td>The registered factor model does not stably explain held-out scatter, and the selected halo edges are not specifically necessary beyond matched controls.</td></tr>
  </tbody></table></div>
  </details>

  <h3>7.1 What is established—and what is not</h3>
  <ul>
    <li><span class="pill">Established</span> 两个位置都含 count geometry；prompt centroid trajectory 低维但单样本 noisy。</li>
    <li><span class="pill">Established</span> Qwen endpoint 三维轨迹比 Gemma 更有序、同向，但 absolute position 几乎与 running index 共线；去除 cubic position 后冻结前三维 readout 接近零。</li>
    <li><span class="pill">Established with qualification</span> Active-needle input intervention 的逐层 full-vector state response 已直接测量；raw-RMS needle-minus-ordinary 的整个预冻结 reuse-window 检验仍未跨模型通过，因此不能写成“needle 在任何尺度上都造成更大 state change”。方向明确的 held-out centroid assay 则在两个模型一致通过：ordinary corruption 后 state 仍靠近正确 running-index cloud，needle corruption 将其推离。跨模型成立的是 count-specific organization，不是无条件的 raw displacement 定律。</li>
    <li><span class="pill">Established</span> 完整 needle span 的早层 state 可强力修复 corrupt behavior；endpoint-only restoration 近零。State deformation 与 restoration 必须并读：前者问 forward 是否不同，后者问差异是否仍能被后续 computation 使用。</li>
    <li><span class="pill">Established</span> 晚层相邻 answer-query boundaries 的局部三维 centroid maps 在 held-out seeds 上可预测、对 seed bootstrap 稳定；连续 ambient operators 的 cosine 较高但低于 1，表示方向更连续但并非固定不变。</li>
    <li><span class="pill">Established</span> broad-ranked answer-query heads、局部 retrieval rank-3、late answer residual state 与最终数字之间存在 matched-control causal effects；retrieval rank-3 effect 只在 Qwen L21–23 / Gemma L29 出现。</li>
    <li><span class="pill">Established with qualification</span> 同一 forward 的 nested intervention 在两模型均给出 source→retrieval→late→output 的三项正向诊断；两模型的 100-row paired-effects 表均已保留，并以 10 个 seed means 为独立单位完成 10,000 次 cluster bootstrap 与 1,024 种 exact sign-flip。该结果支持 ordered partial serial mediation；负 interaction 与剩余 repair 限制了唯一、穷尽通路的主张。</li>
    <li><span class="pill">Falsified strong version</span> Synthetic relation-following head 的存在不足以建立 canonical classical-induction mechanism；冻结 previous-successor edge removal 未超过 attention/distance-matched ordinary-edge control。</li>
    <li><span class="pill">Falsified registered attribution</span> Identity/context/position factorial 的 held-out full model 为负 R²，selected outside-halo edges 也未同时超过两个 matched controls；因此不能把 prompt scatter 归因于这套简单受控分解。</li>
    <li><span class="pill">Not established</span> running-index direction与answer count direction是同一轴；事实上两者可近正交，因为位置、basis、局部 computation 与写入坐标系不同。</li>
    <li><span class="pill">Not established</span> 某一个 prompt token 或某一个 attention head 单独存储/计算完整整数。</li>
    <li><span class="pill">Scope boundary / closed</span> 不声称 endpoint trajectory 对 final count N 不变；当前 running-index capture 的 N 固定为 10，position control 说明 ordinal-position 解释不可忽略，因此不再为本文追加 cross-final-N experiment。</li>
    <li><span class="pill">Optional finer attribution</span> 早层 nonlinear/distributed code 如何被各 broad head 的 QK routing 与 V content 分别读取，仍可用 position-resolved path patching 细分；但 span-level restoration 已足以支持本文粒度的 distributed-evidence claim。</li>
  </ul>

  <details class="paper-appendix"><summary>展开复现账本与 artifact provenance</summary>
  <h3>Reproducibility ledger</h3>
  <ol class="source-list">
    <li><code>v4_4_extension/geometry/*.csv</code>：rank、regression、clustering。</li>
    <li><code>v4_4_extension/counter_properties/*</code>：frozen 3-PC counter quantities、grouped cubic position control、跨层图与 PASS audit。</li>
    <li><code>v4_4_extension/all_token/{'{'}all_token_frozen_pca_projections.csv.gz,gated_curve_formula_tests.csv{'}'}</code>：endpoint/interior/ordinary/hard-negative formula controls。</li>
    <li><code>v4_4_causal_v2/baseline_by_count.csv</code>：按 gold count 的行为 accuracy / absolute error。</li>
    <li><code>v4_4_extension/classification/classification_all_*/answer_classifier_metrics.csv</code>：仅 answer-query seed-held-out classification；本报告读取 <code>logistic_l2</code> 与 <code>nearest_centroid</code> 两行系列。</li>
    <li><code>v4_4_extension/token_corruption/token_corruption_statistics.csv</code>。</li>
    <li><code>v4_4_extension/prompt_subspace_ablation/subspace_ablation_statistics.csv</code>。</li>
    <li><code>v4_4_extension/endpoint_attention_mask/*earlier_span_head_confirmation.csv</code>。</li>
    <li><code>v4_4/realistic_niah_v4_head_atlas.csv</code> 与 <code>v4_4_causal_v2/full_span_topk/{'{'}full_span_topk_membership,full_span_topk_primary_statistics{'}'}.csv</code>。</li>
    <li><code>v4_4_report_additions/full_span_topk_raw_arms.csv</code>：由 formal 2,400-row/model top-K detail 重新汇总的 ranked/random 原始两臂均值与 seed-bootstrap CI；逐 K 的 raw difference 与预注册 contrast 最大差异 ≤2.23×10<sup>−16</sup>。<code>qwen_attention_gallery.json</code>：exact-tokenizer 重建的 discovery-frozen top-4 heads × seed1254、N=3/6/9 固定文本网格，共 12 条全文/token-level attention rows；逐条审计 sequence/query、64-token bin sum、region/span mass 与 Filestream raw/metric SHA。<code>qwen_attention_example_l27h18.json</code> 保留旧单例作为 provenance；全部由 <code>report_additions_audit.json</code> 标记 PASS。</li>
    <li><code>v4_4/v4_4_answer_query_patching.csv</code> 与 <code>v4_4_causal_v2/correct_patching_pooled.csv</code>。</li>
    <li><code>v4_4_extension/layerwise_subspace/{'{'}layer_maps,map_causal_link,answer_query_removal,transport{'}'}/*.csv</code>：三维 map、gauge-aligned stability、removal 与单剂量 transport。</li>
    <li><code>v4_4_4/*analysis.json</code>：Qwen natural OV/read-write/upstream chain。</li>
    <li><code>v4_4_4/gemma/residual/k2/*analysis.json</code>：Gemma K2 residual path。</li>
    <li><code>v4_4_5_followup/campaign_summary.json</code>、<code>v4_4_5_followup/span_restoration/{{needle_minus_ordinary_specificity,full_minus_endpoint,layerwise_seed_statistics}}.*</code> 与 <code>plans/nonthinking-followup-experiment-log-20260813.md</code>：72,000-row dense restoration、逐层 seed-cluster statistics、23,400 answer-state rows、3,000 retrieval states、2,800 retrieval-subspace rows及 persistent-copy audits。</li>
    <li><code>v4_4_5_followup/span_restoration/attention_response_canonical.csv</code> 保留 Qwen L0–35 Top-32；<code>v4_4_5_top6_followup/attention_response/attention_response_topk.csv</code> 从同一 Filestream-audited broad summary 重新汇总 Gemma L0–41 exact Top-6。两者均覆盖 canonical 30 seeds×counts 1–10，并保存 needle/ordinary response 与 specificity。</li>
    <li><code>v4_4_5_followup/span_state_deformation/{{layerwise_state_deformation.csv,summary.json,analysis_audit.json}}</code>：100 confirmation prompts/model 的 clean/needle-corrupt/ordinary-corrupt full-vector capture；Qwen L0–35 共 3,600 rows、Gemma L0–41 共 4,200 rows，50,000 seed-bootstrap 与 exact 2<sup>10</sup> sign-flip。完整 versioned run、代码快照与 162-file persistence audit 保存在 Filestream <code>runs/nonthinking_v445_span_state_deformation_20260817</code>；manifest SHA-256 <code>77032e1e…a3c240</code>。</li>
    <li><code>v4_4_5_followup/state_retention/analysis/{{layerwise_state_retention,window_state_retention,window_condition_levels}}.csv</code> 与 <code>{{summary,selection,analysis_audit}}.json</code>：final N=10 的 20 discovery clean seeds + 10 held-out 三臂 confirmation seeds；endpoint/span-mean、Qwen L0–35 与 Gemma L0–41，50,000 seed-bootstrap、exact 2<sup>10</sup> sign-flip 与 within-family Holm。完整 187-file run/代码/centroid banks 保存在 Filestream <code>runs/nonthinking_v445_state_retention_20260817</code>；source/destination path-size manifest 均为 <code>673338d1…bf767d</code>，per-file content-SHA aggregate 均为 <code>b6bb5acd…e916351</code>。</li>
    <li><code>v4_4_5_followup/exp19/Qwen3-8B/{{paired_serial_effects.csv,serial_summary.json,analysis_audit.json,seed_cluster_reanalysis_audit.json}}</code> 与 <code>v4_4_5_top6_followup/serial_mediation/Gemma4-E4B/serial_summary.json</code>：每模型 1,100-row same-forward partial serial mediation 与 100-row paired-effects 表；两模型均以 10 个 seed means 为独立单位运行 10,000 次 cluster bootstrap 和 1,024 种 exact sign-flip。Qwen 的恢复审计同时记录原始远端/本地逐文件 SHA-256、重新生成 paired CSV 与原文件的 byte-identical 校验，以及旧 seed–count 区间停用。</li>
    <li><code>v4_4_5_followup/exp22_v3/Qwen3-8B</code> 与 <code>v4_4_5_top6_followup/induction_v3/Gemma4-E4B</code>：独立 induction-like gate、300-row canonical matched-edge confirmation 与限定性 negative verdict。Gemma 的行为干预未变，只将 auxiliary broad-bank readout 改为 Top-6，并通过逐行 invariance audit。</li>
    <li><code>v4_4_5_followup/exp23_v2/Qwen3-8B</code> 与 <code>v4_4_5_top6_followup/noise_factorial_v2/Gemma4-E4B</code>：240-row factorial、2,400 endpoint states、400-row outside-context panel、100 exact edge audits与双 control decision。Gemma 的非 broad-bank 字段与原 run 逐行完全一致。</li>
    <li><code>v4_4_5_top6_followup/{{config_diff_audit,FINAL_AUDIT,behavior_invariance_audit,original_preservation_audit}}.json</code>：确认 extension configs 只改变 Gemma retrieval membership 与透明 metadata，冻结 global Top-6 及各层实际 subset，审计 Q22/Q23 非-bank 行为字段的 exact equality，并以 bytes、mtime 与 SHA-256 三重一致性确认 14 个历史 Top-8 source artifacts 未被覆盖。</li>
    <li><code>v4_4_5_top6_followup/comparison_sources</code>：保留历史 Top-8 retrieval geometry 与 L29/L35 subspace 汇总；与原 K8 sweep、原 serial summary、原 attention-response curve 联合生成 Top-6/Top-8 claim-preservation 对照表。该表比较效应量与科学判定，但不冒充预注册的统计等效性检验。</li>
    <li><code>v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html</code>、<code>docs/realistic_niah_v4_4_2.md</code> 与 <code>scripts/analyze_realistic_niah_v4_4_2_counter_geometry.py</code>：Q21 opening-definition cue removal 的冻结 paired states、逐层 CKA/readout 与 intervention 边界。</li>
  </ol>
  </details>
</section>

<section id="extension-audit">
  <h2>8. Non-thinking extension 问题审计：验证了多少，证伪了多少？</h2>
  <p class="lead">为避免把一个宽泛问题同时算作“回答”和“未回答”，这里把 <code>non-thinking extension.md</code> 的提问拆成 25 个可判定命题，并保留原始题号。<strong>已验证</strong>表示有直接实验支持限定后的正命题；<strong>已证伪</strong>只否定表中写出的强版本，不等于证明所有替代理论；<strong>部分回答</strong>表示证据已约束答案但命题本身不允许唯一识别；<strong>未完成</strong>保留为状态类别，但本轮计数为 0；<strong>已关闭</strong>表示当前论文主动不提出相应强主张，因此不再追加实验。原问题 19 已作为整链因果验证放在三个机制步骤之后；原问题 22、23 的完整结果见 Appendix B–C；原问题 21 的完整 cue-removal 证据见 Appendix A；原问题 24 作为主动关闭的范围边界保留在表尾。</p>
  <div class="audit-summary" aria-label="Extension audit status counts">
    <div class="audit-card"><strong>{status_counts['verified']}</strong><span>已验证 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['falsified']}</strong><span>已证伪 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['partial']}</strong><span>部分回答 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['open']}</strong><span>未完成 / 25</span></div>
    <div class="audit-card"><strong>{status_counts['closed']}</strong><span>已关闭 / 25</span></div>
  </div>
  <p>最重要的更新有四条：第一，prompt endpoint 确有可读的 counter-like ordering，但绝对位置足以解释冻结前三维的大部分信号；第二，endpoint 局部寄存器假说被否定，而 full-span distributed evidence 得到强因果支持；第三，broad-bank count subspace 的自然因果使用只出现在 Qwen L21–L23 / Gemma L29，之后转入不同的 late residual representation；第四，同一-forward experiment 19 正式闭合有序部分中介，而 experiments 22/23 分别否定 canonical classical-induction specificity 与简单 prompt-noise attribution package。<strong>目前总评：</strong>证据支持阶段性“形成—局部聚合—晚层执行”，反对一枚固定三维 counter、单一 induction edge registry 或一组简单 nuisance factors 穷尽机制。</p>
  <details class="collapsible-list"><summary>展开 25 项实验设置—结果—结论对照表</summary>
    <div class="table-wrap"><table class="extension-audit"><thead><tr><th>#</th><th>原问题 / 可判定命题</th><th>状态</th><th>实验设置与 control</th><th>具体结果</th><th>目前结论</th></tr></thead><tbody>{extension_audit_rows}</tbody></table></div>
  </details>
  <div class="claim"><strong>Q19 已作为正文最终闭环。</strong>整链实验的直观设计、三项顺序判据、两模型 seed-cluster 结果与 Qwen 数据恢复审计均集中在 Step 3 之后的“整链闭环 · Q19”一节；本审计区只保留第 19 行的状态和结论，避免重复结果或误用旧 seed–count 区间。</div>
  <div class="claim boundary"><strong>审计结论。</strong>25 项中，12 项限定后的正命题已有直接支持，9 个过强机制版本被 matched-control 或 paired robustness experiments 否定，3 项已有实质约束但仍属部分回答，没有仍待运行的预注册机制实验，另有 1 项因当前论文不提出相应强主张而关闭。Q21 只证伪“opening definition cue 必要”，并未删除计数问题与输出指令。尤其不能把“position-confounded counter-like geometry”升级为抽象计数器，也不能把“partial serial mediation”升级为唯一通道。</div>
</section>

<section id="limitations">
  <h2>9. What remains：还差什么没做，以及哪些并非当前主张所必需</h2>
  <p class="lead">现有证据已完成本文中尺度机制链的预注册补全：non-thinking counting 依赖分布式 prompt evidence，经局部 broad retrieval/aggregation 进入晚层可执行 answer state；同一-forward 串联中介在两模型均通过，Qwen 与 Gemma 则采用不同粒度的写入实现。Q22 与 Q23 也已运行并给出限定性负结果。因此当前没有尚待运行、会改变中心机制结论的 GPU 实验；剩余工作主要属于 token-level 精细归因或跨任务外部有效性。</p>
  <div class="table-wrap"><table><thead><tr><th>Open question</th><th>Why it matters</th><th>Decisive experiment</th><th>Priority for the present paper</th></tr></thead><tbody>
    <tr><td>Attention endpoint peaks 的 QK address 与 V content 如何分工？</td><td>这会把 Stage II 从 span/head-bank 层级推进到 token-level circuit；attention map 本身不承担因果结论。</td><td>严格 token registration 后分别 patch Q/K routing、V content 与 pre-O z，并加入 span-interior 和 position-matched controls。</td><td><strong>低优先级。</strong>解析成本高；当前论文不需要声称唯一 token-level path。</td></tr>
    <tr><td>机制是否泛化到更长上下文、更多 count range 与非 city needles？</td><td>现有结论严格限定于冻结的 V4.4 panel、counts 1–10 与当前模板。</td><td>预注册 held-out templates、不同 needle semantics、不同 context lengths 与超出训练范围的 counts。</td><td><strong>投稿前最值得补的泛化实验之一，</strong>但不影响当前 panel 内的因果识别。</td></tr>
  </tbody></table></div>
  <div class="claim boundary"><strong>已证伪与主动关闭的范围。</strong>原问题 21 的强必要性版本已被证伪：只删除开头两句 task/record 定义后，浅层 running ordering 与 seed-held-out readout 几乎保留；但 counting query 与 numeric-output instruction 仍在，所以这不是“全部 task instructions 均非必要”的实验，完整证据见 Appendix A。原问题 24 才是主动关闭：当前报告只主张 position-confounded counter-like record，不主张 final-N invariant abstract counter。原问题 25 的固定 prompt rank-3→answer rank-3 直传强版本也已被证伪，且不是补全原问题 19 的必要条件。</div>
  <div class="claim"><strong>最终结论与优先级。</strong>中心机制链已经闭环，Q22/Q23 的注册强版本也已有负判定；继续租 GPU 不应再用于重复这三项。若资源仍可用，投稿前更高价值的下一步是跨模板、长度、needle semantics 与 count range 的预注册 generalization。QK/V address-content 拆分只在论文要升级为 token-level unique-circuit 主张时才值得做。</div>
</section>

<section id="appendix">
  <h2>Appendix · Q21、Q22、Q23 的完整实验定义与审计结果</h2>
  <p class="lead">Q19 的整链因果实验已放在正文三个机制步骤之后；Qwen 与 Gemma 的 seed-level 推断均已闭合，Qwen 的数据恢复与哈希核验记录在同节。本附录保留主文不宜展开的 robustness、微电路与负结果：Appendix A–C 均已完成并通过 coverage/audit；A 证伪 opening-cue necessity 的强版本，B 与 C 分别否定 classical-induction specificity 与简单 prompt-noise attribution package，同时保留清楚的解释边界。模型特定的 Qwen/Gemma writer 分解紧随其后作为 Appendix D。</p>

  <details class="paper-appendix"><summary>Appendix A · Q21：opening counting-definition cue 的必要性被证伪</summary>
  <h3>Q21 完整实验</h3>
  <div class="experiment"><div class="experiment-label">Completed · V4.4.2</div><div><h4>删除了什么，保留了什么</h4><p>paired intervention 只删除 prompt 开头两句定义：“需要数 passage 中的 city-score audit records”以及“record 的定义”。passage、全部 slots、计数问题、<code>Total:&lt;integer&gt;</code> numeric-output instruction 与 assistant formatting 均保持不变。正式 panel 使用 seeds 1234–1243；下面的 prompt running-index geometry 对每个模型使用 10 个 final-N=10 prompts，每个 prompt 读取第 1–10 个 needle endpoints，因此共有 100 对 cue-present/cue-absent endpoint states。V4.4.2 没有 discovery/confirmation split；ridge 在两种 cue 条件共同拟合的 shared six-PC basis 中使用固定 <span class="math">α=1</span>，并做 leave-one-seed-out prediction，不能把这 10 seeds 重新称为独立 confirmation。</p></div></div>
  <div class="formula"><strong>Centroid-topology linear CKA。</strong>在每个 layer，把十个 running-index centroids 排成矩阵 <span class="math">C<sup>+</sup></span>（cue present）与 <span class="math">C<sup>−</sup></span>（cue absent），按列中心化后形成 Gram matrices <span class="math">K<sup>+</sup>=C̃<sup>+</sup>C̃<sup>+T</sup></span>、<span class="math">K<sup>−</sup>=C̃<sup>−</sup>C̃<sup>−T</sup></span>，再计算 <span class="math">CKA=⟨K<sup>+</sup>,K<sup>−</sup>⟩<sub>F</sub>/(‖K<sup>+</sup>‖<sub>F</sub>‖K<sup>−</sup>‖<sub>F</sub>)</span>。它比较十个 count centroids 之间的关系是否保留，对全局旋转和统一缩放不敏感。<span class="example">例：若 cue removal 只把整条 centroid curve 旋转并放大两倍，两张 Gram matrix 只差统一比例，CKA=1；CKA 接近 0 才表示两种条件下的 centroid relations 不再对齐。</span></div>
  <div class="formula"><strong>Count η² 与 paired interaction η²。</strong><span class="math">η²<sub>count</sub>=SS<sub>between-count</sub>/SS<sub>total</sub></span>，表示完整 hidden-state variation 中由 running-index 分组解释的比例。interaction 先对每个 matched endpoint 求 cue displacement <span class="math">δ=h<sup>−</sup>−h<sup>+</sup></span>，再计算 <span class="math">δ</span> 的 count η²；因此它问“cue 造成的位移是否随 running index 系统变化”，不是行为 accuracy。<span class="example">例：若 displacement 的总平方能量为 100，其中 count-group means 占 48，则 paired interaction η²=0.48；它不表示 accuracy 改变了 48%。</span></div>
  <figure><h4 class="figure-title">图 A1 · 删除 opening definition cue 后，running-index geometry 随层变化</h4><div class="chart-pair"><div>{cue_cka_chart}</div><div>{cue_ridge_chart}</div></div><figcaption>左图横轴是 zero-based transformer layer，纵轴是同层 cue-present 与 cue-absent 十个 count centroids 的 linear CKA；纵轴从 0.94 起截断，用于放大小偏差，虚线 1 表示 centroid relations 完全一致。右图只显示预先用于 prompt geometry 的 Qwen L8 与 Gemma L9；横轴为 model/layer/condition，纵轴是在 pooled shared six-PC basis 中计算的 leave-one-seed-out ridge <span class="math">R²</span>（固定 <span class="math">α=1</span>）。两图共同问 low-dimensional ordering 是否在 cue removal 后仍保留，而不是 cue 是否对完整 residual 完全无影响。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q21 代表层数值</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Layer</th><th>Centroid CKA</th><th>Ridge R² present / absent</th><th>Count η² present / absent</th><th>Paired interaction η²</th></tr></thead><tbody>{cue_appendix_rows}</tbody></table></div></details>
  <div class="claim"><strong>Q21 的精确结论。</strong>Qwen L8 的 CKA=0.9995、ridge R² 0.845→0.840、count η² 0.645→0.633；Gemma L9 的 CKA=0.9999、ridge R² 0.343→0.355、count η² 0.440→0.433。由此可证伪“这两句 opening definitions 是形成有序 running geometry 的必要条件”。但 paired interaction η² 仍为 Qwen 0.484、Gemma 0.332，说明 cue 会以 count-dependent 方式调制完整 state。最重要的边界是：计数问题和输出指令没有删除，所以不能外推成“模型在没有任何 task instruction 时也会形成同一 geometry”。</div>
  </details>

  <details class="paper-appendix"><summary>Appendix B · Q22：是否存在 classical induction-head micro-circuit</summary>
  <h3>Q22 完整实验</h3>
  <details class="collapsible-list"><summary>展开完整实验定义（已完成；audit PASS）</summary>
    <p><strong>为什么现有结果不够。</strong>某个 head 在 needle endpoint 回看较早 spans，只能说明 earlier-span preference；classical induction 还要求它跟随“当前重复 identity → 上一次同 identity 后面的 successor”这一关系，而不是只跟随绝对位置、相对距离或通用 record marker。</p>
    <p><strong>两阶段冻结。</strong>先只用 canonical discovery seeds 1234–1253，按已有 endpoint→earlier-span preference 冻结 candidate heads。随后让这些同一 heads 接受一个独立的 standard induction assay：为每个模型从稳定 single-token pool 生成 30 个固定 base sequences，并各自构造四种完全 token/position-matched 版本。<code>repeated-consistent</code> 含重复 anchor→successor pairs；<code>unique-anchor</code> 消除 previous identity match；<code>successor-reassignment</code> 固定两个 earlier successor 的内容与位置，只交换它们前面的等长 anchor identities，使“当前 anchor 的 previous-match successor”从一个位置移动到另一个位置；<code>same-position ordinary-repeat</code> 保留相同重复/位置统计但打破 anchor→successor relation。该 assay 共 30×4=120 forwards/model，不用 confirmation NIAH outcomes 选 head。</p>
    <div class="formula"><strong>Relation-following score。</strong>对 candidate head <span class="math">h</span>，在当前 anchor query <span class="math">q_t</span> 上，定义 <span class="math">I<sub>h</sub>=mean α<sub>h</sub>(q<sub>t</sub>, successor(previous matching identity))−mean α<sub>h</sub>(q<sub>t</sub>, matched non-successor)</span>。<span class="example">例：若 head 对 identity-defined successor 的平均 attention mass 为 0.20，对同距离 control 为 0.05，则 I<sub>h</sub>=0.15。在 successor-reassignment 中，matching anchor 从 earlier position 1 换到 position 2，而两个 successor positions 均不移动；relation-following mass 应从 successor 1 转到 successor 2。若仍盯原位置，更像 positional routing。</span></div>
    <p><strong>Canonical causal confirmation。</strong>只保留同时满足 canonical earlier-span preference 与 synthetic induction score 的冻结 heads；再在 discovery data 中为每个 retained head 冻结 repeated record-template anchor、current-anchor query offset 与 previous-occurrence successor key offset。在 NIAH confirmation seeds 1254–1263 × counts 1–10 上运行三 arms：natural、candidate-edge removal、matched-control removal，共 300 condition rows/model。candidate arm 在真实 pre-O head slice 上减去每条冻结 natural edge 的 <span class="math">α(q,k)V(k)</span> contribution；control 在同 layer/head 中删除相同 edge 数、相同 key-distance bins 且 natural attention mass 匹配的 non-successor contribution。该操作保留其余 frozen forward，不重新归一化 attention logits，因此是 natural edge-contribution removal，不是 fully renormalized QK counterfactual。随后比较 frozen broad retrieval、correct margin 与最终 expected/strict count。只有 synthetic relation-following、unique-anchor collapse 与 canonical downstream matched-control effect 三者同时成立，才把“induction-like”升级为“classical induction-head mechanism”。</p>
  </details>
  <figure><h4 class="figure-title">图 B1 · Synthetic gate 通过，但 canonical matched-block gate 失败</h4>{exp22_chart}<figcaption>横轴是 counts 2–10 主分析中，冻结 previous-successor candidate-edge removal 相对同 layer/head、同距离、同 edge 数且 natural attention mass 匹配的 ordinary-edge removal 所增加的 expected-count absolute error。正值才支持 classical-induction edge specificity；圆点是 10 个 seed 的平均，横线是 10,000-draw seed bootstrap 95% CI。Qwen 均值和完整 CI 为负；Gemma CI 跨 0，均未满足正向 gate。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q22 synthetic 与 canonical 数值</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Frozen head</th><th>Repeated relation</th><th>Reassignment following</th><th>Unique / ordinary absolute response</th><th>Canonical expected-error candidate−control</th><th>Decision</th></tr></thead><tbody>{exp22_rows}</tbody></table></div></details>
  <div class="claim boundary"><strong>Q22 目前结论。</strong>独立 synthetic assay 确实找到 induction-like relation-following heads，但把同一注册关系带回 canonical NIAH 后，candidate edge 并不比严格 matched ordinary edge 更必要。因此<strong>预注册的 classical induction-head specificity 不受支持</strong>。这只否定该 frozen current-anchor→previous-successor αV contribution 的特异必要性；它不否定 earlier-span routing、其他 head/path registry，也不是 fully renormalized QK deletion。</div>
  </details>

  <details class="paper-appendix"><summary>Appendix C · Q23：identity、context、position 与 outside-context synergy</summary>
  <h3>Q23 完整实验</h3>
  <details class="collapsible-list"><summary>展开完整实验定义（已完成；audit PASS）</summary>
    <p><strong>为什么需要这组补充。</strong><code>docs/realistic_niah_v4.md</code> 中冻结的旧 panel 已构成逐级放松的 robustness ladder：V4.1 固定 position/order/content，V4.2 放开 position，V4.3 再放开固定 fact set 的 order，V4.4 再放开 city-score content。它说明 geometry 不只存在于一个完全固定 prompt，但旧因素不是完整交叉操纵，不能分别估计 identity、context、position 的受控变形及交互。Q23 因而用 factorial 检验一个简单 held-out nuisance model，并用局部 edge removal 检验先前粗粒度 outside-mask 现象能否获得 matched-control specificity。</p>
    <p><strong>Phase A：2×2×2 paired factorial。</strong>在 Qwen L8、Gemma L9 固定读取 running-index states，不重新选层。三个二值因素分别是：（I）active records 的 city/score surface identities 保持原样或用 tokenizer-length-matched pool 随机替换；（C）各 record 周围的 ordinary context 保持原样或在相同 length/depth bins 内跨 slot 置换；（P）record 保持原位置或与 exact-token-length ordinary carrier 交换到预先冻结的 gap-jittered slots，同时保持 record order、总 prompt length 与 answer-query position。每个模型先独立做 tokenizer/span audit。使用全部 30 base seeds × 8 cells × final N=10，共 240 prompt-forwards/model、2,400 endpoint states/model；discovery seeds 1234–1253 只拟合 nuisance model，confirmation seeds 1254–1263 报告效应。</p>
    <div class="formula"><strong>Held-out incremental R²。</strong>先在冻结三维 basis 中减去每个 running index 的 discovery centroid，得到 within-count residual。以 seed 为 group 做 held-out multivariate regression，full model 含 I/C/P 的 main effects 与 interactions；因素 <span class="math">F</span> 的增量定义为 <span class="math">ΔR²<sub>F</sub>=R²(full)−R²(full without every term containing F)</span>。<span class="example">例：full model 在 held-out seeds 上解释 60% residual variance，删去 position 及其 interactions 后只解释 20%，则 ΔR²<sub>P</sub>=0.60−0.20=0.40。它表示这组受控 position manipulations 对 scatter 的增量预测力，不等于自然数据中“40% 神经元由位置产生”。</span></div>
    <p><strong>Phase B：targeted outside-context natural-edge removal。</strong>Discovery seeds 在候选 heads 中冻结一个 source head，并在每个 confirmation unit 内按 natural attention 排序 ordinary halo keys；最多保留 16 条，同时受每个 64-token distance bin 的 distinct non-halo control capacity 约束。Seeds 1254–1263 × counts 1–10 使用四 arms：natural、candidate halo-edge removal、exact-distance random control、同 bin attention-mass control，共 400 rows/model。三种 removal arms 的 edge 数严格相等且大于 0，每条 control 与 candidate 同 distance bin、key 不重用。干预在 answer-query source-head pre-O slice 减去 frozen natural <span class="math">αV</span> contribution；不重算归一化 QK。只有 candidate removal 在 expected error 上同时超过两个 controls，才支持注册的 specificity claim。</p>
    <p><strong>解释边界。</strong>Factorial 的 ΔR² 是受控 deformation 对 held-out residual 的增量预测力，不是自然方差份额；Phase B 识别一个冻结 source-head registry 的特异必要性，不是 outside-context token census 或 pathway-uniqueness test。Negative result 因而约束当前简单解释，但不能把 observational scatter 唯一分配给某个来源，也不能推出 distributed outside-context synergy 不存在。</p>
  </details>
  <figure><h4 class="figure-title">图 C1 · 三类受控变形没有形成稳定 held-out nuisance model</h4>{exp23_factor_chart}<figcaption>横轴是 confirmation seeds 上的 incremental ΔR²：完整 I/C/P 主效应与交互模型的 held-out R²，减去删除所有含该因素项后的 R²。正值表示该因素在这套受控 manipulation 下增加 held-out prediction；零线表示没有增量。Qwen position 为 +0.0175，Gemma identity 为 +0.0031，其余接近或低于 0；更关键的是两个 full-model held-out R² 本身为 −0.0221 与 −0.0893，说明整套模型不如用 confirmation mean 预测。故这些条形不能解释为自然 prompt noise 的 variance share。</figcaption></figure>
  <figure><h4 class="figure-title">图 C2 · Selected outside-halo edge removal 未超过两个 matched controls</h4>{exp23_specificity_chart}<figcaption>横轴是 candidate halo-edge removal 相对各 matched control 多造成的 expected-count absolute error，单位 counts；圆点为 10-seed 均值，横线为 10,000-draw seed bootstrap 95% CI。Distance-random control 严格匹配 layer/head、edge count 和每条 key 的 distance bin；attention-mass control 还在同一 distance bin 内匹配 natural pre-intervention attention。两模型四个 CI 全部跨 0，因此注册判据 <code>candidate_exceeds_both_controls</code> 均为 false。</figcaption></figure>
  <details class="collapsible-list"><summary>展开 Q23 精确结果</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th>Full held-out R²</th><th>ΔR² I / C / P</th><th>Frozen source head</th><th>Expected error vs distance control</th><th>Expected error vs attention-mass control</th><th>Decision</th></tr></thead><tbody>{exp23_rows}</tbody></table></div></details>
  <div class="claim boundary"><strong>Q23 目前结论。</strong>预注册的强解释包未通过：I/C/P manipulation 没有形成稳定的 held-out additive/interaction model，selected outside-halo edges 也没有超出两个 matched controls 的特异必要性。这个负结果不等于 identity、context 或 position 在自然数据中“没有作用”，也不等于所有 outside context 都无用；它只说明当前简单分解不能唯一解释 observational scatter，且冻结的单一 source-head αV halo registry 不是可确认的特异通路。</div>
  </details>
</section>

</main>
</article>
<script>
const GEOMETRY={payload_json};
const COLORS=['#2563eb','#0891b2','#0f766e','#65a30d','#ca8a04','#d97706','#ea580c','#dc2626','#9333ea','#4f46e5'];

function collapseReportLists() {{
  const targets=[...document.querySelectorAll('.table-wrap, main ul, main ol')];
  for(const node of targets) {{
    if(node.closest('details.collapsible-list')) continue;
    const details=document.createElement('details');
    details.className='collapsible-list';
    const summary=document.createElement('summary');
    const isTable=node.classList.contains('table-wrap');
    const count=isTable ? node.querySelectorAll('tbody tr').length : node.children.length;
    summary.textContent=isTable ? `展开表格明细（${{count}} 行）` : `展开列表明细（${{count}} 项）`;
    node.parentNode.insertBefore(details,node);
    details.append(summary,node);
  }}
}}
collapseReportLists();

class PointCloud3D {{
  constructor(canvasId, modelSelectId, layerSelectId, site, defaults) {{
    this.canvas=document.getElementById(canvasId);
    this.modelSelect=document.getElementById(modelSelectId);
    this.layerSelect=document.getElementById(layerSelectId);
    this.site=site;
    this.defaults=defaults;
    this.ctx=this.canvas.getContext('2d');
    this.yaw=-0.7; this.pitch=0.35; this.dragging=false; this.last=null;
    this.modelSelect.addEventListener('change',()=>{{this.layerSelect.value='';this.populateLayers();this.draw();}});
    this.layerSelect.addEventListener('change',()=>this.draw());
    this.canvas.addEventListener('pointerdown',e=>{{this.dragging=true;this.last=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);}});
    this.canvas.addEventListener('pointermove',e=>{{if(!this.dragging)return;const dx=e.clientX-this.last[0],dy=e.clientY-this.last[1];this.yaw+=dx*.009;this.pitch=Math.max(-1.35,Math.min(1.35,this.pitch+dy*.009));this.last=[e.clientX,e.clientY];this.draw();}});
    this.canvas.addEventListener('pointerup',()=>{{this.dragging=false;}});
    this.canvas.addEventListener('pointercancel',()=>{{this.dragging=false;}});
    this.canvas.addEventListener('dblclick',()=>{{this.yaw=-0.7;this.pitch=.35;this.draw();}});
    new ResizeObserver(()=>this.draw()).observe(this.canvas);
    this.populateLayers();
    this.draw();
  }}
  populateLayers() {{
    const model=this.modelSelect.value;
    const layers=Object.keys(GEOMETRY[this.site][model]).map(Number).sort((a,b)=>a-b);
    const previousRaw=this.layerSelect.value;
    const previous=previousRaw===''?Number.NaN:Number(previousRaw);
    const preferred=Number.isFinite(previous)&&layers.includes(previous)?previous:this.defaults[model];
    this.layerSelect.replaceChildren(...layers.map(layer=>{{const option=document.createElement('option');option.value=String(layer);option.textContent=`L${{layer}}`;return option;}}));
    this.layerSelect.value=String(layers.includes(preferred)?preferred:layers[0]);
  }}
  rotate(p) {{
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const x=cy*p[0]+sy*p[2], z=-sy*p[0]+cy*p[2];
    return [x,cp*p[1]-sp*z,sp*p[1]+cp*z];
  }}
  draw() {{
    const model=this.modelSelect.value;
    const layer=this.layerSelect.value;
    const payload=GEOMETRY[this.site][model][layer];
    if(!payload)return;
    const rows=payload.rows;
    const rect=this.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
    if(rect.width<20)return;
    this.canvas.width=Math.round(rect.width*dpr);this.canvas.height=Math.round(rect.height*dpr);
    const ctx=this.ctx;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
    const raw=rows.map(r=>[+r[6],+r[7],+r[8]]),mins=[0,1,2].map(k=>Math.min(...raw.map(p=>p[k]))),maxs=[0,1,2].map(k=>Math.max(...raw.map(p=>p[k])));
    const center=mins.map((v,k)=>(v+maxs[k])/2),span=Math.max(...maxs.map((v,k)=>v-mins[k]))||1;
    const norm=p=>p.map((v,k)=>(v-center[k])/span*2);
    const scale=Math.min(rect.width*.34,rect.height*.34),ox=rect.width*.50,oy=rect.height*.48;
    const project=p=>{{const r=this.rotate(p),persp=1/(1+r[2]*.13);return [ox+r[0]*scale*persp,oy-r[1]*scale*persp,r[2]];}};
    const axes=[[[0,0,0],[1.15,0,0],'PC1'],[[0,0,0],[0,1.15,0],'PC2'],[[0,0,0],[0,0,1.15],'PC3']];
    ctx.lineWidth=1.2;ctx.strokeStyle='#9aa5b4';ctx.fillStyle='#536074';ctx.font='12px system-ui';
    axes.forEach(a=>{{const p0=project(a[0]),p1=project(a[1]);ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p1[0],p1[1]);ctx.stroke();ctx.fillText(a[2],p1[0]+5,p1[1]-5);}});
    const points=rows.map(r=>{{const label=this.site==='prompt'?+r[5]:+r[5];const p=project(norm([+r[6],+r[7],+r[8]]));return {{p,label}};}}).sort((a,b)=>a.p[2]-b.p[2]);
    points.forEach(o=>{{ctx.globalAlpha=.42;ctx.fillStyle=COLORS[o.label-1];ctx.beginPath();ctx.arc(o.p[0],o.p[1],2.8,0,Math.PI*2);ctx.fill();}});ctx.globalAlpha=1;
    const grouped=new Map();rows.forEach(r=>{{const label=+r[5];if(!grouped.has(label))grouped.set(label,[]);grouped.get(label).push([+r[6],+r[7],+r[8]]);}});
    const centroids=[...grouped.entries()].sort((a,b)=>a[0]-b[0]).map(([label,ps])=>{{const mean=[0,1,2].map(k=>ps.reduce((s,p)=>s+p[k],0)/ps.length);return {{label,p:project(norm(mean))}};}});
    ctx.strokeStyle='#26364d';ctx.lineWidth=2;ctx.beginPath();centroids.forEach((o,i)=>{{if(i===0)ctx.moveTo(o.p[0],o.p[1]);else ctx.lineTo(o.p[0],o.p[1]);}});ctx.stroke();
    centroids.sort((a,b)=>a.p[2]-b.p[2]).forEach(o=>{{ctx.fillStyle=COLORS[o.label-1];ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(o.p[0],o.p[1],8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font='600 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(o.label),o.p[0],o.p[1]+.5);}});
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px system-ui';
    const legendY=rect.height-24,step=Math.min(42,(rect.width-28)/10);for(let n=1;n<=10;n++){{const x=14+(n-1)*step;ctx.fillStyle=COLORS[n-1];ctx.beginPath();ctx.arc(x,legendY,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#4b5565';ctx.fillText(String(n),x+8,legendY+4);}}
    ctx.fillStyle='#4b5565';ctx.font='600 12px system-ui';ctx.fillText(`${{model}} · L${{payload.layer}} · ${{rows.length}} states`,14,22);
  }}
}}
new PointCloud3D('prompt-canvas','prompt-model','prompt-layer','prompt',{{'Qwen3-8B':8,'Gemma4-E4B':9}});
new PointCloud3D('answer-canvas','answer-model','answer-layer','answer',{{'Qwen3-8B':28,'Gemma4-E4B':37}});

function setupAttentionGallery(root) {{
  const head=root.querySelector('[data-gallery-head]');
  const prompt=root.querySelector('[data-gallery-prompt]');
  const panels=[...root.querySelectorAll('[data-gallery-panel]')];
  const refresh=()=>{{
    for(const panel of panels) {{
      panel.hidden=!(panel.dataset.head===head.value && panel.dataset.prompt===prompt.value);
    }}
  }};
  head.addEventListener('change',refresh);
  prompt.addEventListener('change',refresh);
  refresh();
}}
document.querySelectorAll('[data-attention-gallery]').forEach(setupAttentionGallery);
</script>
</body>
</html>
"""

    # Keep the source template convenient to maintain while enforcing the
    # publication-facing, mechanism-first narrative in the rendered report.
    section_order = (
        "summary",
        "baseline",
        "representation",
        "formation",
        "retrieval",
        "write",
        "integrated-chain",
        "ledger",
        "extension-audit",
        "limitations",
        "appendix",
        "ov-write",
    )
    main_start = html_doc.index("<main>") + len("<main>")
    main_end = html_doc.index("</main>", main_start)
    main_html = html_doc[main_start:main_end]
    section_matches = list(
        re.finditer(r'<section id="([^"]+)">.*?</section>', main_html, re.DOTALL)
    )
    rendered_sections = {match.group(1): match.group(0) for match in section_matches}
    if tuple(rendered_sections) != (
        "baseline",
        "summary",
        "integrated-chain",
        "representation",
        "formation",
        "retrieval",
        "write",
        "ov-write",
        "ledger",
        "extension-audit",
        "limitations",
        "appendix",
    ):
        raise RuntimeError(
            "Unexpected report sections before mechanism-first reordering: "
            f"{tuple(rendered_sections)}"
        )
    reordered_main = "\n\n" + "\n\n".join(
        rendered_sections[section_id] for section_id in section_order
    ) + "\n"
    html_doc = html_doc[:main_start] + reordered_main + html_doc[main_end:]

    # Every result figure begins with a plain-language reading contract.  This
    # keeps the reader from having to infer the estimand from the axes or wait
    # until the technical caption after the plot.  The three fields are kept
    # separate on purpose: object, reading rule, and a concrete one-unit example.
    figure_guides: dict[str, tuple[str, str, str]] = {
        "机制图 M0 · 从 needle-span evidence 到数字输出的分层计算链": (
            "这是整篇报告的<strong>机制示意图</strong>，不是新增的一组数值结果。上排把一次完整 forward 压成三步：形成 span evidence、由 answer query 取回并聚合、将聚合结果整合为可执行 answer state 后输出；下排把三步映射到 Qwen 与 Gemma 已被实验约束的大致层段。",
            "从左向右读信息流：needle 内容先改变完整 spans 内多个 token 的 hidden states；answer query 再用 broad attention 选择远端 positions，并把各 head 的 post-O writes 聚合进 query residual；后续 blocks 将结果重参数化为可执行 state并写向数字 logits。层段允许重叠，不能把相邻色块理解为单层硬开关。",
            "假设 passage 有三条有效 records：将它们换成等长普通文本会改变对应 span hidden states；固定这份 corrupt 输入、只把三段 clean states 写回，答案又可被救回。末尾 query 随后回看三处并合计 content，形成更接近 count=3 的 late state，再把数字 3 的 logit 推高。",
        ),
        "机制图 M1 · 同一 forward 中的有序 partial serial mediation": (
            "六根柱回答三个简单问题：恢复正确的 needle spans 能救回多少答案；救回之后，阻断 retrieval 会损失多少收益；阻断更晚的 count state 又会损失多少收益。绿色是 Qwen，紫色是 Gemma。",
            "横轴单位是 counts。柱越长，表示该操作相对严格 matched control 对答案的影响越大。Source、retrieval、late 三根柱来自不同配对问题，<strong>不能相加</strong>成“总解释比例”。",
            "正确答案为 8：恢复真实 spans 后模型从倾向答 3 变成倾向答 6，相对恢复无关文本多减少 3 counts 的误差，source repair 就是 3。若删除 retrieval 的正确方向比删除同样大的无关方向多让答案退回 0.5 count，retrieval mediation 就是 0.5。",
        ),
        "图 1a · Qwen3-8B baseline": (
            "这张热图把 Qwen 在每个真实 needle 数量 <span class=\"math\">N=0…10</span> 上的外部行为压成两行：上行是答对比例，下行是平均差几个 count。",
            "从左到右选择 gold count；上排数字越高、绿色越深越好，下排 absolute error 越低越好。每个格子聚合 30 个 prompts，因此这里没有 hidden state 或干预效应。",
            "看 <span class=\"math\">N=8</span>：上格 23% 表示 30 例中约 7 例精确答 8；下格 1.73 表示输出平均离 8 相差 1.73 counts。",
        ),
        "图 1b · Gemma4-E4B baseline": (
            "这张图与左侧 Qwen 使用完全相同的 30-sample-per-count 行为统计，只换成 Gemma，便于判断两个机制解释面对的错误分布是否相同。",
            "列仍是 gold count，上排是 exact accuracy，下排是 mean absolute error；不要把颜色当作模型置信度。",
            "看 <span class=\"math\">N=6</span>：10% 表示约 3/30 例答对；1.53 表示平均离正确答案 6 相差 1.53 counts。",
        ),
        "图 2a · Prompt needle-end running-index manifold": (
            "这是一张可旋转的三维<strong>描述性点云</strong>：每个小点是某个 seed 在第 n 条 needle 的末端 hidden state；颜色/编号是当时已经读到第几条 needle。",
            "大圆点是同一 n 的跨-seed centroid，连线只帮助看 n=1→10 的平均轨迹。PC1–PC3 只在当前层定义，轴没有 count 单位；点云分开表示三维中可分，不表示模型自然依赖这三维。",
            "若 30 个 seeds 的第 3 条 needle states 聚在编号 3 附近，而第 4 条整体沿同一方向移动，就形成一个 counter-like step；若两团大量重叠，单样本紧致度仍然低。",
        ),
        "图 2b · Answer-query consolidated-count manifold": (
            "这张三维点云画的是生成首个数字前 answer query 的 hidden state；每个点的标签是整篇 prompt 的最终 gold count，而不是当前 running index。",
            "同色小点是一类 count 的不同 seeds，大编号圆点是 centroid。图只展示 frozen PC1–PC3 投影；真实 classifier 使用更高维 PCA-32，因此视觉重叠不等同于完全不可读。",
            "Gold=8 的十个 confirmation states 若大多靠近 centroid 8，nearest-centroid 会判对；若它们散到 7/9 的云中，三维图仍可有一条平均轨迹，但类别并不紧致。",
        ),
        "图 2c · Answer-query classifier accuracy 随 gold count 的变化": (
            "这条曲线把代表层 grouped-OOF classifier 的整体 accuracy 拆成 N=1 到 N=10 十个 gold-count 条件，检查较大 count 是否系统性更难区分。",
            "横轴是 gold count，纵轴是该 count 的 20 个 held-out seed states 中被分对的比例；竖线是 pointwise Wilson interval。它是 hidden-state separability，不是模型生成答案的行为准确率。",
            "Qwen N=6 的 7/20 给出 35% accuracy；N=10 的 12/20 给出 60%。所以中高 count 较难是总体趋势，但“随 N 单调下降”被 N=10 的边界回升直接否定。",
        ),
        "图 3 · Counter manifold 的可读性与低维结构如何跨层变化（descriptive baseline）": (
            "这组折线逐层回答两个描述性问题：count 是否能从 held-out state 读出，以及全部样本/十个 centroids 的方差有多少落在前三个 PCA 方向。",
            "横轴是 layer。Probe/classifier 越高表示 held-out 可读性越强；rank-3 capture 越高表示相应对象更低维。它们都是 representation diagnostics，不是 causal effect，也不能据峰值自动选机制层。",
            "Answer classifier accuracy=0.50 可读作 100 个 held-out seed–count states 中约 50 个分到正确 count；centroid rank-3=0.98 只表示十个平均点的 98% 方差在三维内，不表示 98% 单样本能分类。",
        ),
        "图 3b · Frozen PC1–PC3 的 counter-property 跨层审计": (
            "这四个 panel 不再只问“能不能预测 count”，而问十个 prompt-end centroids 是否像逐步累加器：近直线、距离随 count gap 增长、相邻 steps 同向、并跨 seed split 稳定。",
            "横轴都是 layer；四个纵轴越接近 1 越像稳定有序轨迹。大圆点只是报告代表层。这里没有 removal/patch，因此仍不能证明 causal counter。",
            "理想一维 centroids [0,1,2,3] 会给出 line R²≈1、distance-gap ρ≈1、step cosine≈1；若 steps 是 [+3,−1,+3]，count 也许仍可回归，但同向性会下降。",
        ),
        "图 3c · Active-needle input intervention 引起的逐层 full-span state effect": (
            "这组图直接测量第一条因果箭头：同一个 prompt 保持长度与 query 不变，分别不改输入、替换全部 active needles、或替换等长度/等总 token-budget 的 ordinary spans；随后在每一层读取注册位置的完整 residual vectors。每模型上图画两种替换造成的原始 RMS state change，下图画 needle 减 ordinary 的 matched specificity。",
            "横轴是 zero-based post-block layer。上图纵轴是每个 hidden coordinate 的 RMS change，单位为该层 residual 数值；下图纵轴是两臂之差，零表示 needle replacement 并不比普通文本替换造成更大的 state deformation。实心点只表示该层 exact seed sign-flip nominal p<0.05；竖线是由独立 behavior-restoration 实验冻结的 reuse boundary。",
            "若 L8 的 needle/ordinary RMS changes 分别为 0.42/0.40，说明两种输入替换都会让 state 改变，但 needle-specific effect 只有 0.02；若为 0.42/0.10，matched effect 才是 0.32。即使深层差值仍为正，也要看图 4b 的 state restoration 是否还能救答案，才能判断该差异是否仍可被行为使用。",
        ),
        "图 3d · Needle corruption 是否把 span state 推离正确 running-index manifold": (
            "这组图不再只量 state 改了多少，而是问第 k 条 occurrence 的 endpoint/全 span token 均值在输入干预后，是否仍靠近由独立 clean seeds 定义的第 k 条 running-index centroid。第一张条形图汇总三种条件的绝对距离，后两张折线画 needle-corrupt 减 ordinary-corrupt 的逐层差值。",
            "条形图横轴与折线纵轴都以对应 discovery class 的 RMS radius 为单位；约 1 是普通 held-out clean state 的典型离心尺度。折线大于 0 表示 needle corruption 比等 token-budget ordinary corruption 更强地破坏正确 count cloud。竖线是独立 behavior-restoration 冻结的窗口边界；逐层点不用于重选层。",
            "若第 5 条 state 到正确 centroid 的 clean/ordinary/needle distances 为 0.8/1.0/2.6，则主差值是 1.6 radii。它说明 needles 维持 count-specific organization，但只有随后把 clean full-span states 写回并救回答案，才能建立 state→count 的第二条因果箭头。",
        ),
        "图 4 · “输入 evidence 必要”与“decoded endpoint subspace 必要”不是同一命题": (
            "这张条形图把两个不同强度的 causal question 放在同一 count-error 单位上：替换完整 active-needle 文本是否伤行为，以及只删除 endpoint 的线性 rank-3 direction 是否比等范数 nuisance removal 更伤。",
            "横轴是 candidate 相对 matched control 多造成的 absolute error。长柱支持 active input evidence 必要；短柱接近 0 表示当前 endpoint rank-3 没显示局部方向特异必要性。两类操作规模不同，不是在比较同一个 component 的大小。",
            "Gold=8 时，needle/control token replacement 若输出 1/7，则 specificity=7−1=6 counts；rank-3/orthogonal removal 若输出 6/7，则 specificity=2−1=1 count。",
        ),
        "图 4b · Canonical dense span restoration 的完整逐层曲线": (
            "这张图对<strong>每个候选层单独重跑一个 corrupt forward</strong>：只在该层把所有 active needle spans 的 clean hidden states 恢复一次，然后看最终 expected-count error 能否比等 token-budget ordinary-span restoration 更小。",
            "横轴是 restoration layer，不是“后层正在读取哪个历史层”；纵轴 <span class=\"math\">S_restore(ℓ)</span> 是多修复了多少 counts。正值表示在该深度恢复 evidence 还来得及救答案；曲线突然降到 0 表示该 source 已错过可复用窗口。",
            "Gold=8 的 corrupt run error=5；在 L10 恢复 full spans 后 error=2，而 ordinary restoration 后仍为 4，则 L10 specificity=4−2=2 counts。换到 L24 若两者都为 4，L24 点就是 0。",
        ),
        "图 4c · Canonical full-span restoration 对后续 broad retrieval attention 的逐层影响": (
            "这张图沿用图 4b 的逐层 source restoration，但结果不读最终答案，而读同一次 forward 中后续 answer query 的 broad-head attention 是否重新覆盖 active needle spans。",
            "横轴是 source 被恢复的 layer；纵轴是 true-needle restoration 引起的 attention response 减 ordinary restoration response。蓝线看总 needle mass，虚线 broad score 还要求覆盖多个 spans。它定位“恢复到多深仍会重配置后续 retrieval”，不表示 retrieval head直接读取了某个旧层。",
            "若 L16 full-span restoration 使 broad score 从 0.20 升到 0.50，而 ordinary restoration 只升到 0.22，则 ΔB=0.28；若在 L27 两者都不变，则 ΔB≈0。",
        ),
        "图 4d · Discovery 预冻结 landmarks 在 confirmation 上的 readout": (
            "这张条形图不是重新寻找最佳层，而是把 discovery seeds 预先定义的 early plateau、half-boundary 和 near-zero boundary 拿到独立 confirmation seeds 上读取 effect。",
            "横轴仍是 full-span restoration 相对 ordinary control 的 error repair，单位 counts；每根 bar 对应一个事先冻结的阶段/层。正值越大表示该 landmark 在新 seeds 上仍可复用。",
            "若 discovery 把 L19 定义为 half-boundary，confirmation 上该层 full/ordinary errors 为 1.7/3.0，则 bar=1.3；这验证的是 L19 这个预注册位置，不是从 confirmation 再挑一个更高点。",
        ),
        "图 5 · Answer-query broad-score attention maps": (
            "这是 layer×head 的<strong>聚合 routing 地图</strong>。每个格子把某一 head 从最终 answer query 指向所有完整 active needle spans 的 attention mass 与多-span coverage 合成 broad score。",
            "横轴=head，纵轴=layer；颜色越深表示 discovery prompts 上平均更广泛地回看 needles。黑框和数字是随后冻结做 causal test 的 heads。它不是单个 prompt 的 token×token attention map，也不显示写入内容。",
            "某个 L21H7 格子很深，只表示该 head 在多个样本中把较多 attention 分给多条 needles；要说它参与计数，还必须看到消融它比同层 random head 更伤答案。",
        ),
        "图 5b · 自然 forward 的全文 attention routing：needle 城市、haystack 分箱与 span mass": (
            "这是<strong>全文尺度</strong>的 answer-query attention：可在 discovery-frozen Qwen top-4 heads 与预先固定的 seed 1254、N=3/6/9 三条自然文本之间切换。它同时回答三件事：每条城市记录在全文哪里、普通 haystack 的 attention 分布在哪里、每条 needle 总共得到多少 mass。",
            "顶部组成条先给出五类区域的精确总 mass。A 图从左到右、换行继续，每格固定 64 tokens；浅粉→深红表示格内<strong>总</strong> attention 的单调对数色阶，tooltip 给精确值。普通格不写字，城市名只放在其 needle 起始格内；深红上边标 needle，黄色下边标附近等长普通文本对照（冻结字段名 hard_negative）。黄色格不要求与红格坐标重合。B 图红条是整条 needle 的 token weights 之和，12 个 panels 共用百分比上限。",
            "若 Vilnius 位于 token 1,967，它的城市名就嵌在覆盖该位置的 64-token 格内；该格颜色深浅描述局部窗口的总 mass，而下方 Vilnius=13.72% 才是整条 record 的 span mass。周围无红黄边的 passage 格显示普通 haystack mass；这些 routing 数值仍不是 causal contribution。",
        ),
        "图 5c · 选定 head–prompt 的 needle-token attention 细图": (
            "这张图对图 5b 当前选中的同一 head–prompt row 做<strong>token 级放大</strong>：只展开 active needle spans，不把整篇普通文本逐词铺开。",
            "每行对应一个 needle；span mass 是整行权重之和，token 红色越深表示该 token 的 attention 越大。顶部小条仍统计全序列五类位置，因此能看见针内细节而不丢掉全文分母。不同 panel 的 token 色深按各自最大值归一，精确跨 panel 比较应读百分比。",
            "一条 record 中若城市 token 得到 0.2%、句号得到 4%，句号会更深；这可能反映模板边界 routing，但不能据此说‘句号存了 count’，因为图没有计算 value/output contribution。",
        ),
        "图 6a · Broad-head ablation 的 absolute-shift 剂量曲线": (
            "这条剂量曲线逐步同时消融 discovery-frozen top-K broad heads，并把 top-K 原始行为损伤、同层 random-head 原始损伤及二者差值画在一起。",
            "横轴是 K 占 discovery-eligible heads 的比例。模型色实线=top-K raw shift，灰线=random raw shift，紫色点线=ranked−random；前两者回答‘各自移动多少’，紫线才回答候选 heads 是否比删同层普通 heads 更特异。",
            "Qwen K32 的实色/灰色点是 1.750/0.127 counts，紫色点因此是 1.623；Gemma K6 的对应值为 0.990/0.165/+0.825，且 K6 明确标为 post-hoc dose。",
        ),
        "图 6b · Clean-correct correct→wrong damage 的剂量曲线": (
            "这张图只保留模型原本答对的 prompts，把 ranked/random 两臂各自将正确答案变错的概率与两者差值同时画出。",
            "横轴仍是 eligible-head proportion。模型色实线/灰色虚线是两臂 raw correct→wrong rate，紫色点线是 ranked−random；只有紫色差值隔离了删掉相同层数、相同 head 数本身的损伤。",
            "若 ranked 令 30% 的 clean-correct prompts 变错、random 令 10% 变错，图上三条线分别在 0.30、0.10、0.20；紫色 0.20 就是额外 20 percentage points。",
        ),
        "图 6c · 4.3 实际测量的对象：从多头读取到一个合计写入向量": (
            "这是<strong>计算对象示意图</strong>，不是新的结果：它说明 representation analysis 分类的并非 raw attention，而是 frozen broad heads 在同一层、answer query 处的实际 post-O writes 之和。",
            "按箭头从左到右读：多个 spans→多头读取→每头写入 residual→同层求和→held-out readout。只有最后的合计向量进入 classifier。",
            "若三个 heads 的 post-O writes 分别是向量 [1,0]、[0,2]、[−0.2,0.5]，被分析的 broad-bank state 是它们的和 [0.8,2.5]，不是三张 attention map。",
        ),
        "图 6d · Broad-bank 合计输出的 held-out exact-count readout": (
            "这张折线图逐个 frozen layer 测量图 6c 的 broad-bank sum 能否在未见 seeds 上区分最终 count；它描述 retrieved content 的可读性。",
            "横轴是实际测试的 layer，纵轴是十类 accuracy；两条线分别是线性 classifier 与 nearest centroid，0.10 虚线是 chance。相邻点连线仅便于看，不代表未测试层。",
            "Qwen 某层 accuracy=0.54 表示 100 个 confirmation seed–count states 中约 54 个被分到正确 count；它不表示该层贡献了 54% 的行为。",
        ),
        "图 6e · 正常运行依赖与上游修复中介的四条件配对设计": (
            "这是四条件 matched-intervention 设计图，而非效应量结果图。左侧从正常 forward 开始；右侧从一个严格定义的 restored receiver 开始：输入 needles 仍被等长 ordinary text 覆盖，但 post-block L8 的全部 needle-position states 被同一 prompt 的 clean states 替换一次。",
            "Restored receiver 不改 answer-query state，L9 以后不再 clamp；两侧随后都在同一 frozen retrieval layer、同一 answer-query output span 删除 matched-norm 分量。两者的 aligned-minus-orthogonal 计算相同，差别仅是上游 span evidence 是否已在 L8 被恢复。",
            "Gold=8：若恢复 8 个 span tokens 后，restored+aligned/orthogonal 两臂的误差为 1.2/0.8，则上游修复中介效应=0.4 count；它表示 aligned component 承接了部分修复收益，不表示恢复了 40% 的 tokens。",
        ),
        "图 6f · Frozen retrieval rank-3 subspace 的逐层依赖与中介效应": (
            "该图将正常运行依赖效应与上游修复中介效应显示在预冻结 retrieval layers 上，分别定位同一 fitted rank-3 basis 的自然使用窗口和 source-repair mediation window。",
            "横轴是实际 intervention layer；纵轴是 count-aligned removal 相对 equal-norm orthogonal removal 多造成的 expected-count error。实线为正常运行依赖效应，虚线为上游修复中介效应；0 只表示当前层未检测到该 frozen basis 的特异作用。",
            "Qwen L23 的实线 0.333 表示：在 100 个 paired seed–count units 上，count-aligned removal 比 orthogonal control 平均多造成 0.333 count error。它不是 33.3%，也不表示该层解释了三分之一机制。",
        ),
        "图 7 · Answer state 的因果可执行性在中后层出现": (
            "这条曲线做 full-state donor patch：在某一层把 donor prompt 的完整 answer-query residual 写到 receiver 的同层 query，再让 receiver 继续 forward，观察输出是否采用 donor count。",
            "横轴是 patch layer；纵轴是 eligible donor-prediction adoption rate。接近 1 表示完整 state 在该深度几乎足以驱动 donor 答案；它不说明 state 中哪一个线性方向负责。",
            "Receiver gold=3、donor gold=8：若在 L29 patch 后模型由答 3 改答 8，该 pair 记 adoption=1；仍答 3 记 0。100 pairs 中 80 个采用 donor，就是 0.80。",
        ),
        "图 8 · Answer-query count subspace 的必要性随深度增加": (
            "这张逐层曲线在 answer query 删除 discovery-fitted count rank-3，并与同层、同位置、同实际删除 norm 的 orthogonal component 配对，问晚层线性 count direction 是否被自然使用。",
            "横轴是 removal layer；纵轴是 aligned removal 相对 orthogonal removal 多增加的 absolute error。正值越大，方向特异必要性越强；这与图 7 的 full-state sufficiency 是互补命题。",
            "Gold=8 时 aligned removal 输出 6、orthogonal removal 输出 7，误差 2/1，因此该样本的方向特异损伤=1 count。",
        ),
        "图 9 · Answer-query 三维相邻层映射：error 与跨层 cosine": (
            "这四幅图检查相邻层 count geometry 是否能由一个局部三维 map 预测，以及连续两个 maps 在完整 hidden space 中是否朝相近方向；它描述跨 block 的坐标重参数化。",
            "横轴标的是 map 的 target layer，所以 29 代表 L28→29。左列 error 越低越可预测/可复现；右列 operator cosine 越接近 1 越连续。高 cosine 不要求三维坐标轴逐项相同。",
            "若用 L28 的三个 centroid coordinates 经拟合 map 能把 L29 centroids 预测到很小误差，L28→29 点较低；若该 ambient operator 与 L29→30 的 cosine=0.9，说明两步方向相近但不完全相同。",
        ),
        "图 10 · Aligned 1× 跨一个 block 的方向选择性传播": (
            "这张条形图在每个 confirmation receiver prompt 的 source-layer answer query 注入一个 frozen centroid-chord direction 或等 norm orthogonal direction，只继续一个 transformer block，再测 hidden-state change 沿 target-layer chord 的分量。每条 bar 汇总 10 seeds×四个预注册方向（1→2、2→1、5→6、6→5）。",
            "横轴是传播量 <span class=\"math\">F</span>，单位为 target centroid chord；aligned 接近 1 且 orthogonal 接近 0 表示下一 block 选择性保留 count-aligned change。1 只是沿该方向移动一个 chord unit，不等于完整到达某个 donor centroid。",
            "以 receiver=1、donor=2 为例：若下一层 change 沿 frozen 1→2 target chord 移动 0.95 个 chord unit，则 F≈0.95；同 norm orthogonal injection 的 F≈0 才提供方向特异性。这里 donor 是 discovery centroid label，不是另一条 confirmation prompt 的完整 state。",
        ),
        "图 A1 · 删除 opening definition cue 后，running-index geometry 随层变化": (
            "这组 appendix 曲线比较保留/删除开头 counting instruction 时，十个 count centroids 的关系是否保持，以及代表层的 held-out running-index readout 是否仍然存在。",
            "左图横轴=layer、纵轴=两条件 centroid geometry 的 linear CKA；右图是代表层 ridge R²。CKA 近 1 只表示相对几何近似，不表示完整 states 或行为完全相同。",
            "若 cue-present/absent centroids 只是整体旋转，CKA 可接近 1；但若删 cue 后 classifier R² 从 0.8 降到 0.4，说明可读强度仍被调制。",
        ),
        "图 B1 · Synthetic gate 通过，但 canonical matched-block gate 失败": (
            "这张区间图检验 classical induction 的关键 causal specificity：删除真实 previous→successor candidate edges 是否比删除同 head、同距离、同 edge 数、attention mass 匹配的 ordinary edges 更伤计数。",
            "横轴是 candidate-minus-control 的 expected-count error；正值且区间完全大于 0 才支持 registered classical-induction edge。点是 seed 均值，横线是 95% bootstrap CI。",
            "若 candidate removal 增加 0.30 error、matched ordinary removal 增加 0.10，则 contrast=+0.20；若为 −0.05，说明候选 edge 不比 control 特异，不能因 synthetic pattern 好看就命名为 mechanism。",
        ),
        "图 C1 · 三类受控变形没有形成稳定 held-out nuisance model": (
            "这张图问 identity、context、position 三类受控 manipulation 是否能在 discovery-fitted rank-3 coordinates 中，对未见 confirmation seeds 提供稳定增量预测。",
            "横轴 incremental ΔR²=完整模型 held-out R²−删除该因素所有项后的 R²。正值表示该因素在这套 manipulation 中增添预测；但若完整模型 R² 本身为负，单根小正柱也不能解释为自然 variance share。",
            "完整模型 R²=0.10，去掉 position 后 0.08，则 position ΔR²=0.02；若完整模型 R²=−0.05，ΔR²=+0.02 仍表示整体模型比预测 confirmation mean 更差。",
        ),
        "图 C2 · Selected outside-halo edge removal 未超过两个 matched controls": (
            "这张区间图检验 outside-context halo edges 是否具有注册的自然因果特异性：candidate removal 必须同时比 distance-random 与 attention-mass-matched controls 更伤。",
            "横轴是 candidate error−control error，单位 counts；每个模型有两项 contrast。点为 seed 均值，95% CI 若跨 0 就不能确认 candidate 超过该 control；两项都为正才通过 gate。",
            "Candidate removal error=1.2、distance control=1.0、mass control=1.1，则两项 effects 为 +0.2/+0.1；若第二项 CI=[−0.1,0.3]，仍不能声称 halo edges 比两个 controls 都特异。",
        ),
    }
    figure_pattern = re.compile(
        r'(<figure>\s*)<h4 class="figure-title">(.*?)</h4>', re.DOTALL
    )
    rendered_figure_titles = [
        match.group(2).strip() for match in figure_pattern.finditer(html_doc)
    ]
    if len(rendered_figure_titles) != len(set(rendered_figure_titles)):
        raise RuntimeError("Figure titles must be unique before primer insertion")
    missing_guides = sorted(set(rendered_figure_titles) - set(figure_guides))
    unused_guides = sorted(set(figure_guides) - set(rendered_figure_titles))
    if missing_guides or unused_guides:
        raise RuntimeError(
            f"Figure-guide mismatch: missing={missing_guides}, unused={unused_guides}"
        )

    def insert_figure_primer(match: re.Match[str]) -> str:
        prefix, title = match.group(1), match.group(2).strip()
        what, reading, example = figure_guides[title]
        primer = (
            f'<div class="figure-primer" data-for="{html.escape(title)}">'
            '<div class="figure-primer-header">先看懂这张图</div>'
            '<div class="figure-primer-grid">'
            f'<p><strong>这张图画什么。</strong>{what}</p>'
            f'<p><strong>怎么读。</strong>{reading}</p>'
            f'<p class="primer-example"><strong>一个例子。</strong>{example}</p>'
            '</div></div>'
        )
        return f'{prefix}{primer}<h4 class="figure-title">{title}</h4>'

    html_doc = figure_pattern.sub(insert_figure_primer, html_doc)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "NiaH_Non-thinking_report.html",
    )
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
