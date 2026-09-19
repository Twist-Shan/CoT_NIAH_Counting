#!/usr/bin/env python3
"""Build the registered overall P0 grammar-specific ablation dose figure.

The dose curve is confirmation-only and registered-anchor weighted. Discovery
results at the registered primary K are reported separately as a design audit.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _full_split_summary(payload: dict, split: str) -> dict:
    selected_n = selected_failures = random_n = random_failures = 0
    for grammar in payload.get("grammars", []):
        row = grammar.get("by_split", {}).get(split, {})
        selected_n += int(row.get("selected", 0))
        selected_failures += int(row.get("selected_failures", 0))
        random_n += int(
            row.get(
                "random",
                row.get("global_random", row.get("layer_matched_random", 0)),
            )
        )
        random_failures += int(
            row.get(
                "random_failures",
                row.get(
                    "global_random_failures",
                    row.get("layer_matched_random_failures", 0),
                ),
            )
        )
    selected_rate = _rate(selected_failures, selected_n)
    random_rate = _rate(random_failures, random_n)
    return {
        "split": split,
        "anchors": selected_n,
        "selected_failure_rate": selected_rate,
        "random_failure_rate": random_rate,
        "selected_minus_random_failure_rate": (
            selected_rate - random_rate
            if selected_rate is not None and random_rate is not None
            else None
        ),
    }


def _svg(points: list[dict], title: str) -> str:
    width, height = 980, 560
    left, right, top, bottom = 92, 36, 70, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [
        float(row[key])
        for row in points
        for key in (
            "selected_failure_rate",
            "random_failure_rate",
            "selected_minus_random_failure_rate",
        )
        if row.get(key) is not None
    ]
    y_min = min(0.0, min(values, default=0.0))
    y_max = max(0.1, max(values, default=0.1))
    pad = max(0.03, 0.08 * (y_max - y_min))
    y_min = max(-1.0, y_min - (pad if y_min < 0 else 0))
    y_max = min(1.0, y_max + pad)

    def x_at(index: int) -> float:
        return left + (plot_w * index / max(1, len(points) - 1))

    def y_at(value: float) -> float:
        return top + (y_max - value) * plot_h / (y_max - y_min)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{left}" y="34" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#19212b">{html.escape(title)}</text>',
        f'<text x="{left}" y="56" font-family="Arial,sans-serif" font-size="13" fill="#59636f">Confirmation only · registered-anchor weighted · persistent P0 pre-O ablation</text>',
    ]
    ticks = 5
    for tick in range(ticks + 1):
        value = y_min + (y_max - y_min) * tick / ticks
        y = y_at(value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#d9d6cf" stroke-width="1"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#59636f">{100 * value:.0f}%</text>',
            ]
        )
    if y_min <= 0 <= y_max:
        zero_y = y_at(0)
        elements.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#7d838a" stroke-width="1.4"/>'
        )
    for index, row in enumerate(points):
        x = x_at(index)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eeeae2" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#26313c">K{int(row["bank_size"])}</text>',
            ]
        )
    series = [
        ("selected_failure_rate", "Selected bank failure", "#a43b32", ""),
        ("random_failure_rate", "Random control failure", "#315f78", "7 5"),
        (
            "selected_minus_random_failure_rate",
            "Selected − random",
            "#7b4ea3",
            "3 3",
        ),
    ]
    for key, label, color, dash in series:
        coords = [
            (x_at(i), y_at(float(row[key])))
            for i, row in enumerate(points)
            if row.get(key) is not None
        ]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if coords:
            path = " ".join(
                ("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}"
                for i, (x, y) in enumerate(coords)
            )
            elements.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>'
            )
            for x, y in coords:
                elements.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#fbfaf7" stroke="{color}" stroke-width="2.5"/>'
                )
        legend_x = left + 8 + series.index((key, label, color, dash)) * 250
        legend_y = height - 28
        elements.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 32}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash_attr}/>',
                f'<text x="{legend_x + 40}" y="{legend_y + 4}" font-family="Arial,sans-serif" font-size="12" fill="#26313c">{html.escape(label)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#26313c">Failure rate / specificity</text>',
            f'<text x="{left + plot_w / 2}" y="{height - 54}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#26313c">Number of ablated ranked heads (Top-K)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose-completion", type=Path, required=True)
    parser.add_argument("--full-completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    dose = json.loads(args.dose_completion.read_text(encoding="utf-8"))
    full = json.loads(args.full_completion.read_text(encoding="utf-8"))
    points = sorted(
        (
            row
            for row in dose.get("overall", [])
            if row.get("scope") == "all_registered_grammars"
        ),
        key=lambda row: int(row["bank_size"]),
    )
    if not points:
        raise ValueError("Dose completion contains no registered overall curve")
    control_by_k: dict[int, set[str]] = {}
    for row in dose.get("rows", []):
        control_by_k.setdefault(int(row["bank_size"]), set()).add(
            row.get("random_condition", "unspecified_random")
        )
    grouped_controls = {}
    for k, conditions in sorted(control_by_k.items()):
        condition_label = "+".join(sorted(conditions))
        grouped_controls.setdefault(condition_label, []).append(k)
    control_note = "; ".join(
        f"K={','.join(str(k) for k in ks)} use {condition}"
        for condition, ks in grouped_controls.items()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "overall_confirmation_dose_response.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    svg_path = args.output / "overall_confirmation_dose_response.svg"
    svg_path.write_text(_svg(points, args.title), encoding="utf-8")

    split_rows = [_full_split_summary(full, split) for split in ("discovery", "confirmation")]
    grammar_rows = []
    for grammar in full.get("grammars", []):
        confirmation_n = int(
            grammar.get("by_split", {}).get("confirmation", {}).get("selected", 0)
        )
        grammar_rows.append(
            {
                "grammar": grammar.get("grammar"),
                "anchors": int(grammar.get("anchors", 0)),
                "confirmation_anchors": confirmation_n,
                "status": "exploratory" if confirmation_n < 10 else "primary",
            }
        )
    split_table = "".join(
        "<tr>"
        f"<td>{html.escape(row['split'])}</td>"
        f"<td>{row['anchors']}</td>"
        f"<td>{_pct(row['selected_failure_rate'])}</td>"
        f"<td>{_pct(row['random_failure_rate'])}</td>"
        f"<td>{_pct(row['selected_minus_random_failure_rate'])}</td>"
        "</tr>"
        for row in split_rows
    )
    grammar_table = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(row['grammar']))}</code></td>"
        f"<td>{row['anchors']}</td>"
        f"<td>{row['confirmation_anchors']}</td>"
        f"<td>{row['status']}</td>"
        "</tr>"
        for row in grammar_rows
    )
    report = f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>body{{font:16px/1.55 system-ui;max-width:1080px;margin:36px auto;color:#202832;background:#fbfaf7}}h1,h2{{line-height:1.2}}figure{{margin:24px 0}}img{{width:100%;border:1px solid #d9d6cf}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border-bottom:1px solid #d9d6cf;padding:8px;text-align:left}}code{{background:#eeece6;padding:2px 5px}}.note{{border-left:4px solid #9a6b25;padding:10px 14px;background:#f8efd9}}</style>
<h1>{html.escape(args.title)}</h1>
<p>The primary dose response is evaluated only on the frozen confirmation split. Each grammar uses its own P0-event ranking, and rates are pooled by registered anchors rather than giving rare grammars equal weight.</p>
<figure><img src="overall_confirmation_dose_response.svg"><figcaption>Overall confirmation dose response. The x-axis is the nested Top-K head-bank size. The y-axis shows selected-bank failure, random-control failure, and their difference. {html.escape(control_note)}. Every control condition has three registered repeats.</figcaption></figure>
<div class="note">Discovery and confirmation are never pooled. Discovery is shown below only at the registered primary K; the complete K-grid line is confirmation-only.</div>
<h2>Registered primary-K split audit</h2>
<table><thead><tr><th>Split</th><th>Anchors</th><th>Selected failure</th><th>Random failure</th><th>Selected−random</th></tr></thead><tbody>{split_table}</tbody></table>
<h2>Grammar inference status</h2>
<table><thead><tr><th>Grammar</th><th>Full anchors</th><th>Confirmation anchors</th><th>Status</th></tr></thead><tbody>{grammar_table}</tbody></table>
<p>Grammars with fewer than 10 confirmation anchors are labelled exploratory. They remain in the registered overall panel with their natural anchor weight, but are not used for standalone generalization claims.</p>
"""
    (args.output / "overall_dose_response_report.html").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
