#!/usr/bin/env python3
"""Build the consolidated Native-thinking V5 representation and causal report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
GRAMMAR_LABELS = {
    "adjacent_rank_after_city": "adjacent · city → rank",
    "adjacent_rank_before_city": "adjacent · rank → city",
    "same_unit_rank_after_city": "same unit · city → rank",
    "same_unit_rank_before_city": "same unit · rank → city",
    "structural_explicit_rank_before_city": "structural explicit rank",
    "structural_invariant_bullet": "invariant bullet",
    "structural_unmarked": "structural unmarked",
    "evidence_sequence_unranked": "unranked evidence sequence",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"invalid JSONL {path}:{number}") from exc


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def f(value: Any) -> float:
    return float(value)


def i(value: Any) -> int:
    return int(float(value))


def pct(value: Any, digits: int = 1) -> str:
    return f"{100 * f(value):.{digits}f}%"


def db(value: Any) -> str:
    return f"{f(value):+.2f} dB"


def one(rows: Iterable[Mapping[str, Any]], **conditions: Any) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]
    require(len(matches) == 1, f"expected one row for {conditions}, got {len(matches)}")
    return matches[0]


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{esc(cls)}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def load_representation(causal_root: Path, dual_root: Path) -> dict[str, Any]:
    site_rows = read_csv(causal_root / "site_selected.csv")
    alignment_rows = read_csv(causal_root / "legacy_vs_causal_item_end.csv")
    result: dict[str, Any] = {}
    for model in MODELS:
        running = dict(one(site_rows, model_label=model, site_kind="item_end"))
        final_rows = read_csv(
            dual_root / model / "pca16_whiten" / "final_count_selected.csv"
        )
        final = dict(
            one(
                final_rows,
                model_label=model,
                endpoint="final_count",
                mode="native_thinking",
            )
        )
        alignment = dict(one(alignment_rows, model_label=model))
        result[model] = {"running": running, "final": final, "alignment": alignment}
    return result


def load_hybrid(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        analysis = root / model / "analysis_hybrid_supplement_registered_v1"
        dose = read_json(analysis / "hybrid_dose_grid_complete.json")
        full = read_json(analysis / "hybrid_full_panel_complete.json")
        require(dose.get("status") == "PASS", f"{model} hybrid dose is not PASS")
        require(full.get("status") == "PASS", f"{model} hybrid full panel is not PASS")
        require(dose.get("persistent_ablation") is True, f"{model} is not persistent")
        require(
            dose.get("intervention_start_anchor_role") == "p0_item_end",
            f"{model} intervention does not start at P0",
        )
        overall = [
            row for row in dose["overall"] if row["scope"] == "all_registered_grammars"
        ]
        overall.sort(key=lambda row: i(row["bank_size"]))
        primary_k = max(i(row["bank_size"]) for row in overall)
        grammar_rows = [
            row
            for row in dose["rows"]
            if i(row["bank_size"]) == primary_k
        ]
        result[model] = {
            "analysis": analysis,
            "dose": dose,
            "full": full,
            "overall": overall,
            "primary": dict(one(overall, bank_size=primary_k)),
            "primary_k": primary_k,
            "grammar": grammar_rows,
        }
    return result


def load_legacy_qwen(root: Path, selection_path: Path) -> dict[str, Any]:
    raw = read_csv(root / "raw_arm_rates.csv")
    estimands = read_csv(root / "estimands.csv")
    selection = read_json(selection_path)
    development = selection["development_selection"]
    require(development["primary_bank_size"] == 125, "legacy Qwen primary K changed")
    require(
        development["head_ranking_source_grammar"] == "adjacent_rank_before_city"
        and development["head_ranking_source_anchor"] == "post_marker",
        "legacy Qwen ranking contract changed",
    )
    filters = {
        "registered_bank_size": "125",
        "evaluation_scope": "confirmation",
        "analysis_population": "all_examples",
    }
    pooled_selected = one(raw, **filters, grammar_class="pooled", arm="selected_bank")
    pooled_random = one(
        raw,
        **filters,
        grammar_class="pooled",
        arm="layer_matched_random_mean",
    )
    pooled_effect = one(estimands, **filters, grammar_class="pooled")
    by_grammar: dict[str, dict[str, Mapping[str, Any]]] = {}
    for grammar in (
        "adjacent_rank_after_city",
        "adjacent_rank_before_city",
        "same_unit_rank_before_city",
        "structural_unmarked",
    ):
        by_grammar[grammar] = {
            "selected": one(raw, **filters, grammar_class=grammar, arm="selected_bank"),
            "random": one(
                raw,
                **filters,
                grammar_class=grammar,
                arm="layer_matched_random_mean",
            ),
        }
    return {
        "raw": raw,
        "estimands": estimands,
        "selection": selection,
        "pooled_selected": pooled_selected,
        "pooled_random": pooled_random,
        "pooled_effect": pooled_effect,
        "by_grammar": by_grammar,
    }


def load_duplicates(registry: Path) -> dict[str, int]:
    duplicates = [
        row
        for row in read_jsonl(registry)
        if row.get("trace_category") == "full_coverage_with_duplicates"
    ]
    qwen = [row for row in duplicates if row.get("model_label") == "Qwen3-8B"]
    gemma = [row for row in duplicates if row.get("model_label") == "Gemma4-E4B"]
    qwen_extra = [row for row in qwen if i(row["gold_count"]) >= 2]
    gemma_extra = [row for row in gemma if i(row["gold_count"]) >= 2]
    require((len(qwen), len(qwen_extra)) == (17, 14), "Qwen duplicate audit changed")
    require((len(gemma), len(gemma_extra)) == (10, 0), "Gemma duplicate audit changed")
    return {
        "qwen_total": len(qwen),
        "qwen_extra": len(qwen_extra),
        "gemma_total": len(gemma),
        "gemma_extra": len(gemma_extra),
    }


def load_atlas(atlas_root: Path) -> dict[str, Any]:
    head_rows = read_csv(atlas_root / "p0_targeted_retrieval_head_scores.csv")
    mass_rows = read_csv(atlas_root / "p0_significant_head_attention_masses.csv")
    result: dict[str, Any] = {}
    examples = {"Qwen3-8B": (20, 30), "Gemma4-E4B": (29, 4)}
    for model in MODELS:
        top = [
            row
            for row in head_rows
            if row["model_label"] == model
            and row["grammar"] == "all"
            and i(row["rank"]) <= 5
        ]
        top.sort(key=lambda row: i(row["rank"]))
        layer, head = examples[model]
        rows = [
            row
            for row in mass_rows
            if row["model_label"] == model
            and i(row["layer"]) == layer
            and i(row["head"]) == head
        ]
        by_event: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_event[(row["from_occurrence"], row["to_occurrence"])].append(row)
        target_masses: list[float] = []
        target_shares: list[float] = []
        top1: list[float] = []
        for event_rows in by_event.values():
            target = sum(f(row["raw_attention_mass"]) for row in event_rows if row["is_target"] == "True")
            needle = sum(
                f(row["raw_attention_mass"])
                for row in event_rows
                if row["region"] == "needle_record"
            )
            competitors = [
                f(row["raw_attention_mass"])
                for row in event_rows
                if row["region"] == "needle_record" and row["is_target"] != "True"
            ]
            target_masses.append(target)
            target_shares.append(target / needle if needle else 0.0)
            top1.append(float(target >= max(competitors, default=-1.0)))
        result[model] = {
            "top": top,
            "example_layer": layer,
            "example_head": head,
            "target_mass": sum(target_masses) / len(target_masses),
            "target_share": sum(target_shares) / len(target_shares),
            "target_top1": sum(top1) / len(top1),
        }
    return result


def representation_svg(rep: Mapping[str, Any]) -> str:
    panels = [("Running index · exact P0", "running"), ("Final count · answer query", "final")]
    parts = [
        '<svg class="chart" viewBox="0 0 1060 360" role="img" aria-labelledby="rep-title rep-desc">',
        '<title id="rep-title">Native-thinking representation confirmation balanced accuracy</title>',
        '<desc id="rep-desc">Two panels compare Logistic and nearest-centroid balanced accuracy for running-index and final-count representations in Qwen and Gemma.</desc>',
    ]
    for panel_index, (title, endpoint) in enumerate(panels):
        ox = 20 + panel_index * 520
        parts.append(f'<rect x="{ox}" y="16" width="500" height="318" rx="16" fill="#fbfaf6" stroke="#d9d6cd"/>')
        parts.append(f'<text x="{ox+24}" y="48" class="chart-title">{esc(title)}</text>')
        for model_index, model in enumerate(MODELS):
            row = rep[model][endpoint]
            logistic = f(row["confirmation_logistic_balanced_accuracy"])
            ncc = f(row["confirmation_ncc_balanced_accuracy"])
            y = 92 + model_index * 112
            parts.append(f'<text x="{ox+24}" y="{y}" class="chart-model">{SHORT[model]}</text>')
            for metric_index, (label, value, color) in enumerate(
                (("Logistic", logistic, "#14766f"), ("NCC", ncc, "#8b918e"))
            ):
                by = y + 18 + metric_index * 34
                width = 360 * value
                parts.append(f'<text x="{ox+24}" y="{by+14}" class="chart-axis">{label}</text>')
                parts.append(f'<rect x="{ox+104}" y="{by}" width="360" height="18" rx="4" fill="#e9e7e0"/>')
                parts.append(f'<rect x="{ox+104}" y="{by}" width="{width:.1f}" height="18" rx="4" fill="{color}"/>')
                parts.append(f'<text x="{ox+474}" y="{by+14}" text-anchor="end" class="chart-value">{100*value:.1f}%</text>')
        parts.append(f'<text x="{ox+104}" y="318" class="chart-axis">0%</text>')
        parts.append(f'<text x="{ox+464}" y="318" text-anchor="end" class="chart-axis">100% balanced accuracy</text>')
    parts.append("</svg>")
    return "".join(parts)


def alignment_svg(rep: Mapping[str, Any]) -> str:
    parts = [
        '<svg class="chart" viewBox="0 0 1060 310" role="img" aria-labelledby="align-title align-desc">',
        '<title id="align-title">Legacy item-end versus exact causal P0 representation</title>',
        '<desc id="align-desc">Slope chart of Logistic and nearest-centroid balanced accuracy before and after exact causal-site restriction.</desc>',
    ]
    for model_index, model in enumerate(MODELS):
        ox = 30 + model_index * 520
        row = rep[model]["alignment"]
        parts.append(f'<rect x="{ox}" y="14" width="490" height="270" rx="16" fill="#fbfaf6" stroke="#d9d6cd"/>')
        parts.append(f'<text x="{ox+24}" y="46" class="chart-title">{SHORT[model]}</text>')
        x0, x1 = ox + 126, ox + 400
        parts.append(f'<line x1="{x0}" y1="72" x2="{x0}" y2="232" stroke="#d4d2ca"/>')
        parts.append(f'<line x1="{x1}" y1="72" x2="{x1}" y2="232" stroke="#d4d2ca"/>')
        pairs = (
            ("Logistic", "legacy_confirmation_logistic_balanced_accuracy", "causal_confirmation_logistic_balanced_accuracy", "#14766f"),
            ("NCC", "legacy_confirmation_ncc_balanced_accuracy", "causal_confirmation_ncc_balanced_accuracy", "#777e7b"),
        )
        for label, before_key, after_key, color in pairs:
            before, after = f(row[before_key]), f(row[after_key])
            y0, y1 = 242 - 180 * before, 242 - 180 * after
            parts.append(f'<line x1="{x0}" y1="{y0:.1f}" x2="{x1}" y2="{y1:.1f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<circle cx="{x0}" cy="{y0:.1f}" r="5" fill="{color}"/>')
            parts.append(f'<circle cx="{x1}" cy="{y1:.1f}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{x0-10}" y="{y0+4:.1f}" text-anchor="end" class="chart-value">{label} {100*before:.1f}%</text>')
            parts.append(f'<text x="{x1+10}" y="{y1+4:.1f}" class="chart-value">{100*after:.1f}%</text>')
        parts.append(f'<text x="{x0}" y="262" text-anchor="middle" class="chart-axis">all item ends</text>')
        parts.append(f'<text x="{x1}" y="262" text-anchor="middle" class="chart-axis">exact P0 commits</text>')
    parts.append("</svg>")
    return "".join(parts)


def dose_svg(hybrid: Mapping[str, Any]) -> str:
    parts = [
        '<svg class="chart" viewBox="0 0 1060 390" role="img" aria-labelledby="dose-title dose-desc">',
        '<title id="dose-title">Hybrid targeted-retrieval confirmation dose response</title>',
        '<desc id="dose-desc">Two panels show selected-bank failure, random-control failure, and their difference across nested K for Qwen and Gemma.</desc>',
    ]
    for model_index, model in enumerate(MODELS):
        ox = 26 + model_index * 520
        rows = hybrid[model]["overall"]
        parts.append(f'<rect x="{ox}" y="14" width="494" height="344" rx="16" fill="#fbfaf6" stroke="#d9d6cd"/>')
        parts.append(f'<text x="{ox+24}" y="46" class="chart-title">{SHORT[model]} · confirmation</text>')
        x_start, x_end = ox + 66, ox + 454
        y_top, y_bottom = 72, 306
        for tick in range(0, 101, 25):
            y = y_bottom - (y_bottom - y_top) * tick / 100
            parts.append(f'<line x1="{x_start}" y1="{y:.1f}" x2="{x_end}" y2="{y:.1f}" stroke="#e2e0d9"/>')
            parts.append(f'<text x="{x_start-10}" y="{y+4:.1f}" text-anchor="end" class="chart-axis">{tick}%</text>')
        xs = [x_start + (x_end - x_start) * idx / max(1, len(rows) - 1) for idx in range(len(rows))]
        series = (
            ("selected", "selected_failure_rate", "#14766f", ""),
            ("random", "random_failure_rate", "#8b918e", ""),
            ("difference", "selected_minus_random_failure_rate", "#343a38", "5 5"),
        )
        for _, key, color, dash in series:
            points = []
            for x, row in zip(xs, rows):
                y = y_bottom - (y_bottom - y_top) * f(row[key])
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
            for x, row in zip(xs, rows):
                y = y_bottom - (y_bottom - y_top) * f(row[key])
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        for x, row in zip(xs, rows):
            parts.append(f'<text x="{x:.1f}" y="328" text-anchor="middle" class="chart-axis">K{row["bank_size"]}</text>')
        legend_y = 350
        for offset, (label, _, color, dash) in enumerate(series):
            lx = ox + 86 + offset * 132
            parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+24}" y2="{legend_y}" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/>')
            parts.append(f'<text x="{lx+31}" y="{legend_y+4}" class="chart-axis">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grammar_rows(model: str, data: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in sorted(data["grammar"], key=lambda item: (-i(item["confirmation_anchors"]), item["grammar"])):
        anchors = i(row["confirmation_anchors"])
        selected = i(row["selected_failures"]) / anchors if anchors else 0.0
        random_denominator = anchors * 3 if anchors else 0
        random = i(row["random_failures"]) / random_denominator if random_denominator else 0.0
        status = "confirmatory" if anchors >= 10 else "exploratory"
        if anchors == 0:
            status = "no confirmation anchor"
        rows.append(
            [
                f"<code>{esc(row['grammar'])}</code><br><span class=muted>{esc(GRAMMAR_LABELS.get(row['grammar'], row['grammar']))}</span>",
                esc(row["selection_anchor_role"]),
                str(anchors),
                pct(selected) if anchors else "—",
                pct(random) if anchors else "—",
                pct(selected - random) if anchors else "—",
                f'<span class="status {"ok" if anchors >= 10 else "explore"}">{status}</span>',
            ]
        )
    return rows


def build_report(
    rep: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    legacy_qwen: Mapping[str, Any],
    atlas: Mapping[str, Any],
    duplicates: Mapping[str, int],
    generated: str,
    hashes: Mapping[str, str],
) -> str:
    q, g = hybrid["Qwen3-8B"], hybrid["Gemma4-E4B"]
    q_primary, g_primary = q["primary"], g["primary"]
    representation_rows = []
    for model in MODELS:
        running, final = rep[model]["running"], rep[model]["final"]
        representation_rows.extend(
            [
                [SHORT[model], "Running index", "<code>p0_item_end</code>", f"L{running['layer']}", pct(running["confirmation_logistic_balanced_accuracy"]), pct(running["confirmation_ncc_balanced_accuracy"]), db(running["confirmation_class_balanced_snr_db"]), running["confirmation_rows"]],
                [SHORT[model], "Final count", "<code>answer_query_v3</code>", f"L{final['layer']}", pct(final["confirmation_logistic_balanced_accuracy"]), pct(final["confirmation_ncc_balanced_accuracy"]), db(final["confirmation_class_balanced_snr_db"]), final["confirmation_rows"]],
            ]
        )
    alignment_rows = []
    for model in MODELS:
        row = rep[model]["alignment"]
        alignment_rows.append(
            [
                SHORT[model],
                f"{pct(row['legacy_confirmation_logistic_balanced_accuracy'])} → {pct(row['causal_confirmation_logistic_balanced_accuracy'])}",
                f"{pct(row['legacy_confirmation_ncc_balanced_accuracy'])} → {pct(row['causal_confirmation_ncc_balanced_accuracy'])}",
                f"{db(row['legacy_confirmation_snr_db'])} → {db(row['causal_confirmation_snr_db'])}",
            ]
        )
    top_rows = []
    for model in MODELS:
        heads = atlas[model]["top"]
        top_rows.append(
            [
                SHORT[model],
                ", ".join(
                    f"L{row['layer']}H{row['head']} ({f(row['score']):.3f})" for row in heads
                ),
                "20 discovery seeds",
            ]
        )
    primary_rows = []
    for model, data in (("Qwen3-8B", q), ("Gemma4-E4B", g)):
        row = data["primary"]
        primary_rows.append(
            [
                SHORT[model],
                f"K{data['primary_k']}",
                row["confirmation_anchors"],
                pct(row["selected_failure_rate"]),
                pct(row["random_failure_rate"]),
                pct(row["selected_minus_random_failure_rate"]),
            ]
        )
    legacy_comparison_rows = []
    for grammar in (
        "adjacent_rank_after_city",
        "adjacent_rank_before_city",
        "same_unit_rank_before_city",
        "structural_unmarked",
    ):
        old = legacy_qwen["by_grammar"][grammar]
        new = one(q["grammar"], grammar=grammar)
        anchors = i(new["confirmation_anchors"])
        old_selected = f(old["selected"]["anchor_weighted_mean"])
        old_random = f(old["random"]["anchor_weighted_mean"])
        new_selected = i(new["selected_failures"]) / anchors if anchors else 0.0
        new_random = i(new["random_failures"]) / (3 * anchors) if anchors else 0.0
        legacy_comparison_rows.append(
            [
                f"<code>{esc(grammar)}</code>",
                str(anchors),
                f"{pct(old_selected)} / {pct(old_random)}",
                f"{pct(new_selected)} / {pct(new_random)}",
                pct((new_selected - new_random) - (old_selected - old_random)),
            ]
        )
    dose_rows = []
    for model in MODELS:
        for row in hybrid[model]["overall"]:
            dose_rows.append(
                [SHORT[model], f"K{row['bank_size']}", row["confirmation_anchors"], pct(row["selected_failure_rate"]), pct(row["random_failure_rate"]), pct(row["selected_minus_random_failure_rate"])]
            )
    q_atlas = atlas["Qwen3-8B"]
    g_atlas = atlas["Gemma4-E4B"]
    ledger = "<br>".join(f"{esc(path)}: <code>{digest}</code>" for path, digest in hashes.items())
    css = """
:root{--paper:#f3f0e8;--surface:#fffefb;--ink:#202624;--muted:#66706c;--line:#d8d4ca;--soft:#ebe8df;--accent:#14766f;--accent-soft:#dcece8;--nav:#252b29;--code:#eef1ed}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Geist","Aptos","Segoe UI",sans-serif;line-height:1.7}a{color:var(--accent);text-underline-offset:3px}code,.mono,.formula,.audit,td:nth-last-child(-n+4){font-family:"Geist Mono","Cascadia Mono",monospace}code{background:var(--code);border:1px solid #dce1dc;border-radius:4px;padding:.08rem .28rem}.layout{max-width:1510px;margin:auto;display:grid;grid-template-columns:250px minmax(0,1fr);gap:0}.side{position:sticky;top:0;height:100dvh;padding:28px 22px;background:var(--nav);color:#e9eeeb;overflow:auto}.side strong{display:block;font-size:15px;letter-spacing:.05em;margin-bottom:22px}.side a{display:block;color:#bdc7c3;text-decoration:none;padding:7px 0;border-bottom:1px solid #363e3b;font-size:13px}.side a:hover{color:#fff}.content{min-width:0;background:var(--surface);box-shadow:0 0 46px rgba(32,38,36,.08)}main{max-width:1160px;padding:0 54px 80px}header{padding:72px 54px 56px;background:#fbfaf6;border-bottom:1px solid var(--line)}.eyebrow,.section-tag{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.hero-grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,.65fr);gap:54px;align-items:end}h1{font-size:clamp(42px,5vw,70px);line-height:1.02;letter-spacing:-.055em;margin:18px 0 22px;max-width:880px}h2{font-size:32px;line-height:1.16;letter-spacing:-.035em;margin:0 0 22px}h3{font-size:20px;letter-spacing:-.018em;margin:34px 0 12px}p{max-width:82ch}.lead{font-size:19px;color:#4f5a56}.hero-note{border-top:3px solid var(--accent);padding-top:18px;color:var(--muted);font-size:14px}.scope{display:inline-flex;border:1px solid var(--accent);color:var(--accent);padding:5px 10px;border-radius:99px;font-size:11px;font-weight:800;letter-spacing:.08em}section{padding:58px 0;border-bottom:1px solid var(--line)}.summary-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:34px}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));border-top:1px solid var(--line)}.metric{padding:20px 14px 18px 0;border-bottom:1px solid var(--line)}.metric strong{display:block;font:700 34px/1 "Geist Mono","Cascadia Mono",monospace;color:var(--accent)}.metric span{display:block;color:var(--muted);font-size:13px;margin-top:8px}.purpose,.conclusion,.boundary,.example,.note{padding:16px 18px;border-left:3px solid var(--accent);background:var(--accent-soft);margin:22px 0}.purpose .label,.conclusion .label,.boundary .label,.example .label,.note .label{display:block;color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}.boundary,.note{border-left-color:#6f7874;background:#efefea}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:14px;align-items:center;margin:26px 0}.flow-node{min-height:126px;padding:20px;border-top:3px solid var(--accent);background:#f7f6f1}.flow-node span{color:var(--muted);font-size:14px}.arrow{color:var(--accent);font-size:25px}.two{display:grid;grid-template-columns:1fr 1fr;gap:28px}.table-wrap{overflow:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:22px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px 13px;border-bottom:1px solid #e4e1d9;vertical-align:top}th{background:#f0eee7;color:#49524f;font-size:11px;letter-spacing:.045em;text-transform:uppercase;position:sticky;top:0}tr:last-child td{border-bottom:0}.muted{color:var(--muted);font-size:12px}.status{display:inline-block;border-radius:99px;padding:2px 7px;font-size:10px;font-family:"Geist",sans-serif}.status.ok{background:var(--accent-soft);color:var(--accent)}.status.explore{background:#ecebe6;color:#6d7471}.formula{white-space:pre-wrap;background:#252b29;color:#edf4f1;padding:18px;border-radius:6px;overflow:auto;font-size:13px}.chart{display:block;width:100%;height:auto;margin:14px 0}.chart-title{font:700 16px "Geist",sans-serif;fill:#27302d}.chart-model{font:700 14px "Geist",sans-serif;fill:#27302d}.chart-axis{font:11px "Geist Mono",monospace;fill:#68716d}.chart-value{font:700 11px "Geist Mono",monospace;fill:#333a37}figure{margin:28px 0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fbfaf6}figcaption{font-size:13px;color:var(--muted);margin-top:10px;max-width:92ch}.figure-grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.figure-grid figure{margin:0}.wide-image{overflow:auto;border:1px solid var(--line);background:#fbfaf6}.wide-image img{min-width:780px;border:0}.links{display:flex;flex-wrap:wrap;gap:8px}.links a{border:1px solid var(--line);padding:7px 10px;text-decoration:none;font-size:12px;background:#fbfaf6}.audit{font-size:11px;word-break:break-all;color:var(--muted)}details{margin:18px 0}summary{cursor:pointer;color:var(--accent);font-weight:700}.print-only{display:none}@media(max-width:980px){.layout{display:block}.side{position:relative;height:auto}.side a{display:inline-block;margin-right:14px}.hero-grid,.summary-grid,.two,.figure-grid{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg);text-align:center}header{padding:48px 24px}main{padding:0 24px 60px}h1{font-size:44px}}@media print{body{background:#fff}.layout{display:block}.side{display:none}.content{box-shadow:none}header,main{padding-left:0;padding-right:0}.links{display:none}section{break-inside:auto}figure,.table-wrap,.purpose,.conclusion{break-inside:avoid}.print-only{display:block}}
"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Native-thinking V5 · Representation, Retrieval Atlas, and Causal Ablation</title><style>{css}</style></head>
<body><div class="layout"><nav class="side"><strong>Native-thinking V5</strong><a href="#summary">结论摘要</a><a href="#scope">1 · 样本与 token routing</a><a href="#definitions">2 · 三层证据与定义</a><a href="#representation">A · Representation</a><a href="#atlas">B · Attention atlas</a><a href="#causal">C · Causal ablation</a><a href="#grammar">C2 · Grammar 分解</a><a href="#synthesis">机制综合</a><a href="#audit">复现账本</a></nav><div class="content">
<header><div class="hero-grid"><div><div class="eyebrow">REALISTIC NIAH · NATIVE-THINKING V5 · CONSOLIDATED REPORT</div><h1>进度状态、检索定位与因果必要性</h1><p class="lead">一个页面区分三种问题：hidden state 中是否存在计数相关信息；attention head 在哪个 query 位点指向正确 needle；关闭这些 heads 是否真的破坏下一次检索。最终因果设定采用 grammar-routed hybrid ranking，但所有干预统一从 P0 开始并持续至 decode 结束。</p><span class="scope">FINAL HYBRID · REPRESENTATION + ATLAS + ABLATION</span></div><aside class="hero-note"><strong>核心口径</strong><br>P0 是 item k 的提交点与统一干预起点。若下一项 marker 先于 city 出现，head bank 在 marker 已生成的 P2 query 上排序；这说明 marker-conditioned query 更容易局部化 retrieval heads，不等于已证明“P2 写入”。</aside></div></header>
<main>
<section id="summary"><div class="section-tag">Executive synthesis</div><h2>结论摘要：三段证据互补，但不应互相替代</h2><div class="summary-grid"><div><div class="flow"><div class="flow-node"><strong>1 · Representation</strong><br><span>P0 可解码 running index；answer query 可解码 final count。</span></div><div class="arrow">→</div><div class="flow-node"><strong>2 · Localization</strong><br><span>高分 heads 从 query 指向下一条正确 prompt record。</span></div><div class="arrow">→</div><div class="flow-node"><strong>3 · Necessity</strong><br><span>selected bank 持续关闭比同规模 random 更易使首次 city 出错。</span></div></div><p>Representation 证明“状态中有什么”，atlas 证明“attention 在看哪里”，ablation 证明“一个有序 head 集合是否必要”。当前最稳健的机制表述是：模型在完成 item 后维护可解码进度，并通过依赖 surface grammar 的查询位点调用 targeted-retrieval head bank；但还没有 restoration/mediation 将 progress representation 与 bank activity 连成唯一因果路径。</p></div><div class="metrics"><div class="metric"><strong>{pct(rep['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy'])}</strong><span>Qwen running-index Logistic BA</span></div><div class="metric"><strong>{pct(rep['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy'])}</strong><span>Gemma running-index Logistic BA</span></div><div class="metric"><strong>{pct(q_primary['selected_minus_random_failure_rate'])}</strong><span>Qwen K{q['primary_k']} hybrid selected−random</span></div><div class="metric"><strong>{pct(g_primary['selected_minus_random_failure_rate'])}</strong><span>Gemma K{g['primary_k']} hybrid selected−random</span></div></div></div><div class="conclusion"><span class="label">目前能得到的结论</span>两模型都存在可解码的计数相关状态和具有 selection specificity 的 targeted-retrieval bank。Gemma 的有效 bank 更窄；Qwen 需要更宽的 bank，并且最终效应主要由 rank-before-city 的 marker-conditioned P2 ranking 贡献。</div></section>

<section id="scope"><div class="section-tag">Frozen cohort and routing</div><h2>1 · 样本、parser 与最终 token-site 规则</h2><div class="purpose"><span class="label">本节目的</span>先固定什么是一个 transition、head 在哪里排序、ablation 从哪里开始，避免把 trace 表面顺序和因果干预位置混为同一概念。</div><p>Mother panel 为 N=1…10、每个 N 30 个 seeds，共 300 prompts。Discovery seeds 1234–1253 只用于选层与排 head；registered confirmation seeds 1254–1263 只用于评估。N=1 没有 k→k+1 transition，因此不进入 targeted retrieval；Qwen 另有 14 条 N≥2 trace 因模型自身重复列举同一 city 而无法唯一配对，最终 causal registry 为 256。Gemma 的 10 条 duplicate 全部位于 N=1，因此 N=2…10 的 270 条全部保留。</p>{table(['Model','Mother panel','N=1 not applicable','N≥2 duplicate exclusion','Final causal registry','Registry SHA'],[['Qwen','300','30',str(duplicates['qwen_extra']),'256','<code>23134cfaf617…</code>'],['Gemma','300','30',str(duplicates['gemma_extra']),'270','<code>021f1e5d3b95…</code>']])}<h3>最终 hybrid routing</h3><div class="two"><div><h3>P0：统一 commit / intervention 起点</h3><p><code>p0_item_end</code> 是完整 item k 的最后一个真实 output token。无论 head bank 在 P0 还是 P2 排名，selected pre-O head slices 都从这个 P0 token 起持续置零到 decode 结束（<code>decode_head_ablation_steps=-1</code>）。</p></div><div><h3>P2：marker-conditioned ranking query</h3><p>当 rank/marker 先于 city 出现时，head 的 attention score在 marker semantic core 的最后一个 token（<code>post_marker</code>）计算；此时 marker 已进入 residual stream，而 city k+1 尚未输出。其他 grammar 仍在 exact P0 排名。</p></div></div>{table(['Model','Grammar family','Head ranking query','Ablation start','Persistence'],[['Qwen','rank-before-city','<code>post_marker</code> (P2)','<code>p0_item_end</code>','to decode end'],['Qwen','rank-after / unmarked / invariant / evidence','<code>p0_item_end</code>','<code>p0_item_end</code>','to decode end'],['Gemma','rank-before-city','<code>post_marker</code> (P2)','<code>p0_item_end</code>','to decode end'],['Gemma','rank-after / invariant','<code>p0_item_end</code>','<code>p0_item_end</code>','to decode end']])}<div class="example"><span class="label">简单例子</span>在 <code>(Record 2: Riga, 60)</code> 中，P0 是 item 1 的结束 token；等模型输出 <code>Record 2</code> 后，P2 是 rank marker “2”的最后一个 token。最终实验用 P2 attention 给 heads 排名，却从更早的 P0 起持续关闭这些 heads，因此检验的是“这组 marker-conditioned retrieval heads 是否属于从 transition 开始就不可用的必要通路”。</div><div class="conclusion"><span class="label">本节结论</span>P0 与 P2 承担不同实验角色：P0 统一定义 transition 与 intervention；P2 只在 marker-first grammar 中提供更干净的 head-localization query。数据不要求把它们解释为严格的“读取/写入”两相时钟。</div></section>

<section id="definitions"><div class="section-tag">Estimands</div><h2>2 · 三层证据分别如何计算</h2><h3>2.1 Representation</h3><p>Discovery 内拟合 StandardScaler 与 whitened PCA-16，再以 seed-grouped out-of-fold Logistic/NCC 选层；confirmation 只在冻结层上评估。Balanced accuracy 对十个 count/index 类等权：</p><div class="formula">BA = (1 / C) · Σ_c  TP_c / (TP_c + FN_c)
SNR_dB = 10 · log10(signal power / within-class noise power)</div><p>BA 高说明类别可解码；SNR 为负只说明 within-class variation 仍大于 centroid 间能量，不能解释为“没有 counter”。</p><h3>2.2 Targeted-retrieval attention score</h3><p>对 grammar g，在每个 discovery seed 内先平均所有 eligible transition events，再对 seeds 等权。event 基础量是 head 从 query 指向 next needle 完整 prompt-record span 的 raw attention mass：</p><div class="formula">S_g(ℓ,h) = (1 / |D_g|) Σ_{{s∈D_g}} (1 / |E_{{s,g}}|) Σ_{{e∈E_{{s,g}}}} Σ_{{t∈R(target(e))}} A_{{ℓ,h}}(q_e,t)</div><p>这样 event 更多的 seed 不会获得更高权重。<code>all</code> atlas 同样先在 seed 内平均全部 eligible events，再对 discovery seeds 等权；它是描述性共同排序，不替代 grammar-specific bank。</p><h3>2.3 Causal endpoint 与对照</h3><p>对每个 registered anchor 自由生成，读取 ablation 后出现的第一个 semantic city。若它不等于 registry 中 next needle city，则计一次 failure。主描述量为：</p><div class="formula">Δ_failure(K) = selected-bank failure(K) − pooled random-control failure(K)</div><p>Qwen K32–112 使用三个 layer-matched random banks，K128 因容量限制使用三个 global same-K banks；Gemma K1/2/4/6/8 均使用三个 layer-matched random banks。Confirmation 少于 10 anchors 的 grammar 标为 exploratory。</p><div class="conclusion"><span class="label">本节结论</span>高 attention score 只提供定位；只有 selected failure 显著高于同规模 random damage，才支持 selection-specific causal necessity。本页的最终 hybrid 合并目前报告注册的 failure rates，未把旧 P0 bootstrap CI 冒充为新 hybrid 的置信区间。</div></section>

<section id="representation"><div class="section-tag">Experiment A</div><h2>A · Representation：状态里是否携带进度与最终答案</h2><div class="purpose"><span class="label">实验目的</span>分别在 item commit 与最终数字生成前，检验 hidden state 是否包含可跨 seed 解码的 running index k 和 final count N。</div><figure>{representation_svg(rep)}<figcaption>图 1 · Confirmation balanced accuracy。左 panel 的横条表示 exact <code>p0_item_end</code> 上 running index 的 Logistic/NCC BA；右 panel 表示 <code>answer_query_v3</code> 上 final count 的 BA。横轴均为 0–100%，十类 chance 为 10%。颜色区分 Logistic（绿色）与 NCC（灰色），不表示因果方向。</figcaption></figure>{table(['Model','Endpoint','Token site','Layer','Logistic BA','NCC BA','SNR','Rows'],representation_rows)}<p>Qwen 的 running state 为 {pct(rep['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Qwen3-8B']['running']['confirmation_ncc_balanced_accuracy'])}，Gemma 为 {pct(rep['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Gemma4-E4B']['running']['confirmation_ncc_balanced_accuracy'])}。Final-count state 则呈不同分工：Qwen 在 answer query 达到 {pct(rep['Qwen3-8B']['final']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Qwen3-8B']['final']['confirmation_ncc_balanced_accuracy'])}，Gemma 为 {pct(rep['Gemma4-E4B']['final']['confirmation_logistic_balanced_accuracy'])}/{pct(rep['Gemma4-E4B']['final']['confirmation_ncc_balanced_accuracy'])}。</p><div class="example"><span class="label">简单例子</span>对未参与选层的 seed1258，在第 4 个 item 结束处取 hidden vector；如果 discovery-frozen Logistic 与最近 centroid 都预测 running index=4，记为 held-out decoding 正确。它不要求模型下一步一定生成正确 city，因此不能单独作为 retrieval 的因果证据。</div><h3>A2 · causal-site 对齐是否让 counter 更紧致</h3><figure>{alignment_svg(rep)}<figcaption>图 2 · 从所有 parser-observed item ends 收窄到 exact causal P0 commits。横向位置表示 cohort 定义，纵向位置表示 confirmation BA；每条线连接同一分类器。图下表同时报告 SNR 变化。</figcaption></figure>{table(['Model','Logistic BA','NCC BA','SNR'],alignment_rows)}<p>Qwen Logistic/NCC 仅增加约 1.7/1.2 个百分点；Gemma 增加约 2.4/3.4 个百分点。两模型 SNR 都没有随更精确的 site 变得更高。</p><div class="conclusion"><span class="label">Experiment A 结论</span>item commit 上确实存在可解码 running progress，answer query 上存在 final-count state；更精确的 causal site 让分类略有改善，但没有把 progress representation 压成低噪声、单轴、grammar-invariant 的标量计数器。</div></section>

<section id="atlas"><div class="section-tag">Experiment B · descriptive localization</div><h2>B · P0 targeted-retrieval atlas：哪些 heads 指向正确 needle</h2><div class="purpose"><span class="label">实验目的</span>在因果干预之前，直接展示 discovery attention ranking 的空间分布、随 needle ordinal 的变化，以及显著单头究竟把 attention 放到 prompt 的哪个 record。</div><div class="note"><span class="label">适用边界</span>下面 atlas 严格来自 exact P0 discovery queries。它解释最终 P0-ranked grammar bank 与跨 grammar 共同结构；rank-before-city 的最终因果 bank 已改在 P2 排名，因此不能把这里的 P0 rank 次序当成最终 P2 bank。</div><h3>B1 · 全类别 head map</h3><div class="figure-grid"><figure><img src="v5_native_p0_head_atlas/Qwen3-8B_all_p0_head_map.svg" alt="Qwen all-grammar P0 targeted retrieval head map"><figcaption>图 3a · Qwen 全类别 P0 head map。横轴=head index，纵轴=decoder layer；颜色为 seed-equal targeted-retrieval score。白框是描述性 global Top-128，不是最终 hybrid intervention bank。</figcaption></figure><figure><img src="v5_native_p0_head_atlas/Gemma4-E4B_all_p0_head_map.svg" alt="Gemma all-grammar P0 targeted retrieval head map"><figcaption>图 3b · Gemma 全类别 P0 head map。坐标与 Qwen 相同；白框是描述性 global Top-8。模型间色标不同，因此跨模型比较层分布与 rank，不直接比较颜色绝对值。</figcaption></figure></div>{table(['Model','All-scope Top-5 heads (P0 score)','Coverage'],top_rows)}<p>Qwen 的 P0 高分 heads 集中在 L20–L24；Gemma 主要集中在 L17 与 L29。grammar 切换会改变次序与强度，但这些核心层反复出现，说明 retrieval routing 同时具有共享骨架与 grammar-specific reweighting。</p><h3>B2 · Needle ordinal × ranked head</h3><figure class="wide-image"><img src="v5_native_p0_head_atlas/Qwen3-8B_all_p0_needle_ordinal_by_head.svg" alt="Qwen needle ordinal by ranked head"><figcaption>图 4a · Qwen。横轴每列是完整 layer–head identity，按 all-scope P0 score 从高到低排列；纵轴是正在检索的 target needle #2–#10；颜色是正确 record span 的 raw attention mass，灰色表示没有 eligible event。</figcaption></figure><figure><img src="v5_native_p0_head_atlas/Gemma4-E4B_all_p0_needle_ordinal_by_head.svg" alt="Gemma needle ordinal by ranked head"><figcaption>图 4b · Gemma。横轴只展示 Top-8；纵轴和色义与 Qwen 相同。同一列跨 ordinal 保持明亮表示 head 被稳定复用，局部亮带则提示按 retrieval step 的分工。</figcaption></figure><h3>B3 · 单头逐 record attention</h3><div class="figure-grid"><figure><img src="v5_native_p0_head_atlas/Qwen3-8B_adjacent_rank_after_city_L20H30_p0_attention.svg" alt="Qwen L20H30 single-head attention distribution"><figcaption>图 5a · Qwen L{q_atlas['example_layer']}H{q_atlas['example_head']}。横轴是连续 P0 transition queries，纵轴是 prompt 中各 needle record 与非-needle context；红框为每列正确 target。Mean target mass={q_atlas['target_mass']:.3f}，target/all-needle={pct(q_atlas['target_share'])}，target top-1={pct(q_atlas['target_top1'])}。</figcaption></figure><figure><img src="v5_native_p0_head_atlas/Gemma4-E4B_adjacent_rank_after_city_L29H4_p0_attention.svg" alt="Gemma L29H4 single-head attention distribution"><figcaption>图 5b · Gemma L{g_atlas['example_layer']}H{g_atlas['example_head']}。坐标与 Qwen 相同。Mean target mass={g_atlas['target_mass']:.3f}，target/all-needle={pct(g_atlas['target_share'])}，target top-1={pct(g_atlas['target_top1'])}。</figcaption></figure></div><div class="example"><span class="label">简单例子</span>当当前 transition 是 4→5 时，红框落在 prompt 的第 5 条 needle record。若某 head 在这一列把大部分 needle attention 放到第 5 条而不是其他九条，它具有 targeted localization；但只有后续 ablation 能判断模型是否依赖它。</div><div class="conclusion"><span class="label">Experiment B 结论</span>P0 attention 不是均匀铺在所有 heads 上，而是在少数层形成可解释的 next-record routing；这些 heads 能跨多个 ordinal 复用。Atlas 证明 localization，不证明单头或 bank 的因果必要性。</div></section>

<section id="causal"><div class="section-tag">Experiment C · final hybrid intervention</div><h2>C · Causal ablation：关闭被定位的 heads 是否破坏下一次检索</h2><div class="purpose"><span class="label">实验目的</span>在 confirmation 上比较 grammar-specific selected bank 与同规模 random bank，检验 targeted-retrieval ranking 是否提供超出 generic model damage 的因果预测。</div><p>干预清零 selected heads 的 pre-O output slice，而不是删除整个 attention layer。清零从 exact P0 query 开始，并持续作用于后续自由生成的每个 token；这防止模型在第一步失败后通过额外 token 逃出短暂 ablation 窗口。Discovery 与 confirmation 从不混合排 head。</p><h3>C0 · 为什么 Qwen 旧共享 K125 是 78.7%，最终 hybrid K128 只有 40.6%</h3><p>旧实验不是当前 grammar-specific bank 的早期副本。它只在最纯净的 <code>adjacent_rank_before_city</code> P2 events 上排出一个共享 K125 bank，再把同一个 bank 迁移到全部 grammar。其 seed-equal contrast 为 {pct(legacy_qwen['pooled_effect']['mean'])}；按 anchor 加权的 selected/random 为 {pct(legacy_qwen['pooled_selected']['anchor_weighted_mean'])}/{pct(legacy_qwen['pooled_random']['anchor_weighted_mean'])}，差值 {pct(f(legacy_qwen['pooled_selected']['anchor_weighted_mean'])-f(legacy_qwen['pooled_random']['anchor_weighted_mean']))}。最终 hybrid 改为每个 grammar 在自己的 events 内独立排序；K128 的 selected/random 为 {pct(q_primary['selected_failure_rate'])}/{pct(q_primary['random_failure_rate'])}。</p>{table(['Grammar','Anchors','Old shared P2 bank: selected / random','New grammar-specific: selected / random','Change in difference'],legacy_comparison_rows)}<p>整体下降来自 selected failures 从 72/87 降到 40/87；random 只从 11/261 变为 14/261。具体说，<code>adjacent_rank_after_city</code> 少了 23 次 selected failure，<code>same_unit_rank_before_city</code> 少了 11 次，而最纯净的 <code>adjacent_rank_before_city</code> 反而从 17/19 增到 19/19。因此旧共享 P2 bank 捕获到一组跨 grammar 更有因果作用的 heads；按 grammar 内 raw target mass 独立排序虽然更对称，却不保证更接近 causal importance。旧 ranking 还使用 <code>seed_first_equal_anchor_mean</code>，新 ranking 在每 seed 内平均该 grammar 的全部 events，这也是次要的 membership 变化来源。</p><div class="note"><span class="label">如何报告</span>78.7% 保留为“adjacent-P2 shared bank 的强 transfer comparator”；40.6% 是“grammar-specific、统一 P0 intervention 的最终对称设计”。前者说明共享 causal bottleneck 很强，后者说明 raw-attention ranking 的 grammar 内最优性并不等于 causal 最优性。</div><figure>{dose_svg(hybrid)}<figcaption>图 6 · 最终 hybrid confirmation dose response。每个 panel 横轴是 grammar-specific 排序的嵌套 Top-K，纵轴是首次 semantic city 的 failure rate。绿色=selected bank，灰色=三个 random controls pooled failure，虚线=selected−random。Qwen 与 Gemma 的 K 不是同一容量单位，不做横向数值等同。</figcaption></figure>{table(['Model','K','Confirmation anchors','Selected failure','Random failure','Selected−random'],dose_rows)}<h3>C1 · 冻结 primary K</h3>{table(['Model','Primary K','Confirmation anchors','Selected failure','Random failure','Selected−random'],primary_rows)}<p>Qwen 的效应随 bank 变宽，在 K112 后明显上升；K128 为 selected {pct(q_primary['selected_failure_rate'])}、random {pct(q_primary['random_failure_rate'])}，差值 {pct(q_primary['selected_minus_random_failure_rate'])}。Gemma K1 已出现 {pct(g['overall'][0]['selected_minus_random_failure_rate'])} 差值，K6 达到曲线最大 {pct(g['overall'][3]['selected_minus_random_failure_rate'])}；冻结 K8 时仍有 {pct(g_primary['selected_minus_random_failure_rate'])}，但 random failure 上升，因此曲线不应解释为每新增一个 head 都单调贡献正效应。</p><div class="example"><span class="label">简单例子</span>K64 包含某 grammar 排名前 64 个 heads，K96 完整保留它们并加入 rank 65–96。若 selected failure 随 K 上升、而三个同规模 random banks 仍较低，最合理的集合级解释是存在旁路或分布式冗余；它不证明第 96 个 head 单独必要。</div><div class="conclusion"><span class="label">Experiment C1 结论</span>最终 hybrid 设定仍显示 selection specificity：Qwen 需要宽 bank 才出现明显破坏，Gemma 用极少 heads 即产生效应。旧共享 P2 bank 更强，表明 causal heads 比 grammar 内 raw-attention 排名所暗示的更共享；query site、selection scope 与 event aggregation 都会实质改变 bank membership。</div></section>

<section id="grammar"><div class="section-tag">Experiment C2 · heterogeneity</div><h2>C2 · Grammar 分解：整体曲线由哪些 trace 类型组成</h2><div class="purpose"><span class="label">实验目的</span>检查 overall effect 是否由单一高频 grammar 驱动，并区分具有足够 confirmation anchors 的结果与低样本 exploratory 观察。</div><h3>Qwen · K{q['primary_k']}</h3>{table(['Grammar','Ranking query','Anchors','Selected','Random','Difference','Status'],grammar_rows('Qwen3-8B', q))}<h3>Gemma · K{g['primary_k']}</h3>{table(['Grammar','Ranking query','Anchors','Selected','Random','Difference','Status'],grammar_rows('Gemma4-E4B', g))}<p>Qwen 的 <code>adjacent_rank_before_city</code> 在 19 个 confirmation anchors 上达到 100% selected failure，而 random 为 15.8%，是最终整体效应的主要来源；<code>same_unit_rank_before_city</code> 的效应较弱。Gemma 则同时在 <code>adjacent_rank_after_city</code> 与高样本 <code>same_unit_rank_before_city</code> 上显示效应，但后者 K8 random damage 也更高。</p><div class="boundary"><span class="label">低样本边界</span>Qwen structural/evidence grammars 与 Gemma adjacent-rank-before/invariant grammars 的 confirmation anchors 少于 10，只能描述，不能据此声称该 grammar 的稳定总体效应。表中的 0% 或 100% 在 n=1 时尤其不能泛化。</div><div class="conclusion"><span class="label">Experiment C2 结论</span>“targeted retrieval”不是一个对所有 surface grammar 完全同质的统一 bank。更准确的表述是共享候选层上的 grammar-routed banks：P0-ranked 与 marker-conditioned P2-ranked 子回路共同组成 overall effect，且两模型的主导 grammar 不同。</div></section>

<section id="synthesis"><div class="section-tag">Mechanistic synthesis</div><h2>机制综合：当前最窄、可证伪的模型</h2><div class="flow"><div class="flow-node"><strong>1 · Commit progress at P0</strong><br><span>item k 完成；hidden state 携带可解码 running index。</span></div><div class="arrow">→</div><div class="flow-node"><strong>2 · Form a grammar-conditioned query</strong><br><span>city-first grammar 可在 P0 局部化；marker-first grammar 在 P2 更易局部化。</span></div><div class="arrow">→</div><div class="flow-node"><strong>3 · Retrieve and answer</strong><br><span>selected bank 从 prompt record 取回 next city；最终形成 count-specific answer state。</span></div></div><p>数据支持“P0 commit + grammar-conditioned retrieval query + bank-level necessity”的功能分解。P2 的作用目前只能称为更干净的 marker-conditioned localization site：因为 ablation 从 P0 已开始，我们知道这组 heads 从 transition 开始不可用会造成损害；但没有分离式 QK/OV patching 来证明 P0 专门读取、P2 专门写入。</p><div class="boundary"><span class="label">尚未证明</span>没有证明 selected bank 是唯一通路、每个 head 单独必要、progress state 必须经由这些 heads 传递，也没有用 restoration 把被破坏的 needle retrieval 救回。Representation、attention localization 与 causal necessity 已在同一报告对齐，但仍是三种不同 estimand。</div><div class="conclusion"><span class="label">当前总论</span>Native-thinking 模型在 item commits 与 answer query 上形成可解码的计数相关状态，并调用依赖 surface grammar 与 query stage 的 targeted-retrieval head banks。Gemma 回路窄、早期即显效；Qwen 回路宽，且 marker-first grammar 的 P2 ranking 是获得强因果效应的关键。</div></section>

<section id="audit"><div class="section-tag">Reproducibility</div><h2>复现账本与底层数据</h2><div class="links"><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><a href="v5_native_p0_head_atlas/p0_targeted_retrieval_head_scores.csv">P0 head scores</a><a href="v5_native_p0_head_atlas/p0_needle_ordinal_by_head.csv">Ordinal × head data</a><a href="v5_native_hybrid_supplement/native_hybrid_supplement_8gpu_complete.json">Hybrid completion</a><a href="NiaH_Native-Thinking_Causal_Ablation_report.html">Prior component page</a><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">Full P0 atlas</a></div><p>本页已吸收 causal component page 的冻结设计、dose/grammar 结果解释，以及 P0 atlas 的 score 定义、全类别 head map、ordinal decomposition 与单头 attention examples。旧页面保留为详细 provenance；主结论以 <code>v5_native_hybrid_supplement</code> 的 PASS manifests 为准。</p><details><summary>输入文件 SHA256</summary><p class="audit">Generated UTC: {esc(generated)}<br>{ledger}<br>Report schema: realistic_niah_v5_native_thinking_consolidated_v4</p></details><div class="conclusion"><span class="label">审计结论</span>Hybrid completion、两模型 dose/full-panel manifests、P0 atlas manifest 与 representation selected rows 均在生成时验证存在；两个 hybrid manifests 均要求 <code>status=PASS</code>、统一 P0 intervention start 与 persistent ablation，否则 builder 拒绝输出。</div></section>
</main></div></div></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-geometry-root", type=Path, default=Path("reports/v5_native_causal_aligned_geometry"))
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--hybrid-root", type=Path, default=Path("reports/v5_native_hybrid_supplement"))
    parser.add_argument("--atlas-root", type=Path, default=Path("reports/v5_native_p0_head_atlas"))
    parser.add_argument("--legacy-qwen-root", type=Path, default=Path("reports/v5_native_targeted_retrieval/Qwen3-8B"))
    parser.add_argument(
        "--legacy-qwen-selection",
        type=Path,
        default=Path("configs/realistic_niah_v5_qwen_shared_k125_full300_selection.json"),
    )
    parser.add_argument("--trajectory-registry", type=Path, default=Path("reports/v5_native_causal_site_review/trajectory_registry.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/NiaH_Native-Thinking_report.html"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/v5_native_thinking_mechanism_20260813/manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rep = load_representation(args.causal_geometry_root, args.dual_endpoint_root)
    hybrid = load_hybrid(args.hybrid_root)
    legacy_qwen = load_legacy_qwen(args.legacy_qwen_root, args.legacy_qwen_selection)
    atlas = load_atlas(args.atlas_root)
    duplicates = load_duplicates(args.trajectory_registry)
    completion = args.hybrid_root / "native_hybrid_supplement_8gpu_complete.json"
    completion_data = read_json(completion)
    require(completion_data.get("status") == "PASS", "hybrid completion is not PASS")
    inputs = [
        args.causal_geometry_root / "site_selected.csv",
        args.causal_geometry_root / "legacy_vs_causal_item_end.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        args.hybrid_root / "Qwen3-8B" / "analysis_hybrid_supplement_registered_v1" / "hybrid_dose_grid_complete.json",
        args.hybrid_root / "Qwen3-8B" / "analysis_hybrid_supplement_registered_v1" / "hybrid_full_panel_complete.json",
        args.hybrid_root / "Gemma4-E4B" / "analysis_hybrid_supplement_registered_v1" / "hybrid_dose_grid_complete.json",
        args.hybrid_root / "Gemma4-E4B" / "analysis_hybrid_supplement_registered_v1" / "hybrid_full_panel_complete.json",
        completion,
        args.legacy_qwen_root / "analysis_manifest.json",
        args.legacy_qwen_root / "raw_arm_rates.csv",
        args.legacy_qwen_root / "estimands.csv",
        args.legacy_qwen_selection,
        args.atlas_root / "p0_head_atlas_manifest.json",
        args.atlas_root / "p0_targeted_retrieval_head_scores.csv",
        args.atlas_root / "p0_needle_ordinal_by_head.csv",
        args.atlas_root / "p0_significant_head_attention_masses.csv",
        args.trajectory_registry,
    ]
    hashes = {str(path): sha256(path) for path in inputs}
    generated = datetime.now(timezone.utc).isoformat()
    document = build_report(rep, hybrid, legacy_qwen, atlas, duplicates, generated, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    manifest = {
        "schema_version": "realistic_niah_v5_native_thinking_consolidated_v4",
        "status": "complete",
        "generated_at": generated,
        "scope": [
            "representation",
            "p0_attention_atlas",
            "legacy_shared_bank_comparator",
            "hybrid_targeted_retrieval_ablation",
        ],
        "causal_contract": {
            "ranking": "grammar-specific P0 except rank-before-city uses post_marker P2",
            "intervention_start": "p0_item_end",
            "persistent_to_decode_end": True,
            "primary_split": "confirmation",
        },
        "input_sha256": hashes,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": manifest["output_sha256"], "manifest": str(args.manifest.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
