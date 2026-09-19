#!/usr/bin/env python3
"""Build the final Native-thinking V5 synthesis report.

The report intentionally contains only two evidence families:

1. representation geometry at frozen progress/final-answer token sites; and
2. grammar-routed targeted-retrieval head-bank ablation.

Parser and token-site definitions are included as experimental setup, not as
independent mechanism evidence. Superseded pilots and unconfirmed patching or
aggregation experiments are deliberately absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODELS = ("Qwen3-8B", "Gemma4-E4B")
MODEL_SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
MODEL_COLOR = {"Qwen3-8B": "#315f78", "Gemma4-E4B": "#17736b"}
EXPECTED_DOSES = {
    "Qwen3-8B": (32, 64, 80, 96, 112, 125),
    "Gemma4-E4B": (1, 2, 4, 6, 8),
}
REGISTRY_SHA = {
    "Qwen3-8B": "ed75562232fed47312eecc2562c80c825f7b9c48022ee5b27b4e783fc0ccbf12",
    "Gemma4-E4B": "021f1e5d3b95f232c0cf69c08236b592dbf2f54e8e04df6abaa27c097a0a96f8",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing hash input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float:
    result = float(value)
    require(math.isfinite(result), f"non-finite value: {value!r}")
    return result


def as_int(value: Any) -> int:
    return int(value)


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * as_float(value):.{digits}f}%"


def db(value: Any) -> str:
    return f"{as_float(value):+.2f} dB"


def pvalue(value: Any) -> str:
    if value in (None, ""):
        return "—"
    result = as_float(value)
    return f"{result:.4f}" if result >= 0.0001 else f"{result:.2e}"


def one(rows: Iterable[Mapping[str, str]], **conditions: Any) -> dict[str, str]:
    hits = [
        dict(row)
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in conditions.items())
    ]
    require(len(hits) == 1, f"expected one row for {conditions}, found {len(hits)}")
    return hits[0]


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{esc(cls)}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"invalid JSONL at {path}:{line_number}") from exc


def load_eligibility_audit(
    trajectory_registry: Path, qwen_generations: Path
) -> dict[str, Any]:
    duplicate_rows = [
        row
        for row in read_jsonl(trajectory_registry)
        if row.get("trace_category") == "full_coverage_with_duplicates"
    ]
    qwen_all = [row for row in duplicate_rows if row["model_label"] == "Qwen3-8B"]
    qwen_extra = [row for row in qwen_all if as_int(row["gold_count"]) >= 2]
    gemma_all = [row for row in duplicate_rows if row["model_label"] == "Gemma4-E4B"]

    require(len(qwen_all) == 17, f"Qwen duplicate traces changed: {len(qwen_all)}")
    require(len(qwen_extra) == 14, f"Qwen N>=2 duplicate traces changed: {len(qwen_extra)}")
    require(sum(as_int(row["gold_count"]) == 1 for row in qwen_all) == 3, "Qwen N=1 duplicate count")
    require(len(gemma_all) == 10, f"Gemma duplicate traces changed: {len(gemma_all)}")
    require(all(as_int(row["gold_count"]) == 1 for row in gemma_all), "Gemma duplicates escaped N=1")
    require(all(row["exact_count"] for row in gemma_all), "Gemma N=1 duplicate final answer changed")

    requested = {row["request_id"] for row in qwen_extra}
    generations = {
        row["request_id"]: row
        for row in read_jsonl(qwen_generations)
        if row.get("request_id") in requested
    }
    require(set(generations) == requested, "Qwen generation audit is missing duplicate traces")

    audited_rows: list[dict[str, Any]] = []
    for trace in sorted(qwen_extra, key=lambda row: (as_int(row["gold_count"]), as_int(row["seed"]))):
        generation = generations[trace["request_id"]]
        gold_records = generation["gold_records"]
        gold_cities = [record["city"] for record in gold_records]
        require(len(gold_cities) == as_int(trace["gold_count"]), f"{trace['request_id']}: gold record count")
        require(len(set(gold_cities)) == len(gold_cities), f"{trace['request_id']}: duplicate city in gold records")

        literal_counts = []
        for record in gold_records:
            sentence = (
                f"In the 2024 city score audit, {record['city']} "
                f"received a score of {record['score']}."
            )
            literal_counts.append(generation["rendered_prompt"].count(sentence))
        require(all(count == 1 for count in literal_counts), f"{trace['request_id']}: prompt literal multiplicity")

        event_counts: dict[str, int] = {}
        for event in trace["events"]:
            city = event["city"]
            event_counts[city] = event_counts.get(city, 0) + 1
        repeated = [(city, count) for city, count in event_counts.items() if count > 1]
        require(repeated, f"{trace['request_id']}: no repeated parsed event")

        audited_rows.append(
            {
                "seed": as_int(trace["seed"]),
                "gold_count": as_int(trace["gold_count"]),
                "split": trace["split"],
                "gold_cities": gold_cities,
                "repeated_events": repeated,
                "final_count": as_int(trace["parsed_count"]),
                "exact_count": bool(trace["exact_count"]),
            }
        )

    wrong = sum(not row["exact_count"] for row in audited_rows)
    confirmation = sum(row["split"] == "confirmation" for row in audited_rows)
    require(wrong == 11, f"Qwen duplicate wrong-answer count changed: {wrong}")
    require(confirmation == 3, f"Qwen confirmation duplicate count changed: {confirmation}")
    require(all(not row["exact_count"] for row in audited_rows if row["split"] == "confirmation"), "Qwen confirmation duplicate outcomes")

    return {
        "qwen_duplicate_total": len(qwen_all),
        "qwen_duplicate_n1": len(qwen_all) - len(qwen_extra),
        "qwen_duplicate_extra": len(qwen_extra),
        "qwen_wrong": wrong,
        "qwen_correct": len(audited_rows) - wrong,
        "qwen_confirmation_duplicate": confirmation,
        "qwen_rows": audited_rows,
        "gemma_duplicate_total": len(gemma_all),
        "gemma_duplicate_seeds": sorted(as_int(row["seed"]) for row in gemma_all),
    }


def load_representation(
    causal_geometry_root: Path, dual_endpoint_root: Path
) -> dict[str, dict[str, dict[str, str]]]:
    site_selected = read_csv(causal_geometry_root / "site_selected.csv")
    legacy_rows = read_csv(causal_geometry_root / "legacy_vs_causal_item_end.csv")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for model in MODELS:
        running = one(site_selected, model_label=model, site_kind="item_end")
        legacy = one(legacy_rows, model_label=model)
        final_rows = read_csv(
            dual_endpoint_root / model / "pca16_whiten" / "final_count_selected.csv"
        )
        final = one(
            final_rows,
            endpoint="final_count",
            model_label=model,
            mode="native_thinking",
        )
        require(as_int(running["confirmation_seed_count"]) == 10, f"{model}: running confirmation seeds")
        require(as_int(final["confirmation_seed_count"]) == 10, f"{model}: final confirmation seeds")
        result[model] = {"running": running, "final": final, "alignment": legacy}
    return result


def load_causal(
    causal_root: Path, selection_paths: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        root = causal_root / model
        manifest = read_json(root / "analysis_manifest.json")
        selection = read_json(selection_paths[model])
        require(manifest["analysis_status"] == "complete", f"{model}: analysis not complete")
        require(manifest["selection_contract_validated"], f"{model}: selection contract")
        doses = tuple(sorted(as_int(row["registered_bank_size"]) for row in manifest["runs"]))
        require(doses == EXPECTED_DOSES[model], f"{model}: unexpected K grid {doses}")
        require(
            manifest["clean_reference_anchor_registry_sha256"] == REGISTRY_SHA[model],
            f"{model}: registry SHA mismatch",
        )
        primary_k = as_int(selection["development_selection"]["primary_bank_size"])
        for run in manifest["runs"]:
            if as_int(run["registered_bank_size"]) == primary_k:
                expected_units = as_int(selection["anchor_registry"]["eligible_selected_prompts"])
            else:
                expected_units = as_int(selection["anchor_registry"]["registered_confirmation_support"]["anchors"])
            require(as_int(run["complete_five_arm_anchor_units"]) == expected_units, f"{model} K{run['registered_bank_size']}: incomplete anchors")
            require(as_int(run["incomplete_anchor_units"]) == 0, f"{model} K{run['registered_bank_size']}: incomplete units")
            require(run["anchor_registry_sha256"] == REGISTRY_SHA[model], f"{model}: run registry")

        estimands = read_csv(root / "estimands.csv")
        raw = read_csv(root / "raw_arm_rates.csv")
        count_estimands = read_csv(root / "count_estimands.csv")
        dose_rows = [
            one(
                estimands,
                registered_bank_size=k,
                evaluation_scope="confirmation",
                analysis_population="all_examples",
                grammar_class="pooled",
                estimand="selected_failure_minus_mean_random_failure",
            )
            for k in EXPECTED_DOSES[model]
        ]
        raw_rows = {
            arm: one(
                raw,
                registered_bank_size=primary_k,
                evaluation_scope="confirmation",
                analysis_population="all_examples",
                grammar_class="pooled",
                arm=arm,
                metric="failure_rate",
            )
            for arm in ("clean", "selected_bank", "layer_matched_random_mean")
        }
        scope_rows = {
            scope: one(
                estimands,
                registered_bank_size=primary_k,
                evaluation_scope=scope,
                analysis_population="all_examples",
                grammar_class="pooled",
                estimand="selected_failure_minus_mean_random_failure",
            )
            for scope in ("confirmation", "full_panel", "discovery")
        }
        clean_correct = one(
            estimands,
            registered_bank_size=primary_k,
            evaluation_scope="confirmation",
            analysis_population="clean_correct_only",
            grammar_class="pooled",
            estimand="selected_failure_minus_mean_random_failure",
        )
        grammar_rows = [
            row
            for row in estimands
            if as_int(row["registered_bank_size"]) == primary_k
            and row["evaluation_scope"] == "confirmation"
            and row["analysis_population"] == "all_examples"
            and row["estimand"] == "selected_failure_minus_mean_random_failure"
            and row["grammar_class"] != "pooled"
        ]
        counts = [
            row
            for row in count_estimands
            if as_int(row["registered_bank_size"]) == primary_k
            and row["evaluation_scope"] == "confirmation"
            and row["analysis_population"] == "all_examples"
            and row["estimand"] == "selected_failure_minus_mean_random_failure"
        ]
        counts.sort(key=lambda row: as_int(row["gold_count"]))
        result[model] = {
            "root": root,
            "manifest": manifest,
            "selection": selection,
            "primary_k": primary_k,
            "dose": dose_rows,
            "raw": raw_rows,
            "scope": scope_rows,
            "clean_correct": clean_correct,
            "grammar": grammar_rows,
            "counts": counts,
        }
    return result


def evidence_map_svg() -> str:
    return """<svg viewBox="0 0 980 250" role="img" aria-labelledby="emap-title emap-desc">
<title id="emap-title">Native-thinking V5 evidence map</title><desc id="emap-desc">Representation locates decodable progress and final-count states; ablation tests the necessity of grammar-routed retrieval head banks. A dotted connector marks the still untested mediation chain.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#69777b"/></marker></defs>
<rect x="28" y="42" width="280" height="144" rx="8" class="emap-box"/><text x="52" y="73" class="emap-kicker">REPRESENTATION</text><text x="52" y="105" class="emap-head">Progress state</text><text x="52" y="132" class="emap-copy">p0_item_end · running index</text><text x="52" y="160" class="emap-head">Final answer state</text><text x="52" y="184" class="emap-copy">answer_query_v3 · final count</text>
<rect x="350" y="42" width="280" height="144" rx="8" class="emap-box strong"/><text x="374" y="73" class="emap-kicker">ABLATION</text><text x="374" y="105" class="emap-head">Targeted retrieval</text><text x="374" y="132" class="emap-copy">same-site attention ranking</text><text x="374" y="160" class="emap-head">Persistent intervention</text><text x="374" y="184" class="emap-copy">selected vs layer-matched random</text>
<rect x="672" y="42" width="280" height="144" rx="8" class="emap-box muted"/><text x="696" y="73" class="emap-kicker">CURRENT BOUNDARY</text><text x="696" y="105" class="emap-head">Two established links</text><text x="696" y="132" class="emap-copy">state is decodable</text><text x="696" y="158" class="emap-copy">retrieval bank is necessary</text><text x="696" y="184" class="emap-copy">full mediation remains open</text>
<line x1="308" y1="114" x2="350" y2="114" class="emap-arrow dashed" marker-end="url(#arrow)"/><line x1="630" y1="114" x2="672" y2="114" class="emap-arrow" marker-end="url(#arrow)"/>
</svg>"""


def representation_svg(rep: Mapping[str, Mapping[str, Mapping[str, str]]]) -> str:
    width, height = 1000, 430
    top, bottom = 72.0, 310.0
    plot_h = bottom - top

    def y(value: float) -> float:
        return bottom - value * plot_h

    groups = [
        ("running", "Qwen3-8B", 150),
        ("running", "Gemma4-E4B", 350),
        ("final", "Qwen3-8B", 650),
        ("final", "Gemma4-E4B", 850),
    ]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="rep-title rep-desc">',
        '<title id="rep-title">Registered confirmation representation decoding</title>',
        '<desc id="rep-desc">Two panels compare Logistic and nearest-centroid balanced accuracy at the discovery-selected running progress and final answer layers for Qwen and Gemma.</desc>',
        '<text x="250" y="30" text-anchor="middle" class="chart-title">Running progress · exact p0_item_end</text>',
        '<text x="750" y="30" text-anchor="middle" class="chart-title">Final count · answer_query_v3</text>',
        '<line x1="500" y1="48" x2="500" y2="374" class="panel-divider"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(tick)
        parts.append(f'<line x1="54" y1="{yy:.1f}" x2="946" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="46" y="{yy + 4:.1f}" text-anchor="end" class="axis">{100*tick:.0f}%</text>')
    chance_y = y(0.1)
    parts.append(f'<line x1="54" y1="{chance_y:.1f}" x2="946" y2="{chance_y:.1f}" class="chance"/>')
    parts.append(f'<text x="950" y="{chance_y + 4:.1f}" class="axis">10% chance</text>')
    bar_w = 42
    metric_specs = (
        ("confirmation_logistic_balanced_accuracy", -24, "#17736b"),
        ("confirmation_ncc_balanced_accuracy", 24, "#b06f25"),
    )
    for endpoint, model, center in groups:
        row = rep[model][endpoint]
        for key, offset, color in metric_specs:
            value = as_float(row[key])
            xx = center + offset - bar_w / 2
            yy = y(value)
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w}" height="{bottom-yy:.1f}" fill="{color}" rx="3"/>')
            parts.append(f'<text x="{center+offset:.1f}" y="{yy-8:.1f}" text-anchor="middle" class="value">{100*value:.1f}%</text>')
        parts.append(f'<text x="{center}" y="338" text-anchor="middle" class="axis-label">{MODEL_SHORT[model]}</text>')
        parts.append(f'<text x="{center}" y="360" text-anchor="middle" class="axis">L{row["layer"]} · SNR {as_float(row["confirmation_class_balanced_snr_db"]):+.2f} dB</text>')
    parts.extend(
        [
            '<rect x="355" y="397" width="16" height="10" rx="2" fill="#17736b"/><text x="379" y="406" class="axis">Logistic balanced accuracy</text>',
            '<rect x="565" y="397" width="16" height="10" rx="2" fill="#b06f25"/><text x="589" y="406" class="axis">Nearest-centroid balanced accuracy</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def alignment_svg(rep: Mapping[str, Mapping[str, Mapping[str, str]]]) -> str:
    top, bottom = 62.0, 245.0
    low, high = 0.5, 0.85

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    parts = [
        '<svg viewBox="0 0 980 340" role="img" aria-labelledby="align-title align-desc">',
        '<title id="align-title">Legacy item-end versus exact causal progress commit</title>',
        '<desc id="align-desc">Slope chart compares confirmation Logistic and nearest-centroid balanced accuracy before and after restricting running states to exact causal primary progress commits.</desc>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8):
        yy = y(tick)
        parts.append(f'<line x1="54" y1="{yy:.1f}" x2="926" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="46" y="{yy+4:.1f}" text-anchor="end" class="axis">{100*tick:.0f}%</text>')
    for model, x0, x1 in (("Qwen3-8B", 130, 350), ("Gemma4-E4B", 600, 820)):
        row = rep[model]["alignment"]
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="28" text-anchor="middle" class="chart-title">{MODEL_SHORT[model]}</text>')
        for old_key, new_key, color, label in (
            ("legacy_confirmation_logistic_balanced_accuracy", "causal_confirmation_logistic_balanced_accuracy", "#17736b", "Logistic"),
            ("legacy_confirmation_ncc_balanced_accuracy", "causal_confirmation_ncc_balanced_accuracy", "#b06f25", "NCC"),
        ):
            old, new = as_float(row[old_key]), as_float(row[new_key])
            parts.append(f'<line x1="{x0}" y1="{y(old):.1f}" x2="{x1}" y2="{y(new):.1f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<circle cx="{x0}" cy="{y(old):.1f}" r="5" fill="{color}"/>')
            parts.append(f'<circle cx="{x1}" cy="{y(new):.1f}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{x0-8}" y="{y(old)-8:.1f}" text-anchor="end" class="value">{label} {100*old:.1f}%</text>')
            parts.append(f'<text x="{x1+8}" y="{y(new)-8:.1f}" class="value">{100*new:.1f}%</text>')
        parts.append(f'<text x="{x0}" y="272" text-anchor="middle" class="axis-label">all parsed item_end</text>')
        parts.append(f'<text x="{x1}" y="272" text-anchor="middle" class="axis-label">exact causal commit</text>')
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="302" text-anchor="middle" class="axis">SNR {as_float(row["legacy_confirmation_snr_db"]):+.2f} → {as_float(row["causal_confirmation_snr_db"]):+.2f} dB</text>')
    parts.append('</svg>')
    return "".join(parts)


def raw_arm_svg(causal: Mapping[str, Mapping[str, Any]]) -> str:
    top, bottom = 62.0, 285.0

    def y(value: float) -> float:
        return bottom - value * (bottom - top)

    labels = {
        "clean": ("Clean", "#96a19f"),
        "selected_bank": ("Ranked bank", "#9a4a44"),
        "layer_matched_random_mean": ("Matched random", "#315f78"),
    }
    parts = [
        '<svg viewBox="0 0 980 370" role="img" aria-labelledby="raw-title raw-desc">',
        '<title id="raw-title">Primary-K raw failure rates</title>',
        '<desc id="raw-desc">Grouped bars compare clean, ranked-bank ablation, and the mean of three layer-matched random banks on registered confirmation anchors.</desc>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(tick)
        parts.append(f'<line x1="54" y1="{yy:.1f}" x2="930" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="46" y="{yy+4:.1f}" text-anchor="end" class="axis">{100*tick:.0f}%</text>')
    for model, center in (("Qwen3-8B", 280), ("Gemma4-E4B", 700)):
        k = causal[model]["primary_k"]
        parts.append(f'<text x="{center}" y="30" text-anchor="middle" class="chart-title">{MODEL_SHORT[model]} · K{k}</text>')
        for index, arm in enumerate(labels):
            label, color = labels[arm]
            value = as_float(causal[model]["raw"][arm]["mean"])
            xx = center + (index - 1) * 106
            yy = y(value)
            parts.append(f'<rect x="{xx-34}" y="{yy:.1f}" width="68" height="{bottom-yy:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{xx}" y="{yy-9:.1f}" text-anchor="middle" class="value">{100*value:.1f}%</text>')
            parts.append(f'<text x="{xx}" y="312" text-anchor="middle" class="axis-label">{label}</text>')
    parts.append('<text x="492" y="350" text-anchor="middle" class="axis">纵轴：首次生成 city 未等于注册 next needle 的比例</text>')
    parts.append('</svg>')
    return "".join(parts)


def dose_svg(causal: Mapping[str, Mapping[str, Any]]) -> str:
    panels = (("Qwen3-8B", 54, 468), ("Gemma4-E4B", 526, 940))
    top, bottom = 68.0, 340.0

    def y(value: float) -> float:
        return bottom - value * (bottom - top)

    parts = [
        '<svg viewBox="0 0 980 450" role="img" aria-labelledby="dose-title dose-desc">',
        '<title id="dose-title">Registered confirmation dose response</title>',
        '<desc id="dose-desc">Two panels show selected-bank minus mean layer-matched-random failure across nested head-bank sizes, with seed-cluster bootstrap confidence intervals.</desc>',
    ]
    for model, left, right in panels:
        center = (left + right) / 2
        parts.append(f'<text x="{center:.1f}" y="30" text-anchor="middle" class="chart-title">{MODEL_SHORT[model]}</text>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y(tick)
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
            if left == panels[0][1]:
                parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" class="axis">{100*tick:.0f}%</text>')
        rows = causal[model]["dose"]
        step = (right - left) / max(1, len(rows) - 1)
        points = []
        for index, row in enumerate(rows):
            xx = left + index * step
            mean, lo, hi = as_float(row["mean"]), as_float(row["ci95_low"]), as_float(row["ci95_high"])
            points.append((xx, y(mean)))
            parts.append(f'<line x1="{xx:.1f}" y1="{y(lo):.1f}" x2="{xx:.1f}" y2="{y(hi):.1f}" class="ci"/>')
            parts.append(f'<line x1="{xx-5:.1f}" y1="{y(lo):.1f}" x2="{xx+5:.1f}" y2="{y(lo):.1f}" class="ci"/>')
            parts.append(f'<line x1="{xx-5:.1f}" y1="{y(hi):.1f}" x2="{xx+5:.1f}" y2="{y(hi):.1f}" class="ci"/>')
            parts.append(f'<text x="{xx:.1f}" y="368" text-anchor="middle" class="axis-label">K{row["registered_bank_size"]}</text>')
            parts.append(f'<text x="{xx:.1f}" y="{y(mean)-11:.1f}" text-anchor="middle" class="value">{100*mean:.1f}%</text>')
        path = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in points)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{MODEL_COLOR[model]}" stroke-width="3"/>')
        for xx, yy in points:
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.5" fill="{MODEL_COLOR[model]}" stroke="#fffdfa" stroke-width="2"/>')
    parts.append('<text x="497" y="420" text-anchor="middle" class="axis">横轴：嵌套 ranked head-bank 大小 K；纵轴：selected failure − mean(random failure)</text>')
    parts.append('</svg>')
    return "".join(parts)


CSS = r"""
:root{--paper:#f3efe7;--surface:#fffdfa;--ink:#1f2b30;--muted:#657177;--line:#d2cbc0;--deep:#20383a;--teal:#17736b;--teal-soft:#e5f0ed;--blue:#315f78;--blue-soft:#e7eef2;--amber:#a86718;--amber-soft:#f7ecd8;--red:#9a4a44;--red-soft:#f3e4e1}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.68 "Segoe UI Variable","Aptos","Noto Sans SC",system-ui,sans-serif}.layout{display:grid;grid-template-columns:250px minmax(0,1160px);gap:30px;max-width:1490px;margin:auto;padding:28px}nav{position:sticky;top:18px;align-self:start;max-height:calc(100dvh - 36px);overflow:auto;padding:20px 18px;background:#ebe5da;border-top:3px solid var(--teal)}nav strong{display:block;color:var(--deep);margin-bottom:12px}nav a{display:block;padding:7px 0;color:#4c585d;text-decoration:none;border-bottom:1px solid #d7d0c5;font-size:13px}main{min-width:0}header{padding:42px 46px;background:var(--deep);color:#f9fbf8;border-radius:12px;box-shadow:0 18px 45px #263c3920}.eyebrow{letter-spacing:.13em;font-size:12px;font-weight:850;color:#a9cec7}.scope-pill{display:inline-block;margin-top:8px;padding:5px 10px;border:1px solid #86aaa4;color:#d9ebe7;font-size:12px;font-weight:800}h1{font-size:42px;line-height:1.12;letter-spacing:-.045em;margin:9px 0 16px;max-width:920px}h2{font-size:29px;line-height:1.2;letter-spacing:-.03em;margin:0 0 18px;color:var(--deep)}h3{font-size:19px;line-height:1.35;margin:28px 0 10px;color:var(--blue)}p{max-width:82ch}.lead{font-size:18px;max-width:900px}.meta{color:#c5d6d2}.small{font-size:12px;color:var(--muted)}section{scroll-margin-top:18px;margin-top:22px;padding:32px 36px;background:var(--surface);border:1px solid var(--line);border-radius:10px}.summary-strip{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:18px 0}.summary-strip>div{padding:17px 16px;border-right:1px solid var(--line)}.summary-strip>div:last-child{border-right:0}.summary-strip strong{display:block;font:750 24px/1.15 "Cascadia Mono",Consolas,monospace;color:var(--teal)}.summary-strip span{font-size:12px;color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.two>div{min-width:0;padding:20px}.two>div+div{border-left:1px solid var(--line)}.purpose,.definition,.conclusion,.example,.boundary{padding:16px 19px;margin:15px 0}.purpose{background:var(--blue-soft);border-left:4px solid var(--blue)}.definition{background:#f0eee8;border-left:4px solid #7c8588}.conclusion{background:var(--teal-soft);border-left:4px solid var(--teal)}.example{background:var(--amber-soft);border-left:4px solid var(--amber)}.boundary{background:var(--red-soft);border-left:4px solid var(--red)}.label{display:block;margin-bottom:4px;font-size:11px;letter-spacing:.12em;font-weight:850;color:var(--muted)}.formula{margin:14px 0;padding:16px 18px;background:#29383c;color:#f6f4ef;overflow:auto;font:13px/1.75 "Cascadia Mono",Consolas,monospace;white-space:pre-wrap}.table-wrap{overflow:auto;border:1px solid var(--line);margin:15px 0 21px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 11px;text-align:left;vertical-align:top;border-bottom:1px solid #ded8ce}th{background:#ece7dd;color:#344047;white-space:nowrap}tr:last-child td{border-bottom:0}.audit-table{min-width:900px}.audit-table td:nth-child(4){min-width:300px}code{font-family:"Cascadia Mono",Consolas,monospace;background:#efebe3;padding:.08em .3em;border-radius:3px}figure{margin:23px 0;padding:18px 0 14px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);overflow:auto}figure svg{display:block;width:100%;min-width:720px;height:auto}figcaption{max-width:96ch;margin:11px auto 0;color:var(--muted);font-size:13px}.chart-title{font:750 15px "Segoe UI Variable","Aptos",sans-serif;fill:var(--ink)}.axis-label{font:650 12px "Segoe UI Variable","Aptos",sans-serif;fill:var(--ink)}.axis,.value{font:12px "Cascadia Mono",Consolas,monospace;fill:#5e696e}.value{font-weight:750}.grid{stroke:#d8d2c8;stroke-width:1}.chance{stroke:#9d6e34;stroke-width:1.5;stroke-dasharray:5 5}.panel-divider{stroke:#b9b1a6;stroke-width:1}.ci{stroke:#536f76;stroke-width:3;stroke-linecap:round}.emap-box{fill:#f7f3eb;stroke:#bfc6c4;stroke-width:1.2}.emap-box.strong{fill:#e4efec;stroke:#78a19a}.emap-box.muted{fill:#eeeae2;stroke:#c5bdb1}.emap-kicker{font:800 11px "Cascadia Mono",Consolas,monospace;letter-spacing:.09em;fill:#5e6c70}.emap-head{font:750 17px "Segoe UI Variable","Aptos",sans-serif;fill:#20383a}.emap-copy{font:13px "Segoe UI Variable","Aptos",sans-serif;fill:#536167}.emap-arrow{stroke:#69777b;stroke-width:2}.emap-arrow.dashed{stroke-dasharray:5 5}.mechanism-line{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:12px;margin:20px 0}.mechanism-node{padding:16px;border-top:2px solid var(--teal);background:#f2f0ea}.mechanism-arrow{color:var(--muted);font:18px "Cascadia Mono",monospace}.links{display:flex;gap:9px;flex-wrap:wrap}.links a{padding:8px 11px;border:1px solid var(--line);color:var(--blue);text-decoration:none;background:#f7f3eb}.audit{font:11px/1.6 "Cascadia Mono",Consolas,monospace;color:#697176;word-break:break-all}details{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--blue);font-weight:750}@media(max-width:960px){.layout{display:block;padding:13px}nav{position:relative;top:0;max-height:none;margin-bottom:13px}.two{grid-template-columns:1fr}.two>div+div{border-left:0;border-top:1px solid var(--line)}header,section{padding:25px}.summary-strip{grid-template-columns:1fr 1fr}.summary-strip>div:nth-child(2){border-right:0}.mechanism-line{grid-template-columns:1fr}.mechanism-arrow{text-align:center;transform:rotate(90deg)}h1{font-size:34px}}@media(max-width:560px){header,section{padding:21px 17px}.table-wrap{margin-left:-17px;margin-right:-17px;border-left:0;border-right:0}.summary-strip{grid-template-columns:1fr}.summary-strip>div{border-right:0;border-bottom:1px solid var(--line)}.summary-strip>div:last-child{border-bottom:0}h2{font-size:25px}}@media print{body{background:white}.layout{display:block;padding:0}nav{display:none}header,section{box-shadow:none;break-inside:avoid}figure{break-inside:avoid}}
"""


def build_report(
    rep: Mapping[str, Mapping[str, Mapping[str, str]]],
    causal: Mapping[str, Mapping[str, Any]],
    eligibility: Mapping[str, Any],
    *,
    generated: str,
    input_hashes: Mapping[str, str],
) -> str:
    q_run, g_run = rep["Qwen3-8B"]["running"], rep["Gemma4-E4B"]["running"]
    q_final, g_final = rep["Qwen3-8B"]["final"], rep["Gemma4-E4B"]["final"]
    q_primary, g_primary = causal["Qwen3-8B"]["dose"][-1], causal["Gemma4-E4B"]["dose"][-1]
    setting_rows = []
    for model in MODELS:
        selection = causal[model]["selection"]
        route = "rank-before → post_marker；其余 → p0_item_end" if model == "Qwen3-8B" else "全部 grammar → p0_item_end"
        setting_rows.append((esc(MODEL_SHORT[model]), esc(route), str(selection["anchor_registry"]["registered_confirmation_support"]["anchors"]), f"K{selection['development_selection']['primary_bank_size']}", f"<code>{REGISTRY_SHA[model][:12]}…</code>"))
    representation_rows = []
    for model in MODELS:
        for endpoint, label in (("running", "Running progress"), ("final", "Final count")):
            row = rep[model][endpoint]
            representation_rows.append((esc(MODEL_SHORT[model]), esc(label), f"<code>{row['token_site'] if 'token_site' in row else row['site_kind']}</code>", f"L{row['layer']}", pct(row["confirmation_logistic_balanced_accuracy"]), pct(row["confirmation_ncc_balanced_accuracy"]), db(row["confirmation_class_balanced_snr_db"]), f"{row['confirmation_rows']} / {row['confirmation_seed_count']}"))
    alignment_rows = []
    for model in MODELS:
        row = rep[model]["alignment"]
        alignment_rows.append((esc(MODEL_SHORT[model]), f"{pct(row['legacy_confirmation_logistic_balanced_accuracy'])} → {pct(row['causal_confirmation_logistic_balanced_accuracy'])}", f"{pct(row['legacy_confirmation_ncc_balanced_accuracy'])} → {pct(row['causal_confirmation_ncc_balanced_accuracy'])}", f"{db(row['legacy_confirmation_snr_db'])} → {db(row['causal_confirmation_snr_db'])}"))
    raw_rows = []
    for model in MODELS:
        for arm, label in (("clean", "Clean"), ("selected_bank", "Ranked bank"), ("layer_matched_random_mean", "3× layer-matched random mean")):
            row = causal[model]["raw"][arm]
            raw_rows.append((esc(MODEL_SHORT[model]), f"K{causal[model]['primary_k']}", esc(label), pct(row["mean"]), f"{pct(row['ci95_low'])} – {pct(row['ci95_high'])}", row["n_anchor_units"]))
    dose_rows = []
    for model in MODELS:
        for row in causal[model]["dose"]:
            dose_rows.append((esc(MODEL_SHORT[model]), f"K{row['registered_bank_size']}", pct(row["mean"]), f"{pct(row['ci95_low'])} – {pct(row['ci95_high'])}", pvalue(row["holm_p"]), row["n_seeds"], row["n_anchor_units"]))
    scope_rows = []
    for model in MODELS:
        for scope, label in (("confirmation", "Registered confirmation"), ("full_panel", "Full 300 coverage"), ("discovery", "Discovery")):
            row = causal[model]["scope"][scope]
            scope_rows.append((esc(MODEL_SHORT[model]), esc(label), pct(row["mean"]), f"{pct(row['ci95_low'])} – {pct(row['ci95_high'])}", row["n_seeds"], row["n_anchor_units"]))
    grammar_labels = {"macro_primary_grammars": "Equal-primary-grammar macro", "adjacent_rank_after_city": "adjacent rank-after-city", "adjacent_rank_before_city": "adjacent rank-before-city", "same_unit_rank_before_city": "same-unit rank-before-city", "structural_unmarked": "structural unmarked", "structural_invariant_bullet": "invariant bullet", "evidence_sequence_unranked": "evidence sequence unranked"}
    grammar_rows = []
    for model in MODELS:
        ordered = sorted(causal[model]["grammar"], key=lambda row: (0 if row["grammar_class"] == "macro_primary_grammars" else 1, -as_int(row["n_anchor_units"]), row["grammar_class"]))
        for row in ordered:
            if row["grammar_class"] in grammar_labels:
                grammar_rows.append((esc(MODEL_SHORT[model]), esc(grammar_labels[row["grammar_class"]]), pct(row["mean"]), f"{pct(row['ci95_low'])} – {pct(row['ci95_high'])}", row["n_seeds"], row["n_anchor_units"], esc(row["inferential_status"])))
    q_counts = {as_int(row["gold_count"]): row for row in causal["Qwen3-8B"]["counts"]}
    g_counts = {as_int(row["gold_count"]): row for row in causal["Gemma4-E4B"]["counts"]}
    count_rows = [(str(count), pct(q_counts[count]["mean"]) if count in q_counts else "N/A", pct(g_counts[count]["mean"]) if count in g_counts else "N/A", f"{q_counts[count]['n_anchor_units'] if count in q_counts else 0} / {g_counts[count]['n_anchor_units'] if count in g_counts else 0}") for count in range(1, 11)]
    ledger = "<br>".join(f"{esc(path)}: {esc(digest)}" for path, digest in sorted(input_hashes.items()))
    q_macro = one(causal["Qwen3-8B"]["grammar"], grammar_class="macro_primary_grammars")
    g_macro = one(causal["Gemma4-E4B"]["grammar"], grammar_class="macro_primary_grammars")
    sample_flow_rows = [
        (
            "Qwen",
            "300",
            "30",
            f"{eligibility['qwen_duplicate_total']}（N=1: {eligibility['qwen_duplicate_n1']}；N≥2: {eligibility['qwen_duplicate_extra']}）",
            str(eligibility["qwen_duplicate_extra"]),
            "256",
        ),
        (
            "Gemma",
            "300",
            "30",
            f"{eligibility['gemma_duplicate_total']}（全部 N=1）",
            "0",
            "270",
        ),
    ]
    duplicate_audit_rows = []
    for row in eligibility["qwen_rows"]:
        gold = " → ".join(esc(city) for city in row["gold_cities"])
        repeated = "；".join(f"{esc(city)} × {count}" for city, count in row["repeated_events"])
        duplicate_audit_rows.append(
            (
                str(row["seed"]),
                str(row["gold_count"]),
                esc(row["split"]),
                gold,
                repeated,
                str(row["final_count"]),
                "正确" if row["exact_count"] else "错误",
            )
        )
    gemma_duplicate_seed_text = ", ".join(str(seed) for seed in eligibility["gemma_duplicate_seeds"])

    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native-thinking V5 · Representation and Ablation</title><style>{CSS}</style></head><body><div class="layout">
<nav><strong>Native-thinking V5</strong><a href="#summary">结论摘要</a><a href="#setting">1 · 任务与冻结设定</a><a href="#metrics">2 · 定义与计算</a><a href="#rep-running">A1 · Running representation</a><a href="#rep-final">A2 · Final-count representation</a><a href="#rep-align">A3 · Causal-site 对齐</a><a href="#ablation-design">B1 · Ablation 设计</a><a href="#ablation-results">B2 · Raw arms 与 dose</a><a href="#transfer">B3 · Grammar / count</a><a href="#synthesis">机制拼接与边界</a><a href="#audit">复现账本</a></nav><main>
<header><div class="eyebrow">REALISTIC NIAH · NATIVE THINKING V5 · FINAL SYNTHESIS</div><h1>Native-thinking 如何计数：可解码的进度状态与 grammar-routed targeted retrieval</h1><p class="lead">本报告只保留两类已经形成清晰证据链的实验：representation 用来定位“模型状态里有什么”，targeted-retrieval ablation 用来判断“哪些 attention heads 对下一次检索是必要的”。两者严格分开解释，再在最后给出当前能够支持的最窄机制。</p><span class="scope-pill">FINAL · REPRESENTATION + ABLATION ONLY</span><p class="meta">Qwen3-8B / Gemma4-E4B · old300 · discovery 1234–1253 · registered confirmation 1254–1263</p></header>
<section id="summary"><h2>结论摘要：现在有两段证据，但还不是完整的因果链</h2><div class="summary-strip"><div><strong>{pct(q_run['confirmation_logistic_balanced_accuracy'])}</strong><span>Qwen running-index Logistic BA</span></div><div><strong>{pct(g_run['confirmation_logistic_balanced_accuracy'])}</strong><span>Gemma running-index Logistic BA</span></div><div><strong>{pct(q_primary['mean'])}</strong><span>Qwen K125 selected−random failure</span></div><div><strong>{pct(g_primary['mean'])}</strong><span>Gemma K8 selected−random failure</span></div></div><figure>{evidence_map_svg()}<figcaption>图 1 · 当前证据地图。左侧 representation 只回答 hidden state 是否包含可解码的 running index / final count；中间 ablation 直接测试 retrieval head bank 的必要性；右侧是目前可支持的联合解释。左到中的虚线表示尚未通过 restoration/mediation 证明“progress representation 必然经由该 bank 被使用”，因此不能把两段结果写成完整闭环。</figcaption></figure><p>Qwen 与 Gemma 都在完成一个 trace item 后形成可解码的 progress state，也都在输出最终数字前形成 final-count state。因果上，两者都依赖按 discovery attention mass 排出的 retrieval bank，但宽度差异很大：Qwen 需要约 K112–K125 才进入高破坏区，Gemma 的 K8 已经产生强效应。</p><div class="boundary"><span class="label">解释边界</span>本报告只纳入已经形成冻结设定与确认性结果的 representation 和 targeted-retrieval ablation；其他尚未形成稳健确认性结果的实验不进入正文。</div><div class="conclusion"><span class="label">目前能得到的结论</span>Native-thinking 已确认“进度/答案状态可读”和“下一条 needle 的 attention-head retrieval bank 具有集合必要性”；尚未确认两者之间的完整中介链，也未证明单个 head 是独立计数器。</div></section>
<section id="setting"><h2>1 · 任务与冻结设定：比较的是同一批 prompts 上的状态与干预</h2><div class="purpose"><span class="label">本节目的</span>在读任何图之前，先固定样本、transition、token-site 与 inference unit，避免把“trace 写法”“token 位置”和“机制效应”混成同一个变量。</div><p>Panel 包含 gold count N=1…10，每个 N 有 30 个 seeds，共 300 prompts。Seeds 1234–1253 用于 discovery（选层、排 head、冻结 K）；1254–1263 用于 registered confirmation。早期 parser/site smoke 曾接触旧 panel 中的个别 prompt，因此我们称 registered confirmation，而不是 pristine held-out。</p><div class="two"><div><h3>Representation 的观测位置</h3><p><code>p0_item_end</code> 是 transition k→k+1 中完整 item k 的最后一个真实 output token；<code>answer_query_v3</code> 是 thinking trace 最后一个字面 token、紧邻最终数字生成之前的位置。</p></div><div><h3>Ablation 的干预位置</h3><p>Gemma 的所有 grammar 都从 <code>p0_item_end</code> 开始；Qwen 的 rank-before-city grammar 从 target marker 的 rank-core 末 token（<code>post_marker</code>）开始，其余 grammar 从 <code>p0_item_end</code> 开始。</p></div></div>{table(['Model','Frozen routing','Confirmation anchors','Primary K','Registry SHA'],setting_rows)}<div class="example"><span class="label">简单例子</span>若 trace 已输出 <code>(Record 2: Riga, 60)</code>，那么 <code>p0_item_end</code> 是右括号所在的真实 token；此时第 2 项已经提交，第 3 项的 marker 与 city 尚未生成。若下一项写成 <code>3. Osaka</code>，Qwen 的 <code>post_marker</code> 则落在表达“3”的 rank semantic core 末 token，仍在 <code>Osaka</code> 首 token 之前。</div><h3>1.1 Ablation 样本流与 duplicate trace 审计</h3><p>N=1 没有 k→k+1 transition，因此保留在 300-prompt mother panel 中，但 targeted-retrieval ablation 记为 not applicable。这里的 <em>duplicate trace</em> 指 parser 在模型的 reasoning trace 中识别到同一 target city 的重复 event；它不等于输入 prompt 本身重复了该 city-score record。</p>{table(['Model','Original panel','N=1 no transition','Duplicate traces in reasoning','Extra exclusions in N=2…10','Final causal anchors'],sample_flow_rows)}<p>Gemma 并非完全没有 duplicate：共有 {eligibility['gemma_duplicate_total']} 条（seeds {gemma_duplicate_seed_text}），但全部位于 N=1，最终答案也全部为 1。它们已包含在 30 条 N=1 not-applicable prompts 中，因此不会再从 N=2…10 的 270 条里扣除样本；Gemma 最终保留全部 270 个 causal anchors。</p><p>Qwen 在 300 条中共有 {eligibility['qwen_duplicate_total']} 条 duplicate trace：其中 {eligibility['qwen_duplicate_n1']} 条位于 N=1，另有 {eligibility['qwen_duplicate_extra']} 条位于 N≥2，后者才使 causal registry 从 270 降到 256。对这 14 条逐一检查原始 <code>rendered_prompt</code> 后，每条 active city-score audit sentence 都只出现一次，原文不存在重复 target record；重复是模型在 reasoning 中自行再次提及同一 city。最终答案中 {eligibility['qwen_wrong']}/14 错误、{eligibility['qwen_correct']}/14 正确；3 条 confirmation duplicates 全部答错。</p><details><summary>展开 Qwen 14 条 N≥2 duplicate trace：原文与最终答案逐条审计</summary><p>“原文 target cities”按输入中实际顺序列出，每个对应的完整 audit sentence 在 <code>rendered_prompt</code> 中恰好出现一次；“trace 重复”来自 parser-observed reasoning events。Final 是模型最终输出的 count，而不是 event 数的事后重算。</p>{table(['Seed','N','Split','Original target cities (each once)','Repeated in reasoning trace','Final','Outcome'],duplicate_audit_rows,'audit-table')}</details><div class="example"><span class="label">duplicate 例子</span>Qwen seed1234、N=2 的原文只含 <code>Chicago → Baku</code>，两条 audit sentence 各出现一次；模型 reasoning 却把 <code>Baku</code> 当成第三条再次列出，最后回答 3，因此错误。也有三条 trace 虽重复提及 city，最终仍回到正确 count；所以 exclusion 依据的是“第 k 个 occurrence 是否仍可无歧义对应第 k 个 unique needle”，而不是按答案对错筛样本。</div><p>每个可用 prompt 至多选择一个 registered transition；统计时先在 seed 内平均 prompts，再让 seeds 等权，避免某个 seed 或 grammar 因 anchor 较多而主导结果。</p><div class="conclusion"><span class="label">本节结论</span>两个模型都出现过模型内部的 duplicate reasoning event；Gemma 的 10 条全部落在本来就无 transition 的 N=1，因此没有额外样本损失。Qwen 的 14 条 N≥2 duplicates 均不是输入原文重复，也不是 token re-encode/parser 对齐失败，而是 clean trace 无法提供唯一的 k→k+1 occurrence mapping，故冻结 registry 为 256。</div></section>
<section id="metrics"><h2>2 · 通用测量框架：新概念如何计算</h2><div class="purpose"><span class="label">本节目的</span>明确每个指标的计算方法及它能支持的推断，防止把“高可解码性”误写成“因果必要性”，或把 generic damage 误写成 targeted retrieval。</div><h3>2.1 Representation：discovery-only 选层，confirmation 只评估</h3><p>每层 hidden states 先用 discovery rows 拟合 StandardScaler，再拟合 whitened PCA-16。Discovery 的 grouped out-of-fold folds 以 seed 分组；在每个 fold 内分别拟合 L2 multinomial Logistic 与 nearest-centroid classifier（NCC）。选层分数为两者 balanced accuracy 的平均：</p><div class="formula">BA = (1/C) · Σ_c TP_c / (TP_c + FN_c)\nselection_score(layer) = 0.5 · [BA_Logistic(layer) + BA_NCC(layer)]\nC = 10, therefore chance balanced accuracy = 0.10</div><p>选出 discovery score 最大的 layer 后才读取该层 confirmation 指标。NCC 直接把测试点分给 discovery PCA space 中最近的 class centroid；Logistic 允许线性 decision boundaries。两者都高说明 count 结构不只依赖一种 decoder。</p><p>Class-balanced isotropic SNR 先计算每个 count centroid 到 class-balanced grand mean 的平方距离均值（signal），再计算每类样本到本类 centroid 的平方残差均值（noise）：</p><div class="formula">S = mean_c || μ_c − mean_j μ_j ||²\nW = mean_c mean_{{i:y_i=c}} || z_i − μ_c ||²\nSNR_dB = 10 · log10(S / W)</div><p>负 dB 不表示“没有信息”；它表示样本内噪声能量仍大于 centroid 间信号。高 BA 与负 SNR 可以同时出现，意味着可分结构存在但并不紧致。</p><div class="example"><span class="label">简单例子</span>如果三个 count 类别的 recall 分别为 80%、50%、70%，balanced accuracy 是 (0.8+0.5+0.7)/3=66.7%，不会因为某一类样本更多而被加权放大。</div><h3>2.2 Ablation：同位排名、逐层匹配 control、自由生成 endpoint</h3><p>对每个 discovery anchor i、head h，在与干预相同的 query token qᵢ 上，累计该 head 对注册 next-needle prompt-record span Sᵢ 的 attention mass：</p><div class="formula">m(i,h) = Σ_{{t ∈ S_i}} A_h(q_i,t)\nscore(h) = mean_seed [ mean_anchor-within-seed m(i,h) ]</div><p>按 score 得到冻结有序 bank，K 较小的 bank 是较大 bank 的精确前缀。干预把这些 heads 的 pre-O slice 在 query token 以及所有后续 cached decode forwards 中持续清零；三组 random banks 在每一层的 head 数与 selected bank 完全一致。</p><div class="formula">failure(i,arm) = 1[first semantic generated city ≠ registered next city]\nΔ_i = failure(i,selected) − (1/3) Σ_{{r=1}}^3 failure(i,random_r)</div><p>先按 seed 聚合 Δ，再对 seeds 等权。95% CI 使用 10,000 次 seed-cluster bootstrap；双侧 p 值使用 seed-level sign-flip，K-grid 内做 Holm 校正。</p><div class="example"><span class="label">简单例子</span>Clean 和三个 random runs 都首先输出注册 city <code>Prague</code>，selected-bank run 却没有输出可识别 city，则 selected failure=1、random mean failure=0，本 anchor 的 Δ=1。若 selected 与 random 都失败，Δ 接近 0，不能算 targeted specificity。</div><div class="conclusion"><span class="label">本节结论</span>Representation 的主单位是 held-out state decoding；ablation 的主单位是 paired transition failure contrast。两个指标不可互换，只有后者直接提供 bank-level necessity 证据。</div></section>
<section id="rep-running"><h2>Experiment A1 · Running representation：完成 item k 后，模型是否知道当前进度 k？</h2><div class="purpose"><span class="label">实验目的</span>检验 exact causal progress commit 上的 hidden state 是否包含可泛化到新 seeds 的 running index，而不是只反映某一种 marker 字符或训练样本身份。</div><p>Primary site 固定为 causal compiler 注册的 <code>p0_item_end</code>；只在 discovery 上搜索 decoder layer。Qwen 选择 L{q_run['layer']}，Gemma 选择 L{g_run['layer']}。Confirmation 分别包含 {q_run['confirmation_rows']} 与 {g_run['confirmation_rows']} 个 parser-observed states，来自 10 个 seed-disjoint seeds。</p>{table(['Model','Endpoint','Token site','Layer','Logistic BA','NCC BA','SNR','Rows / seeds'],[row for row in representation_rows if row[1]=='Running progress'])}<p>Qwen 的 Logistic/NCC 为 {pct(q_run['confirmation_logistic_balanced_accuracy'])}/{pct(q_run['confirmation_ncc_balanced_accuracy'])}，Gemma 为 {pct(g_run['confirmation_logistic_balanced_accuracy'])}/{pct(g_run['confirmation_ncc_balanced_accuracy'])}，均明显高于十类 chance 10%。但 SNR 仍为 {db(q_run['confirmation_class_balanced_snr_db'])} 与 {db(g_run['confirmation_class_balanced_snr_db'])}，说明 progress state 可线性/centroid 解码，却仍是分布式且有较大 trajectory variation 的状态。多种 grammar 与 seed-disjoint confirmation 降低了“只记住单一 marker 字符”的解释，但当前结果仍不足以证明一个完全不受 surface grammar 影响的标量 counter。</p><div class="example"><span class="label">简单例子</span>把未参与选层的 seed1258 中第 4 个 completed item 的 hidden vector 投影到 discovery-frozen PCA16；若 Logistic 与最近 centroid 都把它判为 running index 4，就算一次正确 held-out decoding。这个实验不要求模型接下来一定输出正确 city。</div><div class="conclusion"><span class="label">本实验结论</span>两个模型都在 item commit 位置显式携带 running progress；Gemma 的 held-out decodability 更高，但二者都不是“低噪声单轴计数器”，也尚不能称为完全 marker-invariant。</div></section>
<section id="rep-final"><h2>Experiment A2 · Final-count representation：输出最终数字前是否形成可执行答案状态？</h2><div class="purpose"><span class="label">实验目的</span>区分“中间 trace 能记录进度”与“最终 answer query 已经组织出 gold count”这两个不同阶段。</div><p><code>answer_query_v3</code> 固定为最后一个 thinking-trace literal token，紧邻数值答案生成之前。每个 count 在 discovery 有 20 个 states、confirmation 有 10 个 states；layer 仍只由 discovery selection score 决定。</p>{table(['Model','Endpoint','Token site','Layer','Logistic BA','NCC BA','SNR','Rows / seeds'],[row for row in representation_rows if row[1]=='Final count'])}<figure>{representation_svg(rep)}<figcaption>图 2 · 两个冻结 endpoint 的 registered-confirmation 解码结果。横轴按 endpoint 分为 running progress（左）和 final count（右），每个 endpoint 内分别显示 Qwen、Gemma；纵轴为十类 balanced accuracy，虚线 10% 是 chance。绿色柱为 Logistic，棕色柱为 NCC；组下方标注 discovery-selected layer 与 confirmation SNR。图只解释被选层的 held-out point estimate，不把所有 confirmation layers 用于重新选层。</figcaption></figure><p>Qwen final state 在 L{q_final['layer']} 达到 Logistic {pct(q_final['confirmation_logistic_balanced_accuracy'])}、NCC {pct(q_final['confirmation_ncc_balanced_accuracy'])}，SNR 转为 {db(q_final['confirmation_class_balanced_snr_db'])}；Gemma L{g_final['layer']} 为 {pct(g_final['confirmation_logistic_balanced_accuracy'])}/{pct(g_final['confirmation_ncc_balanced_accuracy'])}，SNR {db(g_final['confirmation_class_balanced_snr_db'])}。因此 Qwen 的最终答案几何几乎离散化，Gemma 则仍可解码但更重叠。</p><div class="example"><span class="label">简单例子</span>当 gold count 是 7 时，只读取模型尚未输出数字前的 <code>answer_query_v3</code> hidden state。若 discovery-fitted decoder 在 confirmation seed 上预测 7，则表明“7”在输出前已存在于状态中；它不等价于证明哪个 component 写入了该状态。</div><div class="conclusion"><span class="label">本实验结论</span>Native-thinking 不只是沿 trace 保留 running progress；在最终输出前还形成了 count-specific answer state。Qwen 的 final state 比 running state 显著更紧致，Gemma 的整合较弱但仍远高于 chance。</div></section>
<section id="rep-align"><h2>Experiment A3 · Causal-site 对齐：更精确的 token 位置是否让 counter 更紧致？</h2><div class="purpose"><span class="label">实验目的</span>直接检验把旧的“所有 parser-observed item_end”收窄为 causal compiler 实际使用的 primary progress commits，是否实质改变 representation 结论。</div><figure>{alignment_svg(rep)}<figcaption>图 3 · Token-site 收窄前后的 confirmation decodability。每个模型中，横轴从旧 all-item-end cohort 移到 exact causal primary commits；纵轴为 balanced accuracy，绿色线是 Logistic，棕色线是 NCC。下方同时给出 SNR_dB 的变化。该图比较的是 cohort/token 对齐，不是干预效应。</figcaption></figure>{table(['Model','Logistic BA','NCC BA','SNR_dB'],alignment_rows)}<p>精确对齐后 Qwen Logistic/NCC 仅增加约 1.7/1.2 个百分点；Gemma 增加约 2.4/3.4 个百分点。与此同时 SNR 没有提高，Qwen {db(rep['Qwen3-8B']['alignment']['legacy_confirmation_snr_db'])}→{db(rep['Qwen3-8B']['alignment']['causal_confirmation_snr_db'])}，Gemma {db(rep['Gemma4-E4B']['alignment']['legacy_confirmation_snr_db'])}→{db(rep['Gemma4-E4B']['alignment']['causal_confirmation_snr_db'])}。</p><div class="example"><span class="label">简单例子</span>旧 cohort 可能把同一 trace 中多个完整 item endpoint 都纳入；新 cohort 只保留真正对应一次 k→k+1 progress commit 的 endpoint。若“token 选准”会产生紧致 counter，我们应同时看到 BA 与 SNR 上升；实际只看到 BA 小幅上升、SNR 不升。</div><div class="conclusion"><span class="label">本实验结论</span>新的 causal token-site 让“测的是哪个 transition”更干净，并略微提高 held-out decodability；但它没有把 running representation 压缩成更低噪声的单一 counter manifold。</div></section>
<section id="ablation-design"><h2>Experiment B1 · Targeted-retrieval ablation：选中的 heads 是否比同层随机 heads 更必要？</h2><div class="purpose"><span class="label">实验目的</span>把“某些 heads 在 query 上看向目标 needle”升级为行为因果检验：持续关闭同一批 heads 后，模型是否更常无法自由生成注册的 next city。</div><p>Qwen 的 ranking 只使用 discovery 中 <code>adjacent_rank_before_city</code> 的 <code>post_marker</code> source writes，覆盖 15 个 discovery seeds；Gemma 使用 <code>same_unit_rank_before_city</code> 的 <code>p0_item_end</code> source writes，覆盖 19 个 discovery seeds。冻结 bank 随后跨注册 grammar 复用，confirmation seeds 从未参与 head 排名。</p><p>“pre-O slice 清零”是把每层每个入选 head 的 attention output <code>Σ_t A(q,t)V(t)</code> 在进入该层 output projection W_O 之前设为零；不是删除整层、不是改 logits，也不是只在一个 token 做瞬时扰动。干预从 routed query 持续到所有后续 decode forwards，以防模型先偏离、再逃出干预窗口后自我修正。</p><div class="example"><span class="label">简单例子</span>若选中 bank 在 L17 有 3 个 heads、L29 有 5 个 heads，则每个 random bank 也必须在 L17 随机取 3 个、L29 随机取 5 个未入选 heads。这样 selected−random 主要比较“head identity”，而不是层位置或清零数量。</div><div class="conclusion"><span class="label">本实验设定结论</span>Head ranking 与 ablation 起点完全同位，干预持续且 random bank 逐层匹配；因此主 contrast 可以解释为冻结 head identity 的 targeted necessity，而不是 generic model damage。</div></section>
<section id="ablation-results"><h2>Experiment B2 · Raw arms 与 dose response：效应有多大、需要多少 heads？</h2><div class="purpose"><span class="label">实验目的</span>先用 raw arms 排除 generic damage，再用嵌套 K-grid 判断必要回路的有效宽度与旁路程度。</div><figure>{raw_arm_svg(causal)}<figcaption>图 4 · Primary-K registered-confirmation raw failure rates。横轴是 clean、ranked selected bank 和三个逐层匹配 random banks 的均值；纵轴是首次语义 city 未等于注册 next needle 的比例。Ranked 明显高于 random 才能说明 selection specificity；clean 显示任务本身在这些 anchors 上的基线失败。</figcaption></figure>{table(['Model','K','Arm','Failure','95% CI','Anchors'],raw_rows)}<p>Qwen K125 的 clean/ranked/random failure 为 0.0%/{pct(causal['Qwen3-8B']['raw']['selected_bank']['mean'])}/{pct(causal['Qwen3-8B']['raw']['layer_matched_random_mean']['mean'])}；Gemma K8 为 {pct(causal['Gemma4-E4B']['raw']['clean']['mean'])}/{pct(causal['Gemma4-E4B']['raw']['selected_bank']['mean'])}/{pct(causal['Gemma4-E4B']['raw']['layer_matched_random_mean']['mean'])}。两模型的 ranked damage 均远高于逐层匹配 random。</p><figure>{dose_svg(causal)}<figcaption>图 5 · Registered-confirmation dose response。每个 panel 的横轴是 discovery-frozen、有序且严格嵌套的 K；纵轴是 seed-equal <code>selected failure − mean(random failure)</code>，竖线为 95% seed-cluster bootstrap CI。Qwen 与 Gemma 的 K 网格不同，只比较各自回路宽度，不把 K 数值当成跨模型同一容量单位。</figcaption></figure>{table(['Model','K','Selected−random failure','95% CI','Holm p','Seeds','Anchors'],dose_rows)}<p>Qwen 从 K32 的 {pct(causal['Qwen3-8B']['dose'][0]['mean'])} 逐步升到 K96 的 {pct(causal['Qwen3-8B']['dose'][3]['mean'])}，在 K112/K125 跃升为 {pct(causal['Qwen3-8B']['dose'][4]['mean'])}/{pct(causal['Qwen3-8B']['dose'][5]['mean'])}。Gemma K1 已有 {pct(causal['Gemma4-E4B']['dose'][0]['mean'])}，K4/K6 约 {pct(causal['Gemma4-E4B']['dose'][2]['mean'])}/{pct(causal['Gemma4-E4B']['dose'][3]['mean'])}，K8 达 {pct(causal['Gemma4-E4B']['dose'][4]['mean'])}。Gemma 曲线并非严格单调（K2 低于 K1），因此应读作嵌套 bank 的整体剂量趋势，而非每新增一个 head 都独立贡献正效应。</p><div class="example"><span class="label">简单例子</span>Qwen 的 K64 bank 是排名前 64 个 heads，K96 在完整保留这 64 个的基础上再加入第 65–96 个；因此从 K96 到 K112 的效应跃升说明“仅关闭最尖锐的一小撮 heads 仍有旁路”，而不是第 112 个 head 单独造成了全部失败。</div>{table(['Model','Scope','Effect','95% CI','Seeds','Anchors'],scope_rows)}<p>Primary-K confirmation 与 full-panel/discovery coverage 同方向，clean-correct sensitivity 也保持：Qwen {pct(causal['Qwen3-8B']['clean_correct']['mean'])}，Gemma {pct(causal['Gemma4-E4B']['clean_correct']['mean'])}。这降低了结果完全由 clean 本身失败所驱动的可能性。</p><div class="conclusion"><span class="label">本实验结论</span>两个模型都存在具有强 selection specificity 的 targeted-retrieval bank。Gemma 的有效 bank 紧凑（K8）；Qwen 的回路更宽并出现明显阈值/旁路特征（约 K112 以后进入高失败区）。</div></section>
<section id="transfer"><h2>Experiment B3 · 跨 grammar 与 count：统一 bank 是否只在一种 trace 写法上有效？</h2><div class="purpose"><span class="label">实验目的</span>检验 discovery ranking 得到的同一 bank 能否迁移到不同 surface grammar 与 N=2…10，而不是只记住 ranking grammar 的局部格式。</div>{table(['Model','Grammar / summary','Effect','95% CI','Seeds','Anchors','Status'],grammar_rows)}<p>Qwen equal-primary-grammar macro 为 {pct(q_macro['mean'])}，Gemma 为 {pct(g_macro['mean'])}。Qwen 的 adjacent rank-after-city 在 10 seeds/45 anchors 上形成确认性效应；Gemma 的 same-unit rank-before-city 在 10 seeds/58 anchors 上形成确认性效应。其他 grammar 因 supporting seeds 或 anchors 太少，只保留描述，不升级为稳定迁移结论。</p>{table(['Gold count N','Qwen primary-K effect','Gemma primary-K effect','Qwen / Gemma anchors'],count_rows)}<div class="example"><span class="label">简单例子</span>Bank 可以在 <code>(Record 2: Riga, 60)</code> 这类 same-unit rank-before trace 上被选出，却在 <code>Riga, 95. Fifth.</code> 这类 rank-after trace 上继续造成失败；这比在同一种格式内重复测试更能支持“共享 retrieval module”。但只有一个 bullet anchor 时，即使 effect 是 0% 或 100% 都不能代表该 grammar 的总体机制。</div><div class="conclusion"><span class="label">本实验结论</span>主效应不是单一 prompt 格式造成：统一 bank 在至少一个非 selection 的主 grammar 上保持方向一致且有确认性支持，并覆盖 N=2…10。稀有 grammar 的证据量仍不足。</div></section>
<section id="synthesis"><h2>机制拼接：当前最窄、可证伪的 Native-thinking 模型</h2><div class="mechanism-line"><div class="mechanism-node"><strong>1 · Commit progress</strong><br><span class="small">p0_item_end 上携带可解码 running index</span></div><div class="mechanism-arrow">→</div><div class="mechanism-node"><strong>2 · Retrieve next needle</strong><br><span class="small">grammar-routed head bank 从 prompt record 取回 next city</span></div><div class="mechanism-arrow">→</div><div class="mechanism-node"><strong>3 · Form final answer state</strong><br><span class="small">answer_query_v3 上形成 count-specific state</span></div></div><p>Representation 支持第 1 与第 3 个节点的状态存在；ablation 支持第 2 个节点的 bank-level necessity。Qwen 的 final state 更紧致、retrieval bank 更宽；Gemma 的 running state 更易解码、retrieval bank 更紧凑。这说明两个模型可能实现同一功能分解，但没有必要使用同样宽度或同样几何形态的回路。</p><div class="boundary"><span class="label">还不能声称</span>尚未通过 restoration/mediation 证明 p0 progress state 的信息必须经由当前 retrieval bank 传递到 final answer state；也没有证明 selected bank 是唯一通路、每个入选 head 单独必要，或负 SNR 意味着不存在 counter。</div><div class="conclusion"><span class="label">当前总论</span>最稳健的表述是：Native-thinking 在 item commits 与 final answer query 上形成可解码的计数相关状态，并使用 grammar-routed attention-head bank 执行下一条 needle 的 targeted retrieval。Qwen 的必要回路宽而具阈值，Gemma 的回路紧凑；完整的 representation→retrieval→answer 中介链仍是下一阶段问题。</div></section>
<section id="audit"><h2>复现账本与详细报告</h2><div class="links"><a href="NiaH_Geometry_Comparison.html">Full representation geometry</a><a href="NiaH_Native-Thinking_Causal_Ablation_report.html">Full causal ablation</a><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 head map / attention atlas</a><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><a href="v5_native_targeted_retrieval/Qwen3-8B/targeted_retrieval_report.html">Qwen model report</a><a href="v5_native_targeted_retrieval/Gemma4-E4B/targeted_retrieval_report.html">Gemma model report</a></div><p>本页是 synthesis，不替代各份细节报告。Representation 图表只读取 discovery-frozen selected rows；causal 图表只读取 complete registered analysis manifests 与 paired estimands；P0 atlas 则单独展示 grammar-specific discovery attention ranking 与显著单头的逐 needle attention。样本资格审计直接读取完整 trajectory registry，并对 Qwen N≥2 duplicate rows 回查原始 generation 的 <code>rendered_prompt</code>。</p><details><summary>输入文件 SHA256</summary><p class="audit">Generated UTC: {esc(generated)}<br>{ledger}<br>Report schema: realistic_niah_v5_native_thinking_synthesis_v3</p></details><div class="conclusion"><span class="label">审计结论</span>两个 causal manifests 均为 complete，K-grid、anchor registry、五臂完整性与 selection contract 已通过；duplicate 数量、原文 multiplicity 与最终答案对错由 builder 断言，若底层数据改变将拒绝生成报告。</div></section>
</main></div></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-geometry-root", type=Path, default=Path("reports/v5_native_causal_aligned_geometry"))
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--causal-root", type=Path, default=Path("reports/v5_native_targeted_retrieval"))
    parser.add_argument("--qwen-selection", type=Path, default=Path("configs/realistic_niah_v5_qwen_shared_k125_full300_selection.json"))
    parser.add_argument("--gemma-selection", type=Path, default=Path("configs/realistic_niah_v5_gemma_shared_k8_full300_selection.json"))
    parser.add_argument("--parser-report", type=Path, default=Path("reports/NiaH_Native-Thinking_Parser_and_Token_Sites.html"))
    parser.add_argument("--geometry-report", type=Path, default=Path("reports/NiaH_Geometry_Comparison.html"))
    parser.add_argument("--causal-report", type=Path, default=Path("reports/NiaH_Native-Thinking_Causal_Ablation_report.html"))
    parser.add_argument("--p0-atlas", type=Path, default=Path("reports/NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html"))
    parser.add_argument("--trajectory-registry", type=Path, default=Path("reports/v5_native_causal_site_review/trajectory_registry.jsonl"))
    parser.add_argument("--qwen-generations", type=Path, default=Path("work/remote_native_traces_68_209_74_38/Qwen3-8B/generations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/NiaH_Native-Thinking_report.html"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/v5_native_thinking_mechanism_20260813/manifest.json"))
    parser.add_argument("--report-root", help=argparse.SUPPRESS)
    parser.add_argument("--analysis-root", help=argparse.SUPPRESS)
    parser.add_argument("--geometry-root", help=argparse.SUPPRESS)
    parser.add_argument("--reference-report", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection_paths = {"Qwen3-8B": args.qwen_selection, "Gemma4-E4B": args.gemma_selection}
    rep = load_representation(args.causal_geometry_root, args.dual_endpoint_root)
    causal = load_causal(args.causal_root, selection_paths)
    eligibility = load_eligibility_audit(args.trajectory_registry, args.qwen_generations)
    input_paths = [
        args.causal_geometry_root / "audit.json",
        args.causal_geometry_root / "site_selected.csv",
        args.causal_geometry_root / "legacy_vs_causal_item_end.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        args.causal_root / "Qwen3-8B" / "analysis_manifest.json",
        args.causal_root / "Qwen3-8B" / "estimands.csv",
        args.causal_root / "Qwen3-8B" / "raw_arm_rates.csv",
        args.causal_root / "Qwen3-8B" / "count_estimands.csv",
        args.causal_root / "Gemma4-E4B" / "analysis_manifest.json",
        args.causal_root / "Gemma4-E4B" / "estimands.csv",
        args.causal_root / "Gemma4-E4B" / "raw_arm_rates.csv",
        args.causal_root / "Gemma4-E4B" / "count_estimands.csv",
        args.qwen_selection,
        args.gemma_selection,
        args.parser_report,
        args.geometry_report,
        args.causal_report,
        args.p0_atlas,
        args.trajectory_registry,
        args.qwen_generations,
    ]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    generated = datetime.now(timezone.utc).isoformat()
    report = build_report(rep, causal, eligibility, generated=generated, input_hashes=input_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "realistic_niah_v5_native_thinking_synthesis_v3",
        "status": "complete",
        "generated_at": generated,
        "scope": ["representation", "targeted_retrieval_ablation"],
        "excluded_from_report": "experiments without frozen confirmatory evidence",
        "models": list(MODELS),
        "registered_confirmation_seeds": list(range(1254, 1264)),
        "eligibility_audit": {
            "qwen_duplicate_traces_total": eligibility["qwen_duplicate_total"],
            "qwen_duplicate_traces_n1": eligibility["qwen_duplicate_n1"],
            "qwen_extra_duplicate_exclusions_n_ge_2": eligibility["qwen_duplicate_extra"],
            "qwen_duplicate_final_wrong": eligibility["qwen_wrong"],
            "qwen_duplicate_final_correct": eligibility["qwen_correct"],
            "gemma_duplicate_traces_total": eligibility["gemma_duplicate_total"],
            "gemma_duplicates_all_n1": True,
            "gemma_extra_duplicate_exclusions_n_ge_2": 0,
        },
        "input_sha256": input_hashes,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "sha256": manifest["output_sha256"], "manifest": str(args.manifest.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # The concise Non-thinking-ordered builder supersedes this historical
    # synthesis while this module remains importable for helper/test
    # compatibility.
    from build_v5_native_thinking_report_restructured import main as final_main

    raise SystemExit(final_main())
