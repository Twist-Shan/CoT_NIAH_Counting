#!/usr/bin/env python3
"""Add trace proportions, paired clarity evidence, and Qwen band diagnostics.

This is an idempotent augmentation step for an already-generated, self-contained
``NiaH_Geometry_Comparison.html``.  It is intentionally separate from the heavy
geometry builder so that descriptive parser/band audits can be refreshed without
reloading every hidden-state tensor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS = ("Qwen3-8B", "Gemma4-E4B")
TRACE_CATEGORIES = (
    "one_to_one",
    "full_coverage_with_duplicates",
    "partial_with_duplicates",
    "partial_unique",
    "synthetic_unverified",
)
CAPTURE_MARKERS = (
    "indexed",
    "ordinal",
    "bullet",
    "audit_sentence",
    "completion_recap",
)
FULL_LEGACY_MARKERS = (*CAPTURE_MARKERS, "unresolved")
HYBRID_MARKERS = (
    "inline_count",
    "indexed",
    "ordinal",
    "bullet",
    "audit_sentence",
    "completion_recap",
    "evidence_sequence",
)
SPLITS = ("all", "discovery", "confirmation")
BEGIN = "<!-- BEGIN geometry-diagnostic-augmentation-v1 -->"
END = "<!-- END geometry-diagnostic-augmentation-v1 -->"
STYLE_BEGIN = "/* BEGIN geometry-diagnostic-augmentation-v1 */"
STYLE_END = "/* END geometry-diagnostic-augmentation-v1 */"
PALETTE = {
    "one_to_one": "#00A88F",
    "full_coverage_with_duplicates": "#D6B52C",
    "partial_with_duplicates": "#E76F51",
    "partial_unique": "#6750E8",
    "synthetic_unverified": "#8E5DB7",
}
MARKER_PALETTE = {
    "completion_recap": "#6750E8",
    "indexed": "#00A88F",
    "ordinal": "#D6B52C",
    "audit_sentence": "#D94B86",
    "bullet": "#00A9D8",
    "evidence_sequence": "#E76F51",
    "inline_count": "#A7C957",
    "unresolved": "#8A838E",
}
COUNT_PALETTE = (
    "#6750E8",
    "#00A9D8",
    "#00A88F",
    "#2DBE77",
    "#A7C957",
    "#D6B52C",
    "#F29E4C",
    "#E76F51",
    "#D94B86",
    "#8E5DB7",
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def trace_category_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODELS:
        model_rows = [row for row in rows if str(row.get("model_label")) == model]
        if len(model_rows) != 300:
            raise ValueError(f"{model}: expected 300 parser-audit rows, got {len(model_rows)}")
        for split in SPLITS:
            selected = (
                model_rows
                if split == "all"
                else [row for row in model_rows if str(row.get("split")) == split]
            )
            expected = 300 if split == "all" else (200 if split == "discovery" else 100)
            if len(selected) != expected:
                raise ValueError(
                    f"{model}/{split}: expected {expected} rows, got {len(selected)}"
                )
            counts = Counter(str(row.get("trace_category")) for row in selected)
            unknown = sorted(set(counts) - set(TRACE_CATEGORIES))
            if unknown:
                raise ValueError(f"{model}/{split}: unknown categories {unknown}")
            result[model][split] = {
                "total": len(selected),
                "counts": {category: int(counts.get(category, 0)) for category in TRACE_CATEGORIES},
            }
    return result


def marker_summary(
    rows_by_model: Mapping[str, list[dict[str, Any]]],
    *,
    marker_kinds: tuple[str, ...],
    expected_totals: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODELS:
        model_rows = list(rows_by_model[model])
        if len(model_rows) != int(expected_totals["all"]):
            raise ValueError(
                f"{model}: expected {expected_totals['all']} marker rows, "
                f"got {len(model_rows)}"
            )
        for split in SPLITS:
            selected = (
                model_rows
                if split == "all"
                else [row for row in model_rows if str(row.get("split")) == split]
            )
            expected = int(expected_totals[split])
            if len(selected) != expected:
                raise ValueError(
                    f"{model}/{split}: expected {expected} marker rows, "
                    f"got {len(selected)}"
                )
            counts = Counter(str(row.get("marker_kind")) for row in selected)
            unknown = sorted(set(counts) - set(marker_kinds))
            if unknown:
                raise ValueError(f"{model}/{split}: unknown markers {unknown}")
            result[model][split] = {
                "total": len(selected),
                "counts": {
                    marker: int(counts.get(marker, 0)) for marker in marker_kinds
                },
            }
    return result


def capture_marker_summary(
    native_capture_root: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[Path]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    inputs = []
    for model in MODELS:
        path = native_capture_root / model / "capture_index.jsonl"
        rows_by_model[model] = read_jsonl(path)
        inputs.append(path)
    return (
        marker_summary(
            rows_by_model,
            marker_kinds=CAPTURE_MARKERS,
            expected_totals={"all": 30, "discovery": 20, "confirmation": 10},
        ),
        inputs,
    )


def hybrid_marker_summary(
    parser_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    rows_by_model = {
        model: [row for row in parser_rows if str(row.get("model_label")) == model]
        for model in MODELS
    }
    return marker_summary(
        rows_by_model,
        marker_kinds=HYBRID_MARKERS,
        expected_totals={"all": 300, "discovery": 200, "confirmation": 100},
    )


def legacy_compatible_marker_summary(
    parser_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Summarize the full panel using the hybrid audit's old-parser label."""

    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parser_rows:
        marker = row.get("old_marker_kind")
        rows_by_model[str(row["model_label"])].append(
            {
                "split": row["split"],
                "marker_kind": "unresolved" if marker is None else str(marker),
            }
        )
    return marker_summary(
        rows_by_model,
        marker_kinds=FULL_LEGACY_MARKERS,
        expected_totals={"all": 300, "discovery": 200, "confirmation": 100},
    )


def _reasoning_text(raw_output_text: str) -> str:
    start = raw_output_text.find("<think>")
    stop = raw_output_text.rfind("</think>")
    if start >= 0 and stop > start:
        value = raw_output_text[start + len("<think>") : stop]
    else:
        value = raw_output_text
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def unresolved_trace_examples(
    parser_rows: list[dict[str, Any]], native_trace_root: Path | None
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Audit old-taxonomy misses and optionally attach their raw reasoning."""

    unresolved = [row for row in parser_rows if row.get("old_marker_kind") is None]
    records: dict[str, dict[str, Any]] = {}
    inputs: list[Path] = []
    if native_trace_root is not None:
        ids_by_model: dict[str, set[str]] = defaultdict(set)
        for row in unresolved:
            ids_by_model[str(row["model_label"])].add(str(row["request_id"]))
        for model, request_ids in sorted(ids_by_model.items()):
            path = native_trace_root / model / "generations.jsonl"
            inputs.append(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    request_id = str(record.get("request_id"))
                    if request_id in request_ids:
                        records[request_id] = record
            missing = sorted(request_ids - set(records))
            if missing:
                raise ValueError(f"unresolved raw traces missing from {path}: {missing}")

    result = []
    for row in sorted(
        unresolved,
        key=lambda value: (
            str(value["model_label"]),
            int(value["seed"]),
            int(value["gold_count"]),
        ),
    ):
        request_id = str(row["request_id"])
        record = records.get(request_id)
        old_parser = (
            record.get("trace_parse", {}).get("parser", {}) if record else {}
        )
        marker = str(row["marker_kind"])
        if marker == "inline_count":
            surface_description = (
                "段落散文；city 后的 one/second/third 等 rank/count 词嵌在句内，"
                "没有独立行首列表。"
            )
        elif marker == "evidence_sequence":
            surface_description = (
                "段落散文；没有可接受的连续 1…M span，只能按 city+score "
                "evidence 的出现顺序建立审计序列。"
            )
        else:
            surface_description = "旧五类未命中；current hybrid parser 使用其他证据。"
        selected = int(row["item_count"])
        gold = int(row["gold_count"])
        if str(row["trace_category"]) == "synthetic_unverified":
            hybrid_description = (
                f"evidence_sequence：{selected} 个 score-supported items；人工赋予"
                "顺序标签 1…M，并明确标为 synthetic_unverified。"
            )
        else:
            hybrid_description = (
                f"{marker}：选择 {selected}/{gold} 个有 local rank evidence 的 items；"
                f"trace_category={row['trace_category']}。"
            )
            if selected < gold:
                hybrid_description += " 后续无 rank 的城市不会用最终 Total 补齐。"
        raw = str(record.get("raw_output_text", "")) if record else ""
        result.append(
            {
                "request_id": request_id,
                "model_label": str(row["model_label"]),
                "seed": int(row["seed"]),
                "split": str(row["split"]),
                "gold_count": gold,
                "final_parsed_count": row.get("final_parsed_count"),
                "old_status": str(
                    old_parser.get("status", "raw trace not supplied")
                ),
                "old_candidates": old_parser.get("candidates_considered"),
                "surface_description": surface_description,
                "hybrid_description": hybrid_description,
                "hybrid_marker": marker,
                "hybrid_item_count": selected,
                "trace_category": str(row["trace_category"]),
                "reasoning_text": _reasoning_text(raw) if raw else None,
            }
        )
    return result, inputs


def marker_definitions_html() -> str:
    """Operational definitions of the five capture-time marker labels."""

    definitions = (
        (
            "indexed",
            "显式阿拉伯数字列表",
            "item 在逻辑行首以 1. / 1) 开始；序列必须从 1 起严格递增，每项恰含一个注册 gold city，并在换行或保守句号边界结束。",
            "1. Paris  /  2) Lima",
        ),
        (
            "ordinal",
            "英文序数列表",
            "item 在行首以 First/Secondly/… 开始并严格递增；Then 只可作为第 2 项，Finally 作为当前期望的末项。其余 city 与边界约束同 indexed。",
            "First, Paris  /  Second: Lima",
        ),
        (
            "bullet",
            "无编号项目符号列表",
            "item 在行首以 -, • 或非粗体 * 开始；每项恰含一个注册 gold city 并有结束边界。它表示逐项结构，但 marker 本身不携带 k。",
            "- Paris  /  • Lima",
        ),
        (
            "audit_sentence",
            "固定审计句式链",
            "没有逐项序号；仅匹配 V4.4 的窄模板 “In the 2024 city score audit, CITY received a score of NUMBER.”，以连续匹配到的句子作为 items。",
            "In the 2024 city score audit, Paris received a score of 87.",
        ),
        (
            "completion_recap",
            "末段紧凑复述",
            "没有逐项 marker。只在显式列表未命中后，寻找 reasoning 后 55% 中带 count/recount/tally/there are N cities 等强 cue 的紧凑 city recap；cue 可在同句、前一句，或由下一句的匹配总数确认。",
            "Let me count again: Paris, Lima, Oslo. There are three cities.",
        ),
    )
    cards = []
    for marker, title, definition, example in definitions:
        cards.append(
            f'<div><h3><code>{esc(marker)}</code> · {esc(title)}</h3>'
            f'<p>{esc(definition)}</p><p><strong>canonical example:</strong> '
            f'<code>{esc(example)}</code></p></div>'
        )
    return (
        '<details open><summary>五类 marker 的 operational definition</summary>'
        '<div class="callout"><strong>标签选择原则：</strong>这五类描述的是 parser '
        '最终选中的表面结构，不是互斥的语言学本体。parser 先找第一个可终止的显式 '
        'indexed/ordinal/bullet span；只有显式 span 未命中时才尝试 completion recap 与 '
        'exact audit-sentence fallback。</div>'
        f'<div class="definitions">{"".join(cards)}</div></details>'
    )


def unresolved_examples_html(examples: list[dict[str, Any]]) -> str:
    rows = []
    raw_details = []
    for value in examples:
        candidates = value["old_candidates"]
        old_result = esc(value["old_status"])
        if candidates is not None:
            old_result += f"<br><span class=\"muted\">candidates={int(candidates)}</span>"
        final = value["final_parsed_count"]
        final_text = (
            f"{int(final)} / {int(value['gold_count'])} · correct"
            if final is not None and int(final) == int(value["gold_count"])
            else f"{esc(final)} / {int(value['gold_count'])}"
        )
        rows.append(
            (
                f"<code>seed{int(value['seed'])} · N={int(value['gold_count'])}</code>",
                esc(value["split"]),
                old_result,
                esc(value["surface_description"]),
                esc(value["hybrid_description"]),
                final_text,
            )
        )
        if value["reasoning_text"] is not None:
            raw_details.append(
                f'<details><summary>seed{int(value["seed"])} · N={int(value["gold_count"])} '
                f'· current <code>{esc(value["hybrid_marker"])}</code></summary>'
                f'<pre class="trace">{esc(value["reasoning_text"])}</pre></details>'
            )
    raw_block = (
        '<h4>五条原始 reasoning（默认折叠）</h4>' + "".join(raw_details)
        if raw_details
        else '<p class="small">本次构建未提供 raw native-trace root，因此未嵌入原文。</p>'
    )
    return f"""
<details open><summary><code>unresolved</code> 到底是什么：{len(examples)} 条旧 taxonomy miss</summary>
<p class="small"><code>unresolved</code> 只表示 capture-time parser 没有给出上述五类之一；它不表示模型答错，也不表示 trace 中没有计数证据。这里五条均为 Qwen 段落式 reasoning，旧 parser 均未形成可终止候选。</p>
{table(['trajectory','split','旧 parser','原始表面形态','current hybrid 处理','最终 Total'], rows)}
<div class="callout warning"><strong>最重要的审计边界：</strong><code>evidence_sequence</code> 的 1…M 是 parser 按 score-supported city evidence 的出现顺序赋予的审计索引，不是模型表面写出的 marker；因此这三条保持 <code>synthetic_unverified</code>。另外 N=6/seed1256 虽最终答对 6，hybrid 只保留有 local rank evidence 的前 2 项，不以最终答案补齐其余 4 项。</div>
{raw_block}</details>"""


def normalized_mutual_information(counts: Mapping[str, Mapping[str, int]]) -> float:
    """Arithmetic-mean normalized mutual information for a contingency table."""

    row_totals = {
        row: float(sum(int(value) for value in columns.values()))
        for row, columns in counts.items()
    }
    column_names = sorted(
        {column for columns in counts.values() for column in columns}
    )
    column_totals = {
        column: float(
            sum(int(columns.get(column, 0)) for columns in counts.values())
        )
        for column in column_names
    }
    total = float(sum(row_totals.values()))
    if total <= 0:
        return float("nan")
    mutual_information = 0.0
    for row, columns in counts.items():
        for column in column_names:
            value = float(columns.get(column, 0))
            if value > 0:
                mutual_information += (value / total) * math.log(
                    value * total / (row_totals[row] * column_totals[column])
                )
    row_entropy = -sum(
        (value / total) * math.log(value / total)
        for value in row_totals.values()
        if value > 0
    )
    column_entropy = -sum(
        (value / total) * math.log(value / total)
        for value in column_totals.values()
        if value > 0
    )
    denominator = row_entropy + column_entropy
    return 0.0 if denominator <= 0 else 2.0 * mutual_information / denominator


def fisher_exact_two_sided(table_2x2: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """Two-sided Fisher exact p-value using fixed margins."""

    (a, b), (c, d) = table_2x2
    row_one = a + b
    row_two = c + d
    column_one = a + c
    total = row_one + row_two
    denominator = math.comb(total, row_one)

    def probability(value: int) -> float:
        return (
            math.comb(column_one, value)
            * math.comb(total - column_one, row_one - value)
            / denominator
        )

    observed = probability(a)
    lower = max(0, row_one - (total - column_one))
    upper = min(row_one, column_one)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(lower, upper + 1)
            if probability(value) <= observed + 1e-15
        ),
    )


def qwen_band_marker_analysis(
    point_rows: list[dict[str, str]], parser_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare capture-time and hybrid-parser marker labels with Qwen bands."""

    parser_by_request = {
        str(row["request_id"]): row
        for row in parser_rows
        if str(row.get("model_label")) == "Qwen3-8B"
    }
    missing = sorted(
        {str(row["request_id"]) for row in point_rows} - set(parser_by_request)
    )
    if missing:
        raise ValueError(f"Qwen band points missing parser rows: {missing}")

    legacy_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    hybrid_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_request: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in point_rows:
        request_id = str(row["request_id"])
        band = str(row["band"])
        family = (
            "completion_recap"
            if str(row["marker_kind"]) == "completion_recap"
            else "other_marker"
        )
        hybrid = str(parser_by_request[request_id]["marker_kind"])
        legacy_family_counts[family][band] += 1
        hybrid_counts[hybrid][band] += 1
        by_request[request_id].append(row)

    trajectory_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trajectory_rows = []
    for request_id, rows in sorted(by_request.items()):
        band_counts = Counter(str(row["band"]) for row in rows)
        majority_band, majority_count = max(
            band_counts.items(), key=lambda item: (item[1], item[0])
        )
        legacy_marker = str(rows[0]["marker_kind"])
        family = (
            "completion_recap"
            if legacy_marker == "completion_recap"
            else "other_marker"
        )
        trajectory_counts[family][majority_band] += 1
        trajectory_rows.append(
            {
                "request_id": request_id,
                "seed": int(rows[0]["seed"]),
                "legacy_marker": legacy_marker,
                "hybrid_marker": str(parser_by_request[request_id]["marker_kind"]),
                "majority_band": majority_band,
                "majority_fraction": majority_count / len(rows),
            }
        )

    fisher_table = (
        (
            int(trajectory_counts["completion_recap"]["lower"]),
            int(trajectory_counts["completion_recap"]["upper"]),
        ),
        (
            int(trajectory_counts["other_marker"]["lower"]),
            int(trajectory_counts["other_marker"]["upper"]),
        ),
    )
    hybrid_plain = {
        marker: {band: int(values.get(band, 0)) for band in ("lower", "upper")}
        for marker, values in sorted(hybrid_counts.items())
    }
    total_states = len(point_rows)
    hybrid_purity = (
        sum(max(values.values()) for values in hybrid_plain.values()) / total_states
    )
    return {
        "legacy_family_counts": {
            family: {
                band: int(legacy_family_counts[family].get(band, 0))
                for band in ("lower", "upper")
            }
            for family in ("completion_recap", "other_marker")
        },
        "trajectory_counts": {
            family: {
                band: int(trajectory_counts[family].get(band, 0))
                for band in ("lower", "upper")
            }
            for family in ("completion_recap", "other_marker")
        },
        "trajectory_fisher_two_sided_p": fisher_exact_two_sided(fisher_table),
        "hybrid_counts": hybrid_plain,
        "hybrid_nmi": normalized_mutual_information(hybrid_plain),
        "hybrid_weighted_band_purity": hybrid_purity,
        "trajectory_rows": trajectory_rows,
    }


def category_section(
    capture_markers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    full_legacy_markers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    hybrid_markers: Mapping[str, Mapping[str, Mapping[str, Any]]],
    trace_categories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    unresolved_examples: list[dict[str, Any]],
    *,
    legacy_geometry_capture: bool,
) -> str:
    bars = []
    for model in MODELS:
        payload = full_legacy_markers[model]["all"]
        total = int(payload["total"])
        segments = []
        for marker in FULL_LEGACY_MARKERS:
            count = int(payload["counts"][marker])
            percentage = 100.0 * count / total
            segments.append(
                f'<span class="cat-segment" style="width:{percentage:.6f}%;background:{MARKER_PALETTE[marker]}" '
                f'title="{esc(marker)}: {count}/{total} ({percentage:.1f}%)"></span>'
            )
        bars.append(
            f'<div class="cat-bar-row"><strong>{esc(model)}</strong><div class="cat-bar">{"".join(segments)}</div></div>'
        )
    legend = "".join(
        f'<span><i style="background:{MARKER_PALETTE[marker]}"></i><code>{esc(marker)}</code></span>'
        for marker in FULL_LEGACY_MARKERS
    )
    full_legacy_rows = []
    for model in MODELS:
        for split in SPLITS:
            payload = full_legacy_markers[model][split]
            total = int(payload["total"])
            values = []
            for marker in FULL_LEGACY_MARKERS:
                count = int(payload["counts"][marker])
                values.append(
                    f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>"
                )
            full_legacy_rows.append((esc(model), esc(split), str(total), *values))
    capture_rows = []
    for model in MODELS:
        for split in SPLITS:
            payload = capture_markers[model][split]
            total = int(payload["total"])
            values = []
            for marker in CAPTURE_MARKERS:
                count = int(payload["counts"][marker])
                values.append(f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>")
            capture_rows.append((esc(model), esc(split), str(total), *values))
    hybrid_rows = []
    for model in MODELS:
        for split in SPLITS:
            payload = hybrid_markers[model][split]
            total = int(payload["total"])
            values = []
            for marker in HYBRID_MARKERS:
                count = int(payload["counts"][marker])
                values.append(
                    f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>"
                )
            hybrid_rows.append((esc(model), esc(split), str(total), *values))
    category_rows = []
    for model in MODELS:
        for split in SPLITS:
            payload = trace_categories[model][split]
            total = int(payload["total"])
            values = []
            for category in TRACE_CATEGORIES:
                count = int(payload["counts"][category])
                values.append(
                    f"{count} <span class=\"muted\">({100*count/total:.1f}%)</span>"
                )
            category_rows.append((esc(model), esc(split), str(total), *values))
    status = (
        '<div class="callout warning"><strong>当前数据状态：</strong>主比例覆盖完整 300 条/模型，'
        '使用 hybrid audit 保留的 <code>old_marker_kind</code> 兼容标签来回答“五类 marker 各占多少”；'
        'Qwen 有 5 条旧 parser 未赋类，显式列为 <code>unresolved</code>，不从分母中删除。当前 '
        'hidden-state 点云实际使用的 N=10 capture composition 另列在下方，因为它与完整 300 的'
        '格式混合不同。</div>'
        if legacy_geometry_capture
        else '<div class="callout"><strong>当前数据状态：</strong>主 hidden-state 分析已使用'
        '全注册 panel；旧五类 capture taxonomy 仍保留作为与现有图的兼容审计。</div>'
    )
    return f"""
<section id="trace-proportions"><h2>Trace marker 的构成比例</h2>
{status}
<h3>完整 300 trajectories：五类 legacy-compatible marker</h3>
<p>比例按 trajectory 计算，不按 state 数计算；分母固定为每模型 300 条（discovery 200，confirmation 100）。灰色 <code>unresolved</code> 不是第六种 marker，而是旧 parser 没有赋予五类之一的审计状态。</p>
<div class="cat-bars">{''.join(bars)}</div><div class="cat-legend">{legend}</div>
{table(['模型','split','轨迹数',*FULL_LEGACY_MARKERS], full_legacy_rows)}
<div class="callout"><strong>完整 panel：</strong>Qwen 为 audit_sentence 33.7%、indexed 30.7%、completion_recap 26.0%、ordinal 7.0%、bullet 1.0%，另有 unresolved 1.7%；Gemma 为 bullet 75.7%、audit_sentence 15.0%、indexed 9.0%、completion_recap 0.3%、ordinal 0。</div>
{marker_definitions_html()}
{unresolved_examples_html(unresolved_examples)}
<details open><summary>当前 geometry capture · N=10 的五类 marker composition</summary>
<p class="small">这张表才与本页现有 hidden-state points 一一对应：30 traces/模型（discovery 20，confirmation 10）。完整 300 的比例不能反向当作截图 92 个 Qwen confirmation states 的格式配比。</p>
{table(['模型','split','轨迹数',*CAPTURE_MARKERS], capture_rows)}
<p class="small"><strong>当前点云：</strong>Qwen 为 completion_recap 40.0%、indexed 33.3%、audit_sentence 16.7%、ordinal 6.7%、bullet 3.3%；Gemma 为 bullet 63.3%、indexed 23.3%、audit_sentence 13.3%，其余两类为 0。</p></details>
<details><summary>全 300 trajectories · current hybrid parser marker 比例</summary>
<p class="small">current hybrid parser 把连续的 “Count: k” 事件识别为 <code>inline_count</code>，并为兜底序列增加 <code>evidence_sequence</code>，所以实际是七类 marker。它回答全数据的表面格式构成，但不应反向替换旧 hidden-state capture manifest 的 marker 标签。</p>
{table(['模型','split','轨迹数',*HYBRID_MARKERS], hybrid_rows)}</details>
<details><summary>补充：trace completeness category 比例</summary>
{table(['模型','split','轨迹数',*TRACE_CATEGORIES], category_rows)}</details>
</section>"""


def selected_pairs(dual_root: Path) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        directory = dual_root / model / "pca16_whiten"
        for endpoint, filename in (
            ("running index", "running_index_selected.csv"),
            ("final count", "final_count_selected.csv"),
        ):
            rows = read_csv(directory / filename)
            group = "all_traces" if endpoint == "running index" else "all_counts"
            matches = [row for row in rows if str(row.get("analysis_group")) == group]
            by_mode = {str(row["mode"]): row for row in matches}
            if set(by_mode) != {"non_thinking", "native_thinking"}:
                raise ValueError(f"{model}/{endpoint}: missing selected modes")
            result.append(
                {
                    "model": model,
                    "endpoint": endpoint,
                    "non": by_mode["non_thinking"],
                    "native": by_mode["native_thinking"],
                }
            )
    return result


def percent(row: Mapping[str, Any], field: str) -> float:
    return 100.0 * float(row[field])


def dumbbell_row(label: str, non: float, native: float) -> str:
    low = min(non, native)
    high = max(non, native)
    return f"""<div class="clarity-row"><div class="clarity-name">{esc(label)}</div>
<div class="clarity-track"><span class="clarity-link" style="left:{low:.4f}%;width:{high-low:.4f}%"></span>
<span class="clarity-dot non" style="left:{non:.4f}%" title="non-thinking {non:.1f}%"></span>
<span class="clarity-dot native" style="left:{native:.4f}%" title="native-thinking {native:.1f}%"></span></div>
<div class="clarity-values">{non:.1f}% → {native:.1f}%</div></div>"""


def clarity_section(
    pairs: list[dict[str, Any]], *, legacy_geometry_capture: bool
) -> str:
    charts = []
    snr_rows = []
    qwen_running_snr: tuple[float, float] | None = None
    for pair in pairs:
        non = pair["non"]
        native = pair["native"]
        non_log = percent(non, "confirmation_logistic_balanced_accuracy")
        native_log = percent(native, "confirmation_logistic_balanced_accuracy")
        non_ncc = percent(non, "confirmation_ncc_balanced_accuracy")
        native_ncc = percent(native, "confirmation_ncc_balanced_accuracy")
        non_snr = float(non["confirmation_class_balanced_snr_db"])
        native_snr = float(native["confirmation_class_balanced_snr_db"])
        non_snr_ratio = 10.0 ** (non_snr / 10.0)
        native_snr_ratio = 10.0 ** (native_snr / 10.0)
        if pair["model"] == "Qwen3-8B" and pair["endpoint"] == "running index":
            qwen_running_snr = (non_snr, native_snr)
        probes_agree = native_log > non_log and native_ncc > non_ncc
        snr_agrees = native_snr > non_snr
        if probes_agree and snr_agrees:
            verdict = "Logistic、NCC、SNR 三项同向；可写更可解码且类间/类内比更高。"
        elif probes_agree:
            verdict = "两种 probe 同向，但 SNR 未提高；只能写更可解码，不能笼统写更紧。"
        else:
            verdict = "两种 probe 未同向；不支持总体更清晰的表述。"
        charts.append(
            f"""<article class="clarity-card"><h3>{esc(pair['model'])} · {esc(pair['endpoint'])}</h3>
{dumbbell_row('Logistic BA', non_log, native_log)}
{dumbbell_row('Nearest-centroid BA', non_ncc, native_ncc)}
<p>{esc(verdict)}</p><div class="muted">独立最佳层：NT L{int(float(non['layer']))} / Native L{int(float(native['layer']))}；held-out states {int(float(non['confirmation_rows']))} / {int(float(native['confirmation_rows']))}</div></article>"""
        )
        snr_rows.append(
            (
                esc(pair["model"]),
                esc(pair["endpoint"]),
                f"{non_snr:.2f} dB <span class=\"muted\">(S/N={non_snr_ratio:.3f})</span>",
                f"{native_snr:.2f} dB <span class=\"muted\">(S/N={native_snr_ratio:.3f})</span>",
                f"{native_snr-non_snr:+.2f} dB",
            )
        )
    if qwen_running_snr is None:
        raise ValueError("missing Qwen3-8B running-index SNR row")
    qwen_non_ratio = 10.0 ** (qwen_running_snr[0] / 10.0)
    qwen_native_ratio = 10.0 ** (qwen_running_snr[1] / 10.0)
    scope_warning = (
        '<div class="callout warning"><strong>当前估计边界：</strong>这版 dual '
        'running-index capture 只覆盖 gold N=10 的 30 条轨迹，而不是 10 counts × '
        '30 seeds；Qwen confirmation 的 92 个 states 来自 10 条 N=10 trace。旧版 '
        'final-count 指标也只含 5 个 confirmation seeds（50 states）。因此这些图可以'
        '说明当前 capture 中的现象和更合适的画法，不能替代待补的全 300 native-running '
        'capture。</div>'
        if legacy_geometry_capture
        else '<div class="callout"><strong>估计范围：</strong>dual running-index 与 '
        'final-count 都使用注册的 10 counts × 30 seeds panel；每个 panel 仍以表内实际 '
        'states 与逐类支持为准。</div>'
    )
    return f"""
<section id="clarity-evidence"><h2>如何可视化 “native-thinking 更清晰”</h2>
<p>主图改用 paired held-out evidence：同一 endpoint 内，non-thinking 与 native-thinking 各取自己由 discovery 选出的 token site/layer，再比较冻结后的 confirmation balanced accuracy。圆点位置是绝对 accuracy，连线只表示差值；不把不同模型层号强行对齐。</p>
<div class="clarity-legend"><span><i class="clarity-dot non"></i>non-thinking</span><span><i class="clarity-dot native"></i>native-thinking</span><span>横轴 0–100%</span></div>
<div class="clarity-grid">{''.join(charts)}</div>
<h3>SNR 如何解释</h3>
<div class="definitions two"><div><h3>形式化定义</h3><p>在 discovery 拟合并冻结的 PCA16-whitened 空间中，对 confirmation 的每个 count 类计算 centroid μ<sub>k</sub>。令 class-balanced grand centroid μ̄ = K<sup>−1</sup>Σ<sub>k</sub>μ<sub>k</sub>，则 signal S = K<sup>−1</sup>Σ<sub>k</sub>||μ<sub>k</sub>−μ̄||²；noise N = K<sup>−1</sup>Σ<sub>k</sub>[n<sub>k</sub><sup>−1</sup>Σ<sub>i:yᵢ=k</sub>||zᵢ−μ<sub>k</sub>||²]。报告 SNR = S/N 与 SNR<sub>dB</sub> = 10 log<sub>10</sub>(S/N)。各类先各自平均，因此 late-position 支持较少的类不会被早期类淹没。</p></div><div><h3>直觉与边界</h3><p>0 dB 表示 centroid separation energy 与 within-class scatter 相等；正值表示前者更大；负值表示后者更大。SNR 是全局、各方向等权的紧致度描述，不是分类 accuracy。Logistic 可以利用少数判别方向和各向异性边界，而 marker/template nuisance direction 会抬高类内总散布，因此“accuracy 更高但 SNR 略低”并不矛盾。</p></div></div>
{table(['模型','endpoint','Non-thinking','Native-thinking','Native − Non'], snr_rows)}
<p class="small">Qwen running-index 的 S/N 为 {qwen_non_ratio:.3f} → {qwen_native_ratio:.3f}，等价于 noise/signal 约 {1/qwen_non_ratio:.2f} → {1/qwen_native_ratio:.2f}。因此 native-thinking 在这里是<strong>更线性可解码</strong>，却没有表现为<strong>全局类内散布相对更小</strong>。由于两种 mode 各自用 discovery 拟合 PCA basis，SNR 可在同一 protocol 与维数下作描述性比较，但不应解释为共享坐标系中的绝对距离差。</p>
{scope_warning}
<div class="callout"><strong>最稳妥的 claim：</strong>当前四个模型×endpoint 对比中，native-thinking 的两种 held-out probe 均高于 non-thinking；但 Qwen running-index 的 SNR 从 −2.27 dB 变为 −2.57 dB，方向相反。这里应写“在各自 discovery-selected 表征上，native-thinking 的 count index 更线性可解码”，而不是笼统写“所有类簇都更紧”。</div>
</section>"""


def rotate_point(x: float, y: float, z: float) -> tuple[float, float]:
    yaw = -0.72
    pitch = 0.46
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x1 = cy * x + sy * z
    z1 = -sy * x + cy * z
    return x1, cp * y - sp * z1


def scatter_svg(points: list[dict[str, str]], *, centered: bool) -> str:
    coordinates = []
    for row in points:
        prefix = "centered_" if centered else ""
        x, y = rotate_point(
            float(row[f"{prefix}pc1"]),
            float(row[f"{prefix}pc2"]),
            float(row[f"{prefix}pc3"]),
        )
        coordinates.append((x, y, row))
    xs = [value[0] for value in coordinates]
    ys = [value[1] for value in coordinates]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    width, height, pad = 540, 330, 26

    def sx(value: float) -> float:
        return pad + (value - xmin) / dx * (width - 2 * pad)

    def sy(value: float) -> float:
        return height - pad - (value - ymin) / dy * (height - 2 * pad)

    circles = []
    for x, y, row in coordinates:
        if centered:
            color = COUNT_PALETTE[int(row["occurrence"]) - 1]
            label = f"k={row['occurrence']} · seed={row['seed']}"
        else:
            color = MARKER_PALETTE.get(row["marker_kind"], "#626A74")
            label = f"{row['marker_kind']} · seed={row['seed']} · k={row['occurrence']}"
        circles.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4.1" fill="{color}" fill-opacity=".72"><title>{esc(label)}</title></circle>'
        )
    title = (
        "Per-trajectory centered · color = running k"
        if centered
        else "Raw PCA3 · color = marker_kind"
    )
    return f"""<figure class="band-figure"><h3>{esc(title)}</h3><svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><rect x=".5" y=".5" width="{width-1}" height="{height-1}" fill="#F8F4EC" stroke="#DDD5C9"/>{''.join(circles)}</svg></figure>"""


def band_section(
    audit: Mapping[str, Any],
    point_rows: list[dict[str, str]],
    parser_rows: list[dict[str, Any]],
) -> str:
    raw = audit["display_pca3"]["confirmation_two_band"]
    centered = audit["within_trajectory_centered_pca3"]
    trajectory = audit["trajectory_band_summary"]
    associations = {row["column"]: row for row in audit["categorical_associations"]}
    centered_associations = {
        row["column"]: row for row in centered["categorical_associations"]
    }
    numeric = {row["column"]: row for row in audit["numeric_associations"]}
    marker = associations["marker_kind"]
    boundary = associations["boundary_kind"]
    occurrence = numeric["occurrence"]
    between_fraction = float(
        audit["hidden_space_variance"][
            "between_trajectory_fraction_after_discovery_scaling"
        ]
    )
    sensitivity = audit["site_layer_sensitivity"]
    site_by_name = {
        str(row["site_kind"]): row for row in sensitivity["selected_layer_sites"]
    }
    site_rows = []
    for row in sensitivity["selected_layer_sites"]:
        site_rows.append(
            (
                f"<code>{esc(row['site_kind'])}</code>",
                f"{float(row['raw_silhouette']):.3f}",
                f"{float(row['raw_marker_kind_nmi']):.3f}",
                f"{float(row['raw_occurrence_nmi']):.3f}",
                f"{100*float(row['raw_mean_trajectory_purity']):.1f}%",
                f"{float(row['centered_marker_kind_nmi']):.3f}",
            )
        )
    peak_marker = sensitivity["selected_site_peak_marker_nmi"]
    marker_analysis = qwen_band_marker_analysis(point_rows, parser_rows)
    legacy_family = marker_analysis["legacy_family_counts"]
    trajectory_family = marker_analysis["trajectory_counts"]
    legacy_state_purity = (
        legacy_family["completion_recap"]["lower"]
        + legacy_family["other_marker"]["upper"]
    ) / len(point_rows)
    marker_rows = []
    for kind in ("completion_recap", "indexed", "ordinal", "audit_sentence"):
        counts = marker["counts"].get(kind, {"upper": 0, "lower": 0})
        marker_rows.append(
            (
                f"<code>{esc(kind)}</code>",
                str(int(counts.get("lower", 0))),
                str(int(counts.get("upper", 0))),
                str(int(counts.get("lower", 0)) + int(counts.get("upper", 0))),
            )
        )
    family_rows = []
    for family in ("completion_recap", "other_marker"):
        counts = legacy_family[family]
        family_rows.append(
            (
                f"<code>{esc(family)}</code>",
                str(counts["lower"]),
                str(counts["upper"]),
                str(counts["lower"] + counts["upper"]),
            )
        )
    trajectory_family_rows = []
    for family in ("completion_recap", "other_marker"):
        counts = trajectory_family[family]
        trajectory_family_rows.append(
            (
                f"<code>{esc(family)}</code>",
                str(counts["lower"]),
                str(counts["upper"]),
                str(counts["lower"] + counts["upper"]),
            )
        )
    hybrid_rows = []
    for kind in HYBRID_MARKERS:
        if kind not in marker_analysis["hybrid_counts"]:
            continue
        counts = marker_analysis["hybrid_counts"][kind]
        hybrid_rows.append(
            (
                f"<code>{esc(kind)}</code>",
                str(counts["lower"]),
                str(counts["upper"]),
                str(counts["lower"] + counts["upper"]),
            )
        )
    raw_metrics = audit["ordinal_decodability"]["raw"]
    centered_metrics = audit["ordinal_decodability"][
        "within_trajectory_centered_diagnostic"
    ]
    return f"""
<section id="qwen-bands"><h2>Qwen native-thinking 为什么分成上下两层</h2>
<p><strong>结论：</strong>旧 capture taxonomy 下，上下层几乎等价于两个 <em>marker family</em>：<code>completion_recap</code> 在下层，其余 indexed/ordinal/audit 格式在上层；但它<strong>不是两个单独 marker 一一对应</strong>。current hybrid parser 重标后，<code>inline_count</code> 横跨两层，说明更稳妥的机制是整条 trace 的 surface-template / boundary offset，而不是 count 类别或两个内部计数器。分析对象严格对应截图：<code>post_boundary @ L18</code>、gold N=10、10 条 confirmation trajectories、92 个实际 item states；PCA3 仍只在 discovery 拟合。</p>
<div class="band-grid">{scatter_svg(point_rows, centered=False)}{scatter_svg(point_rows, centered=True)}</div>
<div class="band-legends"><div><strong>Raw marker_kind：</strong>{''.join(f'<span><i style="background:{color}"></i>{esc(kind)}</span>' for kind,color in MARKER_PALETTE.items() if kind in {'completion_recap','indexed','ordinal','audit_sentence'})}</div><div><strong>Centered：</strong>颜色仍是 k=1…10；去均值只用于 nuisance 诊断，不是部署时可用的 estimator。</div></div>
{table(['marker_kind','下层','上层','合计'], marker_rows)}
<h3>“两层是不是两种 marker？”的直接检验</h3>
<div class="definitions two"><div><h3>Capture-time 五类标签</h3>{table(['legacy marker family','下层 states','上层 states','合计'], family_rows)}<p><code>completion_recap</code> 有 {legacy_family['completion_recap']['lower']}/{sum(legacy_family['completion_recap'].values())} states 在下层；其他 marker 有 {legacy_family['other_marker']['upper']}/{sum(legacy_family['other_marker'].values())} states 在上层，按 state 描述性 purity = {100*legacy_state_purity:.1f}%。同一 trace 内的 states 不独立，所以这张 state 表不用于显著性检验。</p></div><div><h3>以 trajectory 为独立单位</h3>{table(['legacy marker family','下层-majority traces','上层-majority traces','合计'], trajectory_family_rows)}<p>4 条 recap trajectories 都以下层为主，6 条其他 trajectories 都以上层为主；two-sided Fisher exact <em>p</em> = {float(marker_analysis['trajectory_fisher_two_sided_p']):.4f}。这是截图后提出的 post-hoc 检验且只有 n=10，证据很强但仍需在新增 confirmation seeds 上预注册复现。</p></div></div>
<details open><summary>用 current hybrid parser 重标同一批 traces</summary>
{table(['hybrid marker_kind','下层 states','上层 states','合计'], hybrid_rows)}
<p class="small">hybrid marker 与 band 的 NMI = {float(marker_analysis['hybrid_nmi']):.3f}，按 marker 取多数 band 的 state purity = {100*float(marker_analysis['hybrid_weighted_band_purity']):.1f}%。关联仍在，但比 capture-time legacy marker NMI {float(marker['nmi']):.3f} 弱；关键是 <code>inline_count</code> 同时包含上下层。这排除了“新版 parser 的两个离散 marker 类恰好就是两层”的简单解释。</p></details>
<div class="definitions two"><div><h3>原始 band</h3><p>K-means PCA3 silhouette = {float(raw['silhouette']):.3f}；上下层为 {int(raw['cluster_sizes']['upper'])}/{int(raw['cluster_sizes']['lower'])} states。平均 trajectory 内 band purity = {100*float(trajectory['mean_within_trajectory_band_purity']):.1f}%，{int(trajectory['fully_single_band_trajectories'])}/10 条 trace 完全落在单一 band；discovery-scaled hidden space 中 {100*between_fraction:.1f}% 的总平方和位于 trajectory means 之间。</p></div><div><h3>格式归因</h3><p><code>marker_kind</code> 与 band 的 NMI = {float(marker['nmi']):.3f}；<code>boundary_kind</code> NMI = {float(boundary['nmi']):.3f}。<code>completion_recap/recap_period</code> 几乎全在下层；indexed/ordinal 的 newline 与 audit_sentence_period 几乎全在上层。相反，上/下层平均 k 仅为 {float(occurrence['upper_mean']):.2f}/{float(occurrence['lower_mean']):.2f}，标准化均值差 {float(occurrence['standardized_mean_difference']):.3f}。</p></div></div>
<h3>站点与层的敏感性</h3>
{table(['L18 token site','silhouette','marker NMI','occurrence NMI','trajectory purity','去 trajectory mean 后 marker NMI'], site_rows)}
<p class="small"><code>city_end</code> 的两簇主要跟 occurrence 走（marker NMI {float(site_by_name['city_end']['raw_marker_kind_nmi']):.3f}，occurrence NMI {float(site_by_name['city_end']['raw_occurrence_nmi']):.3f}）；到 <code>item_end</code>/<code>post_boundary</code> 后，marker NMI 升至 {float(site_by_name['item_end']['raw_marker_kind_nmi']):.3f}/{float(site_by_name['post_boundary']['raw_marker_kind_nmi']):.3f}。<code>post_boundary</code> 的格式分离也不是 L18 偶然：36 层扫描的 peak marker NMI = {float(peak_marker['raw_marker_kind_nmi']):.3f} @ L{int(peak_marker['layer'])}。旧 capture 未保存 <code>pre_city</code> 与 <code>city_unit_end</code>，因此这两站点暂不能加入该敏感性表。</p>
<div class="callout"><strong>关键去混杂：</strong>对每条 trajectory 在原 hidden space 内减去自己的 state 均值后，原 band 与新坐标聚类的 NMI = {float(centered['raw_vs_centered_band_nmi']):.6f}，marker-kind NMI 降到 {float(centered_associations['marker_kind']['nmi']):.3f}。与此同时 held-out Logistic BA {100*float(raw_metrics['confirmation_logistic_balanced_accuracy']):.1f}% → {100*float(centered_metrics['confirmation_logistic_balanced_accuracy']):.1f}%，NCC {100*float(raw_metrics['confirmation_ncc_balanced_accuracy']):.1f}% → {100*float(centered_metrics['confirmation_ncc_balanced_accuracy']):.1f}%，SNR {float(raw_metrics['confirmation_class_balanced_snr_db']):.2f} → {float(centered_metrics['confirmation_class_balanced_snr_db']):.2f} dB。也就是说，去掉 trace-level offset 后 count 可解码性基本保留。</div>
<div class="callout warning"><strong>机制解释边界：</strong><code>post_boundary</code> 是统一的 parser 名称，却不是统一的表面 token：completion recap 落在 recap period，indexed/ordinal 多落在 newline，audit sentence 落在句号。当前证据支持“格式/边界方向叠加在 count manifold 上”，但尚不能区分是标点 token、换行、模板句法还是更长程的 trace-style state 导致。下一步应在全 300 capture 上报告 marker-kind 着色、format-stratified probes，以及 trajectory-centered sensitivity；不要把上下 band 当成两个内部计数器。</div>
</section>"""


def remove_block(text: str, begin: str, end: str) -> str:
    if begin not in text:
        return text
    start = text.index(begin)
    stop = text.index(end, start) + len(end)
    while start > 0 and text[start - 1] in "\r\n":
        start -= 1
    while stop < len(text) and text[stop] in "\r\n":
        stop += 1
    return text[:start] + text[stop:]


def update_report(
    *,
    report: Path,
    manifest: Path | None,
    parser_audit: Path,
    native_capture_root: Path,
    native_trace_root: Path | None,
    dual_root: Path,
    band_audit: Path,
    band_points: Path,
) -> None:
    document = report.read_text(encoding="utf-8")
    document = remove_block(document, BEGIN, END)
    document = remove_block(document, STYLE_BEGIN, STYLE_END)
    parser_rows = read_jsonl(parser_audit)
    trace_summary = trace_category_summary(parser_rows)
    full_legacy_summary = legacy_compatible_marker_summary(parser_rows)
    hybrid_summary = hybrid_marker_summary(parser_rows)
    capture_summary, capture_inputs = capture_marker_summary(native_capture_root)
    unresolved_examples, unresolved_inputs = unresolved_trace_examples(
        parser_rows, native_trace_root
    )
    pairs = selected_pairs(dual_root)
    schemas = {
        str(
            read_json(
                dual_root / model / "pca16_whiten" / "dual_endpoint_geometry_audit.json"
            ).get("schema_version")
        )
        for model in MODELS
    }
    legacy_geometry_capture = schemas.isdisjoint(
        {
            "realistic_niah_dual_endpoint_geometry_v3_all_counts",
            "realistic_niah_dual_endpoint_geometry_v4_pooled_all_counts",
        }
    )
    diagnostic = read_json(band_audit)
    points = read_csv(band_points)
    inserted = (
        BEGIN
        + category_section(
            capture_summary,
            full_legacy_summary,
            hybrid_summary,
            trace_summary,
            unresolved_examples,
            legacy_geometry_capture=legacy_geometry_capture,
        )
        + clarity_section(
            pairs, legacy_geometry_capture=legacy_geometry_capture
        )
        + band_section(diagnostic, points, parser_rows)
        + END
    )
    design_start = document.index('<section id="design">')
    design_end = document.index("</section>", design_start) + len("</section>")
    document = document[:design_end] + inserted + document[design_end:]
    if legacy_geometry_capture:
        document = document.replace(
            "同一份报告并列展示 non-thinking、经过 one-to-one 结构清洗的 native-thinking，"
            "以及按实际出现 ordinal 对齐的 native-thinking；每个模型都从完整的 10 counts × "
            "30 seeds = 300 条注册轨迹出发。",
            "本报告按 10 counts × 30 seeds 的注册设计组织三列比较。当前嵌入的 running-index "
            "hidden-state 点云仍是早期 N=10 capture；主 marker 比例与该 30-trace capture 严格"
            "对应，全 300 hybrid-parser marker 分布另作补充。因此几何结果按实际 displayed "
            "trajectories/states 解读。",
            1,
        )
    document = document.replace(
        '<a href="#trace-proportions">五类比例</a>',
        '<a href="#trace-proportions">Marker 比例</a>',
    )
    if '<a href="#trace-proportions">Marker 比例</a>' not in document:
        document = document.replace(
            '<a href="#design">口径</a>',
            '<a href="#design">口径</a><a href="#trace-proportions">Marker 比例</a><a href="#clarity-evidence">清晰度</a><a href="#qwen-bands">Qwen 分层</a>',
            1,
        )
    styles = f"""
{STYLE_BEGIN}
.cat-bars{{max-width:1080px;margin:20px 0 10px}}.cat-bar-row{{display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:center;margin:9px 0}}.cat-bar{{height:28px;display:flex;background:#E1DBD0;overflow:hidden}}.cat-segment{{display:block;height:100%}}.cat-legend,.band-legends,.clarity-legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:10px 0 18px}}.cat-legend span,.band-legends span{{display:inline-flex;align-items:center;gap:5px}}.cat-legend i,.band-legends i{{display:inline-block;width:10px;height:10px;border-radius:50%}}.clarity-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}}.clarity-card{{background:var(--surface);border:1px solid var(--line);padding:15px}}.clarity-card h3{{color:var(--indigo);font-size:16px;margin:0 0 12px}}.clarity-card p{{font-size:12px;color:var(--muted)}}.clarity-row{{display:grid;grid-template-columns:128px minmax(180px,1fr) 110px;gap:10px;align-items:center;margin:10px 0;font-size:12px}}.clarity-track{{height:22px;position:relative;background:linear-gradient(to right,#E5DED3 1px,transparent 1px);background-size:25% 100%;border-left:1px solid #BDB5A8;border-right:1px solid #BDB5A8}}.clarity-link{{position:absolute;top:10px;height:2px;background:#8A838E}}.clarity-dot{{position:absolute;top:5px;width:12px;height:12px;border-radius:50%;transform:translateX(-50%);z-index:2}}.clarity-dot.non{{background:#20242D}}.clarity-dot.native{{background:#00A88F}}.clarity-legend .clarity-dot{{position:static;display:inline-block;transform:none;vertical-align:-2px;margin-right:5px}}.clarity-values{{font:11px/1.4 Consolas,monospace}}.band-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0}}.band-figure{{margin:0;background:var(--surface);border:1px solid var(--line);padding:12px}}.band-figure h3{{font-size:15px;color:var(--indigo);margin:0 0 8px}}.band-figure svg{{display:block;width:100%;height:auto}}@media(max-width:900px){{.clarity-grid,.band-grid{{grid-template-columns:1fr}}.clarity-row{{grid-template-columns:110px 1fr 96px}}}}@media(max-width:560px){{.cat-bar-row{{grid-template-columns:1fr}}.clarity-row{{grid-template-columns:1fr}}.clarity-track{{width:100%}}}}
{STYLE_END}
"""
    document = document.replace("</style>", styles + "</style>", 1)
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report)

    if manifest is not None:
        value = read_json(manifest)
        value["diagnostic_augmentation"] = {
            "schema_version": "geometry_diagnostic_augmentation_v2",
            "trace_category_denominator": "trajectory",
            "legacy_unresolved_trajectories": len(unresolved_examples),
            "qwen_band_scope": diagnostic["scope"],
            "inputs": {
                str(path.resolve()): sha256(path)
                for path in (
                    parser_audit,
                    *capture_inputs,
                    *unresolved_inputs,
                    band_audit,
                    band_points,
                )
            },
            "dual_endpoint_root": str(dual_root.resolve()),
        }
        value["output_sha256"] = sha256(report)
        temporary_manifest = manifest.with_name(manifest.name + ".tmp")
        temporary_manifest.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parser-audit", type=Path, required=True)
    parser.add_argument("--native-capture-root", type=Path, required=True)
    parser.add_argument("--native-trace-root", type=Path)
    parser.add_argument("--dual-endpoint-root", type=Path, required=True)
    parser.add_argument("--band-audit", type=Path, required=True)
    parser.add_argument("--band-points", type=Path, required=True)
    args = parser.parse_args()
    update_report(
        report=args.report.resolve(),
        manifest=None if args.manifest is None else args.manifest.resolve(),
        parser_audit=args.parser_audit.resolve(),
        native_capture_root=args.native_capture_root.resolve(),
        native_trace_root=(
            None if args.native_trace_root is None else args.native_trace_root.resolve()
        ),
        dual_root=args.dual_endpoint_root.resolve(),
        band_audit=args.band_audit.resolve(),
        band_points=args.band_points.resolve(),
    )
    print(args.report.resolve())


if __name__ == "__main__":
    main()
