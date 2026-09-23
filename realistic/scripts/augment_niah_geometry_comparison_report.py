"""Trace-category and frozen-band diagnostics retained for statistical reproducibility."""

from __future__ import annotations

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

SPLITS = ("all", "discovery", "confirmation")

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
                "Prose with inline rank/count words (one/second/third) after cities; "
                "no separate line-initial list."
            )
        elif marker == "evidence_sequence":
            surface_description = (
                "Prose without an acceptable contiguous 1...M span; audit order follows "
                "the occurrence of city-plus-score evidence."
            )
        else:
            surface_description = "None of the five legacy categories matched; the hybrid parser uses other evidence."
        selected = int(row["item_count"])
        gold = int(row["gold_count"])
        if str(row["trace_category"]) == "synthetic_unverified":
            hybrid_description = (
                f"evidence_sequence: {selected} score-supported items; assigned ordinal "
                "labels 1...M, explicitly marked synthetic_unverified."
            )
        else:
            hybrid_description = (
                f"{marker}: selected {selected}/{gold} items with local rank evidence; "
                f"trace_category={row['trace_category']}."
            )
            if selected < gold:
                hybrid_description += " Later unranked cities are not filled in using the final Total."
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


def normalized_mutual_information_labels(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("NMI inputs must be non-empty and equally sized")
    total = float(len(left))

    def entropy(values: list[str]) -> float:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    joint: dict[tuple[str, str], int] = {}
    for one, two in zip(left, right):
        left_counts[one] = left_counts.get(one, 0) + 1
        right_counts[two] = right_counts.get(two, 0) + 1
        joint[(one, two)] = joint.get((one, two), 0) + 1
    mutual_information = 0.0
    for (one, two), count in joint.items():
        probability = count / total
        mutual_information += probability * math.log(
            probability
            / ((left_counts[one] / total) * (right_counts[two] / total))
        )
    denominator = (entropy(left) + entropy(right)) / 2.0
    return 1.0 if denominator == 0.0 else mutual_information / denominator
