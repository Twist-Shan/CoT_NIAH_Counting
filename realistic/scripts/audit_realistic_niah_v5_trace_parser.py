#!/usr/bin/env python3
"""Audit V5 trace-local 1..M parsing without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.parsing import parse_trace_record  # noqa: E402


def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield value


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def audit_rows(
    records: Iterable[Mapping[str, Any]], *, gold_count: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        row_gold_count = int(
            record.get("gold_count", len(record.get("gold_records", ())))
        )
        if gold_count is not None and row_gold_count != gold_count:
            continue
        parsed = parse_trace_record(record)
        parser = parsed["parser"]
        sites = parsed["char_sites"]
        episode = parsed["episode_parse"]
        old_parser = record.get("trace_parse", {}).get("parser", {})
        item_count = int(parser["item_count"])
        item_labels = list(range(1, item_count + 1))
        audit = {
            "request_id": parsed.get("request_id"),
            "stimulus_id": parsed.get("stimulus_id"),
            "model_label": parsed.get("model_label"),
            "model_family": parsed.get("model_family"),
            "seed": parsed.get("seed"),
            "split": parsed.get("split"),
            "gold_count": row_gold_count,
            "final_parsed_count": parsed.get("parsed_count"),
            "old_item_count": old_parser.get("item_count"),
            "old_marker_kind": old_parser.get("marker_kind"),
            "item_count": item_count,
            "item_labels": item_labels,
            "surface_item_markers": list(parser["item_markers"]),
            "item_gold_cities": list(parser["item_gold_cities"]),
            "labels_are_trace_local_one_to_m": item_labels
            == list(range(1, item_count + 1)),
            "agrees_with_final_total": parsed.get("parsed_count") == item_count,
            "marker_kind": parser.get("marker_kind"),
            "sequence_source": parsed["sequence_source"],
            "selection_reason": episode["selection_reason"],
            "rank_supported_event_count": int(
                episode["rank_supported_event_count"]
            ),
            "raw_sequence_count": int(episode["raw_sequence_count"]),
            "selected_terminal_rank": episode.get("selected_terminal_rank"),
            "selected_evidence_kinds": (
                list(
                    episode["sequences"][episode["selected_sequence_index"]][
                        "evidence_kinds"
                    ]
                )
                if episode["selected_sequence_index"] is not None
                else []
            ),
            "trace_category": parser.get("trace_category"),
            "trace_one_to_one": bool(parser.get("trace_one_to_one")),
            "pre_marker_sites": sum(
                site["site_kind"] == "pre_marker" for site in sites
            ),
            "marker_end_sites": sum(
                site["site_kind"] == "marker_end" for site in sites
            ),
            "pre_city_sites": sum(site["site_kind"] == "pre_city" for site in sites),
            "city_unit_end_sites": sum(
                site["site_kind"] == "city_unit_end" for site in sites
            ),
            "item_end_sites": sum(site["site_kind"] == "item_end" for site in sites),
            "answer_query_v3_sites": sum(
                site["site_kind"] == "answer_query_v3" for site in sites
            ),
        }
        audited.append(audit)
        by_model[str(audit["model_label"])].append(audit)

    model_summaries: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        model_summaries[model] = {
            "traces": len(rows),
            "parser_hit_traces": sum(int(row["item_count"]) > 0 for row in rows),
            "split_trajectory_counts": dict(
                sorted(Counter(str(row["split"]) for row in rows).items())
            ),
            "trace_items": sum(int(row["item_count"]) for row in rows),
            "item_end_sites": sum(int(row["item_end_sites"]) for row in rows),
            "pre_city_sites": sum(int(row["pre_city_sites"]) for row in rows),
            "city_unit_end_sites": sum(
                int(row["city_unit_end_sites"]) for row in rows
            ),
            "pre_marker_sites": sum(int(row["pre_marker_sites"]) for row in rows),
            "marker_end_sites": sum(int(row["marker_end_sites"]) for row in rows),
            "answer_query_v3_sites": sum(
                int(row["answer_query_v3_sites"]) for row in rows
            ),
            "old_trace_items": sum(int(row["old_item_count"] or 0) for row in rows),
            "trace_local_one_to_m_label_traces": sum(
                bool(row["labels_are_trace_local_one_to_m"]) for row in rows
            ),
            "final_total_agreement_traces": sum(
                bool(row["agrees_with_final_total"]) for row in rows
            ),
            "one_to_one_traces": sum(bool(row["trace_one_to_one"]) for row in rows),
            "one_to_one_split_trajectory_counts": {
                split: sum(
                    str(row["split"]) == split and bool(row["trace_one_to_one"])
                    for row in rows
                )
                for split in ("discovery", "confirmation")
            },
            "item_count_distribution": dict(
                sorted(Counter(int(row["item_count"]) for row in rows).items())
            ),
            "marker_kind_distribution": dict(
                sorted(Counter(str(row["marker_kind"]) for row in rows).items())
            ),
            "sequence_source_distribution": dict(
                sorted(Counter(str(row["sequence_source"]) for row in rows).items())
            ),
            "selection_reason_distribution": dict(
                sorted(Counter(str(row["selection_reason"]) for row in rows).items())
            ),
            "rank_supported_event_count": sum(
                int(row["rank_supported_event_count"]) for row in rows
            ),
            "raw_rank_episode_count": sum(
                int(row["raw_sequence_count"]) for row in rows
            ),
            "rank_evidence_kind_distribution": dict(
                sorted(
                    Counter(
                        kind
                        for row in rows
                        for kind in row["selected_evidence_kinds"]
                    ).items()
                )
            ),
        }
    summary = {
        "gold_count_filter": gold_count,
        "traces": len(audited),
        "models": model_summaries,
    }
    return audited, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--gold-count",
        type=int,
        default=0,
        help="Audit one registered gold count; the default 0 audits all counts.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold_count = None if args.gold_count == 0 else int(args.gold_count)
    rows, summary = audit_rows(_read_jsonl(args.inputs), gold_count=gold_count)
    if args.output is not None:
        _atomic_jsonl(args.output, rows)
        summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
