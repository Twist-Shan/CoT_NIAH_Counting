from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .parsing import parse_trace_record
from .spec import V5Config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(target)
    return count


def registered_records(
    rows: Iterable[Mapping[str, Any]],
    config: V5Config,
    *,
    model_label: str | None = None,
) -> list[dict[str, Any]]:
    config.validate()
    selected: list[dict[str, Any]] = []
    for row in rows:
        count = int(row.get("gold_count", len(row.get("gold_records", row.get("gold_pairs", [])))))
        seed = int(row.get("seed", -1))
        if str(row.get("design_variant", "v4.4")) != config.design_variant:
            continue
        if count not in config.counts or seed not in config.all_seeds:
            continue
        if model_label is not None and row.get("model_label") not in {None, model_label}:
            continue
        value = dict(row)
        value["split"] = (
            "discovery" if seed in config.discovery_seeds else "confirmation"
        )
        selected.append(value)
    return selected


def parse_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_input: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        parsed = parse_trace_record(row)
        output.append(({**dict(row), "trace_parse": parsed}) if include_input else parsed)
    return output

