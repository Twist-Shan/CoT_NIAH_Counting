from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        row = {"event": str(event), "unix_time": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[v4.4.2] {event} "
            + " ".join(f"{key}={value}" for key, value in payload.items()),
            flush=True,
        )

    @contextmanager
    def timer(self, name: str, **payload: Any) -> Iterator[None]:
        started = time.perf_counter()
        self.write(f"{name}_start", **payload)
        try:
            yield
        except Exception as error:
            self.write(
                f"{name}_failed",
                elapsed_seconds=time.perf_counter() - started,
                error_type=type(error).__name__,
                error=str(error),
                **payload,
            )
            raise
        self.write(
            f"{name}_complete",
            elapsed_seconds=time.perf_counter() - started,
            **payload,
        )


def select_stimuli(
    stimuli_path: str | Path,
    *,
    variants: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    counts: Sequence[int] | None = None,
    split: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(stimuli_path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    variant_set = None if variants is None else {str(value) for value in variants}
    seed_set = None if seeds is None else {int(value) for value in seeds}
    count_set = None if counts is None else {int(value) for value in counts}
    selected = [
        row
        for row in rows
        if (
            (variant_set is None or str(row["design_variant"]) in variant_set)
            and (seed_set is None or int(row["seed"]) in seed_set)
            and (count_set is None or int(row["gold_count"]) in count_set)
            and (split is None or str(row["split"]) == str(split))
        )
    ]
    if not selected:
        raise ValueError("V4.4.2 stimulus filters selected no rows")
    return selected
