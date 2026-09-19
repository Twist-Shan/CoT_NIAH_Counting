from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch


RUNTIME_COLUMNS = (
    "event_id",
    "scope",
    "block",
    "position_encoding",
    "mode",
    "step",
    "started_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "status",
    "num_examples",
    "num_batches",
    "device",
    "resumed_or_cached",
    "error_type",
    "peak_cuda_memory_bytes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id(values: dict[str, Any]) -> str:
    identity = {
        key: values.get(key)
        for key in ("scope", "block", "position_encoding", "mode", "step")
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _atomic_runtime_row(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "tables" / "runtime_events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    frame = pd.DataFrame([{column: row.get(column) for column in RUNTIME_COLUMNS}])
    combined = pd.concat((existing, frame), ignore_index=True)
    combined = combined.drop_duplicates(["event_id"], keep="last")
    combined = combined.sort_values(
        ["scope", "position_encoding", "mode", "step", "block"],
        na_position="first",
    ).reset_index(drop=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def _cuda_sync(device: str | torch.device | None) -> None:
    if device is None:
        return
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class TimingEvent:
    num_examples: int | None = None
    num_batches: int | None = None


@contextmanager
def timed_event(
    run_dir: str | Path,
    *,
    scope: str,
    block: str,
    position_encoding: str | None = None,
    mode: str | None = None,
    step: int | None = None,
    device: str | torch.device | None = None,
    num_examples: int | None = None,
    num_batches: int | None = None,
    cached: bool = False,
) -> Iterator[TimingEvent]:
    """Log one deterministic, resumable wall-clock event without touching RNG state."""

    run_dir = Path(run_dir)
    fields: dict[str, Any] = {
        "scope": scope,
        "block": block,
        "position_encoding": position_encoding,
        "mode": mode,
        "step": step,
        "device": str(device) if device is not None else None,
    }
    fields["event_id"] = _event_id(fields)
    started_at = _utc_now()
    label = " ".join(
        part
        for part in (
            f"scope={scope}",
            f"variant={position_encoding}/{mode}" if position_encoding and mode else "",
            f"step={step}" if step is not None else "",
            f"block={block}",
        )
        if part
    )
    print(f"[timing:start] {label}", flush=True)
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    _cuda_sync(device)
    started = time.perf_counter()
    status = "cached" if cached else "complete"
    error_type: str | None = None
    event = TimingEvent(num_examples=num_examples, num_batches=num_batches)
    try:
        yield event
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        _cuda_sync(device)
        duration = time.perf_counter() - started
        peak = None
        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated(device))
        finished_at = _utc_now()
        row = {
            **fields,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "duration_seconds": float(duration),
            "status": status,
            "num_examples": event.num_examples,
            "num_batches": event.num_batches,
            "resumed_or_cached": bool(cached),
            "error_type": error_type,
            "peak_cuda_memory_bytes": peak,
        }
        _atomic_runtime_row(run_dir, row)
        print(
            f"[timing:done] {label} seconds={duration:.2f} status={status}",
            flush=True,
        )


def record_cached_event(
    run_dir: str | Path,
    *,
    scope: str,
    block: str,
    position_encoding: str | None = None,
    mode: str | None = None,
    step: int | None = None,
    device: str | torch.device | None = None,
) -> None:
    with timed_event(
        run_dir,
        scope=scope,
        block=block,
        position_encoding=position_encoding,
        mode=mode,
        step=step,
        device=device,
        cached=True,
    ):
        pass


def record_duration_event(
    run_dir: str | Path,
    *,
    scope: str,
    block: str,
    duration_seconds: float,
    position_encoding: str | None = None,
    mode: str | None = None,
    step: int | None = None,
    device: str | torch.device | None = None,
    num_examples: int | None = None,
    num_batches: int | None = None,
    status: str = "complete",
) -> None:
    """Persist an already measured duration, such as optimizer time between evaluations."""

    fields: dict[str, Any] = {
        "scope": scope,
        "block": block,
        "position_encoding": position_encoding,
        "mode": mode,
        "step": step,
        "device": str(device) if device is not None else None,
    }
    fields["event_id"] = _event_id(fields)
    now = _utc_now()
    row = {
        **fields,
        "started_at_utc": None,
        "finished_at_utc": now,
        "duration_seconds": float(duration_seconds),
        "status": status,
        "num_examples": num_examples,
        "num_batches": num_batches,
        "resumed_or_cached": False,
        "error_type": None,
        "peak_cuda_memory_bytes": None,
    }
    _atomic_runtime_row(Path(run_dir), row)
    label = " ".join(
        part
        for part in (
            f"scope={scope}",
            f"variant={position_encoding}/{mode}" if position_encoding and mode else "",
            f"step={step}" if step is not None else "",
            f"block={block}",
        )
        if part
    )
    print(
        f"[timing:done] {label} seconds={duration_seconds:.2f} status={status}",
        flush=True,
    )
