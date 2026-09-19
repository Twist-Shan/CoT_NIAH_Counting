#!/usr/bin/env python3
"""Materialize a site-filtered V5 capture without another model forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = "realistic_niah_v5_filtered_capture_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reusable_output(
    output_dir: Path,
    *,
    source_index_sha256: str,
    requested_site_kinds: set[str],
    expected_rows: int,
) -> Path | None:
    """Return a previously completed filtered index, never a partial output."""

    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise FileExistsError(f"Filtered output is not a directory: {output_dir}")
    if not any(output_dir.iterdir()):
        return None
    manifest_path = output_dir / "capture_manifest.json"
    index_path = output_dir / "capture_index.jsonl"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileExistsError(
            f"Filtered output is partial; use a new output directory: {output_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or str(manifest.get("source_capture_index_sha256"))
        != source_index_sha256
        or set(map(str, manifest.get("selected_site_kinds", [])))
        != requested_site_kinds
        or int(manifest.get("rows", -1)) != expected_rows
    ):
        raise FileExistsError(
            f"Filtered output does not match this request; use a new output "
            f"directory: {output_dir}"
        )
    rows = read_jsonl(index_path)
    if len(rows) != expected_rows:
        raise FileExistsError(
            f"Filtered output index is incomplete; use a new output directory: "
            f"{output_dir}"
        )
    for row in rows:
        row_manifest_path = output_dir / str(row["manifest_path"])
        row_states_path = output_dir / str(row["states_path"])
        if not row_manifest_path.is_file() or not row_states_path.is_file():
            raise FileExistsError(
                f"Filtered output is missing shard files; use a new output "
                f"directory: {output_dir}"
            )
        row_manifest = json.loads(row_manifest_path.read_text(encoding="utf-8"))
        row_kinds = {
            str(site.get("site_kind"))
            for site in row_manifest.get("site_rows", [])
        }
        if row_kinds != requested_site_kinds:
            raise FileExistsError(
                f"Filtered output contains the wrong site kinds; use a new "
                f"output directory: {output_dir}"
            )
    return index_path


def filter_capture(
    source_index: Path,
    output_dir: Path,
    *,
    site_kinds: Iterable[str],
) -> Path:
    requested = tuple(dict.fromkeys(map(str, site_kinds)))
    if not requested:
        raise ValueError("At least one site kind is required")
    requested_set = set(requested)
    source_rows = read_jsonl(source_index)
    if not source_rows:
        raise ValueError(f"Empty source capture index: {source_index}")
    source_index_sha256 = sha256(source_index)
    reusable = reusable_output(
        output_dir,
        source_index_sha256=source_index_sha256,
        requested_site_kinds=requested_set,
        expected_rows=len(source_rows),
    )
    if reusable is not None:
        return reusable

    output_rows: list[dict[str, Any]] = []
    site_counts: list[int] = []
    for row_number, source_row in enumerate(source_rows):
        source_manifest_path = source_index.parent / str(
            source_row["manifest_path"]
        )
        source_states_path = source_index.parent / str(source_row["states_path"])
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        retained_indices = [
            index
            for index, site in enumerate(manifest.get("site_rows", []))
            if str(site.get("site_kind")) in requested_set
        ]
        if not retained_indices:
            raise ValueError(
                f"{source_row.get('request_id')}: none of {requested} are present"
            )
        if requested == ("answer_query_v3",) and len(retained_indices) != 1:
            raise ValueError(
                f"{source_row.get('request_id')}: expected exactly one "
                f"answer_query_v3 site, found {len(retained_indices)}"
            )
        with np.load(source_states_path, allow_pickle=False) as archive:
            layer_indices = np.asarray(archive["layer_indices"])
            site_states = np.asarray(archive["site_states"])[retained_indices]

        relative_dir = Path("shards") / f"row_{row_number:04d}"
        states_path = output_dir / relative_dir / "states.npz"
        manifest_path = output_dir / relative_dir / "capture_manifest.json"
        atomic_npz(
            states_path,
            layer_indices=layer_indices,
            site_states=site_states,
        )
        filtered_manifest = dict(manifest)
        filtered_manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "source_capture_schema_version": manifest.get("schema_version"),
                "source_manifest_path": str(source_manifest_path.resolve()),
                "source_states_sha256": sha256(source_states_path),
                "selected_site_kinds": sorted(requested_set),
                "capture_span_pooling": False,
                "site_rows": [
                    manifest["site_rows"][index] for index in retained_indices
                ],
                "span_site_ids": [],
                "states_file": "states.npz",
                "site_states_shape": list(site_states.shape),
            }
        )
        atomic_json(manifest_path, filtered_manifest)
        filtered_row = dict(source_row)
        filtered_row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "source_row_schema_version": source_row.get("schema_version"),
                "manifest_path": (
                    relative_dir / "capture_manifest.json"
                ).as_posix(),
                "states_path": (relative_dir / "states.npz").as_posix(),
            }
        )
        output_rows.append(filtered_row)
        site_counts.append(len(retained_indices))

    output_index = output_dir / "capture_index.jsonl"
    atomic_jsonl(output_index, output_rows)
    atomic_json(
        output_dir / "capture_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_capture_index": str(source_index.resolve()),
            "source_capture_index_sha256": source_index_sha256,
            "selected_site_kinds": sorted(requested_set),
            "rows": len(output_rows),
            "site_rows": int(sum(site_counts)),
            "sites_per_trajectory_min": int(min(site_counts)),
            "sites_per_trajectory_max": int(max(site_counts)),
            "split_trajectory_counts": {
                split: sum(str(row.get("split")) == split for row in output_rows)
                for split in ("discovery", "confirmation")
            },
            "gold_counts": sorted(
                {int(row["gold_count"]) for row in output_rows}
            ),
        },
    )
    return output_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-kinds", nargs="+", required=True)
    args = parser.parse_args()
    result = filter_capture(
        args.source_index.resolve(),
        args.output.resolve(),
        site_kinds=args.site_kinds,
    )
    print(result)


if __name__ == "__main__":
    main()
