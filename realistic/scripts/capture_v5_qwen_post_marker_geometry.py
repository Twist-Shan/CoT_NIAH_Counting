#!/usr/bin/env python3
"""Capture exact Qwen pre/post-marker residual states for geometry analysis.

The final causal compiler locates ``post_marker`` at an exact output-token
index, but the historical representation shards did not save that position.
This restartable capture replays the frozen native-thinking completions and
saves only two paired states per eligible event:

``pre_marker``
    The output token immediately before the marker surface begins.

``post_marker``
    The compiler-registered retrieval query token after the marker and before
    the target city.

No text is regenerated and no intervention is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import (  # noqa: E402
    capture_post_block_states,
    load_registered_model,
)
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.parsing import (  # noqa: E402
    output_token_ids,
    prompt_token_ids,
)


SCHEMA_VERSION = "realistic_niah_v5_post_marker_geometry_capture_v1"
DEFAULT_GRAMMARS = (
    "adjacent_rank_before_city",
    "same_unit_rank_before_city",
    "structural_explicit_rank_before_city",
)
SITE_ORDER = ("pre_marker", "post_marker")


@dataclass(frozen=True)
class ModelInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def atomic_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_events(
    path: str | Path,
    *,
    model_label: str,
    grammars: Iterable[str],
) -> list[dict[str, Any]]:
    allowed = set(map(str, grammars))
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("model_label")) != model_label:
                continue
            if str(row.get("grammar_class")) not in allowed:
                continue
            if not truthy(row.get("retrieval_eligible")):
                continue
            required = (
                "request_id",
                "occurrence",
                "rank_surface_output_token_start",
                "retrieve_query_output_token",
                "retrieve_query_token_id",
            )
            if any(str(row.get(key, "")).strip() == "" for key in required):
                raise ValueError(f"Incomplete post-marker event: {row}")
            events.append(dict(row))
    if not events:
        raise ValueError("No eligible post-marker events remain")
    keys = [(row["request_id"], int(row["occurrence"])) for row in events]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate request/occurrence post-marker events")
    return sorted(
        events,
        key=lambda row: (str(row["request_id"]), int(row["occurrence"])),
    )


def truncated_input(row: Mapping[str, Any], output_end: int) -> ModelInput:
    prompt = prompt_token_ids(row)
    output = output_token_ids(row)
    if not 0 < int(output_end) <= len(output):
        raise ValueError(f"Invalid output prefix end {output_end}/{len(output)}")
    mask_value = row.get("attention_mask")
    prompt_mask = (
        tuple(int(value) for value in mask_value)
        if mask_value is not None
        else (1,) * len(prompt)
    )
    if len(prompt_mask) != len(prompt):
        raise ValueError("Prompt attention-mask length mismatch")
    output_prefix = output[: int(output_end)]
    return ModelInput(
        input_ids=prompt + output_prefix,
        attention_mask=prompt_mask + (1,) * len(output_prefix),
    )


def event_sites(
    event: Mapping[str, Any],
    *,
    output_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_start = int(event["rank_surface_output_token_start"])
    pre_marker = marker_start - 1
    post_marker = int(event["retrieve_query_output_token"])
    if not 0 <= pre_marker < marker_start <= post_marker < len(output_ids):
        raise ValueError(
            "Invalid pre/post-marker order for "
            f"{event['request_id']} occurrence={event['occurrence']}: "
            f"pre={pre_marker}, marker_start={marker_start}, post={post_marker}, "
            f"output={len(output_ids)}"
        )
    expected_id = int(event["retrieve_query_token_id"])
    actual_id = int(output_ids[post_marker])
    if actual_id != expected_id:
        raise ValueError(
            "Registry post-marker token id mismatch for "
            f"{event['request_id']} occurrence={event['occurrence']}: "
            f"index={post_marker}, expected={expected_id}, actual={actual_id}"
        )
    common = {
        "occurrence": int(event["occurrence"]),
        "grammar_class": str(event["grammar_class"]),
        "primary_full_chain_event": truthy(event.get("primary_full_chain_event")),
        "rank_surface_text": str(event.get("rank_surface_text", "")),
        "city": str(event.get("city", "")),
    }
    return (
        {
            **common,
            "site": "pre_marker",
            "output_token_index": pre_marker,
            "token_id": int(output_ids[pre_marker]),
        },
        {
            **common,
            "site": "post_marker",
            "output_token_index": post_marker,
            "token_id": actual_id,
            "token_text": str(event.get("retrieve_query_token_text", "")),
        },
    )


def shard_name(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:20]


@torch.inference_mode()
def capture_request(
    model: Any,
    adapter: Any,
    row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    layers: Sequence[int] | None,
    source_hashes: Mapping[str, str],
    overwrite: bool,
) -> dict[str, Any]:
    request_id = str(row["request_id"])
    shard_dir = output_dir / "shards" / shard_name(request_id)
    manifest_path = shard_dir / "capture_manifest.json"
    if manifest_path.is_file() and not overwrite:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Incompatible existing shard: {manifest_path}")
        if saved.get("source_hashes") != dict(source_hashes):
            raise RuntimeError(f"Existing shard source hashes differ: {manifest_path}")
        return saved

    output_ids = output_token_ids(row)
    site_pairs = [event_sites(event, output_ids=output_ids) for event in events]
    site_rows = [site for pair in site_pairs for site in pair]
    output_positions = [int(site["output_token_index"]) for site in site_rows]
    output_end = max(output_positions) + 1
    encoding = truncated_input(row, output_end)
    prompt_count = len(prompt_token_ids(row))
    sequence_positions = [prompt_count + position for position in output_positions]
    started = time.perf_counter()
    _logits, captured = capture_post_block_states(
        model,
        adapter,
        encoding,
        sequence_positions,
        layers=layers,
    )
    layer_indices = tuple(sorted(captured))
    stacked = torch.stack([captured[layer] for layer in layer_indices], dim=1)
    if stacked.shape[0] != len(site_rows):
        raise RuntimeError("Captured site axis does not match metadata")
    states = stacked.reshape(len(events), len(SITE_ORDER), len(layer_indices), -1)
    states_path = shard_dir / "states.npz"
    atomic_npz(
        states_path,
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        site_states=states.numpy().astype(np.float16, copy=False),
    )
    grouped_sites = []
    for event_index, pair in enumerate(site_pairs):
        grouped_sites.append(
            {
                "event_index": event_index,
                "occurrence": int(pair[0]["occurrence"]),
                "grammar_class": str(pair[0]["grammar_class"]),
                "primary_full_chain_event": bool(pair[0]["primary_full_chain_event"]),
                "rank_surface_text": str(pair[0]["rank_surface_text"]),
                "city": str(pair[0]["city"]),
                "sites": {site["site"]: site for site in pair},
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "stimulus_id": row.get("stimulus_id"),
        "model_label": row.get("model_label"),
        "seed": int(row["seed"]),
        "split": str(row["split"]),
        "gold_count": int(row.get("gold_count", len(row.get("gold_records", [])))),
        "site_order": list(SITE_ORDER),
        "events": grouped_sites,
        "layers": list(layer_indices),
        "state_shape": list(states.shape),
        "states_file": "states.npz",
        "output_prefix_tokens": int(output_end),
        "sequence_length": int(encoding.sequence_length),
        "elapsed_seconds": time.perf_counter() - started,
        "source_hashes": dict(source_hashes),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen3-8B", choices=["Qwen3-8B"])
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--event-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--grammars", nargs="+", default=list(DEFAULT_GRAMMARS))
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "generations_sha256": sha256(args.generations),
        "event_registry_sha256": sha256(args.event_registry),
        "capture_script_sha256": sha256(Path(__file__)),
    }
    generation_rows = [
        row
        for row in read_jsonl(args.generations)
        if str(row.get("model_label")) == args.model
    ]
    row_by_request = {str(row["request_id"]): row for row in generation_rows}
    if len(row_by_request) != len(generation_rows):
        raise ValueError("Duplicate generation request ids")
    events = read_events(
        args.event_registry,
        model_label=args.model,
        grammars=args.grammars,
    )
    events_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        request_id = str(event["request_id"])
        if request_id not in row_by_request:
            raise KeyError(f"Event registry request missing from generations: {request_id}")
        events_by_request[request_id].append(event)
    request_ids = sorted(events_by_request)
    if args.limit_requests is not None:
        request_ids = request_ids[: int(args.limit_requests)]
    if not request_ids:
        raise ValueError("No requests selected")

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    index_rows = []
    for request_index, request_id in enumerate(request_ids, start=1):
        manifest = capture_request(
            model,
            adapter,
            row_by_request[request_id],
            events_by_request[request_id],
            output_dir=args.output,
            layers=args.layers,
            source_hashes=source_hashes,
            overwrite=args.overwrite,
        )
        relative = Path("shards") / shard_name(request_id)
        index_rows.append(
            {
                "request_id": request_id,
                "model_label": args.model,
                "seed": int(manifest["seed"]),
                "split": str(manifest["split"]),
                "gold_count": int(manifest["gold_count"]),
                "event_count": len(manifest["events"]),
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            f"[post-marker capture] {request_index}/{len(request_ids)} "
            f"events={len(manifest['events'])} request={request_id}",
            flush=True,
        )

    atomic_jsonl(args.output / "capture_index.jsonl", index_rows)
    selected_events = [
        event
        for request_id in request_ids
        for event in events_by_request[request_id]
    ]
    grammar_counts = Counter(str(event["grammar_class"]) for event in selected_events)
    split_counts = Counter(
        str(row_by_request[str(event["request_id"])]["split"])
        for event in selected_events
    )
    occurrence_counts = Counter(int(event["occurrence"]) for event in selected_events)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "model_label": args.model,
        "requests": len(request_ids),
        "events": len(selected_events),
        "sites_per_event": len(SITE_ORDER),
        "grammars": dict(sorted(grammar_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "occurrences": {str(key): value for key, value in sorted(occurrence_counts.items())},
        "layers": index_rows and json.loads(
            (args.output / index_rows[0]["manifest_path"]).read_text(encoding="utf-8")
        )["layers"],
        "source_hashes": source_hashes,
        "capture_index_sha256": sha256(args.output / "capture_index.jsonl"),
        "full_sequence_hidden_states_materialized": False,
        "generation_replayed_without_resampling": True,
        "intervention_applied": False,
    }
    atomic_json(args.output / "capture_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
