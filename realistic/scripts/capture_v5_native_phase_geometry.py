#!/usr/bin/env python3
"""Replay frozen native-thinking completions at exact post-marker/city sites.

This supplement deliberately does not regenerate text.  It augments the
historical V5 representation capture with token positions that were not saved:

``post_marker``
    The causal compiler's registered query token after an explicit rank marker
    and before the target city.  It is defined only for rank-before-city
    events and is a marker-conditioned positive control, not an implicit
    counter estimate.

``marker_end``
    The last original output token of the explicit rank core.  This broader
    lexical control is also available when the marker follows the city.

``post_city``
    The first *original output token* strictly after the city-containing token.
    It is captured only when that token still belongs to the registered item.
    We never re-tokenize a truncated whitespace suffix to manufacture a token.

Every selected position for one request is captured in a single forward pass.
The output is restartable and contains a flat site axis because not every event
has every site.
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
from typing import Any, Mapping, Sequence

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


SCHEMA_VERSION = "realistic_niah_v5_native_phase_geometry_capture_v1"
SITE_ORDER = ("post_marker", "marker_end", "post_city")


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


def optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    return None if not text else int(text)


def read_events(path: str | Path, *, model_label: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("model_label")) != model_label:
                continue
            required = ("request_id", "occurrence", "city_output_token_end")
            if any(str(row.get(key, "")).strip() == "" for key in required):
                raise ValueError(f"Incomplete event row: {row}")
            events.append(dict(row))
    if not events:
        raise ValueError(f"No events for {model_label}")
    keys = [(str(row["request_id"]), int(row["occurrence"])) for row in events]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate request/occurrence event keys")
    return sorted(events, key=lambda row: (str(row["request_id"]), int(row["occurrence"])))


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


def classify_surface(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return "whitespace"
    if all(not char.isalnum() for char in stripped):
        return "syntax"
    if any(char.isalpha() for char in stripped):
        return "lexical"
    if any(char.isdigit() for char in stripped):
        return "numeric"
    return "other"


def event_sites(
    event: Mapping[str, Any],
    *,
    output_ids: Sequence[int],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    occurrence = int(event["occurrence"])
    city_end = int(event["city_output_token_end"])
    commit = optional_int(event.get("commit_output_token"))
    common = {
        "occurrence": occurrence,
        "city": str(event.get("city", "")),
        "gold_count": int(event["gold_count"]),
        "grammar_class": str(event.get("grammar_class", "")),
        "surface_order": str(event.get("surface_order", "")),
        "marker_kind": str(event.get("marker_kind", "")),
        "causal_cohort": str(event.get("causal_cohort", "")),
        "primary_full_chain_event": truthy(event.get("primary_full_chain_event")),
        "progress_commit_eligible": truthy(event.get("progress_commit_eligible")),
        "trace_category": str(event.get("trace_category", "")),
    }
    result: list[dict[str, Any]] = []

    # Exact registered query after a visible rank phrase and before the city.
    query = optional_int(event.get("retrieve_query_output_token"))
    if (
        query is not None
        and str(event.get("surface_order")) == "rank_before_city"
        and truthy(event.get("marker_control_eligible"))
        and 0 <= query < len(output_ids)
    ):
        token_text = tokenizer.decode(
            [int(output_ids[query])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result.append(
            {
                **common,
                "site": "post_marker",
                "output_token_index": query,
                "token_id": int(output_ids[query]),
                "token_text": token_text,
                "token_surface_class": classify_surface(token_text),
                "label_semantics": "completed_count_k_before_city_k",
            }
        )

    # Broader lexical control: state after the explicit rank core itself.
    rank_end = optional_int(event.get("rank_output_token_end"))
    if (
        rank_end is not None
        and int(event.get("rank_right_spill_chars") or 0) == 0
        and 0 < rank_end <= len(output_ids)
    ):
        index = rank_end - 1
        token_text = tokenizer.decode(
            [int(output_ids[index])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result.append(
            {
                **common,
                "site": "marker_end",
                "output_token_index": index,
                "token_id": int(output_ids[index]),
                "token_text": token_text,
                "token_surface_class": classify_surface(token_text),
                "label_semantics": "explicit_rank_core_contains_k",
            }
        )

    # First baseline token after the city-containing token.  Require it to be
    # no later than the accepted item commit, so it is event-local.
    post_city = city_end
    if (
        int(event.get("city_right_spill_chars") or 0) == 0
        and commit is not None
        and 0 <= post_city <= commit
        and post_city < len(output_ids)
    ):
        token_text = tokenizer.decode(
            [int(output_ids[post_city])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result.append(
            {
                **common,
                "site": "post_city",
                "output_token_index": post_city,
                "token_id": int(output_ids[post_city]),
                "token_text": token_text,
                "token_surface_class": classify_surface(token_text),
                "tokens_after_city_to_commit": int(commit - post_city),
                "label_semantics": "completed_count_k_after_city_k",
            }
        )
    return sorted(result, key=lambda row: (int(row["output_token_index"]), SITE_ORDER.index(str(row["site"]))))


def shard_name(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:20]


@torch.inference_mode()
def capture_request(
    model: Any,
    adapter: Any,
    tokenizer: Any,
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
    site_rows = [
        site
        for event in events
        for site in event_sites(event, output_ids=output_ids, tokenizer=tokenizer)
    ]
    if not site_rows:
        raise ValueError(f"No exact phase sites for {request_id}")
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
    states = torch.stack([captured[layer] for layer in layer_indices], dim=1)
    if states.shape[0] != len(site_rows):
        raise RuntimeError("Captured site axis does not match metadata")
    atomic_npz(
        shard_dir / "states.npz",
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        site_states=states.numpy().astype(np.float16, copy=False),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "stimulus_id": row.get("stimulus_id"),
        "model_label": row.get("model_label"),
        "seed": int(row["seed"]),
        "split": str(row["split"]),
        "gold_count": int(row.get("gold_count", len(row.get("gold_records", [])))),
        "site_rows": site_rows,
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
    parser.add_argument("--model", required=True, choices=["Qwen3-8B", "Gemma4-E4B"])
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--event-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--layers", type=int, nargs="+")
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
        row for row in read_jsonl(args.generations) if str(row.get("model_label")) == args.model
    ]
    row_by_request = {str(row["request_id"]): row for row in generation_rows}
    if len(row_by_request) != len(generation_rows):
        raise ValueError("Duplicate generation request ids")
    events = read_events(args.event_registry, model_label=args.model)
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
    index_rows: list[dict[str, Any]] = []
    excluded_requests: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(request_ids, start=1):
        preview_sites = [
            site
            for event in events_by_request[request_id]
            for site in event_sites(
                event,
                output_ids=output_token_ids(row_by_request[request_id]),
                tokenizer=tokenizer,
            )
        ]
        if not preview_sites:
            excluded_requests.append(
                {
                    "request_id": request_id,
                    "events": len(events_by_request[request_id]),
                    "reason": "no_exact_phase_site",
                }
            )
            print(
                f"[phase capture] {request_index}/{len(request_ids)} "
                f"excluded=no_exact_phase_site request={request_id}",
                flush=True,
            )
            continue
        manifest = capture_request(
            model,
            adapter,
            tokenizer,
            row_by_request[request_id],
            events_by_request[request_id],
            output_dir=args.output,
            layers=args.layers,
            source_hashes=source_hashes,
            overwrite=args.overwrite,
        )
        relative = Path("shards") / shard_name(request_id)
        site_counts = Counter(str(row["site"]) for row in manifest["site_rows"])
        index_rows.append(
            {
                "request_id": request_id,
                "model_label": args.model,
                "seed": int(manifest["seed"]),
                "split": str(manifest["split"]),
                "gold_count": int(manifest["gold_count"]),
                "event_count": len(events_by_request[request_id]),
                "site_counts": dict(sorted(site_counts.items())),
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            f"[phase capture] {request_index}/{len(request_ids)} "
            f"sites={len(manifest['site_rows'])} request={request_id}",
            flush=True,
        )

    atomic_jsonl(args.output / "capture_index.jsonl", index_rows)
    selected_manifests = [
        json.loads((args.output / row["manifest_path"]).read_text(encoding="utf-8"))
        for row in index_rows
    ]
    all_sites = [site for manifest in selected_manifests for site in manifest["site_rows"]]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "model_label": args.model,
        "requests": len(index_rows),
        "requests_considered": len(request_ids),
        "excluded_requests": excluded_requests,
        "events_in_registry": sum(len(events_by_request[rid]) for rid in request_ids),
        "captured_sites": len(all_sites),
        "site_counts": dict(sorted(Counter(str(row["site"]) for row in all_sites).items())),
        "site_surface_classes": {
            site: dict(
                sorted(
                    Counter(
                        str(row["token_surface_class"])
                        for row in all_sites
                        if str(row["site"]) == site
                    ).items()
                )
            )
            for site in SITE_ORDER
        },
        "splits": dict(sorted(Counter(str(m["split"]) for m in selected_manifests).items())),
        "layers": selected_manifests[0]["layers"],
        "source_hashes": source_hashes,
        "capture_index_sha256": sha256(args.output / "capture_index.jsonl"),
        "generation_replayed_without_resampling": True,
        "full_sequence_hidden_states_materialized": False,
        "intervention_applied": False,
        "post_marker_warning": (
            "Defined only for rank-before-city events and follows an explicit rank; "
            "treat as a lexical positive control, not implicit counter evidence."
        ),
        "post_city_definition": (
            "First original output token strictly after the city-containing token, "
            "provided it is no later than the registered item commit."
        ),
    }
    atomic_json(args.output / "capture_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
