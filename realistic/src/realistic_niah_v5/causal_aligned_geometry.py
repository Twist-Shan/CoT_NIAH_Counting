"""Load legacy all-layer captures on causal-compiler-aligned progress events.

The V5 representation capture predates the final causal compiler, but it stores
several exact single-token sites for every parsed occurrence.  This module
joins those archived states to the frozen causal ``event_registry.csv`` and
keeps an event only when the archived ``item_end`` token is exactly the causal
``commit_output_token``.  The resulting dataset can therefore be re-analysed
on CPU without pretending that an unavailable site (notably ``post_marker``)
was captured.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from realistic_niah_v5.cross_mode_geometry import ModeDataset


SCHEMA_VERSION = "realistic_niah_v5_causal_aligned_capture_v1"
PRIMARY_PROGRESS_FILTER = {
    "primary_full_chain_event": True,
    "progress_commit_eligible": True,
    "progress_commit_site_resolved": True,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value).strip()


def _metadata_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["request_id"].astype(str)
        + "::"
        + frame["occurrence"].astype(int).astype(str)
    )


def metadata_key_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(_metadata_key(frame).tolist())
    return hashlib.sha256(("\n".join(keys) + "\n").encode("utf-8")).hexdigest()


def _event_rows(
    event_registry: Path,
    *,
    model_label: str,
) -> pd.DataFrame:
    events = pd.read_csv(event_registry)
    events = events.loc[events["model_label"].astype(str).eq(model_label)].copy()
    for column, expected in PRIMARY_PROGRESS_FILTER.items():
        events = events.loc[events[column].map(_as_bool).eq(expected)]
    required = [
        "request_id",
        "occurrence",
        "commit_output_token",
        "grammar_class",
        "city",
        "split",
        "seed",
        "gold_count",
    ]
    missing = sorted(set(required) - set(events.columns))
    if missing:
        raise ValueError(f"Causal event registry is missing columns: {missing}")
    events = events.dropna(subset=["commit_output_token"]).copy()
    events["occurrence"] = events["occurrence"].astype(int)
    events["commit_output_token"] = events["commit_output_token"].astype(int)
    duplicate = events.duplicated(["request_id", "occurrence"], keep=False)
    if duplicate.any():
        examples = events.loc[duplicate, ["request_id", "occurrence"]].head(8)
        raise ValueError(
            "Causal primary progress events are not unique:\n"
            + examples.to_string(index=False)
        )
    return events.sort_values(
        ["split", "seed", "gold_count", "request_id", "occurrence"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_causal_aligned_native_capture(
    capture_index: str | Path,
    event_registry: str | Path,
    *,
    site_kind: str = "item_end",
) -> tuple[ModeDataset, dict[str, Any]]:
    """Return one archived site on exact causal progress-commit events.

    ``item_end`` is the archived name of the same token used as
    ``p0_item_end`` by the causal compiler.  Other archived sites are retained
    as same-event controls, but every row must first pass the item-end/commit
    token equality audit.
    """

    index_path = Path(capture_index)
    event_path = Path(event_registry)
    index_rows = _read_jsonl(index_path)
    if not index_rows:
        raise ValueError(f"Empty capture index: {index_path}")
    model_labels = {str(row["model_label"]) for row in index_rows}
    if len(model_labels) != 1:
        raise ValueError(f"Capture index mixes model labels: {sorted(model_labels)}")
    model_label = next(iter(model_labels))
    events = _event_rows(event_path, model_label=model_label)
    events_by_request = {
        request_id: frame.to_dict(orient="records")
        for request_id, frame in events.groupby("request_id", sort=False)
    }

    counters: Counter[str] = Counter()
    descriptors: list[tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]] = []
    seen_event_keys: set[tuple[str, int]] = set()
    for index_row in index_rows:
        request_id = str(index_row["request_id"])
        request_events = events_by_request.get(request_id)
        if not request_events:
            continue
        manifest_path = index_path.parent / str(index_row["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sites: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
        for site_index, site in enumerate(manifest["site_rows"]):
            occurrence = site.get("occurrence")
            if occurrence is None:
                continue
            key = (str(site.get("site_kind")), int(occurrence))
            if key in sites:
                raise ValueError(f"Duplicate archived site {key} in {manifest_path}")
            sites[key] = (site_index, site)

        for event in request_events:
            occurrence = int(event["occurrence"])
            event_key = (request_id, occurrence)
            commit_pair = sites.get(("item_end", occurrence))
            if commit_pair is None:
                counters["missing_archived_item_end"] += 1
                continue
            _commit_index, commit_site = commit_pair
            archived_commit = int(commit_site["endpoint_token"])
            causal_commit = int(event["commit_output_token"])
            if archived_commit != causal_commit:
                counters["commit_token_mismatch"] += 1
                continue
            archived_city = _clean_text(commit_site.get("city"))
            causal_city = _clean_text(event.get("city"))
            if archived_city != causal_city:
                counters["city_mismatch"] += 1
                continue
            target_pair = sites.get((site_kind, occurrence))
            if target_pair is None:
                counters[f"missing_archived_{site_kind}"] += 1
                continue
            target_index, target_site = target_pair
            descriptors.append((index_row, event, target_index, target_site))
            seen_event_keys.add(event_key)

    counters["registry_events_without_exact_archived_match"] = int(
        len(events) - len(seen_event_keys)
    )
    descriptors.sort(
        key=lambda item: (
            0 if str(item[1]["split"]) == "discovery" else 1,
            int(item[1]["seed"]),
            int(item[1]["gold_count"]),
            int(item[1]["occurrence"]),
            str(item[1]["request_id"]),
        )
    )
    if not descriptors:
        raise ValueError(
            f"No exact causal-aligned {site_kind!r} states in {index_path}"
        )

    first_row, _first_event, _first_site_index, _first_site = descriptors[0]
    first_states_path = index_path.parent / str(first_row["states_path"])
    with np.load(first_states_path, allow_pickle=False) as archive:
        layer_indices = archive["layer_indices"].astype(int)
        hidden_size = int(archive["site_states"].shape[-1])
    states_by_layer = {
        int(layer): np.empty((len(descriptors), hidden_size), dtype=np.float16)
        for layer in layer_indices
    }
    metadata_rows: list[dict[str, Any]] = []
    for row_offset, (index_row, event, site_index, target_site) in enumerate(
        descriptors
    ):
        states_path = index_path.parent / str(index_row["states_path"])
        with np.load(states_path, allow_pickle=False) as archive:
            observed_layers = archive["layer_indices"].astype(int)
            if not np.array_equal(observed_layers, layer_indices):
                raise ValueError(f"Layer mismatch in {states_path}")
            site_states = np.asarray(archive["site_states"])[int(site_index)]
            for layer_axis, layer in enumerate(layer_indices):
                states_by_layer[int(layer)][row_offset] = site_states[layer_axis]
        metadata_rows.append(
            {
                "split": str(event["split"]),
                "seed": int(event["seed"]),
                "occurrence": int(event["occurrence"]),
                "gold_count": int(event["gold_count"]),
                "stimulus_id": str(index_row["stimulus_id"]),
                "request_id": str(event["request_id"]),
                "sequence_source": event.get("sequence_source"),
                "trace_category": event.get("trace_category"),
                "causal_cohort": event.get("causal_cohort"),
                "grammar_class": event.get("grammar_class"),
                "exact_count": _as_bool(event.get("exact_count")),
                "selected_site_kind": site_kind,
                "site_output_token": int(target_site["endpoint_token"]),
                "commit_output_token": int(event["commit_output_token"]),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    dataset = ModeDataset(
        mode="native_thinking_causal_aligned",
        model_label=model_label,
        metadata=metadata,
        states_by_layer=states_by_layer,
    )
    dataset.validate()

    support = (
        metadata.groupby(["split", "occurrence"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "model_label": model_label,
        "site_kind": site_kind,
        "causal_event_filter": dict(PRIMARY_PROGRESS_FILTER),
        "causal_registry_event_count": int(len(events)),
        "matched_state_count": int(len(metadata)),
        "matched_trajectory_count": int(metadata["request_id"].nunique()),
        "metadata_key_sha256": metadata_key_sha256(metadata),
        "layers": [int(value) for value in layer_indices],
        "support": support.to_dict(orient="records"),
        "exclusions": dict(sorted(counters.items())),
        "token_equivalence": (
            "archived item_end endpoint_token equals causal commit_output_token "
            "for every retained row"
        ),
    }
    return dataset, audit


def common_metadata_keys(datasets: Iterable[ModeDataset]) -> set[str]:
    values = [set(_metadata_key(dataset.metadata)) for dataset in datasets]
    if not values:
        return set()
    return set.intersection(*values)
