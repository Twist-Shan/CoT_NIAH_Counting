from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from realistic_niah.entity_domains import resolve_entity_domain


STIMULUS_SCHEMA_VERSION = "realistic_niah_entity_domain_transfer_stimulus_v1"
STIMULUS_MANIFEST_SCHEMA_VERSION = (
    "realistic_niah_entity_domain_transfer_manifest_v1"
)
NONTHINKING_CAPTURE_SCHEMA_VERSION = (
    "realistic_niah_entity_domain_transfer_nonthinking_capture_v1"
)
SITE_INDEX_SCHEMA_VERSION = "realistic_niah_domain_transfer_site_index_v1"
DEFAULT_CONFIRMATION_SEEDS = tuple(range(1254, 1264))
DEFAULT_COUNTS = tuple(range(1, 11))

_CITY_RECORD_RE = re.compile(
    r"In the 2024 city score audit, (?P<entity>.+?) received a score of "
    r"(?P<score>\d+)\."
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL line {line_number} in {path}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def write_capture_site_index(capture_dir: str | Path) -> tuple[Path, Path]:
    """Flatten per-shard capture sites into a reusable top-level catalog.

    ``site_states`` remains shard-local so the catalog is small.  Each catalog
    row records the NPZ path and the exact first-axis index needed to retrieve
    that site's all-layer hidden states.  Native trajectories are intentionally
    ragged: every parser-observed ``item_end`` gets a row, including duplicate
    or partial counting paths, without padding or one-to-one filtering.
    """

    root = Path(capture_dir)
    capture_index_path = root / "capture_index.jsonl"
    capture_rows = read_jsonl(capture_index_path)
    site_rows: list[dict[str, Any]] = []
    layer_registries: dict[str, list[int]] = {}
    for capture_row in capture_rows:
        manifest_path = root / str(capture_row["manifest_path"])
        states_path = root / str(capture_row["states_path"])
        if not manifest_path.is_file() or not states_path.is_file():
            raise FileNotFoundError(
                f"Capture shard is incomplete: {manifest_path} / {states_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_sites = list(manifest.get("site_rows", ()))
        state_shape = [int(value) for value in manifest["site_states_shape"]]
        if len(state_shape) != 3 or state_shape[0] != len(shard_sites):
            raise ValueError(
                f"Site metadata/state shape mismatch in {manifest_path}: "
                f"sites={len(shard_sites)} shape={state_shape}"
            )
        layers = [int(value) for value in manifest["layers"]]
        if state_shape[1] != len(layers):
            raise ValueError(
                f"Layer metadata/state shape mismatch in {manifest_path}: "
                f"layers={len(layers)} shape={state_shape}"
            )
        model_label = str(capture_row["model_label"])
        previous_layers = layer_registries.setdefault(model_label, layers)
        if previous_layers != layers:
            raise ValueError(f"Inconsistent layer registry for {model_label}")
        for inferred_axis, raw_site in enumerate(shard_sites):
            site = dict(raw_site)
            state_axis = int(site.pop("state_axis", inferred_axis))
            if state_axis != inferred_axis:
                raise ValueError(
                    f"Non-contiguous state_axis in {manifest_path}: "
                    f"expected {inferred_axis}, found {state_axis}"
                )
            site_rows.append(
                {
                    **site,
                    "schema_version": SITE_INDEX_SCHEMA_VERSION,
                    "capture_row_index": int(capture_row["row_index"]),
                    "request_id": capture_row.get("request_id"),
                    "stimulus_id": str(capture_row["stimulus_id"]),
                    "source_stimulus_id": capture_row.get("source_stimulus_id"),
                    "model_label": model_label,
                    "mode": str(capture_row["mode"]),
                    "entity_domain": str(capture_row["entity_domain"]),
                    "seed": int(capture_row["seed"]),
                    "split": str(capture_row["split"]),
                    "gold_count": int(capture_row["gold_count"]),
                    "manifest_path": str(capture_row["manifest_path"]),
                    "states_path": str(capture_row["states_path"]),
                    "state_array_key": "site_states",
                    "state_axis": state_axis,
                    "layer_array_key": "layer_indices",
                }
            )

    site_index_path = root / "site_index.jsonl"
    _atomic_jsonl(site_index_path, site_rows)
    site_kinds = sorted({str(row["site_kind"]) for row in site_rows})
    site_manifest_path = root / "site_index_manifest.json"
    _atomic_json(
        site_manifest_path,
        {
            "schema_version": SITE_INDEX_SCHEMA_VERSION,
            "capture_index": capture_index_path.name,
            "site_index": site_index_path.name,
            "site_rows": len(site_rows),
            "running_site_rows": sum(
                str(row["site_kind"]) in {"running_index", "item_end"}
                for row in site_rows
            ),
            "answer_site_rows": sum(
                str(row["site_kind"]) in {"answer_query", "answer_query_v3"}
                for row in site_rows
            ),
            "site_kinds": site_kinds,
            "layer_registries": layer_registries,
            "state_lookup": (
                "np.load(capture_root / states_path)[state_array_key]"
                "[state_axis, layer_axis, :]"
            ),
            "native_running_policy": (
                "all parser-observed item_end sites; ragged paths and duplicates retained"
            ),
            "nonthinking_running_policy": (
                "one prompt active-record span-end per registered occurrence"
            ),
        },
    )
    return site_index_path, site_manifest_path


def _stable_domain_seed(domain: str, seed: int) -> int:
    digest = hashlib.sha256(
        f"niah-domain-transfer-v1:{domain}:{int(seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def entity_panel(entity_domain: str, seed: int) -> tuple[str, ...]:
    domain = resolve_entity_domain(entity_domain)
    if domain.name == "city":
        raise ValueError("City is the source condition, not a transfer name panel")
    if len(domain.entity_names) < max(DEFAULT_COUNTS):
        raise RuntimeError(f"Domain {domain.name} has too few registered entities")
    values = list(domain.entity_names)
    random.Random(_stable_domain_seed(domain.name, int(seed))).shuffle(values)
    return tuple(values[: max(DEFAULT_COUNTS)])


def _domain_record_text(
    source_text: str,
    *,
    entity_domain: str,
    entity: str,
    score: int,
) -> str:
    domain = resolve_entity_domain(entity_domain)
    matches = list(_CITY_RECORD_RE.finditer(source_text))
    if len(matches) != 1:
        raise ValueError(
            "Each active source span must contain exactly one city-score record"
        )
    match = matches[0]
    if int(match.group("score")) != int(score):
        raise ValueError("Source record score disagrees with span metadata")
    return (
        source_text[: match.start()]
        + domain.record_sentence(entity, int(score))
        + source_text[match.end() :]
    )


def _shift_outside_replacements(
    position: int,
    replacements: Sequence[tuple[int, int, str]],
) -> int:
    shift = 0
    for start, end, new_text in replacements:
        if int(position) < int(start):
            break
        if int(start) <= int(position) < int(end):
            raise ValueError(
                f"Position {position} lies inside a replaced active span"
            )
        shift += len(new_text) - (int(end) - int(start))
    return int(position) + shift


def transform_stimulus(
    source: Mapping[str, Any],
    *,
    entity_domain: str,
) -> dict[str, Any]:
    """Replace active city records while preserving the frozen V4.4 backbone.

    Scores, haystack text, source seed, source count, and record insertion order
    are held fixed. Character offsets are recomputed after replacement. Legacy
    ``city`` fields retain arbitrary entity labels so the existing V5 parser can
    use its audited entity-registry path without a second parser implementation.
    """

    domain = resolve_entity_domain(entity_domain)
    if domain.name == "city":
        raise ValueError("transform_stimulus expects flower or animal")
    if str(source.get("design_variant")) != "v4.4":
        raise ValueError("Domain transfer is fixed to the V4.4 backbone")
    if str(source.get("split")) != "confirmation":
        raise ValueError("Domain transfer uses the held-out confirmation panel")
    seed = int(source["seed"])
    gold_count = int(source["gold_count"])
    if seed not in DEFAULT_CONFIRMATION_SEEDS or gold_count not in DEFAULT_COUNTS:
        raise ValueError("Source row is outside the registered 10 x 10 panel")

    names = entity_panel(domain.name, seed)
    name_by_slot = {index + 1: name for index, name in enumerate(names)}
    passage = str(source["passage"])
    source_active = sorted(
        (dict(value) for value in source["active_needle_spans"]),
        key=lambda value: int(value["char_start"]),
    )
    if len(source_active) != gold_count:
        raise ValueError("Source active-span count disagrees with gold_count")

    replacements: list[tuple[int, int, str]] = []
    replacement_by_slot: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    pieces: list[str] = []
    new_cursor = 0
    for active in source_active:
        start = int(active["char_start"])
        end = int(active["char_end"])
        slot_index = int(active["slot_index"])
        if start < cursor or not 0 <= start < end <= len(passage):
            raise ValueError("Source active spans overlap or are out of bounds")
        source_text = passage[start:end]
        if source_text != str(active["text"]):
            raise ValueError("Source active span text does not match the passage")
        new_text = _domain_record_text(
            source_text,
            entity_domain=domain.name,
            entity=name_by_slot[slot_index],
            score=int(active["score"]),
        )
        unchanged = passage[cursor:start]
        pieces.extend((unchanged, new_text))
        new_cursor += len(unchanged)
        new_start = new_cursor
        new_end = new_start + len(new_text)
        replacement_by_slot[slot_index] = (new_start, new_end, new_text)
        new_cursor = new_end
        replacements.append((start, end, new_text))
        cursor = end
    pieces.append(passage[cursor:])
    transformed_passage = "".join(pieces)

    result = copy.deepcopy(dict(source))
    result["schema_version"] = STIMULUS_SCHEMA_VERSION
    result["source_schema_version"] = source.get("schema_version")
    result["source_stimulus_id"] = str(source["stimulus_id"])
    result["stimulus_id"] = (
        f"DT_{domain.name.upper()}_V4_4_T10000_N{gold_count}_seed{seed}"
    )
    result["entity_domain"] = domain.name
    result["entity_singular"] = domain.singular
    result["entity_plural"] = domain.plural
    result["entity_panel"] = list(names)
    result["passage"] = transformed_passage
    result["passage_sha256"] = _sha256_text(transformed_passage)
    result["source_passage_sha256"] = source.get("passage_sha256")
    result["source_target_passage_tokens"] = int(source["target_passage_tokens"])
    result["canonical_offsets_recomputed"] = False
    result["character_offsets_recomputed"] = True
    result["domain_transfer_protocol"] = {
        "scores_held_fixed": True,
        "haystack_held_fixed": True,
        "source_seed_held_fixed": True,
        "source_count_held_fixed": True,
        "active_record_positions_held_approximately": True,
        "entity_names_deterministic_by_domain_and_seed": True,
        "legacy_entity_key": "city",
    }

    slots: list[dict[str, Any]] = []
    for raw_slot in source["slots"]:
        slot = copy.deepcopy(dict(raw_slot))
        slot_index = int(slot["slot_index"])
        slot["city"] = name_by_slot[slot_index]
        slot["entity"] = name_by_slot[slot_index]
        slot["entity_domain"] = domain.name
        for key in (
            "canonical_span_start",
            "canonical_span_end",
            "canonical_token_length",
        ):
            slot.pop(key, None)
        if bool(slot["active"]):
            new_start, new_end, new_text = replacement_by_slot[slot_index]
            slot["char_start"] = new_start
            slot["char_end"] = new_end
            slot["text"] = new_text
            slot["content_text"] = new_text
            slot["text_sha256"] = _sha256_text(new_text)
        else:
            slot["char_start"] = _shift_outside_replacements(
                int(slot["char_start"]), replacements
            )
            slot["char_end"] = _shift_outside_replacements(
                int(slot["char_end"]), replacements
            )
        slots.append(slot)
    result["slots"] = slots

    active_rows: list[dict[str, Any]] = []
    for source_row in sorted(source_active, key=lambda value: int(value["slot_index"])):
        slot_index = int(source_row["slot_index"])
        new_start, new_end, new_text = replacement_by_slot[slot_index]
        active = {
            key: copy.deepcopy(value)
            for key, value in source_row.items()
            if key
            not in {
                "canonical_span_start",
                "canonical_span_end",
                "canonical_token_length",
            }
        }
        active.update(
            {
                "char_start": new_start,
                "char_end": new_end,
                "city": name_by_slot[slot_index],
                "entity": name_by_slot[slot_index],
                "entity_domain": domain.name,
                "text": new_text,
                "text_sha256": _sha256_text(new_text),
            }
        )
        active_rows.append(active)
    result["active_needle_spans"] = active_rows
    result["gold_pairs"] = [
        {
            "slot_index": int(active["slot_index"]),
            "city": str(active["city"]),
            "entity": str(active["entity"]),
            "entity_domain": domain.name,
            "score": int(active["score"]),
        }
        for active in active_rows
    ]

    hard_negatives: list[dict[str, Any]] = []
    for raw_negative in source.get("hard_negative_spans", ()):
        negative = copy.deepcopy(dict(raw_negative))
        negative["char_start"] = _shift_outside_replacements(
            int(negative["char_start"]), replacements
        )
        negative["char_end"] = _shift_outside_replacements(
            int(negative["char_end"]), replacements
        )
        for key in (
            "canonical_span_start",
            "canonical_span_end",
            "canonical_token_length",
        ):
            negative.pop(key, None)
        hard_negatives.append(negative)
    result["hard_negative_spans"] = hard_negatives
    audit_transformed_stimulus(result)
    return result


def audit_transformed_stimulus(row: Mapping[str, Any]) -> None:
    domain = resolve_entity_domain(str(row.get("entity_domain")))
    if domain.name == "city":
        raise ValueError("A transfer stimulus cannot use the city domain")
    passage = str(row["passage"])
    gold_count = int(row["gold_count"])
    active = list(row["active_needle_spans"])
    if len(active) != gold_count or len(row["gold_pairs"]) != gold_count:
        raise ValueError("Transfer stimulus has an incomplete gold registry")
    if passage.count(f"In the 2024 {domain.singular} score audit,") != gold_count:
        raise ValueError("Transfer passage does not contain exactly gold_count records")
    for expected_slot, record in enumerate(active, start=1):
        if int(record["slot_index"]) != expected_slot:
            raise ValueError("Active transfer slots must be 1..N in order")
        start = int(record["char_start"])
        end = int(record["char_end"])
        if passage[start:end] != str(record["text"]):
            raise ValueError("A transformed active span is not character-aligned")
        entity = str(record["entity"])
        sentence = domain.record_sentence(entity, int(record["score"]))
        if str(record["text"]).count(sentence) != 1:
            raise ValueError("A transformed active span has the wrong sentence")
    if row["passage_sha256"] != _sha256_text(passage):
        raise ValueError("Transfer passage SHA-256 is stale")


def prepare_domain_transfer_panel(
    source_rows: Iterable[Mapping[str, Any]],
    *,
    entity_domains: Sequence[str] = ("flower", "animal"),
    seeds: Sequence[int] = DEFAULT_CONFIRMATION_SEEDS,
    counts: Sequence[int] = DEFAULT_COUNTS,
) -> list[dict[str, Any]]:
    seed_set = {int(value) for value in seeds}
    count_set = {int(value) for value in counts}
    source = [
        dict(row)
        for row in source_rows
        if str(row.get("design_variant")) == "v4.4"
        and str(row.get("split")) == "confirmation"
        and int(row.get("seed", -1)) in seed_set
        and int(row.get("gold_count", -1)) in count_set
    ]
    expected_cells = {(seed, count) for seed in seed_set for count in count_set}
    observed_cells = {(int(row["seed"]), int(row["gold_count"])) for row in source}
    if observed_cells != expected_cells or len(source) != len(expected_cells):
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(
            "Source V4.4 confirmation panel is not one-to-one over seed x count: "
            f"rows={len(source)} missing={missing[:5]} extra={extra[:5]}"
        )
    by_cell = {
        (int(row["seed"]), int(row["gold_count"])): row for row in source
    }
    output: list[dict[str, Any]] = []
    for entity_domain in entity_domains:
        domain = resolve_entity_domain(entity_domain)
        if domain.name == "city":
            raise ValueError("The output panel should contain transfer domains only")
        for seed in sorted(seed_set):
            for count in sorted(count_set):
                output.append(
                    transform_stimulus(
                        by_cell[(seed, count)], entity_domain=domain.name
                    )
                )
    return output


def write_domain_transfer_panel(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_path: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    domains = sorted({str(row["entity_domain"]) for row in rows})
    paths: dict[str, Path] = {}
    for domain in domains:
        selected = [dict(row) for row in rows if row["entity_domain"] == domain]
        path = output / f"{domain}_confirmation_100.jsonl"
        if len(selected) != 100:
            raise ValueError(f"Expected 100 {domain} rows, found {len(selected)}")
        _atomic_jsonl(path, selected)
        paths[domain] = path
    combined = output / "flower_animal_confirmation_200.jsonl"
    _atomic_jsonl(combined, rows)
    paths["combined"] = combined
    manifest = {
        "schema_version": STIMULUS_MANIFEST_SCHEMA_VERSION,
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "domains": domains,
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "counts": sorted({int(row["gold_count"]) for row in rows}),
        "rows": len(rows),
        "rows_per_domain": {
            domain: sum(row["entity_domain"] == domain for row in rows)
            for domain in domains
        },
        "same_source_cell_across_domains": True,
        "scores_held_fixed_across_domains": True,
        "haystack_held_fixed_across_domains": True,
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    paths["manifest"] = manifest_path
    return paths


def _should_regenerate_censored(
    existing: Mapping[str, Any] | None,
    decoding: Any,
) -> bool:
    """Return true only when a larger ceiling can repair a censored shard."""

    return bool(
        existing is not None
        and existing.get("generation_truncated")
        and int(existing.get("output_tokens", 0)) < int(decoding.max_new_tokens)
    )


def generate_native_domain_shards(
    model: Any,
    tokenizer: Any,
    stimuli: Iterable[Mapping[str, Any]],
    *,
    model_spec: Any,
    decoding: Any,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Greedily generate native-thinking traces as restartable JSON shards."""

    from realistic_niah_v5.generation import (
        generate_native_trace,
        render_native_prompt,
    )

    output = Path(output_dir)
    ordered_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    for row_index, raw_stimulus in enumerate(stimuli):
        stimulus = dict(raw_stimulus)
        audit_transformed_stimulus(stimulus)
        safe_id = f"{model_spec.label}__native-thinking__{stimulus['stimulus_id']}"
        relative = Path("shards") / f"{safe_id}.json"
        shard_path = output / relative
        existing: dict[str, Any] | None = None
        if shard_path.exists():
            existing = json.loads(shard_path.read_text(encoding="utf-8"))
        regenerate_censored = _should_regenerate_censored(existing, decoding)
        if existing is not None and not overwrite and not regenerate_censored:
            generated = existing
            if generated.get("stimulus_id") != stimulus["stimulus_id"]:
                raise RuntimeError(f"Incompatible generation shard: {shard_path}")
            if generated.get("model_label") != model_spec.label:
                raise RuntimeError(f"Generation model mismatch: {shard_path}")
        else:
            prompt = render_native_prompt(
                stimulus,
                tokenizer=tokenizer,
                model_spec=model_spec,
            )
            generated = generate_native_trace(
                model,
                tokenizer,
                prompt,
                decoding=decoding,
                sampling_seed=int(stimulus["seed"]),
            )
            generated["source_stimulus_id"] = stimulus.get("source_stimulus_id")
            generated["domain_transfer_schema_version"] = STIMULUS_SCHEMA_VERSION
            if regenerate_censored and existing is not None:
                generated["generation_rescue"] = {
                    "reason": "previous_max_new_tokens_censoring",
                    "previous_output_tokens": int(existing["output_tokens"]),
                    "new_max_new_tokens": int(decoding.max_new_tokens),
                    "prompt_and_greedy_decoding_unchanged": True,
                }
            _atomic_json(shard_path, generated)
        ordered_rows.append(generated)
        parser = dict(generated.get("trace_parse", {}).get("parser", {}))
        index_rows.append(
            {
                "row_index": row_index,
                "stimulus_id": generated["stimulus_id"],
                "source_stimulus_id": generated.get("source_stimulus_id"),
                "model_label": generated["model_label"],
                "mode": "native-thinking",
                "entity_domain": generated.get("entity_domain"),
                "seed": int(generated["seed"]),
                "split": generated["split"],
                "gold_count": int(generated["gold_count"]),
                "output_tokens": int(generated["output_tokens"]),
                "generation_truncated": bool(generated["generation_truncated"]),
                "parsed_count": generated.get("trace_parse", {}).get(
                    "parsed_count"
                ),
                "exact_count": generated.get("trace_parse", {}).get("exact_count"),
                "trace_category": parser.get("trace_category"),
                "marker_kind": parser.get("marker_kind"),
                "trace_item_count": int(parser.get("item_count", 0)),
                "shard_path": relative.as_posix(),
            }
        )
        print(
            "[domain-transfer native generation] "
            f"{row_index + 1} model={model_spec.label} "
            f"domain={generated.get('entity_domain')} "
            f"count={generated['gold_count']} tokens={generated['output_tokens']} "
            f"trace={parser.get('trace_category')}",
            flush=True,
        )
    if not ordered_rows:
        raise ValueError("No domain-transfer stimuli were supplied")
    generations_path = output / "generations.jsonl"
    _atomic_jsonl(generations_path, ordered_rows)
    _atomic_jsonl(output / "generation_index.jsonl", index_rows)
    _atomic_json(
        output / "generation_manifest.json",
        {
            "schema_version": "realistic_niah_domain_transfer_generation_v1",
            "rows": len(index_rows),
            "model_label": model_spec.label,
            "model_id": model_spec.model_id,
            "mode": "native-thinking",
            "entity_domains": sorted(
                {str(row["entity_domain"]) for row in index_rows}
            ),
            "seeds": sorted({int(row["seed"]) for row in index_rows}),
            "counts": sorted({int(row["gold_count"]) for row in index_rows}),
            "decoding": {
                key: getattr(decoding, key)
                for key in (
                    "max_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                )
            },
            "truncated_rows": sum(
                bool(row["generation_truncated"]) for row in index_rows
            ),
            "rescued_truncated_rows": sum(
                bool(row.get("generation_rescue")) for row in ordered_rows
            ),
            "exact_count_rows": sum(bool(row["exact_count"]) for row in index_rows),
            "trace_category_counts": {
                category: sum(row["trace_category"] == category for row in index_rows)
                for category in sorted(
                    {str(row["trace_category"]) for row in index_rows}
                )
            },
            "restartable_shards": True,
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )
    return generations_path


def capture_native_domain_shards(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    records: Iterable[Mapping[str, Any]],
    *,
    config: Any,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Capture native item-end and answer-query-v3 states with exclusions."""

    from realistic_niah_v5.capture import capture_trace_record

    output = Path(output_dir)
    selected_site_kinds = ("item_end", "answer_query_v3")
    index_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    for row_index, raw_row in enumerate(records):
        row = dict(raw_row)
        domain = resolve_entity_domain(str(row.get("entity_domain")))
        request_id = str(row.get("request_id", row.get("stimulus_id", row_index)))
        safe_id = request_id.replace("/", "__").replace("\\", "__")
        relative = Path("shards") / safe_id
        try:
            manifest = capture_trace_record(
                model,
                adapter,
                tokenizer,
                row,
                config=config,
                output_dir=output / relative,
                layers=layers,
                site_kinds=selected_site_kinds,
                capture_span_pooling=False,
                overwrite=overwrite,
            )
        except ValueError as error:
            if str(error) != "No aligned registered V5 trace sites":
                raise
            exclusion = {
                "row_index": row_index,
                "request_id": request_id,
                "stimulus_id": row.get("stimulus_id"),
                "model_label": row.get("model_label"),
                "mode": "native-thinking",
                "entity_domain": domain.name,
                "seed": row.get("seed"),
                "gold_count": row.get("gold_count"),
                "reason": str(error),
            }
            exclusions.append(exclusion)
            print(
                "[domain-transfer native capture exclusion] "
                f"{row_index + 1} {request_id} reason={error}",
                flush=True,
            )
            continue
        site_rows = list(manifest["site_rows"])
        running_count = sum(
            str(site.get("site_kind")) == "item_end" for site in site_rows
        )
        answer_count = sum(
            str(site.get("site_kind")) == "answer_query_v3" for site in site_rows
        )
        index_rows.append(
            {
                "schema_version": manifest["schema_version"],
                "row_index": row_index,
                "request_id": request_id,
                "stimulus_id": manifest["stimulus_id"],
                "source_stimulus_id": row.get("source_stimulus_id"),
                "model_label": manifest["model_label"],
                "mode": "native-thinking",
                "entity_domain": manifest.get("entity_domain", domain.name),
                "seed": manifest["seed"],
                "split": manifest["split"],
                "gold_count": manifest["gold_count"],
                "parsed_count": manifest["parsed_count"],
                "exact_count": manifest["exact_count"],
                "trace_one_to_one": manifest["parser"]["trace_one_to_one"],
                "trace_category": manifest["parser"]["trace_category"],
                "marker_kind": manifest["parser"]["marker_kind"],
                "trace_item_count": int(manifest["parser"]["item_count"]),
                "output_tokens": int(row.get("output_tokens", 0)),
                "generation_truncated": bool(row.get("generation_truncated")),
                "generation_rescue": row.get("generation_rescue"),
                "running_site_count": int(running_count),
                "answer_site_count": int(answer_count),
                "sequence_source": manifest["sequence_source"],
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            "[domain-transfer native capture] "
            f"{row_index + 1} model={manifest['model_label']} "
            f"domain={domain.name} count={manifest['gold_count']} "
            f"running={running_count} answer={answer_count} "
            f"elapsed={float(manifest['elapsed_seconds']):.2f}s",
            flush=True,
        )
    if not index_rows and not exclusions:
        raise ValueError("No native domain-transfer records were supplied")
    index_path = output / "capture_index.jsonl"
    _atomic_jsonl(index_path, index_rows)
    _atomic_jsonl(output / "capture_exclusions.jsonl", exclusions)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": "realistic_niah_domain_transfer_native_capture_v1",
            "rows_requested": len(index_rows) + len(exclusions),
            "rows_captured": len(index_rows),
            "rows_excluded": len(exclusions),
            "model_labels": sorted({str(row["model_label"]) for row in index_rows}),
            "mode": "native-thinking",
            "entity_domains": sorted(
                {str(row["entity_domain"]) for row in index_rows}
            ),
            "selected_site_kinds": list(selected_site_kinds),
            "running_site_semantics": "parsed native-thinking item-end",
            "answer_site_semantics": (
                "last literal output token immediately before the final numeric answer"
            ),
            "total_running_states": sum(
                int(row["running_site_count"]) for row in index_rows
            ),
            "total_answer_states": sum(
                int(row["answer_site_count"]) for row in index_rows
            ),
            "generation_rescue_rows": sum(
                bool(row.get("generation_rescue")) for row in index_rows
            ),
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )
    write_capture_site_index(output)
    return index_path


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _candidate_prediction(
    logits: Any,
    encoding: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    values = logits.detach().float().cpu()
    top_token = int(values.argmax().item())
    answer_rows = dict(encoding.count_candidate_answer_token_ids)
    first_tokens = {
        int(count): int(token_ids[0]) for count, token_ids in answer_rows.items()
    }
    unique = len(set(first_tokens.values())) == len(first_tokens)
    single_token = all(len(token_ids) == 1 for token_ids in answer_rows.values())
    inverse = {token: count for count, token in first_tokens.items()}
    predicted_count = inverse.get(top_token) if unique else None
    candidate_counts = sorted(first_tokens)
    return {
        "next_token_id": top_token,
        "next_token_text": tokenizer.decode(
            [top_token],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "candidate_count": predicted_count,
        "candidate_first_tokens_unique": unique,
        "all_candidate_answers_single_token": single_token,
        "candidate_counts": candidate_counts,
        "candidate_first_token_ids": [first_tokens[count] for count in candidate_counts],
        "candidate_first_token_logits": [
            float(values[first_tokens[count]].item()) for count in candidate_counts
        ],
    }


def capture_nonthinking_domain_shards(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    stimuli: Iterable[Mapping[str, Any]],
    *,
    model_spec: Any,
    config: Any,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    save_dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Capture prompt running-index and final-answer states in one prefill.

    Each shard stores all post-block decoder layers at the final token
    overlapping every active record span plus the prompt-final ``Total:``
    endpoint. The exact model input IDs are retained for later audits.
    """

    from realistic_niah_v4.modeling import capture_post_block_states
    from realistic_niah_v4.prompts import render_v4_prompt

    dtype = np.dtype(str(save_dtype))
    if dtype not in {np.dtype("float16"), np.dtype("float32")}:
        raise ValueError("save_dtype must be float16 or float32")
    selected_layers = (
        tuple(range(int(adapter.num_layers)))
        if layers is None
        else tuple(sorted({int(value) for value in layers}))
    )
    if not selected_layers:
        raise ValueError("At least one layer is required")
    if any(not 0 <= layer < int(adapter.num_layers) for layer in selected_layers):
        raise ValueError("A requested layer is outside the decoder")

    output = Path(output_dir)
    index_rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    for row_index, raw_stimulus in enumerate(stimuli):
        stimulus = dict(raw_stimulus)
        audit_transformed_stimulus(stimulus)
        encoding = render_v4_prompt(
            stimulus,
            tokenizer=tokenizer,
            model_spec=model_spec,
            config=config,
            answer_format="numeric",
        )
        domain = str(stimulus["entity_domain"])
        safe_id = f"{model_spec.label}__nonthinking__{encoding.stimulus_id}"
        relative = Path("shards") / safe_id
        shard_dir = output / relative
        state_path = shard_dir / "states.npz"
        manifest_path = shard_dir / "capture_manifest.json"
        if state_path.exists() and manifest_path.exists() and not overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != NONTHINKING_CAPTURE_SCHEMA_VERSION:
                raise RuntimeError(f"Incompatible existing shard: {manifest_path}")
            if manifest.get("layers") != list(selected_layers):
                raise RuntimeError(f"Existing layer grid differs: {manifest_path}")
        else:
            running_positions = [span.end - 1 for span in encoding.needle_spans]
            positions = [*running_positions, int(encoding.query_position)]
            started = time.perf_counter()
            logits, captured = capture_post_block_states(
                model,
                adapter,
                encoding,
                positions,
                layers=selected_layers,
            )
            state_values = np.stack(
                [captured[layer].numpy() for layer in selected_layers], axis=1
            ).astype(dtype, copy=False)
            if state_values.shape[:2] != (
                len(positions),
                len(selected_layers),
            ) or not np.isfinite(state_values).all():
                raise RuntimeError("Invalid non-thinking domain-transfer states")
            prediction = _candidate_prediction(logits, encoding, tokenizer)
            active = sorted(
                stimulus["active_needle_spans"],
                key=lambda value: int(value["slot_index"]),
            )
            site_rows = [
                {
                    "site_id": f"running_index:{occurrence}",
                    "site_kind": "running_index",
                    "occurrence": occurrence,
                    "entity": str(record["entity"]),
                    "legacy_city": str(record["city"]),
                    "score": int(record["score"]),
                    "token_position": int(running_positions[occurrence - 1]),
                    "token_endpoint_exclusive": int(
                        encoding.needle_spans[occurrence - 1].end
                    ),
                    "boundary_kind": "prompt_active_record_span_end",
                    "state_axis": occurrence - 1,
                }
                for occurrence, record in enumerate(active, start=1)
            ]
            site_rows.append(
                {
                    "site_id": "answer_query",
                    "site_kind": "answer_query",
                    "occurrence": None,
                    "entity": None,
                    "score": None,
                    "token_position": int(encoding.query_position),
                    "token_endpoint_exclusive": int(encoding.query_position + 1),
                    "boundary_kind": "prompt_final_total_colon",
                    "state_axis": len(running_positions),
                }
            )
            _save_npz(
                state_path,
                layer_indices=np.asarray(selected_layers, dtype=np.int64),
                site_states=state_values,
                input_ids=np.asarray(encoding.input_ids, dtype=np.int64),
                attention_mask=np.asarray(encoding.attention_mask, dtype=np.int8),
                candidate_counts=np.asarray(
                    prediction["candidate_counts"], dtype=np.int64
                ),
                candidate_first_token_ids=np.asarray(
                    prediction["candidate_first_token_ids"], dtype=np.int64
                ),
                candidate_first_token_logits=np.asarray(
                    prediction["candidate_first_token_logits"], dtype=np.float32
                ),
            )
            manifest = {
                "schema_version": NONTHINKING_CAPTURE_SCHEMA_VERSION,
                "stimulus_id": encoding.stimulus_id,
                "source_stimulus_id": stimulus.get("source_stimulus_id"),
                "model_label": model_spec.label,
                "model_id": model_spec.model_id,
                "mode": "non-thinking",
                "entity_domain": domain,
                "seed": int(encoding.seed),
                "split": str(encoding.split),
                "gold_count": int(encoding.count),
                "prompt_sha256": _sha256_text(encoding.text),
                "prompt_suffix": encoding.text[-240:],
                "sequence_length": int(encoding.sequence_length),
                "query_position": int(encoding.query_position),
                "layers": list(selected_layers),
                "site_rows": site_rows,
                "site_states_shape": list(state_values.shape),
                "running_site_count": len(running_positions),
                "answer_site_count": 1,
                "save_dtype": str(dtype),
                "states_file": "states.npz",
                "prediction": {
                    key: value
                    for key, value in prediction.items()
                    if key
                    not in {
                        "candidate_counts",
                        "candidate_first_token_ids",
                        "candidate_first_token_logits",
                    }
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            _atomic_json(manifest_path, manifest)

        prediction = dict(manifest.get("prediction", {}))
        index_rows.append(
            {
                "schema_version": NONTHINKING_CAPTURE_SCHEMA_VERSION,
                "row_index": row_index,
                "stimulus_id": manifest["stimulus_id"],
                "source_stimulus_id": manifest.get("source_stimulus_id"),
                "model_label": manifest["model_label"],
                "mode": "non-thinking",
                "entity_domain": manifest["entity_domain"],
                "seed": int(manifest["seed"]),
                "split": manifest["split"],
                "gold_count": int(manifest["gold_count"]),
                "sequence_length": int(manifest["sequence_length"]),
                "running_site_count": int(manifest["running_site_count"]),
                "answer_site_count": int(manifest["answer_site_count"]),
                "candidate_count": prediction.get("candidate_count"),
                "candidate_count_correct": (
                    prediction.get("candidate_count") == int(manifest["gold_count"])
                    if prediction.get("candidate_count") is not None
                    else None
                ),
                "manifest_path": (relative / "capture_manifest.json").as_posix(),
                "states_path": (relative / "states.npz").as_posix(),
            }
        )
        print(
            "[domain-transfer non-thinking] "
            f"{row_index + 1} model={model_spec.label} domain={domain} "
            f"count={manifest['gold_count']} sites="
            f"{manifest['running_site_count'] + manifest['answer_site_count']} "
            f"elapsed={float(manifest['elapsed_seconds']):.2f}s",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No domain-transfer stimuli were supplied")
    index_path = output / "capture_index.jsonl"
    _atomic_jsonl(index_path, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": NONTHINKING_CAPTURE_SCHEMA_VERSION,
            "rows": len(index_rows),
            "model_labels": sorted({row["model_label"] for row in index_rows}),
            "modes": ["non-thinking"],
            "entity_domains": sorted(
                {row["entity_domain"] for row in index_rows}
            ),
            "seeds": sorted({int(row["seed"]) for row in index_rows}),
            "counts": sorted({int(row["gold_count"]) for row in index_rows}),
            "layers": list(selected_layers),
            "running_site_semantics": "prompt active-record span-end",
            "answer_site_semantics": "last token of teacher-forced Total:",
            "total_running_states": sum(
                int(row["running_site_count"]) for row in index_rows
            ),
            "total_answer_states": sum(
                int(row["answer_site_count"]) for row in index_rows
            ),
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
            "elapsed_seconds": time.perf_counter() - started_run,
        },
    )
    write_capture_site_index(output)
    return index_path
