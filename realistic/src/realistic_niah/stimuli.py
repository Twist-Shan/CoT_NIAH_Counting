from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from dataset_generation.dynamic_niah import TokenizerAdapter
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    generate_dynamic_niah_dataset_v2,
)

from .spec import (
    CANONICAL_TOKENIZER,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
)

CITY_SCORE_RE = re.compile(
    r"In the 2024 city score audit, (?P<city>.+?) "
    r"received a score of (?P<score>\d+)\."
)


@dataclass(frozen=True)
class FreezeSpec:
    passage_lengths: tuple[int, ...] = PASSAGE_LENGTHS
    needle_counts: tuple[int, ...] = NEEDLE_COUNTS
    seeds: tuple[int, ...] = SEEDS
    canonical_tokenizer: str = CANONICAL_TOKENIZER
    canonical_tokenizer_revision: str | None = None
    tokenizer_cache_dir: str | None = None
    max_search_attempts: int = 12
    max_window_retries: int = 4
    minimum_filler_tokens: int = 128
    haystack_source_mode: str = "multi_file_no_repeat"
    haystack_dir: str = "data/haystacks/paul_graham"
    haystack_corpus_manifest: str | None = None
    haystack_corpus_manifest_sha256: str | None = None
    insertion_depth_min_fraction: float = 0.0
    insertion_depth_max_fraction: float = 1.0


@dataclass(frozen=True)
class FreezeProtocol:
    protocol_version: str = "realistic_niah_v2"
    stimulus_schema_version: str = "realistic_niah_master_v1"
    manifest_schema_version: str = "realistic_niah_manifest_v1"
    audit_schema_version: str = "realistic_niah_audit_v1"
    stimulus_id_prefix: str = ""


V2_FREEZE_PROTOCOL = FreezeProtocol()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _compact_needle(
    needle: dict[str, Any],
    *,
    passage_characters: int,
) -> dict[str, Any]:
    record = needle["record"]
    span_start = int(needle["context_span_start"])
    span_end = int(needle["context_span_end"])
    return {
        "needle_id": needle["needle_id"],
        "text": needle["decoded_text"],
        "city": record["city"],
        "score": int(record["score"]),
        "standalone_needle_token_length": int(needle["token_length"]),
        "char_start": int(needle["char_start"]),
        "char_end": int(needle["char_end"]),
        "canonical_span_start": span_start,
        "canonical_span_end": span_end,
        "canonical_context_span_length": span_end - span_start,
        "token_span_alignment": needle.get(
            "token_span_alignment", "offset_mapping_overlap"
        ),
        "token_span_verified": bool(needle["token_span_verified"]),
        "normalized_depth": float(needle["char_start"])
        / max(1, passage_characters),
    }


def _assert_valid_candidate(
    row: dict[str, Any],
    *,
    target_passage_tokens: int,
    num_needles: int,
    tokenizer: TokenizerAdapter,
) -> None:
    passage = row["context"]
    actual_tokens = len(tokenizer.encode(passage))
    if actual_tokens != target_passage_tokens:
        raise ValueError(
            f"Passage length mismatch: {actual_tokens} != {target_passage_tokens}"
        )
    if len(row["relevant_records"]) != num_needles:
        raise ValueError("Gold record count does not match requested needle count")
    if len(row["realized_insertions"]) != num_needles:
        raise ValueError("Realized insertion count does not match requested count")

    cities = [str(item["city"]) for item in row["relevant_records"]]
    scores = [int(item["score"]) for item in row["relevant_records"]]
    if len(cities) != len(set(cities)):
        raise ValueError("Cities must be unique within a stimulus")
    if len(scores) != len(set(scores)):
        raise ValueError("Scores must be unique within a stimulus")

    expected_pairs = Counter(zip(cities, scores))
    observed_pairs = Counter(
        (match.group("city"), int(match.group("score")))
        for match in CITY_SCORE_RE.finditer(passage)
    )
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Filler contamination or damaged needle detected: "
            f"expected={expected_pairs}, observed={observed_pairs}"
        )
    for needle in row["needles"]:
        if passage.count(needle["decoded_text"]) != 1:
            raise ValueError(
                f"Needle {needle['needle_id']} is missing, truncated, or duplicated"
            )
        if not needle.get("token_span_verified"):
            raise ValueError(f"Needle {needle['needle_id']} has an unverified span")


def _next_untried_budget(
    proposed: int,
    *,
    seen: set[int],
    minimum: int,
    maximum: int,
) -> int | None:
    for radius in range(0, maximum - minimum + 1):
        candidates = (proposed + radius, proposed - radius)
        for candidate in candidates:
            if minimum <= candidate <= maximum and candidate not in seen:
                return candidate
    return None


def freeze_stimulus(
    *,
    target_passage_tokens: int,
    num_needles: int,
    seed: int,
    tokenizer: TokenizerAdapter,
    spec: FreezeSpec,
    haystack_dir: str | None = None,
    entities_path: str = "data/entities/cities.csv",
    fact_templates_path: str = "data/templates/niah_fact_single_template.txt",
    protocol: FreezeProtocol | None = None,
) -> dict[str, Any]:
    resolved_protocol = protocol or V2_FREEZE_PROTOCOL
    resolved_haystack_dir = haystack_dir or spec.haystack_dir
    initial_budget = max(
        spec.minimum_filler_tokens,
        target_passage_tokens - 18 * num_needles,
    )
    search_log: list[dict[str, int]] = []

    for retry_index in range(spec.max_window_retries):
        budget = initial_budget
        seen: set[int] = set()
        for attempt_index in range(spec.max_search_attempts):
            if budget in seen:
                break
            seen.add(budget)
            retry_seed = seed + retry_index * 1_000_003
            cfg = DynamicNiahV2Config(
                task_type="match_count",
                tokenizer_name=spec.canonical_tokenizer,
                num_examples=1,
                target_haystack_tokens=budget,
                num_needles=num_needles,
                insertion_positions=(0,) * num_needles,
                randomize_needle_insertion=True,
                randomize_needle_seed=seed,
                sentence_level_insertion=True,
                word_level_insertion=False,
                text_insertion_min_fraction=(
                    spec.insertion_depth_min_fraction
                ),
                text_insertion_max_fraction=(
                    spec.insertion_depth_max_fraction
                ),
                prompt_style="vanilla_no_cue",
                global_random_seed=seed,
                haystack_seed=retry_seed,
                haystack_source_mode=spec.haystack_source_mode,
                needle_seed=seed,
                haystack_dir=resolved_haystack_dir,
                entities_path=entities_path,
                fact_templates_path=fact_templates_path,
            )
            row = generate_dynamic_niah_dataset_v2(
                cfg,
                tokenizer_adapter=tokenizer,
            )[0]
            actual = len(tokenizer.encode(row["context"]))
            search_log.append(
                {
                    "retry_index": retry_index,
                    "attempt_index": attempt_index,
                    "clean_filler_tokens": budget,
                    "realized_passage_tokens": actual,
                }
            )
            if actual == target_passage_tokens:
                _assert_valid_candidate(
                    row,
                    target_passage_tokens=target_passage_tokens,
                    num_needles=num_needles,
                    tokenizer=tokenizer,
                )
                passage = row["context"]
                compact_needles = [
                    _compact_needle(
                        item,
                        passage_characters=len(passage),
                    )
                    for item in row["needles"]
                ]
                observed_depths = [
                    float(item["normalized_depth"])
                    for item in compact_needles
                ]
                if any(
                    depth < spec.insertion_depth_min_fraction
                    or depth > spec.insertion_depth_max_fraction
                    for depth in observed_depths
                ):
                    raise RuntimeError(
                        "Final needle depth fell outside the registered "
                        f"bounds [{spec.insertion_depth_min_fraction}, "
                        f"{spec.insertion_depth_max_fraction}]: "
                        f"{observed_depths}"
                    )
                stimulus_id = (
                    f"{resolved_protocol.stimulus_id_prefix}"
                    f"T{target_passage_tokens}_N{num_needles}_seed{seed}"
                )
                return {
                    "schema_version": (
                        resolved_protocol.stimulus_schema_version
                    ),
                    **(
                        {
                            "protocol_version": (
                                resolved_protocol.protocol_version
                            )
                        }
                        if resolved_protocol != V2_FREEZE_PROTOCOL
                        else {}
                    ),
                    "stimulus_id": stimulus_id,
                    "seed": seed,
                    "target_passage_tokens": target_passage_tokens,
                    "num_needles": num_needles,
                    "nominal_density_per_1k": num_needles
                    / (target_passage_tokens / 1000),
                    "canonical_tokenizer": spec.canonical_tokenizer,
                    "canonical_tokenizer_revision": (
                        spec.canonical_tokenizer_revision
                    ),
                    "canonical_passage_tokens": actual,
                    "clean_filler_tokens": budget,
                    "passage": passage,
                    "passage_sha256": _sha256_text(passage),
                    "gold_count": num_needles,
                    "gold_pairs": [
                        {
                            "needle_id": item["needle_id"],
                            "city": item["city"],
                            "score": item["score"],
                        }
                        for item in compact_needles
                    ],
                    "needles": compact_needles,
                    "realized_insertions": row["realized_insertions"],
                    **(
                        {
                            "insertion_depth_policy": {
                                "coordinate": (
                                    "needle_char_start/"
                                    "final_passage_characters"
                                ),
                                "minimum_inclusive": (
                                    spec.insertion_depth_min_fraction
                                ),
                                "maximum_inclusive": (
                                    spec.insertion_depth_max_fraction
                                ),
                                "candidate_filter": (
                                    "conservative_preinsertion_then_"
                                    "final_validation"
                                ),
                                "observed_minimum": min(observed_depths),
                                "observed_maximum": max(observed_depths),
                            }
                        }
                        if resolved_protocol != V2_FREEZE_PROTOCOL
                        else {}
                    ),
                    "haystack": {
                        key: value
                        for key, value in row["haystack"].items()
                        if key != "base_tokens"
                    },
                    "length_search": {
                        "attempts": len(search_log),
                        "retry_index": retry_index,
                        "history": search_log,
                        "post_insertion_truncation": False,
                    },
                }
            proposed = budget + (target_passage_tokens - actual)
            next_budget = _next_untried_budget(
                proposed,
                seen=seen,
                minimum=spec.minimum_filler_tokens,
                maximum=target_passage_tokens,
            )
            if next_budget is None:
                break
            budget = next_budget

    raise RuntimeError(
        "Unable to hit exact post-insertion passage length "
        f"for T={target_passage_tokens}, N={num_needles}, seed={seed}; "
        f"attempts={search_log}"
    )


def freeze_grid(
    *,
    output_dir: str | Path,
    spec: FreezeSpec | None = None,
    require_huggingface_tokenizer: bool = True,
    overwrite: bool = False,
    protocol: FreezeProtocol | None = None,
) -> dict[str, Path]:
    resolved_protocol = protocol or V2_FREEZE_PROTOCOL
    resolved_spec = spec or FreezeSpec()
    if resolved_spec.haystack_corpus_manifest is not None:
        corpus_manifest = Path(
            resolved_spec.haystack_corpus_manifest
        ).resolve()
        if not corpus_manifest.is_file():
            raise FileNotFoundError(
                f"Haystack corpus manifest does not exist: {corpus_manifest}"
            )
        observed_corpus_manifest_sha256 = hashlib.sha256(
            corpus_manifest.read_bytes()
        ).hexdigest()
        expected_corpus_manifest_sha256 = (
            resolved_spec.haystack_corpus_manifest_sha256
        )
        if (
            expected_corpus_manifest_sha256 is not None
            and expected_corpus_manifest_sha256
            != observed_corpus_manifest_sha256
        ):
            raise ValueError(
                "Haystack corpus manifest SHA256 mismatch: "
                f"{observed_corpus_manifest_sha256} != "
                f"{expected_corpus_manifest_sha256}"
            )
        resolved_spec = replace(
            resolved_spec,
            haystack_corpus_manifest=str(corpus_manifest),
            haystack_corpus_manifest_sha256=(
                observed_corpus_manifest_sha256
            ),
        )
    if resolved_spec.canonical_tokenizer_revision is None:
        if resolved_spec.canonical_tokenizer == "simple":
            immutable_revision = None
        else:
            from huggingface_hub import model_info

            info = model_info(resolved_spec.canonical_tokenizer)
            if not info.sha:
                raise RuntimeError(
                    "Unable to resolve an immutable canonical tokenizer revision"
                )
            immutable_revision = str(info.sha)
        resolved_spec = replace(
            resolved_spec,
            canonical_tokenizer_revision=immutable_revision,
        )
    output = Path(output_dir)
    stimuli_path = output / "stimuli.jsonl"
    if stimuli_path.exists() and not overwrite:
        raise FileExistsError(
            f"{stimuli_path} already exists; pass overwrite=True explicitly"
        )
    tokenizer = TokenizerAdapter(
        resolved_spec.canonical_tokenizer,
        revision=resolved_spec.canonical_tokenizer_revision,
        cache_dir=resolved_spec.tokenizer_cache_dir,
    )
    if require_huggingface_tokenizer and tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Canonical Hugging Face tokenizer failed to load; refusing the "
            f"silent {tokenizer.backend!r} fallback: {tokenizer.load_error}"
        )

    expected = (
        len(resolved_spec.passage_lengths)
        * len(resolved_spec.needle_counts)
        * len(resolved_spec.seeds)
    )
    rows: list[dict[str, Any]] = []
    for target in resolved_spec.passage_lengths:
        for num_needles in resolved_spec.needle_counts:
            for seed in resolved_spec.seeds:
                row = freeze_stimulus(
                    target_passage_tokens=target,
                    num_needles=num_needles,
                    seed=seed,
                    tokenizer=tokenizer,
                    spec=resolved_spec,
                    haystack_dir=resolved_spec.haystack_dir,
                    protocol=resolved_protocol,
                )
                rows.append(row)
                print(
                    f"[freeze {len(rows)}/{expected}] "
                    f"{row['stimulus_id']} "
                    f"attempts={row['length_search']['attempts']}",
                    flush=True,
                )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, generated {len(rows)}")
    ids = [row["stimulus_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate stimulus IDs detected")

    counts = Counter(
        (row["target_passage_tokens"], row["num_needles"]) for row in rows
    )
    expected_per_cell = len(resolved_spec.seeds)
    if any(value != expected_per_cell for value in counts.values()):
        raise RuntimeError("Each grid cell must contain all paired seeds")

    jsonl_payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    manifest_spec = asdict(resolved_spec)
    if resolved_protocol == V2_FREEZE_PROTOCOL:
        manifest_spec.pop("insertion_depth_min_fraction", None)
        manifest_spec.pop("insertion_depth_max_fraction", None)
    manifest = {
        "schema_version": resolved_protocol.manifest_schema_version,
        **(
            {
                "protocol_version": resolved_protocol.protocol_version,
                "stimulus_protocol": asdict(resolved_protocol),
            }
            if resolved_protocol != V2_FREEZE_PROTOCOL
            else {}
        ),
        "spec": manifest_spec,
        "rows": len(rows),
        "cells": len(counts),
        "rows_per_cell": expected_per_cell,
        "duplicate_stimulus_id": len(ids) - len(set(ids)),
        "canonical_passage_length_mismatch": sum(
            row["canonical_passage_tokens"] != row["target_passage_tokens"]
            for row in rows
        ),
        "truncated_or_missing_needles": 0,
        "task_or_template_tokens_in_T": 0,
    }
    contamination = {
        "rows_checked": len(rows),
        "extra_city_score_pairs": 0,
        "missing_or_damaged_needles": 0,
    }
    cell_counts = [
        {"target_passage_tokens": key[0], "num_needles": key[1], "rows": value}
        for key, value in sorted(counts.items())
    ]

    paths = {
        "stimuli": stimuli_path,
        "manifest": output / "manifest.json",
        "contamination_audit": output / "contamination_audit.json",
        "cell_counts": output / "cell_counts.json",
        "sha256": output / "SHA256SUMS",
    }
    _atomic_write(paths["stimuli"], jsonl_payload)
    _atomic_write(paths["manifest"], _json_bytes(manifest))
    _atomic_write(paths["contamination_audit"], _json_bytes(contamination))
    _atomic_write(paths["cell_counts"], _json_bytes(cell_counts))

    checksum_lines = []
    for name in ("stimuli", "manifest", "contamination_audit", "cell_counts"):
        payload = paths[name].read_bytes()
        checksum_lines.append(
            f"{hashlib.sha256(payload).hexdigest()}  {paths[name].name}"
        )
    _atomic_write(paths["sha256"], ("\n".join(checksum_lines) + "\n").encode())
    return paths


def load_stimuli(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit_frozen_grid(
    *,
    stimuli_path: str | Path,
    manifest_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    require_huggingface_tokenizer: bool = True,
    protocol: FreezeProtocol | None = None,
) -> dict[str, Any]:
    stimuli_file = Path(stimuli_path).resolve()
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else stimuli_file.with_name("manifest.json")
    )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    saved_protocol = manifest.get("stimulus_protocol")
    resolved_protocol = (
        protocol
        or (
            FreezeProtocol(**saved_protocol)
            if isinstance(saved_protocol, dict)
            else V2_FREEZE_PROTOCOL
        )
    )
    if manifest.get("schema_version") != (
        resolved_protocol.manifest_schema_version
    ):
        raise ValueError("Frozen manifest schema does not match the protocol")
    if manifest.get(
        "protocol_version",
        V2_FREEZE_PROTOCOL.protocol_version,
    ) != resolved_protocol.protocol_version:
        raise ValueError("Frozen manifest protocol version mismatch")
    manifest_spec = manifest["spec"]
    tokenizer_name = str(manifest_spec["canonical_tokenizer"])
    tokenizer_revision = manifest_spec.get("canonical_tokenizer_revision")
    tokenizer = TokenizerAdapter(
        tokenizer_name,
        revision=tokenizer_revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    if require_huggingface_tokenizer and tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Canonical Hugging Face tokenizer failed to load during audit: "
            f"{tokenizer.load_error}"
        )

    rows = load_stimuli(stimuli_file)
    errors: list[str] = []
    cell_counts: Counter[tuple[int, int]] = Counter()
    cell_seeds: dict[tuple[int, int], set[int]] = {}
    ids: list[str] = []
    for row in rows:
        identifier = str(row.get("stimulus_id", "<missing>"))
        ids.append(identifier)
        try:
            target = int(row["target_passage_tokens"])
            num_needles = int(row["num_needles"])
            seed = int(row["seed"])
            cell = (target, num_needles)
            cell_counts[cell] += 1
            cell_seeds.setdefault(cell, set()).add(seed)
            passage = str(row["passage"])
            passage_ids, passage_offsets = tokenizer.encode_with_offsets(passage)
            if len(passage_ids) != target:
                raise ValueError(
                    f"canonical length {len(passage_ids)} != {target}"
                )
            if int(row["canonical_passage_tokens"]) != target:
                raise ValueError("saved canonical passage length mismatch")
            if row["canonical_tokenizer"] != tokenizer_name:
                raise ValueError("canonical tokenizer name mismatch")
            if (
                row.get("canonical_tokenizer_revision")
                != tokenizer_revision
            ):
                raise ValueError("canonical tokenizer revision mismatch")
            if row["passage_sha256"] != _sha256_text(passage):
                raise ValueError("passage SHA256 mismatch")
            if int(row["gold_count"]) != num_needles:
                raise ValueError("gold_count does not equal num_needles")
            if len(row["gold_pairs"]) != num_needles:
                raise ValueError("gold pair count does not equal num_needles")
            if len(row["needles"]) != num_needles:
                raise ValueError("needle metadata count does not equal num_needles")
            if len(row["realized_insertions"]) != num_needles:
                raise ValueError(
                    "realized insertion count does not equal num_needles"
                )
            if row.get("schema_version") != (
                resolved_protocol.stimulus_schema_version
            ):
                raise ValueError("stimulus schema version mismatch")
            if bool(row["length_search"]["post_insertion_truncation"]):
                raise ValueError("post-insertion truncation is forbidden")
            if manifest_spec.get("haystack_source_mode") == (
                "multi_file_no_repeat"
            ):
                haystack = row["haystack"]
                if haystack.get("source_mode") != "multi_file_no_repeat":
                    raise ValueError("haystack source mode mismatch")
                if bool(haystack.get("source_repeated_to_target")):
                    raise ValueError("repeated haystack source is forbidden")
                if int(haystack.get("source_repeat_count", 0)) != 1:
                    raise ValueError("haystack source repeat count must be one")

            gold = [
                (str(item["city"]), int(item["score"]))
                for item in row["gold_pairs"]
            ]
            if len(gold) != len(set(gold)):
                raise ValueError("duplicate gold city-score pair")
            if len({city for city, _ in gold}) != num_needles:
                raise ValueError("duplicate city")
            if len({score for _, score in gold}) != num_needles:
                raise ValueError("duplicate score")
            observed = Counter(
                (match.group("city"), int(match.group("score")))
                for match in CITY_SCORE_RE.finditer(passage)
            )
            if observed != Counter(gold):
                raise ValueError(
                    f"filler contamination or damaged fact: {observed}"
                )

            for needle in row["needles"]:
                text = str(needle["text"])
                char_start = int(needle["char_start"])
                char_end = int(needle["char_end"])
                span_start = int(needle["canonical_span_start"])
                span_end = int(needle["canonical_span_end"])
                if passage[char_start:char_end] != text:
                    raise ValueError(
                        f"{needle['needle_id']} character span mismatch"
                    )
                overlapping = [
                    index
                    for index, (token_start, token_end) in enumerate(
                        passage_offsets
                    )
                    if token_end > token_start
                    and token_start < char_end
                    and token_end > char_start
                ]
                if not overlapping:
                    raise ValueError(
                        f"{needle['needle_id']} has no overlapping canonical token"
                    )
                expected_span = (overlapping[0], overlapping[-1] + 1)
                if (span_start, span_end) != expected_span:
                    raise ValueError(
                        f"{needle['needle_id']} canonical offset span mismatch: "
                        f"saved={(span_start, span_end)}, "
                        f"expected={expected_span}"
                    )
                if (
                    int(needle["canonical_context_span_length"])
                    != span_end - span_start
                ):
                    raise ValueError(
                        f"{needle['needle_id']} canonical span length mismatch"
                    )
                if needle.get("token_span_alignment") != "offset_mapping_overlap":
                    raise ValueError(
                        f"{needle['needle_id']} lacks exact offset-map alignment"
                    )
                if not bool(needle.get("token_span_verified")):
                    raise ValueError(
                        f"{needle['needle_id']} token span is not verified"
                    )
                if passage.count(text) != 1:
                    raise ValueError(
                        f"{needle['needle_id']} missing or duplicated"
                    )
                minimum_depth = float(
                    manifest_spec.get(
                        "insertion_depth_min_fraction",
                        0.0,
                    )
                )
                maximum_depth = float(
                    manifest_spec.get(
                        "insertion_depth_max_fraction",
                        1.0,
                    )
                )
                if not minimum_depth <= float(
                    needle["normalized_depth"]
                ) <= maximum_depth:
                    raise ValueError(
                        f"{needle['needle_id']} normalized depth out of range"
                    )
            expected_id = (
                f"{resolved_protocol.stimulus_id_prefix}"
                f"T{target}_N{num_needles}_seed{seed}"
            )
            if identifier != expected_id:
                raise ValueError(
                    f"stimulus ID mismatch: expected {expected_id}"
                )
        except Exception as exc:
            errors.append(f"{identifier}: {type(exc).__name__}: {exc}")

    expected_lengths = tuple(int(x) for x in manifest_spec["passage_lengths"])
    expected_counts = tuple(int(x) for x in manifest_spec["needle_counts"])
    expected_seeds = tuple(int(x) for x in manifest_spec["seeds"])
    expected_cells = {
        (target, num_needles)
        for target in expected_lengths
        for num_needles in expected_counts
    }
    if set(cell_counts) != expected_cells:
        errors.append(
            "grid cells mismatch: "
            f"observed={sorted(cell_counts)}, expected={sorted(expected_cells)}"
        )
    if any(value != len(expected_seeds) for value in cell_counts.values()):
        errors.append("not every cell contains the registered seed count")
    for cell in expected_cells:
        if cell_seeds.get(cell, set()) != set(expected_seeds):
            errors.append(
                f"paired seed mismatch for T={cell[0]}, N={cell[1]}: "
                f"observed={sorted(cell_seeds.get(cell, set()))}, "
                f"expected={sorted(expected_seeds)}"
            )
    if len(ids) != len(set(ids)):
        errors.append("duplicate stimulus IDs")
    if 0 in expected_counts:
        errors.append("N=0 is forbidden in this experiment")

    checksum_file = stimuli_file.with_name("SHA256SUMS")
    checksum_status: dict[str, bool] = {}
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, filename = line.split(maxsplit=1)
            target_file = checksum_file.parent / filename.strip()
            matches = (
                target_file.exists()
                and hashlib.sha256(target_file.read_bytes()).hexdigest()
                == digest
            )
            checksum_status[filename.strip()] = matches
            if not matches:
                errors.append(f"checksum mismatch: {filename.strip()}")
    else:
        errors.append("SHA256SUMS is missing")

    return {
        "schema_version": resolved_protocol.audit_schema_version,
        **(
            {"protocol_version": resolved_protocol.protocol_version}
            if resolved_protocol != V2_FREEZE_PROTOCOL
            else {}
        ),
        "passed": not errors,
        "stimuli_path": str(stimuli_file),
        "stimuli_sha256": hashlib.sha256(stimuli_file.read_bytes()).hexdigest(),
        "canonical_tokenizer": tokenizer_name,
        "canonical_tokenizer_revision": tokenizer_revision,
        "rows_checked": len(rows),
        "cells_checked": len(cell_counts),
        "rows_per_cell": sorted(set(cell_counts.values())),
        "needle_counts": list(expected_counts),
        "checksum_status": checksum_status,
        "errors": errors,
    }


def select_stimuli(
    rows: Iterable[dict[str, Any]],
    *,
    passage_lengths: Iterable[int] | None = None,
    needle_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    allowed_t = set(passage_lengths) if passage_lengths is not None else None
    allowed_n = set(needle_counts) if needle_counts is not None else None
    allowed_s = set(seeds) if seeds is not None else None
    return [
        row
        for row in rows
        if (allowed_t is None or row["target_passage_tokens"] in allowed_t)
        and (allowed_n is None or row["num_needles"] in allowed_n)
        and (allowed_s is None or row["seed"] in allowed_s)
    ]
