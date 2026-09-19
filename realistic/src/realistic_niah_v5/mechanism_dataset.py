from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from realistic_niah.prompts import build_messages, query_block
from realistic_niah_v4.prompts import V4_NUMERIC_QUERY_BLOCK

from .generation import build_v5_user_text


LEGACY_PAIRED_SCHEMA_VERSION = "realistic_niah_count_mechanism_dataset_v1"
LEGACY_PAIRED_PROTOCOL_VERSION = "v4_4_shared_backbone_non_native_pair_v1"
DATASET_SCHEMA_VERSION = "realistic_niah_count_geometry_shared_v2"
DATASET_PROTOCOL_VERSION = "v4_4_shared_geometry_mode_contracts_v2"
MODE_CONTRACT_SCHEMA_VERSION = "realistic_niah_count_mode_contracts_v1"
CAUSAL_REGISTRY_SCHEMA_VERSION = "realistic_niah_count_causal_registry_v1"
GEOMETRY_PANEL_ID = "v4_4_seed1234_1263_count1_10_shared_geometry_v1"
PASSAGE_PLACEHOLDER = "{{passage}}"
_PASSAGE_SENTINEL = "__REALISTIC_NIAH_SHARED_PASSAGE_SENTINEL__"
MODE_CONTRACT_IDS = {
    "non_thinking": "v4_4_non_thinking_direct_numeric_v1",
    "native_thinking": "v4_4_native_thinking_concise_v1",
}
EXPECTED_SOURCE_STIMULI_SHA256 = (
    "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9da5078f1b340aea440b2dcb51caa3c135873fb2daa9f53462fa4086b0b162ef"
)
EXPECTED_PAIR_BACKBONE_SHA256 = (
    "f70a62c0a9cb10d4f80f950566aa321f84fb96df67479b2f60c424f6c300238e"
)
MODES = ("non_thinking", "native_thinking")
SPLITS = ("discovery", "confirmation")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True).encode("utf-8")
        + b"\n"
        for row in rows
    )
    _write_bytes(path, payload)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def build_non_thinking_user_text(passage: str) -> str:
    messages = build_messages(str(passage), prompt_mode="direct")
    registered = query_block("direct")
    if len(messages) != 1 or not messages[0]["content"].endswith(registered):
        raise RuntimeError("Registered direct prompt layout changed")
    return messages[0]["content"][: -len(registered)] + V4_NUMERIC_QUERY_BLOCK


def _mode_user_text(mode: str, passage: str) -> str:
    if mode == "non_thinking":
        return build_non_thinking_user_text(passage)
    if mode == "native_thinking":
        return build_v5_user_text(passage)
    raise ValueError(f"Unknown mechanism dataset mode: {mode}")


def build_mode_contracts() -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rendered = _mode_user_text(mode, _PASSAGE_SENTINEL)
        if rendered.count(_PASSAGE_SENTINEL) != 1:
            raise RuntimeError(f"{mode} prompt does not contain one passage sentinel")
        template = rendered.replace(_PASSAGE_SENTINEL, PASSAGE_PLACEHOLDER)
        enable_thinking = mode == "native_thinking"
        if mode == "non_thinking":
            assistant_prefix = "Total:"
            expected_final_line_template = "Total:{gold_count}"
            runtime_note = (
                "Apply the registered model chat template with native thinking "
                "disabled, then append assistant_prefix before scoring/generation."
            )
        else:
            assistant_prefix = ""
            expected_final_line_template = "Total: {gold_count}"
            runtime_note = (
                "Apply the registered model chat template with native thinking "
                "enabled; do not inject a synthetic reasoning scaffold."
            )
        contracts[mode] = {
            "schema_version": MODE_CONTRACT_SCHEMA_VERSION,
            "contract_id": MODE_CONTRACT_IDS[mode],
            "dataset_mode": mode,
            "native_thinking": enable_thinking,
            "chat_template_thinking_enabled": enable_thinking,
            "user_text_template": template,
            "user_text_template_sha256": _sha256_bytes(template.encode("utf-8")),
            "assistant_prefix": assistant_prefix,
            "expected_final_line_template": expected_final_line_template,
            "runtime_note": runtime_note,
            "treatment_note": (
                "The registered mode treatment jointly includes this prompt contract "
                "and the chat-template thinking control; it is not a flag-only contrast."
            ),
        }
    return contracts


def render_mode_user_text(contract: Mapping[str, Any], passage: str) -> str:
    template = str(contract["user_text_template"])
    if template.count(PASSAGE_PLACEHOLDER) != 1:
        raise ValueError("Mode contract must contain exactly one passage placeholder")
    return template.replace(PASSAGE_PLACEHOLDER, str(passage))


def render_expected_final_line(contract: Mapping[str, Any], gold_count: int) -> str:
    return str(contract["expected_final_line_template"]).format(
        gold_count=int(gold_count)
    )


def _backbone_sha256(stimulus: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(dict(stimulus)))


def _validate_stimulus(row: Mapping[str, Any]) -> None:
    if row.get("design_variant") != "v4.4":
        raise ValueError("Mechanism dataset accepts V4.4 rows only")
    passage = str(row["passage"])
    if _sha256_bytes(passage.encode("utf-8")) != row["passage_sha256"]:
        raise ValueError(f"Passage SHA mismatch for {row.get('stimulus_id')}")
    count = int(row["gold_count"])
    spans = list(row["active_needle_spans"])
    gold = list(row["gold_pairs"])
    if len(spans) != count or len(gold) != count:
        raise ValueError(f"Needle/gold count mismatch for {row.get('stimulus_id')}")
    for span, pair in zip(spans, gold):
        start, end = int(span["char_start"]), int(span["char_end"])
        if not 0 <= start < end <= len(passage):
            raise ValueError(f"Invalid needle char span for {row.get('stimulus_id')}")
        if passage[start:end] != str(span["text"]):
            raise ValueError(f"Needle span text mismatch for {row.get('stimulus_id')}")
        if int(span["slot_index"]) != int(pair["slot_index"]):
            raise ValueError(f"Needle/gold slot mismatch for {row.get('stimulus_id')}")
        if str(span["city"]) != str(pair["city"]):
            raise ValueError(f"Needle/gold city mismatch for {row.get('stimulus_id')}")
        if int(span["score"]) != int(pair["score"]):
            raise ValueError(f"Needle/gold score mismatch for {row.get('stimulus_id')}")


def paired_record(stimulus: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown mechanism dataset mode: {mode}")
    _validate_stimulus(stimulus)
    source = dict(stimulus)
    source_schema = str(source.pop("schema_version"))
    source_protocol = str(source.pop("protocol_version"))
    passage = str(source["passage"])
    if mode == "non_thinking":
        user_text = build_non_thinking_user_text(passage)
        enable_thinking = False
        assistant_prefix = "Total:"
        expected_final_line = f"Total:{int(source['gold_count'])}"
        runtime_note = (
            "Apply the registered model chat template with native thinking disabled, "
            "then append assistant_prefix before scoring/generation."
        )
    else:
        user_text = build_v5_user_text(passage)
        enable_thinking = True
        assistant_prefix = ""
        expected_final_line = f"Total: {int(source['gold_count'])}"
        runtime_note = (
            "Apply the registered model chat template with native thinking enabled; "
            "do not inject a synthetic reasoning scaffold."
        )
    backbone = {
        "schema_version": source_schema,
        "protocol_version": source_protocol,
        **source,
    }
    pair_id = str(source["stimulus_id"])
    return {
        "schema_version": LEGACY_PAIRED_SCHEMA_VERSION,
        "dataset_protocol_version": LEGACY_PAIRED_PROTOCOL_VERSION,
        "dataset_mode": mode,
        "row_id": f"{mode}:{pair_id}",
        "pair_id": pair_id,
        "backbone_sha256": _backbone_sha256(backbone),
        "source_schema_version": source_schema,
        "source_protocol_version": source_protocol,
        "native_thinking": enable_thinking,
        "chat_template_thinking_enabled": enable_thinking,
        "user_text": user_text,
        "user_text_sha256": _sha256_bytes(user_text.encode("utf-8")),
        "messages": [{"role": "user", "content": user_text}],
        "assistant_prefix": assistant_prefix,
        "expected_final_line": expected_final_line,
        "runtime_note": runtime_note,
        **source,
    }


_SHARED_ROW_METADATA_FIELDS = {
    "schema_version",
    "dataset_protocol_version",
    "geometry_panel_id",
    "row_id",
    "pair_id",
    "backbone_sha256",
    "source_schema_version",
    "source_protocol_version",
    "available_modes",
    "mode_views",
}


def shared_geometry_record(
    stimulus: Mapping[str, Any],
    *,
    mode_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _validate_stimulus(stimulus)
    contracts = dict(mode_contracts or build_mode_contracts())
    if set(contracts) != set(MODES):
        raise ValueError("Mode contracts must define non_thinking and native_thinking")
    source = dict(stimulus)
    source_schema = str(source.pop("schema_version"))
    source_protocol = str(source.pop("protocol_version"))
    backbone = {
        "schema_version": source_schema,
        "protocol_version": source_protocol,
        **source,
    }
    passage = str(source["passage"])
    gold_count = int(source["gold_count"])
    mode_views: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        contract = contracts[mode]
        rendered = render_mode_user_text(contract, passage)
        if rendered != _mode_user_text(mode, passage):
            raise RuntimeError(f"{mode} contract no longer renders the registered prompt")
        mode_views[mode] = {
            "contract_id": str(contract["contract_id"]),
            "contract_sha256": _sha256_bytes(_canonical_json(contract)),
            "user_text_sha256": _sha256_bytes(rendered.encode("utf-8")),
            "expected_final_line": render_expected_final_line(contract, gold_count),
        }
    pair_id = str(source["stimulus_id"])
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "geometry_panel_id": GEOMETRY_PANEL_ID,
        "row_id": pair_id,
        "pair_id": pair_id,
        "backbone_sha256": _backbone_sha256(backbone),
        "source_schema_version": source_schema,
        "source_protocol_version": source_protocol,
        "available_modes": list(MODES),
        "mode_views": mode_views,
        **source,
    }


def audit_paired_records(
    non_thinking: Sequence[Mapping[str, Any]],
    native_thinking: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(non_thinking) != len(native_thinking):
        errors.append("mode row counts differ")
    pairs = []
    for left, right in zip(non_thinking, native_thinking):
        pair_id = str(left.get("pair_id"))
        if pair_id != str(right.get("pair_id")):
            errors.append(f"pair order/id mismatch at {pair_id}")
            continue
        if left.get("backbone_sha256") != right.get("backbone_sha256"):
            errors.append(f"backbone mismatch for {pair_id}")
        for key in (
            "passage",
            "passage_sha256",
            "gold_count",
            "gold_pairs",
            "slots",
            "active_needle_spans",
            "hard_negative_spans",
            "design",
            "seed",
            "split",
        ):
            if left.get(key) != right.get(key):
                errors.append(f"{key} differs for {pair_id}")
        passage = str(left.get("passage", ""))
        if str(left.get("user_text", "")).count(passage) != 1:
            errors.append(f"non-thinking prompt does not contain one passage: {pair_id}")
        if str(right.get("user_text", "")).count(passage) != 1:
            errors.append(f"native prompt does not contain one passage: {pair_id}")
        pairs.append((pair_id, str(left.get("backbone_sha256"))))
    return {
        "passed": not errors,
        "errors": errors,
        "paired_rows": len(pairs),
        "pair_backbone_sha256": _sha256_bytes(_canonical_json(pairs)),
        "all_passages_identical_between_modes": not any(
            "passage differs" in error for error in errors
        ),
        "all_needle_spans_identical_between_modes": not any(
            "active_needle_spans differs" in error for error in errors
        ),
        "all_gold_pairs_identical_between_modes": not any(
            "gold_pairs differs" in error for error in errors
        ),
    }


def audit_shared_geometry_records(
    records: Sequence[Mapping[str, Any]],
    *,
    mode_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    contracts = dict(mode_contracts or build_mode_contracts())
    errors: list[str] = []
    seen_pair_ids: set[str] = set()
    pairs: list[tuple[str, str]] = []
    prompt_hash_checks = 0
    for index, row in enumerate(records):
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            errors.append(f"row {index} has no pair_id")
            continue
        if pair_id in seen_pair_ids:
            errors.append(f"duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        if row.get("row_id") != pair_id or row.get("stimulus_id") != pair_id:
            errors.append(f"row/stimulus id mismatch for {pair_id}")
        if row.get("schema_version") != DATASET_SCHEMA_VERSION:
            errors.append(f"schema version mismatch for {pair_id}")
        if row.get("dataset_protocol_version") != DATASET_PROTOCOL_VERSION:
            errors.append(f"dataset protocol mismatch for {pair_id}")
        if row.get("geometry_panel_id") != GEOMETRY_PANEL_ID:
            errors.append(f"geometry panel mismatch for {pair_id}")
        if list(row.get("available_modes", [])) != list(MODES):
            errors.append(f"available modes mismatch for {pair_id}")

        source = {
            key: value
            for key, value in row.items()
            if key not in _SHARED_ROW_METADATA_FIELDS
        }
        backbone = {
            "schema_version": str(row.get("source_schema_version", "")),
            "protocol_version": str(row.get("source_protocol_version", "")),
            **source,
        }
        try:
            _validate_stimulus(backbone)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid shared stimulus for {pair_id}: {exc}")
            continue
        backbone_sha = _backbone_sha256(backbone)
        if row.get("backbone_sha256") != backbone_sha:
            errors.append(f"backbone SHA mismatch for {pair_id}")
        pairs.append((pair_id, backbone_sha))

        views = row.get("mode_views")
        if not isinstance(views, Mapping) or set(views) != set(MODES):
            errors.append(f"mode views mismatch for {pair_id}")
            continue
        passage = str(row["passage"])
        gold_count = int(row["gold_count"])
        for mode in MODES:
            contract = contracts.get(mode)
            view = views.get(mode)
            if not isinstance(contract, Mapping) or not isinstance(view, Mapping):
                errors.append(f"missing {mode} contract/view for {pair_id}")
                continue
            if view.get("contract_id") != contract.get("contract_id"):
                errors.append(f"{mode} contract id mismatch for {pair_id}")
            expected_contract_sha = _sha256_bytes(_canonical_json(contract))
            if view.get("contract_sha256") != expected_contract_sha:
                errors.append(f"{mode} contract SHA mismatch for {pair_id}")
            rendered = render_mode_user_text(contract, passage)
            if rendered != _mode_user_text(mode, passage):
                errors.append(f"{mode} registered prompt mismatch for {pair_id}")
            rendered_sha = _sha256_bytes(rendered.encode("utf-8"))
            if view.get("user_text_sha256") != rendered_sha:
                errors.append(f"{mode} prompt SHA mismatch for {pair_id}")
            expected_line = render_expected_final_line(contract, gold_count)
            if view.get("expected_final_line") != expected_line:
                errors.append(f"{mode} expected line mismatch for {pair_id}")
            prompt_hash_checks += 1
    return {
        "passed": not errors,
        "errors": errors,
        "geometry_shared_rows": len(records),
        "unique_pair_ids": len(seen_pair_ids),
        "mode_contracts": list(MODES),
        "prompt_hash_checks": prompt_hash_checks,
        "pair_backbone_sha256": _sha256_bytes(_canonical_json(pairs)),
    }


def _dataset_card(source_stimuli_sha256: str) -> str:
    return f"""---
pretty_name: Realistic NIAH Count Mechanism Analysis
task_categories:
- text-generation
language:
- en
size_categories:
- n<1K
configs:
- config_name: geometry_shared
  default: true
  data_files:
  - split: discovery
    path: data/geometry_shared/discovery.jsonl
  - split: confirmation
    path: data/geometry_shared/confirmation.jsonl
---

# Realistic NIAH count mechanism analysis

Version 2 stores the paired geometry panel once. The default
`geometry_shared` configuration contains 300 unique V4.4 stimulus rows: 200
discovery rows (seeds 1234-1253) and 100 held-out confirmation rows (seeds
1254-1263), with counts 1-10 balanced within every seed. Each `pair_id` is now
one row rather than two duplicated mode rows.

The common row contains the passage, gold records, slots, active needle spans,
hard negatives, design metadata, and a byte-canonical `backbone_sha256`.
`mode_views` stores only per-row prompt hashes and expected final lines for the
registered `non_thinking` and `native_thinking` contracts. Full prompt/runtime
contracts are stored once in `contracts/MODE_CONTRACTS.json`.

## Frozen source

- Source protocol: `realistic_niah_v4_nonthinking_v3`
- Design variant: `v4.4`
- Canonical passage length: 10,000 Qwen3-8B tokenizer tokens
- Full source `stimuli.jsonl` SHA-256: `{source_stimuli_sha256}`
- Original full-grid audit: passed (1,200 rows across V4.1-V4.4)

The uploaded geometry config contains only the 300 V4.4 rows. `source/`
retains the original audit and a V4.4-only excerpt of the source manifest.

This repository contains inputs only: no model generations, parsed traces,
hidden states, attention tables, representation outputs, causal results, or
mechanism result schemas are included.

## Prompt/runtime contract

To render a mode prompt, select its entry in `contracts/MODE_CONTRACTS.json`
and replace the single `{{{{passage}}}}` placeholder in `user_text_template`
with the row's `passage`. The resulting SHA-256 must equal
`mode_views[mode].user_text_sha256`.

- `non_thinking`: disable native thinking, use the registered direct numeric
  prompt, and append the contract's `assistant_prefix` (`Total:`) before
  scoring or generation.
- `native_thinking`: enable native thinking, use the registered concise
  reasoning prompt, and do not inject a synthetic reasoning scaffold.

The comparison treatment jointly includes the registered prompt contract and
the chat-template thinking control. It is not a flag-only contrast.

## Mode-specific causal extensions

Future causal inputs or result tables must not modify `geometry_shared`.
Register them in `causal/REGISTRY.json` and store files under
`data/causal/<mode>/<experiment_id>/<split>.jsonl`. Each registry entry records
its own row schema, seeds, counts, controls, hashes, and discovery/confirmation
policy. Non-thinking and native-thinking extensions may therefore add seeds
independently while retaining an immutable shared geometry panel.

When an extension is published through Dataset Viewer, expose it as a distinct
config named `causal_<mode>_<experiment_id>`; do not combine heterogeneous
experiment schemas into the default geometry config.

## Migration from version 1

The previous `non_thinking` and `native_thinking` configs duplicated all 300
backbones. Their exact mode prompts remain reproducible from the v2 contracts
and per-row hashes. Pin the previous Hub revision if the legacy two-config row
shape is required.

Rendered token IDs are intentionally not stored because they depend on the
model family and pinned tokenizer revision. Exact prompt-token needle spans
must be reconstructed from each row's half-open passage character spans after
applying the registered model chat template.

"""


def mode_contracts_document(
    contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved = dict(contracts or build_mode_contracts())
    return {
        "schema_version": MODE_CONTRACT_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "passage_placeholder": PASSAGE_PLACEHOLDER,
        "contracts": resolved,
    }


def causal_extension_registry() -> dict[str, Any]:
    return {
        "schema_version": CAUSAL_REGISTRY_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "geometry_panel_id": GEOMETRY_PANEL_ID,
        "layout": "data/causal/{mode}/{experiment_id}/{split}.jsonl",
        "dataset_viewer_config_template": "causal_{mode}_{experiment_id}",
        "rules": [
            "Do not edit or duplicate geometry_shared rows in a causal extension.",
            "Each extension declares its own row schema, seeds, counts, controls, and file hashes.",
            "Discovery and confirmation membership is immutable after registration.",
            "Mode-specific extra seeds are allowed and must be listed explicitly.",
            "Cross-mode causal contrasts must identify their shared paired confirmation subset.",
        ],
        "required_entry_fields": [
            "experiment_id",
            "mode",
            "dataset_viewer_config",
            "row_schema_version",
            "files",
            "discovery_seeds",
            "confirmation_seeds",
            "counts",
            "controls",
            "file_sha256",
            "source_commit",
        ],
        "extension_entry_template": {
            "experiment_id": "<stable_experiment_id>",
            "mode": "<non_thinking|native_thinking>",
            "dataset_viewer_config": "causal_<mode>_<experiment_id>",
            "row_schema_version": "<experiment_specific_schema>",
            "files": {
                "discovery": "data/causal/<mode>/<experiment_id>/discovery.jsonl",
                "confirmation": "data/causal/<mode>/<experiment_id>/confirmation.jsonl",
            },
            "discovery_seeds": [],
            "confirmation_seeds": [],
            "counts": [],
            "controls": [],
            "shared_confirmation_pair_ids_file": None,
            "file_sha256": {},
            "source_commit": None,
            "notes": None,
        },
        "modes": {
            "non_thinking": {"extensions": []},
            "native_thinking": {"extensions": []},
        },
    }


def build_mechanism_dataset(
    *,
    source_dataset_dir: str | Path,
    output_dir: str | Path,
    repository_head: str | None = None,
) -> dict[str, Any]:
    source = Path(source_dataset_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stimuli_path = source / "stimuli.jsonl"
    manifest_path = source / "manifest.json"
    audit_path = source / "audit.json"
    stimuli_sha = sha256_file(stimuli_path)
    manifest_sha = sha256_file(manifest_path)
    if stimuli_sha != EXPECTED_SOURCE_STIMULI_SHA256:
        raise ValueError(
            f"Unexpected source stimuli SHA-256: {stimuli_sha}; "
            f"expected {EXPECTED_SOURCE_STIMULI_SHA256}"
        )
    if manifest_sha != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            f"Unexpected source manifest SHA-256: {manifest_sha}; "
            f"expected {EXPECTED_SOURCE_MANIFEST_SHA256}"
        )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_audit.get("passed") is not True:
        raise ValueError("The original V4 source audit did not pass")
    stimuli = [
        row for row in load_jsonl(stimuli_path) if row.get("design_variant") == "v4.4"
    ]
    if len(stimuli) != 300:
        raise ValueError(f"Expected 300 frozen V4.4 rows, found {len(stimuli)}")
    if len({str(row["stimulus_id"]) for row in stimuli}) != len(stimuli):
        raise ValueError("V4.4 stimulus IDs are not unique")
    expected_seeds = set(range(1234, 1264))
    expected_counts = set(range(1, 11))
    if {int(row["seed"]) for row in stimuli} != expected_seeds:
        raise ValueError("V4.4 seed grid is not 1234-1263")
    if {int(row["gold_count"]) for row in stimuli} != expected_counts:
        raise ValueError("V4.4 count grid is not 1-10")
    cells = Counter((int(row["seed"]), int(row["gold_count"])) for row in stimuli)
    if set(cells.values()) != {1} or len(cells) != 300:
        raise ValueError("V4.4 seed/count grid is incomplete or duplicated")

    mode_contracts = build_mode_contracts()
    records = [
        shared_geometry_record(row, mode_contracts=mode_contracts) for row in stimuli
    ]
    geometry_audit = audit_shared_geometry_records(
        records, mode_contracts=mode_contracts
    )
    if not geometry_audit["passed"]:
        raise RuntimeError(
            "Shared geometry export audit failed: "
            + "; ".join(geometry_audit["errors"])
        )
    if geometry_audit["pair_backbone_sha256"] != EXPECTED_PAIR_BACKBONE_SHA256:
        raise RuntimeError(
            "Shared pair backbone SHA changed: "
            f"{geometry_audit['pair_backbone_sha256']}"
        )

    data_paths: dict[str, Path] = {}
    for split in SPLITS:
        path = output / "data" / "geometry_shared" / f"{split}.jsonl"
        split_rows = [row for row in records if row["split"] == split]
        expected = 200 if split == "discovery" else 100
        if len(split_rows) != expected:
            raise ValueError(f"Expected {expected} rows for geometry_shared/{split}")
        _write_jsonl(path, split_rows)
        data_paths[f"geometry_shared/{split}"] = path

    contracts_path = output / "contracts" / "MODE_CONTRACTS.json"
    causal_registry_path = output / "causal" / "REGISTRY.json"
    _write_json(contracts_path, mode_contracts_document(mode_contracts))
    _write_json(causal_registry_path, causal_extension_registry())

    source_excerpt = {
        **{key: value for key, value in source_manifest.items() if key != "families"},
        "families": [
            item
            for item in source_manifest["families"]
            if item.get("design_variant") == "v4.4"
        ],
        "source_stimuli_sha256": stimuli_sha,
        "source_manifest_sha256": manifest_sha,
        "exported_design_variant": "v4.4",
        "exported_rows": 300,
    }
    _write_json(output / "source" / "manifest_v4_4_excerpt.json", source_excerpt)
    _write_json(output / "source" / "original_audit.json", source_audit)
    _write_bytes(output / "README.md", _dataset_card(stimuli_sha).encode("utf-8"))

    file_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "realistic_niah_count_mechanism_manifest_v2",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "geometry_panel_id": GEOMETRY_PANEL_ID,
        "repository_head": repository_head,
        "source": {
            "artifact": "exports/run_20260731_v4_numeric_presentation_v3/dataset",
            "stimuli_sha256": stimuli_sha,
            "manifest_sha256": manifest_sha,
            "original_rows": int(source_manifest["rows"]),
            "original_audit_passed": True,
            "exported_design_variant": "v4.4",
        },
        "configs": {
            "geometry_shared": {
                "default": True,
                "rows": len(records),
                "discovery_rows": sum(
                    row["split"] == "discovery" for row in records
                ),
                "confirmation_rows": sum(
                    row["split"] == "confirmation" for row in records
                ),
                "seeds": list(range(1234, 1264)),
                "counts": list(range(1, 11)),
                "available_modes": list(MODES),
                "immutable": True,
            }
        },
        "mode_contracts": {
            "path": contracts_path.relative_to(output).as_posix(),
            "contract_ids": {
                mode: mode_contracts[mode]["contract_id"] for mode in MODES
            },
        },
        "causal_extensions": {
            "registry": causal_registry_path.relative_to(output).as_posix(),
            "registered": 0,
            "modes": list(MODES),
        },
        "migration": {
            "from_dataset_protocol_version": LEGACY_PAIRED_PROTOCOL_VERSION,
            "legacy_configs": list(MODES),
            "replacement_config": "geometry_shared",
            "legacy_rows": 600,
            "shared_rows": 300,
        },
        "geometry_audit": geometry_audit,
        "files": file_hashes,
    }
    _write_json(output / "DATASET_MANIFEST.json", manifest)
    audit = {
        "schema_version": "realistic_niah_count_mechanism_audit_v2",
        "passed": geometry_audit["passed"],
        "source_full_grid_audit_passed": True,
        "source_stimuli_sha256_verified": True,
        "source_manifest_sha256_verified": True,
        "v4_4_rows": 300,
        "geometry_shared_rows": 300,
        "discovery_rows": 200,
        "confirmation_rows": 100,
        "mode_contracts_verified": geometry_audit["prompt_hash_checks"],
        "causal_registry_extensions": 0,
        **geometry_audit,
    }
    _write_json(output / "AUDIT.json", audit)
    final_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    checksum = "".join(f"{digest}  {name}\n" for name, digest in final_hashes.items())
    _write_bytes(output / "SHA256SUMS", checksum.encode("utf-8"))
    return {
        "output_dir": str(output),
        "manifest": str(output / "DATASET_MANIFEST.json"),
        "audit": audit,
        "data_files": {key: str(value) for key, value in data_paths.items()},
    }
