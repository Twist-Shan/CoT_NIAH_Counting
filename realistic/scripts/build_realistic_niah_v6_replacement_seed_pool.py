#!/usr/bin/env python3
"""Build the frozen-amendment V4.4 reserve stimuli for V6 format failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generation.dynamic_niah import TokenizerAdapter  # noqa: E402
from realistic_niah_v4.spec import V4Config  # noqa: E402
from realistic_niah_v4.stimuli import (  # noqa: E402
    ControlledFreezeSpec,
    build_controlled_family,
)
from realistic_niah_v6.pipeline import (  # noqa: E402
    EXPECTED_SOURCE_STIMULI_SHA256,
    read_jsonl,
    sha256_file,
)
from realistic_niah_v6.replacement import (  # noqa: E402
    POOL_SCHEMA_VERSION,
    load_replacement_policy,
)
from realistic_niah_v6.spec import V6Config  # noqa: E402


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def _anchor_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stimulus_id": str(row["stimulus_id"]),
        "passage_sha256": str(row["passage_sha256"]),
        "gold_pairs": row["gold_pairs"],
        "active_needle_spans": row["active_needle_spans"],
        "slot_final_starts": row["design"]["slot_final_starts"],
        "content_permutation_zero_based": row["design"][
            "content_permutation_zero_based"
        ],
    }


def _load_frozen_base_pool(
    *,
    base_pool: Path,
    base_policy_path: Path,
    current_policy: Mapping[str, Any],
    config: V6Config,
    source_stimuli: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate and reuse a frozen pool before generating an exact seed suffix."""

    base_stimuli = base_pool / "stimuli.jsonl"
    base_manifest_path = base_pool / "manifest.json"
    if not base_stimuli.is_file() or not base_manifest_path.is_file():
        raise ValueError("Frozen base pool is missing stimuli.jsonl or manifest.json")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Frozen base pool has the wrong schema")
    if base_manifest.get("status") != "PASS_AMENDMENT_RESERVE_POOL":
        raise ValueError("Frozen base pool did not pass construction")
    if base_manifest.get("stimuli_sha256") != sha256_file(base_stimuli):
        raise ValueError("Frozen base pool stimulus hash changed")
    if base_manifest.get("replacement_policy_sha256") != sha256_file(
        base_policy_path
    ):
        raise ValueError("Frozen base pool policy hash changed")
    if base_manifest.get("source_stimuli_sha256") != sha256_file(source_stimuli):
        raise ValueError("Frozen base pool used a different source stimulus file")

    base_policy = load_replacement_policy(base_policy_path, config)
    base_discovery = tuple(
        map(int, base_policy["discovery_replacement_seed_pool"])
    )
    base_confirmation = tuple(
        map(int, base_policy["confirmation_replacement_seed_pool"])
    )
    current_discovery = tuple(
        map(int, current_policy["discovery_replacement_seed_pool"])
    )
    current_confirmation = tuple(
        map(int, current_policy["confirmation_replacement_seed_pool"])
    )
    if current_discovery[: len(base_discovery)] != base_discovery:
        raise ValueError("Current discovery pool is not a suffix extension of the base")
    if current_confirmation[: len(base_confirmation)] != base_confirmation:
        raise ValueError("Current confirmation pool is not a suffix extension of the base")
    extension_discovery = current_discovery[len(base_discovery) :]
    extension_confirmation = current_confirmation[len(base_confirmation) :]
    if not extension_discovery and not extension_confirmation:
        raise ValueError("Current policy does not extend the frozen base pool")

    if list(map(int, base_manifest.get("discovery_seeds", ()))) != list(
        base_discovery
    ):
        raise ValueError("Frozen base pool discovery seeds changed")
    if list(map(int, base_manifest.get("confirmation_seeds", ()))) != list(
        base_confirmation
    ):
        raise ValueError("Frozen base pool confirmation seeds changed")
    expected_contract = {
        "design_variant": config.design_variant,
        "counts": list(map(int, config.counts)),
        "original_discovery_seeds": list(map(int, config.discovery_seeds)),
        "original_confirmation_seeds": list(map(int, config.confirmation_seeds)),
    }
    if base_manifest.get("v6_stimulus_contract") != expected_contract:
        raise ValueError("Frozen base pool changed the V6 stimulus contract")

    rows = read_jsonl(base_stimuli)
    role_by_seed = {
        **{seed: "discovery" for seed in base_discovery},
        **{seed: "confirmation" for seed in base_confirmation},
    }
    expected_keys = {
        (seed, count)
        for seed in role_by_seed
        for count in map(int, config.counts)
    }
    observed_keys = [
        (int(row.get("seed", -1)), int(row.get("gold_count", -1)))
        for row in rows
    ]
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
        raise ValueError("Frozen base pool does not contain its exact seed/count grid")
    for row in rows:
        seed = int(row["seed"])
        if str(row.get("split")) != role_by_seed[seed]:
            raise ValueError("Frozen base pool seed role changed")
        if row.get("v6_replacement_candidate") is not True:
            raise ValueError("Frozen base pool lost its replacement-candidate marker")
        if row.get("v6_replacement_selection_outcomes_available") is not False:
            raise ValueError("Frozen base pool exposes selection outcomes")

    metadata = {
        "construction_mode": "incremental_frozen_pool_extension",
        "base_pool": str(base_pool.resolve()),
        "base_pool_manifest_sha256": sha256_file(base_manifest_path),
        "base_pool_stimuli_sha256": sha256_file(base_stimuli),
        "base_pool_policy": str(base_policy_path.resolve()),
        "base_pool_policy_sha256": sha256_file(base_policy_path),
        "base_rows_reused": len(rows),
        "extension_discovery_seeds": list(extension_discovery),
        "extension_confirmation_seeds": list(extension_confirmation),
    }
    return (
        [dict(row) for row in rows],
        [dict(value) for value in base_manifest.get("families", ())],
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    parser.add_argument("--source-stimuli", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-pool",
        type=Path,
        help="Validated frozen pool to reuse before generating only policy suffixes",
    )
    parser.add_argument(
        "--base-policy",
        type=Path,
        help="Policy whose hash and seed grid define --base-pool",
    )
    parser.add_argument("--haystack-dir", default="data/haystacks/paul_graham")
    parser.add_argument("--entities", default="data/entities/cities.csv")
    parser.add_argument(
        "--fact-templates", default="data/templates/niah_fact_single_template.txt"
    )
    args = parser.parse_args()

    config = V6Config.load(args.v6_config)
    policy = load_replacement_policy(args.replacement_policy, config)
    if sha256_file(args.source_stimuli) != EXPECTED_SOURCE_STIMULI_SHA256:
        raise ValueError("The frozen V4 source stimulus hash changed")
    discovery_pool = tuple(
        map(int, policy["discovery_replacement_seed_pool"])
    )
    confirmation_pool = tuple(
        map(int, policy["confirmation_replacement_seed_pool"])
    )
    if (args.base_pool is None) != (args.base_policy is None):
        raise ValueError("--base-pool and --base-policy must be provided together")
    # Two original anchors prove that this script is using the exact V4.4
    # deterministic construction procedure, one from each registered split.
    discovery_anchor = int(config.discovery_seeds[0])
    confirmation_anchor = int(config.confirmation_seeds[0])
    v4_config = V4Config(
        seeds=(
            discovery_anchor,
            *discovery_pool,
            confirmation_anchor,
            *confirmation_pool,
        ),
        discovery_seeds=(discovery_anchor, *discovery_pool),
        confirmation_seeds=(confirmation_anchor, *confirmation_pool),
    )
    v4_config.validate()
    tokenizer = TokenizerAdapter(
        v4_config.canonical_tokenizer,
        revision=v4_config.canonical_tokenizer_revision,
        cache_dir=str(args.cache_dir),
    )
    if tokenizer.backend != "huggingface":
        raise RuntimeError(
            "Pinned Hugging Face tokenizer is required for V6 reserve stimuli: "
            f"{tokenizer.load_error}"
        )
    freeze_spec = ControlledFreezeSpec(
        config=v4_config,
        haystack_dir=args.haystack_dir,
        entities_path=args.entities,
        fact_templates_path=args.fact_templates,
        tokenizer_cache_dir=str(args.cache_dir),
    )

    source_rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in read_jsonl(args.source_stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    anchor_audits: list[dict[str, Any]] = []
    for role, seed in (
        ("discovery", discovery_anchor),
        ("confirmation", confirmation_anchor),
    ):
        family, _ = build_controlled_family(
            variant="v4.4",
            seed=seed,
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
        )
        mismatches = []
        for row in family:
            key = (seed, int(row["gold_count"]))
            source = source_rows.get(key)
            if source is None or _anchor_signature(source) != _anchor_signature(row):
                mismatches.append(int(row["gold_count"]))
        if mismatches:
            raise RuntimeError(
                f"V4.4 reserve procedure failed {role} anchor seed={seed}: "
                f"counts={mismatches}"
            )
        anchor_audits.append(
            {
                "role": role,
                "seed": seed,
                "matched_counts": list(map(int, config.counts)),
                "status": "EXACT_MATCH",
            }
        )

    role_by_seed = {
        **{seed: "discovery" for seed in discovery_pool},
        **{seed: "confirmation" for seed in confirmation_pool},
    }
    incremental_metadata: dict[str, Any] = {
        "construction_mode": "full_deterministic_construction",
        "base_rows_reused": 0,
        "extension_discovery_seeds": list(discovery_pool),
        "extension_confirmation_seeds": list(confirmation_pool),
    }
    if args.base_pool is not None and args.base_policy is not None:
        rows, families, incremental_metadata = _load_frozen_base_pool(
            base_pool=args.base_pool,
            base_policy_path=args.base_policy,
            current_policy=policy,
            config=config,
            source_stimuli=args.source_stimuli,
        )
    else:
        rows = []
        families = []
    reused_seeds = {int(row["seed"]) for row in rows}
    extension_seeds = tuple(
        seed
        for seed in discovery_pool + confirmation_pool
        if seed not in reused_seeds
    )
    extension_rows_before = len(rows)
    current_policy_sha256 = sha256_file(args.replacement_policy)
    for index, seed in enumerate(extension_seeds, start=1):
        family, metadata = build_controlled_family(
            variant="v4.4",
            seed=seed,
            tokenizer=tokenizer,
            freeze_spec=freeze_spec,
        )
        role = role_by_seed[seed]
        if {int(row["gold_count"]) for row in family} != set(config.counts):
            raise RuntimeError(f"Replacement family seed={seed} is incomplete")
        for raw in family:
            row = dict(raw)
            row["split"] = role
            row["v6_replacement_candidate"] = True
            row["v6_replacement_seed_role"] = role
            row["v6_replacement_policy_sha256"] = current_policy_sha256
            row["v6_replacement_selection_outcomes_available"] = False
            rows.append(row)
        families.append({"seed_role": role, **metadata})
        print(
            f"[v6 replacement pool] {index}/{len(extension_seeds)} "
            f"role={role} seed={seed}",
            flush=True,
        )

    rows.sort(key=lambda row: (int(row["seed"]), int(row["gold_count"])))
    if len(rows) != len(role_by_seed) * len(config.counts):
        raise RuntimeError("V6 replacement stimulus row count changed")
    if len({str(row["stimulus_id"]) for row in rows}) != len(rows):
        raise RuntimeError("V6 replacement stimulus IDs are not unique")
    args.output.mkdir(parents=True, exist_ok=True)
    stimuli = args.output / "stimuli.jsonl"
    _atomic_jsonl(stimuli, rows)
    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "status": "PASS_AMENDMENT_RESERVE_POOL",
        "design_variant": "v4.4",
        "counts": list(map(int, config.counts)),
        "discovery_seeds": list(discovery_pool),
        "confirmation_seeds": list(confirmation_pool),
        "rows": len(rows),
        "rows_per_seed": len(config.counts),
        "source_stimuli": str(args.source_stimuli.resolve()),
        "source_stimuli_sha256": sha256_file(args.source_stimuli),
        "replacement_policy": str(args.replacement_policy.resolve()),
        "replacement_policy_sha256": sha256_file(args.replacement_policy),
        "v6_config_sha256": sha256_file(args.v6_config),
        "v6_stimulus_contract": {
            "design_variant": config.design_variant,
            "counts": list(map(int, config.counts)),
            "original_discovery_seeds": list(map(int, config.discovery_seeds)),
            "original_confirmation_seeds": list(
                map(int, config.confirmation_seeds)
            ),
        },
        "stimuli_sha256": sha256_file(stimuli),
        "anchor_regeneration_audits": anchor_audits,
        "original_generation_outputs_existed_before_pool_construction": True,
        "original_failure_rate_may_have_informed_pool_capacity_only": True,
        "reserve_model_outputs_available_during_pool_construction": False,
        "causal_intervention_outcomes_read_during_pool_construction": False,
        "intervention_outcomes_read": False,
        "families": families,
        **incremental_metadata,
        "extension_rows_generated": len(rows) - extension_rows_before,
        "historical_rows_preserved_byte_for_byte": bool(args.base_pool),
        "procedure_source_sha256": {
            "realistic_niah_v4/spec.py": hashlib.sha256(
                (SRC / "realistic_niah_v4" / "spec.py").read_bytes()
            ).hexdigest(),
            "realistic_niah_v4/stimuli.py": hashlib.sha256(
                (SRC / "realistic_niah_v4" / "stimuli.py").read_bytes()
            ).hexdigest(),
        },
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "rows": len(rows),
                "stimuli_sha256": manifest["stimuli_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
