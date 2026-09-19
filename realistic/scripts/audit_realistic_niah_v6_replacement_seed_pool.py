#!/usr/bin/env python3
"""Independently audit a frozen V6 replacement-stimulus pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--replacement-policy", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.pool / "manifest.json"
    stimuli_path = args.pool / "stimuli.jsonl"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy: dict[str, Any] = json.loads(
        args.replacement_policy.read_text(encoding="utf-8")
    )
    stimulus_lines = [
        line
        for line in stimuli_path.read_bytes().splitlines(keepends=True)
        if line.strip()
    ]
    rows = [json.loads(line) for line in stimulus_lines]

    policy_sha256 = sha256_file(args.replacement_policy)
    stimuli_sha256 = sha256_file(stimuli_path)
    discovery_seeds = [int(value) for value in policy["discovery_replacement_seed_pool"]]
    confirmation_seeds = [
        int(value) for value in policy["confirmation_replacement_seed_pool"]
    ]
    counts = [int(value) for value in policy["counts"]]
    expected_seed_roles = {
        **{seed: "discovery" for seed in discovery_seeds},
        **{seed: "confirmation" for seed in confirmation_seeds},
    }
    expected_pairs = {
        (seed, count)
        for seed in expected_seed_roles
        for count in counts
    }
    observed_pairs = [(int(row["seed"]), int(row["gold_count"])) for row in rows]

    require(len(observed_pairs) == len(set(observed_pairs)), "duplicate seed/count row")
    require(set(observed_pairs) == expected_pairs, "seed/count Cartesian grid mismatch")
    require(
        manifest["discovery_seeds"] == discovery_seeds,
        "manifest discovery seed order mismatch",
    )
    require(
        manifest["confirmation_seeds"] == confirmation_seeds,
        "manifest confirmation seed order mismatch",
    )
    require(manifest["counts"] == counts, "manifest count grid mismatch")
    require(manifest["rows"] == len(rows), "manifest row count mismatch")
    require(manifest["rows_per_seed"] == len(counts), "rows_per_seed mismatch")
    require(manifest["stimuli_sha256"] == stimuli_sha256, "stimuli SHA-256 mismatch")
    require(
        manifest["replacement_policy_sha256"] == policy_sha256,
        "manifest replacement-policy SHA-256 mismatch",
    )
    construction_mode = str(
        manifest.get("construction_mode", "full_deterministic_construction")
    )
    row_policy_sha256_counts = Counter(
        str(row["v6_replacement_policy_sha256"]) for row in rows
    )
    if construction_mode == "incremental_frozen_pool_extension":
        base_pool = Path(str(manifest.get("base_pool", "")))
        base_manifest_path = base_pool / "manifest.json"
        base_stimuli_path = base_pool / "stimuli.jsonl"
        require(base_manifest_path.is_file(), "incremental base manifest is missing")
        require(base_stimuli_path.is_file(), "incremental base stimuli are missing")
        require(
            sha256_file(base_manifest_path)
            == manifest.get("base_pool_manifest_sha256"),
            "incremental base manifest SHA-256 mismatch",
        )
        require(
            sha256_file(base_stimuli_path)
            == manifest.get("base_pool_stimuli_sha256"),
            "incremental base stimuli SHA-256 mismatch",
        )
        base_policy_sha256 = str(manifest.get("base_pool_policy_sha256", ""))
        base_policy_path = Path(str(manifest.get("base_pool_policy", "")))
        require(base_policy_path.is_file(), "incremental base policy is missing")
        require(
            sha256_file(base_policy_path) == base_policy_sha256,
            "incremental base policy SHA-256 mismatch",
        )
        base_lines = [
            line
            for line in base_stimuli_path.read_bytes().splitlines(keepends=True)
            if line.strip()
        ]
        base_rows_reused = int(manifest.get("base_rows_reused", -1))
        extension_rows_generated = int(manifest.get("extension_rows_generated", -1))
        require(base_rows_reused == len(base_lines), "base row reuse count mismatch")
        require(
            extension_rows_generated == len(rows) - len(base_lines),
            "extension row generation count mismatch",
        )
        require(
            manifest.get("historical_rows_preserved_byte_for_byte") is True,
            "historical row preservation flag is not true",
        )
        require(
            stimulus_lines[: len(base_lines)] == base_lines,
            "historical base rows were not preserved byte-for-byte",
        )
        extension_seeds = {
            *map(int, manifest.get("extension_discovery_seeds", ())),
            *map(int, manifest.get("extension_confirmation_seeds", ())),
        }
        require(extension_seeds, "incremental pool has no declared extension seeds")
        for row in rows:
            expected_policy_sha256 = (
                policy_sha256
                if int(row["seed"]) in extension_seeds
                else base_policy_sha256
            )
            require(
                row["v6_replacement_policy_sha256"] == expected_policy_sha256,
                f"row replacement-policy SHA-256 mismatch for seed {row['seed']}",
            )
    else:
        require(
            set(row_policy_sha256_counts) == {policy_sha256},
            "row replacement-policy SHA-256 mismatch",
        )

    for row in rows:
        seed = int(row["seed"])
        role = expected_seed_roles[seed]
        require(row["split"] == role, f"split mismatch for seed {seed}")
        require(
            row["v6_replacement_seed_role"] == role,
            f"replacement role mismatch for seed {seed}",
        )
        require(row["v6_replacement_candidate"] is True, "candidate flag is not true")
        require(
            row["v6_replacement_selection_outcomes_available"] is False,
            "selection outcomes were marked available",
        )

    family_roles = {
        int(family["seed"]): family["seed_role"] for family in manifest["families"]
    }
    require(family_roles == expected_seed_roles, "family seed/role coverage mismatch")
    require(manifest["intervention_outcomes_read"] is False, "outcomes-read flag true")
    require(
        manifest["causal_intervention_outcomes_read_during_pool_construction"]
        is False,
        "causal outcomes-read flag true",
    )
    require(
        manifest["reserve_model_outputs_available_during_pool_construction"] is False,
        "reserve model outputs existed during pool construction",
    )

    discovery_amendment = policy.get("pool_exhaustion_amendment")
    if discovery_amendment is not None:
        for key in (
            "frozen_before_extension_model_outputs",
            "confirmation_pool_unchanged",
        ):
            require(
                discovery_amendment[key] is True,
                f"discovery amendment flag {key} is not true",
            )
        for key in (
            "intervention_outcomes_read",
            "hidden_states_read",
            "attention_scores_read",
        ):
            require(
                discovery_amendment[key] is False,
                f"discovery amendment flag {key} is not false",
            )

    confirmation_amendment = policy.get("confirmation_pool_exhaustion_amendment")
    if confirmation_amendment is not None:
        for key in (
            "frozen_before_extension_model_outputs",
            "discovery_pool_unchanged",
        ):
            require(
                confirmation_amendment[key] is True,
                f"confirmation amendment flag {key} is not true",
            )
        for key in (
            "intervention_outcomes_read",
            "hidden_states_read",
            "attention_scores_read",
        ):
            require(
                confirmation_amendment[key] is False,
                f"confirmation amendment flag {key} is not false",
            )
        require(
            manifest.get("extension_confirmation_seeds")
            == confirmation_amendment.get("confirmation_extension_seeds"),
            "manifest/policy confirmation extension mismatch",
        )

    audits = {
        (entry["role"], int(entry["seed"])): entry
        for entry in manifest["anchor_regeneration_audits"]
    }
    anchor_specs = (
        ("discovery", int(policy["original_discovery_seeds"][0])),
        ("confirmation", int(policy["original_confirmation_seeds"][0])),
    )
    for key in anchor_specs:
        audit = audits.get(key)
        require(audit is not None, f"missing anchor audit {key}")
        require(audit["status"] == "EXACT_MATCH", f"anchor audit failed {key}")
        require(audit["matched_counts"] == counts, f"anchor count audit mismatch {key}")

    result = {
        "schema_version": "realistic_niah_v6_replacement_seed_pool_independent_audit_v1",
        "status": "PASS_AMENDMENT_RESERVE_POOL_INDEPENDENT_AUDIT",
        "manifest_sha256": sha256_file(manifest_path),
        "stimuli_sha256": stimuli_sha256,
        "replacement_policy_sha256": policy_sha256,
        "rows": len(rows),
        "rows_by_split": dict(Counter(row["split"] for row in rows)),
        "seeds_by_split": {
            role: sum(1 for value in expected_seed_roles.values() if value == role)
            for role in ("discovery", "confirmation")
        },
        "counts": counts,
        "construction_mode": construction_mode,
        "base_rows_reused": int(manifest.get("base_rows_reused", 0)),
        "extension_rows_generated": int(
            manifest.get("extension_rows_generated", len(rows))
        ),
        "historical_rows_preserved_byte_for_byte": bool(
            manifest.get("historical_rows_preserved_byte_for_byte", False)
        ),
        "row_policy_sha256_counts": dict(row_policy_sha256_counts),
        "anchor_regeneration_audits": list(manifest["anchor_regeneration_audits"]),
        "selection_outcomes_available": False,
        "intervention_outcomes_read": False,
        "hidden_states_read": False,
        "attention_scores_read": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
