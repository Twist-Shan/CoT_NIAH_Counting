#!/usr/bin/env python3
"""Restore the frozen full-commit -> targeted-query analysis for V6.

The V6 native-loop runner prospectively generated the full-donor, self-patch,
count-subspace and norm-matched orthogonal arms inherited from the Native-
thinking protocol.  The first V6 report-tail analyzer only gated the narrower
count-subspace intervention.  This adapter analyzes the already-generated
full-state arms with the original frozen contrast while preserving V6's
analysis-slot -> true-source-seed replacement contract.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import sha256_file  # noqa: E402
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from scripts import (  # noqa: E402
    analyze_realistic_niah_v5_commit_state_to_targeted_query as legacy,
)
from scripts.analyze_realistic_niah_v6_native_loop import (  # noqa: E402
    _coherent_contract,
)


SCHEMA_VERSION = "realistic_niah_v6_full_commit_to_targeted_query_analysis_v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def analyze_with_true_source_seeds(
    frame: Any,
    *,
    phase: str,
    true_source_seeds: Sequence[int],
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run the frozen V5 contrast on V6 true-source identities.

    V6 replacement rows retain their original analysis-slot membership while
    all seed-level inference must use the actual source seed.  The legacy
    analyzer validates a module-level seed contract, so this adapter changes
    that validation constant only and restores it even if analysis fails.
    """

    expected = tuple(map(int, true_source_seeds))
    if len(expected) not in (10, 20) or len(set(expected)) != len(expected):
        raise ValueError("V6 full-commit analysis requires 10 or 20 unique seeds")
    observed = set(frame["seed"].astype(int))
    if observed != set(expected):
        raise ValueError(
            "Full-commit trials changed the true-source panel: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    attribute = (
        "COUNT_STREAM_DISCOVERY_SEEDS"
        if phase == "discovery"
        else "COUNT_STREAM_CONFIRMATION_SEEDS"
    )
    previous = getattr(legacy, attribute)
    setattr(legacy, attribute, expected)
    try:
        estimands, seed_effects, gates = legacy.analyze(
            frame,
            phase=phase,
            bootstrap_samples=int(bootstrap_samples),
            random_seed=int(random_seed),
        )
    finally:
        setattr(legacy, attribute, previous)
    gates = dict(gates)
    gates.update(
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_status": (
                "FROZEN_NATIVE_REPORT_CONTRAST_RESTORED_DISCOVERY"
                if phase == "discovery"
                else "FROZEN_NATIVE_REPORT_CONTRAST_RESTORED_CONFIRMATION"
            ),
            "registered_true_source_seeds": list(expected),
            "preexisting_arm_generation": True,
            "model_trials_recomputed": False,
            "frozen_k_changed": False,
            "scientific_scope": (
                "full completed-item commit hidden state -> frozen targeted-query "
                "attention and teacher-forced next-city distribution"
            ),
            "gate_interpretation": (
                "The direct edge is established by full-donor minus self and "
                "full-donor minus norm-matched-orthogonal targeted-attention "
                "contrasts. Greedy city adoption is a downstream diagnostic, "
                "not a required gate."
            ),
        }
    )
    return estimands, seed_effects, gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    parser.add_argument("--confirmation-freeze", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = V6Config.load(args.v6_config)
    if args.phase == "confirmation":
        if args.confirmation_freeze is None:
            raise ValueError("Full-commit confirmation analysis requires a freeze")
        validate_confirmation_freeze(
            args.confirmation_freeze,
            prompt_mode=config.prompt_mode,
            model_label=args.model,
        )
    elif args.confirmation_freeze is not None:
        raise ValueError("Discovery analysis must not open a confirmation freeze")

    contract = _coherent_contract(
        args.cohort_registry,
        config=config,
        model_label=args.model,
        phase=args.phase,
    )
    frame = legacy._read_trials(args.trials)
    estimands, seed_effects, gates = analyze_with_true_source_seeds(
        frame,
        phase=args.phase,
        true_source_seeds=contract["true_source_seeds"],
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    gates["model_label"] = args.model
    gates["prompt_mode"] = config.prompt_mode
    gates["panel_membership_identity"] = "analysis_slot_seed"
    gates["statistical_identity"] = "true_source_seed"
    legacy._atomic_csv(args.output / "estimands.csv", estimands)
    legacy._atomic_csv(args.output / "seed_effects.csv", seed_effects)
    legacy._atomic_json(args.output / "claim_gates.json", gates)

    manifest = {
        "schema_version": (
            "realistic_niah_v6_full_commit_to_query_analysis_manifest_v1"
        ),
        "status": "PASS",
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "phase": args.phase,
        "contract": contract,
        "trial_root": str(args.trials.resolve()),
        "trial_manifest_sha256": sha256_file(args.trials / "manifest.json"),
        "claim_gates_sha256": sha256_file(args.output / "claim_gates.json"),
        "confirmation_freeze_sha256": (
            sha256_file(args.confirmation_freeze)
            if args.confirmation_freeze is not None
            else None
        ),
        "model_trials_recomputed": False,
        "existing_full_state_arms_reused": True,
        "seed_aliasing": False,
        "frozen_k_changed": False,
    }
    _atomic_json(args.output / "v6_analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase": args.phase,
                "seed_count": len(contract["true_source_seeds"]),
                "directional_signal_pass": bool(gates["directional_signal_pass"]),
                "strong_direct_gate_pass": bool(gates["strong_direct_gate_pass"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
