"""Shared estimands for archived Thinking and Enumeration outcomes.

This module changes reporting only. It does not refit probes, reparse text,
select successful trials, or make historical cohorts experimentally matched.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def exact_prefix_depth(observed: list[int], expected: list[int]) -> int:
    depth = 0
    for actual, target in zip(observed, expected):
        if actual != target:
            break
        depth += 1
    return depth


def continuation_counts(rows: Iterable[Mapping[str, Any]], hop: int) -> dict[str, Any]:
    """Count eligible horizons separately from prior-prefix successes.

    A truncation/empty parse stays in the horizon denominator. Success means an
    exact prefix of the saved city ordinals, not a correct score or final count.
    """
    if type(hop) is not int or hop < 1:
        raise ValueError("hop must be a positive integer")
    per_seed: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    total = truncated = 0
    for row in rows:
        k, n = row["donor_occurrence_k"], row["gold_count"]
        observed = row["generated_known_city_ordinals_any_surface"]
        if type(k) is not int or type(n) is not int or not 0 <= k < n:
            raise ValueError("Invalid donor progress / count")
        if not isinstance(observed, list) or any(type(x) is not int for x in observed):
            raise ValueError("Saved city ordinals must be a list of integers")
        if type(row["seed"]) is not int or type(row["generation_truncated"]) is not bool:
            raise ValueError("Expected a true-source integer seed and Boolean truncation flag")
        counts = per_seed[row["seed"]]
        total += 1
        truncated += int(row["generation_truncated"])
        if n - k < hop:
            continue
        depth = exact_prefix_depth(observed, list(range(k + 1, n + 1)))
        counts[0] += 1  # All cases with enough remaining records.
        counts[1] += int(depth >= hop - 1)
        counts[2] += int(depth >= hop)
    if total == 0:
        raise ValueError("No continuation rows")
    return {"hop": hop, "total_trials": total, "truncated_trials": truncated,
            "seed_counts": {str(s): per_seed[s] for s in sorted(per_seed)}}


def continuation_summary(rows: list[Mapping[str, Any]], hop: int, *,
                         draws: int, random_seed: int) -> dict[str, Any]:
    if draws < 2:
        raise ValueError("At least two bootstrap draws are required")
    out = continuation_counts(rows, hop)
    counts = np.asarray(list(out["seed_counts"].values()), dtype=np.int64)
    horizon, conditional, successes = counts.sum(axis=0).tolist()
    out.update(horizon_eligible=horizon, conditional_eligible=conditional,
               successes=successes, horizon_not_applicable=out["total_trials"] - horizon,
               source_seed_count=len(counts),
               aggregation="pooled trial ratio; bootstrap resamples true-source seed clusters")
    rng = np.random.default_rng(random_seed)
    boot = counts[rng.integers(0, len(counts), (draws, len(counts)))].sum(axis=1)
    for name, column, denominator in (("unconditional", 0, horizon),
                                      ("conditional", 1, conditional)):
        valid = boot[:, column] > 0
        usable = len(counts) >= 2 and valid.mean() >= .99
        ci = np.quantile(boot[valid, 2] / boot[valid, column], [.025, .975]).tolist() if usable else None
        out[name] = {"estimate": successes / denominator if denominator else None,
                     "ci95": ci, "valid_bootstrap_draws": int(valid.sum()),
                     "bootstrap_draws": draws, "bootstrap_seed": random_seed,
                     "ci_status": "available" if usable else "insufficient_nonzero_denominators"}
    return out


def relay_point(natural: float, remaining: float, *, epsilon: float = 1e-8) -> dict[str, Any]:
    if not np.isfinite([natural, remaining]).all():
        raise ValueError("Relay damage must be finite")
    estimable = abs(natural) > epsilon
    return {"natural_damage": float(natural), "remaining_signed_damage": float(remaining),
            "paired_interaction": float(natural - remaining),
            "signed_reduction": float(1 - remaining / natural) if estimable else None,
            "absolute_reduction": float(1 - abs(remaining) / abs(natural)) if estimable else None,
            "ratio_status": "available" if estimable else "natural_damage_near_zero"}


def relay_summary(rows: list[Mapping[str, Any]], *, draws: int,
                  random_seed: int) -> dict[str, Any]:
    """Average directed pairs within seed, then seeds equally; paired bootstrap."""
    per_seed: dict[int, list[list[float]]] = defaultdict(list)
    for row in rows:
        per_seed[int(row["seed"])].append([row["natural_damage"], row["remaining_signed_damage"]])
    if len(per_seed) < 2 or draws < 2:
        raise ValueError("Relay intervals require at least two seeds and two draws")
    values = np.asarray([np.mean(per_seed[s], axis=0) for s in sorted(per_seed)], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite relay pair values")
    out = relay_point(*values.mean(axis=0))
    rng = np.random.default_rng(random_seed)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    valid = np.abs(boot[:, 0]) > 1e-8
    metrics = {"natural_damage": boot[:, 0], "remaining_signed_damage": boot[:, 1],
               "paired_interaction": boot[:, 0] - boot[:, 1]}
    if valid.mean() >= .99 and out["ratio_status"] == "available":
        metrics.update(signed_reduction=1 - boot[valid, 1] / boot[valid, 0],
                       absolute_reduction=1 - np.abs(boot[valid, 1] / boot[valid, 0]))
    out.update(ci95={k: np.quantile(v, [.025, .975]).tolist() for k, v in metrics.items()},
               source_seed_count=len(values), directed_pair_count=len(rows),
               valid_ratio_bootstrap_draws=int(valid.sum()), bootstrap_draws=draws,
               bootstrap_seed=random_seed, aggregation="equal-weight seed means of paired effects",
               ratio_ci_status="available" if "signed_reduction" in metrics else "unstable_denominator")
    return out
