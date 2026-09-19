"""CPU checks and source-seed estimands for frozen answer and retrieval assays."""
from __future__ import annotations
import ast
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from .read_audit import require


def unique_grid(rows, fields, expected_keys):
    indexed = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in indexed, f"Duplicate trial: {key}")
        indexed[key] = row
    expected = set(expected_keys)
    require(set(indexed) == expected,
            f"Incomplete grid: missing={len(expected - indexed.keys())}, extra={len(indexed.keys() - expected)}")
    return indexed


def verify_generation(raw, max_tokens):
    ids = raw["generated_token_ids"]
    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids), "Invalid saved token IDs")
    require(0 < len(ids) == raw["generated_token_count"] <= max_tokens, "Generation length mismatch")
    eos = ids[-1] in raw["generation_eos_token_ids"]
    require(raw["stopped_on_eos"] == eos, "EOS flag mismatch")
    require(raw["generation_truncated"] == (len(ids) >= max_tokens and not eos), "Truncation flag mismatch")
    require(eos or len(ids) == max_tokens, "Unexplained early generation stop")
    require(all(isinstance(raw[k], str) for k in ("completion_text", "completion_text_raw")), "Missing saved output text")


def seed_mean(observations, *, repetitions=10000, random_seed=20260915):
    """Average repetitions/pairs within each seed, then weight seeds equally."""
    require(repetitions >= 2, "Bootstrap needs at least two draws")
    groups = defaultdict(list)
    for source_seed, value in observations:
        require(type(source_seed) is int and np.isfinite(value), "Invalid source seed or nonfinite outcome")
        groups[source_seed].append(float(value))
    require(bool(groups), "Empty outcome population")
    seeds = sorted(groups)
    values = np.asarray([np.mean(groups[s]) for s in seeds], dtype=float)
    interval = None
    if len(seeds) >= 2:
        rng = np.random.default_rng(random_seed)
        boot = values[rng.integers(0, len(seeds), (repetitions, len(seeds)))].mean(axis=1)
        interval = np.quantile(boot, [.025, .975]).tolist()
    return {"estimate": float(values.mean()), "ci95": interval, "independent_seed_count": len(seeds),
            "observation_count": sum(map(len, groups.values())), "observation_sum": sum(map(sum, groups.values())),
            "by_seed": [{"seed": s, "observations": len(groups[s]), "mean": float(values[i])} for i, s in enumerate(seeds)],
            "aggregation": "equal-weight source-seed means", "bootstrap_draws": repetitions,
            "bootstrap_seed": random_seed, "ci_status": "available" if interval else "fewer_than_two_seeds"}


def load_frozen_city_scorer(path):
    """Load only the frozen pure-text definitions, avoiding GPU-only imports.

    The AST nodes are compiled unchanged. No parser source is copied, edited,
    or substituted; callers record the SHA256 of the entire frozen source.
    """
    path = Path(path)
    functions = {"_first_generated_gold_city", "_first_generated_city_record", "_retrieval_behavior_score"}
    constants = {"_CITY_NAME_FRAGMENT", "_GENERATED_CITY_RECORD_PATTERNS", "_NON_CITY_RECORD_LABELS"}
    selected, found = [], set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        names = set()
        if isinstance(node, ast.FunctionDef):
            names = {node.name} & functions
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)} & constants
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = {node.target.id} & constants
        if names:
            selected.append(node)
            found.update(names)
    require(found == functions | constants, "Frozen city parser definitions incomplete")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {"re": re}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_retrieval_behavior_score"]
