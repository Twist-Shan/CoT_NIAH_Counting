"""Frozen inputs and fail-closed audits for the September Native supplement.

This module imports no model runtime. Historical rows remain immutable, and
new dense curves are assembled exclusively from measurements in the new run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

MODELS = ("Qwen3-8B", "Gemma4-E4B")
LAYERS = {"Qwen3-8B": 36, "Gemma4-E4B": 42}
OLD_LAYERS = {
    "Qwen3-8B": [0, 5, 10, 15, 20, 25, 30, 35],
    "Gemma4-E4B": [0, 6, 12, 18, 23, 29, 35, 41],
}
ANSWER_SEEDS = list(range(1254, 1264))
PROGRESS_SEEDS = list(range(1276, 1286))
ANSWER_ARMS = ("self_patch", "full_donor_patch")
PROGRESS_ARMS = ("receiver_self", "native_donor", "donor_to_receiver")
SCHEMA = "native_thinking_supplement_20260911_v1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    # Literal U+2028/U+2029 in archived passages are NOT JSONL delimiters.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_pairs(pairs: list[dict], model: str) -> list[list[int]]:
    ids = [r["pair_id"] for r in pairs]
    keys = [[int(r["seed"]), int(r["receiver_count"]), int(r["donor_count"])] for r in pairs]
    if len(ids) != 40 or len(set(ids)) != 40 or len({tuple(k) for k in keys}) != 40:
        raise ValueError("Expected exactly 40 unique directed pairs")
    if Counter(k[0] for k in keys) != Counter({s: 4 for s in ANSWER_SEEDS}):
        raise ValueError("Seed panel changed")
    for r, (seed, receiver, target) in zip(pairs, keys):
        if r["model_label"] != model or r["split"] != "confirmation":
            raise ValueError("Model/split changed")
        if receiver == target or not {receiver, target} <= set(range(1, 11)):
            raise ValueError("Invalid Target/Receiver counts")
        if r.get("alignment_key") != [seed, receiver, target]:
            raise ValueError("Alignment key disagrees with pair")
        if r.get("pair_selection_uses_patch_outcome") is not False:
            raise ValueError("Pair selection must remain outcome blind")
        if not r.get("receiver_exact_count") or not r.get("donor_exact_count"):
            raise ValueError("Both archived traces must be clean-correct")
        if {r["receiver_site_id"], r["donor_site_id"]} != {"answer_query_v3"}:
            raise ValueError("Patch site changed")
    return keys


def answer_key(row: dict) -> tuple[str, int, str]:
    return str(row["pair_id"]), int(row["layer"]), str(row["condition"])


def validate_answer(rows: list[dict], pairs: list[dict], layers: list[int], *, complete: bool = True) -> dict:
    registry = {r["pair_id"]: r for r in pairs}
    expected = {(p, l, a) for p in registry for l in layers for a in ANSWER_ARMS}
    actual: set[tuple] = set()
    for r in rows:
        key = answer_key(r)
        if key in actual or key not in expected:
            raise ValueError(f"Duplicate/unregistered answer trial: {key}")
        actual.add(key)
        p = registry[key[0]]
        for field, required in {
            "request_id": p["receiver_request_id"], "donor_request_id": p["donor_request_id"],
            "model_label": p["model_label"], "seed": p["seed"], "gold_count": p["receiver_count"],
            "donor_count": p["donor_count"], "receiver_site_id": "answer_query_v3", "donor_site_id": "answer_query_v3",
        }.items():
            if r.get(field) != required:
                raise ValueError(f"Answer trial metadata mismatch: {key}, {field}")
        if r["condition"] == "self_patch" and r.get("prediction") != p["receiver_count"]:
            raise ValueError(f"Self patch did not regenerate Receiver gold: {key}")
        if not isinstance(r.get("generated_token_count"), int) or not 0 < r["generated_token_count"] <= 16:
            raise ValueError(f"Invalid generation length: {key}")
        if not isinstance(r.get("completion_text_raw"), str):
            raise ValueError(f"Missing literal continuation: {key}")
    if complete and actual != expected:
        raise ValueError(f"Incomplete answer grid: {len(actual)}/{len(expected)}")
    return {"status": "PASS", "trials": len(actual), "expected_trials": len(expected), "layers": layers}


def progress_key(row: dict) -> tuple[int, int, int, str]:
    return int(row["seed"]), int(row["receiver_occurrence_j"]), int(row["donor_occurrence_k"]), str(row["condition"])


def validate_progress(rows: list[dict], targets: list[int], *, direction: str) -> dict:
    offset = {"forward": -1, "backward": 1}[direction]
    expected = {(s, k + offset, k, a) for s in PROGRESS_SEEDS for k in targets for a in PROGRESS_ARMS}
    actual: set[tuple] = set()
    for r in rows:
        key = progress_key(r)
        if key in actual or key not in expected:
            raise ValueError(f"Duplicate/unregistered progress trial: {key}")
        actual.add(key)
        if r.get("layer") != 16 or r.get("gold_count") != 10 or r.get("patch_scope") != "item_span":
            raise ValueError(f"Progress intervention changed: {key}")
        if r.get("cohort_mode") != "prompt_conditioned_noindex" or r.get("patch_applications") != 1:
            raise ValueError(f"Progress condition changed: {key}")
        if r.get("donor_successor") != key[2] + 1 or r.get("receiver_successor") != key[1] + 1:
            raise ValueError(f"Successor labels changed: {key}")
        for field in ("donor_vs_receiver_sum_logodds", "donor_vs_receiver_attention_log_ratio"):
            if not math.isfinite(float(r[field])):
                raise ValueError(f"Nonfinite progress readout: {key}")
        if r["condition"] != "native_donor":
            if "generated_known_city_ordinals_any_surface" not in r or "completion_text" not in r:
                raise ValueError(f"Missing greedy continuation: {key}")
            if not 0 < r.get("generated_token_count", 0) <= 96:
                raise ValueError(f"Invalid continuation budget: {key}")
            computed = r.get("first_generated_known_city_ordinal") == key[2] + 1
            if r.get("greedy_donor_successor_adoption") != computed:
                raise ValueError(f"Adoption flag disagrees with first city: {key}")
    if actual != expected:
        raise ValueError(f"Incomplete progress grid: {len(actual)}/{len(expected)}")
    return {"status": "PASS", "trials": len(actual), "seed_clusters": 10, "direction": direction}


def compare_replay(old: list[dict], new: list[dict], *, progress: bool = False) -> dict:
    key_fn = progress_key if progress else answer_key
    left, right = ({key_fn(r): r for r in group} for group in (old, new))
    if len(left) != len(old) or len(right) != len(new) or left.keys() != right.keys():
        raise ValueError("Replay key multiset differs")
    fields = (
        ("completion_text", "first_generated_known_city_ordinal", "donor_successor_argmax", "shared_commit_position", "patch_width", "targeted_bank_sha256")
        if progress else
        ("prediction", "completion_text_raw", "generated_token_count", "source_positions")
    )
    changes = [
        {"key": list(k), "fields": [f for f in fields if left[k].get(f) != right[k].get(f)]}
        for k in sorted(left) if any(left[k].get(f) != right[k].get(f) for f in fields)
    ]
    return {"status": "PASS" if not changes else "MISMATCH", "compared_trials": len(left), "changed_trials": len(changes), "changes": changes}


def verify_bundle(bundle: Path) -> dict:
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    for relative, expected in plan["files_sha256"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != expected:
            raise ValueError(f"Frozen bundle mismatch: {relative}")
    core = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if canonical_digest(core) != plan["plan_sha256"]:
        raise ValueError("Plan hash mismatch")
    return plan


def validate_answer_shards(root: Path, pairs: list[dict], layers: list[int]) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        try:
            layer = int(path.parent.name.removeprefix("L"))
            index = int(path.stem) - 1
            if layer not in layers or not 0 <= index < len(pairs):
                raise ValueError("Unregistered shard path")
            validate_answer(read_rows(path), [pairs[index]], [layer])
        except (ValueError, KeyError) as error:
            raise ValueError(f"Invalid resume shard {path}: {error}") from error


def summarize_backward(rows: list[dict]) -> dict:
    """Seed-cluster estimates; availability is structural, truncations stay failures."""
    import numpy as np

    audit = validate_progress(rows, [4, 6, 8], direction="backward")
    by_key = {progress_key(r): r for r in rows}
    per_seed = []
    hop_rows = []
    for seed in PROGRESS_SEEDS:
        effects = []
        for k in (4, 6, 8):
            self_row, patch = (by_key[(seed, k + 1, k, a)] for a in ("receiver_self", "donor_to_receiver"))
            effects.append(int(patch["greedy_donor_successor_adoption"]) - int(self_row["greedy_donor_successor_adoption"]))
            cities = patch["generated_known_city_ordinals_any_surface"]
            for hop in range(1, 5):
                available = k + hop <= 10
                previous = cities[:hop - 1] == list(range(k + 1, k + hop))
                eligible = available and (hop == 1 or previous)
                hit = eligible and len(cities) >= hop and cities[hop - 1] == k + hop
                hop_rows.append({"seed": seed, "target_k": k, "hop": hop, "available": available, "eligible": eligible, "success": hit})
        per_seed.append({"seed": seed, "adoption_gain": float(np.mean(effects))})
    values = np.array([r["adoption_gain"] for r in per_seed])
    rng = np.random.default_rng(20260911)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {
        **audit, "schema_version": SCHEMA,
        "claim_scope": "Gemma prompt-conditioned backward extension on an already observed cohort",
        "pristine_confirmation": False, "target_k": [4, 6, 8],
        "adoption_gain_patch_minus_self": float(values.mean()),
        "adoption_gain_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "bootstrap_unit": "seed cluster after equal averaging over the three target steps",
        "bootstrap_repetitions": 10000, "per_seed": per_seed,
        "stepwise": [
            {"hop": h, "successes": sum(r["success"] for r in hop_rows if r["hop"] == h),
             "eligible": sum(r["eligible"] for r in hop_rows if r["hop"] == h),
             "structurally_available": sum(r["available"] for r in hop_rows if r["hop"] == h)}
            for h in range(1, 5)
        ],
        "hop_rows": hop_rows,
        "truncation_policy": "missing expected cities count as failures; exclude only k+hop>N",
    }
