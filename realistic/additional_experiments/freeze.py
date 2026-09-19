"""Freeze inputs and legacy mechanisms before new-task model inference."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from protocol import (TASKS, audit_case, make_case, read_jsonl, sha256,
                      text_hash, user_prompt, validate_config, write_json)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_HEAD_MEMBERSHIP = REPO / "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv"
DEFAULT_NATIVE_BASIS_ROOT = REPO / "work/v5_native_count_stream/representation_20260820"


def freeze(config_path: Path, source: Path, output: Path, *, smoke: bool = False,
           head_membership: Path | None = None, native_basis_root: Path | None = None) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if smoke:
        config["discovery_seeds"] = config["discovery_seeds"][:1]
        config["confirmation_seeds"] = config["confirmation_seeds"][:1]
        config["levels"] = [config["levels"][0], config["levels"][-1]]
    config["run_type"] = "smoke" if smoke else "exploratory_pilot"
    validate_config(config)
    membership = Path(head_membership) if head_membership is not None else DEFAULT_HEAD_MEMBERSHIP
    basis_root = Path(native_basis_root) if native_basis_root is not None else DEFAULT_NATIVE_BASIS_ROOT
    basis_files = [
        (model, name, basis_root / model / name)
        for model in config["models"]
        for name in ("item_end_discovery_basis.json", "item_end_discovery_basis.npz")
    ]
    missing = [str(path) for path in (source, membership, *(p for _, _, p in basis_files))
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing additional-task prerequisites (no output created):\n" +
                                "\n".join(missing) +
                                "\nSet --source, --head-membership, and --native-basis-root; see docs/REPRODUCING.md.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for _, name, path in basis_files:
        if name.endswith(".json"):
            json.loads(path.read_text(encoding="utf-8"))
    seeds = config["discovery_seeds"] + config["confirmation_seeds"]
    source_rows = [r for r in read_jsonl(source) if r["design_variant"] == "v4.4" and r["seed"] in seeds]
    lookup = {(r["seed"], r["gold_count"]): r for r in source_rows}
    if len(lookup) != len(source_rows):
        raise ValueError("Duplicate source seed/count keys")
    cases = []
    for seed in seeds:
        split = "discovery" if seed in config["discovery_seeds"] else "confirmation"
        for level in config["levels"]:
            for task in TASKS:
                n = level if task == "count_all" else config["total_records_new_tasks"]
                if (seed, n) not in lookup:
                    raise ValueError(f"Missing source row {seed}/{n}")
                cases.append(make_case(lookup[seed, n], task, level, split))
    for case in cases:
        audit_case(case)
    with membership.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    banks = {}
    for model, source_n, keep_n in (("Qwen3-8B", 32, 32), ("Gemma4-E4B", 8, 6)):
        selected = sorted((r for r in rows if r["model_label"] == model and
                           int(r["top_n"]) == source_n and int(r["rank"]) <= keep_n),
                          key=lambda r: int(r["rank"]))
        if len(selected) != keep_n or len({(r["layer"], r["head"]) for r in selected}) != keep_n:
            raise ValueError("Legacy head membership failed")
        banks[model] = [[int(r["layer"]), int(r["head"])] for r in selected]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "frozen_banks.json", {
        "source_file": str(membership.resolve()), "source_sha256": sha256(membership),
        "source_mode": "nonthinking", "banks": banks,
        "new_task_outcomes_used_for_selection": False,
        "note": "Same legacy broad bank tested in both modes; this is not a native item-retrieval head bank.",
    })
    with (output / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with (output / "user_prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            for mode in config["modes"]:
                prompt = user_prompt(case, mode)
                f.write(json.dumps({"case_id": case["case_id"], "mode": mode,
                                    "user_text": prompt, "user_text_sha256": text_hash(prompt)}, ensure_ascii=False) + "\n")
                if case["seed"] == config["confirmation_seeds"][0] and case["level"] == config["levels"][0]:
                    example = output / "prompt_examples" / f"{case['task']}__{mode}.txt"
                    example.parent.mkdir(exist_ok=True)
                    example.write_text(prompt, encoding="utf-8")
    bases = []
    for model, name, src in basis_files:
        dest = output / "legacy_native_bases" / model / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        bases.append({"path": str(dest.relative_to(output)), "source": str(src.resolve()), "sha256": sha256(dest)})
    dependencies = {str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "src").rglob("*.py"))}
    write_json(output / "legacy_source_hashes.json", dependencies)
    audit = {
        "status": "PASS", "source": str(source.resolve()), "source_sha256": sha256(source),
        "source_mutated": False, "case_count": len(cases),
        "user_prompt_count": len(cases) * len(config["modes"]),
        "expected_model_mode_cases": len(cases) * len(config["models"]) * len(config["modes"]),
        "gold_and_offsets_checked": True, "new_task_total_records_fixed": config["total_records_new_tasks"],
        "legacy_native_bases": bases,
        "files": {name: sha256(output / name) for name in
                  ("config.json", "cases.jsonl", "user_prompts.jsonl", "frozen_banks.json", "legacy_source_hashes.json")},
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_inference_completed": False,
    }
    write_json(output / "freeze_audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_v1.json")
    parser.add_argument("--source", type=Path, default=REPO / "work/nonthinking_report_filestream_stage3/stimuli.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-membership", type=Path, default=DEFAULT_HEAD_MEMBERSHIP,
                        help="full_span_topk_membership.csv produced by analyze_realistic_niah_v4_4_full_span_topk.py")
    parser.add_argument("--native-basis-root", type=Path, default=DEFAULT_NATIVE_BASIS_ROOT,
                        help="Archived pilot input directory containing <model>/item_end_discovery_basis.{json,npz}; see docs/COMPLETENESS.md for the unresolved export source")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, args.source, args.output, smoke=args.smoke,
                           head_membership=args.head_membership, native_basis_root=args.native_basis_root), indent=2))
