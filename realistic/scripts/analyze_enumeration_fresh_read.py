"""Verify/extract a completed Read backup and audit its five-arm results on CPU."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.read_audit import audit_cell, read, require, sha


def frozen_parser(root):
    folder = root / "fresh_causal_v1/code/src/realistic_niah"
    name = "_frozen_read_parser"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    for child in ("spec", "parsing"):
        spec = importlib.util.spec_from_file_location(f"{name}.{child}", folder / f"{child}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module.parse_total


def extract_checked(archive, target):
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            require(member.isfile(), f"Non-file archive member: {member.name}")
            destination = (target / member.name).resolve()
            require(destination.is_relative_to(target.resolve()) and destination not in seen,
                    f"Unsafe or duplicate archive path: {member.name}")
            seen.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            with destination.open("xb") as output:
                for block in iter(lambda: source.read(1048576), b""):
                    output.write(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    require(sha(args.archive) == args.expected_sha256, "Archive transfer hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    extracted = args.output / "source"
    extract_checked(args.archive, extracted)
    manifest = read(extracted / "backup_manifest.json")
    for name, record in manifest["files"].items():
        file = extracted / name
        require(file.stat().st_size == record["bytes"] and sha(file) == record["sha256"],
                f"Archived member hash mismatch: {name}")
    timings = {"extract_and_hash_seconds": time.monotonic() - tick}
    cfg = read(extracted / "fresh_v1/protocol.json")
    parse_total = frozen_parser(extracted)
    cells = []
    for model in cfg["models"]:
        for mode in cfg["modes"]:
            cell_tick = time.monotonic()
            cell = audit_cell(extracted, model, mode, parse_total, cfg)
            cell["audit_seconds"] = time.monotonic() - cell_tick
            cells.append(cell)
            print(json.dumps({"model": model, "mode": mode, "audit": cell["audit"],
                              "rows": cell["formal_rows"], "seconds": cell["audit_seconds"]}), flush=True)
    report = {"schema": "enumeration_fresh_read_audit_v1", "status": "PASS",
              "utc": datetime.now(timezone.utc).isoformat(), "cells": cells,
              "archive_sha256": args.expected_sha256,
              "backup_manifest_sha256": sha(extracted / "backup_manifest.json"),
              "member_hashes_verified": len(manifest["files"]),
              "script_sha256": sha(__file__), "module_sha256": sha(ROOT / "src/realistic_niah_v6/read_audit.py"),
              "frozen_parser_sha256": sha(extracted / "fresh_causal_v1/code/src/realistic_niah/parsing.py"),
              "bootstrap": cfg["bootstrap"], "command": sys.argv,
              "estimand": "Input-weighted accuracy on legal-query inputs; five matched arms, no baseline-correctness filtering.",
              "limitations": ["No GPU rerun or tokenizer decode in this CPU audit.",
                  "Generation parsing retains the frozen Total parser, including valid answers emitted before truncation.",
                  "Two unavailable Qwen Thinking queries remain unexecuted, not scored as failures.",
                  "Gemma prompted FOUND is auxiliary, not natural Native Read.",
                  "Intervals resample source seeds; count inputs are not independent repetitions."],
              "timings": {**timings, "total_seconds": time.monotonic() - tick}}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# Fresh Read audit", "", "All archived input/output hashes and five-arm integrity checks passed.",
             "Accuracy uses all legal-query inputs; original 100-input coverage is retained.",
             "Intervals are 95% source-seed cluster bootstrap percentiles (10,000 draws, seed 20260915).",
             "Truncated completions remain in the denominator and follow the frozen Total parser.", "",
             "| Model / mode | Condition | Correct / eligible | 95% CI (%) | Parsed | Truncated | Correct & truncated |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for cell in cells:
        label = f'{cell["model"]} / {cell["mode"]}'
        if cell["comparison_role"].startswith("prompted"):
            label += " (prompted auxiliary)"
        for condition, item in cell["conditions"].items():
            stat = item["accuracy"]
            ci = stat["ci95"]
            ci_text = "descriptive only" if ci is None else f"{100*ci[0]:.1f}--{100*ci[1]:.1f}"
            lines.append(f'| {label} | {condition} | {int(stat["numerator"])}/{stat["denominator"]} | '
                         f'{ci_text} | {item["parsed"]} | {item["truncated"]} | {item["correct_but_truncated"]} |')
    lines.extend(["", "## Coverage and clean replay", ""])
    for cell in cells:
        lines.append(f'- {cell["model"]} / {cell["mode"]}: {cell["eligible_inputs"]}/100 legal queries; '
                     f'{cell["baseline_correct_population"]}/100 original baseline correct; '
                     f'{len(cell["clean_replay_disagreements"])} clean replay prediction changes.')
    lines.extend(["", "CPU audit verifies saved evidence and recomputes statistics; it does not repeat GPU inference."])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_rows": sum(c["formal_rows"] for c in cells),
                      "hashes": len(manifest["files"]), "seconds": report["timings"]["total_seconds"]}))


if __name__ == "__main__":
    main()

