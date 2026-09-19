"""Verify a completed Update archive and recompute paired continuation metrics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v6.update_audit import audit_cell
from realistic_niah_v6.read_audit import read, require, sha
from analyze_enumeration_fresh_read import extract_checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    require(sha(a.archive) == a.expected_sha256, "Archive transfer hash mismatch")
    if not a.source.exists():
        extract_checked(a.archive, a.source)
    inventory = read(a.source / "backup_manifest.json")
    require(inventory["schema"] == "enumeration_update_backup_v1", "Wrong backup type")
    for name, record in inventory["files"].items():
        file = a.source / name
        require(file.stat().st_size == record["bytes"] and sha(file) == record["sha256"], f"Member mismatch: {name}")
    update = a.source / "fresh_native_update_v1"
    manifest, assembly = read(update / "manifest.json"), read(update / "assembly_audit.json")
    for path, expected in assembly["source_files_sha256"].items():
        local = a.source / PurePosixPath(path).relative_to(PurePosixPath(inventory["source_root"]))
        require(sha(local) == expected, "Assembly source hash mismatch")
    for name, expected in (("fresh_v1/manifest.json", manifest["baseline_manifest_sha256"]),
                           ("fresh_causal_v1/code_manifest.json", manifest["causal_code_manifest_sha256"]),
                           ("fresh_causal_v1/cohorts.json", manifest["old_cohorts_sha256"]),
                           ("fresh_native_update_v1/protocol.json", manifest["protocol_sha256"])):
        require(sha(a.source / name) == expected, "Frozen provenance mismatch")
    baseline = read(a.source / "fresh_v1/manifest.json")
    require(sha(a.source / "fresh_v1/protocol.json") == baseline["protocol_sha256"], "Baseline protocol mismatch")
    code = read(a.source / "fresh_causal_v1/code_manifest.json")
    for name, expected in {**code["original_code_sha256"], **code["additive_code_sha256"]}.items():
        if name.startswith(("src/", "scripts/")):
            require(sha(a.source / "fresh_causal_v1/code" / name) == expected, f"Frozen code mismatch: {name}")
    require(len(manifest["cells"]) == 4 and {(c["model"], c["mode"]) for c in manifest["cells"]} ==
            {(m, mode) for m in ("Qwen3-8B", "Gemma4-E4B") for mode in ("enumeration_index", "enumeration_bullet")},
            "Incomplete or duplicate model/mode population")
    parser_path = a.source / "fresh_causal_v1/code/src/realistic_niah_v5/same_site_progress_transplant.py"
    spec = importlib.util.spec_from_file_location("_frozen_update_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = read(a.source / "fresh_v1/protocol.json")
    cells = []
    for cell in manifest["cells"]:
        start = time.monotonic()
        result = audit_cell(a.source, cell, manifest, cfg, module.generated_bullet_city_ordinals, inventory["source_root"])
        result["seconds"] = time.monotonic() - start
        cells.append(result)
        print(json.dumps({k: result[k] for k in ("model", "mode", "audit", "formal_rows", "seconds")}), flush=True)
    total = sum(c["formal_rows"] for c in cells)
    require(total == manifest["primary_expected_rows"] == assembly["primary_rows"] == 2160, "Missing primary rows")
    report = {"schema": "enumeration_fresh_update_audit_v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
              "formal_rows": total, "cells": cells, "archive_sha256": a.expected_sha256,
              "backup_manifest_sha256": sha(a.source / "backup_manifest.json"), "verified_files": len(inventory["files"]),
              "source_files": len(assembly["source_files_sha256"]), "parser_sha256": sha(parser_path),
              "script_sha256": sha(__file__), "module_sha256": sha(ROOT / "src/realistic_niah_v6/update_audit.py"),
              "reporting_module_sha256": sha(ROOT / "src/realistic_niah_v6/aligned_reporting.py"),
              "bootstrap": cfg["bootstrap"], "command": sys.argv, "seconds": time.monotonic() - tick,
              "estimand": "Paired donor-to-receiver minus receiver-self next-city adoption; all selected rows retained.",
              "limitations": ["Ten source seeds per model/mode; modes have separate baseline-eligible populations.",
                  "Sampling alignment was amended after the shared-cohort experiments began; it was not the original prospective protocol.",
                  "City-prefix success is not complete record fidelity or final-count accuracy.",
                  "The fixed 96-token budget and frozen known-city parser are retained; empty parses and truncations remain in denominators.",
                  "k=8 has no donor horizon at hops 3/4; empty conditional denominators are null, never zero.",
                  "Item-span patches also transfer item content and syntax; positive effects do not isolate arithmetic.",
                  "Donor and receiver absolute sites match, but their endpoint surface tokens can differ; Target and self have identical input hashes.",
                  "CPU audit rechecks saved source evidence and statistics, not GPU inference or tokenizer decoding.",
                  "Confidence intervals are descriptive source-seed bootstrap intervals without multiplicity adjustment.",
                  "All-zero or all-one source clusters yield degenerate percentile intervals; this does not establish a population probability of zero or one."]}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# Fresh primary Update audit", "", f"PASS: {total} rows; {len(inventory['files'])} archived file hashes verified.",
             "All three conditions generated on each selected seed/k/direction/scope; raw outputs and two prefill hooks match.",
             "No outcome-based exclusion; 10,000 source-seed bootstrap draws, seed 20260915.",
             "Discovery-selected layers (one-based): " + "; ".join(f'{c["model"]}/{c["mode"]}: L{c["layer_one_based"]}' for c in cells) + ".",
             "", "## Next-city adoption (all k pooled within direction)", "",
             "| Model / mode | Scope | Direction | Target | Self | Native donor | Target − self (pp; 95% CI) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for cell in cells:
        for group in cell["groups"]:
            if group["donor_k"] != "all":
                continue
            values = [group["conditions"][c]["continuation"]["1"] for c in ("donor_to_receiver", "receiver_self", "native_donor")]
            text = [f'{v["successes"]}/{v["horizon_eligible"]}' for v in values]
            stat = group["paired_target_minus_self_adoption"]
            ci = stat["ci95"]
            lines.append(f'| {cell["model"]} / {cell["mode"]} | {group["scope"]} | {group["direction"]} | '
                         + " | ".join(text) + f' | {100*stat["value"]:.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}] |')
    lines += ["", "## Donor-target continuation by k (item span)", "",
              "Each entry is successes / horizon denominator; successes / previous-prefix denominator.",
              "The same numerator is used in unconditional and conditional rates. A zero denominator means not estimable.",
              "", "| Model / mode | Direction | k | Hop 1 | Hop 2 | Hop 3 | Hop 4 |", "| --- | --- | ---: | --- | --- | --- | --- |"]
    for cell in cells:
        for group in cell["groups"]:
            if group["scope"] != "item_span" or group["donor_k"] == "all":
                continue
            cells_text = []
            for h in (1, 2, 3, 4):
                stat = group["conditions"]["donor_to_receiver"]["continuation"][str(h)]
                conditional = (f'{stat["successes"]}/{stat["conditional_eligible"]}'
                               if stat["conditional_eligible"] else "N/A (0 eligible)")
                cells_text.append("N/A" if not stat["horizon_eligible"] else
                                  f'{stat["successes"]}/{stat["horizon_eligible"]}; {conditional}')
            lines.append(f'| {cell["model"]} / {cell["mode"]} | {group["direction"]} | {group["donor_k"]} | '
                         + " | ".join(cells_text) + " |")
    lines += ["", "Full by-k statistics for all scopes and three controls, denominators and seed-bootstrap intervals are in audit.json.",
              "", "## Limits", ""] + ["- " + v for v in report["limitations"]]
    (a.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_rows": total, "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
