"""Verify the completed N10 Update grid against the frozen selection and raw evidence."""
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
from realistic_niah_v6.read_audit import read, require, sha
from realistic_niah_v6.update_audit import audit_cell
from realistic_niah_v6.update_n10 import validate_selection
from analyze_enumeration_fresh_relay import verify_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "inputs", "discovery", "selection-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--selection-audit-sha256", required=True)
    a = p.parse_args()
    tick = time.monotonic()
    inventory = verify_inventory(a.source)
    require(inventory["schema"] == "enumeration_n10_update_incremental_backup_v1", "Wrong N10 backup")
    for name, folder in (("inputs", a.inputs), ("discovery", a.discovery)):
        require(sha(folder / "backup_manifest.json") == inventory["dependencies"][name]["manifest_sha256"],
                "Dependency inventory mismatch")
        verify_inventory(folder)
    stage = a.source / "fresh_n10_update_v1"
    manifest, selection, assembly, state, protocol = [read(stage / name) for name in
        ("manifest.json", "selection_manifest.json", "assembly_audit.json", "gpu_pipeline_status.json", "protocol.json")]
    require(sha(stage / "manifest.json") == inventory["manifest_sha256"], "N10 run identity mismatch")
    require(sha(stage / "selection_manifest.json") == inventory["selection_manifest_sha256"] == assembly["selection_manifest_sha256"],
            "N10 selection identity mismatch")

    def relocate(path):
        relative = PurePosixPath(path).relative_to(PurePosixPath(inventory["source_root"]))
        for folder in (a.source, a.inputs, a.discovery):
            file = folder / relative
            if file.is_file():
                return file
        raise ValueError(f"Missing frozen dependency: {relative}")

    for relative, expected in manifest["code_sha256"].items():
        require(sha(stage / "code" / relative) == expected, f"Frozen N10 code changed: {relative}")
    for relative, expected in manifest["contract_sha256"].items():
        require(sha(stage / relative) == expected, f"Frozen N10 contract changed: {relative}")
    for remote, expected in {**manifest["source_sha256"], **assembly["source_files_sha256"]}.items():
        require(sha(relocate(remote)) == expected, f"Frozen source/assembly changed: {remote}")
    require(assembly["status"] == "PASS" and state["status"] == "COMPLETE" and
            state["phase"] == "N10_UPDATE_GPU_COMPLETE_ANALYSIS_PENDING", "Incomplete N10 stage")
    require(selection["protocol_sha256"] == sha(stage / "protocol.json"), "Selection protocol changed")
    require(selection["confirmation_is_reused"] is True, "Reused confirmation disclosure missing")
    require(sha(a.selection_audit) == a.selection_audit_sha256, "Prior selection CPU audit changed")
    selection_audit = read(a.selection_audit)
    require(selection_audit["status"] == "PASS" and selection_audit["selection_manifest_sha256"] == sha(stage / "selection_manifest.json"),
            "Prior selection CPU audit does not cover this selection")
    discovered = read(a.discovery / "backup_manifest.json")
    require(discovered["n10_manifest_sha256"] == sha(stage / "manifest.json"), "Discovery belongs to another run")
    # Recheck the exact files already covered by the complete CPU selection replay.
    # The immutable tensors remain in that archive; duplicated contracts must agree.
    for name, record in discovered["files"].items():
        local = a.source / name
        if local.exists():
            require(sha(local) == record["sha256"], f"Discovery evidence changed after audit: {name}")
    for name, record in inventory["tensor_dependencies"].items():
        require(record == discovered["files"][name], "Tensor dependency identity mismatch")
    expected_cells = {(m, f) for m in ("Qwen3-8B", "Gemma4-E4B") for f in ("enumeration_index", "enumeration_bullet")}
    for cells in (manifest["cells"], selection["cells"], selection_audit["cells"]):
        require(len(cells) == 4 and {(c["model"], c["mode"]) for c in cells} == expected_cells, "Missing or duplicate N10 cell")
    jobs = read(stage / "jobs.json")
    require([j["id"] for j in jobs] == [j["id"] for j in state["jobs"]], "Job order or completeness mismatch")
    require(max(i for i, j in enumerate(jobs) if j["id"].startswith("smoke/")) <
            min(i for i, j in enumerate(jobs) if j["id"].startswith("formal/")), "Formal preceded complete smoke grid")
    for job, done in zip(jobs, state["jobs"]):
        status = read(relocate(job["status_path"]))
        require(done["status"] == status["status"] == "COMPLETE" and done["returncode"] == 0, "Failed pipeline job")
        require(done["command"] == job["command"], "Executed command changed")
        if job["expected_rows"] is not None:
            require(done["rows"] == job["expected_rows"] == status.get("completed_rows", status.get("completed")), "Worker count mismatch")
    require(read(stage / "cpu_validation.json")["status"] == "PASS", "Frozen contract CPU checks did not pass")
    cfg = read(a.inputs / "fresh_v1/protocol.json")
    for key in ("donor_k", "directions", "scopes", "max_new_tokens"):
        require(cfg["update"][key] == protocol["update"][key], f"Numerical Update protocol changed: {key}")
    parser_path = stage / "code/src/realistic_niah_v5/same_site_progress_transplant.py"
    spec = importlib.util.spec_from_file_location("_frozen_n10_update_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells, smoke = [], []
    for cell in manifest["cells"]:
        chosen = next(c for c in selection["cells"] if (c["model"], c["mode"]) == (cell["model"], cell["mode"]))
        validate_selection(selection, cell["model"], cell["mode"], cell["confirmation_seeds"])
        require(sha(relocate(chosen["selection_path"])) == chosen["selection_sha256"], "Selected layer report changed")
        kwargs = dict(inputs_root=a.inputs, selection=selection)
        start = time.monotonic()
        smoke.append(audit_cell(a.source, cell, manifest, cfg, module.generated_bullet_city_ordinals,
                                inventory["source_root"], phase="smoke", statistics=False, **kwargs))
        result = audit_cell(a.source, cell, manifest, cfg, module.generated_bullet_city_ordinals,
                            inventory["source_root"], **kwargs)
        result["seconds"] = time.monotonic() - start
        cells.append(result)
        print(json.dumps({k: result[k] for k in ("model", "mode", "audit", "formal_rows", "layer_one_based", "seconds")}), flush=True)
    require(sum(c["formal_rows"] for c in cells) == assembly["primary_rows"] == manifest["expected_formal_rows"] == 2160,
            "N10 formal total mismatch")
    require(sum(c["smoke_rows"] for c in smoke) == manifest["expected_smoke_rows"] == 216, "N10 smoke total mismatch")
    report = {"schema": "enumeration_n10_update_audit_v1", "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
        "formal_rows": 2160, "smoke_rows": 216, "cells": cells, "smoke": smoke,
        "manifest_sha256": sha(stage / "manifest.json"), "selection_manifest_sha256": sha(stage / "selection_manifest.json"),
        "selection_cpu_audit_sha256": sha(a.selection_audit), "backup_manifest_sha256": sha(a.source / "backup_manifest.json"),
        "script_sha256": sha(__file__), "module_sha256": sha(ROOT / "src/realistic_niah_v6/update_audit.py"),
        "reporting_sha256": sha(ROOT / "src/realistic_niah_v6/aligned_reporting.py"), "parser_sha256": sha(parser_path),
        "bootstrap": cfg["bootstrap"], "command": sys.argv, "seconds": time.monotonic() - tick,
        "limits": ["The N10 amendment follows observation of old mixed-N outcomes; the ten confirmation inputs per cell are reused.",
            "All four intervention grids were executed anew; no old intervention rows were reused.",
            "All failures and 96-token truncations are retained. Empty conditional denominators and inapplicable donor horizons are null.",
            "Item spans transfer syntax and record content together with progress; city-prefix success is not complete-record or final-count accuracy.",
            "There are ten source-seed clusters per cell. Pointwise bootstrap intervals do not adjust for multiple tests; degenerate intervals do not prove population certainty.",
            "CPU verification checks saved tokens, hooks, source evidence and statistics; it does not repeat GPU inference or tokenizer decoding."]}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# N10 Update audit", "", "PASS: 2,160 formal rows and 216 smoke rows; complete paired controls and raw evidence.", "",
        "| Cell | Layer | Scope | Direction | Target | Self | Native donor | Hop 2 conditional | Target-self (95% CI) |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for cell in cells:
        for group in cell["groups"]:
            if group["donor_k"] != "all":
                continue
            values = [group["conditions"][c]["continuation"]["1"] for c in ("donor_to_receiver", "receiver_self", "native_donor")]
            counts = [f'{v["successes"]}/{v["horizon_eligible"]}' for v in values]
            hop = group["conditions"]["donor_to_receiver"]["continuation"]["2"]
            conditional = f'{hop["successes"]}/{hop["conditional_eligible"]}' if hop["conditional_eligible"] else "N/A (0 eligible)"
            effect = group["paired_target_minus_self_adoption"]
            lines.append(f'| {cell["model"]}/{cell["mode"]} | {cell["layer_one_based"]} | {group["scope"]} | {group["direction"]} | '
                         + " | ".join(counts) + f' | {conditional} | {effect["value"]:.3f} {effect["ci95"]} |')
    lines += ["", "Full per-k, three-scope and both conditional/unconditional continuation statistics are in audit.json.", ""]
    lines.extend("- " + v for v in report["limits"])
    (a.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "formal_rows": 2160, "smoke_rows": 216, "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
