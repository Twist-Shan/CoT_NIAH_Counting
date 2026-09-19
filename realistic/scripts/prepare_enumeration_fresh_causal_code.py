"""Freeze additive Read/Update entrypoints without changing the baseline snapshot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

SCRIPTS = (
    "enumeration_fresh_geometry.py", "prepare_enumeration_fresh_causal_registry.py",
    "freeze_enumeration_fresh_cohorts.py", "run_enumeration_fresh_read.py",
    "run_enumeration_fresh_update.py", "prepare_enumeration_fresh_causal_code.py",
    "inspect_enumeration_fresh.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--extensions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    baseline = json.loads((a.bundle / "manifest.json").read_text())
    for name, expected in baseline["code_sha256"].items():
        assert sha(a.bundle / "code" / name) == expected, name
    a.output.mkdir(parents=True, exist_ok=False)
    code = a.output / "code"
    shutil.copytree(a.bundle / "code", code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "work"))
    for name in SCRIPTS:
        assert not (code / "scripts" / name).exists(), name
        shutil.copy2(a.extensions / name, code / "scripts" / name)
    shutil.copy2(a.extensions / "test_enumeration_fresh_geometry.py", code / "tests/test_enumeration_fresh_geometry.py")
    record = {"status": "CODE_FROZEN_PENDING_REMOTE_TESTS_AND_BASELINE_REGISTRIES",
              "baseline_bundle": str(a.bundle), "baseline_manifest_sha256": sha(a.bundle / "manifest.json"),
              "original_code_sha256": baseline["code_sha256"],
              "additive_code_sha256": {f"scripts/{name}": sha(code / "scripts" / name) for name in SCRIPTS},
              "tests_sha256": sha(code / "tests/test_enumeration_fresh_geometry.py"),
              "stages_implemented": ["baseline_Read_Update_geometry", "Read_blanking", "Update"],
              "remaining_stage_implementation": ["Retrieve_selection_and_lesions", "answer_state_patch", "terminal_relay"],
              "numerical_kernel_changes": False, "baseline_snapshot_modified": False,
              "command": sys.argv, "seconds": time.monotonic() - tick}
    (a.output / "code_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "seconds": record["seconds"],
                      "manifest_sha256": sha(a.output / "code_manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
