"""Run representative original tests in separate component processes."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "realistic": [
        "tests/test_realistic_niah_v3_1.py",
        "tests/test_realistic_niah_v4_4_2.py",
        "tests/test_realistic_niah_v5_parser_v2.py",
        "tests/test_realistic_niah_v6.py",
        "tests/test_enumeration_fresh_behavior_audit.py",
        "tests/test_enumeration_fresh_read_audit.py",
        "tests/test_enumeration_fresh_update_audit.py",
        "tests/test_enumeration_update_n10.py",
        "tests/test_enumeration_qwen_layer_diagnostic.py",
        "tests/test_release_preparation.py",
        "tests/test_niah_geometry_report_diagnostics.py",
        "tests/test_realistic_niah_v4_4_3.py",
        "tests/test_realistic_niah_v4_4_3_set.py",
        "tests/test_realistic_niah_v4_4_4.py",
        "tests/test_realistic_niah_v4_4_4_readwrite.py",
        "tests/test_realistic_niah_v4_4_4_relay.py",
        "tests/test_realistic_niah_v4_4_4_upstream_path.py",
        "tests/test_realistic_niah_v5_integrated_branch_ledger.py",
        "tests/test_realistic_niah_v5_integrated_mediator_restoration.py",
        "tests/test_realistic_niah_v5_prospective_evidence.py",
        "tests/test_realistic_niah_v6_answer_trace_extension.py",
        "tests/test_v5_native_supplement.py",
        "additional_experiments/test_category_trace_parser.py",
        "additional_experiments/test_kth_retrieval.py",
        "additional_experiments/test_task_scoring.py",
        "additional_experiments/test_local_selection.py",
        "additional_experiments/test_transfer_registry.py",
        "additional_experiments/test_fresh_task_local.py",
        "additional_experiments/test_protocol.py",
    ],
    "synthetic": [
        "tests/test_synthetic_counting_v58.py",
        "tests/test_v58_alignment_supplement.py",
        "tests/test_v58_cached_nonthinking.py",
        "tests/test_v58_commit_query.py",
        "tests/test_v58_native_continuation.py",
        "tests/test_v58_top1to8_aligned.py",
    ],
}


def main() -> int:
    start = time.perf_counter()
    failed = False
    for family, tests in SUITES.items():
        cwd = ROOT / family
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(cwd / part) for part in ("src", "scripts", ".", "additional_experiments"))
        env["PYTHONUTF8"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        print(f"Testing {family}", flush=True)
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=cwd, env=env)
        failed |= result.returncode != 0
    print(f"Elapsed: {time.perf_counter()-start:.1f} seconds", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
