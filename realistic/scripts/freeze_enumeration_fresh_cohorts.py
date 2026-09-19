"""Freeze cross-mode Update cohorts from completed CPU eligibility ledgers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.enumeration_fresh_geometry import common_eligible_seeds
from scripts.prepare_enumeration_fresh_causal_registry import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--registries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    baseline = json.loads((a.bundle / "manifest.json").read_text())
    cfg = json.loads((a.bundle / "protocol.json").read_text())
    assert sha(a.bundle / "protocol.json") == baseline["protocol_sha256"]
    if a.output.exists():
        raise FileExistsError(a.output)
    result = {"status": "FROZEN_BEFORE_INTERVENTION", "models": {}, "registries": {},
              "baseline_manifest_sha256": sha(a.bundle / "manifest.json"),
              "selection": "first common baseline-format-and-all-update-geometry eligible seeds in registered order",
              "selection_used_final_correctness": False, "intervention_outcomes_accessed": False,
              "quota": cfg["causal_target_common_seeds_per_model"]}
    for model in cfg["models"]:
        ledgers = []
        for mode in cfg["modes"]:
            folder = a.registries / model / mode
            manifest = json.loads((folder / "manifest.json").read_text())
            status = json.loads((folder / "status.json").read_text())
            assert status["status"] == "COMPLETE"
            assert manifest["baseline_manifest_sha256"] == result["baseline_manifest_sha256"]
            assert sha(folder / "ledger.json") == manifest["ledger_sha256"]
            ledger = json.loads((folder / "ledger.json").read_text())
            ledgers.append(ledger)
            result["registries"][f"{model}/{mode}"] = {
                "manifest_sha256": sha(folder / "manifest.json"), "ledger_sha256": manifest["ledger_sha256"]}
        selected = common_eligible_seeds(ledgers, baseline["confirmation_candidates"],
                                        field="update_eligible", quota=result["quota"])
        discovery = common_eligible_seeds(ledgers, baseline["discovery_seeds"],
                                          field="update_eligible", quota=len(baseline["discovery_seeds"]))
        result["models"][model] = {"confirmation_seeds": selected, "actual_confirmation_n": len(selected),
                                  "confirmation_shortfall": result["quota"] - len(selected),
                                  "discovery_geometry_common_seeds": discovery,
                                  "layers_one_based": cfg["update"]["layers_one_based"][model]}
    result["cross_model_common_confirmation_seeds"] = sorted(set.intersection(
        *(set(value["confirmation_seeds"]) for value in result["models"].values())))
    result["seconds"] = time.monotonic() - tick
    result["command"] = sys.argv
    result["entrypoint_sha256"] = sha(__file__)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, result)
    print(json.dumps({"status": result["status"], "models": result["models"]}), flush=True)


if __name__ == "__main__":
    main()
