"""Print a compact, read-only fresh-pipeline snapshot for the progress monitor."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    state = json.loads((a.root / "fresh_pipeline_v1_status.json").read_text())
    cells = []
    for phase in ("smoke", "baseline"):
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            for mode in ("enumeration_index", "enumeration_bullet", "thinking"):
                folder = a.root / "fresh_v1" / phase / model / mode
                path = folder / "status.json"
                row = json.loads(path.read_text()) if path.exists() else {"status": "PENDING"}
                cells.append({"phase": phase, "model": model, "mode": mode,
                              **{k: row[k] for k in ("status", "completed", "total", "seconds", "error") if k in row}})
    workers = [{k: row[k] for k in ("id", "pid", "status", "returncode") if k in row}
               for row in state.get("jobs", []) if row["status"] != "COMPLETE"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "status": state["status"], "phase": state["phase"],
              "error": state.get("error"), "workers": workers, "cells": cells,
              "baseline_completed_rows": sum(row.get("completed", 0) for row in cells if row["phase"] == "baseline"),
              "gpu": gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
