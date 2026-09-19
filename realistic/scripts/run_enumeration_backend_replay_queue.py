#!/usr/bin/env python3
"""Run the frozen replay jobs serially, retaining errors and elapsed times."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    args = p.parse_args()
    q = json.loads(args.queue.read_text())
    status_path = args.queue.parent / "queue_status.json"
    if status_path.exists():
        raise FileExistsError("Queue already started; inspect outputs before deciding how to recover")
    start = time.time()
    status = {"status": "RUNNING", "pid": os.getpid(), "started_unix": start,
              "queue_sha256": hashlib.sha256(args.queue.read_bytes()).hexdigest(), "jobs": []}

    def save():
        status["elapsed_seconds"] = time.time() - start
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(status_path)

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
    save()
    try:
        for job in q["jobs"]:
            path = Path(job["command"][-1])
            if hashlib.sha256(path.read_bytes()).hexdigest() != job["job_sha256"]:
                raise RuntimeError(f"Frozen job changed: {job['id']}")
            record = {"id": job["id"], "status": "RUNNING", "started_unix": time.time()}
            status["jobs"].append(record)
            save()
            with (path.parent / "process.log").open("x") as log:
                child = subprocess.Popen(job["command"], cwd=q["code"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                save()
                record["returncode"] = child.wait()
            record["seconds"] = time.time() - record["started_unix"]
            record["status"] = "COMPLETE" if record["returncode"] == 0 else "FAILED"
            save()
            if record["returncode"]:
                raise RuntimeError(f"Replay failed: {job['id']}; see its process.log")
        status["status"] = "COMPLETE"
    except BaseException as error:
        status["status"] = "FAILED"
        status["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
