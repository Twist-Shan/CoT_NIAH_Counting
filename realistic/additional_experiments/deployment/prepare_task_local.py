"""Freeze a new task-local selection campaign from audited source inputs."""
from pathlib import Path
import argparse
import hashlib
import shlex
import json
import shutil
import sys
import tarfile
import time

B = Path(__file__).resolve().parents[1]
REPO = B.parent
VERSION = 'task_local_disjoint_first_20260908_v3'
R = B / 'runs' / VERSION / 'package'
OLD = B / "runs/topk_completion_20260907_v1/downloaded"
sys.path.insert(0, str(B))
from local_selection import rank_discovery, select_heads, random_control, control_audit


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")
def local(remote):
    run, rel = remote.split("/additional_experiments/", 1)[1].split("/", 1)
    return B / "runs" / run / "downloaded" / rel


def copy_snapshot(destination):
    for p in (REPO / "src").rglob("*.py"):
        dest = destination / "src" / p.relative_to(REPO / "src")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for name in ["run.py", "protocol.py", "diagnostics.py", "local_selection.py", "task_scoring.py"]:
        shutil.copy2(B / name, destination / name)
    for name in ["run_task_local.py", "analyze_task_local.py"]:
        shutil.copy2(B / "deployment" / name, destination / name)
    shutil.copy2(B / "TASK_LOCAL_DISJOINT_FIRST_20260908.md", destination / "PROTOCOL.md")
    # The released modeling source already snapshots all nested configs before
    # mutation and restores them in finally. Copy it unchanged; do not reapply
    # the obsolete text patch for the earlier working-tree implementation.
    shutil.copy2(B / "deployment/task_local_inputs.py", destination / "task_local_inputs.py")
    shutil.copy2(B / "deployment/compile_transfer_registry.py", destination / "compile_transfer_registry.py")


def launcher(python='python'):
    # PYTHON can point to the environment on the machine executing the package.
    return ['#!/bin/bash', 'set -euo pipefail', 'cd "$(dirname "$0")"',
            'PYTHON=${PYTHON:-' + shlex.quote(python) + '}',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1']


def main():
    global OLD, R
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-package', type=Path, default=OLD)
    ap.add_argument('--alignment-audit', type=Path, default=B / 'runs/report_refresh_20260908_v1/alignment_audit.json')
    ap.add_argument('--output', type=Path, default=R)
    ap.add_argument('--python', default='python', help='Interpreter used by launch.sh; PYTHON overrides it at execution')
    args = ap.parse_args()
    OLD, R = args.previous_package.resolve(), args.output.resolve()
    started = time.perf_counter()
    assert read(args.alignment_audit)["status"] == "PASS"
    if R.exists():
        raise FileExistsError("Frozen package already exists; use an explicitly versioned new directory")
    R.mkdir(parents=True)
    old = read(OLD / "protocol.json")
    copy_snapshot(R)
    widths = {}
    differences = []
    for model in old["sources"]:
        widths[model] = read(OLD / "full" / model / "environment.json")["widths"]
        for task in ["kth", "category"]:
            plans = read(OLD / "plans" / f"{task}_{model}.json")
            assert len(plans) == 600
            write(R / "plans" / f"{task}_{model}.json", plans)
            for mode in ["nonthinking", "native_thinking"]:
                observations = []
                source_files = {}
                for plan in plans:
                    if plan["split"] != "discovery" or plan["mode"] != mode:
                        continue
                    remote = old["sources"][model][task] + f"/full/{model}/{task}/attention/{mode}_{plan['case_id']}.json"
                    assert sha(local(remote)) == old["source_hashes"][remote]
                    row = read(local(remote))
                    observations.append(row)
                    source_files[remote] = sha(local(remote))
                ranking = rank_discovery(observations)
                sizes = [1, 2, 4, 8, 16, 32, 64, 128] if model.startswith("Qwen") else [1, 2, 4, 6, 8]
                bank = dict(model=model, task=task, mode=mode, assay="broad", ranking=ranking,
                            sizes=sizes, selection="new-task discovery seed-equal global Top-K; no layer cap",
                            source_hashes=source_files, conditions={})
                for k in sizes:
                    heads = select_heads(ranking, k)
                    bank["conditions"][str(k)] = dict(selected=heads, random=[random_control(heads, widths[model], 7000+i) for i in range(3)])
                    bank['conditions'][str(k)]['control_audit'] = control_audit(heads,widths[model],bank['conditions'][str(k)]['random'])
                write(R / "banks" / f"{task}_{model}_{mode}_broad.json", bank)
                previous = read(OLD / "banks" / f"{task}_{model}.json")[mode + "_broad"]["heads"]
                heads = bank["conditions"][str(max(sizes))]["selected"]
                differences.append(dict(model=model, task=task, mode=mode, k=max(sizes),
                                        changed_head_count=len(set(map(tuple, heads)) - set(map(tuple, previous)))))
    cfg = dict(version=VERSION, cache=old["cache"], widths=widths,
               models=list(old["sources"]), tasks=["kth", "category"],
               target_sizes={"Qwen3-8B": [32,64,80,96,112,128], "Gemma4-E4B": [1,2,4,6,8]},
               discovery_seeds=list(range(1234,1254)), confirmation_seeds=list(range(1254,1264)),
               target_selection="new-task raw target-source-record attention mass at frozen Native discovery anchors; seed-equal global ranking",
               control_policy='disjoint_first_minimum_overlap',
               controls="layer-count matched; uniformly sample unselected heads without replacement when sufficient; otherwise include all unselected and sample only the deficit from selected; random repeats may overlap",
               layer_cap=False, broad_max_tokens=64, targeted_max_tokens=256,
               endpoints="case-specific sites located on each new task's own natural trace; Broad before its final answer; Targeted at its own frozen registered next-record event",
               scoring="Broad: final task answer exact accuracy. Targeted: first semantic record entity equals the registered next record; if no record is identifiable, use the main assay's exact original target-token prefix fallback at offset zero. Same definitions for both tasks and all conditions.",
               stats=dict(bootstrap=20000, seed=20260907, unit="seed", correction="exact two-sided seed sign-flip; Holm across task/model/mode/dose within assay and population; pointwise CIs"),
               old_results_reused=False, natural_outputs_reused=True,
               expected_full_points=6739, broad_membership_changes=differences)
    cfg["files"] = {str(p.relative_to(R)).replace("\\", "/"): sha(p) for p in R.rglob("*") if p.is_file()}
    write(R / "protocol.json", cfg)
    py = '"$PYTHON"'
    launch = launcher(args.python)
    for stage in ["discover", "canary", "full"]:
        for model in cfg["models"]:
            launch.append(f'{py} run_task_local.py --model {model} --stage {stage}')
    launch += [f'{py} analyze_task_local.py --root .',
               'tar -czf results.tgz protocol.json plans banks full canary discovery analysis', 'echo TASK_LOCAL_COMPLETE']
    (R / "launch.sh").write_bytes(("\n".join(launch) + "\n").encode("utf-8"))
    archive = R.parent / (VERSION+'.tgz')
    with tarfile.open(archive, "w:gz") as tar:
        for p in R.rglob("*"):
            if p.is_file(): tar.add(p, arcname=str(p.relative_to(R)))
    write(R.parent / "preparation.json", dict(status="PREPARED_NOT_RUN", files=len(cfg["files"]),
          archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
          elapsed_seconds=time.perf_counter()-started, broad_membership_changes=differences))
    print(json.dumps(read(R.parent / "preparation.json")))


if __name__ == "__main__": main()
