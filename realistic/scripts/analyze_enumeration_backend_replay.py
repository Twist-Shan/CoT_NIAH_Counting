#!/usr/bin/env python3
"""Audit completed Enumeration helper replays and compare paired raw outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    base = args.queue.parent
    queue = read_json(args.queue)
    status = read_json(base / "queue_status.json")
    if status["status"] != "COMPLETE" or len(status["jobs"]) != 8:
        raise ValueError("All eight jobs must finish before the final audit")
    if any(j["status"] != "COMPLETE" or j["returncode"] != 0 for j in status["jobs"]):
        raise ValueError("Queue contains a failed or incomplete job")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def sha(path):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return hashes[path]

    if sha(args.queue) != status["queue_sha256"]:
        raise ValueError("Frozen queue changed")
    loaded = {}
    job_audits = []
    runtime_signatures = []
    for item in queue["jobs"]:
        folder = base / item["id"]
        command_path = Path(item["command"][-1])
        if sha(command_path) != item["job_sha256"]:
            raise ValueError(f"Frozen command changed: {item['id']}")
        job = read_json(command_path)
        for filename, expected in job["input_sha256"].items():
            if sha(filename) != expected:
                raise ValueError(f"Frozen input changed: {filename}")
        for filename, expected in job["code_sha256"].items():
            if sha(Path(queue["code"]) / filename) != expected:
                raise ValueError(f"Frozen code changed: {filename}")
        process = read_json(folder / "process_status.json")
        if process["status"] != "COMPLETE":
            raise ValueError(f"Incomplete process: {item['id']}")
        shards = sorted((folder / "results/shards").glob("*.jsonl"))
        for f in shards:
            sha(f)
        rows = [r for f in shards for r in read_rows(f)]
        endpoint = "/endpoint_routing/" in item["id"]
        expected_shards = 60 if endpoint else 10
        conditions = (set(("clean", "self_patch", "full_donor_patch", "count_subspace_transplant",
                           "norm_matched_orthogonal_patch", "count_component_removed", "count_component_restored"))
                      if endpoint else set(("clean", "prompt_all_blank", "prompt_records_blank",
                                            "trace_all_blank", "prompt_and_trace_blank")))
        if len(shards) != expected_shards or len(rows) != expected_shards * len(conditions):
            raise ValueError(f"Incomplete result count: {item['id']}")
        keyed = {}
        groups = defaultdict(set)
        for row in rows:
            if row["status"] != "ok":
                raise ValueError(f"Non-ok result row: {item['id']}: {row['status']}")
            identity = row["pair_sha256"] if endpoint else row["request_id"]
            key = (identity, row["condition"])
            if key in keyed:
                raise ValueError(f"Duplicate trial: {key}")
            keyed[key] = row
            groups[identity].add(row["condition"])
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise ValueError(f"Nonfinite top-level scalar: {key}")
        if len(groups) != expected_shards or any(v != conditions for v in groups.values()):
            raise ValueError(f"Condition grid changed: {item['id']}")
        events_path = folder / "backend_events.jsonl"
        sha(events_path)
        events = read_rows(events_path)
        switches = [e for e in events if e["event"] == "temporary_attention_backend"]
        generations = [e for e in events if e["event"].startswith("generate_answer_completion")]
        if len(switches) != len(rows) or len(generations) != len(rows):
            raise ValueError(f"Observed calls do not cover every trial: {item['id']}")
        mismatches = sum(e["before"] != e["after"] for e in switches)
        fixed = job["backend_variant"] == "fixed"
        if fixed and (mismatches or any(e["after"]["text"] != "sdpa" for e in switches)):
            raise ValueError(f"Repaired helper failed to restore state: {item['id']}")
        model_events = [e for e in events if e["event"] == "model_loaded"]
        if len(model_events) != 1:
            raise ValueError("Expected exactly one model load per job")
        model_event = model_events[0]
        runtime = read_json(folder / "runtime.json")
        signature = {k: runtime[k] for k in ("python", "cuda", "packages", "gpus")}
        signature["model"] = {k: v for k, v in model_event.items() if k not in ("seconds", "event")}
        runtime_signatures.append(signature)
        audit = dict(id=item["id"], shards=len(shards), rows=len(rows),
                     conditions=dict(Counter(r["condition"] for r in rows)),
                     helper_calls=len(switches), generation_calls=len(generations),
                     backend_restore_mismatches=mismatches,
                     eager_after_count=sum(e["after"]["text"] == "eager" for e in switches),
                     seconds=process["seconds"])
        job_audits.append(audit)
        loaded[item["id"]] = (job, keyed, generations)
    if any(s != runtime_signatures[0] for s in runtime_signatures):
        raise ValueError("Paired jobs did not use identical model/runtime settings")
    comparisons = []
    for mode in ("enumeration_index", "enumeration_bullet"):
        for experiment in ("endpoint_routing", "answer_blanking"):
            prefix = f"{mode}/{experiment}"
            old_job, old_rows, old_gen = loaded[prefix + "/legacy"]
            new_job, new_rows, new_gen = loaded[prefix + "/fixed"]
            old_args, new_args = list(old_job["args"]), list(new_job["args"])
            for cli in (old_args, new_args):
                cli[cli.index("--output") + 1] = "<OUTPUT>"
            if old_args != new_args or old_job["input_sha256"] != new_job["input_sha256"] or old_job["code_sha256"] != new_job["code_sha256"]:
                raise ValueError(f"Non-helper paired command/input difference: {prefix}")
            if old_rows.keys() != new_rows.keys():
                raise ValueError(f"Paired support mismatch: {prefix}")
            fields = sorted(set.intersection(*(set(r) for r in old_rows.values()), *(set(r) for r in new_rows.values())))
            changed = {}
            for field in fields:
                diffs = [(old_rows[k][field], new_rows[k][field]) for k in sorted(old_rows)]
                count = sum(a != b for a, b in diffs)
                if not count:
                    continue
                info = {"changed_rows": count, "paired_rows": len(diffs)}
                if all(isinstance(a, (int, float)) and isinstance(b, (int, float)) for a, b in diffs):
                    delta = [float(b) - float(a) for a, b in diffs]
                    info.update(mean_legacy=statistics.mean(float(a) for a, b in diffs),
                                mean_fixed=statistics.mean(float(b) for a, b in diffs),
                                mean_fixed_minus_legacy=statistics.mean(delta), max_abs_change=max(map(abs, delta)))
                changed[field] = info
            generation_differences = []
            for a, b in zip(old_gen, new_gen):
                if (a["event"], a["index"]) != (b["event"], b["index"]):
                    raise ValueError(f"Generation call-order mismatch: {prefix}")
                if a["generation"]["generated_token_ids"] != b["generation"]["generated_token_ids"]:
                    generation_differences.append({"call_index": a["index"],
                                                   "legacy": a["generation"], "fixed": b["generation"]})
            condition_metrics = {}
            metric_names = (("donor_city_adoption", "receiver_city_retention",
                             "donor_vs_receiver_city_log_odds", "donor_vs_receiver_path_log_odds",
                             "donor_successor_attention_mass", "receiver_successor_attention_mass",
                             "local_generation_truncated")
                            if experiment == "endpoint_routing" else
                            ("exact_count", "absolute_error", "gold_first_answer_token_log_probability",
                             "generation_truncated"))
            for condition in sorted({k[1] for k in old_rows}):
                paired = [(old_rows[k], new_rows[k]) for k in sorted(old_rows) if k[1] == condition]
                details = {"paired_rows": len(paired), "metrics": {}}
                for metric in metric_names:
                    values = [(a[metric], b[metric]) for a, b in paired]
                    eligible = [(float(a), float(b)) for a, b in values
                                if isinstance(a, (int, float)) and isinstance(b, (int, float))]
                    item = {"paired_numeric_rows": len(eligible),
                            "changed_rows": sum(a != b for a, b in values)}
                    if eligible:
                        item.update(mean_legacy=statistics.mean(a for a, b in eligible),
                                    mean_fixed=statistics.mean(b for a, b in eligible),
                                    mean_fixed_minus_legacy=statistics.mean(b - a for a, b in eligible),
                                    max_abs_change=max(abs(b - a) for a, b in eligible))
                    if all(isinstance(a, bool) and isinstance(b, bool) for a, b in values):
                        item.update(false_to_true=sum(not a and b for a, b in values),
                                    true_to_false=sum(a and not b for a, b in values))
                    details["metrics"][metric] = item
                text_field = "local_completion_text" if experiment == "endpoint_routing" else "completion_text_raw"
                details["changed_completion_text_rows"] = sum(a[text_field] != b[text_field] for a, b in paired)
                condition_metrics[condition] = details
            comparisons.append({"id": prefix, "paired_rows": len(old_rows),
                                "changed_generation_calls": len(generation_differences),
                                "generation_calls": len(old_gen), "changed_fields": changed,
                                "condition_metrics": condition_metrics,
                                "generation_differences": generation_differences})
    report = {"schema_version": "enumeration_backend_replay_audit_v1", "status": "PASS",
              "scope": "Paired helper effect in one pinned runtime using historical inputs; no fresh confirmation.",
              "jobs": job_audits, "comparisons": comparisons, "runtime": runtime_signatures[0],
              "verified_file_sha256": hashes, "seconds": time.monotonic() - started}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "jobs": len(job_audits),
                      "condition_rows": sum(j["rows"] for j in job_audits),
                      "verified_files": len(hashes),
                      "comparisons": [{k: c[k] for k in ("id", "paired_rows", "changed_generation_calls")}
                                      for c in comparisons], "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
