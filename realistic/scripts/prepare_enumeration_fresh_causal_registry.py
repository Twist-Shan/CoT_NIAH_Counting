"""Compile Read/Update baseline eligibility for one completed fresh cell on CPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.enumeration_fresh_geometry import adapt_update_row, build_read_geometry, build_update_geometry, json_sha


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--mode", choices=("enumeration_index", "enumeration_bullet", "thinking"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    cfg = json.loads((args.bundle / "protocol.json").read_text())
    assert sha(args.bundle / "protocol.json") == manifest["protocol_sha256"]
    baseline = args.bundle / "baseline" / args.model / args.mode
    status = json.loads((baseline / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["completed"] == manifest["rows_per_cell"]
    source = baseline / "generations.jsonl"
    assert sha(source) == status["generations_sha256"]
    # Numerical kernels and existing parser/audit scripts must match the
    # baseline snapshot. The new entrypoints are separately hashed below.
    for name, expected in manifest["code_sha256"].items():
        if name.startswith(("src/", "scripts/")):
            assert sha(ROOT / name) == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "geometry").mkdir()
    state = {"status": "RUNNING", "completed": 0, "total": manifest["rows_per_cell"]}
    write(args.output / "status.json", state)
    from realistic_niah_v4.modeling import load_registered_tokenizer
    from realistic_niah_v4.spec import resolve_model_spec
    if args.mode != "thinking":
        from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
        install_v6_kernel_adapters()
        install_v6_specialized_geometry(args.mode)
    tokenizer = load_registered_tokenizer(resolve_model_spec(args.model), cache_dir=args.cache_dir)
    ledger, seen = [], set()
    try:
        with source.open() as handle, (args.output / "adapted_generations.jsonl").open("x") as output:
            for line in handle:
                row = json.loads(line)
                key = int(row["seed"]), int(row["gold_count"])
                assert key not in seen
                seen.add(key)
                assert row["alignment_mode"] == args.mode and row["model_label"] == args.model
                adapted, metadata = adapt_update_row(row, model=args.model, mode=args.mode)
                geometry = {"seed": key[0], "gold_count": key[1], "request_id": row["request_id"],
                            "roles": row["alignment_roles"], "mode": args.mode, "model": args.model,
                            "stimulus_passage_sha256": row["stimulus_passage_sha256"],
                            "source_row_sha256": json_sha(row), "adapted_row_sha256": json_sha(adapted),
                            "update_metadata": metadata}
                entry = {name: geometry[name] for name in ("seed", "gold_count", "request_id", "roles")}
                entry.update(format_eligible=metadata["format_eligible"], read_eligible=False, update_eligible=False)
                try:
                    _, _, audit = build_read_geometry(row, tokenizer, mode=args.mode)
                    geometry["read"] = audit
                    entry["read_eligible"] = True
                except ValueError as error:
                    entry["read_exclusion"] = str(error)
                if key[1] == 10:
                    try:
                        geometry["update"] = build_update_geometry(adapted, tokenizer, metadata=metadata, cfg=cfg["update"])
                        entry["update_eligible"] = True
                    except ValueError as error:
                        entry["update_exclusion"] = str(error)
                entry["baseline_parsed_count"] = row.get("trace_parse", {}).get("parsed_count")
                entry["baseline_exact_count"] = row.get("trace_parse", {}).get("exact_count")
                entry["geometry_file"] = f"geometry/seed{key[0]}_N{key[1]}.json"
                write(args.output / entry["geometry_file"], geometry)
                entry["geometry_sha256"] = sha(args.output / entry["geometry_file"])
                ledger.append(entry)
                output.write(json.dumps(adapted, ensure_ascii=True) + "\n")
                output.flush()
                state.update(completed=len(ledger), seconds=time.monotonic() - tick)
                write(args.output / "status.json", state)
        assert len(ledger) == manifest["rows_per_cell"]
        read_rows = [entry for entry in ledger if "unfiltered_read" in entry["roles"]]
        assert len(read_rows) == 100
        write(args.output / "ledger.json", ledger)
        record = {"status": "FROZEN_BASELINE_GEOMETRY", "model": args.model, "mode": args.mode,
                  "baseline_manifest_sha256": sha(args.bundle / "manifest.json"),
                  "baseline_generations_sha256": sha(source), "ledger_sha256": sha(args.output / "ledger.json"),
                  "adapted_generations_sha256": sha(args.output / "adapted_generations.jsonl"),
                  "new_entrypoints_sha256": {name: sha(ROOT / "scripts" / name) for name in (
                      "enumeration_fresh_geometry.py", "prepare_enumeration_fresh_causal_registry.py")},
                  "unfiltered_read_population": 100, "read_eligible": sum(entry["read_eligible"] for entry in read_rows),
                  "update_eligible_n10": sum(entry["update_eligible"] for entry in ledger),
                  "selection_used_final_correctness": False, "intervention_outcomes_accessed": False,
                  "command": sys.argv, "seconds": time.monotonic() - tick}
        write(args.output / "manifest.json", record)
        state.update(status="COMPLETE", manifest_sha256=sha(args.output / "manifest.json"))
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
