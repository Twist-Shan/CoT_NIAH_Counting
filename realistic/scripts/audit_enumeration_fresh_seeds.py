"""Conservatively inventory historical seed declarations before fresh generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

SEED_VALUE = re.compile(rb'"[^"\r\n]*seed[^"\r\n]*"\s*:\s*(\d+|\[[\d,\s]+\])', re.I)
SEED_NAME = re.compile(rb'seed[_-]?(\d+)', re.I)
SKIP = {"cache", "hf_cache", "models", ".git", ".venv", "venv", "__pycache__", "node_modules", ".uv-cache"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, nargs="+", required=True)
    p.add_argument("--exclude", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--first-candidate", type=int, default=2026091600)
    p.add_argument("--candidate-count", type=int, default=70)
    a = p.parse_args()
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    excluded = [p.resolve() for p in a.exclude] + [a.output.resolve()]
    candidates = set(range(a.first_candidate, a.first_candidate + a.candidate_count))
    used, seen, conflicts = set(), set(), set()
    files, errors, archives = [], [], []
    candidate_re = re.compile(rb'(?<!\d)(?:' + b'|'.join(str(x).encode() for x in sorted(candidates)) + rb')(?!\d)')
    for root in a.root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory, dirs, names in os.walk(root):
            here = Path(directory).resolve()
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("venv_")
                             and not any((here / d) == e or e in (here / d).parents for e in excluded))
            for name in sorted(names):
                path = here / name
                if path.is_symlink() or path in seen or any(path == e or e in path.parents for e in excluded):
                    continue
                if name.endswith((".tar.gz", ".tgz", ".zip")):
                    archives.append(str(path))
                    continue
                if path.suffix not in (".json", ".jsonl", ".csv", ".tsv"):
                    continue
                seen.add(path)
                try:
                    before = path.stat()
                    raw = path.read_bytes()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError("File changed during audit")
                    seeds = {int(v) for m in SEED_VALUE.finditer(raw) for v in re.findall(rb'\d+', m[1])}
                    seeds.update(int(m[1]) for m in SEED_NAME.finditer(raw))
                    seeds.update(int(m[1]) for m in SEED_NAME.finditer(str(path).encode()))
                    # CSV seed columns can lack JSON-style key/value declarations.
                    if path.suffix in (".csv", ".tsv"):
                        with path.open(encoding="utf-8-sig", newline="") as f:
                            reader = csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ",")
                            cols = [c for c in (reader.fieldnames or []) if "seed" in c.lower()]
                            if cols:
                                for row in reader:
                                    for col in cols:
                                        value = str(row.get(col, "")).strip()
                                        if value.isdigit():
                                            seeds.add(int(value))
                    # Also reject prospective IDs mentioned anywhere in the bytes,
                    # including nonstandard source mappings and incomplete logs.
                    hits = {int(m[0]) for m in candidate_re.finditer(raw)}
                    conflicts.update(hits | (seeds & candidates))
                    used.update(seeds)
                    files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                  "bytes": len(raw), "seed_values": sorted(seeds),
                                  "prospective_id_mentions": sorted(hits)})
                    if len(files) % 500 == 0:
                        print(json.dumps({"files": len(files), "unique_seed_values": len(used),
                                          "seconds": time.monotonic() - started}), flush=True)
                except Exception as exc:
                    errors.append({"path": str(path), "error": repr(exc)})
    report = {"schema": "enumeration_seed_inventory_v1", "roots": list(map(str, a.root)),
              "excluded": list(map(str, excluded)), "files": files, "used_seed_values": sorted(used),
              "candidate_seeds": sorted(candidates), "conflicts": sorted(conflicts), "errors": errors,
              "archives_not_opened": archives, "seconds": time.monotonic() - started,
              "status": "PASS_UNPACKED_INVENTORY" if not errors and not conflicts else "FAIL",
              "limitation": "Audits unpacked JSON/JSONL and CSV/TSV artifacts; archive-only history and seed ranges require separate review."}
    (a.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "conflicts", "errors", "seconds")}
                     | {"files": len(files), "unique_seed_values": len(used), "archives": archives}), flush=True)
    if errors or conflicts:
        raise RuntimeError("History inventory is incomplete or prospective seeds overlap")


if __name__ == "__main__":
    main()
