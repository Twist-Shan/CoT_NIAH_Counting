"""Freeze fresh stimuli and an isolated code snapshot for the alignment supplement."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-code", "extensions", "config", "history-audit", "history-review", "data-root", "source-stimuli", "cache-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    tick = time.monotonic()
    cfg = json.loads(a.config.read_text())
    history = json.loads(a.history_audit.read_text())
    review = json.loads(a.history_review.read_text())
    assert history["status"] == "PASS_UNPACKED_INVENTORY"
    assert review["status"] == "PASS" and review["history_audit_sha256"] == sha(a.history_audit)
    start = cfg["first_seed"]
    discovery = list(range(start, start + cfg["discovery_seed_count"]))
    confirmation = list(range(discovery[-1] + 1, discovery[-1] + 1 + cfg["confirmation_candidate_count"]))
    read_seeds = confirmation[:cfg["read_seed_count"]]
    assert history["candidate_seeds"] == discovery + confirmation and not history["conflicts"]
    a.output.mkdir(parents=True, exist_ok=False)
    code = a.output / "code"
    shutil.copytree(a.source_code, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "work"))
    for source in a.extensions.glob("*.py"):
        shutil.copy2(source, code / "scripts" / source.name)
    shutil.copy2(a.config, a.output / "protocol.json")
    shutil.copy2(a.history_review, a.output / "history_review.json")
    sys.path[:0] = [str(code / "src"), str(code)]
    from dataset_generation.dynamic_niah import TokenizerAdapter
    from realistic_niah_v4.spec import V4Config
    from realistic_niah_v4.stimuli import ControlledFreezeSpec, build_controlled_family
    from scripts.build_realistic_niah_v6_replacement_seed_pool import _anchor_signature
    v4 = V4Config(seeds=tuple([1234, 1254] + discovery + confirmation),
                  discovery_seeds=tuple([1234] + discovery), confirmation_seeds=tuple([1254] + confirmation))
    v4.validate()
    tokenizer = TokenizerAdapter(v4.canonical_tokenizer, revision=v4.canonical_tokenizer_revision, cache_dir=str(a.cache_dir))
    if tokenizer.backend != "huggingface":
        raise RuntimeError(tokenizer.load_error)
    data = a.data_root
    freeze = ControlledFreezeSpec(config=v4, haystack_dir=str(data / "haystacks/paul_graham"),
                                 entities_path=str(data / "entities/cities.csv"),
                                 fact_templates_path=str(data / "templates/niah_fact_single_template.txt"),
                                 tokenizer_cache_dir=str(a.cache_dir))
    anchors = {}
    with a.source_stimuli.open() as f:
        for line in f:
            row = json.loads(line)
            if row["seed"] in (1234, 1254) and row["design_variant"] == "v4.4":
                anchors[row["seed"], row["gold_count"]] = row
    for seed in (1234, 1254):
        family, _ = build_controlled_family(variant="v4.4", seed=seed, tokenizer=tokenizer, freeze_spec=freeze)
        for row in family:
            assert _anchor_signature(row) == _anchor_signature(anchors[seed, row["gold_count"]]), (seed, row["gold_count"])
    rows, families = [], []
    for i, seed in enumerate(discovery + confirmation):
        family, meta = build_controlled_family(variant="v4.4", seed=seed, tokenizer=tokenizer, freeze_spec=freeze)
        families.append(meta)
        for row in family:
            if seed not in read_seeds and row["gold_count"] != cfg["causal_count"]:
                continue
            row["split"] = "discovery" if seed in discovery else "confirmation"
            row["alignment_roles"] = (["unfiltered_read"] if seed in read_seeds else []) + (["causal_discovery" if seed in discovery else "causal_candidate"] if row["gold_count"] == 10 else [])
            rows.append(row)
        print(f"STIMULI {i+1}/{len(discovery+confirmation)} seed={seed}", flush=True)
    # First row is the same preregistered confirmation N=10 technical smoke.
    rows.sort(key=lambda r: (not (r["seed"] == read_seeds[0] and r["gold_count"] == 10), r["seed"] not in read_seeds, r["seed"], r["gold_count"]))
    assert len(rows) == 160 and len({(r["seed"], r["gold_count"]) for r in rows}) == 160
    inputs = a.output / "stimuli.jsonl"
    with inputs.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    code_hashes = {str(f.relative_to(code)): sha(f) for d in ("src", "scripts", "configs", "tests")
                   for f in sorted((code / d).rglob("*")) if f.suffix in (".json", ".py")}
    data_hashes = {str(f): sha(f) for sub in ("haystacks/paul_graham", "entities", "templates")
                   for f in sorted((data / sub).rglob("*")) if f.is_file()}
    manifest = {"status": "FROZEN_BEFORE_GENERATION", "schema": "enumeration_fresh_bundle_v1",
                "discovery_seeds": discovery, "confirmation_candidates": confirmation, "read_seeds": read_seeds,
                "rows_per_cell": len(rows), "baseline_cells": 6, "baseline_rows": 960,
                "stimuli_sha256": sha(inputs), "protocol_sha256": sha(a.output / "protocol.json"),
                "history_audit": str(a.history_audit), "history_audit_sha256": sha(a.history_audit),
                "source_stimuli_sha256": sha(a.source_stimuli), "anchor_seeds_exact_match": [1234,1254],
                "code_sha256": code_hashes, "data_sha256": data_hashes, "families": families,
                "seconds": time.monotonic() - tick}
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status":manifest["status"],"rows":len(rows),"seconds":manifest["seconds"]}),flush=True)


if __name__ == "__main__":
    main()
