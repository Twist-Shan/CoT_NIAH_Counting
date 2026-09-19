#!/usr/bin/env python3
"""Select one well-supported native trace grammar using discovery seeds only.

The analysis keeps the causal-aligned ``item_end`` counter site fixed.  A
grammar/layer pair is eligible only when both registered splits contain all ten
running-index classes and confirmation has at least two states per class.  The
two-state floor keeps Gemma's rare k=10 stratum; its uncertainty is reported.
Selection uses grouped discovery OOF balanced accuracy; confirmation is read
only after the winner has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realistic_niah_v5.causal_aligned_geometry import load_causal_aligned_native_capture
from realistic_niah_v5.cross_mode_geometry import CLASSES, ModeDataset
from realistic_niah_v5.trace_stratified_geometry import (
    confirmation_metrics,
    grouped_discovery_cv_metrics,
)

MODELS = ("Qwen3-8B", "Gemma4-E4B")
SCHEMA = "realistic_niah_v5_native_clean_grammar_geometry_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def subset(dataset: ModeDataset, mask: np.ndarray) -> ModeDataset:
    result = ModeDataset(
        mode=dataset.mode,
        model_label=dataset.model_label,
        metadata=dataset.metadata.loc[mask].reset_index(drop=True),
        states_by_layer={k: v[mask] for k, v in dataset.states_by_layer.items()},
    )
    result.validate()
    return result


def support(metadata: pd.DataFrame) -> tuple[bool, int, list[dict]]:
    frame = metadata.groupby(["split", "occurrence"]).size().rename("n").reset_index()
    for split in ("discovery", "confirmation"):
        got = set(frame.loc[frame.split.eq(split), "occurrence"].astype(int))
        if got != set(CLASSES):
            return False, 0, frame.to_dict("records")
    cmin = int(frame.loc[frame.split.eq("confirmation"), "n"].min())
    return cmin >= 2, cmin, frame.to_dict("records")


def metrics(states, metadata, pca_dim, folds, seed):
    return {
        **grouped_discovery_cv_metrics(
            states, metadata, CLASSES, pca_dim=pca_dim,
            random_state=seed, folds=folds, pca_whiten=True,
        ),
        **confirmation_metrics(
            states, metadata, CLASSES, pca_dim=pca_dim,
            random_state=seed, pca_whiten=True,
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--running-root", type=Path, default=ROOT / "work/v5_geometry_full_panel/running")
    p.add_argument("--event-registry", type=Path, default=ROOT / "reports/v5_native_causal_site_review/event_registry.csv")
    p.add_argument("--output", type=Path, default=ROOT / "reports/v5_native_clean_grammar_geometry")
    p.add_argument("--pca-dim", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    candidates, supports, winners, payload = [], [], [], {}
    for model in MODELS:
        ds, audit = load_causal_aligned_native_capture(
            a.running_root / model / "capture_index.jsonl", a.event_registry,
            site_kind="item_end",
        )
        payload[model] = {"loader_audit": audit, "grammars": {}}
        for grammar in sorted(ds.metadata.grammar_class.dropna().astype(str).unique()):
            mask = ds.metadata.grammar_class.astype(str).eq(grammar).to_numpy()
            gd = subset(ds, mask)
            eligible, cmin, table = support(gd.metadata)
            supports.append({
                "model_label": model, "grammar_class": grammar,
                "rows": len(gd.metadata), "trajectories": gd.metadata.request_id.nunique(),
                "eligible": eligible, "confirmation_min_per_class": cmin,
                "support": json.dumps(table, ensure_ascii=False),
            })
            if not eligible:
                continue
            for layer, states in sorted(gd.states_by_layer.items()):
                row = {"model_label": model, "grammar_class": grammar, "layer": layer,
                       "rows": len(gd.metadata), **metrics(states, gd.metadata, a.pca_dim, a.folds, a.seed)}
                candidates.append(row)

        frame = pd.DataFrame([r for r in candidates if r["model_label"] == model])
        if frame.empty:
            raise ValueError(f"No eligible grammar for {model}")
        win = frame.sort_values(
            ["discovery_selection_score", "discovery_oof_ncc_balanced_accuracy",
             "discovery_oof_logistic_balanced_accuracy", "rows", "layer", "grammar_class"],
            ascending=[False, False, False, False, True, True], kind="mergesort",
        ).iloc[0].to_dict()
        winners.append(win)
        grammar, layer = str(win["grammar_class"]), int(win["layer"])
        mask = ds.metadata.grammar_class.astype(str).eq(grammar).to_numpy()
        gd = subset(ds, mask)
        discovery = gd.metadata.split.astype(str).eq("discovery").to_numpy()
        pca = PCA(n_components=3, random_state=a.seed).fit(gd.states_by_layer[layer][discovery].astype(np.float32))
        xyz = pca.transform(gd.states_by_layer[layer].astype(np.float32))
        payload[model]["selected"] = {"grammar_class": grammar, "layer": layer,
            "pca3_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "points": [{
                "x": float(v[0]), "y": float(v[1]), "z": float(v[2]),
                "split": str(m.split), "seed": int(m.seed), "occurrence": int(m.occurrence),
                "request_id": str(m.request_id),
            } for v, m in zip(xyz, gd.metadata.itertuples(index=False))]}
        print(model, grammar, f"L{layer}", f"disc={win['discovery_selection_score']:.3f}",
              f"conf={win['confirmation_logistic_balanced_accuracy']:.3f}/{win['confirmation_ncc_balanced_accuracy']:.3f}")

    a.output.mkdir(parents=True, exist_ok=True)
    paths = {"candidates": a.output / "grammar_layer_candidates.csv",
             "support": a.output / "grammar_support.csv",
             "selected": a.output / "selected_clean_grammar.csv",
             "payload": a.output / "geometry_payload.json"}
    pd.DataFrame(candidates).to_csv(paths["candidates"], index=False)
    pd.DataFrame(supports).to_csv(paths["support"], index=False)
    pd.DataFrame(winners).to_csv(paths["selected"], index=False)
    paths["payload"].write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    audit = {"schema_version": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
             "selection_rule": "grammar and layer selected only by grouped discovery OOF mean Logistic/NCC balanced accuracy; confirmation evaluated after freeze",
             "eligibility": "both splits cover occurrences 1..10; confirmation minimum >=2 per occurrence (Gemma k=10 has n=2)",
             "site": "causal-aligned item_end/p0_item_end", "pca_dim": a.pca_dim,
             "outputs": {str(p.resolve()): sha256(p) for p in paths.values()}}
    (a.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
