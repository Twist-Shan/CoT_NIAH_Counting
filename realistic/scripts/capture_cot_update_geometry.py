"""Probe no-index item endpoints at one previously frozen geometry layer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
import torch

from realistic_niah_v5.counting_mechanism_transfer import build_first_pass_tstar_answer_source_registry
from realistic_niah_v5.trace_stratified_geometry import _fit_projection_and_predict, confirmation_metrics
from scripts.run_realistic_niah_v5_same_site_progress_transplant import (
    _chunk_forward, _encoding_tensors, _experiment_model, _read_rows, _tensor_from_output,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--attention-backend", default="sdpa")
    args = p.parse_args()
    start = time.monotonic()
    frozen = json.loads((args.bundle / "manifest.json").read_text())
    info = frozen["models"][args.model]
    source = args.bundle / "inputs" / f"{args.model}_geometry.jsonl"
    rel = str(source.relative_to(args.bundle)).replace("\\", "/")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen["input_code_sha256"][rel]
    prompted = info["cohort_mode"] == "prompt_conditioned_noindex"
    population = "gemma_prompt_conditioned_noindex_found_v3" if prompted else "first_pass_noindex_enumeration"
    eligibility = "primary_eligible_prompt_conditioned_noindex" if prompted else "primary_eligible_prefix_clean"
    rows = _read_rows(source, gold_count=10, seeds=info["discovery_seeds"] + info["confirmation_seeds"],
                      max_seeds=None, selection_population=population, eligibility_field=eligibility)
    model, tokenizer, adapter = _experiment_model(args)
    layer = int(info["layer_zero_based"])
    assert not model.training and 0 <= layer < adapter.num_layers
    arrays, metadata, timings = [], [], []
    loaded = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            tick = time.monotonic()
            encoding, registry = build_first_pass_tstar_answer_source_registry(
                row, tokenizer, candidate_counts=tuple(range(1, 11)),
                selection_population=population, eligibility_field=eligibility)
            positions = [int(end) - 1 for _, end in registry.trace_items]
            assert len(positions) == 10 and len(set(positions)) == 10
            input_ids, attention_mask = _encoding_tensors(model, encoding)
            captured = {}
            chunk_start = 0

            def hook(_module, _arguments, output):
                hidden = _tensor_from_output(output)
                for pos in positions:
                    if chunk_start <= pos < chunk_start + hidden.shape[1]:
                        if pos in captured:
                            raise RuntimeError("Endpoint captured twice")
                        captured[pos] = hidden[0, pos - chunk_start].detach().float().cpu().numpy()

            handle = adapter.layers[layer].register_forward_hook(hook)
            previous = None
            try:
                for chunk_start in range(0, max(positions) + 1, 512):
                    previous = _chunk_forward(model, adapter, input_ids, attention_mask,
                                              start=chunk_start, end=min(max(positions) + 1, chunk_start + 512),
                                              previous=previous)
            finally:
                handle.remove()
            assert set(captured) == set(positions)
            for k, pos in enumerate(positions, 1):
                value = captured[pos]
                assert np.isfinite(value).all()
                arrays.append(value)
                metadata.append(dict(seed=int(row["seed"]), occurrence=k,
                                     split="discovery" if row["seed"] in info["discovery_seeds"] else "confirmation",
                                     position=pos, token_id=int(encoding.input_ids[pos]), request_id=row["request_id"]))
            del previous
            elapsed = time.monotonic() - tick
            timings.append(dict(seed=row["seed"], seconds=elapsed))
            print(f"CAPTURE {len(timings)}/30 seed={row['seed']} seconds={elapsed:.2f}", flush=True)
    states = np.asarray(arrays, dtype=np.float32)
    meta = pd.DataFrame(metadata)
    assert states.shape[0] == 300
    train = meta.split.eq("discovery").to_numpy()
    labels = np.arange(1, 11)
    assert train.sum() == 200 and (~train).sum() == 100
    metrics = confirmation_metrics(states, meta, labels, pca_dim=16, random_state=0, pca_whiten=True)
    logistic, ncc, _, _ = _fit_projection_and_predict(states[train], meta.loc[train, "occurrence"].to_numpy(),
                                                     states[~train], labels, pca_dim=16,
                                                     random_state=0, pca_whiten=True)
    predicted = meta.loc[~train].copy()
    predicted["ncc_prediction"] = ncc
    predicted["logistic_prediction"] = logistic
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "selected_states.npz", states=states.astype(np.float16))
    meta.to_csv(args.output / "state_metadata.csv", index=False)
    predicted.to_csv(args.output / "confirmation_predictions.csv", index=False)
    metrics.update(model=args.model, layer_zero_based=layer, layer_one_based=layer + 1,
                   site="last token of parsed item", discovery_states=200, confirmation_states=100,
                   probe="discovery StandardScaler + whitened PCA16; NCC and class-weighted L2 logistic C=1",
                   layer_selection=frozen.get("layer_selection_basis", "Frozen full-cohort geometry"),
                   cohort_mode=info["cohort_mode"], input_sha256=frozen["input_code_sha256"][rel])
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "process_status.json").write_text(json.dumps(dict(
        status="COMPLETE", seconds=time.monotonic() - start, load_seconds=loaded - start, captures=timings,
        python=sys.version, packages={name: importlib.metadata.version(name) for name in
                                     ("torch", "transformers", "numpy", "scikit-learn")}), indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
