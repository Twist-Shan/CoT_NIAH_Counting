"""Select a layer using only the frozen no-index discovery cohort."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_protocol(args):
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["models"][args.model]
    source = bundle / "inputs" / f"{args.model}_geometry.jsonl"
    relative = source.relative_to(bundle).as_posix()
    assert sha(source) == manifest["input_code_sha256"][relative]
    discovery, confirmation = info["discovery_seeds"], info["confirmation_seeds"]
    assert len(discovery) == len(set(discovery)) == 20
    assert not set(discovery) & set(confirmation)
    code = bundle / "code"
    required = [
        "src/realistic_niah_v5/trace_stratified_geometry.py",
        "src/realistic_niah_v5/counting_mechanism_transfer.py",
        "scripts/run_realistic_niah_v5_same_site_progress_transplant.py",
    ]
    for relative_code in required:
        assert sha(code / relative_code) == manifest["input_code_sha256"]["code/" + relative_code]
    sys.path[:0] = [str(code), str(code / "src")]
    expected_layers = {"Qwen3-8B": 36, "Gemma4-E4B": 42}[args.model]
    protocol = dict(
        model=args.model, cohort_mode=info["cohort_mode"], input_sha256=sha(source),
        discovery_seeds=discovery, states=200, labels=list(range(1, 11)),
        expected_layers=expected_layers, previous_layer_zero_based=info["layer_zero_based"],
        selection="Maximize discovery five-fold seed-grouped NCC balanced accuracy; ties use logistic, then earlier layer; scores rounded to 12 decimals.",
        probe="Per-fold StandardScaler, whitened PCA16, NCC without shrinkage; class-balanced L2 logistic C=1.",
        pca_random_state=0, folds=5, chunk_size=512, attention_backend=args.attention_backend,
        torch_dtype=args.torch_dtype, endpoint="Last token of each parsed item, before prompt-filler deletion",
        confirmation_used=False, script_sha256=sha(__file__),
        frozen_code_sha256={relative_code: sha(code / relative_code) for relative_code in required},
    )
    return source, info, protocol


def capture(args, source, info, protocol):
    import torch
    from realistic_niah_v5.counting_mechanism_transfer import build_first_pass_tstar_answer_source_registry
    from scripts.run_realistic_niah_v5_same_site_progress_transplant import (
        _chunk_forward, _encoding_tensors, _experiment_model, _read_rows, _tensor_from_output,
    )

    prompted = info["cohort_mode"] == "prompt_conditioned_noindex"
    population = "gemma_prompt_conditioned_noindex_found_v3" if prompted else "first_pass_noindex_enumeration"
    eligibility = "primary_eligible_prompt_conditioned_noindex" if prompted else "primary_eligible_prefix_clean"
    rows = _read_rows(source, gold_count=10, seeds=info["discovery_seeds"], max_seeds=None,
                      selection_population=population, eligibility_field=eligibility)
    assert len(rows) == 20 and {int(r["seed"]) for r in rows} == set(info["discovery_seeds"])
    assert torch.cuda.is_available()
    assert float((torch.ones(4, device="cuda") * 2).sum()) == 8.0
    model, tokenizer, adapter = _experiment_model(args)
    assert not model.training and adapter.num_layers == protocol["expected_layers"]
    arrays, metadata, durations = [], [], []
    with torch.inference_mode():
        for row in rows:
            tick = time.monotonic()
            encoding, registry = build_first_pass_tstar_answer_source_registry(
                row, tokenizer, candidate_counts=tuple(range(1, 11)),
                selection_population=population, eligibility_field=eligibility)
            positions = [int(end) - 1 for _, end in registry.trace_items]
            assert len(positions) == len(set(positions)) == 10
            input_ids, attention_mask = _encoding_tensors(model, encoding)
            captured = {}
            chunk_start = 0

            def make_hook(layer):
                def hook(_module, _arguments, output):
                    hidden = _tensor_from_output(output)
                    local = [(k, pos - chunk_start) for k, pos in enumerate(positions)
                             if chunk_start <= pos < chunk_start + hidden.shape[1]]
                    if local:
                        values = hidden[0, [pos for _, pos in local]].detach().float().cpu().numpy()
                        for (k, _), value in zip(local, values):
                            assert (k, layer) not in captured
                            captured[k, layer] = value
                return hook

            handles = [block.register_forward_hook(make_hook(layer))
                       for layer, block in enumerate(adapter.layers)]
            previous = None
            try:
                for chunk_start in range(0, max(positions) + 1, 512):
                    previous = _chunk_forward(model, adapter, input_ids, attention_mask,
                                              start=chunk_start, end=min(max(positions) + 1, chunk_start + 512),
                                              previous=previous)
            finally:
                for handle in handles:
                    handle.remove()
            assert len(captured) == 10 * adapter.num_layers
            values = np.stack([np.stack([captured[k, layer] for layer in range(adapter.num_layers)])
                               for k in range(10)])
            assert np.isfinite(values).all()
            arrays.append(values)
            for k, pos in enumerate(positions, 1):
                metadata.append(dict(seed=int(row["seed"]), occurrence=k, split="discovery", position=pos,
                                     token_id=int(encoding.input_ids[pos]), request_id=row["request_id"]))
            del previous
            durations.append(dict(seed=int(row["seed"]), seconds=time.monotonic() - tick))
            print(f"CAPTURE {args.model} {len(arrays)}/20 seed={row['seed']}", flush=True)
    states = np.concatenate(arrays, axis=0)
    frame = pd.DataFrame(metadata)
    assert states.shape[:2] == (200, protocol["expected_layers"])
    np.savez_compressed(args.output / "discovery_all_layers.npz", states=states,
                        layer_indices=np.arange(adapter.num_layers))
    frame.to_csv(args.output / "state_metadata.csv", index=False)

    reference_dir = args.bundle / "geometry" / args.model
    with np.load(reference_dir / "selected_states.npz") as archive:
        reference = archive["states"]
    reference_meta = pd.read_csv(reference_dir / "state_metadata.csv")
    reference_rows = reference_meta[reference_meta["split"].eq("discovery")]
    source_keys = [(int(r.seed), int(r.occurrence)) for r in reference_rows.itertuples()]
    target_keys = [(int(r.seed), int(r.occurrence)) for r in frame.itertuples()]
    assert source_keys == target_keys
    assert np.array_equal(frame[["position", "token_id"]].to_numpy(),
                          reference_rows[["position", "token_id"]].to_numpy())
    actual = states[:, info["layer_zero_based"]].astype(np.float16)
    expected = reference[reference_meta["split"].eq("discovery").to_numpy()]
    exact = bool(np.array_equal(actual, expected))
    audit = dict(status="PASS" if exact else "FAIL", previous_layer_states_exact=exact,
                 previous_layer_one_based=info["layer_one_based"], compared_states=200,
                 max_abs_difference=float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))),
                 captures=durations, gpu=torch.cuda.get_device_name(0),
                 packages={name: importlib.metadata.version(name) for name in
                           ["numpy", "scikit-learn", "torch", "transformers"]})
    save_json(args.output / "capture_audit.json", audit)
    assert exact, "Fixed-layer states differ from the previous no-index capture; inspect before selection"


def analyze(args, protocol):
    from realistic_niah_v5.trace_stratified_geometry import grouped_discovery_cv_metrics

    frame = pd.read_csv(args.output / "state_metadata.csv")
    with np.load(args.output / "discovery_all_layers.npz") as archive:
        states, layers = archive["states"], archive["layer_indices"]
    assert states.shape[:2] == (200, protocol["expected_layers"])
    assert np.array_equal(layers, np.arange(protocol["expected_layers"]))
    assert np.isfinite(states).all() and set(frame["split"]) == {"discovery"}
    assert set(frame["seed"]) == set(protocol["discovery_seeds"])
    assert all(sorted(group["occurrence"]) == list(range(1, 11)) for _, group in frame.groupby("seed"))
    metrics = []
    for layer in layers:
        tick = time.monotonic()
        values = grouped_discovery_cv_metrics(states[:, layer], frame, np.arange(1, 11),
                                              pca_dim=16, random_state=0, folds=5, pca_whiten=True)
        assert values["discovery_oof_rows"] == 200 and values["discovery_fold_count"] == 5
        metrics.append(dict(layer_zero_based=int(layer), layer_one_based=int(layer) + 1,
                            **values, seconds=time.monotonic() - tick))
        print(f"CV {args.model} L{layer+1} NCC={values['discovery_oof_ncc_balanced_accuracy']:.4f}", flush=True)
    candidates = pd.DataFrame(metrics)
    candidates.to_csv(args.output / "discovery_layer_metrics.csv", index=False)
    ranked = sorted(metrics, key=lambda row: (-round(row["discovery_oof_ncc_balanced_accuracy"], 12),
                                              -round(row["discovery_oof_logistic_balanced_accuracy"], 12),
                                              row["layer_zero_based"]))
    save_json(args.output / "selection.json", dict(status="PASS", protocol=protocol, selected=ranked[0],
                                                  top_five=ranked[:5], layers_evaluated=len(metrics),
                                                  states_sha256=sha(args.output / "discovery_all_layers.npz"),
                                                  metadata_sha256=sha(args.output / "state_metadata.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["capture", "analyze", "all"], default="all")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    args = parser.parse_args()
    tick = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        source, info, protocol = load_protocol(args)
        config = args.output / "protocol.json"
        if config.exists():
            assert json.loads(config.read_text()) == protocol, "Output directory belongs to another protocol"
        else:
            save_json(config, protocol)
        save_json(args.output / "status.json", dict(state="RUNNING", command=sys.argv))
        if args.phase in ["capture", "all"]:
            assert not (args.output / "discovery_all_layers.npz").exists(), "Existing capture must be preserved"
            capture(args, source, info, protocol)
        if args.phase in ["analyze", "all"]:
            analyze(args, protocol)
        save_json(args.output / "status.json", dict(state="COMPLETE", phase=args.phase,
                                                   seconds=time.monotonic() - tick, command=sys.argv))
    except Exception:
        save_json(args.output / "status.json", dict(state="FAILED", seconds=time.monotonic() - tick,
                                                   traceback=traceback.format_exc(), command=sys.argv))
        raise


if __name__ == "__main__":
    main()
