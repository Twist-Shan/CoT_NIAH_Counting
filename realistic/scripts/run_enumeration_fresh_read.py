"""Run the five fixed Read source-blanking arms on the unfiltered fresh pool."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.enumeration_fresh_geometry import build_read_geometry, json_sha
from scripts.prepare_enumeration_fresh_causal_registry import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    tick = time.monotonic()
    manifest = json.loads((args.registry / "manifest.json").read_text())
    cfg = json.loads((args.bundle / "protocol.json").read_text())
    assert manifest["status"] == "FROZEN_BASELINE_GEOMETRY"
    assert sha(args.bundle / "manifest.json") == manifest["baseline_manifest_sha256"]
    assert sha(args.registry / "ledger.json") == manifest["ledger_sha256"]
    assert sha(args.registry / "adapted_generations.jsonl") == manifest["adapted_generations_sha256"]
    for name, expected in manifest["new_entrypoints_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    baseline_manifest = json.loads((args.bundle / "manifest.json").read_text())
    assert sha(args.bundle / "protocol.json") == baseline_manifest["protocol_sha256"]
    for name, expected in baseline_manifest["code_sha256"].items():
        if name.startswith(("src/", "scripts/")):
            assert sha(ROOT / name) == expected, name
    ledger = json.loads((args.registry / "ledger.json").read_text())
    population = [row for row in ledger if "unfiltered_read" in row["roles"]]
    assert len(population) == 100
    eligible = [row for row in population if row["read_eligible"]]
    if not eligible:
        raise ValueError("No legal Read query in the registered 100-input pool")
    selected = eligible[:1] if args.smoke else eligible
    rows = {}
    with (args.registry / "adapted_generations.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["seed"]), int(row["gold_count"])] = row
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "completed": 0, "total": len(selected) * len(cfg["read"]["conditions"]),
             "population": 100, "eligible_inputs": len(eligible), "smoke": args.smoke}
    write(args.output / "status.json", state)
    write(args.output / "coverage.json", population)
    import torch
    from realistic_niah_v4.modeling import generate_answer_completion, load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.causal import completion_metrics
    from realistic_niah_v5.token_level_ablation import blank_token_states
    try:
        model, tokenizer, adapter = load_registered_model(resolve_model_spec(manifest["model"]),
            cache_dir=args.cache_dir, device_map="auto", torch_dtype="bfloat16", attention_backend="sdpa")
        model.eval()
        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        runtime = {"model": manifest["model"], "mode": manifest["mode"], "torch": torch.__version__,
                   "transformers": importlib.metadata.version("transformers"), "python": sys.version,
                   "gpu": torch.cuda.get_device_name(), "dtype": str(next(model.parameters()).dtype),
                   "backend": "sdpa", "model_revision": resolve_model_spec(manifest["model"]).revision,
                   "model_source_sha256": sha(sys.modules[type(model).__module__].__file__),
                   "layers": adapter.num_layers, "model_load_seconds": time.monotonic() - tick,
                   "registry_sha256": sha(args.registry / "manifest.json"),
                   "runner_sha256": sha(__file__), "command": sys.argv,
                   "max_new_tokens": cfg["read"]["max_new_tokens"],
                   "head_attention_readout": "not_collected; behavioral source-blanking assay"}
        write(args.output / "runtime.json", runtime)
        with (args.output / "trials.jsonl").open("x") as output:
            for entry in selected:
                row = rows[entry["seed"], entry["gold_count"]]
                path = args.registry / entry["geometry_file"]
                assert sha(path) == entry["geometry_sha256"]
                frozen = json.loads(path.read_text())
                assert json_sha(row) == frozen["adapted_row_sha256"]
                encoding, masks, geometry = build_read_geometry(row, tokenizer, mode=manifest["mode"])
                assert json_sha(geometry) == json_sha(frozen["read"])
                for condition in cfg["read"]["conditions"]:
                    positions = masks[condition]
                    probe = []
                    def check_zero(_module, _inputs, result):
                        if positions and result.shape[1] > 1:
                            active = [p for p in positions if p < result.shape[1]]
                            if active:
                                probe.append(float(result[:, active, :].detach().abs().max()))
                    with blank_token_states(model, adapter, positions) as hooks:
                        handle = model.get_input_embeddings().register_forward_hook(check_zero)
                        try:
                            generated = generate_answer_completion(model, tokenizer, encoding,
                                max_new_tokens=cfg["read"]["max_new_tokens"])
                        finally:
                            handle.remove()
                    if positions:
                        assert probe and max(probe) == 0.0, "Blank embedding probe did not observe exact zero"
                    else:
                        assert hooks["blank_embedding_hook_applications"] == 0
                    assert model.config._attn_implementation == text_cfg._attn_implementation == "sdpa"
                    assert json_sha(list(encoding.input_ids)) == geometry["input_ids_sha256"]
                    assert json_sha(list(encoding.attention_mask)) == geometry["attention_mask_sha256"]
                    trial = {"seed": entry["seed"], "gold_count": entry["gold_count"],
                             "request_id": entry["request_id"], "model": manifest["model"], "mode": manifest["mode"],
                             "condition": condition, "status": "ok", "blank_token_count": len(positions),
                             "blank_positions_sha256": json_sha(list(positions)),
                             "geometry_sha256": entry["geometry_sha256"], "blank_hook_audit": hooks,
                             "embedding_zero_probe": probe, "generated": generated,
                             **completion_metrics(generated, gold_count=encoding.count)}
                    output.write(json.dumps(trial, ensure_ascii=True) + "\n")
                    output.flush()
                    state.update(completed=state["completed"] + 1, seed=entry["seed"],
                                 gold_count=entry["gold_count"], seconds=time.monotonic() - tick)
                    write(args.output / "status.json", state)
                    print(json.dumps(state), flush=True)
        state.update(status="COMPLETE", trials_sha256=sha(args.output / "trials.jsonl"))
    except BaseException as error:
        state.update(status="FAILED", error=repr(error))
        raise
    finally:
        state["seconds"] = time.monotonic() - tick
        write(args.output / "status.json", state)


if __name__ == "__main__":
    main()
