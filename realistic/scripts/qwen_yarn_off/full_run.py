"""Reproduce the Gemma stimulus grid with Qwen3-32B and unscaled RoPE.

Inputs and prompt hashes are checked against archived runs. Each worker owns
disjoint seeds, checkpoints every batch, and counts every truncated output wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def request_key(row):
    return (f"L{int(row['target_passage_tokens']):06d}_"
            f"N{int(row['num_needles']):02d}_seed{int(row['seed'])}_"
            f"{row['prompt_mode']}")


def expected_keys(config):
    return {request_key(dict(target_passage_tokens=L, num_needles=N, seed=s, prompt_mode=m))
            for L in config["passage_lengths"] for N in config["needle_counts"]
            for s in config["seeds"] for m in config["prompt_modes"]}


def score_with_truncation_policy(evaluation):
    result = dict(evaluation)
    result["parsed_exact_count"] = bool(result["exact_count"])
    result["exact_count"] = bool(result["exact_count"] and not result["truncated"])
    result["scoring_policy"] = "parsed_exact_count_and_not_truncated_v1"
    return result


def check_rope(value):
    for key in ("rope_scaling", "rope_parameters"):
        params = value.get(key)
        if params is not None:
            assert isinstance(params, dict), (key, params)
            assert params.get("rope_type", params.get("type", "default")) == "default", params
            assert params.get("factor", 1.0) == 1.0, params


def archived_files(config, label):
    b = Path(config["backup_root"])
    short = b / "realistic_niah_v3_1/20260819_formal"
    long = b / "realistic_niah_v3_3_long_context/20260906_holdout"
    return [*(short / f"shards/{label}__{m}/main/requests.jsonl" for m in config["prompt_modes"]),
            long / f"formal/{label}/worker-0/main/requests.jsonl",
            long / f"formal/{label}/worker-1/main/requests.jsonl"]


def prepare(root, config):
    started = time.perf_counter()
    wanted = expected_keys(config)
    assert len(wanted) == config["expected_requests"]
    refs, source_files = {}, []
    for label in ("Gemma4-31B", "Qwen3-32B"):
        found = set()
        for path in archived_files(config, label):
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for line in f:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["prompt_mode"] not in config["prompt_modes"]:
                        continue
                    key = request_key(row)
                    assert key in wanted and key not in found, (label, key)
                    found.add(key)
                    basic = {k: row[k] for k in ("stimulus_id", "target_passage_tokens", "num_needles", "seed", "prompt_mode", "passage_sha256")}
                    if label == "Gemma4-31B":
                        refs[key] = basic
                    else:
                        assert all(refs[key][k] == v for k, v in basic.items()), key
                        assert row["model_revision"] == config["model_revision"]
                        refs[key].update(prompt_payload_hashes=row["prompt_payload_hashes"],
                                         model_input_tokens=row["model_input_tokens"], decoding=row["decoding"])
            source_files.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size))
        assert found == wanted, (label, len(found), len(wanted - found))
        print(f"REFERENCE_GRID_OK model={label} requests={len(found)}", flush=True)
    b = Path(config["backup_root"])
    datasets = [b / "realistic_niah_v3_1/20260819_formal/dataset/stimuli.jsonl",
                b / "realistic_niah_v3_3_long_context/20260906_holdout/dataset/stimuli.jsonl"]
    index = {}
    for path in datasets:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row["stimulus_id"]
                assert sid not in index
                assert sha(row["passage"].encode()) == row["passage_sha256"], sid
                assert row["gold_count"] == len(row["gold_pairs"]) == row["num_needles"], sid
                matches = re.findall(r"In the 2024 city score audit, (.+?) received a score of (\d+)\.", row["passage"])
                assert len(matches) == row["num_needles"], sid
                for mode in config["prompt_modes"]:
                    ref = refs[request_key(row | {"prompt_mode": mode})]
                    assert ref["stimulus_id"] == sid and ref["passage_sha256"] == row["passage_sha256"]
                index[sid] = {k: row[k] for k in ("target_passage_tokens", "num_needles", "seed", "passage_sha256")}
                index[sid].update(path=str(path), offset=offset, bytes=len(line))
        source_files.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size))
        print(f"DATASET_INDEXED path={path} total_stimuli={len(index)}", flush=True)
    assert len(index) * len(config["prompt_modes"]) == config["expected_requests"]
    code_sha = {}
    for package in ("realistic_niah", "realistic_niah_v3", "realistic_niah_v3_1", "dataset_generation"):
        source = Path(config["source_code_root"]) / package
        for path in source.rglob("*.py"):
            rel = path.relative_to(Path(config["source_code_root"]))
            dest = root / "source_snapshot" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != dest.resolve():
                shutil.copy2(path, dest)
            code_sha[str(rel)] = sha(dest.read_bytes())
    save(root / "reference_requests.json", refs)
    save(root / "dataset_index.json", index)
    save(root / "preparation.json", dict(passed=True, created_at_utc=utc(), config_sha256=json_sha(config),
         expected_requests=len(wanted), unique_stimuli=len(index), source_files=source_files,
         source_code_sha256=code_sha, reference_requests_sha256=sha((root / "reference_requests.json").read_bytes()),
         dataset_index_sha256=sha((root / "dataset_index.json").read_bytes()),
         elapsed_seconds=time.perf_counter() - started))
    print(f"PREPARATION_COMPLETE requests={len(wanted)} seconds={time.perf_counter()-started:.1f}", flush=True)


def schedule(config, refs, worker):
    assigned = {k: v for k, v in refs.items() if config["seeds"].index(v["seed"]) % config["workers"] == worker}
    remaining = dict(assigned)
    batches = []
    seed = min(v["seed"] for v in assigned.values())
    # A prespecified small cross-section measures runtime before the long sweep.
    for L in (25000, 50000, 100000):
        for mode in config["prompt_modes"]:
            keys = [request_key(dict(target_passage_tokens=L, num_needles=N, seed=seed, prompt_mode=mode)) for N in (3, 20)]
            batches.append(("initial_runtime_sample", [remaining.pop(k) | {"request_key": k} for k in keys]))
    for L in config["execution_length_order"]:
        for seed in config["seeds"]:
            if config["seeds"].index(seed) % config["workers"] != worker:
                continue
            counts = config["needle_counts"]
            for i in range(0, len(counts), config["engine"]["request_batch_size"]):
                for mode in config["prompt_modes"]:
                    batch = []
                    for N in counts[i:i + config["engine"]["request_batch_size"]]:
                        key = request_key(dict(target_passage_tokens=L, num_needles=N, seed=seed, prompt_mode=mode))
                        if key in remaining:
                            batch.append(remaining.pop(key) | {"request_key": key})
                    if batch:
                        batches.append(("full_grid", batch))
    assert not remaining
    flattened = [r["request_key"] for _, rows in batches for r in rows]
    assert len(flattened) == len(set(flattened)) == len(assigned)
    assert set(flattened) == set(assigned)
    return batches, assigned


def summary(rows):
    rows = list(rows)
    return dict(completed_requests=len(rows), exact_correct=sum(bool(r["evaluation"]["exact_count"]) for r in rows),
                truncations=sum(bool(r["evaluation"]["truncated"]) for r in rows),
                parse_failures=sum(r["evaluation"]["parse_status"] != "ok" for r in rows),
                output_tokens=sum(r["output_tokens"] for r in rows))


def read_stimulus(index, ref):
    location = index[ref["stimulus_id"]]
    with Path(location["path"]).open("rb") as f:
        f.seek(location["offset"])
        stimulus = json.loads(f.read(location["bytes"]))
    assert stimulus["stimulus_id"] == ref["stimulus_id"]
    assert sha(stimulus["passage"].encode()) == ref["passage_sha256"]
    return stimulus


def verify_decoded_text(tokenizer, candidate):
    ids = list(candidate.token_ids)
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded == candidate.text:
        return "exact"
    assert candidate.finish_reason == "stop" and ids and ids[-1] == tokenizer.eos_token_id
    decoded = tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert decoded == candidate.text, "Generated text differs from token readback"
    return "terminal_eos_omitted"


def run_worker(root, config, worker):
    started = time.perf_counter()
    work = root / f"worker-{worker}"
    work.mkdir(parents=True, exist_ok=True)
    config_hash = json_sha(config)
    prep = json.loads((root / "preparation.json").read_text())
    assert prep["passed"] and prep["config_sha256"] == config_hash
    for rel, digest in prep["source_code_sha256"].items():
        assert sha((root / "source_snapshot" / rel).read_bytes()) == digest
    for filename, field in (("reference_requests.json", "reference_requests_sha256"), ("dataset_index.json", "dataset_index_sha256")):
        assert sha((root / filename).read_bytes()) == prep[field]
    refs = json.loads((root / "reference_requests.json").read_text())
    index = json.loads((root / "dataset_index.json").read_text())
    batches, assigned = schedule(config, refs, worker)
    done = {}
    for part in sorted((work / "parts").glob("*.json")):
        for row in json.loads(part.read_text()):
            key = row["request_key"]
            assert key in assigned and key not in done
            assert row["config_sha256"] == config_hash
            assert row["prompt_payload_hashes"] == assigned[key]["prompt_payload_hashes"]
            assert not row["evaluation"]["truncated"] or not row["evaluation"]["exact_count"]
            done[key] = row
    old_times = {r["batch_id"]: r["batch_wall_time_seconds"] for r in done.values()}
    generation_seconds = sum(old_times.values())
    manifest = dict(status="loading", worker=worker, config_sha256=config_hash, started_at_utc=utc(),
                    expected_requests=len(assigned), resumed_requests=len(done), **summary(done.values()))
    save(work / "run_manifest.json", manifest)
    assert config["gpu_binding"] == "scheduler"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert len(visible.split(",")) == config["engine"]["tensor_parallel_size"], visible
    assert os.environ.get("SLURM_JOB_ID"), "GPU inference requires a Slurm allocation"
    sys.path.insert(0, str(root / "source_snapshot"))
    from transformers import AutoConfig, AutoTokenizer
    from realistic_niah.prompts import build_messages, render_generation_prompt, reasoning_expected
    from realistic_niah.parsing import evaluate_generation
    from realistic_niah.runner import EngineConfig, decoding_config, load_vllm_runtime, _sampling_params_kwargs
    from realistic_niah_v3_1.spec import MODEL_SPECS
    spec = MODEL_SPECS[config["model_label"]]
    cache = config["model_cache_dir"]
    source = AutoConfig.from_pretrained(config["model_id"], revision=config["model_revision"], cache_dir=cache, local_files_only=True)
    check_rope(source.to_dict())
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], cache_dir=cache, local_files_only=True)
    packages = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "huggingface-hub")}
    manifest.update(packages=packages, python=sys.version, hardware=subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], text=True).splitlines(),
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], source_model_config=source.to_dict(),
        full_run_script_sha256=sha(Path(__file__).read_bytes()))
    save(work / "run_manifest.json", manifest)
    loading = time.perf_counter()
    runtime = load_vllm_runtime(model_spec=spec, revision=config["model_revision"],
        engine_config=EngineConfig(**config["engine"]), cache_dir=cache,
        additional_engine_overrides=config["additional_engine_overrides"])
    engine = runtime.llm.llm_engine
    effective = engine.vllm_config.model_config.hf_config.to_dict()
    check_rope(effective)
    assert engine.vllm_config.model_config.max_model_len == config["engine"]["max_model_len"]
    assert effective["max_position_embeddings"] >= config["engine"]["max_model_len"]
    for key in ("rope_parameters", "rope_scaling", "rope_theta"):
        assert effective.get(key) == source.to_dict().get(key), key
    manifest.update(status="running", model_load_seconds=time.perf_counter()-loading,
                    effective_model_config=effective, effective_engine_overrides=runtime.engine_overrides,
                    original_rope_frequencies_preserved=True, yarn_enabled=False)
    save(work / "run_manifest.json", manifest)
    print(f"MODEL_READY worker={worker} yarn=False batch=2 requests={len(assigned)}", flush=True)
    for phase, planned in batches:
        batch = [r for r in planned if r["request_key"] not in done]
        if not batch:
            continue
        mode = batch[0]["prompt_mode"]
        assert all(r["prompt_mode"] == mode for r in batch)
        decode = decoding_config(spec, mode)
        assert asdict(decode) == config["decoding"][mode]
        prepared, parameters = [], []
        for ref in batch:
            stimulus = read_stimulus(index, ref)
            messages = build_messages(stimulus["passage"], prompt_mode=mode)
            rendered = render_generation_prompt(tokenizer, messages, model_spec=spec, prompt_mode=mode)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            hashes = dict(messages_sha256=json_sha(messages), rendered_prompt_sha256=sha(rendered.encode()), input_ids_sha256=json_sha(ids))
            assert hashes == ref["prompt_payload_hashes"], ref["request_key"]
            assert len(ids) == ref["model_input_tokens"]
            assert len(ids) + decode.max_tokens <= config["engine"]["max_model_len"]
            assert asdict(decode) | {"seed": ref["seed"]} == ref["decoding"]
            prepared.append((ref, stimulus, ids, hashes))
            parameters.append(runtime.sampling_params_class(**_sampling_params_kwargs(decode, seed=ref["seed"])))
        batch_id = json_sha([r["request_key"] for r in batch])
        print(json.dumps(dict(event="BATCH_START", phase=phase, keys=[r["request_key"] for r in batch])), flush=True)
        t = time.perf_counter()
        outputs = runtime.llm.generate([{"prompt_token_ids": ids} for _, _, ids, _ in prepared], parameters, use_tqdm=False)
        elapsed = time.perf_counter() - t
        generation_seconds += elapsed
        assert len(outputs) == len(prepared)
        rows = []
        for (ref, stimulus, ids, hashes), generated in zip(prepared, outputs):
            assert list(generated.prompt_token_ids) == ids
            assert len(generated.outputs) == 1
            candidate = generated.outputs[0]
            assert candidate.finish_reason in ("stop", "length"), candidate.finish_reason
            text_readback = verify_decoded_text(tokenizer, candidate)
            evaluation = evaluate_generation(candidate.text, prompt_mode=mode, reasoning_expected=reasoning_expected(spec, mode),
                gold_pairs=stimulus["gold_pairs"], finish_reason=candidate.finish_reason,
                output_tokens=len(candidate.token_ids), max_output_tokens=decode.max_tokens)
            evaluation = score_with_truncation_policy(evaluation)
            row = dict(schema_version="qwen_yarn_off_gemma_grid_v1", request_key=ref["request_key"],
                config_sha256=config_hash, model_label=config["model_label"], model_id=config["model_id"], model_revision=config["model_revision"],
                stimulus_id=ref["stimulus_id"], target_passage_tokens=ref["target_passage_tokens"], num_needles=ref["num_needles"],
                seed=ref["seed"], prompt_mode=mode, gold_count=stimulus["gold_count"], gold_pairs=stimulus["gold_pairs"],
                passage_sha256=ref["passage_sha256"], prompt_payload_hashes=hashes, model_input_tokens=len(ids),
                decoding=asdict(decode) | {"seed": ref["seed"]}, raw_output_text=candidate.text,
                output_token_ids=list(candidate.token_ids), output_tokens=len(candidate.token_ids),
                finish_reason=candidate.finish_reason, stop_reason=candidate.stop_reason, evaluation=evaluation,
                batch_id=batch_id, batch_size=len(batch), batch_wall_time_seconds=elapsed, phase=phase,
                worker=worker, completed_at_utc=utc(), token_text_readback=text_readback)
            assert row["request_key"] not in done
            rows.append(row)
        save(work / "parts" / f"{batch_id}.json", rows)
        for row in rows:
            done[row["request_key"]] = row
        manifest.update(**summary(done.values()), generation_seconds=generation_seconds,
                        last_updated_at_utc=utc(), current_phase=phase, last_batch_seconds=elapsed,
                        last_batch_keys=[r["request_key"] for r in rows])
        save(work / "run_manifest.json", manifest)
        print(json.dumps(dict(event="BATCH_COMPLETE", worker=worker, completed=len(done), expected=len(assigned),
            seconds=round(elapsed,3), results=[{k:r[k] for k in ("request_key", "output_tokens", "finish_reason")} |
            {"prediction":r["evaluation"]["predicted_count"], "correct":r["evaluation"]["exact_count"]} for r in rows])), flush=True)
    assert set(done) == set(assigned)
    path = work / "requests.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for key in sorted(done):
            f.write(json.dumps(done[key], ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    manifest.update(status="complete", completed_at_utc=utc(), elapsed_process_seconds=time.perf_counter()-started,
                    requests_sha256=sha(path.read_bytes()))
    save(work / "run_manifest.json", manifest)
    print(f"WORKER_COMPLETE worker={worker} requests={len(done)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Directory for preparation and worker outputs.")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/qwen_yarn_off.json")
    p.add_argument("--backup-root", type=Path, help="Archive containing the short and long benchmark runs; preparation only.")
    p.add_argument("--cache-dir", type=Path, help="Local model cache; preparation only.")
    p.add_argument("--source-code-root", type=Path, default=Path(__file__).resolve().parents[2] / "src")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", type=int, choices=(0, 1, 2, 3))
    args = p.parse_args()
    if args.prepare:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config["backup_root"] = str((args.backup_root or Path(config["backup_root"])).resolve())
        config["model_cache_dir"] = str((args.cache_dir or Path(config["model_cache_dir"])).resolve())
        config["source_code_root"] = str(args.source_code_root.resolve())
        saved_config = args.root / "config.json"
        if saved_config.exists() and json.loads(saved_config.read_text()) != config:
            raise ValueError("The run already contains a different configuration; choose a new --root.")
        save(saved_config, config)
    else:
        if args.backup_root is not None or args.cache_dir is not None:
            p.error("Input/cache overrides are preparation options; workers reuse the saved configuration.")
        config = json.loads((args.root / "config.json").read_text(encoding="utf-8"))
    try:
        if args.prepare:
            prepare(args.root, config)
        else:
            assert args.worker is not None
            run_worker(args.root, config, args.worker)
    except Exception as exc:
        failure = args.root / (f"worker-{args.worker}" if args.worker is not None else "preparation_failure")
        save(failure / "failure.json", dict(time_utc=utc(), error_type=type(exc).__name__, message=str(exc)))
        manifest_path = failure / "run_manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            m.update(status="failed", last_updated_at_utc=utc(), error_type=type(exc).__name__, error=str(exc))
            save(manifest_path, m)
        raise


if __name__ == "__main__":
    main()
