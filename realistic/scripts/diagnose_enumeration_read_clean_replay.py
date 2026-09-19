"""Diagnose original-generation versus answer-query replay without changing cohorts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(stage):
    manifest = read(stage / "manifest.json")
    if sha(__file__) != manifest["script_sha256"]:
        raise ValueError("Diagnostic script differs from frozen manifest")
    for filename, expected in manifest["input_sha256"].items():
        if sha(filename) != expected:
            raise ValueError(f"Diagnostic input hash mismatch: {filename}")
    return manifest


def process_command(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()


def worker(stage):
    manifest = verify(stage)
    root = Path(manifest["root"])
    causal = root / "fresh_causal_v1"
    sys.path[:0] = [str(causal / "code/src"), str(causal / "code")]
    import torch
    from scripts.enumeration_fresh_geometry import build_read_geometry
    from realistic_niah_v4 import modeling
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5 import causal as kernel
    from realistic_niah.parsing import parse_total
    helper_path = root / "fresh_answer_patch_gpu_v1/code/scripts/run_enumeration_fresh_answer_patch.py"
    spec = importlib.util.spec_from_file_location("_frozen_answer_patch_observer", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = root / "fresh_v1/baseline/Gemma4-E4B/enumeration_bullet/generations.jsonl"
    rows = {(r["seed"], r["gold_count"]): r for r in map(json.loads, source.read_text().splitlines())}
    tick = time.monotonic()
    model, tokenizer, adapter = modeling.load_registered_model(resolve_model_spec("Gemma4-E4B"),
        cache_dir=Path(manifest["cache_dir"]), device_map="auto", torch_dtype="bfloat16", attention_backend="sdpa")
    model.eval()
    results = []
    with torch.inference_mode(), (stage / "trials.jsonl").open("x", encoding="utf-8") as output:
        for case in manifest["cases"]:
            start = time.monotonic()
            row = rows[case["seed"], case["gold_count"]]
            encoding, _, geometry = build_read_geometry(row, tokenizer, mode="enumeration_bullet")
            if geometry["selected_site"]["alignment_strategy"] != "literal_baseline_token_prefix":
                raise ValueError("Diagnostic requires exact original token prefix")
            torch.manual_seed(row["sampling_seed"])
            torch.cuda.manual_seed_all(row["sampling_seed"])
            device = model.get_input_embeddings().weight.device
            ids = torch.tensor([row["input_ids"]], device=device)
            mask = torch.tensor([row["attention_mask"]], device=device)
            generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                       max_new_tokens=4096, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            autoregressive = generated[0, ids.shape[1]:].cpu().tolist()
            autoregressive_text = tokenizer.decode(autoregressive, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
            del generated
            clean32_a = modeling.generate_answer_completion(model, tokenizer, encoding, max_new_tokens=32)
            clean32_b = modeling.generate_answer_completion(model, tokenizer, encoding, max_new_tokens=32)
            clean16 = modeling.generate_answer_completion(model, tokenizer, encoding, max_new_tokens=16)
            layers = [0, adapter.num_layers - 1]
            logits, states = modeling.capture_post_block_states(model, adapter, encoding,
                                                                [encoding.query_position], layers=layers)
            values, indices = torch.topk(logits.reshape(-1).float(), 5)
            no_cache_logits = [{"token_id": i, "text": tokenizer.decode([i]), "logit": v}
                               for i, v in zip(indices.cpu().tolist(), values.cpu().tolist())]
            self_panels = []
            for layer in layers:
                with helper.retain_patch_audits(kernel, modeling) as audits:
                    raw = kernel.generate_with_residual_interventions(model, tokenizer, adapter, encoding,
                        {layer: ([encoding.query_position], states[layer][0])}, max_new_tokens=16)
                if len(audits) != 1:
                    raise ValueError("Missing self-patch diagnostic audit")
                self_panels.append({"layer": layer, "raw": raw, "audit": audits[0],
                    "matches_clean_tokens": raw["generated_token_ids"] == clean16["generated_token_ids"],
                    "prediction": parse_total(raw["full_answer_text"])})
            result = {"case": case, "geometry": geometry,
                "autoregressive_token_ids": autoregressive, "autoregressive_text": autoregressive_text,
                "autoregressive_matches_saved_tokens": autoregressive == row["output_token_ids"],
                "autoregressive_prediction": parse_total(autoregressive_text),
                "clean32_a": clean32_a, "clean32_b": clean32_b, "clean16": clean16,
                "replay32_repeat_matches": clean32_a["generated_token_ids"] == clean32_b["generated_token_ids"],
                "replay_prediction": parse_total(clean16["full_answer_text"]),
                "no_cache_top5": no_cache_logits, "self_panels": self_panels, "seconds": time.monotonic() - start}
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            print(json.dumps({"case": case, "autoregressive_matches_saved_tokens": result["autoregressive_matches_saved_tokens"],
                              "replay_prediction": result["replay_prediction"], "seconds": result["seconds"]}), flush=True)
    gold_regenerated = all(r["replay_prediction"] == r["case"]["gold_count"] and
                          all(p["prediction"] == r["case"]["gold_count"] and p["matches_clean_tokens"] and
                              p["audit"]["realized_delta_norm"] == 0 for p in r["self_panels"]) for r in results)
    summary = {"status": "COMPLETE", "utc": datetime.now(timezone.utc).isoformat(),
               "cases": len(results), "all_sampled_native_regeneration_checks_pass": gold_regenerated,
               "all_autoregressive_tokens_reproduced": all(r["autoregressive_matches_saved_tokens"] for r in results),
               "all_answer_replays_stable": all(r["replay32_repeat_matches"] for r in results),
               "trials_sha256": sha(stage / "trials.jsonl"), "seconds": time.monotonic() - tick,
               "scope": "Diagnostic only; original Read outcomes and answer-patch pairs are unchanged.",
               "torch": torch.__version__, "model_source_sha256": sha(sys.modules[type(model).__module__].__file__)}
    write(stage / "summary.json", summary)


def driver(stage):
    manifest = verify(stage)
    tick = time.monotonic()
    status_path = stage / "status.json"
    state = {"status": "RUNNING", "phase": "WAIT_PRIMARY_UPDATE_COMPLETE", "pid": os.getpid(), "command": sys.argv}
    write(status_path, state)
    relay_pid = manifest["paused_relay_pid"]
    answer_pid = manifest["paused_answer_pid"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    try:
        while True:
            previous = read(Path(manifest["root"]) / "fresh_native_update_v1/gpu_pipeline_status.json")
            if previous["status"] == "FAILED":
                raise RuntimeError("Primary Update failed; diagnostic did not run")
            if previous["status"] == "COMPLETE":
                if previous["phase"] != "ENUMERATION_UPDATE_PRIMARY_COMPLETE":
                    raise RuntimeError("Wrong Update completion phase")
                break
            if time.monotonic() - tick > 36 * 3600:
                raise TimeoutError("Primary Update did not finish")
            state["seconds"] = time.monotonic() - tick
            write(status_path, state)
            time.sleep(10)
        active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                text=True, capture_output=True, check=True).stdout.strip()
        if active:
            raise RuntimeError(f"GPU occupied after Update; no overlapping diagnostic launched: {active}")
        state["phase"] = "DIAGNOSE_SIX_FIXED_INPUTS"
        with (stage / "worker.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, __file__, "--stage", str(stage), "--worker"],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = child.pid
            write(status_path, state)
            try:
                code = child.wait(timeout=3600)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        if code:
            raise RuntimeError(f"Diagnostic worker exited {code}; inspect worker.log")
        summary = read(stage / "summary.json")
        # Releasing answer-state requires the original Native success contract, not a relaxed guard.
        if summary["all_sampled_native_regeneration_checks_pass"]:
            if "fresh_answer_patch_gpu_v1" not in process_command(answer_pid):
                raise RuntimeError("Answer queue PID changed")
            os.kill(answer_pid, signal.SIGCONT)
            state["answer_queue_resumed"] = True
        else:
            state["answer_queue_resumed"] = False
        state.update(status="COMPLETE", phase="DIAGNOSTIC_COMPLETE" if state["answer_queue_resumed"]
                     else "DIAGNOSTIC_COMPLETE_ANSWER_PATCH_REVIEW_REQUIRED")
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), answer_queue_resumed=False)
        raise
    finally:
        # A failed prerequisite/diagnostic never strands independent Relay work.
        if "fresh_relay_native_gpu_v1" in process_command(relay_pid):
            os.kill(relay_pid, signal.SIGCONT)
            state["relay_queue_resumed"] = True
        state["seconds"] = time.monotonic() - tick
        write(status_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    (worker if args.worker else driver)(args.stage)
