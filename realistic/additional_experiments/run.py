"""GPU capture and paired causal interventions, isolated from V4/V5 outputs."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch

from diagnostics import native_endpoint, norm_matched_orthogonal, strip_end_tokens, trace_record_sites
from protocol import (audit_case, encode_ids, matched_random_heads, parse_answer,
                      read_jsonl, sha256, text_hash, token_end, user_prompt, write_json)
from realistic_niah.parsing import split_reasoning_and_final
from realistic_niah_v4.modeling import (
    capture_post_block_states, generate_answer_completion, generate_with_head_ablation,
    generate_with_residual_interventions, load_registered_model, query_attention_rows,
)
from realistic_niah_v4.spec import resolve_model_spec


def verify_frozen(frozen: Path) -> dict:
    audit = json.loads((frozen / "freeze_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise ValueError("Freeze audit not PASS")
    for rel, expected in audit["files"].items():
        if sha256(frozen / rel) != expected:
            raise ValueError(f"Changed frozen input: {rel}")
    for row in audit["legacy_native_bases"]:
        if sha256(frozen / row["path"].replace("\\", "/")) != row["sha256"]:
            raise ValueError("Changed legacy basis")
    dependencies = json.loads((frozen / "legacy_source_hashes.json").read_text(encoding="utf-8"))
    for rel, expected in dependencies.items():
        if sha256(REPO / rel.replace("\\", "/")) != expected:
            raise ValueError(f"Source drift from freeze: {rel}")
    return json.loads((frozen / "config.json").read_text(encoding="utf-8"))


def render(case: dict, mode: str, tokenizer):
    user = user_prompt(case, mode)
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": mode == "native_thinking"}
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": user}], **kwargs)
    prefill = case["answer_prefix"] if mode == "nonthinking" else ""
    rendered += prefill
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if rendered.count(case["passage"]) != 1:
        raise ValueError("Rendered passage not unique")
    return encode_ids(encoded["input_ids"]), encoded["offset_mapping"], {
        "user_text": user, "rendered_prompt": rendered,
        "user_text_sha256": text_hash(user), "rendered_prompt_sha256": text_hash(rendered),
        "chat_template_kwargs": kwargs, "assistant_prefill_text": prefill,
        "input_ids": encoded["input_ids"], "attention_mask": [1] * len(encoded["input_ids"]),
    }


def summarize_generation(result: dict, case: dict, *, mode: str, prefixed: bool = False) -> dict:
    raw = result["completion_text_raw"]
    if prefixed:
        final, reasoning = case["answer_prefix"] + strip_end_tokens(raw), ""
    else:
        reasoning, final = split_reasoning_and_final(raw, prompt_mode="native_thinking", reasoning_expected=True)
        final = strip_end_tokens(final)
    parsed = parse_answer(final, case)
    ordinal = None
    if case["task"] == "kth_needle" and parsed["prediction"]:
        ordinal = next((r["ordinal"] for r in case["records"]
                        if parsed["prediction"] == f"{r['city']}|{r['score']}"), None)
    numeric = ordinal if case["task"] == "kth_needle" else (
        int(parsed["prediction"]) if parsed["prediction"] is not None else None)
    return {**parsed, "final_text": final, "reasoning_text": reasoning,
            "predicted_ordinal_or_count": numeric,
            "absolute_error": abs(numeric - case["level"]) if numeric is not None else None,
            "generation_truncated": result["generation_truncated"],
            "generated_token_count": result["generated_token_count"]}


def capture_one(model, tokenizer, adapter, case, mode, config, output: Path, *, resume=False):
    target = output / "captures" / mode / case["case_id"]
    if (target / "complete.json").exists():
        if not resume:
            raise FileExistsError(target)
        saved = json.loads((target / "complete.json").read_text(encoding="utf-8"))
        for name, expected in saved["files"].items():
            if sha256(target / name) != expected:
                raise ValueError("Resume capture hash mismatch")
        return
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    audit_case(case)
    enc, offsets, prompt = render(case, mode, tokenizer)
    write_json(target / "prompt.json", prompt)
    max_tokens = config["max_new_tokens_native"] if mode == "native_thinking" else config["max_new_tokens_answer"]
    raw = generate_answer_completion(model, tokenizer, enc, max_new_tokens=max_tokens)
    behavior = summarize_generation(raw, case, mode=mode, prefixed=mode == "nonthinking")
    write_json(target / "generation.json", {**raw, **behavior, "case_id": case["case_id"], "mode": mode})
    endpoint, endpoint_reason = enc, None
    trace_sites, trace_reason = [], "nonthinking_has_no_native_trace"
    fulltext = prompt["rendered_prompt"]
    if mode == "native_thinking":
        result, endpoint_reason = native_endpoint(tokenizer, fulltext, raw, case["answer_prefix"])
        if result is None:
            endpoint = None
        else:
            endpoint, offsets, combined, close_end = result
            trace_sites, trace_reason = trace_record_sites(case, raw["completion_text_raw"], len(fulltext), close_end, offsets)
            fulltext = combined
    # Prompt sites are available even when native reasoning/answer parsing fails.
    capture_encoding = endpoint if endpoint is not None else enc
    pstart = prompt["rendered_prompt"].index(case["passage"])
    sites, record_spans, total, filtered = [], [], 0, 0
    for record in case["records"]:
        total += 1
        filtered += int(record["is_target"])
        a, b = pstart + record["char_start"], pstart + record["char_end"]
        pos = token_end(offsets, a, b)
        indices = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
        record_spans.append((indices[0], indices[-1] + 1))
        sites.append({"kind": "prompt_record_end", "position": pos, "ordinal": total,
                      "total_prefix": total, "target_prefix": filtered, "is_target": record["is_target"]})
    for i, item in enumerate(trace_sites, 1):
        sites.append({**item, "kind": "native_item_end", "total_prefix": i,
                      "target_prefix": sum(int(s["is_target"]) for s in trace_sites[:i])})
    if endpoint is not None:
        sites.append({"kind": "answer_query", "position": endpoint.query_position,
                      "ordinal": None, "total_prefix": case["total_records"],
                      "target_prefix": case["level"]})
    layers = config["capture_layers"][config["current_model"]]
    _, states = capture_post_block_states(model, adapter, capture_encoding,
                                  [s["position"] for s in sites], layers=layers)
    arrays = {f"layer_{l}": states[l].numpy() for l in layers}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite hidden state")
    np.savez_compressed(target / "states.npz", **arrays)
    attention = []
    if endpoint is not None:
        rows, starts = query_attention_rows(model, adapter, endpoint)
        for layer, head in config["current_bank"]:
            alpha, key_start = rows[layer][head], starts[layer]
            masses = []
            for a, b in record_spans:
                lo, hi = max(0, a - key_start), min(len(alpha), b - key_start)
                masses.append(float(alpha[lo:hi].sum()) if hi > lo else 0.0)
            if case["task"] == "kth_needle":
                selected = [i == case["level"] - 1 for i in range(len(masses))]
            else:
                selected = [r["is_target"] for r in case["records"]]
            attention.append({"layer": layer, "head": head, "record_masses": masses,
                              "target_mass": sum(m for m, yes in zip(masses, selected) if yes),
                              "other_record_mass": sum(m for m, yes in zip(masses, selected) if not yes)})
    write_json(target / "sites.json", {
        "sites": sites, "record_spans": record_spans, "prompt_token_count": enc.sequence_length,
        "endpoint_input_ids": list(endpoint.input_ids) if endpoint is not None else None,
        "endpoint_unavailable_reason": endpoint_reason, "native_site_unavailable_reason": trace_reason,
        "native_site_contract": "Unique city+score lines, monotone passage order; line-end token; grammar not forced",
        "native_record_ordinals": [s["ordinal"] for s in trace_sites],
        "all_native_sites_explicitly_indexed": bool(trace_sites) and all(s["explicit_index"] for s in trace_sites),
        "attention": attention,
    })
    names = ("prompt.json", "generation.json", "sites.json", "states.npz")
    write_json(target / "complete.json", {"status": "PASS", "files": {n: sha256(target / n) for n in names},
                                         "elapsed_seconds": time.perf_counter() - started})


def read_capture(output, mode, case):
    folder = output / "captures" / mode / case["case_id"]
    meta = json.loads((folder / "sites.json").read_text(encoding="utf-8"))
    return folder, meta


def causal_one(model, tokenizer, adapter, case, donor, mode, config, output, *, resume=False):
    target = output / "causal" / mode / (case["case_id"] + ".json")
    if target.exists():
        if not resume:
            raise FileExistsError(target)
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing["donor_case_id"] != donor["case_id"]:
            raise ValueError("Changed donor on resume")
        return
    started = time.perf_counter()
    folder, meta = read_capture(output, mode, case)
    d_folder, d_meta = read_capture(output, mode, donor)
    donor_behavior = json.loads((d_folder / "generation.json").read_text(encoding="utf-8"))
    base = {"case_id": case["case_id"], "donor_case_id": donor["case_id"], "mode": mode,
            "seed": case["seed"], "task": case["task"], "level": case["level"], "gold": case["gold"],
            "donor_gold": donor["gold"], "donor_model_prediction": donor_behavior["prediction"],
            "endpoint": "final_answer_query", "arms": []}
    if meta["endpoint_input_ids"] is None or d_meta["endpoint_input_ids"] is None:
        write_json(target, {**base, "status": "UNAVAILABLE", "reason": "recipient_or_donor_native_endpoint_missing"})
        return
    enc = encode_ids(meta["endpoint_input_ids"])
    budget = config["max_new_tokens_answer"]
    clean = generate_answer_completion(model, tokenizer, enc, max_new_tokens=budget)
    clean_summary = summarize_generation(clean, case, mode=mode, prefixed=True)
    original = json.loads((folder / "generation.json").read_text(encoding="utf-8"))
    base["endpoint_replay_prediction_equal"] = clean_summary["prediction"] == original["prediction"]
    base["native_reasoning_held_fixed"] = mode == "native_thinking"
    base["native_reasoning_may_already_contain_answer"] = mode == "native_thinking"
    if not base["endpoint_replay_prediction_equal"]:
        write_json(target, {**base, "status": "UNAVAILABLE", "reason": "natural_endpoint_replay_mismatch",
                            "clean_replay": clean_summary, "original_prediction": original["prediction"]})
        return

    def add_arm(name, result, extra=None):
        audit = result.get("intervention_hook_applications", {})
        if any(n != 1 for n in audit.values()):
            raise ValueError("One-shot intervention applied an unexpected number of times")
        base["arms"].append({"arm": name, **summarize_generation(result, case, mode=mode, prefixed=True),
                             "generated_token_ids": result["generated_token_ids"],
                             "completion_text_raw": result["completion_text_raw"],
                             "hook_applications": audit, **(extra or {})})

    add_arm("clean", clean)
    bank = config["current_bank"]
    for name, heads in [("legacy_broad_bank", bank)] + [
        (f"random_bank_{i}", matched_random_heads(bank, adapter.num_heads, config["random_seed"] + i))
        for i in range(config["random_head_replicates"])
    ]:
        generated = generate_with_head_ablation(model, tokenizer, adapter, enc, heads,
                                                scope="answer_query", max_new_tokens=budget)
        add_arm(name, generated, {"heads": heads})
    with np.load(folder / "states.npz") as z, np.load(d_folder / "states.npz") as dz:
        for layer in config["patch_layers"][config["current_model"]]:
            recipient, donor_state = z[f"layer_{layer}"][-1], dz[f"layer_{layer}"][-1]
            ctrl, norm_audit = norm_matched_orthogonal(recipient, donor_state, config["random_seed"] + layer + case["seed"])
            dtype = next(model.parameters()).dtype
            r = torch.from_numpy(recipient.copy()).to(dtype).float().numpy()
            d = torch.from_numpy(donor_state.copy()).to(dtype).float().numpy()
            c = torch.from_numpy(ctrl.copy()).to(dtype).float().numpy()
            realized_delta, realized_control = d - r, c - r
            norm_audit.update({"realized_norm_ratio": float(np.linalg.norm(realized_control) / np.linalg.norm(realized_delta)),
                               "realized_cosine": float(np.dot(realized_control, realized_delta) /
                                   (np.linalg.norm(realized_control) * np.linalg.norm(realized_delta)))})
            if abs(norm_audit["realized_norm_ratio"] - 1) > .02 or abs(norm_audit["realized_cosine"]) > .02:
                raise ValueError(f"Realized dtype-matched control failed: {norm_audit}")
            for name, replacement in (("self_patch", recipient), ("donor_patch", donor_state), ("orthogonal_patch", ctrl)):
                result = generate_with_residual_interventions(
                    model, tokenizer, adapter, enc,
                    {layer: ([enc.query_position], torch.from_numpy(replacement.copy()))}, max_new_tokens=budget)
                if name == "self_patch" and result["generated_token_ids"] != clean["generated_token_ids"]:
                    raise ValueError("Self-patch changed the clean continuation")
                add_arm(f"{name}_L{layer}", result, {"layer": layer, "norm_audit": norm_audit})
    write_json(target, {**base, "status": "PASS", "elapsed_seconds": time.perf_counter() - started})


def execute(args):
    started = time.perf_counter()
    config = verify_frozen(args.frozen)
    config["current_model"] = args.model
    banks = json.loads((args.frozen / "frozen_banks.json").read_text(encoding="utf-8"))
    config["current_bank"] = banks["banks"][args.model]
    cases = read_jsonl(args.frozen / "cases.jsonl")
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume only with unchanged inputs/code")
    output.mkdir(parents=True, exist_ok=True)
    spec = resolve_model_spec(args.model)
    invariant = {
        "config": config, "frozen_audit_sha256": sha256(args.frozen / "freeze_audit.json"),
        "code_hashes": {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
        "model_id": spec.model_id, "model_revision": spec.revision,
        "cache_dir": str(args.cache_dir.resolve()), "device_map": args.device_map,
        "tokenizer_padding_side": "left", "batch_size": 1, "do_sample": False,
    }
    manifest = output / "run_contract.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != invariant:
        raise ValueError("Changed code/config/environment inputs on resume")
    write_json(manifest, invariant)
    versions = {n: importlib.metadata.version(n) for n in ("torch", "numpy", "transformers", "accelerate")}
    write_json(output / "environment.json", {"argv": sys.argv, "python": sys.version, "platform": platform.platform(),
               "versions": versions, "cuda_available": torch.cuda.is_available(),
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "determinism": "Greedy decoding with fixed RNG; CUDA kernels may remain nondeterministic."})
    if not torch.cuda.is_available() and args.device_map != "cpu":
        raise RuntimeError("GPU unavailable; use --device-map cpu only for an explicitly intended CPU run")
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    model, tokenizer, adapter = load_registered_model(spec, cache_dir=args.cache_dir, device_map=args.device_map,
                                                     torch_dtype=config["torch_dtype"], attention_backend=config["attention_backend"])
    tokenizer.padding_side = "left"
    write_json(output / "model_load.json", {"elapsed_seconds": time.perf_counter() - started,
               "num_layers": adapter.num_layers, "num_heads": adapter.num_heads,
               "chat_template_sha256": text_hash(str(tokenizer.chat_template))})
    if args.phase in ("all", "capture"):
        for mode in config["modes"]:
            for i, case in enumerate(cases, 1):
                capture_one(model, tokenizer, adapter, case, mode, config, output, resume=args.resume)
                print(f"capture {args.model} {mode} {i}/{len(cases)} {case['case_id']}", flush=True)
    if args.phase in ("all", "causal"):
        levels = config["levels"]
        lookup = {(r["seed"], r["task"], r["level"]): r for r in cases}
        for mode in config["modes"]:
            for case in cases:
                if case["split"] != "confirmation":
                    continue
                idx = levels.index(case["level"])
                donor_level = levels[idx + 1] if idx < len(levels) - 1 else levels[idx - 1]
                donor = lookup[case["seed"], case["task"], donor_level]
                causal_one(model, tokenizer, adapter, case, donor, mode, config, output, resume=args.resume)
                print(f"causal {args.model} {mode} {case['case_id']}", flush=True)
    complete = list((output / "captures").glob("*/*/complete.json"))
    causal = list((output / "causal").glob("*/*.json"))
    expected_captures = len(cases) * len(config["modes"])
    expected_causal = sum(r["split"] == "confirmation" for r in cases) * len(config["modes"])
    if len(complete) != expected_captures or (args.phase != "capture" and len(causal) != expected_causal):
        raise RuntimeError("Completion coverage mismatch")
    write_json(output / f"{args.phase}_complete.json", {"status": "PASS", "captures": len(complete), "causal_cases": len(causal),
               "causal_unavailable": sum(json.loads(p.read_text(encoding="utf-8"))["status"] != "PASS" for p in causal),
               "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--phase", choices=("all", "capture", "causal"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        execute(args)
    except Exception:
        failure = args.output.parent / (args.output.name + "_failure_" + time.strftime("%Y%m%dT%H%M%S") + ".json")
        write_json(failure, {"status": "FAIL", "argv": sys.argv, "traceback": traceback.format_exc()})
        raise
