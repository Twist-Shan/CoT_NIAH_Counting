#!/usr/bin/env python3
"""Run one frozen CLI with observational backend instrumentation.

The legacy comparison substitutes only the archived context manager, before
dependent modules import it. It deliberately preserves backend state between
trials. Commands and inputs are supplied by a separately frozen job manifest.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib.metadata
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend_state(model):
    root = model.config
    text = root.get_text_config() if hasattr(root, "get_text_config") else root
    return {"root": getattr(root, "_attn_implementation", None),
            "text": getattr(text, "_attn_implementation", None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    folder = args.job.parent
    import torch
    import realistic_niah_v4.modeling as modeling
    if not torch.cuda.is_available():
        raise RuntimeError("GPU replay requires working CUDA")
    for path, expected in job["input_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    for rel, expected in job["code_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"Frozen code changed: {rel}")
    legacy_source = None
    if job["backend_variant"] == "legacy":
        archived = Path(job["legacy_modeling"])
        source = archived.read_text()
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_temporary_attention_backend")
        legacy_source = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
        namespace = dict(vars(modeling))
        exec(compile("from __future__ import annotations\n" + legacy_source,
                     str(archived), "exec"), namespace)
        modeling._temporary_attention_backend = contextlib.contextmanager(namespace[node.name])
    elif job["backend_variant"] != "fixed":
        raise ValueError("Unknown backend variant")
    events = (folder / "backend_events.jsonl").open("x", encoding="utf-8")
    counters = {}

    def emit(value):
        events.write(json.dumps(value, sort_keys=True) + "\n")
        events.flush()

    original_backend_context = modeling._temporary_attention_backend

    @contextlib.contextmanager
    def observed_backend_context(model, backend):
        before = backend_state(model)
        started = time.monotonic()
        try:
            with original_backend_context(model, backend):
                yield
        finally:
            after = backend_state(model)
            name = "temporary_attention_backend"
            counters[name] = counters.get(name, 0) + 1
            emit({"event": name, "index": counters[name], "requested": backend,
                  "before": before, "after": after, "seconds": time.monotonic() - started})
            if job["backend_variant"] == "fixed" and before != after:
                raise RuntimeError(f"Fixed helper did not restore backend: {before} -> {after}")

    modeling._temporary_attention_backend = observed_backend_context

    def instrument(name, owner=modeling):
        original = getattr(owner, name)

        @functools.wraps(original)
        def observed(model, *a, **kw):
            before = backend_state(model)
            started = time.monotonic()
            result = original(model, *a, **kw)
            after = backend_state(model)
            counters[name] = counters.get(name, 0) + 1
            record = {"event": name, "index": counters[name], "before": before,
                      "after": after, "seconds": time.monotonic() - started}
            if name.startswith("generate_answer_completion"):
                record["generation"] = result
            emit(record)
            if job["backend_variant"] == "fixed" and (before != after or after["text"] != "sdpa"):
                raise RuntimeError(f"Fixed backend state changed during {name}: {before} -> {after}")
            return result

        setattr(owner, name, observed)

    for name in ["position_attention_outputs", "generate_answer_completion"]:
        instrument(name)
    restoration = importlib.import_module("realistic_niah_v4_4_5.restoration")
    instrument("generate_answer_completion_from_prefill", restoration)
    entry = ROOT / "scripts" / job["entry"]
    # V6 dispatches into several V5 entry points. Instrument their shared loader
    # before importing the entry so direct imports cannot retain an unaudited copy.
    loader_module_name = job.get("loader_module")
    if loader_module_name:
        loader_module = importlib.import_module(loader_module_name)
    else:
        spec = importlib.util.spec_from_file_location("cot_backend_entry", entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loader_module = module
    loader_name = job["loader_name"]
    original_load = getattr(loader_module, loader_name)

    def audited_load(*arguments, **kwargs):
        started = time.monotonic()
        model, tokenizer, adapter = original_load(*arguments, **kwargs)
        config = model.config
        audit = {"event": "model_loaded", "backend": backend_state(model),
                 "revision": getattr(config, "_commit_hash", None),
                 "model_class": type(model).__name__, "layers": adapter.num_layers,
                 "dtype": str(next(model.parameters()).dtype), "training": model.training,
                 "seconds": time.monotonic() - started,
                 "model_source_sha256": sha(sys.modules[type(model).__module__].__file__)}
        emit(audit)
        if model.training or audit["dtype"] != "torch.bfloat16" or audit["backend"]["text"] != "sdpa":
            raise RuntimeError(f"Unexpected loaded model: {audit}")
        return model, tokenizer, adapter

    setattr(loader_module, loader_name, audited_load)
    if loader_module_name:
        spec = importlib.util.spec_from_file_location("cot_backend_entry", entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    runtime = {"python": sys.version, "cuda": torch.version.cuda,
               "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "accelerate", "numpy", "pandas"]},
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "legacy_function": legacy_source, "job_sha256": sha(args.job)}
    (folder / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    sys.argv = [str(entry), *job["args"]]
    started = time.monotonic()
    status = "FAILED"
    try:
        module.main()
        status = "COMPLETE"
    finally:
        events.close()
        (folder / "process_status.json").write_text(json.dumps(
            {"status": status, "seconds": time.monotonic() - started, "calls": counters}, indent=2) + "\n")


if __name__ == "__main__":
    main()
