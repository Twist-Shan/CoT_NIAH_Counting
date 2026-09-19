"""Generate every frozen baseline input; never filter on correctness or formatting."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    p.add_argument("--mode", choices=("enumeration_index", "enumeration_bullet", "thinking"), required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    tick = time.monotonic()
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    cfg = json.loads((a.bundle / "protocol.json").read_text())
    assert sha(a.bundle / "protocol.json") == manifest["protocol_sha256"]
    assert sha(a.bundle / "stimuli.jsonl") == manifest["stimuli_sha256"]
    for filename, expected in manifest["code_sha256"].items():
        assert sha(ROOT / filename) == expected, filename
    with (a.bundle / "stimuli.jsonl").open() as f:
        stimuli = [json.loads(line) for line in f]
    if a.limit:
        stimuli = stimuli[:a.limit]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "shards").mkdir()
    state = dict(status="RUNNING", completed=0, total=len(stimuli), model=a.model, mode=a.mode)
    save(a.output / "status.json", state)
    import torch
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v5.generation import generate_native_trace, render_native_prompt
    from realistic_niah_v5.spec import DecodingSpec as NativeDecoding
    from realistic_niah_v6.generation import generate_structured_enumeration, render_structured_prompt
    from realistic_niah_v6.spec import DecodingSpec as EnumerationDecoding
    spec = resolve_model_spec(a.model)
    try:
        model, tokenizer, adapter = load_registered_model(spec, cache_dir=a.cache_dir, device_map="auto", torch_dtype="bfloat16", attention_backend="sdpa")
        model.eval()
        config = model.config
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        assert next(model.parameters()).dtype == torch.bfloat16 and not model.training
        runtime = {"python":sys.version,"torch":torch.__version__,"transformers":importlib.metadata.version("transformers"),
                   "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"model_revision":spec.revision,
                   "model_source_sha256":sha(sys.modules[type(model).__module__].__file__),
                   "dtype":"bfloat16","backend":"sdpa","layers":adapter.num_layers,
                   "manifest_sha256":sha(a.bundle / "manifest.json"),"command":sys.argv,"model_load_seconds":time.monotonic()-tick}
        save(a.output / "runtime.json", runtime)
        with (a.output / "generations.jsonl").open("x",encoding="utf-8") as output:
            for source in stimuli:
                if a.mode != "thinking":
                    prompt = render_structured_prompt(source,tokenizer=tokenizer,model_spec=spec,prompt_mode=a.mode)
                    row = generate_structured_enumeration(model,tokenizer,prompt,decoding=EnumerationDecoding(max_new_tokens=cfg["baseline_max_new_tokens"]),sampling_seed=source["seed"])
                elif a.model == "Qwen3-8B":
                    prompt = render_native_prompt(source,tokenizer=tokenizer,model_spec=spec)
                    row = generate_native_trace(model,tokenizer,prompt,decoding=NativeDecoding(max_new_tokens=cfg["baseline_max_new_tokens"]),sampling_seed=source["seed"])
                else:
                    from scripts.run_realistic_niah_v5_gemma_prompt_conditioned_noindex import prompt_from_source, audit_prompt_conditioned_noindex_row, AUDIT_KEY
                    from realistic_niah_v5.parsing import parse_trace_record
                    prompt, audit = prompt_from_source(source,tokenizer,spec)
                    row = generate_native_trace(model,tokenizer,prompt,decoding=NativeDecoding(max_new_tokens=cfg["baseline_max_new_tokens"]),sampling_seed=source["seed"])
                    row["generated_continuation_token_count"] = len(row["output_token_ids"])
                    ids = list(audit["assistant_prefix_token_ids"]) + row["output_token_ids"]
                    row.update(rendered_prompt=audit["base_rendered_prompt"],input_ids=list(audit["base_input_ids"]),
                               attention_mask=list(audit["base_attention_mask"]),prompt_token_count=len(audit["base_input_ids"]),
                               output_token_ids=ids,output_tokens=len(ids),
                               raw_output_text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                               clean_output_text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                               gemma_prompt_conditioned_noindex_prompt_audit=audit,
                               prefilled_output_token_count=len(audit["assistant_prefix_token_ids"]))
                    row["trace_parse"] = parse_trace_record(row)
                    row[AUDIT_KEY] = audit_prompt_conditioned_noindex_row(row)
                assert row["seed"] == source["seed"] and row["gold_count"] == source["gold_count"]
                assert row["input_ids"] and row["output_token_ids"]
                assert config._attn_implementation == "sdpa" and text_config._attn_implementation == "sdpa"
                row.update(alignment_mode=a.mode,alignment_roles=source["alignment_roles"],
                           alignment_protocol_sha256=manifest["protocol_sha256"],stimulus_passage_sha256=source["passage_sha256"],
                           alignment_manifest_sha256=runtime["manifest_sha256"],baseline_correctness_filter_applied=False)
                name=f"seed{source['seed']}_N{source['gold_count']}.json"
                save(a.output / "shards" / name, row)
                output.write(json.dumps(row,ensure_ascii=True)+"\n")
                output.flush()
                state.update(completed=state["completed"]+1,seed=source["seed"],gold_count=source["gold_count"],seconds=time.monotonic()-tick)
                save(a.output / "status.json",state)
                print(json.dumps(state),flush=True)
        state.update(status="COMPLETE",generations_sha256=sha(a.output / "generations.jsonl"))
    except BaseException as e:
        state.update(status="FAILED",error=repr(e))
        raise
    finally:
        state["seconds"]=time.monotonic()-tick
        save(a.output / "status.json",state)


if __name__ == "__main__":
    main()
