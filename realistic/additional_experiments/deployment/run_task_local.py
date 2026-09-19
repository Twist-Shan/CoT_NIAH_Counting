"""New-task discovery attention, frozen selection, and overlap-permitted ablation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

R = Path(__file__).resolve().parent
sys.path[:0] = [str(R), str(R / "src")]
from local_selection import rank_discovery, select_heads, random_control, target_span, control_audit
from task_scoring import score_generation, target_token_geometry


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["discover", "canary", "full"], required=True)
    ap.add_argument("--cache-dir", type=Path, help="Override model cache after relocating a frozen package")
    args = ap.parse_args()
    started = time.perf_counter()
    cfg = read(R / "protocol.json")
    assert cfg['control_policy'] == 'disjoint_first_minimum_overlap'
    for name, expected in cfg["files"].items():
        assert sha(R / name) == expected, (name, "changed frozen source")
    if args.stage == "full":
        for model in cfg["models"]:
            assert read(R / "canary" / model / "complete.json")["status"] == "PASS"
    import torch
    from protocol import encode_ids
    from run import summarize_generation
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v4.modeling import (load_registered_model, query_attention_rows,
        generate_with_head_ablation, generate_answer_completion, _is_prompt_prefill, _text_config)
    from realistic_niah_v5.causal import _first_generated_city_record
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this stage")
    torch.manual_seed(20260908)
    model, tok, adapter = load_registered_model(resolve_model_spec(args.model), cache_dir=args.cache_dir or cfg["cache"])
    model.eval()
    widths = list(adapter.num_heads)
    assert widths == cfg["widths"][args.model]
    def backend():
        values = [getattr(c, "_attn_implementation", None) for c in [model.config, _text_config(model)] if c is not None]
        assert values and all(v == "sdpa" for v in values), values
        return values
    out = R / ("discovery" if args.stage == "discover" else args.stage) / args.model
    write(out / "environment.json", dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
          widths=widths, backend=backend(), protocol_sha256=sha(R / "protocol.json")))

    def inputs(plan):
        source = Path(plan["source"])
        if not source.is_absolute():
            source = R / source
        for name, expected in plan["source_hashes"].items():
            assert sha(source / name) == expected
        p, g = read(source / "prompt.json"), read(source / "generation.json")
        return p, g, p["input_ids"] + g["generated_token_ids"]

    def generate(ids, heads, assay, max_tokens):
        enc = encode_ids(ids)
        backend()
        if assay == "broad":
            result = generate_with_head_ablation(model, tok, adapter, enc, heads, scope="answer_query", max_new_tokens=max_tokens)
        elif not heads:
            result = generate_answer_completion(model, tok, enc, max_new_tokens=max_tokens)
        else:
            hooks, groups, calls = [], defaultdict(list), Counter()
            for l, h in heads: groups[l].append(h)
            for l, hs in groups.items():
                def hook(module, values, l=l, hs=hs):
                    x = values[0]; changed = x.clone()
                    positions = [enc.query_position] if _is_prompt_prefill(x, enc) else slice(None)
                    for h in hs:
                        changed[:, positions, h*adapter.head_dims[l]:(h+1)*adapter.head_dims[l]] = 0
                    calls[l] += 1
                    return (changed, *values[1:])
                hooks.append(adapter.output_projections[l].register_forward_pre_hook(hook))
            try:
                result = generate_answer_completion(model, tok, enc, max_new_tokens=max_tokens)
            finally:
                for h in hooks: h.remove()
            assert all(calls[l] for l in groups)
            result["hook_calls"] = dict(calls)
        backend()
        return result

    def score(g, plan, assay, target_meta=None):
        return score_generation(g,plan['case'],mode=plan['mode'],assay=assay,
                                target=(plan['target'] or {}).get('target_city'),
                                **(target_meta or {}))

    for task in cfg["tasks"]:
        plans = read(R / "plans" / f"{task}_{args.model}.json")
        if args.stage == "discover":
            if cfg.get('input_mode') == 'fresh_natural_generations':
                from task_local_inputs import broad_spans, broad_score
                for mode in ['nonthinking', 'native_thinking']:
                    broad_observations, broad_files = [], {}
                    for plan in plans:
                        if plan['mode'] != mode or plan['split'] != 'discovery':
                            continue
                        path = out / task / 'broad' / mode / (plan['case_id']+'.json')
                        if path.exists():
                            row = read(path)
                            assert row['protocol_sha256'] == sha(R / 'protocol.json')
                        else:
                            row = dict(case_id=plan['case_id'], seed=plan['seed'], split='discovery',
                                mode=mode, heads=[], protocol_sha256=sha(R / 'protocol.json'))
                            if plan['broad_prefix']:
                                prompt, generation, ids = inputs(plan)
                                spans = broad_spans(plan, prompt, tok)
                                enc = encode_ids(ids[:plan['broad_prefix']])
                                backend(); matrices, starts = query_attention_rows(model, adapter, enc); backend()
                                for layer, (matrix, start) in enumerate(zip(matrices, starts)):
                                    for head, alpha in enumerate(matrix):
                                        masses = [float(alpha[max(0, a-start):min(len(alpha), b-start)].sum())
                                                  if b > start and a-start < len(alpha) else 0.
                                                  for a, b in spans]
                                        row['heads'].append([layer, head, broad_score(masses)])
                                row.update(prefix_length=plan['broad_prefix'], spans=spans, backend=backend())
                            else:
                                row['unavailable'] = plan['unavailable'].get('broad')
                            write(path, row)
                        if row['heads']:
                            broad_observations.append(row)
                        broad_files[path.relative_to(R).as_posix()] = sha(path)
                    ranking = rank_discovery(broad_observations)
                    bank = dict(model=args.model, task=task, mode=mode, assay='broad', ranking=ranking,
                        sizes=cfg['broad_sizes'][args.model], source_hashes=broad_files, conditions={})
                    for k in bank['sizes']:
                        selected = select_heads(ranking, k)
                        randoms = [random_control(selected, widths, 7000+i) for i in range(3)]
                        bank['conditions'][str(k)] = dict(selected=selected, random=randoms,
                            control_audit=control_audit(selected, widths, randoms))
                    bankpath = R / 'banks' / f'{task}_{args.model}_{mode}_broad.json'
                    if bankpath.exists():
                        assert read(bankpath) == bank
                    else:
                        write(bankpath, bank)
            observations = []
            for p in plans:
                if p["mode"] != "native_thinking" or p["split"] != "discovery": continue
                dest = out / task / f"{p['case_id']}.json"
                if dest.exists():
                    row = read(dest)
                    assert row["protocol_sha256"] == sha(R / "protocol.json")
                    if row["heads"]: observations.append(row)
                    continue
                row = dict(case_id=p["case_id"], seed=p["seed"], split=p["split"], heads=[], protocol_sha256=sha(R / "protocol.json"))
                if p["target"]:
                    prompt, generation, ids = inputs(p)
                    encoded = tok(prompt["rendered_prompt"], add_special_tokens=False, return_offsets_mapping=True)
                    assert encoded["input_ids"] == prompt["input_ids"]
                    a, b = target_span(p["case"], prompt["rendered_prompt"], encoded["offset_mapping"], p["target"]["target_city"])
                    enc = encode_ids(ids[:p["target"]["prefix_length"]])
                    backend(); matrices, starts = query_attention_rows(model, adapter, enc); backend()
                    for l, (matrix, start) in enumerate(zip(matrices, starts)):
                        for h, alpha in enumerate(matrix):
                            value = float(alpha[max(0,a-start):min(len(alpha),b-start)].float().sum()) if b > start else 0.
                            row["heads"].append([l, h, value])
                    row.update(target_span=[a,b], prefix_length=p["target"]["prefix_length"], target=p["target"]["target_city"], backend=backend())
                    observations.append(row)
                else: row["unavailable"] = p["unavailable"].get("targeted")
                write(dest, row)
                print(json.dumps(dict(stage="discovery", model=args.model, task=task, case_id=p["case_id"])), flush=True)
            ranking = rank_discovery(observations)
            bank = dict(model=args.model, task=task, mode="native_thinking", assay="targeted", ranking=ranking,
                        sizes=cfg["target_sizes"][args.model], discovery_cases=len(observations),
                        source_hashes={str(p.relative_to(R)):sha(p) for p in (out/task).glob("*.json")}, conditions={})
            for k in bank["sizes"]:
                heads = select_heads(ranking,k)
                bank["conditions"][str(k)] = dict(selected=heads, random=[random_control(heads,widths,6000+i) for i in range(3)])
                bank['conditions'][str(k)]['control_audit'] = control_audit(heads,widths,bank['conditions'][str(k)]['random'])
            dest = R / "banks" / f"{task}_{args.model}_native_thinking_targeted.json"
            if dest.exists(): assert read(dest) == bank
            else: write(dest, bank)
            continue
        banks = {(mode,assay): R/"banks"/f"{task}_{args.model}_{mode}_{assay}.json" for mode,assay in [("nonthinking","broad"),("native_thinking","broad"),("native_thinking","targeted")]}
        panel = [p for p in plans if p["split"] == "confirmation"]
        if args.stage == "canary":
            cid = next(p["case_id"] for p in panel if p["mode"]=="native_thinking" and p["broad_prefix"] and p["target"])
            panel = [p for p in panel if p["case_id"]==cid]
        for p in panel:
            prompt, generation, ids = inputs(p)
            target_meta = None
            if p['target']:
                target_meta = target_token_geometry(tok,generation['completion_text_raw'],
                    generation['generated_token_ids'],p['target']['target_city_start'],
                    p['target']['target_city'],p['target']['prefix_length']-len(prompt['input_ids']))
            for assay in (["broad"] if p["mode"]=="nonthinking" else ["broad","targeted"]):
                dest = out/task/(p["mode"]+"_"+assay)/p["case_id"]
                length = p["broad_prefix"] if assay=="broad" else (p["target"] or {}).get("prefix_length")
                if not length:
                    write(dest/"unavailable.json",dict(reason=p["unavailable"].get(assay),case_id=p["case_id"]))
                    continue
                prefix_generated_ids=ids[len(prompt['input_ids']):length]
                max_tokens=cfg['broad_max_tokens'] if assay=='broad' else cfg['targeted_max_tokens']
                prefix_text=tok.decode(prefix_generated_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                assert generation['completion_text_raw'].startswith(prefix_text)
                bankpath=banks[p["mode"],assay]; bank=read(bankpath)
                sizes=bank["sizes"] if args.stage=="full" else sorted({bank["sizes"][0],bank["sizes"][-1]})
                cleanfile=dest/"clean.json"
                if cleanfile.exists():
                    saved=read(cleanfile); assert saved["prefix_length"]==length and saved["protocol_sha256"]==sha(R/"protocol.json")
                    clean=saved["arm"]
                else:
                    gen=generate(ids[:length],[],assay,max_tokens); clean=dict(name="clean",heads=[],generation=gen,score=score(gen,p,assay,target_meta))
                    write(cleanfile,dict(prefix_length=length,protocol_sha256=sha(R/"protocol.json"),arm=clean))
                for k in sizes:
                    f=dest/f"K{k}.json"
                    if f.exists():
                        prev=read(f);assert prev["protocol_sha256"]==sha(R/"protocol.json") and prev["bank_sha256"]==sha(bankpath)
                        continue
                    heads=bank["conditions"][str(k)]; expected=[("selected",heads["selected"])]+[(f"random_{i}",hh) for i,hh in enumerate(heads["random"])]
                    assert heads['control_audit'] == control_audit(heads['selected'],widths,heads['random'])
                    arms=[clean]
                    for name,hh in expected:
                        gen=generate(ids[:length],hh,assay,max_tokens)
                        arms.append(dict(name=name,heads=hh,generation=gen,score=score(gen,p,assay,target_meta)))
                    overlap=[len(set(map(tuple,heads["selected"])) & set(map(tuple,hh))) for hh in heads["random"]]
                    write(f,dict(model=args.model,task=task,case_id=p["case_id"],seed=p["seed"],mode=p["mode"],assay=assay,k=k,
                          prefix_length=length,prefix_generated_text=prefix_text,prefix_generated_token_count=len(prefix_generated_ids),max_new_tokens=max_tokens,arms=arms,overlap_counts=overlap,backend=backend(),
                          protocol_sha256=sha(R/"protocol.json"),bank_sha256=sha(bankpath),target_token_geometry=target_meta,control_audit=heads['control_audit']))
                    print(json.dumps(dict(stage=args.stage,model=args.model,task=task,mode=p["mode"],assay=assay,case_id=p["case_id"],k=k)),flush=True)
    write(out/"complete.json",dict(status="PASS",stage=args.stage,elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha(R/"protocol.json")))


if __name__ == "__main__": main()
