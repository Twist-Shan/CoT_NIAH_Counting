"""Freeze canonical Bullet data and compile geometry, retaining original outputs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

MOUNT = Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5')
OLD = MOUNT/'supplements/enumeration_alignment_20260916'
ROOT = MOUNT/'supplements/enumeration_default_seed_20260918_v2'
MODE = 'enumeration_bullet'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n')
    temp.replace(path)

def main():
    started=time.monotonic()
    ROOT.mkdir(exist_ok=False)
    state=dict(status='RUNNING',phase='FREEZE',started_utc=time.time())
    def save():
        state['seconds']=time.monotonic()-started;write(ROOT/'setup_status.json',state)
    save()
    try:
        code=ROOT/'code'
        shutil.copytree(OLD/'fresh_n10_update_v1/code',code,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
        # Keep the repaired numerical kernels and latest answer/retrieve wrappers.
        for source in (OLD/'fresh_retrieve_gpu_v2/code/scripts',OLD/'fresh_answer_patch_gpu_v3_native_eligibility/code/scripts'):
            for p in source.glob('*.py'): shutil.copy2(p,code/'scripts'/p.name)
        shutil.copy2(__file__,ROOT/Path(__file__).name)
        bundle=ROOT/'fresh_v1';bundle.mkdir();(bundle/'code').symlink_to(code,target_is_directory=True)
        causal=ROOT/'fresh_causal_v1';causal.mkdir();(causal/'code').symlink_to(code,target_is_directory=True)
        cfg=read(OLD/'fresh_v1/protocol.json')
        cfg.update(first_seed=1234,models=MODELS,modes=[MODE],confirmation_candidate_count=10,
            discovery_reserve=list(range(1264,1324)),confirmation_reserve=list(range(1324,1384)),
            chronology='Canonical historical samples reused for alignment; not new independent confirmation.')
        write(bundle/'protocol.json',cfg)
        code_hash={str(p.relative_to(code)):sha(p) for top in ('src','scripts','configs','tests')
                   for p in sorted((code/top).rglob('*')) if p.suffix in ('.py','.json')}
        source=MOUNT/'runs/v6_enumeration_replication_20260828/source_stimuli/stimuli.jsonl'
        with source.open() as f: all_stim=[json.loads(l) for l in f]
        stimuli=[r for r in all_stim if 1234<=r['seed']<=1263 and r['design_variant']=='v4.4']
        expected={(s,n) for s in range(1234,1264) for n in range(1,11)}
        assert len(stimuli)==300 and {(r['seed'],r['gold_count']) for r in stimuli}==expected
        by_key={(r['seed'],r['gold_count']):r for r in stimuli}
        for r in stimuli:
            assert r['split']==('discovery' if r['seed']<1254 else 'confirmation')
            r['alignment_roles']=(['unfiltered_read'] if r['seed']>=1254 else [])+(['causal_discovery' if r['seed']<1254 else 'causal_candidate'] if r['gold_count']==10 else [])
        with (bundle/'stimuli.jsonl').open('x') as f:
            for r in stimuli: f.write(json.dumps(r,ensure_ascii=True)+'\n')
        manifest=dict(status='FROZEN_CANONICAL_REUSE',discovery_seeds=list(range(1234,1254)),
            confirmation_candidates=list(range(1254,1264)),read_seeds=list(range(1254,1264)),
            rows_per_cell=300,baseline_cells=2,baseline_rows=600,code_sha256=code_hash,
            protocol_sha256=sha(bundle/'protocol.json'),stimuli_sha256=sha(bundle/'stimuli.jsonl'),
            source_stimuli=str(source),source_stimuli_sha256=sha(source))
        write(bundle/'manifest.json',manifest)
        write(causal/'code_manifest.json',dict(original_code_sha256=code_hash,additive_code_sha256={},
            status='FROZEN_CANONICAL_REUSE',baseline_manifest_sha256=sha(bundle/'manifest.json')))
        sys.path[:0]=[str(code/'src'),str(code)]
        from realistic_niah_v4.modeling import load_registered_tokenizer
        from realistic_niah_v4.spec import resolve_model_spec
        from realistic_niah_v6.generation import render_structured_prompt
        provenance=[]
        for model in MODELS:
            folder=MOUNT/'runs/v6_enumeration_replication_20260828'/MODE/model/'generation'
            archived=folder/'generations.jsonl'
            with archived.open() as f: rows=[r for r in map(json.loads,f) if 1234<=r['seed']<=1263]
            assert len(rows)==300 and {(r['seed'],r['gold_count']) for r in rows}==expected
            spec=resolve_model_spec(model)
            tokenizer=load_registered_tokenizer(spec,cache_dir=MOUNT/'cache/huggingface')
            dest=bundle/'baseline'/model/MODE;dest.mkdir(parents=True)
            audit=[]
            with (dest/'generations.jsonl').open('x') as f:
                for row in rows:
                    stim=by_key[row['seed'],row['gold_count']]
                    prompt=render_structured_prompt(stim,tokenizer=tokenizer,model_spec=spec,prompt_mode=MODE)
                    assert row['model_revision']==spec.revision and row['split']==stim['split']
                    assert row['input_ids']==list(prompt.input_ids) and row['rendered_prompt']==prompt.rendered_prompt
                    row.update(alignment_mode=MODE,alignment_roles=stim['alignment_roles'],
                        stimulus_passage_sha256=stim['passage_sha256'],alignment_manifest_sha256=sha(bundle/'manifest.json'),
                        alignment_protocol_sha256=manifest['protocol_sha256'],baseline_correctness_filter_applied=False)
                    f.write(json.dumps(row,ensure_ascii=True)+'\n')
                    audit.append(dict(seed=row['seed'],n=row['gold_count'],prompt_exact=True))
            write(dest/'status.json',dict(status='COMPLETE',completed=300,total=300,reused=True,
                generations_sha256=sha(dest/'generations.jsonl'),source=str(archived),source_sha256=sha(archived)))
            write(dest/'reuse_audit.json',audit)
            provenance.append(dict(model=model,source=str(archived),source_sha256=sha(archived),rows=300))
            state.update(phase='REGISTRY',model=model);save()
            cmd=[sys.executable,str(code/'scripts/prepare_enumeration_fresh_causal_registry.py'),
                 '--bundle',str(bundle),'--model',model,'--mode',MODE,'--cache-dir',str(MOUNT/'cache/huggingface'),
                 '--output',str(causal/'registries'/model/MODE)]
            with (ROOT/f'registry_{model}.log').open('x') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(ROOT/'provenance.json',provenance)
        state.update(status='COMPLETE',phase='CANONICAL_GEOMETRY_READY')
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
