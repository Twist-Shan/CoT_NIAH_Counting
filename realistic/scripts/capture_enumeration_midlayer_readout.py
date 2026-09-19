"""Evaluate the already fixed L19 decoder on the new confirmation inputs."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def freeze(root,stage):
    primary=root/'fresh_n10_update_v1';update=root/'fresh_midlayer_update_20260917_v1';um=read(update/'manifest.json')
    assert um['new_layer_one_based']==19
    folder=primary/'cells/Qwen3-8B/enumeration_index'
    paths=[update/'manifest.json',update/'inputs/adapted_generations.jsonl',primary/'code/scripts/run_enumeration_n10_discovery.py',
        folder/'discovery_states.npz',folder/'discovery_metadata.csv',folder/'discovery_layer_metrics.csv',folder/'discovery_capture_audit.json']
    stage.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,stage/Path(__file__).name)
    m=dict(root=str(root),update=str(update),source_code=um['source_code'],source_runtime=um['source_runtime'],
        discovery=str(folder),cache_dir=um['cache_dir'],layer_zero_based=18,confirmation_seeds=um['selected_seeds'],
        source_sha256={str(p):sha(p) for p in paths},script_sha256=sha(__file__),utc=datetime.now(timezone.utc).isoformat(),
        purpose='Report discovery-only-trained readout on the fixed L19 confirmation cohort; no new layer choice')
    write(stage/'manifest.json',m);print(json.dumps(dict(status='FROZEN',manifest_sha256=sha(stage/'manifest.json'))))

def run(stage):
    tick=time.monotonic();m=read(stage/'manifest.json');assert sha(__file__)==m['script_sha256']
    for p,h in m['source_sha256'].items():assert sha(p)==h,p
    assert read(Path(m['update'])/'status.json')['status']=='COMPLETE'
    sys.path[:0]=[m['source_code']+'/src',m['source_code']]
    import numpy as np
    import pandas as pd
    import torch
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters,install_v6_specialized_geometry
    from realistic_niah_v5.trace_stratified_geometry import _fit_projection_and_predict
    install_v6_kernel_adapters();install_v6_specialized_geometry('enumeration_index')
    from realistic_niah_v5.count_stream import build_answer_source_registry
    state=dict(status='RUNNING',phase='CAPTURE',completed_traces=0,pid=os.getpid())
    def save():state['seconds']=time.monotonic()-tick;write(stage/'status.json',state)
    save()
    try:
        path=Path(m['root'])/'fresh_n10_update_v1/code/scripts/run_enumeration_n10_discovery.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='capture_endpoints')
        namespace={};exec(ast.unparse(node),namespace);capture=namespace['capture_endpoints']
        model,tokenizer,adapter=load_registered_model(resolve_model_spec('Qwen3-8B'),cache_dir=m['cache_dir'],device_map='auto',torch_dtype='bfloat16',attention_backend='sdpa');model.eval()
        runtime=dict(model_revision=resolve_model_spec('Qwen3-8B').revision,model_source_sha256=sha(sys.modules[type(model).__module__].__file__),
            dtype=str(next(model.parameters()).dtype),backend='sdpa',layers=adapter.num_layers,torch=torch.__version__,transformers=importlib.metadata.version('transformers'))
        reference=read(m['source_runtime'])
        for k,v in runtime.items():assert reference[k]==v,k
        write(stage/'runtime.json',runtime)
        with (Path(m['update'])/'inputs/adapted_generations.jsonl').open() as f:inputs={r['seed']:r for r in map(json.loads,f)}
        arrays=[];metadata=[];layer=m['layer_zero_based']
        for index,seed in enumerate(m['confirmation_seeds']):
            encoding,reg=build_answer_source_registry(inputs[seed],tokenizer);positions=[int(end)-1 for _,end in reg.trace_items];assert len(positions)==10
            values=capture(model,adapter,encoding,positions,[layer])
            if index==0:assert np.array_equal(values,capture(model,adapter,encoding,positions,[layer]))
            arrays.append(values)
            metadata.extend(dict(seed=seed,gold_count=10,occurrence=k,split='confirmation',position=pos,token_id=int(encoding.input_ids[pos])) for k,pos in enumerate(positions,1))
            state['completed_traces']=index+1;save()
        states=np.concatenate(arrays,axis=0);assert states.shape[:2]==(100,1) and np.isfinite(states).all()
        np.savez_compressed(stage/'confirmation_states.npz',states=states,layer_indices=np.array([layer]));pd.DataFrame(metadata).to_csv(stage/'confirmation_metadata.csv',index=False)
        folder=Path(m['discovery']);audit=read(folder/'discovery_capture_audit.json');assert sha(folder/'discovery_states.npz')==audit['states_sha256']
        with np.load(folder/'discovery_states.npz') as a:discovery=a['states'][:,layer]
        dmeta=pd.read_csv(folder/'discovery_metadata.csv');truth=np.array([r['occurrence'] for r in metadata])
        logistic,ncc,_,components=_fit_projection_and_predict(discovery,dmeta['occurrence'].to_numpy(dtype=int),states[:,0],np.arange(1,11),pca_dim=16,random_state=0,pca_whiten=True)
        metrics=pd.read_csv(folder/'discovery_layer_metrics.csv');d=metrics[metrics['layer_one_based'].eq(layer+1)].iloc[0]
        result=dict(status='PASS',layer_one_based=layer+1,states=100,discovery_ncc=float(d['discovery_oof_ncc_balanced_accuracy']),
            confirmation_ncc=float((ncc==truth).mean()),confirmation_logistic=float((logistic==truth).mean()),pca_components=components,
            used_for_layer_selection=False,first_trace_repeat_exact=True,manifest_sha256=sha(stage/'manifest.json'),
            predictions=[dict(**r,ncc=int(n),logistic=int(l)) for r,n,l in zip(metadata,ncc,logistic)],
            files_sha256={p.name:sha(p) for p in stage.iterdir() if p.is_file() and p.suffix in ['.npz','.csv']})
        write(stage/'readout.json',result);state.update(status='COMPLETE',phase='CONFIRMATION_READOUT_COMPLETE');print(json.dumps({k:result[k] for k in ['status','discovery_ncc','confirmation_ncc','confirmation_logistic']}),flush=True)
    except BaseException as e:state.update(status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--stage',type=Path,required=True);p.add_argument('--root',type=Path);a=p.parse_args();freeze(a.root,a.stage) if a.action=='freeze' else run(a.stage)
