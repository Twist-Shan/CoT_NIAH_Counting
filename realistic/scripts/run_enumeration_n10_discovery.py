"""Prepare, capture and select Enumeration N=10 Update states using Native probes."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from realistic_niah_v6.update_n10 import select_discovery, rank_layers, validate_selection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temp.replace(path)


def prepare(args, manifest, info, folder):
    from scripts.enumeration_fresh_geometry import adapt_update_row, json_sha
    from realistic_niah_v4.modeling import load_registered_tokenizer
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
    install_v6_kernel_adapters()
    install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5.count_stream import build_answer_source_registry
    tokenizer = load_registered_tokenizer(resolve_model_spec(args.model), cache_dir=manifest['cache_dir'])
    registry = Path(info['registry'])
    original = rows(registry / 'adapted_generations.jsonl')
    source = {int(r['seed']): r for r in original if int(r['gold_count']) == 10}
    candidates = list(manifest['initial_discovery_seeds'])
    origins = {s: str(registry / 'adapted_generations.jsonl') for s in candidates + info['confirmation_seeds']}
    if args.model in manifest['reserve_models']:
        reserve = args.stage / 'reserve_baselines' / args.model / args.mode
        finished = read(reserve / 'status.json')
        assert finished['status'] == 'COMPLETE' and finished['completed'] == len(manifest['reserve_seeds'])
        assert sha(reserve / 'generations.jsonl') == finished['generations_sha256']
        additions = rows(reserve / 'generations.jsonl')
        assert [r['seed'] for r in additions] == manifest['reserve_seeds']
        for row in additions:
            assert row['seed'] not in source and row['gold_count'] == 10 and row['split'] == 'discovery'
            source[row['seed']] = row
            origins[row['seed']] = str(reserve / 'generations.jsonl')
        candidates += manifest['reserve_seeds']
    ledger, adapted, geometry = [], {}, {}
    for seed in candidates + info['confirmation_seeds']:
        row, meta = adapt_update_row(source[seed], model=args.model, mode=args.mode)
        entry = dict(seed=seed, gold_count=10, format_eligible=meta['format_eligible'],
                     endpoint_eligible=False, source=origins[seed], source_row_sha256=json_sha(source[seed]),
                     split='discovery' if seed in candidates else 'confirmation',
                     final_correctness_used=False, intervention_outcomes_used=False)
        if meta['format_eligible']:
            encoding, registry_value = build_answer_source_registry(row, tokenizer)
            positions = [int(end) - 1 for _, end in registry_value.trace_items]
            assert len(positions) == len(set(positions)) == 10
            assert positions == sorted(positions) and positions[0] >= encoding.prompt_token_count
            entry['endpoint_eligible'] = True
            geometry[seed] = dict(positions=positions, token_ids=[int(encoding.input_ids[p]) for p in positions],
                                  input_ids_sha256=json_sha(list(encoding.input_ids)))
        ledger.append(entry)
        adapted[seed] = row
    discovery = select_discovery(ledger, candidates, manifest['all_original_confirmation_candidates'], 20)
    assert all(next(r for r in ledger if r['seed'] == seed)['endpoint_eligible'] for seed in info['confirmation_seeds'])
    write(folder / 'candidate_ledger.json', ledger)
    selected = discovery + info['confirmation_seeds']
    with (folder / 'selected_generations.jsonl').open('x', encoding='utf-8') as handle:
        for seed in selected:
            handle.write(json.dumps(adapted[seed], ensure_ascii=True) + '\n')
    write(folder / 'geometry.json', {str(s): geometry[s] for s in selected})
    cell = dict(model=args.model, mode=args.mode, gold_count=10, discovery_seeds=discovery,
                confirmation_seeds=info['confirmation_seeds'], discovery_states=200, confirmation_states=100,
                selection_used_final_correctness=False, intervention_outcomes_accessed=False,
                inputs_sha256=sha(folder / 'selected_generations.jsonl'), geometry_sha256=sha(folder / 'geometry.json'),
                ledger_sha256=sha(folder / 'candidate_ledger.json'), source_file_sha256={p: sha(p) for p in set(origins.values())},
                original_discovery_eligible=sum(r['endpoint_eligible'] for r in ledger if r['seed'] in manifest['initial_discovery_seeds']),
                reserve_selected=[s for s in discovery if s in manifest['reserve_seeds']])
    write(folder / 'cohort.json', cell)


def capture_endpoints(model, adapter, encoding, positions, layer_ids):
    import numpy as np
    import torch
    from scripts.run_realistic_niah_v5_same_site_progress_transplant import _chunk_forward, _encoding_tensors, _tensor_from_output
    ids, mask = _encoding_tensors(model, encoding)
    captured = {}
    chunk_start = 0

    def make_hook(layer):
        def hook(_module, _arguments, output):
            hidden = _tensor_from_output(output)
            local = [(k, p - chunk_start) for k, p in enumerate(positions) if chunk_start <= p < chunk_start + hidden.shape[1]]
            if local:
                values = hidden[0, [p for _, p in local]].detach().float().cpu().numpy()
                for (k, _), value in zip(local, values):
                    assert (k, layer) not in captured
                    captured[k, layer] = value
        return hook
    handles = [adapter.layers[layer].register_forward_hook(make_hook(layer)) for layer in layer_ids]
    previous = None
    try:
        with torch.inference_mode():
            for chunk_start in range(0, max(positions) + 1, 512):
                previous = _chunk_forward(model, adapter, ids, mask, start=chunk_start,
                    end=min(max(positions) + 1, chunk_start + 512), previous=previous)
    finally:
        for handle in handles:
            handle.remove()
        del previous
    assert len(captured) == len(positions) * len(layer_ids)
    values = np.stack([np.stack([captured[k, layer] for layer in layer_ids]) for k in range(10)])
    assert np.isfinite(values).all()
    return values


def capture(args, manifest, folder, state, save):
    import numpy as np
    import torch
    import pandas as pd
    from scripts.enumeration_fresh_geometry import json_sha
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v6.kernel import install_v6_kernel_adapters, install_v6_specialized_geometry
    install_v6_kernel_adapters()
    install_v6_specialized_geometry(args.mode)
    from realistic_niah_v5.count_stream import build_answer_source_registry
    cell = read(folder / 'cohort.json')
    assert sha(folder / 'selected_generations.jsonl') == cell['inputs_sha256']
    assert sha(folder / 'geometry.json') == cell['geometry_sha256']
    geometry = read(folder / 'geometry.json')
    confirm = args.phase == 'confirm'
    selection = read(args.stage / 'selection_manifest.json') if confirm else None
    chosen = validate_selection(selection, args.model, args.mode, cell['confirmation_seeds']) if confirm else None
    seeds = cell['confirmation_seeds'] if confirm else cell['discovery_seeds']
    source = {int(r['seed']): r for r in rows(folder / 'selected_generations.jsonl')}
    spec = resolve_model_spec(args.model)
    tick = time.monotonic()
    model, tokenizer, adapter = load_registered_model(spec, cache_dir=manifest['cache_dir'],
        device_map='auto', torch_dtype='bfloat16', attention_backend='sdpa')
    model.eval()
    assert not model.training and next(model.parameters()).dtype == torch.bfloat16
    expected_layers = {'Qwen3-8B':36,'Gemma4-E4B':42}[args.model]
    assert adapter.num_layers == expected_layers
    layers = [chosen] if confirm else list(range(adapter.num_layers))
    text_cfg = model.config.get_text_config() if hasattr(model.config,'get_text_config') else model.config
    runtime = dict(model_revision=spec.revision, model_source_sha256=sha(sys.modules[type(model).__module__].__file__),
        torch=torch.__version__, transformers=importlib.metadata.version('transformers'), python=sys.version,
        dtype='torch.bfloat16', backend='sdpa', layers=adapter.num_layers, gpu=torch.cuda.get_device_name(),
        command=sys.argv, model_load_seconds=time.monotonic()-tick)
    original = next(c for c in manifest['cells'] if c['model']==args.model and c['mode']==args.mode)
    reference = read(original['source_runtime'])
    for key in ('model_revision','model_source_sha256','torch','transformers','dtype','backend','layers'):
        assert runtime[key] == reference[key], f'Runtime changed: {key}'
    write(folder / f'{args.phase}_runtime.json', runtime)
    arrays, metadata, timings = [], [], []
    for index, seed in enumerate(seeds):
        start = time.monotonic()
        encoding, reg = build_answer_source_registry(source[seed], tokenizer)
        positions = [int(end)-1 for _,end in reg.trace_items]
        expected = geometry[str(seed)]
        assert positions == expected['positions']
        assert [int(encoding.input_ids[p]) for p in positions] == expected['token_ids']
        assert json_sha(list(encoding.input_ids)) == expected['input_ids_sha256']
        values = capture_endpoints(model, adapter, encoding, positions, layers)
        if index == 0:
            repeat = capture_endpoints(model, adapter, encoding, positions, layers)
            assert np.array_equal(values, repeat), 'Repeated first-trace capture differs'
            state['first_trace_repeat_exact'] = True
            del repeat
        assert model.config._attn_implementation == text_cfg._attn_implementation == 'sdpa'
        arrays.append(values)
        metadata.extend(dict(seed=seed, occurrence=k, gold_count=10, split='confirmation' if confirm else 'discovery',
            position=pos, token_id=int(encoding.input_ids[pos]), request_id=source[seed]['request_id']) for k,pos in enumerate(positions,1))
        timings.append(dict(seed=seed, seconds=time.monotonic()-start))
        state.update(completed_traces=index+1, expected_traces=len(seeds), completed_states=(index+1)*10)
        save()
        print(json.dumps({'phase':args.phase,'model':args.model,'mode':args.mode,'traces':index+1,'total':len(seeds)}),flush=True)
    states = np.concatenate(arrays,axis=0)
    assert states.shape[:2] == (len(seeds)*10,len(layers))
    prefix = 'confirmation' if confirm else 'discovery'
    np.savez_compressed(folder / f'{prefix}_states.npz', states=states, layer_indices=np.array(layers))
    pd.DataFrame(metadata).to_csv(folder / f'{prefix}_metadata.csv', index=False)
    write(folder / f'{prefix}_capture_audit.json',dict(status='PASS',states_shape=list(states.shape),timings=timings,
        first_trace_repeat_exact=True, states_sha256=sha(folder/f'{prefix}_states.npz'),metadata_sha256=sha(folder/f'{prefix}_metadata.csv')))
    if confirm:
        from realistic_niah_v5.trace_stratified_geometry import _fit_projection_and_predict
        with np.load(folder / 'discovery_states.npz') as archive:
            discovery = archive['states'][:,chosen]
        dmeta = pd.read_csv(folder / 'discovery_metadata.csv')
        truth = np.array([r['occurrence'] for r in metadata])
        logistic,ncc,_,components = _fit_projection_and_predict(discovery,dmeta['occurrence'].to_numpy(dtype=int),
            states[:,0],np.arange(1,11),pca_dim=16,random_state=0,pca_whiten=True)
        write(folder/'confirmation_readout.json',dict(status='PASS',layer_one_based=chosen+1,states=100,
            ncc_balanced_accuracy=float((ncc==truth).mean()),logistic_balanced_accuracy=float((logistic==truth).mean()),
            pca_components=components,selection_manifest_sha256=sha(args.stage/'selection_manifest.json'),
            confirmation_is_reused=True, predictions=[dict(**row,ncc=int(n),logistic=int(l)) for row,n,l in zip(metadata,ncc,logistic)]))


def analyze(args, manifest, folder, state, save):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from realistic_niah_v5.trace_stratified_geometry import grouped_discovery_cv_metrics
    cell = read(folder / 'cohort.json')
    audit = read(folder / 'discovery_capture_audit.json')
    assert audit['status'] == 'PASS' and sha(folder/'discovery_states.npz') == audit['states_sha256']
    assert sha(folder/'discovery_metadata.csv') == audit['metadata_sha256']
    frame = pd.read_csv(folder/'discovery_metadata.csv')
    with np.load(folder/'discovery_states.npz') as archive:
        states, layers = archive['states'],archive['layer_indices']
    assert states.shape[:2] == (200,{'Qwen3-8B':36,'Gemma4-E4B':42}[args.model])
    assert np.array_equal(layers,np.arange(states.shape[1])) and np.isfinite(states).all()
    assert set(frame['split']) == {'discovery'} and set(frame['gold_count']) == {10}
    assert set(frame['seed']) == set(cell['discovery_seeds'])
    assert all(sorted(group['occurrence']) == list(range(1,11)) for _,group in frame.groupby('seed'))
    folds = []
    for train,test in GroupKFold(5).split(states[:,0],frame['occurrence'],groups=frame['seed']):
        folds.append(dict(train_rows=train.tolist(),test_rows=test.tolist(),
            train_seeds=sorted(set(map(int,frame.iloc[train]['seed']))),test_seeds=sorted(set(map(int,frame.iloc[test]['seed'])))))
    write(folder/'discovery_folds.json',folds)
    metrics = []
    for layer in layers:
        tick = time.monotonic()
        value = grouped_discovery_cv_metrics(states[:,layer],frame,np.arange(1,11),pca_dim=16,
                                             random_state=0,folds=5,pca_whiten=True)
        metrics.append(dict(layer_zero_based=int(layer),layer_one_based=int(layer)+1,**value,seconds=time.monotonic()-tick))
        state.update(completed_layers=len(metrics),expected_layers=len(layers));save()
        print(f'CV {args.model}/{args.mode} L{layer+1} NCC={value["discovery_oof_ncc_balanced_accuracy"]:.4f}',flush=True)
    ranked = rank_layers(metrics,args.model)
    pd.DataFrame(metrics).to_csv(folder/'discovery_layer_metrics.csv',index=False)
    write(folder/'selection.json',dict(status='PASS',model=args.model,mode=args.mode,selected=ranked[0],top_five=ranked[:5],
        candidate_layers_one_based=[1,35 if args.model=='Qwen3-8B' else 22],
        discovery_seeds=cell['discovery_seeds'],discovery_states=200,gold_count=10,confirmation_used=False,
        states_sha256=audit['states_sha256'],metadata_sha256=audit['metadata_sha256'],
        folds_sha256=sha(folder/'discovery_folds.json'),metrics_sha256=sha(folder/'discovery_layer_metrics.csv')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--model',choices=('Qwen3-8B','Gemma4-E4B'),required=True)
    parser.add_argument('--mode',choices=('enumeration_index','enumeration_bullet'),required=True)
    parser.add_argument('--phase',choices=('prepare','capture','analyze','confirm'),required=True)
    args=parser.parse_args()
    manifest=read(args.stage/'manifest.json')
    for relative,digest in manifest['code_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    for path,digest in manifest['source_sha256'].items():
        assert sha(path)==digest,path
    info=next(c for c in manifest['cells'] if c['model']==args.model and c['mode']==args.mode)
    folder=args.stage/'cells'/args.model/args.mode
    folder.mkdir(parents=True,exist_ok=True)
    status=folder/f'{args.phase}_status.json'
    assert not status.exists(),f'Preserve earlier attempt: {status}'
    tick=time.monotonic();state={'status':'RUNNING','phase':args.phase,'command':sys.argv}
    def save():
        state['seconds']=time.monotonic()-tick;write(status,state)
    save()
    try:
        if args.phase=='prepare':prepare(args,manifest,info,folder)
        elif args.phase=='analyze':analyze(args,manifest,folder,state,save)
        else:capture(args,manifest,folder,state,save)
        state['status']='COMPLETE'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    main()
