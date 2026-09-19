"""Within-cell Enumeration geometry; no intersection across formats or models.

The population amendment is descriptive and reuses the original split. It does
not replace N=10 causal-layer selection or constitute new confirmation data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from realistic_niah_v5.cross_mode_geometry import CLASSES, load_native_thinking_capture
from realistic_niah_v5.dual_endpoint_geometry import load_native_thinking_final_count
from realistic_niah_v5.trace_stratified_geometry import grouped_discovery_cv_metrics, confirmation_metrics
from realistic_niah_v6.native_aligned_representation import _validate_original_capture, _artifact, _package_version
from realistic_niah_v6.spec import DISCOVERY_SEEDS, CONFIRMATION_SEEDS


def select_ncc(rows):
    """The frozen paper rule reads discovery fields only."""
    return max(rows, key=lambda r: (round(float(r['discovery_oof_ncc_balanced_accuracy']), 12),
               round(float(r['discovery_oof_logistic_balanced_accuracy']), 12), -int(r['layer'])))


def population(dataset):
    """Validate and describe every loaded row, without dropping other-cell absences."""
    dataset.validate()
    frame = dataset.metadata
    result = {}
    for split, seeds in [('discovery', DISCOVERY_SEEDS), ('confirmation', CONFIRMATION_SEEDS)]:
        sub = frame[frame['split'].eq(split)]
        if set(sub['seed']) != set(seeds) or set(sub['occurrence']) != set(CLASSES):
            raise ValueError('Own-state population lacks registered seeds/classes')
        result[split] = dict(rows=len(sub), seeds=len(set(sub['seed'])),
            label_support={str(int(k)): int(v) for k,v in sub['occurrence'].value_counts().sort_index().items()})
    if len(frame) != sum(x['rows'] for x in result.values()):
        raise ValueError('Unexpected split')
    return result


def fit_display_pca(states, discovery):
    """Use Thinking's discovery-only PCA3 and sklearn's native axis directions."""
    states = np.asarray(states, dtype=np.float32)
    discovery = np.asarray(discovery, dtype=bool)
    if states.ndim != 2 or discovery.shape != (len(states),) or discovery.sum() < 3:
        raise ValueError('Invalid discovery support for display PCA3')
    if not np.isfinite(states).all():
        raise ValueError('Nonfinite display state')
    scaler = StandardScaler().fit(states[discovery])
    pca = PCA(n_components=3, svd_solver='randomized', random_state=0, whiten=False).fit(
        scaler.transform(states[discovery]))
    return scaler, pca, pca.transform(scaler.transform(states))


def analyze(run_root: Path, output: Path):
    start = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    (output/'data').mkdir(); (output/'states').mkdir()
    records = []; coords = []; panels = []; inputs = {}; supports = {}; timing = {}
    selected = {'running_index': [], 'final_count': []}
    for mode in ['enumeration_index', 'enumeration_bullet']:
        for model in ['Qwen3-8B', 'Gemma4-E4B']:
            cell = mode+'|'+model; index = run_root/mode/model/'capture/confirmation_all_sample/capture_index.jsonl'
            inputs[cell] = _validate_original_capture(index, (mode, model))
            # Bind the original captured tensors and site definitions, not just the index.
            sources = {}
            for row in map(json.loads, index.read_text().splitlines()):
                for key in ['manifest_path', 'states_path']:
                    relative = row[key]
                    sources[relative] = _artifact(index.parent/relative)
            (output/'data'/f'{mode}_{model}_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
            for endpoint in ['running_index', 'final_count']:
                tick=time.monotonic()
                site = 'item_end' if endpoint=='running_index' else 'answer_query_v3'
                dataset = (load_native_thinking_capture(index, site_kind=site, site_policy='uniform', cohort='parser_hit')
                           if endpoint=='running_index' else load_native_thinking_final_count(index))
                pop = population(dataset); supports[cell+'|'+endpoint]=pop
                frame = dataset.metadata
                expected_depth = 36 if model=='Qwen3-8B' else 42
                assert sorted(dataset.states_by_layer)==list(range(expected_depth))
                if endpoint=='final_count':
                    assert pop['discovery']['rows']==200 and pop['confirmation']['rows']==100
                candidates=[]
                for layer, states in sorted(dataset.states_by_layer.items()):
                    d=grouped_discovery_cv_metrics(states, frame, CLASSES, pca_dim=16, random_state=0, folds=5, pca_whiten=True)
                    c=confirmation_metrics(states, frame, CLASSES, pca_dim=16, random_state=0, pca_whiten=True)
                    candidates.append(dict(endpoint=endpoint, prompt_mode=mode, model_label=model, token_site=site,
                        layer=int(layer), exact_four_cell_sample_alignment=False, population='own_available_original_states', **d, **c))
                winner=dict(select_ncc(candidates)); selected[endpoint].append(winner); records.extend(candidates)
                layer=winner['layer']; x=np.asarray(dataset.states_by_layer[layer], dtype=np.float32)
                dmask=frame['split'].eq('discovery').to_numpy(); cmask=frame['split'].eq('confirmation').to_numpy()
                scaler,pca,all_coordinates=fit_display_pca(x,dmask)
                zc=all_coordinates[cmask]; signs=np.ones(3)
                label='running' if endpoint=='running_index' else 'final'
                for i,(row,z) in enumerate(zip(frame.loc[cmask].to_dict('records'),zc)):
                    coords.append(dict(format=mode, model=model, endpoint=label, token_site=site,
                        layer_source_zero_based=layer, layer_display_one_based=layer+1, archived_row_index=i,
                        seed=int(row['seed']), gold_count=int(row['gold_count']), count_or_index=int(row['occurrence']),
                        pc1=float(z[0]), pc2=float(z[1]), pc3=float(z[2])))
                panels.append(dict(cell=cell,endpoint=label,token_site=site,layer_display_one_based=layer+1,
                    discovery_fit_rows=pop['discovery']['rows'], confirmation_display_rows=pop['confirmation']['rows'],
                    label_support=pop['confirmation']['label_support'], discovery_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                    discovery_axis_signs=signs.tolist()))
                name=f'{mode}_{model}_{endpoint}'
                frame.to_csv(output/'data'/f'{name}_metadata.csv',index=False)
                np.savez_compressed(output/'states'/f'{name}.npz',states=x, pca_components=pca.components_,
                    scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,pca_mean=pca.mean_,signs=signs)
                timing[name]=time.monotonic()-tick
                print(json.dumps(dict(cell=cell,endpoint=endpoint,rows=pop,layer=layer+1,
                    ncc=winner['confirmation_ncc_balanced_accuracy'],seconds=timing[name])),flush=True)
                del dataset,x,all_coordinates;gc.collect()
    for endpoint in selected:
        pd.DataFrame([r for r in records if r['endpoint']==endpoint]).to_csv(output/f'{endpoint}_candidate_metrics.csv',index=False)
    pd.DataFrame(coords).to_csv(output/'data/enumeration_pca_coordinates.csv',index=False)
    selection=dict(selected=selected,rule='Discovery NCC; logistic tie-break; earliest layer',
        confirmation_used_for_selection=False,source_sha256={f'{endpoint}_candidate_metrics.csv':_artifact(output/f'{endpoint}_candidate_metrics.csv')['sha256'] for endpoint in selected})
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    figs=[dict(name='enumeration_pca_'+mode.split('_')[-1],panels=[r for r in panels if r['cell'].startswith(mode+'|')]) for mode in ['enumeration_index','enumeration_bullet']]
    (output/'pca_manifest.json').write_text(json.dumps(dict(figures=figs,analysis_population='each_cell_own_available_states',
        original_transform='Discovery StandardScaler and unwhitened PCA3, randomized SVD, random_state=0; sklearn native signs; confirmation projection only.',
        points_subsampled=False),indent=2)+'\n')
    manifest=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,populations=supports,
        source_code=_artifact(Path(__file__)),command=sys.argv,seconds=time.monotonic()-start,timing=timing,
        versions={p:_package_version(p) for p in ['numpy','scipy','scikit-learn','pandas']},platform=platform.platform(),
        chronology='User-requested within-cell population amendment after common-support results; original confirmation reused; no new causal inference or matched cross-cell contrast.',
        n10_causal_layer_selection_changed=False,outputs={p.relative_to(output).as_posix():_artifact(p) for p in output.rglob('*') if p.is_file()})
    (output/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',seconds=manifest['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();analyze(args.run_root,args.output)
