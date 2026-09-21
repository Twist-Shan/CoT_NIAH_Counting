"""Render aligned readouts and dependent cue/noise figures in the existing style."""
from pathlib import Path
import csv, json, importlib.util, sys, hashlib, argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def module(name,path,source=None):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    if source is None: spec.loader.exec_module(m)
    else: exec(compile(source,str(path),'exec'),m.__dict__)
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True,help='Final update_answer.py output')
    p.add_argument('--legacy-figures',type=Path,required=True,help='Parent of nonthinking_ncc_selection_20260913 containing frozen selection/coordinates/domain payload')
    p.add_argument('--cue-report',type=Path,required=True,help='Original all-layer cue report containing PROMPT_GEOM')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    RESULTS=args.results.resolve();LEGACY=args.legacy_figures.resolve()
    selection=json.loads((RESULTS/'selection.json').read_text(encoding='utf-8'))['selected']
    old=json.loads((LEGACY/'nonthinking_ncc_selection_20260913/selection.json').read_text(encoding='utf-8'))['selected']
    for model in selection: selection[model]['domain_transfer']=old[model]['domain_transfer']
    (OUT/'selection.json').write_text(json.dumps({'selected':selection},indent=2))
    sweep=pd.read_csv(RESULTS/'layer_sweep.csv')
    # Keep the existing rendering code and layout; only substitute new measured data.
    for model,short in [('Qwen3-8B','qwen'),('Gemma4-E4B','gemma')]:
        for role,name in [('running_index',f'classification_prompt_20_{short}'),('answer_query_scan',f'classification_all_{short}')]:
            dst=OUT/'plot_inputs'/name; dst.mkdir(parents=True,exist_ok=True); rows=[]
            for r in sweep[(sweep.model==model)&(sweep.role==role)].itertuples():
                for a,k in [('nearest_centroid','ncc'),('logistic_l2','logistic')]:
                    rows.append(dict(layer=r.layer,algorithm=a,accuracy=getattr(r,k),rows=200,seeds=20,count_class_count=10,pca_components=16))
            pd.DataFrame(rows).to_csv(dst/'answer_classifier_metrics.csv',index=False)
    source=(ROOT/'figures/nonthinking_ncc_selection_20260913/build_readouts.py').read_text(encoding='utf-8')
    source=source.replace("DATA = ROOT / 'realistic/reports/v4_non-thinking_causal/v4_4_extension/classification'", "DATA = OUT / 'plot_inputs'")
    source=source.replace("PREVIOUS = ROOT / 'figures/nonthinking_count_readouts_20260912'", "PREVIOUS = OUT / 'plot_inputs'")
    source=source.replace("== 32 for r in selected", "== 16 for r in selected")
    start=source.index("    for model, layer, accuracy in [")
    end=source.index("    for ext in ['pdf'",start)
    source=source[:start]+source[end:]
    source=source.replace("'measurements_changed': {'running_index': 'refit on 20 discovery seeds', 'final_count': False}","'measurements_changed': {'running_index': True, 'final_count': True}")
    source=source.replace("'../nonthinking_count_readouts_20260912/running_index_20_audit.json'","'results/audit.json'")
    source=source.replace('PCA32','PCA16')
    source=source.replace("xytext=(0, 8)", "xytext=(0, -16) if source == 'prompt' and model == 'Gemma4-E4B' else (0, 8)")
    (OUT/'build_readouts.py').write_text(source)
    module('aligned_readouts',OUT/'build_readouts.py').main()

    # Use the CURRENT three-domain renderer, preserving its camera and styling.
    prior=LEGACY/'nonthinking_ncc_selection_20260913'
    with (prior/'pca_plot_data.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    rows=[r for r in rows if r['analysis']=='domain']
    metadata=json.loads((prior/'geometry_manifest.json').read_text(encoding='utf-8'))
    html=args.cue_report.read_text(encoding='utf-8')
    marker='const PROMPT_GEOM='
    cue,_=json.JSONDecoder().raw_decode(html[html.index(marker)+len(marker):])
    stats=[]
    for model,letter in [('Qwen3-8B','A'),('Gemma4-E4B','B')]:
        layer=selection[model]['running_index']['layer']; key=f'{model}|prompt_counter|{layer}'
        raw=cue['datasets'][key]
        next(p for p in metadata['panels'] if p['panel']==letter)['evr']=raw['evr_raw'][:3]
        for condition,offset in [('cue_present',4),('cue_absent',10)]:
            for r in raw['rows']:
                rows.append(dict(analysis='cue',model=model,site='needle_end',layer_source_zero_based=layer,layer_display_one_based=layer+1,condition=condition,seed=int(r[0]),count_or_index=int(r[1]),pc1=float(r[offset]),pc2=float(r[offset+1]),pc3=float(r[offset+2])))
        stats.append({**cue['statistics'][key],'model':model,'layer':layer+1})
    with (OUT/'cue_domain_inputs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pd.DataFrame(stats).to_csv(OUT/'cue_control_statistics.csv',index=False)
    (OUT/'geometry_inputs.json').write_text(json.dumps(metadata,indent=2))
    geometry_source=ROOT/'figures/nonthinking_appendix_revision_20260913/build_geometry.py'
    code=geometry_source.read_text(encoding='utf-8').replace("SOURCE = ROOT / 'figures/nonthinking_ncc_selection_20260913'",f"SOURCE = Path({str(prior)!r})")
    code=code.replace('str(p.relative_to(ROOT))','str(p)')
    geo=module('current_geometry',geometry_source,code)
    geo.OUT=OUT; geo.DATA=OUT/'cue_domain_inputs.csv'; geo.META=OUT/'geometry_inputs.json'; geo.metadata=metadata
    geo.main()

    # Recompute relative noise at the newly selected answer-query layers.
    noise=OUT/'noise'; noise.mkdir(exist_ok=True)
    verify=module('noise_verifier',ROOT/'realistic/scripts/verify_nonthinking_noise_alternatives.py')
    rng=np.random.default_rng(1234); d=rng.integers(0,20,(10000,20)); c=rng.integers(0,10,(10000,10))
    wd=np.stack([(d==i).sum(1) for i in range(20)],axis=1); wc=np.stack([(c==i).sum(1) for i in range(10)],axis=1)
    rows=[]; details=[]; summaries={}
    for model in selection:
        with np.load(RESULTS/(model+'_answer_query_scan_selected.npz')) as z:
            train=z['fit'].reshape(20,10,-1).astype(np.float64); test=z['held_out'].reshape(10,10,-1).astype(np.float64)
        layer=selection[model]['answer_query_scan']['layer']+1
        direction=np.diff(train.mean(0),axis=0); direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        q=np.stack([np.stack([test[:,k]@direction[k],test[:,k+1]@direction[k]],axis=-1) for k in range(9)],axis=1)
        sd=np.sqrt(q.var(0,ddof=1).mean(-1)); gap=np.diff(q.mean(0),axis=-1)[:,0]; relative=sd/np.abs(gap)
        bn,bg,br,_=verify.weighted_projection_bootstrap(train,test,wd,wc)
        for b in range(20):
            v=np.diff(train[d[b]].mean(0),axis=0); v/=np.linalg.norm(v,axis=-1,keepdims=True)
            te=test[c[b]]; p=np.stack([np.stack([te[:,k]@v[k],te[:,k+1]@v[k]],axis=-1) for k in range(9)],axis=1)
            np.testing.assert_allclose(np.sqrt(p.var(0,ddof=1).mean(-1)),bn[b],atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(np.diff(p.mean(0),axis=-1)[:,0],bg[b],atol=1e-9,rtol=1e-9)
        ci=np.quantile(br,[.025,.975],axis=0)
        summaries[model]={'layer':layer}
        for name,point,draws in [('relative_noise',relative,br),('pooled_sd',sd,bn),('mean_gap',np.abs(gap),np.abs(bg))]:
            summaries[model][name]={'high_over_low':float(verify.ratios(point[None])[0]),'ci95':np.quantile(verify.ratios(draws),[.025,.975]).tolist()}
        for k in range(9):
            rows.append(dict(model=model,layer=layer,primary_layer=True,N_lower=k+1,relative_noise=relative[k],joint_ci_low=ci[0,k],joint_ci_high=ci[1,k]))
            details.append(dict(model=model,layer=layer,method='local_centroid_direction',N_lower=k+1,relative_noise=relative[k],pooled_sd=sd[k],mean_gap=gap[k]))
    pd.DataFrame(rows).to_csv(noise/'joint_bootstrap_relative_noise.csv',index=False)
    pd.DataFrame(details).to_csv(noise/'adjacent_pair_metrics.csv',index=False)
    (noise/'summary.json').write_text(json.dumps(summaries,indent=2))
    (noise/'verification.json').write_text(json.dumps({'status':'PASS','checks':'20 bootstrap draws per model independently verified by literal resampling; 10000 total draws; fixed CV-selected layers'},indent=2))
    noiseplot=module('aligned_noise',ROOT/'figures/nonthinking_relative_noise_20260919/build.py')
    noiseplot.OUT=OUT; noiseplot.DATA=noise; noiseplot.SELECTION=OUT/'selection.json'; noiseplot.main()
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
