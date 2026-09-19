"""Freeze NCC-only display selections before reading confirmation results."""
from pathlib import Path
import csv, json, hashlib, sys, importlib.util, time
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
R=ROOT/'realistic'
OLD=ROOT/'figures/cot_appendix_20260912/representations/data'
DATA=OUT/'data'
MODELS=['Qwen3-8B','Gemma4-E4B']
RULE='Maximum discovery seed/block-grouped NCC balanced accuracy; ties: logistic balanced accuracy, then earlier layer. Scores rounded to 12 decimals for floating-point ties.'
HASHES={}
def read(p):
    b=p.read_bytes();HASHES[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b.decode('utf-8-sig')
def csvread(p):return list(csv.DictReader(read(p).splitlines()))
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def csvsave(p,v):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
def choose(rows,ncc,log,layer):return max(rows,key=lambda r:(round(float(r[ncc]),12),round(float(r[log]),12),-int(r[layer])))
def main():
    start=time.perf_counter(); selected={};sweeps=[]
    for model in MODELS:
        selected[model]={}
        for ep in ['running_index','final_count']:
            path=R/f'reports/v5_dual_endpoint_geometry_full300/{model}/pca16_whiten/{ep}_candidate_metrics.csv'
            rr=[r for r in csvread(path) if r['mode']=='native_thinking' and r['analysis_group']==('all_traces' if ep=='running_index' else 'all_counts')]
            b=choose(rr,'discovery_oof_ncc_balanced_accuracy','discovery_oof_logistic_balanced_accuracy','layer')
            selected[model][ep]=int(b['layer'])
            sweeps.extend(dict(model=model,endpoint=ep,layer=int(r['layer']),ncc=float(r['discovery_oof_ncc_balanced_accuracy']),logistic=float(r['discovery_oof_logistic_balanced_accuracy'])) for r in rr)
        rr=[r for r in csvread(R/'work/domain_transfer_geometry/analysis/layer_selection_sweep.csv') if r['model_label']==model and r['mode']=='native_thinking']
        b=choose(rr,'cv_ncc_balanced_accuracy','cv_logistic_balanced_accuracy','layer')
        selected[model]['domain_answer']=int(b['layer'])
        sweeps.extend(dict(model=model,endpoint='domain_answer',layer=int(r['layer']),ncc=float(r['cv_ncc_balanced_accuracy']),logistic=float(r['cv_logistic_balanced_accuracy'])) for r in rr)
    for mode in ['nonthinking','thinking']:
        rr=csvread(ROOT/f'synthetic/work/v58_final/analysis/v58_unified_legacy_20260905/{mode}/geometry/clean_layer_metrics.csv')
        selected[mode]={}
        for ep in sorted({r['endpoint'] for r in rr}):
            a=[r for r in rr if r['endpoint']==ep and int(r['layer'])>=1]
            b=choose(a,'discovery_oof_ncc_balanced_accuracy','discovery_oof_logistic_balanced_accuracy','layer')
            selected[mode][ep]=int(b['layer'])
            sweeps.extend(dict(model=mode,endpoint=ep,layer=int(r['layer']),ncc=float(r['discovery_oof_ncc_balanced_accuracy']),logistic=float(r['discovery_oof_logistic_balanced_accuracy'])) for r in a)
    save(OUT/'selection.json',dict(rule=RULE,selected=selected,indexing='LLM zero-based; synthetic physical block 1--4; layer 0 is embedding and not a block',source_sha256=HASHES.copy()))
    csvsave(OUT/'discovery_sweep.csv',sweeps)
    print(json.dumps(selected),flush=True)
    # Selections above are persisted before any confirmation metrics are used.
    meta=json.loads(read(OLD/'metadata.json')); allcoords=json.loads(read(OLD/'canonical_all_layer_coordinates.json'))
    curves=csvread(OLD/'canonical_layerwise_readouts.csv');points=[]
    sys.path.insert(0,str(ROOT/'figures/cot_appendix_20260912/representations'))
    import extract_data as ext
    for model in MODELS:
        for ep in ['running_index','final_count']:
            layer=selected[model][ep]; m=meta['canonical']['models'][model][ep]
            coord=allcoords['models'][model][ep][str(layer)]
            m.update(selected_layer_zero_based=layer,selected_layer_display_one_based=layer+1,default_layer_zero_based=layer,evr=coord['evr'])
            m['selected_metrics_source_row']=next(r for r in csvread(R/f'reports/v5_dual_endpoint_geometry_full300/{model}/pca16_whiten/{ep}_candidate_metrics.csv') if r['mode']=='native_thinking' and r['analysis_group']==('all_traces' if ep=='running_index' else 'all_counts') and int(r['layer'])==layer)
            for r in curves:
                if r['model']==model and r['endpoint']==ep:r['is_selected_layer']=int(int(r['layer_zero_based'])==layer)
            points.extend(ext.flat_point(model,ep,layer,seed,k,gold,[x,y,z],split=split) for split,seed,k,x,y,z,gold in coord['points'])
    meta['canonical']['selection_rule']=RULE
    csvsave(DATA/'canonical_selected_pca.csv',points);csvsave(DATA/'canonical_layerwise_readouts.csv',curves)
    # Transfer geometry at changed layers is recomputed from cached captures.
    sys.path.insert(0,str(R/'scripts'))
    import analyze_niah_domain_endpoint_comparison as dom
    payload=json.loads(read(R/'reports/v5_domain_endpoint_comparison/geometry_payload.json'))
    for model in MODELS:
        layers=dict(running_index=selected[model]['running_index'],answer_token=selected[model]['domain_answer'])
        if any(payload['models'][model]['native_thinking'][e]['layer']!=l for e,l in layers.items()):
            transfer=dom.load_transfer_pair(R/f'work/domain_transfer_geometry/full/native/{model}/site_index.jsonl',mode='native_thinking',layers=layers)
            for ep,layer in layers.items():
                if payload['models'][model]['native_thinking'][ep]['layer']==layer:continue
                cm,cx,site=dom.load_city_selected(model=model,mode='native_thinking',endpoint=ep,layer=layer,
                    nonthinking_root=R/'work/nonthinking_v44_geometry_300_150_136_166_78',
                    native_running_root=R/'work/v5_geometry_full_panel/running',native_final_root=R/'work/v5_geometry_full_panel/final')
                tm,tx,ts=transfer[ep]
                p,_=dom.analyse_endpoint(cm,cx,tm,tx,model=model,mode='native_thinking',endpoint=ep,layer=layer,city_site=site,transfer_site=ts,seed=0)
                payload['models'][model]['native_thinking'][ep]=p
                print(model,ep,layer+1,p['metrics'],flush=True)
            del transfer
    pp=[];rr=[]
    for model in MODELS:
        for ep in ['running_index','answer_token']:
            p=payload['models'][model]['native_thinking'][ep];l=p['layer']
            meta['domain']['models'][model][ep]={k:v for k,v in p.items() if k!='points'}
            meta['domain']['models'][model][ep].update(selected_layer_zero_based=l,selected_layer_display_one_based=l+1)
            for d,m in p['metrics'].items():rr.append(dict(model=model,endpoint=ep,domain=d,layer_zero_based=l,layer_display_one_based=l+1,split='confirmation',states=m['states'],trajectories=m['trajectories'],ncc_balanced_accuracy=m['ncc_balanced_accuracy'],logistic_balanced_accuracy=m['logistic_balanced_accuracy'],chance_balanced_accuracy=.1))
            pp.extend(ext.flat_point(model,ep,l,pnt['seed'],pnt['count'],pnt['gold_count'],[pnt['x'],pnt['y'],pnt['z']],condition=pnt['domain'],domain=pnt['domain']) for pnt in p['points'])
    meta['domain']['layer_selection']={'running_index':'Inherited NCC-first canonical selection','answer_token':RULE}
    meta['domain']['limitations']=[s for s in meta['domain']['limitations'] if not s.startswith('Answer layers')]
    csvsave(DATA/'domain_selected_pca.csv',pp);csvsave(DATA/'domain_readouts.csv',rr)
    save(DATA/'metadata.json',meta);save(DATA/'domain_payload.json',payload)
    save(OUT/'prepare_audit.json',dict(status='PASS',source_sha256=HASHES,elapsed_seconds=time.perf_counter()-start,rule=RULE))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
