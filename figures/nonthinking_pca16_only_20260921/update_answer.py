"""Analyze final-count readability using packed answer-query states.

Classification and relative noise use separate input sets.
"""
from pathlib import Path
import argparse,csv,json,shutil,time,hashlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
import analyze as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--folds',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.perf_counter();mapping=json.loads(args.folds.read_text(encoding='utf-8'));expected=json.loads(args.expected.read_text(encoding='utf-8'));hashes={};checks=[]
    selection=json.loads((args.previous/'selection.json').read_text(encoding='utf-8'))
    sweep=[r for r in csv.DictReader((args.previous/'layer_sweep.csv').open()) if r['role']=='running_index']
    metrics=[r for r in csv.DictReader((args.previous/'metrics.csv').open()) if r['role']=='running_index']
    for model,oldlayer in [('Qwen3-8B',24),('Gemma4-E4B',29)]:
        folder=args.source/'runs/v4_4_counter_channel_20260806/packed/layers'
        rows=[]
        for layer in range(36 if model=='Qwen3-8B' else 42):
            path=folder/f'{model}__answer_query__L{layer:02d}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z: x=z['states'].astype(np.float32); y=z['count'].astype(int); seeds=z['seed'].astype(int)
            assert len(y)==200 and set(seeds)==set(a.FIT)
            folds=np.array([mapping[str(s)] for s in seeds]);scores={}
            for dim in ([32,16] if layer==oldlayer else [16]):
                pn=np.zeros_like(y);pl=pn.copy()
                for k in range(5):
                    tr,te=a.transform(x[folds!=k],x[folds==k],'answer_query_scan',dim)
                    pn[folds==k],pl[folds==k]=a.predict(tr,y[folds!=k],te,'answer_query_scan')
                scores[dim]=[float(balanced_accuracy_score(y,pn)),float(balanced_accuracy_score(y,pl))]
            if 32 in scores:
                np.testing.assert_allclose(scores[32][0],expected[model][0],atol=1e-12)
                checks.append(dict(model=model,layer=layer+1,dimension32=scores[32],historical_expected=expected[model],ncc_historical_reproduced=True,logistic_delta=scores[32][1]-expected[model][1]))
            r=dict(model=model,role='answer_query_scan',layer=layer,ncc=scores[16][0],logistic=scores[16][1]);rows.append(r);sweep.append(r)
            if (layer+1)%6==0:print(model,'layer',layer+1,flush=True)
        chosen=max(rows,key=lambda r:(round(r['ncc'],12),round(r['logistic'],12),-r['layer']))
        selection['selected'][model]['answer_query_scan']=chosen
        a.save(out/'selection.json',selection)
        path=folder/f"{model}__answer_query__L{chosen['layer']:02d}.npz"
        shutil.copy2(path,out/f'{model}_answer_query_original_selected.npz')
        # Preserve the relative-noise assay's original source, now evaluated at the selected layer.
        base=args.source/'runs/v4_4_geometry_comparison_20260816'/model/'numeric/representation/answer_query_all_layers_v1'
        tr,y,ss=a.load(base,'answer_query_scan',a.FIT,hashes,layer=chosen['layer'])
        te,ty,ts=a.load(base,'answer_query_scan',a.TEST,hashes,layer=chosen['layer'])
        np.savez_compressed(out/f'{model}_answer_query_scan_selected.npz',fit=tr,held_out=te,fit_labels=y,held_out_labels=ty,fit_seeds=ss,held_out_seeds=ts,layer=chosen['layer'],usage='Relative-noise assay only; classifier CV uses original packed states')
        metrics.append(dict(model=model,role='answer_query_scan',layer=chosen['layer'],layer_one_based=chosen['layer']+1,cv_ncc=chosen['ncc'],cv_logistic=chosen['logistic'],held_out_ncc='',held_out_logistic='',held_out_ridge_r2='',centroid_pca3_variance=''))
        shutil.copy2(args.previous/f'{model}_running_index_selected.npz',out/f'{model}_running_index_selected.npz')
        print('SELECTED',chosen,flush=True)
    a.write_csv(out/'metrics.csv',metrics);a.write_csv(out/'layer_sweep.csv',sweep)
    shutil.copy2(args.previous/'protocol.json',out/'protocol.json')
    a.save(out/'audit.json',dict(status='PASS',original_answer_sources=hashes,baseline32_checks=checks,running_index_audit=str(args.previous/'audit.json'),elapsed_seconds=time.perf_counter()-start,scope='PCA16 count readouts; relative noise uses its specified input set.'))

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
