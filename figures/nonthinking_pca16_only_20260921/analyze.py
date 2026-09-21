"""Analyze Non-thinking running-index readouts with PCA16.

Run update_answer.py next for the original packed final-count inputs.

Read immutable cached residual states; fit each transform inside seed-grouped CV.
Keep the original five seed folds, save layer choices before reading held-out states.
The analysis reads cached residual states.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time, platform
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import NearestCentroid
from threadpoolctl import threadpool_limits

MODELS = ['Qwen3-8B', 'Gemma4-E4B']
FIT = list(range(1234, 1254))
TEST = list(range(1254, 1264))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def transform(a, b, role, dim=16):
    if role=='running_index':
        pca=PCA(n_components=dim,svd_solver='randomized',random_state=20260806)
        a=pca.fit_transform(a); b=pca.transform(b)
        scale=StandardScaler().fit(a)
        return scale.transform(a),scale.transform(b)
    scale=StandardScaler().fit(a)
    a=scale.transform(a); b=scale.transform(b)
    pca=PCA(n_components=dim,random_state=442)
    return pca.fit_transform(a),pca.transform(b)

def predict(a, y, b, role):
    ncc=NearestCentroid(shrink_threshold=None if role=='running_index' else .1).fit(a,y).predict(b)
    logistic = LogisticRegression(C=1,solver='lbfgs',max_iter=2000 if role=='running_index' else 4000,random_state=20260806 if role=='running_index' else None).fit(a, y).predict(b)
    return ncc, logistic

def load(base, role, seeds, sources, layer=None):
    index = base/'capture_index.jsonl'
    sources[str(index)] = hashlib.sha256(index.read_bytes()).hexdigest()
    rows = [json.loads(s) for s in index.read_text(encoding='utf-8').splitlines() if s.strip()]
    rows = [r for r in rows if r['design_variant']=='v4.4' and int(r['seed']) in seeds and (role != 'running_index' or int(r['count'])==10)]
    rows.sort(key=lambda r:(int(r['seed']),int(r['count'])))
    expected = len(seeds) if role=='running_index' else len(seeds)*10
    assert len(rows)==expected, (base,role,len(rows),expected)
    arr, ys, ss = [], [], []
    for r in rows:
        assert r['split']==('discovery' if int(r['seed']) in FIT else 'confirmation')
        path=base/r['shard_path']; sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:
            indices=z['layer_indices'].tolist()
            a=z['span_end' if role=='running_index' else 'query_states'].astype(np.float32)
            if layer is not None: a=a[indices.index(layer)]
            elif role=='running_index': a=np.moveaxis(a,1,0)
            if role!='running_index': a=a[None]
            arr.append(a)
        labels=list(range(1,11)) if role=='running_index' else [int(r['count'])]
        ys.extend(labels); ss.extend([int(r['seed'])]*len(labels))
    return np.concatenate(arr), np.array(ys), np.array(ss)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--folds',type=Path,required=True)
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'audit.json').exists(), 'Do not overwrite completed analysis'
    start=time.perf_counter(); hashes={}; sweeps=[]; selections={}; metrics=[]; predictions=[]; foldlog=[]
    mapping=json.loads(args.folds.read_text(encoding='utf-8'))
    save(out/'protocol.json',dict(analysis='Non-thinking PCA16 readouts',pca_components=16,whiten=False,running_index='PCA(randomized,20260806) then StandardScaler; NCC no shrinkage; logistic C1/max_iter2000',answer_query='StandardScaler then PCA(auto,442); NCC shrinkage0.1; logistic C1/max_iter4000',fit_seeds=FIT,held_out_seeds=TEST,fold_map=mapping,selection='Maximum pooled out-of-fold NCC balanced accuracy; ties logistic then earlier layer',evaluation='Fitting and held-out seeds are disjoint; layer selection uses fitting seeds only.'))
    for model in MODELS:
        selections[model]={}; cache={}
        for role in ['running_index']:
            root='run_20260731_v4_numeric_presentation_v3' if role=='running_index' else 'v4_4_geometry_comparison_20260816'
            suffix='capture' if role=='running_index' else 'answer_query_all_layers_v1'
            base=args.source/'runs'/root/model/'numeric/representation'/suffix
            x,y,seeds=load(base,role,FIT,hashes)
            assert x.shape[0]==200 and np.isfinite(x).all()
            folds=np.array([mapping[str(s)] for s in seeds]); assert set(folds)==set(range(5))
            rows=[]
            for layer in range(x.shape[1]):
                pn=np.zeros_like(y); pl=np.zeros_like(y)
                for fold in range(5):
                    train=folds!=fold; test=~train
                    assert not set(seeds[train]) & set(seeds[test])
                    a,b=transform(x[train,layer],x[test,layer],role); pn[test],pl[test]=predict(a,y[train],b,role)
                    if layer==0: foldlog.append(dict(model=model,role=role,fold=fold,train_seeds=sorted(set(seeds[train].tolist())),test_seeds=sorted(set(seeds[test].tolist()))))
                row=dict(model=model,role=role,layer=layer,ncc=float(balanced_accuracy_score(y,pn)),logistic=float(balanced_accuracy_score(y,pl)))
                rows.append(row); sweeps.append(row)
                if (layer+1)%6==0: print(model,role,'layer',layer+1,'of',x.shape[1],flush=True)
            selected=max(rows,key=lambda r:(round(r['ncc'],12),round(r['logistic'],12),-r['layer']))
            selections[model][role]=selected
            cache[role]=(base,x,y,seeds)
            print('SELECTED',json.dumps(selected),flush=True)
        save(out/'selection.json',dict(selected=selections)); write_csv(out/'layer_sweep.csv',sweeps)
        # Running-index layer selection is on disk before any held-out capture is loaded.
        for role,(base,x,y,seeds) in cache.items():
            layer=selections[model][role]['layer']; train=x[:,layer]
            test,ty,ts=load(base,role,TEST,hashes,layer=layer)
            assert test.shape[0]==100 and np.isfinite(test).all()
            a,b=transform(train,test,role); pn,pl=predict(a,y,b,role)
            ridge=Ridge(alpha=1).fit(a,y).predict(b)
            c=np.stack([train[y==k].mean(0) for k in range(1,11)]).astype(np.float64)
            sv=np.linalg.svd(c-c.mean(0),compute_uv=False)
            result=dict(model=model,role=role,layer=layer,layer_one_based=layer+1,cv_ncc=selections[model][role]['ncc'],cv_logistic=selections[model][role]['logistic'],held_out_ncc=float(balanced_accuracy_score(ty,pn)),held_out_logistic=float(balanced_accuracy_score(ty,pl)),held_out_ridge_r2=float(r2_score(ty,ridge)),centroid_pca3_variance=float(np.square(sv[:3]).sum()/np.square(sv).sum()))
            metrics.append(result)
            for i in range(len(ty)): predictions.append(dict(model=model,role=role,seed=int(ts[i]),label=int(ty[i]),ncc=int(pn[i]),logistic=int(pl[i]),ridge=float(ridge[i])))
            np.savez_compressed(out/(model+'_'+role+'_selected.npz'),fit=train,held_out=test,fit_labels=y,held_out_labels=ty,fit_seeds=seeds,held_out_seeds=ts,layer=layer)
            print('HELD_OUT',json.dumps(result),flush=True)
        del cache
    write_csv(out/'metrics.csv',metrics); write_csv(out/'predictions.csv',predictions)
    save(out/'audit.json',dict(status='PASS',numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version(),source_sha256=hashes,folds=foldlog,elapsed_seconds=time.perf_counter()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_inference=False))

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
