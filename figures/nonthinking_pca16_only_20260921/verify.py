"""Verify dimension-only change against historical PCA32 and new PCA16 results."""
from pathlib import Path
import csv,json,argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
import analyze as run

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
mapping=json.loads((OUT/'folds.json').read_text(encoding='utf-8'));checks=[]

def cv(x,y,seeds,role,dim):
    folds=np.array([mapping[str(s)] for s in seeds]); pn=np.zeros_like(y);pl=np.zeros_like(y)
    for k in range(5):
        a,b=run.transform(x[folds!=k],x[folds==k],role,dim)
        pn[folds==k],pl[folds==k]=run.predict(a,y[folds!=k],b,role)
    return balanced_accuracy_score(y,pn),balanced_accuracy_score(y,pl)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args();results=args.results.resolve()
assert (results/'metrics.csv').is_file()
with threadpool_limits(limits=2):
    if (results/'metrics.csv').exists():
        for r in csv.DictReader((results/'metrics.csv').open()):
            model,role=r['model'],r['role']
            if role=='answer_query_scan':
                with np.load(results/f'{model}_answer_query_original_selected.npz') as z:
                    x,y,s=z['states'].astype(np.float32),z['count'].astype(int),z['seed'].astype(int)
                scores=cv(x,y,s,role,16)
                np.testing.assert_allclose(scores[0],float(r['cv_ncc']),atol=1e-12)
                checks.append(dict(model=model,role=role,dimension=16,server_ncc_reproduced=True,logistic_local=scores[1],logistic_server=float(r['cv_logistic'])))
                print(checks[-1],flush=True)
                continue
            with np.load(results/f'{model}_{role}_selected.npz') as z:
                x,y,s=z['fit'],z['fit_labels'],z['fit_seeds'];t,ty,ts=z['held_out'],z['held_out_labels'],z['held_out_seeds']
            assert not set(s)&set(ts)
            a,b=run.transform(x,t,role,16);pn,pl=run.predict(a,y,b,role)
            actual=[balanced_accuracy_score(ty,pn),balanced_accuracy_score(ty,pl)]
            np.testing.assert_allclose(actual,[float(r['held_out_ncc']),float(r['held_out_logistic'])],atol=1e-12)
            scores=cv(x,y,s,role,16)
            np.testing.assert_allclose(scores,[float(r['cv_ncc']),float(r['cv_logistic'])],atol=1e-12)
            checks.append(dict(model=model,role=role,dimension=16,server_results_reproduced=True,disjoint_seeds=True))
            print(checks[-1],flush=True)
(results/'verification.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2))
