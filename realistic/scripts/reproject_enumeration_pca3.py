"""Reproject frozen selected states; leave probes, layers and populations unchanged."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from realistic_niah_v6.own_state_representation import fit_display_pca

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main(source,output):
    tick=time.monotonic();manifest=read(source/'analysis_manifest.json')
    for name,record in manifest['outputs'].items():assert sha(source/name)==record['sha256'],name
    reference_path=ROOT/'scripts/build_niah_geometry_comparison_report.py'
    node=next(n for n in ast.parse(reference_path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='fit_dual_display_coordinates')
    reference_source=ast.unparse(node)
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    namespace=dict(np=np,StandardScaler=StandardScaler,PCA=PCA,require=require)
    exec('from __future__ import annotations\n'+reference_source,namespace)
    reference=namespace['fit_dual_display_coordinates']
    output.mkdir(parents=True,exist_ok=False);(output/'data').mkdir();(output/'states').mkdir()
    for name in ['selection.json','running_index_candidate_metrics.csv','final_count_candidate_metrics.csv']:shutil.copy2(source/name,output/name)
    metadata=read(source/'pca_manifest.json');oldcoords=pd.read_csv(source/'data/enumeration_pca_coordinates.csv')
    coordinates=[];comparisons=[]
    for figure in metadata['figures']:
        for panel in figure['panels']:
            mode,model=panel['cell'].split('|');endpoint='running_index' if panel['endpoint']=='running' else 'final_count'
            name=f'{mode}_{model}_{endpoint}';frame=pd.read_csv(source/'data'/f'{name}_metadata.csv')
            original=np.load(source/'states'/f'{name}.npz');x=original['states']
            dmask=frame['split'].eq('discovery').to_numpy();cmask=frame['split'].eq('confirmation').to_numpy()
            scaler,pca,z=fit_display_pca(x,dmask);zc=z[cmask];layer=panel['layer_display_one_based']-1
            # Execute the actual Thinking display function on the identical frozen states.
            ref=reference(SimpleNamespace(mode=mode,metadata=frame,states_by_layer={layer:x}))[str(layer)]
            refc=np.asarray([p[3:6] for p in ref['points'] if p[0]=='confirmation'])
            error=float(np.max(np.abs(zc-refc)));assert error<=5.1e-6,(name,error)
            assert np.max(np.abs(pca.explained_variance_ratio_-ref['evr']))<=5.1e-7
            assert np.array_equal(scaler.mean_,original['scaler_mean']) and np.array_equal(scaler.scale_,original['scaler_scale'])
            old=oldcoords[(oldcoords['format']==mode)&(oldcoords['model']==model)&(oldcoords['endpoint']==panel['endpoint'])]
            assert len(old)==len(zc)==panel['confirmation_display_rows']
            oldz=np.stack([old[f'pc{i}'].to_numpy() for i in (1,2,3)],axis=1)
            reconstructed=((x[cmask]-scaler.mean_)/scaler.scale_-pca.mean_)@pca.components_.T
            reconstruction_error=float(np.max(np.abs(zc-reconstructed)));assert reconstruction_error<1e-4
            angles=np.rad2deg(subspace_angles(original['pca_components'].T,pca.components_.T))
            # Sign-invariant pairwise geometry comparison measures the solver change.
            ix,jx=np.triu_indices(len(zc),k=1)
            olddist=np.linalg.norm(oldz[ix]-oldz[jx],axis=1);newdist=np.linalg.norm(zc[ix]-zc[jx],axis=1)
            comparisons.append(dict(cell=panel['cell'],endpoint=panel['endpoint'],rows=len(zc),layer=layer+1,
                old_axis_signs=original['signs'].tolist(),new_axis_signs=[1,1,1],
                max_subspace_angle_degrees=float(max(angles)),pairwise_distance_correlation=float(np.corrcoef(olddist,newdist)[0,1]),
                max_evr_difference=float(np.max(np.abs(pca.explained_variance_ratio_-panel['discovery_explained_variance_ratio']))),
                thinking_function_max_coordinate_error=error,reconstruction_max_error=reconstruction_error))
            panel.update(discovery_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),discovery_axis_signs=[1,1,1])
            records=old.to_dict('records')
            for record,point in zip(records,zc):
                for i in range(3):record[f'pc{i+1}']=float(point[i])
                coordinates.append(record)
            frame.to_csv(output/'data'/f'{name}_metadata.csv',index=False)
            np.savez_compressed(output/'states'/f'{name}.npz',states=x,pca_components=pca.components_,pca_mean=pca.mean_,
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,signs=np.ones(3))
    metadata['original_transform']='Discovery-only StandardScaler and unwhitened PCA3; randomized SVD, random_state=0; sklearn native axis signs, identical to Thinking.'
    write(output/'pca_manifest.json',metadata);pd.DataFrame(coordinates).to_csv(output/'data/enumeration_pca_coordinates.csv',index=False)
    result=dict(status='PASS',source=str(source),source_manifest_sha256=sha(source/'analysis_manifest.json'),
        reference_function='fit_dual_display_coordinates',reference_source_sha256=sha(reference_path),
        actual_reference_function_executed=True,probe_metrics_and_layer_files_identical=True,
        comparisons=comparisons,versions={p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','scipy']},
        utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick)
    for name in ['selection.json','running_index_candidate_metrics.csv','final_count_candidate_metrics.csv']:assert sha(source/name)==sha(output/name)
    write(output/'solver_audit.json',result)
    write(output/'analysis_manifest.json',dict(status='PASS',source_analysis=manifest,source_manifest_sha256=sha(source/'analysis_manifest.json'),
        source_code_sha256=sha(__file__),helper_source_sha256=sha(ROOT/'src/realistic_niah_v6/own_state_representation.py'),
        outputs={p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()},
        changed_scope='PCA3 only; probes, selected layers, populations and causal results unchanged',seconds=time.monotonic()-tick))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
