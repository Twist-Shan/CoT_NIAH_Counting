#!/usr/bin/env python3
"""Compare paired Qwen states immediately before and after explicit rank markers."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v5.cross_mode_geometry import CLASSES
from realistic_niah_v5.trace_stratified_geometry import confirmation_metrics, grouped_discovery_cv_metrics

SITES = ("pre_marker", "post_marker")

def load(root: Path):
    index = [json.loads(x) for x in (root / "capture_index.jsonl").read_text(encoding="utf-8").splitlines() if x]
    meta, chunks, layers = [], [], None
    for row in index:
        manifest = json.loads((root / row["manifest_path"]).read_text(encoding="utf-8"))
        with np.load(root / row["states_path"], allow_pickle=False) as z:
            now = z["layer_indices"].astype(int); states = z["site_states"]
        if layers is None: layers = now
        if not np.array_equal(layers, now): raise ValueError("layer mismatch")
        chunks.append(states)
        for event in manifest["events"]:
            meta.append({"request_id": row["request_id"], "stimulus_id": row["request_id"],
                         "split": row["split"], "seed": int(row["seed"]),
                         "gold_count": int(row["gold_count"]), "occurrence": int(event["occurrence"]),
                         "grammar_class": event["grammar_class"],
                         "primary_full_chain_event": bool(event["primary_full_chain_event"])})
    return np.concatenate(chunks), pd.DataFrame(meta), layers

def main():
    p=argparse.ArgumentParser(); p.add_argument("--capture",type=Path,default=ROOT/"work/v5_qwen_post_marker_geometry")
    p.add_argument("--output",type=Path,default=ROOT/"reports/v5_qwen_post_marker_geometry"); a=p.parse_args()
    states, meta, layers = load(a.capture); candidates=[]; selected=[]; payload={}
    scopes={"all_rank_before":np.ones(len(meta),bool),
            "adjacent_rank_before_city":meta.grammar_class.eq("adjacent_rank_before_city").to_numpy()}
    for scope, mask in scopes.items():
      m=meta.loc[mask].reset_index(drop=True); payload[scope]={}
      support=m.groupby(["split","occurrence"]).size().rename("n").reset_index()
      if any(set(support.loc[support.split.eq(s),"occurrence"].astype(int)) != set(CLASSES) for s in ("discovery","confirmation")):
        continue
      for si,site in enumerate(SITES):
        rows=[]
        for li,layer in enumerate(layers):
          x=states[mask,si,li].astype(np.float32)
          d=grouped_discovery_cv_metrics(x,m,CLASSES,pca_dim=16,random_state=0,folds=5,pca_whiten=True)
          c=confirmation_metrics(x,m,CLASSES,pca_dim=16,random_state=0,pca_whiten=True)
          row={"scope":scope,"site":site,"layer":int(layer),"rows":len(m),**d,**c}; rows.append(row); candidates.append(row)
        win=pd.DataFrame(rows).sort_values(["discovery_selection_score","discovery_oof_ncc_balanced_accuracy","layer"],ascending=[False,False,True]).iloc[0].to_dict(); selected.append(win)
        li=int(np.where(layers==int(win["layer"]))[0][0]); x=states[mask,si,li].astype(np.float32)
        disc=m.split.eq("discovery").to_numpy(); pc=PCA(3,random_state=0).fit(x[disc]); xyz=pc.transform(x)
        payload[scope][site]={"layer":int(win["layer"]),"evr":pc.explained_variance_ratio_.tolist(),
          "points":[{"x":float(v[0]),"y":float(v[1]),"z":float(v[2]),"split":str(r.split),"seed":int(r.seed),"occurrence":int(r.occurrence),"grammar_class":str(r.grammar_class)} for v,r in zip(xyz,m.itertuples(index=False))],
          "support":support.to_dict("records")}
        print(scope,site,f"L{int(win['layer'])}",f"conf={win['confirmation_logistic_balanced_accuracy']:.3f}/{win['confirmation_ncc_balanced_accuracy']:.3f}")
    a.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(candidates).to_csv(a.output/"site_layer_candidates.csv",index=False)
    pd.DataFrame(selected).to_csv(a.output/"site_selected.csv",index=False)
    (a.output/"geometry_payload.json").write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    (a.output/"audit.json").write_text(json.dumps({"scope":"paired replayed events","sites":list(SITES),"selection":"discovery-only grouped OOF, confirmation frozen","explicit_marker_warning":"post_marker follows a visible ordinal/rank phrase; decoding gain is marker-conditioned retrieval evidence, not an implicit-counter estimate"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
