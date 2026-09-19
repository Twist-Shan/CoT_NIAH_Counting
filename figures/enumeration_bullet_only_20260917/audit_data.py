"""Verify the new figures are an exact Bullet subset of the prior figures."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
PRIOR=ROOT/'figures/enumeration_midlayer_20260917'
REPO=ROOT/'realistic'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

style=module('previous_thinking_style',ROOT/'figures/thinking_appendix_style_20260913/build_figures.py');style.setup()
old=module('previous_enumeration_causal',PRIOR/'build_causal.py');old.style=style
captured={}
def capture(fig,name,records):
    captured[name]=[r for r in records if r['mode']=='enumeration_bullet'];plt.close(fig)
old.save=capture
base=REPO/'outputs/enumeration_alignment_20260916'
retrieve=read(base/'retrieve_audit_1550/audit.json')
old.retrieve(retrieve);old.head_scores(retrieve)
old.update(read(REPO/'outputs/enumeration_replay_midlayer_20260917/midlayer_audit/combined_update_audit.json'))
old.readout(read(REPO/'outputs/enumeration_followup_20260917/audit_v1/answer_audit.json'),read(base/'read_audit_v1/audit.json'))
checks={}
for oldname,rows in captured.items():
    name=oldname.replace('_aligned','');new=read(OUT/'data'/f'{name}.json')
    assert new==rows,name
    checks[name]=dict(rows=len(rows),exact_original_bullet_subset=True)

with (PRIOR/'data/readout_discovery_curves.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=[r for r in csv.DictReader(f) if r['format']=='enumeration_bullet']
for r in rows:
    for k in ['layer_display_one_based','states']:r[k]=int(r[k])
    r['balanced_accuracy']=float(r['balanced_accuracy'])
assert rows==read(OUT/'data/enumeration_representations.json')
checks['enumeration_representations']=dict(rows=len(rows),exact_original_bullet_subset=True)

pca_name='enumeration_pca_bullet'
assert sha(OUT/f'{pca_name}.pdf')==sha(PRIOR/f'{pca_name}.pdf')
with (PRIOR/'data/pca_coordinates.csv').open(encoding='utf-8-sig',newline='') as f:
    original=[r for r in csv.DictReader(f) if r['format']=='enumeration_bullet']
new=read(OUT/'data'/f'{pca_name}.json')
normalize=lambda rows:sorted(json.dumps(r,sort_keys=True) for r in rows)
assert normalize(new)==normalize(original)
checks[pca_name]=dict(rows=len(new),original_coordinates_unchanged=True,original_pdf_unchanged=True)

m=read(OUT/'manifest.json')
for path,digest in m['source_sha256'].items():assert sha(ROOT/path)==digest,path
for name,rec in m['figures'].items():
    for file,digest in rec['artifacts_sha256'].items():assert sha(OUT/file)==digest,file
result=dict(status='PASS',checks=checks,source_and_artifact_hashes_match=True,
            statement='Only the displayed format and layout changed; all Bullet points, intervals, curves, samples and PCA coordinates are preserved.')
(OUT/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
