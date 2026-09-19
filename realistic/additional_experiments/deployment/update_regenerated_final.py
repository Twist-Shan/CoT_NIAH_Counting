import json,csv,shutil,re
from pathlib import Path
b=Path(__file__).resolve().parents[1];p=b/'aligned_final_summary.csv';backup=b/'aligned_final_summary_before_sdpa_regeneration.csv'
if not backup.exists():shutil.copy2(p,backup)
with backup.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f));fields=list(rows[0])
a=json.loads((b/'runs/aligned_gemma_kth_sdpa_20260907_v1/analysis/audit.json').read_text(encoding='utf-8'))
new={(x['mode'],x['assay']):x for x in a['summary']}
for i,r in enumerate(rows):
 if r['model']=='Gemma4-E4B' and r['task']=='kth':rows[i]=dict(new[r['mode'],r['assay']],version='sdpa_regenerated_v1')
with p.open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
assert len(rows)==12
report=b/'GEMMA_KTH_SDPA_RESULTS_20260907.md'
for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):assert (b/link).exists(),link
p=b/'STATUS.json';s=json.loads(p.read_text(encoding='utf-8'));s['gemma_kth_sdpa_regeneration'].update(status='COMPLETE_WITH_DOCUMENTED_CLEAN_DIFFERENCE',report=report.name,archive_sha256='c059252656d7d72908a64c62b0fb9822e9902b9f054fee77f58d8039273f39d9',code_hash_check={'files':235,'status':'PASS'},target_rescore={'cases':94,'arms':470,'prediction_changes':1})
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Updated 3 Gemma kth rows; 9 other rows preserved; report links PASS')
