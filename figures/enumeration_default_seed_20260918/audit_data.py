"""Independently compare every plotted value with its audited source statistic."""
from pathlib import Path
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPORTS = ROOT / 'realistic/outputs/enumeration_default_seed_20260918/reports'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def cell(report, model):
    matches = [c for c in report['cells'] if c['model'] == model]
    assert len(matches) == 1
    return matches[0]
def equal(row, statistic):
    assert abs(row['estimate'] - statistic.get('estimate', statistic.get('value'))) < 1e-12
    assert all(abs(a-b) < 1e-12 for a,b in zip(row['ci95'], statistic['ci95']))

reports = {k: read(REPORTS/'audit_v2'/f'{k}_audit.json') for k in ['retrieve','read','answer','update']}
assert all(r['status'] == 'PASS' for r in reports.values())
checked = {}
for name in ['enumeration_head_scores','enumeration_retrieve','enumeration_update','enumeration_readout']:
    rows = read(OUT/'data'/f'{name}.json')
    assert rows
    for row in rows:
        model = row['model']
        if name == 'enumeration_head_scores':
            source = read(REPORTS/'retrieve/jobs/localize'/model/'enumeration_bullet/ranking.json')[row['rank']-1]
            assert (row['layer'],row['head'],row['score']) == (source['layer']+1,source['head']+1,source['score'])
            assert row['selected'] == (row['rank'] <= (128 if model == 'Qwen3-8B' else 6))
        elif name == 'enumeration_retrieve':
            dose, = [v for v in cell(reports['retrieve'],model)['dose_response'] if v['dose_k'] == row['k']]
            equal(row,dose['metrics'][row['metric']][row['condition']])
        elif name == 'enumeration_update':
            c = cell(reports['update'],model)
            g, = [v for v in c['groups'] if v['scope'] == row['scope'] and v['direction'] == row['direction'] and v['donor_k'] == 'all']
            hop = row.get('hop',1); condition = row.get('condition','donor_to_receiver')
            value = g['conditions'][condition]['continuation'][str(hop)]
            equal(row,value['conditional'])
            assert row['numerator'] == value['successes'] and row['denominator'] == value['conditional_eligible']
            assert row.get('layer',row.get('layer_one_based')) == c['layer_one_based']
        elif row['metric'] == 'answer_donor_adoption':
            layer, = [v for v in cell(reports['answer'],model)['layerwise'] if v['layer_one_based'] == row['layer']]
            equal(row,layer['conditions'][row['condition']]['donor_adoption'])
        else:
            equal(row,cell(reports['read'],model)['conditions'][row['condition']]['accuracy'])
    for ext in ['pdf','png','svg']:
        f = OUT/f'{name}.{ext}'
        assert hashlib.sha256(f.read_bytes()).hexdigest() == read(OUT/'manifest.json')['figures'][name]['artifacts_sha256'][f.name]
    paper = ROOT/'runs/paper_figures/figures/enumeration_bullet'/f'{name}.pdf'
    assert paper.read_bytes() == (OUT/f'{name}.pdf').read_bytes()
    checked[name] = len(rows)
out = dict(status='PASS',plotted_records_verified=checked,total_records=sum(checked.values()))
(OUT/'data_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(out)
