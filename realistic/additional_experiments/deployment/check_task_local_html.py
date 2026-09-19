"""Verify report provenance, displayed numbers, and self-contained SVG structure."""
import csv
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from build_task_local_html import VERSION, NAMES

B=Path(__file__).resolve().parents[1]

class Structure(HTMLParser):
    def __init__(self): super().__init__(); self.ids=[]; self.tags={}; self.external=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs); self.tags[tag]=self.tags.get(tag,0)+1
        if 'id' in d:self.ids.append(d['id'])
        if tag in ['script','link','img'] and any(d.get(k,'').startswith(('https:','http:','//')) for k in ['src','href']):self.external.append(d)

def verify():
    p=B.parents[1]/'NiaH_Additional-tasks_report.html'; other=B.parent/'reports/NiaH_Additional-tasks_report.html'
    raw=p.read_bytes(); assert raw==other.read_bytes()
    s=raw.decode('utf-8'); structure=Structure();structure.feed(s)
    assert len(structure.ids)==len(set(structure.ids)), 'Duplicate HTML/SVG ids'
    assert not structure.external
    data=json.loads(re.search(r'<script type="application/json" id="report-data">(.*?)</script>',s,re.S)[1])
    assert data['version']==VERSION
    extra=data.get('target_category_broad') or {}
    native=data.get('native_full_span_broad') or {}
    expected_figures=(8 if data['complete'] else 2)+(2 if extra.get('complete') else 0)+(2 if native.get('complete') else 0)
    assert len(data['figures'])==structure.tags['figure']==structure.tags['svg']==structure.tags['figcaption']==expected_figures
    for rel,h in data['source_hashes'].items():assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==h
    with (B/'runs/report_refresh_20260908_v1/natural_summary.csv').open(encoding='utf-8') as f: expected=list(csv.DictReader(f))
    assert expected==data['natural']
    visible=re.sub(r'<script.*?</script>|<svg.*?</svg>|<pre.*?</pre>','',s,flags=re.S)
    for row in expected:
        if row['split'] not in ['all','confirmation']:continue
        assert f"{row['correct']}/{row['n']}" in visible
        assert f"{100*float(row['mean']):.2f} [{100*float(row['lower']):.2f}, {100*float(row['upper']):.2f}]" in visible
    if data['complete']:
        assert len(data['ablation'])==370
        assert '进度版：新消融尚未完成' not in s
        analysis=B/'runs'/VERSION/'downloaded/analysis'
        with (analysis/'summary.csv').open(encoding='utf-8') as f: summary=list(csv.DictReader(f))
        with (analysis/'hypothesis_tests.csv').open(encoding='utf-8') as f: tests=list(csv.DictReader(f))
        assert data['ablation']==summary
        policy=data['k_selection']
        assert policy['rule']=='argmax_random_minus_selected' and policy['population']=='all_examples'
        assert policy['interpretation']=='post_hoc_exploratory' and policy['data_split']=='confirmation'
        assert policy['holm_family']=='unchanged_full_grid' and policy['secondary_uses_main_k'] is True
        assert '事后选择' in visible and '逐点95%区间未校正选 K' in visible
        panel_keys={(r['task'],r['model'],r['mode'],r['assay']) for r in summary}
        assert len(policy['panels'])==len(panel_keys)==12
        assert {(r['task'],r['model'],r['mode'],r['assay']) for r in policy['panels']}==panel_keys
        for panel in policy['panels']:
            subset=[r for r in summary if all(r[key]==panel[key] for key in ['task','model','mode','assay'])]
            delta=[r for r in subset if r['metric']=='delta']
            peak=max(float(r['mean']) for r in delta)
            tied=sorted(int(r['k']) for r in delta if float(r['mean'])==peak)
            assert panel['k']==min(tied) and panel['tied_ks']==tied
            ident='best_'+panel['assay']+'_'+panel['mode']
            body=re.search(r'<div id="'+ident+r'">.*?<tbody>(.*?)</tbody>',s,re.S)[1]
            displayed=[[unescape(x) for x in re.findall(r'<td>(.*?)</td>',row,re.S)] for row in re.findall(r'<tr>(.*?)</tr>',body,re.S)]
            label=NAMES[panel['task']]+' / '+panel['model']
            matches=[r for r in displayed if r[0]==label]
            assert len(matches)==1
            row=matches[0]
            assert int(re.match(r'\d+',row[1])[0])==panel['k']
            assert ('并列' in row[1])==(len(tied)>1)
            metrics={r['metric']:r for r in subset if int(r['k'])==panel['k']}
            assert row[2]==metrics['delta']['n']+' / '+metrics['delta']['seeds']
            for idx,metric in [(3,'clean'),(4,'selected'),(5,'random')]:
                assert row[idx]==f"{100*float(metrics[metric]['mean']):.2f}"
            for idx,metric in [(6,'delta'),(7,'clean_drop')]:
                r=metrics[metric]
                assert row[idx]==f"{100*float(r['mean']):.2f} [{100*float(r['lower']):.2f}, {100*float(r['upper']):.2f}]"
            test=next(r for r in tests if all(r[key]==panel[key] for key in ['task','model','mode','assay']) and int(r['k'])==panel['k'] and r['population']=='all_examples')
            assert row[8]==f"{float(test['p_holm']):.4f}"
    else:
        assert data['ablation']==[] and '进度版：新消融尚未完成' in s
    assert 'aligned_final_summary.csv' not in s
    assert '第一条有效语义记录' in visible and '最多继续生成256 tokens' in visible
    extra_rows_verified=0
    if extra.get('complete'):
        assert extra['record_scope']=='question_target_category' and extra['points']==1300
        analysis=B/'runs'/extra['version']/'downloaded/analysis'
        with (analysis/'summary.csv').open(encoding='utf-8') as f:assert list(csv.DictReader(f))==extra['summary']
        with (analysis/'hypothesis_tests.csv').open(encoding='utf-8') as f:assert list(csv.DictReader(f))==extra['tests']
        body=re.search(r'<div id="target-category-best">.*?<tbody>(.*?)</tbody>',s,re.S)[1]
        shown=[[unescape(x) for x in re.findall(r'<td>(.*?)</td>',row,re.S)] for row in re.findall(r'<tr>(.*?)</tr>',body,re.S)]
        names={'all_examples':'合并问题','city_questions':'只看 city 问题','flower_questions':'只看 flower 问题'}
        for model,k in extra['best_k'].items():
            candidates=[r for r in extra['summary'] if r['model']==model and r['population']=='all_examples' and r['metric']=='delta']
            assert k==int(min(candidates,key=lambda r:(-float(r['mean']),int(r['k'])))['k'])
            for population,label in names.items():
                row=next(r for r in shown if r[0]==model and r[1]==label)
                assert int(row[2])==k
                metrics={r['metric']:r for r in extra['summary'] if r['model']==model and int(r['k'])==k and r['population']==population}
                assert int(row[3])==int(metrics['delta']['n'])
                for idx,metric in [(4,'clean'),(5,'selected'),(6,'random')]:assert row[idx]==f"{100*float(metrics[metric]['mean']):.2f}"
                for idx,metric in [(7,'delta'),(8,'delta_gain_vs_all_records')]:
                    r=metrics[metric];assert row[idx]==f"{100*float(r['mean']):.2f} [{100*float(r['lower']):.2f}, {100*float(r['upper']):.2f}]"
                if population=='all_examples':
                    test=next(t for t in extra['tests'] if t['model']==model and int(t['k'])==k and t['population']==population)
                    assert row[9]==f"{float(test['p_holm']):.4f}"
                else:
                    assert row[9]=='描述性'
                extra_rows_verified+=1
        assert len(shown)==extra_rows_verified==6
    native_verified=0
    if native.get('complete'):
        assert native['record_scope']=='registered_generated_full_record_spans'
        analysis=B/'runs'/native['version']/'downloaded/analysis'
        for name,key in [('summary.csv','summary'),('hypothesis_tests.csv','tests'),('coverage.csv','coverage')]:
            with (analysis/name).open(encoding='utf-8') as f:assert list(csv.DictReader(f))==native[key]
        body=re.search(r'<div id="native-full-span-best">.*?<tbody>(.*?)</tbody>',s,re.S)
        if body is None:body=re.search(r'<div id="native-full-span-best" class="tablewrap">.*?<tbody>(.*?)</tbody>',s,re.S)
        assert body is not None
        shown=[[unescape(x) for x in re.findall(r'<td>(.*?)</td>',row,re.S)] for row in re.findall(r'<tr>(.*?)</tr>',body[1],re.S)]
        for panel in native['best_k']:
            subset=[r for r in native['summary'] if r['model']==panel['model'] and r['task']==panel['task'] and r['population']=='all_examples']
            candidates=[r for r in subset if r['metric']=='delta']
            assert panel['k']==int(min(candidates,key=lambda r:(-float(r['mean']),int(r['k'])))['k'])
            row=next(r for r in shown if r[0]==NAMES[panel['task']]+' / '+panel['model'])
            metrics={r['metric']:r for r in subset if int(r['k'])==panel['k']}
            assert int(row[1])==panel['k'] and int(row[2])==int(metrics['delta']['n'])
            for i,metric in [(3,'clean'),(4,'selected'),(5,'random')]:assert row[i]==f"{100*float(metrics[metric]['mean']):.2f}"
            for i,metric in [(6,'delta'),(7,'delta_gain_vs_end_token')]:
                r=metrics[metric];assert row[i]==f"{100*float(r['mean']):.2f} [{100*float(r['lower']):.2f}, {100*float(r['upper']):.2f}]"
            t=next(r for r in native['tests'] if r['model']==panel['model'] and r['task']==panel['task'] and int(r['k'])==panel['k'] and r['population']=='all_examples')
            assert row[8]==f"{float(t['p_holm']):.4f}";native_verified+=1
        assert len(shown)==native_verified==4
    for href in re.findall(r'href="#([^"]+)"',s):assert href in structure.ids,href
    result=dict(status='PASS',complete=data['complete'],figures=len(data['figures']),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),best_k_panels_verified=12 if data['complete'] else 0,target_category_rows_verified=extra_rows_verified,native_full_span_rows_verified=native_verified)
    (B/'runs'/VERSION/'report_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    return result

if __name__=='__main__':verify()
