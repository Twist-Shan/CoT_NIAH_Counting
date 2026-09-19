"""Exercise all ablation report panels using explicitly synthetic layout data.

This never writes the research report or any experiment result directory.
"""
import json
from build_task_local_html import B, VERSION, Report, MODELS, TASKS

root=B/'runs'/VERSION/'no_results_smoke'
try:
    Report(root)
except RuntimeError:
    pass
else:
    raise AssertionError('Missing experiment results were accepted as complete')
report=Report(root,allow_pending=True)
report.complete=True
for task in TASKS:
    for model in MODELS:
        for mode,assay in [('nonthinking','broad'),('native_thinking','broad'),('native_thinking','targeted')]:
            sizes=([1,2,4,8,16,32,64,128] if assay=='broad' else [32,64,80,96,112,128]) if model==MODELS[0] else [1,2,4,6,8]
            for k in sizes:
                key=dict(task=task,model=model,mode=mode,assay=assay,k=k)
                selected=.9-.5*k/max(sizes)
                for metric,value in dict(clean=.9,selected=selected,random=.8,delta=.8-selected,clean_drop=.9-selected).items():
                    r=dict(key,metric=metric,n=100,seeds=10,mean=value,lower=value-.05,upper=value+.05)
                    report.summary.append(r);report.secondary.append(r.copy())
                for population in ['all_examples','clean_correct']:
                    report.tests.append(dict(key,population=population,p_holm=1.0))
assert len(report.summary)==370
parts=['<!doctype html><html><meta charset="utf-8"><title>SYNTHETIC LAYOUT TEST</title><style>',(B/'deployment/additional_report_style.css').read_text(),'</style><main><h1>排版检查：全部为合成测试数据，不是实验结果</h1>']
for mode,assay in [('nonthinking','broad'),('native_thinking','broad'),('native_thinking','targeted')]:
    parts.append(report.ablation_section(mode,assay))
parts.append('</main></html>')
out=B/'runs'/VERSION/'report_layout_smoke.html'
out.write_text(''.join(parts),encoding='utf-8')
assert len(report.figures)==6
print(json.dumps(dict(status='PASS',synthetic_layout=True,figures=6,path=str(out))))
