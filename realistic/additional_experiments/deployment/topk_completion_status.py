from pathlib import Path
import json,datetime,subprocess
R=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_topk_completion_20260907_v1')
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(R),'exists':R.exists()}
if (R/'protocol.json').exists():
    p=json.loads((R/'protocol.json').read_text());out['expected_points']=p['expected_points']
    out['stages']={}
    for stage in ['canary','full']:
        out['stages'][stage]={}
        for model in ['Qwen3-8B','Gemma4-E4B']:
            base=R/stage/model
            out['stages'][stage][model]={'complete':(base/'complete.json').exists(),'points':len(list(base.glob('*/*/*/K*.json'))),'unavailable':len(list(base.glob('*/*/*/unavailable.json')))}
    lines=(R/'run.log').read_text(errors='replace').splitlines() if (R/'run.log').exists() else []
    out['tail']=lines[-6:];out['error_lines']=[s for s in lines if 'Traceback' in s or 'Error:' in s or 'AssertionError' in s][-10:]
    out['archive_ready']=(R/'results.tgz').exists()
out['processes']=subprocess.run(['pgrep','-af','run_topk_completion.py'],text=True,capture_output=True).stdout
print(json.dumps(out,ensure_ascii=False))
