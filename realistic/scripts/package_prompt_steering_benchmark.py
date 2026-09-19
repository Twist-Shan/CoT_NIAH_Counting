"""Prepare an isolated, hashed code snapshot for the authorized SSH instance."""
import hashlib
import json
from pathlib import Path
import zipfile
import argparse

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=root/'outputs/prompt_steering_10k_20260910')
args=p.parse_args()
output=args.output
output.mkdir(parents=True,exist_ok=True)
paths=list((root/'src').rglob('*.py'))
paths += [root/p for p in ('scripts/run_prompt_ridge_steering_10k.py',
    'tests/test_prompt_ridge_steering_10k.py','configs/realistic_niah_v4.json',
    'plans/nonthinking-prompt-steering-10k-20260910.md',
    'plans/nonthinking-prompt-steering-n3-full-20260910.md',
    'scripts/prepare_prompt_steering_n3_seeds.py','scripts/analyze_prompt_steering_n3.py',
    'scripts/launch_prompt_steering_n3.sh')]
manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
with zipfile.ZipFile(output/'code.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,str(p.relative_to(root)))
    z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
(output/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(output/'code.zip'),files=len(paths),bytes=(output/'code.zip').stat().st_size)))
