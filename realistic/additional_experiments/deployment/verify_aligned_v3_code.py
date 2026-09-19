import json,hashlib
from pathlib import Path
r=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_aligned_transfer_20260906_v3')
c=json.loads((r/'full/Gemma4-E4B/contract.json').read_text())
for name,h in c['code'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
print(json.dumps(dict(status='PASS',code_files=len(c['code']))))
