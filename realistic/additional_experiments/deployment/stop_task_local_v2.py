"""Stop only the old campaign explicitly superseded by the user's policy."""
import json
import os
from pathlib import Path
import signal
import time

root=Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_task_local_overlap_20260908_v2')
launch=json.loads((root/'launch.json').read_text())
pid=int(launch['supervisor_pid']);proc=Path('/proc')/str(pid)
if proc.exists() and not (proc/'stat').read_text().split(') ',1)[1].startswith('Z '):
    assert str(root/'launch.posix.sh').encode() in (proc/'cmdline').read_bytes()
    assert (proc/'cwd').resolve()==root.resolve()
    assert os.getpgid(pid)==pid
    os.killpg(pid,signal.SIGTERM)
result=dict(status='SUPERSEDED_BY_USER',supervisor_pid=pid,time=time.time(),reason='Random controls must exclude selected heads where feasible; allow overlap only in layers with insufficient unselected heads. Keep all existing results as history.')
(root/'superseded.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
