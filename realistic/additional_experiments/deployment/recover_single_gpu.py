"""Reload an idle single-GPU VM driver with NVLink disabled for this boot."""
import json
from pathlib import Path
import subprocess
import time

root = Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5_additional_experiments_task_local_overlap_20260908_v2')
history = []

def run(command, check=True):
    p = subprocess.run(command, capture_output=True, text=True, timeout=45)
    row = dict(command=command, code=p.returncode, stdout=p.stdout, stderr=p.stderr, time=time.time())
    history.append(row)
    (root/'gpu_recovery.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(row), flush=True)
    if check and p.returncode:
        raise RuntimeError(f'Failed: {command}')
    return p

p = run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
assert len(p.stdout.strip().splitlines()) == 1 and int(p.stdout.strip()) < 1024
assert not run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip()
devices = [s for s in run(['lspci', '-nn']).stdout.splitlines() if 'NVIDIA' in s]
assert len(devices) == 1 and '3D controller' in devices[0]
assert 'NVreg_NvLinkDisable' in run(['modinfo', '-p', 'nvidia']).stdout
loaded = {s.split()[0] for s in Path('/proc/modules').read_text().splitlines()}
modules = [s for s in ['nvidia_peermem', 'nvidia_drm', 'nvidia_modeset', 'nvidia_uvm', 'nvidia'] if s in loaded]
run(['sudo', '-n', 'systemctl', 'stop', 'nvidia-fabricmanager'])
run(['sudo', '-n', 'systemctl', 'stop', 'nvidia-persistenced'])
try:
    holders = run(['sudo', '-n', 'fuser', '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm'], check=False)
    assert not holders.stdout.strip(), 'A process still holds a GPU device; refusing to reload'
    for module in modules:
        run(['sudo', '-n', 'modprobe', '-r', module])
    run(['sudo', '-n', 'modprobe', 'nvidia', 'NVreg_NvLinkDisable=1'])
finally:
    for module in reversed(modules):
        run(['sudo', '-n', 'modprobe', module], check=False)
    run(['sudo', '-n', 'systemctl', 'start', 'nvidia-persistenced'], check=False)
run(['outputs/external/lambda_nfs_CoT-Native-thinking-v5_venv_v6_20260828_bin_python', '-c',
     'import torch; assert torch.cuda.is_available(); x=torch.ones(32, device="cuda"); assert x.sum().item()==32; print(torch.__version__, torch.cuda.get_device_name())'])
