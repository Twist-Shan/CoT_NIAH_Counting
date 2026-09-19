"""Read-only diagnostics for the approved experiment instance."""
import json
import subprocess

commands = [
    ['lspci', '-nn'],
    ['nvidia-smi', 'nvlink', '--status'],
    ['sudo', '-n', 'journalctl', '-k', '-b', '--no-pager'],
    ['sudo', '-n', 'journalctl', '-u', 'nvidia-fabricmanager', '-b', '--no-pager'],
    ['systemctl', 'cat', 'nvidia-fabricmanager'],
]
for command in commands:
    p = subprocess.run(command, capture_output=True, text=True, timeout=30)
    lines = p.stdout.splitlines()
    if command[0] == 'lspci':
        lines = [s for s in lines if 'nvidia' in s.lower()]
    elif '-k' in command:
        lines = [s for s in lines if any(x in s.lower() for x in ('nvidia', 'nvrm', 'nvlink', 'nvswitch'))][-70:]
    print(json.dumps({'command': command, 'returncode': p.returncode, 'stdout': '\n'.join(lines), 'stderr': p.stderr}), flush=True)
