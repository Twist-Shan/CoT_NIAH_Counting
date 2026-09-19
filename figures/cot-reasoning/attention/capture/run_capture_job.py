"""Run one bounded capture and persist its command, log, and final status."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout_seconds < 1:
        parser.error('A command and a positive timeout are required.')
    args.run_directory.mkdir(parents=True, exist_ok=False)
    record = {'status': 'RUNNING', 'pid': os.getpid(), 'command': command,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'timeout_seconds': args.timeout_seconds}
    state = args.run_directory / 'status.json'
    state.write_text(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    try:
        with (args.run_directory/'capture.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout_seconds, check=False)
        record.update(status='SUCCEEDED' if result.returncode == 0 else 'FAILED',
                      returncode=result.returncode)
    except subprocess.TimeoutExpired:
        record.update(status='TIMEOUT', returncode=124)
    except Exception as exc:
        record.update(status='FAILED', exception_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        record['seconds'] = time.monotonic()-started
        state.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
