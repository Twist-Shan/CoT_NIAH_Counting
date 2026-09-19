#!/usr/bin/env python3
"""Run the two experiment families with their original relative-path conventions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the paper entry points.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and working directory without executing.")
    parser.add_argument("experiment", nargs="?", help="Entry-point name, or realistic/synthetic followed by a script path.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    registry = json.loads((ROOT / "experiments.json").read_text(encoding="utf-8"))
    if args.list:
        for name, entry in registry.items():
            print(f"{name:24} {entry['paper']}: {entry['description']}")
        return 0
    if not args.experiment:
        parser.print_help()
        return 0
    extra = args.arguments
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.experiment in ("realistic", "synthetic"):
        family = args.experiment
        if not extra:
            parser.error(f"{family} requires a relative .py script path")
        cwd = ROOT / family
        target = (cwd / extra[0]).resolve()
        if not target.is_relative_to(cwd) or target.suffix != ".py" or not target.is_file():
            parser.error("Script must be an existing .py file inside the selected component")
        command = [sys.executable, str(target), *extra[1:]]
    else:
        if args.experiment not in registry:
            parser.error(f"Unknown entry point: {args.experiment}; use --list")
        entry = registry[args.experiment]
        family = entry["family"]
        cwd = ROOT / family
        if "module" in entry:
            command = [sys.executable, "-m", entry["module"], *extra]
        else:
            command = [sys.executable, str(cwd / entry["script"]), *extra]
    env = os.environ.copy()
    # Keep each family's scripts isolated: several historical scripts have the
    # same names. Dependencies shared within a family remain importable.
    env["PYTHONPATH"] = os.pathsep.join([str(cwd / "src"), str(cwd / "scripts"), str(cwd)])
    env["PYTHONUTF8"] = "1"
    if args.dry_run:
        print(json.dumps({"cwd": str(cwd), "command": command}, indent=2))
        return 0
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
