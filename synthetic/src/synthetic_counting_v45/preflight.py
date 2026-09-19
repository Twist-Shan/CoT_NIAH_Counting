from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_counting_v44.preflight import run_preflight


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit v45 before optimization")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = run_preflight(args.run_dir, expected_version="v45")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
