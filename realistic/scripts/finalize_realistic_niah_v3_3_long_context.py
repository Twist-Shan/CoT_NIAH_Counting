from __future__ import annotations

import argparse
import json

from realistic_niah_v3_3_long_context.runner import finalize_if_ready, finalize_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and combine both V3.3 long-context worker outputs."
    )
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--if-ready", action="store_true")
    args = parser.parse_args()
    result = (
        finalize_if_ready(stimuli_path=args.stimuli, run_root=args.run_root)
        if args.if_ready
        else finalize_run(stimuli_path=args.stimuli, run_root=args.run_root)
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
