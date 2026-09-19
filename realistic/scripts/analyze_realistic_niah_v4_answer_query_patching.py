from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah_v4.answer_query_patching import (
    analyze_answer_query_patching,
    audit_answer_query_patching,
    write_answer_query_analysis,
    write_answer_query_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and analyze the V4 answer-query residual-patching screen."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    args = parser.parse_args()

    audit = audit_answer_query_patching(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = write_answer_query_audit(
        audit, args.output_dir / "answer_query_patching_audit.json"
    )
    tables = analyze_answer_query_patching(
        args.run_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    table_paths = write_answer_query_analysis(tables, args.output_dir)
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "tables": {key: str(value) for key, value in table_paths.items()},
                "validated": bool(audit["validated"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
