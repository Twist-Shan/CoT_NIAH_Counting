from __future__ import annotations

import argparse
import json

from realistic_niah_v3_1.resume import audit_resume_manifests, parse_resume_commits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit existing V3.1 manifests before a checkpoint resume."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume-from-commits", default="")
    args = parser.parse_args()
    result = audit_resume_manifests(
        args.run_root,
        current_commit=args.expected_commit,
        allowed_commits=parse_resume_commits(args.resume_from_commits),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
