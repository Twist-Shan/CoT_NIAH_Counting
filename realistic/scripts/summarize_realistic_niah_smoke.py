from __future__ import annotations

import argparse
import json
from pathlib import Path

from realistic_niah.smoke_analysis import (
    load_request_rows,
    summarize_guarded_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen V2 native-thinking smoke test."
    )
    parser.add_argument("--requests", nargs="+", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = load_request_rows(args.requests)
    expected_models = None
    expected_requests_per_model = None
    if args.config:
        config = json.loads(
            Path(args.config).read_text(encoding="utf-8")
        )
        expected_models = config.get("models")
        expected_requests_per_model = config.get(
            "expected_requests_per_model"
        )
    summary = summarize_guarded_smoke(
        rows,
        expected_models=expected_models,
        expected_requests_per_model=expected_requests_per_model,
    )
    payload = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    print(payload, end="")


if __name__ == "__main__":
    main()
