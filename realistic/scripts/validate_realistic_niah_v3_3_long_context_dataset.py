from __future__ import annotations

import argparse
import json

from realistic_niah_v3_3_long_context.integrity import validate_frozen_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the sealed V3.3 long-context dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--expected-seal-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_frozen_dataset(
                args.dataset_dir,
                expected_seal_sha256=args.expected_seal_sha256,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
