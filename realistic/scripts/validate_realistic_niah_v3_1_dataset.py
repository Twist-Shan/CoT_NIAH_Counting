from __future__ import annotations

import argparse
import json

from realistic_niah_v3_1.integrity import validate_frozen_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen V3.1 dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--record-source-revision", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_frozen_dataset(
                args.dataset_dir,
                require_source_revision=not args.record_source_revision,
                record_source_revision=args.record_source_revision,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
