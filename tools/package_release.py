"""Build a deterministic source-only ZIP; omit Git metadata and run artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

from validate_repo import ROOT, source_files, validate, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "CoT_NIAH_Counting_anonymous.zip")
    args = parser.parse_args()
    result = validate()
    if result["status"] != "passed":
        raise SystemExit("Source validation failed: " + "; ".join(result["errors"]))
    manifest_errors = verify_manifest()
    if manifest_errors:
        raise SystemExit("Refresh the manifest with tools/validate_repo.py --write-manifest: " + "; ".join(manifest_errors))
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}. Use a new --output path.")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in sorted(source_files()):
            entry = zipfile.ZipInfo("CoT_NIAH_Counting/" + p.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, p.read_bytes())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"Created {output.name} ({output.stat().st_size:,} bytes)\nSHA256 {digest}")


if __name__ == "__main__":
    main()
