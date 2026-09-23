"""Validate the source-only release without importing optional GPU packages."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "work", "outputs", "output", "tmp",
             "reports", "results", "cache", ".cache", ".venv", "artifacts", "colab_results", "checkpoints", "build", "dist"}
SKIP_FILES = {".preparation-marker"}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "token": re.compile(r"\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{24,}\b"),
    "user-home": re.compile(r"(?i)(?:[A-Z]:[/\\]+Users[/\\]+[^/\\\s]+|/(?:home|Users)/[^/\s\"']+/)"),
    "remote-login": re.compile(r"\b[\w.-]+@(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Basic/extension/compatibility Han blocks, including supplementary planes.
HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
              (0x20000, 0x2EE5F), (0x2F800, 0x2FA1F), (0x30000, 0x3347F))
UNICODE_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8}))")


def has_han(text: str) -> bool:
    def decode(match):
        value = int(match.group(1) or match.group(2), 16)
        return chr(value) if value <= 0x10FFFF else match.group(0)
    text = UNICODE_ESCAPE.sub(decode, text)
    return any(any(low <= ord(char) <= high for low, high in HAN_RANGES)
               for char in text if ord(char) > 127)


def source_files():
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".venv"))
        for name in sorted(files):
            path = Path(directory) / name
            if name not in SKIP_FILES and path.suffix not in {".pyc", ".zip", ".log"}:
                yield path


def validate() -> dict:
    errors = []
    counts = {"files": 0, "python": 0, "json": 0, "bytes": 0}
    for p in source_files():
        rel = p.relative_to(ROOT).as_posix()
        counts["files"] += 1
        counts["bytes"] += p.stat().st_size
        if p.is_symlink():
            errors.append(f"Symlink is not allowed in source snapshot: {rel}")
        if p.suffix in {".pt", ".pth", ".safetensors", ".npz", ".npy", ".pem", ".key"} or p.name.startswith(".env"):
            errors.append(f"Unexpected artifact/credential file: {rel}")
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary file: {rel}")
            continue
        if has_han(rel) or has_han(text):
            errors.append(f"Han characters in source, path or escaped text: {rel}")
        try:
            if p.suffix == ".py":
                ast.parse(text, filename=rel)
                counts["python"] += 1
            if p.suffix == ".json":
                json.loads(text)
                counts["json"] += 1
        except (SyntaxError, ValueError) as exc:
            errors.append(f"Invalid syntax: {rel}: {exc}")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"Potential {name} in {rel}")
    registry = json.loads((ROOT / "experiments.json").read_text(encoding="utf-8"))
    for name, item in registry.items():
        component = ROOT / item["family"]
        target = component / item["script"] if "script" in item else component / "src" / (item["module"].replace(".", "/") + ".py")
        if not target.is_file():
            errors.append(f"Missing entry point: {name}")
    # Check local links in the new reader-facing documents.
    documents = [ROOT / "README.md", ROOT / "LICENSE_STATUS.md", ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "figures/README.md", *(ROOT / "docs").glob("*.md")]
    for p in documents:
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                if not (p.parent / target.split("#")[0]).exists():
                    errors.append(f"Broken link in {p.name}: {target}")
    return {"status": "passed" if not errors else "failed", **counts, "errors": errors}


def verify_manifest() -> list[str]:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256 is missing"]
    expected = {}
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT) or relative in expected:
            errors.append(f"Invalid manifest entry: {relative}")
            continue
        expected[relative] = checksum
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            errors.append(f"Checksum mismatch: {relative}")
    actual = {p.relative_to(ROOT).as_posix() for p in source_files() if p != manifest}
    if actual != set(expected):
        errors.append(f"Manifest file set differs: {sorted(actual.symmetric_difference(expected))}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Refresh the anonymous file checksums after validation.")
    parser.add_argument("--check-manifest", action="store_true", help="Verify all anonymous source file hashes.")
    args = parser.parse_args()
    result = validate()
    if result["status"] == "passed" and args.write_manifest:
        manifest = ROOT / "MANIFEST.sha256"
        lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in source_files() if p != manifest]
        manifest.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8", newline="\n")
    if args.check_manifest:
        result["errors"].extend(verify_manifest())
        result["status"] = "failed" if result["errors"] else "passed"
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
