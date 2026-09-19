from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

GITHUB_API_TREE = (
    "https://api.github.com/repos/gkamradt/LLMTest_NeedleInAHaystack/contents/"
    "needlehaystack/PaulGrahamEssays"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/"
    "needlehaystack/PaulGrahamEssays"
)
MIN_FILE_BYTES = 5 * 1024


def _get_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-generation-bot"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-generation-bot"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def _normalize_haystack_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def sync(out_dir: Path, min_file_bytes: int = MIN_FILE_BYTES) -> dict[str, int | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = _get_json(GITHUB_API_TREE)
    txt_items = [x for x in listing if x.get("type") == "file" and str(x.get("name", "")).endswith(".txt")]

    index_path = out_dir / "index.jsonl"
    rows: list[str] = []
    written = 0
    removed_small = 0
    for item in txt_items:
        name = item["name"]
        raw_url = f"{RAW_BASE}/{name}"
        text = _normalize_haystack_text(_get_text(raw_url))
        text_bytes = len(text.encode("utf-8"))
        target = out_dir / name
        if text_bytes < min_file_bytes:
            if target.exists():
                target.unlink()
            removed_small += 1
            continue
        target.write_text(text, encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append(json.dumps({"filename": name, "source_url": raw_url, "sha256": sha, "bytes": text_bytes}))
        written += 1

    index_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return {"files_written": written, "files_removed_too_small": removed_small, "index_path": str(index_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/haystacks/paul_graham")
    p.add_argument("--min-file-bytes", type=int, default=MIN_FILE_BYTES)
    args = p.parse_args()
    result = sync(Path(args.out_dir), min_file_bytes=args.min_file_bytes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
