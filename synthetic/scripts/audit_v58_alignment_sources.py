"""Extract auditable report prose, excluding embedded plot/script payloads."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class ReportProse(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "svg"):
            self.skip.append(tag)
        if not self.skip and tag in ("h1", "h2", "h3", "h4", "p", "li", "tr", "figcaption", "div"):
            self.parts.append("\n")
        if not self.skip and tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
            return
        if tag in ("h1", "h2", "h3", "h4", "p", "li", "tr", "figcaption", "div"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def prose(self):
        return "\n".join(re.sub(r"\s+", " ", s).strip() for s in "".join(self.parts).splitlines() if s.strip()) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("reports", nargs="+", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for source in args.reports:
        parser = ReportProse()
        parser.feed(source.read_text(encoding="utf-8"))
        dest = args.output / (source.stem + ".txt")
        dest.write_text(parser.prose(), encoding="utf-8")
        sources.append({"source": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "prose": str(dest.resolve()), "lines": len(parser.prose().splitlines())})
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(sources, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
