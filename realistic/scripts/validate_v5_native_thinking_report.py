#!/usr/bin/env python3
"""Validate structural invariants of the shipped Native-thinking HTML report."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


class AuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.fragments: list[str] = []
        self.svg_open = 0
        self.svg_close = 0
        self.figure_open = 0
        self.figure_close = 0
        self.figcaption_open = 0
        self.figure_title_open = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))
        href = attributes.get("href")
        if href and href.startswith("#"):
            self.fragments.append(href[1:])
        if tag == "svg":
            self.svg_open += 1
        if tag == "figure":
            self.figure_open += 1
        if tag == "figcaption":
            self.figcaption_open += 1
        if tag == "h3" and "figure-title" in str(attributes.get("class", "")).split():
            self.figure_title_open += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "svg":
            self.svg_close += 1
        if tag == "figure":
            self.figure_close += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    text = args.report.read_text(encoding="utf-8")
    audit = AuditParser()
    audit.feed(text)
    duplicate_ids = sorted(
        {value for value in audit.ids if audit.ids.count(value) > 1}
    )
    missing_fragments = sorted(set(audit.fragments) - set(audit.ids))
    if duplicate_ids:
        raise ValueError(f"Duplicate HTML ids: {duplicate_ids}")
    if missing_fragments:
        raise ValueError(f"Missing fragment targets: {missing_fragments}")
    if audit.svg_open != audit.svg_close:
        raise ValueError("Unbalanced SVG roots")
    if not (
        audit.figure_open
        == audit.figure_close
        == audit.figcaption_open
        == audit.figure_title_open
    ):
        raise ValueError(
            "Every figure must have one visible .figure-title and one figcaption: "
            f"open={audit.figure_open}, close={audit.figure_close}, "
            f"titles={audit.figure_title_open}, captions={audit.figcaption_open}"
        )
    section_order = (
        "summary",
        "baseline",
        "representation",
        "formation",
        "retrieval",
        "write",
        "answer",
        "integrated-chain",
        "ledger",
        "extension-audit",
        "limitations",
        "appendix",
    )
    section_offsets = [text.index(f'<section id="{value}">') for value in section_order]
    if section_offsets != sorted(section_offsets):
        raise ValueError("Main sections no longer follow the Non-thinking-aligned order")
    for required in (
        "本文主张（仅限 Qwen3-8B 的自然 no-index trace）",
        "J.1 显式 index positive control",
        "Qwen L19、Gemma L16",
        "Gemma 尚无对应的自然 no-index 因果结果",
        "图 5c · 同一 L16 state intervention",
        "simulatively confirmed†",
        "可诱发的机制能力，不是自然使用",
        "实验前置 · Parser 与因果设计合同",
        "strict_eligible_no_explicit_count_cue",
        "first_generated_known_city_ordinal",
        "计划内结果均已落盘",
        "图 6d · Answer-query full-state patch 的逐层 donor-count adoption",
        "registered existing-split extension",
        "answer_query_v3",
        "realistic_niah_v5_native_thinking_restructured_v14",
    ):
        if required not in text:
            raise ValueError(f"Required report statement is absent: {required}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.report.read_bytes()).hexdigest()
    if digest != manifest["output_sha256"]:
        raise ValueError("Report hash does not match its manifest")
    print(
        json.dumps(
            {
                "status": "PASS",
                "bytes": len(text.encode("utf-8")),
                "id_count": len(audit.ids),
                "fragment_link_count": len(audit.fragments),
                "svg_count": audit.svg_open,
                "figure_count": audit.figure_open,
                "sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
