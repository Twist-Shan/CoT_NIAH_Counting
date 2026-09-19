#!/usr/bin/env python3
"""Static structural checks for the self-contained non-thinking HTML report."""

from __future__ import annotations

import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.fragments: list[str] = []
        self.svg_count = 0
        self.svg_without_label = 0
        self.details_count = 0
        self.summary_count = 0
        self.figure_count = 0
        self.figcaption_count = 0
        self.class_counts: Counter[str] = Counter()
        self.external_assets: list[str] = []
        self.in_extension_table = False
        self.in_extension_body = False
        self.extension_rows = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        self.class_counts.update(attr.get("class", "").split())
        if identifier := attr.get("id"):
            self.ids.append(identifier)
        if tag == "a" and (href := attr.get("href", "")).startswith("#"):
            self.fragments.append(href[1:])
        if tag == "svg":
            self.svg_count += 1
            if not (attr.get("aria-label") or attr.get("aria-labelledby")):
                self.svg_without_label += 1
        if tag == "details":
            self.details_count += 1
        if tag == "summary":
            self.summary_count += 1
        if tag == "figure":
            self.figure_count += 1
        if tag == "figcaption":
            self.figcaption_count += 1
        if tag == "table" and "extension-audit" in attr.get("class", "").split():
            self.in_extension_table = True
        if tag == "tbody" and self.in_extension_table:
            self.in_extension_body = True
        if tag == "tr" and self.in_extension_body:
            self.extension_rows += 1
        if tag in {"img", "script", "link"}:
            uri = attr.get("src") or attr.get("href") or ""
            if uri and not uri.startswith(("data:", "#")):
                self.external_assets.append(uri)

    def handle_endtag(self, tag: str) -> None:
        if tag == "tbody" and self.in_extension_body:
            self.in_extension_body = False
        if tag == "table" and self.in_extension_table:
            self.in_extension_table = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    text = args.html.read_text(encoding="utf-8")
    report = ReportParser()
    report.feed(text)

    duplicate_ids = sorted({item for item in report.ids if report.ids.count(item) > 1})
    missing_fragments = sorted(set(report.fragments) - set(report.ids))
    stale_phrases = [
        phrase
        for phrase in (
            "Planned experiment 19",
            "Planned experiment 22",
            "Planned experiment 23",
            "experiment 19 has not yet been run",
            "尚未运行",
        )
        if phrase.casefold() in text.casefold()
    ]
    failures: list[str] = []
    if duplicate_ids:
        failures.append(f"duplicate ids: {duplicate_ids}")
    if missing_fragments:
        failures.append(f"missing fragment targets: {missing_fragments}")
    if report.extension_rows != 25:
        failures.append(f"extension audit rows={report.extension_rows}, expected 25")
    if report.svg_count == 0 or report.svg_without_label:
        failures.append(
            f"SVG accessibility: count={report.svg_count}, unlabeled={report.svg_without_label}"
        )
    if report.details_count != report.summary_count:
        failures.append(
            f"details/summary mismatch: {report.details_count}/{report.summary_count}"
        )
    if report.figure_count != report.figcaption_count:
        failures.append(
            f"figure/figcaption mismatch: {report.figure_count}/{report.figcaption_count}"
        )
    if report.class_counts["figure-primer"] != report.figure_count:
        failures.append(
            "figure-primer/figure mismatch: "
            f"{report.class_counts['figure-primer']}/{report.figure_count}"
        )
    for primer_label in (
        "<strong>这张图画什么。</strong>",
        "<strong>怎么读。</strong>",
        "<strong>一个例子。</strong>",
    ):
        if text.count(primer_label) != report.figure_count:
            failures.append(
                f"primer label {primer_label!r} count={text.count(primer_label)}, "
                f"expected {report.figure_count}"
            )
    expected_exact_classes = {
        "stage": 4,
        "protocol-step": 3,
        "chain-row": 5,
    }
    for class_name, expected in expected_exact_classes.items():
        actual = report.class_counts[class_name]
        if actual != expected:
            failures.append(f".{class_name} count={actual}, expected {expected}")
    expected_minimum_classes = {
        "chain-blueprint": 5,
        "evidence-triad": 4,
        "triad-step": 12,
        "chain-purpose": 6,
        "step-heading": 10,
        "experiment": 18,
        "conclusion-line": 2,
    }
    for class_name, minimum in expected_minimum_classes.items():
        actual = report.class_counts[class_name]
        if actual < minimum:
            failures.append(f".{class_name} count={actual}, expected >= {minimum}")
    if text.count("<strong>目的。</strong>") < 15:
        failures.append("fewer than 15 explicit experiment/section purposes")
    if text.count("目前结论") < 15:
        failures.append("fewer than 15 explicit current-conclusion statements")
    if report.external_assets:
        failures.append(f"external assets: {report.external_assets}")
    if stale_phrases:
        failures.append(f"stale phrases: {stale_phrases}")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        "PASS "
        f"ids={len(report.ids)} anchors={len(report.fragments)} "
        f"svgs={report.svg_count} details={report.details_count} "
        f"figures={report.figure_count} primers={report.class_counts['figure-primer']} "
        f"triads={report.class_counts['evidence-triad']} "
        f"extension_rows={report.extension_rows} external_assets=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
