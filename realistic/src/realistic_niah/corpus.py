from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


BLOCK_TAGS = frozenset(
    {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "p",
        "table",
        "td",
        "tr",
    }
)
SKIP_TAGS = frozenset({"script", "style"})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n"
        for value in values
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


class _FirstFontTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.started = False
        self.finished = False
        self.font_depth = 0
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if self.finished:
            return
        if not self.started:
            if tag == "font":
                self.started = True
                self.font_depth = 1
            return
        if tag == "font":
            self.font_depth += 1
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self.started or self.finished:
            return
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "font":
            self.font_depth -= 1
            if self.font_depth == 0:
                self.finished = True

    def handle_data(self, data: str) -> None:
        if (
            self.started
            and not self.finished
            and self.skip_depth == 0
        ):
            self.parts.append(data)


class _VisibleBodyTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_body = False
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if tag == "body":
            self.in_body = True
        if self.in_body and tag in SKIP_TAGS:
            self.skip_depth += 1
        if self.in_body and self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.in_body and tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if self.in_body and self.skip_depth == 0 and tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "body":
            self.in_body = False

    def handle_data(self, data: str) -> None:
        if self.in_body and self.skip_depth == 0:
            self.parts.append(data)


def normalize_corpus_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.split("\n")
    ]
    normalized: list[str] = []
    previous_blank = True
    for line in lines:
        if line:
            normalized.append(line)
            previous_blank = False
        elif not previous_blank:
            normalized.append("")
            previous_blank = True
    return "\n".join(normalized).strip() + "\n"


def extract_html_essay(html: str) -> tuple[str, str]:
    first_font = _FirstFontTextParser()
    first_font.feed(html)
    candidate = normalize_corpus_text("".join(first_font.parts))
    if len(candidate) >= 256:
        return candidate, "first_font_visible_text"

    visible_body = _VisibleBodyTextParser()
    visible_body.feed(html)
    fallback = normalize_corpus_text("".join(visible_body.parts))
    if len(fallback) < 256:
        raise ValueError("HTML page did not contain enough visible essay text")
    return fallback, "body_visible_text_fallback"


def _decode_download(payload: bytes, charset: str | None) -> str:
    encodings = [charset, "utf-8", "cp1252"]
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace")


def _download_one(
    url: str,
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    requested_url = (
        "https://" + url[len("http://") :]
        if url.startswith("http://www.paulgraham.com/")
        else url
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                requested_url,
                headers={
                    "User-Agent": (
                        "Realistic-CoT-NiaH-Corpus-Sync/2.0 "
                        "(research benchmark)"
                    )
                },
            )
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                payload = response.read()
                charset = response.headers.get_content_charset()
                final_url = response.geturl()
                content_type = response.headers.get_content_type()
            return {
                "source_url": url,
                "requested_url": requested_url,
                "final_url": final_url,
                "download_bytes": len(payload),
                "content_type": content_type,
                "charset": charset,
                "payload": payload,
                "attempts": attempt,
            }
        except Exception as exc:  # pragma: no cover - network-specific
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.25 * 2 ** (attempt - 1)))
    assert last_error is not None
    return {
        "source_url": url,
        "requested_url": requested_url,
        "status": "download_failed",
        "error": f"{type(last_error).__name__}: {last_error}",
        "attempts": retries,
    }


def _safe_output_name(url: str, used: set[str]) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or "essay"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "essay"
    prefix = "pg" if "paulgraham.com" in parsed.netloc.lower() else "repo"
    candidate = f"{prefix}_{stem}.txt"
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    candidate = f"{prefix}_{stem}_{suffix}.txt"
    used.add(candidate)
    return candidate


def _prepare_download(download: dict[str, Any]) -> dict[str, Any]:
    if download.get("status") == "download_failed":
        return download
    payload = download.pop("payload")
    text = _decode_download(payload, download.get("charset"))
    source_url = str(download["source_url"])
    is_html = (
        source_url.lower().endswith((".html", ".htm"))
        or str(download.get("content_type", "")).lower() == "text/html"
    )
    try:
        if is_html:
            normalized, extraction = extract_html_essay(text)
        else:
            normalized = normalize_corpus_text(text)
            extraction = "plain_text"
    except Exception as exc:
        return download | {
            "status": "extraction_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload_utf8 = normalized.encode("utf-8")
    return download | {
        "status": "prepared",
        "extraction": extraction,
        "normalized_text": normalized,
        "normalized_bytes": len(payload_utf8),
        "sha256": _sha256_bytes(payload_utf8),
    }


def sync_full_paul_graham_corpus(
    *,
    url_list_path: str | Path,
    output_dir: str | Path,
    min_file_bytes: int = 5 * 1024,
    workers: int = 8,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    minimum_included_files: int = 100,
    maximum_failures: int = 50,
) -> dict[str, Any]:
    url_file = Path(url_list_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Corpus output directory must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    url_payload = url_file.read_bytes()
    urls: list[str] = []
    seen_urls: set[str] = set()
    duplicate_urls = 0
    for raw in url_payload.decode("utf-8").splitlines():
        url = raw.strip()
        if not url:
            continue
        if url in seen_urls:
            duplicate_urls += 1
            continue
        seen_urls.add(url)
        urls.append(url)
    if not urls:
        raise ValueError(f"No URLs found in {url_file}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        downloads = list(
            executor.map(
                lambda url: _download_one(
                    url,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                ),
                urls,
            )
        )
    prepared = [_prepare_download(download) for download in downloads]

    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    seen_content: dict[str, str] = {}
    for item in prepared:
        row = {key: value for key, value in item.items() if key != "normalized_text"}
        if item.get("status") != "prepared":
            rows.append(row)
            continue
        if int(item["normalized_bytes"]) < min_file_bytes:
            rows.append(row | {"status": "excluded_too_small"})
            continue
        sha256 = str(item["sha256"])
        if sha256 in seen_content:
            rows.append(
                row
                | {
                    "status": "excluded_duplicate_content",
                    "duplicate_of": seen_content[sha256],
                }
            )
            continue
        output_name = _safe_output_name(str(item["source_url"]), used_names)
        _atomic_write(
            output / output_name,
            str(item["normalized_text"]).encode("utf-8"),
        )
        seen_content[sha256] = output_name
        rows.append(row | {"status": "included", "filename": output_name})

    included = [row for row in rows if row["status"] == "included"]
    failures = [
        row
        for row in rows
        if row["status"] in {"download_failed", "extraction_failed"}
    ]
    audit_errors: list[str] = []
    for row in included:
        path = output / str(row["filename"])
        payload = path.read_bytes()
        if len(payload) != int(row["normalized_bytes"]):
            audit_errors.append(f"byte mismatch: {path.name}")
        if _sha256_bytes(payload) != row["sha256"]:
            audit_errors.append(f"SHA256 mismatch: {path.name}")

    corpus_lines = [
        f"{row['sha256']}  {row['filename']}  {row['normalized_bytes']}"
        for row in sorted(included, key=lambda value: str(value["filename"]))
    ]
    corpus_payload = ("\n".join(corpus_lines) + "\n").encode("utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": "realistic_niah_haystack_corpus_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_protocol": (
            "NVIDIA/RULER PaulGrahamEssays_URLs full-list sync"
        ),
        "source_list_path": str(url_file),
        "source_list_sha256": _sha256_bytes(url_payload),
        "urls_total": len(urls),
        "duplicate_urls_skipped": duplicate_urls,
        "minimum_file_bytes": min_file_bytes,
        "status_counts": status_counts,
        "included_files": len(included),
        "included_total_bytes": sum(
            int(row["normalized_bytes"]) for row in included
        ),
        "content_unique": len({row["sha256"] for row in included})
        == len(included),
        "corpus_payload_sha256": _sha256_bytes(corpus_payload),
        "audit_passed": not audit_errors,
        "audit_errors": audit_errors,
        "software": {
            "python": sys.version,
            "implementation": "stdlib urllib + HTMLParser",
        },
    }
    _atomic_write(output / "corpus_index.jsonl", _jsonl_bytes(rows))
    _atomic_write(output / "corpus_manifest.json", _json_bytes(manifest))
    _atomic_write(output / "SHA256SUMS", corpus_payload)

    if audit_errors:
        raise RuntimeError(f"Corpus integrity audit failed: {audit_errors}")
    if len(included) < minimum_included_files:
        raise RuntimeError(
            f"Only {len(included)} corpus files met inclusion criteria; "
            f"minimum is {minimum_included_files}"
        )
    if len(failures) > maximum_failures:
        raise RuntimeError(
            f"{len(failures)} corpus URLs failed; maximum is {maximum_failures}"
        )
    return manifest
