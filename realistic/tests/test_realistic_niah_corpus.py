from __future__ import annotations

from realistic_niah.corpus import extract_html_essay, normalize_corpus_text


def test_html_extraction_uses_first_font_and_removes_nonvisible_content() -> None:
    html = """\
<html><body>
<nav>Navigation outside essay.</nav>
<font>
<h1>Essay title</h1>
<p>First paragraph &amp; visible text.</p>
<script>hidden_script()</script>
<p>Second paragraph contains enough substantive prose to represent an essay.
It continues with several sentences so the extractor does not mistake a tiny
navigation fragment for the article body. This is intentionally longer than
the fallback threshold used by the production corpus builder.</p>
<p>Third paragraph adds further visible material for a stable extraction
test and verifies that the first font element remains the selected body.</p>
</font>
<font>Footer translation links.</font>
</body></html>
"""

    text, strategy = extract_html_essay(html)

    assert strategy == "first_font_visible_text"
    assert "Essay title" in text
    assert "First paragraph & visible text." in text
    assert "Second paragraph contains enough substantive prose" in text
    assert "Navigation outside essay." not in text
    assert "hidden_script" not in text
    assert "Footer translation links." not in text


def test_corpus_normalization_is_stable() -> None:
    text = normalize_corpus_text(" Alpha\u00a0 beta. \r\n\r\n\r\n Gamma.  ")

    assert text == "Alpha beta.\n\nGamma.\n"
