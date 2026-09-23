"""Regression checks for source anonymity rules; run with unittest discovery."""
import unittest

from validate_repo import PATTERNS, has_cjk_or_fullwidth, has_han


class SourceAnonymityTests(unittest.TestCase):
    def test_fullwidth_punctuation_is_not_mistaken_for_plain_english(self):
        for codepoint in (0xFF1B, 0xFF5C, 0xFF08, 0xFF09, 0x3002, 0xFE15):
            with self.subTest(codepoint=codepoint):
                text = "English text" + chr(codepoint)
                self.assertFalse(has_han(text))
                self.assertTrue(has_cjk_or_fullwidth(text))

    def test_unicode_escapes_are_checked_as_well_as_literal_characters(self):
        slash = chr(92)
        for escaped in ("uFF1B", "U0000FF5C", "N{FULLWIDTH LEFT PARENTHESIS}",
                        "u4E00", "U00020000"):
            with self.subTest(escaped=escaped):
                self.assertTrue(has_cjk_or_fullwidth(slash + escaped))
        self.assertTrue(has_han(slash + "u4E00"))

    def test_ascii_source_can_describe_required_model_token_codepoints(self):
        self.assertFalse(has_cjk_or_fullwidth("delimiter = chr(0xFF5C)"))
        self.assertFalse(has_cjk_or_fullwidth("English; ASCII punctuation."))
        self.assertFalse(has_cjk_or_fullwidth(chr(92) + "N{UNKNOWN NAME}"))

    def test_private_source_revision_is_distinct_from_public_model_revision(self):
        digest = "a" * 40
        guard = PATTERNS["development-commit"]
        self.assertIsNotNone(guard.search('"inference_git_commit": "' + digest + '"'))
        self.assertIsNone(guard.search('"model_revision": "' + digest + '"'))
        self.assertIsNone(guard.search('"inference_git_commit": "redacted_for_anonymous_review"'))

    def test_geographic_timezone_is_distinct_from_utc(self):
        guard = PATTERNS["geographic-timezone"]
        self.assertIsNotNone(guard.search('"timezone": "' + "Region" + "/" + 'City"'))
        self.assertIsNone(guard.search('"timezone": "UTC"'))
        self.assertIsNone(guard.search('"timezone": "redacted_for_anonymous_review"'))


if __name__ == "__main__":
    unittest.main()
