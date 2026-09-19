from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from protocol import (make_case, user_prompt, parse_answer, matched_random_heads,
                      audit_case, text_hash, read_jsonl)
from diagnostics import native_endpoint, norm_matched_orthogonal, ridge_predict, trace_record_sites


def source(n=10, seed=1234):
    text, records = "", []
    for i in range(n):
        text += f"Ordinary paragraph {i}.\n"
        r = f"\u2029Excerpt:\nIn the 2024 city score audit, City{i} received a score of {70+i}.\nEnd excerpt.\u2029"
        records.append({"text": r, "city": f"City{i}", "score": 70+i,
                        "char_start": len(text), "char_end": len(text) + len(r)})
        text += r
    return {"active_needle_spans": records, "passage": text, "gold_count": n,
            "seed": seed, "stimulus_id": f"test_{seed}_{n}", "canonical_passage_tokens": 10000}


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        result = {"input_ids": list(text.encode("ascii"))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


class ProtocolTests(unittest.TestCase):
    def test_kth_edges_and_source_immutability(self):
        s = source()
        before = text_hash(s["passage"])
        for k in (1, 2, 10):
            c = make_case(s, "kth_needle", k, "confirmation")
            self.assertEqual(c["gold"], f"City{k-1}|{69+k}")
            self.assertEqual(c["passage"], s["passage"])
        self.assertEqual(text_hash(s["passage"]), before)
        for k in (0, 11):
            with self.assertRaises(ValueError):
                make_case(s, "kth_needle", k, "confirmation")

    def test_topic_offsets_counts_and_task_known_before_passage(self):
        for seed in (1234, 1235):
            for n in (1, 4, 10):
                c = make_case(source(seed=seed), "topic_count", n, "discovery")
                audit_case(c)
                self.assertEqual(c["total_records"], 10)
                self.assertEqual(sum(r["is_target"] for r in c["records"]), n)
                self.assertIn(c["target_topic"], user_prompt(c, "nonthinking").split("<passage>")[0])
                self.assertNotIn("Topic:", c["passage"])
                self.assertNotIn(c["target_topic"], c["passage"])

    def test_nested_topic_targets_and_identical_kth_passages(self):
        s = source()
        a, b = [make_case(s, "topic_count", k, "confirmation") for k in (2, 8)]
        self.assertLess({r["ordinal"] for r in a["records"] if r["is_target"]},
                        {r["ordinal"] for r in b["records"] if r["is_target"]})
        self.assertEqual(make_case(s, "kth_needle", 2, "confirmation")["passage_sha256"],
                         make_case(s, "kth_needle", 8, "confirmation")["passage_sha256"])

    def test_reference_native_prompt_byte_exact(self):
        from realistic_niah_v5.generation import build_v5_user_text
        c = make_case(source(4), "count_all", 4, "discovery")
        self.assertEqual(user_prompt(c, "native_thinking"), build_v5_user_text(c["passage"]))

    def test_reference_nonthinking_query_byte_exact(self):
        from realistic_niah_v4.prompts import V4_NUMERIC_QUERY_BLOCK
        c = make_case(source(4), "count_all", 4, "discovery")
        self.assertEqual(user_prompt(c, "nonthinking").split("</passage>\n\n")[1], V4_NUMERIC_QUERY_BLOCK)

    def test_output_parser_does_not_accept_extra_records_or_reasoning(self):
        c = make_case(source(), "topic_count", 2, "confirmation")
        self.assertTrue(parse_answer("Total:2", c)["correct"])
        self.assertFalse(parse_answer("There are 2.\nTotal:2", c)["parse_ok"])
        self.assertFalse(parse_answer("Total:2\nTotal:3", c)["parse_ok"])
        self.assertFalse(parse_answer("Total:-2", c)["parse_ok"])
        k = make_case(source(), "kth_needle", 2, "confirmation")
        self.assertTrue(parse_answer("Needle:City1|71", k)["correct"])
        self.assertFalse(parse_answer("Needle:City1|72", k)["correct"])

    def test_native_boundary_requires_closed_reasoning(self):
        tokenizer = CharacterTokenizer()
        for text, good in (("Total:2", False), ("Total:9\n</think>\nTotal:2", True),
                           ("reasoning\n<channel|>\nTotal:2", True),
                           ("reasoning\n</think>\nTotal:2\nTotal:3", False)):
            generated = {"completion_text_raw": text, "generated_token_ids": list(text.encode("ascii"))}
            found, reason = native_endpoint(tokenizer, "PROMPT", generated, "Total:")
            self.assertEqual(found is not None, good)
            if good:
                self.assertTrue(bytes(found[0].input_ids).decode("ascii").endswith("Total:"))
                self.assertIsNone(reason)

    def test_native_prefix_retokenization_drift_is_rejected(self):
        raw = "</think>\nTotal:2"
        result, reason = native_endpoint(CharacterTokenizer(), "PROMPT", {
            "completion_text_raw": raw, "generated_token_ids": [0] * len(raw)}, "Total:")
        self.assertIsNone(result)
        self.assertEqual(reason, "decoded_text_does_not_reproduce_original_ids")

    def test_trace_repetition_not_silently_selected(self):
        c = make_case(source(), "kth_needle", 2, "confirmation")
        raw = "1. City0: 70\n2. City1: 71\n1. City0: 70\n</think>"
        offsets = [(i, i+1) for i in range(len(raw))]
        sites, reason = trace_record_sites(c, raw, 0, len(raw), offsets)
        self.assertEqual(sites, [])
        self.assertEqual(reason, "repeated_or_out_of_order_natural_records")

    def test_head_controls_match_each_layer_and_are_reproducible(self):
        heads = [[2, 1], [2, 3], [4, 0]]
        a = matched_random_heads(heads, [8] * 5, 1)
        self.assertEqual(a, matched_random_heads(heads, [8] * 5, 1))
        self.assertEqual(Counter(l for l, h in heads), Counter(l for l, h in a))
        self.assertEqual(len(a), len(set(map(tuple, a))))
        with self.assertRaises(ValueError):
            matched_random_heads([[1, 8]], [8] * 5, 1)

    def test_residual_control_equal_norm_orthogonal_and_zero_guard(self):
        a, b = np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32)[::-1].copy()
        control, audit = norm_matched_orthogonal(a, b, 13)
        self.assertLess(abs(audit["delta_cosine"]), 1e-6)
        self.assertAlmostEqual(np.linalg.norm(control-a), np.linalg.norm(b-a), places=4)
        with self.assertRaises(ValueError):
            norm_matched_orthogonal(a, a, 13)

    def test_probe_heldout_linear_signal(self):
        x = np.arange(20).reshape(-1, 1)
        pred = ridge_predict(x, 2*x[:, 0]+1, [[3.5], [8.5]], alpha=1e-8)
        np.testing.assert_allclose(pred, [8, 18], atol=1e-5)
        with self.assertRaises(ValueError):
            ridge_predict([[np.nan], [1]], [0, 1], [[1]])

    def test_jsonl_handles_embedded_unicode_paragraph_separators(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cases.jsonl"
            c = make_case(source(), "topic_count", 2, "discovery")
            p.write_text(json.dumps(c, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl(p)[0], c)

    def test_bootstrap_weights_seeds_equally(self):
        from analyze import interval
        result = interval({1: [0., 0., 0., 0.], 2: [1.]}, 1000)
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["seeds"], 2)
        self.assertIsNone(interval({1: [1.]}, 100)["lower"])

    def test_count_centroid_basis_rank_and_orthonormality(self):
        from analyze import source_basis
        y = np.repeat([1, 2, 3, 4], 3)
        x = np.stack([y, y*y, y*y*y, np.ones(len(y))], axis=1).astype(float)
        center, b = source_basis(x, y)
        self.assertEqual(b.shape, (4, 3))
        np.testing.assert_allclose(b.T @ b, np.eye(3), atol=1e-6)
        self.assertEqual(center.shape, (4,))


if __name__ == "__main__":
    unittest.main()
