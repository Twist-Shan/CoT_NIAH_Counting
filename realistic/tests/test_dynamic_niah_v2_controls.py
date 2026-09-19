import json
import re
from pathlib import Path

import pytest

from dataset_generation import dynamic_niah_v2
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    TokenizerAdapter,
    build_prediction_messages,
    build_uncontrolled_context,
    build_uncontrolled_messages,
    generate_dynamic_niah_dataset_v2,
    insertion_positions_for_example,
    sample_random_insertion_positions,
    write_dynamic_niah_v2,
    _insert_at_sentence_ends,
    _insert_at_word_boundaries,
    _sentence_end_offsets,
)


def test_v2_control_insertions_and_answers() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[True, False, True],
        global_random_seed=17,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["gold_answer"]["city"] is not None
    assert row["control_gold_answer"]["has_answer"] is True
    assert (
        row["control_gold_answer"]["city"] == row["control_relevant_records"][0]["city"]
    )

    realized_by_id = {x["needle_id"]: x for x in row["realized_insertions"]}
    needles_by_id = {x["needle_id"]: x for x in row["needles"]}

    for nid in ("N1", "N3"):
        assert realized_by_id[nid]["is_control"] is True
        assert realized_by_id[nid]["inserted_from"] == "control"
        assert (
            realized_by_id[nid]["control"]["token_length"]
            == needles_by_id[nid]["token_length"]
        )

    assert realized_by_id["N2"]["is_control"] is False
    assert realized_by_id["N2"]["inserted_from"] == "needle"


def test_v2_control_all_true_yields_no_control_answer() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=2,
        insertion_positions=(5, 20),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[True, True],
        global_random_seed=21,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    assert row["control_gold_answer"] == {
        "city": None,
        "score": None,
        "has_answer": False,
    }
    assert row["control_relevant_records"] == []


def test_v2_rows_include_schema_version_and_writer_copies_schema(
    tmp_path: Path,
) -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=1,
        insertion_positions=(5,),
        haystack_dir="data/haystacks/paul_graham",
        output_dir=str(tmp_path),
        data_save_path=str(tmp_path / "dynamic_niah_v2.jsonl"),
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    assert row["schema_version"] == "dynamic_niah_v2_dataset_v1"

    paths = write_dynamic_niah_v2([row], cfg)
    schema_path = Path(paths["schema"])
    assert schema_path.name == "dataset.schema.json"
    assert schema_path.exists()

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["title"] == "Dynamic NIAH v2 JSONL row"
    assert "schema_version" in schema["required"]


def test_v2_count_avg_task_changes_query_schema_and_gold() -> None:
    cfg = DynamicNiahV2Config(
        task_type="count_avg",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[False, True, False],
        global_random_seed=31,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["task_type"] == "count_avg"
    assert "average score" in row["query"]
    assert row["gold_answer"]["count"] == 3
    expected_avg = sum(r["score"] for r in row["relevant_records"]) / 3
    assert row["gold_answer"]["average_score"] == expected_avg
    assert row["control_gold_answer"]["count"] == 2
    assert row["control_gold_answer"]["has_answer"] is True
    assert '{"count":0,"average_score":0.0}' in row["messages"][0]["content"]

    schema = json.loads(Path("dataset.schema.json").read_text(encoding="utf-8"))
    assert "count_avg_answer" in schema["$defs"]


def test_match_count_can_use_single_uniform_fact_template() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        fact_templates_path="data/templates/niah_fact_single_template.txt",
        global_random_seed=97,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert "city score audit records" in row["query"]
    assert "How many cities received a score?" in row["query"]
    assert "Make sure to memorize" not in row["query"]
    assert "Some information about cities are inserted" in row["messages"][0]["content"]
    for needle in row["needles"]:
        assert "city score audit" in needle["decoded_text"]
        assert "received a score of" in needle["decoded_text"]
        assert "Survey note" not in needle["decoded_text"]


def test_match_count_marker_needles_render_vanilla_prompt() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        counting_needle_kind="marker",
        marker_text="[dolphin]",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        prompt_style="vanilla",
        global_random_seed=98,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["counting_needle_kind"] == "marker"
    assert row["marker_text"] == "[dolphin]"
    assert row["gold_answer"] == {"count": 3}
    assert row["context"].count("[dolphin]") == 3
    assert 'The exact marker "[dolphin]" is inserted' in row["messages"][0]["content"]
    assert 'How many times does the exact marker "[dolphin]" appear' in row["query"]
    for needle in row["needles"]:
        assert needle["decoded_text"] == "[dolphin]"
        assert needle["record"]["marker"] == "[dolphin]"


def test_match_count_marker_vanilla_no_cue_removes_instruction_cue() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        counting_needle_kind="marker",
        marker_text="[dolphin]",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=2,
        insertion_positions=(5, 20),
        haystack_dir="data/haystacks/paul_graham",
        prompt_style="vanilla_no_cue",
        global_random_seed=99,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    prompt = row["messages"][0]["content"]
    instruction = prompt.split("\n\nContext:\n", 1)[0]

    assert instruction == (
        'Return ONLY one JSON object on a single line with schema {"count":0}. '
        "No extra text."
    )
    assert 'The exact marker "[dolphin]" is inserted' not in instruction
    assert "Do NOT explain or include reasoning." not in instruction
    assert 'How many times does the exact marker "[dolphin]" appear' in row["query"]


def test_city_score_needles_use_distinct_entities_within_each_example() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=120,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        fact_templates_path="data/templates/niah_fact_single_template.txt",
        global_random_seed=42,
        needle_seed=456,
    )

    rows = generate_dynamic_niah_dataset_v2(cfg)

    for row in rows:
        cities = [needle["record"]["city"] for needle in row["needles"]]
        assert len(cities) == len(set(cities))


def test_short_haystack_is_repeated_to_target_length(tmp_path: Path) -> None:
    haystack_dir = tmp_path / "haystacks"
    haystack_dir.mkdir()
    (haystack_dir / "short.txt").write_text(
        "Alpha beta gamma. " * 360,
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=1200,
        num_needles=1,
        insertion_positions=(50,),
        haystack_dir=str(haystack_dir),
        global_random_seed=101,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert len(row["haystack"]["base_tokens"]) == 1200
    assert row["haystack"]["source_repeated_to_target"] is True
    assert row["haystack"]["source_repeat_count"] > 1
    assert row["haystack"]["expanded_token_count"] >= 1200


def test_multi_file_haystack_deduplicates_sources_without_repeating(
    tmp_path: Path,
) -> None:
    haystack_dir = tmp_path / "haystacks"
    haystack_dir.mkdir()
    source_texts = {
        "alpha.txt": "".join(
            f"Alpha source sentence number {index}. " for index in range(800)
        ),
        "beta.txt": "".join(
            f"Beta source sentence number {index}. " for index in range(800)
        ),
        "gamma.txt": "".join(
            f"Gamma source sentence number {index}. " for index in range(800)
        ),
    }
    for name, text in source_texts.items():
        (haystack_dir / name).write_text(text, encoding="utf-8")
    (haystack_dir / "alpha_copy.txt").write_text(
        source_texts["alpha.txt"],
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=9_000,
        num_needles=1,
        insertion_positions=(50,),
        haystack_dir=str(haystack_dir),
        haystack_source_mode="multi_file_no_repeat",
        global_random_seed=102,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    metadata = row["haystack"]

    assert len(metadata["base_tokens"]) == 9_000
    assert metadata["source_mode"] == "multi_file_no_repeat"
    assert metadata["window_strategy"] == "seed_shuffled_nested_prefix"
    assert metadata["window_start"] == 0
    assert metadata["source_repeated_to_target"] is False
    assert metadata["source_repeat_count"] == 1
    assert metadata["source_file_count"] == 3
    assert not {
        "alpha.txt",
        "alpha_copy.txt",
    }.issubset(set(metadata["source_files"]))
    assert metadata["original_token_count"] >= 9_000
    assert metadata["expanded_token_count"] == metadata["original_token_count"]


def test_sentence_level_insertion_uses_sentence_boundaries_and_spans(
    tmp_path: Path,
) -> None:
    haystack_dir = tmp_path / "haystacks"
    haystack_dir.mkdir()
    (haystack_dir / "sentences.txt").write_text(
        ("Alpha beta gamma. Delta epsilon zeta! Eta theta iota? " * 120),
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, None),
        sentence_level_insertion=True,
        randomize_needle_seed=13,
        haystack_dir=str(haystack_dir),
        global_random_seed=103,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["controls"]["sentence_level_insertion"] is True
    assert row["controls"]["insertion_position_pattern"] == [5, 20, None]
    assert row["controls"]["insertion_positions"][2] is None
    assert row["controls"]["insertion_positions"][:2] != [5, 20]
    assert len(row["realized_insertions"]) == 2
    for insertion in row["realized_insertions"]:
        start = insertion["char_start"]
        end = insertion["char_end"]
        assert row["context"][start - 1] == " "
        assert row["context"][start - 2] in ".!?"
        assert row["context"][start:end] == insertion["decoded_text"]
        assert insertion["context_span_start"] < insertion["context_span_end"]
        assert insertion["token_span_verified"] is True


def test_sentence_end_offsets_reject_urls_paths_and_abbreviations() -> None:
    text = (
        "Visit https://github.com/foo/bar.txt for details. Then continue. "
        "This is e. g. a fragment. Real sentence. Version 1.2.3 is installed. "
        "Final sentence!"
    )

    offsets = _sentence_end_offsets(text)
    snippets = [text[max(0, offset - 28) : offset] for offset in offsets]

    assert any(snippet.endswith("for details.") for snippet in snippets)
    assert any(snippet.endswith("Then continue.") for snippet in snippets)
    assert any(snippet.endswith("Real sentence.") for snippet in snippets)
    assert any(snippet.endswith("is installed.") for snippet in snippets)
    assert any(snippet.endswith("Final sentence!") for snippet in snippets)
    assert not any(snippet.endswith("github.") for snippet in snippets)
    assert not any(snippet.endswith("bar.") for snippet in snippets)
    assert not any(snippet.endswith("This is e.") for snippet in snippets)
    assert not any(snippet.endswith("This is e. g.") for snippet in snippets)
    assert not any(snippet.endswith("Version 1.") for snippet in snippets)
    assert not any(snippet.endswith("Version 1.2.") for snippet in snippets)


def test_sentence_level_insertion_uses_conservative_delimiters() -> None:
    tok = TokenizerAdapter("simple")
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        num_needles=2,
        sentence_level_insertion=True,
        randomize_needle_seed=0,
    )
    needles = [
        {
            "needle_id": f"N{i}",
            "inserted_decoded_text": f"needle {i}.",
            "inserted_tokens": ["needle", str(i), "."],
            "token_length": 3,
            "is_control": False,
            "control": None,
        }
        for i in range(1, 3)
    ]

    out, realized = _insert_at_sentence_ends(
        "Visit https://github.com/foo/bar.txt for details. Then continue.",
        needles,
        (True, True),
        cfg=cfg,
        tok=tok,
        ex_idx=0,
    )

    assert len(realized) == 2
    assert all(item["sentence_delimiter_candidate_count"] == 2 for item in realized)
    assert "github. needle" not in out
    assert "bar. needle" not in out
    for insertion in realized:
        start = insertion["char_start"]
        end = insertion["char_end"]
        assert out[start - 1].isspace()
        assert end == len(out) or out[end].isspace()
        assert out[start:end] == insertion["decoded_text"]
        assert insertion["token_span_verified"] is True


def test_sentence_level_insertion_falls_back_when_no_conservative_delimiters() -> None:
    tok = TokenizerAdapter("simple")
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        num_needles=1,
        sentence_level_insertion=True,
        randomize_needle_seed=0,
    )
    needle = {
        "needle_id": "N1",
        "inserted_decoded_text": "needle one.",
        "inserted_tokens": ["needle", "one", "."],
        "token_length": 3,
        "is_control": False,
        "control": None,
    }

    out, realized = _insert_at_sentence_ends(
        "https://github.com/foo/bar.txt https://example.com/a.b",
        [needle],
        (True,),
        cfg=cfg,
        tok=tok,
        ex_idx=0,
    )

    assert len(realized) == 1
    insertion = realized[0]
    assert insertion["sentence_delimiter_filter_fallback"] == "word_boundary"
    assert insertion["sentence_delimiter_conservative_candidate_count"] == 0
    assert insertion["sentence_delimiter_fallback_candidate_count"] == 1
    assert insertion["token_span_source"] == "sentence_text_offsets_word_boundary_fallback"
    assert "github. needle" not in out
    assert "bar. needle" not in out
    assert out[insertion["char_start"] : insertion["char_end"]] == "needle one."
    assert out[insertion["char_start"] - 1].isspace()
    assert insertion["char_end"] == len(out) or out[insertion["char_end"]].isspace()
    assert insertion["token_span_verified"] is True


def test_sentence_level_insertion_samples_with_replacement_when_needed() -> None:
    tok = TokenizerAdapter("simple")
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        num_needles=3,
        sentence_level_insertion=True,
        randomize_needle_seed=0,
    )
    needles = [
        {
            "needle_id": f"N{i}",
            "inserted_decoded_text": f"needle {i}",
            "inserted_tokens": [f"needle", str(i)],
            "token_length": 2,
            "is_control": False,
            "control": None,
        }
        for i in range(1, 4)
    ]

    out, realized = _insert_at_sentence_ends(
        "Alpha. Beta.",
        needles,
        (True, True, True),
        cfg=cfg,
        tok=tok,
        ex_idx=0,
    )

    assert len(realized) == 3
    assert all(
        item["sentence_delimiter_sampled_with_replacement"] is True
        for item in realized
    )
    assert all(item["sentence_delimiter_candidate_count"] == 2 for item in realized)
    assert any(
        count > 1
        for count in {
            offset: [item["sentence_delimiter_offset"] for item in realized].count(
                offset
            )
            for offset in {item["sentence_delimiter_offset"] for item in realized}
        }.values()
    )
    assert "needle 1 needle 2" in out or "needle 2 needle 3" in out
    for insertion in realized:
        assert out[insertion["char_start"] : insertion["char_end"]] == insertion[
            "decoded_text"
        ]
        assert insertion["context_span_start"] < insertion["context_span_end"]
        assert insertion["token_span_verified"] is True


def test_word_level_insertion_uses_whitespace_boundaries_and_spans(
    tmp_path: Path,
) -> None:
    haystack_dir = tmp_path / "haystacks"
    haystack_dir.mkdir()
    (haystack_dir / "words.txt").write_text(
        ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 800),
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        counting_needle_kind="marker",
        marker_text="[dolphin]",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=80,
        num_needles=3,
        insertion_positions=(5, 20, None),
        word_level_insertion=True,
        randomize_needle_seed=13,
        haystack_dir=str(haystack_dir),
        global_random_seed=104,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["controls"]["word_level_insertion"] is True
    assert row["controls"]["sentence_level_insertion"] is False
    assert row["gold_answer"] == {"count": 2}
    assert row["context"].count("[dolphin]") == 2
    assert len(row["realized_insertions"]) == 2
    for insertion in row["realized_insertions"]:
        start = insertion["char_start"]
        end = insertion["char_end"]
        assert row["context"][start:end] == "[dolphin]"
        assert row["context"][start - 1].isspace()
        assert end == len(row["context"]) or row["context"][end].isspace()
        assert insertion["word_level_insertion"] is True
        assert insertion["word_boundary_candidate_count"] > 0
        assert insertion["context_span_start"] < insertion["context_span_end"]
        assert insertion["token_span_verified"] is True


def test_num_max_needles_samples_variable_counts() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        counting_needle_kind="marker",
        marker_text="[dolphin]",
        tokenizer_name="simple",
        num_examples=12,
        target_haystack_tokens=300,
        num_needles=3,
        num_max_needles=4,
        insertion_positions=(5, 80, 160, 240),
        randomize_needle_insertion=True,
        randomize_needle_seed=7,
        haystack_dir="data/haystacks/paul_graham",
        global_random_seed=104,
    )

    rows = generate_dynamic_niah_dataset_v2(cfg)
    counts = [int(row["gold_answer"]["count"]) for row in rows]

    assert all(1 <= count <= 4 for count in counts)
    assert len(set(counts)) > 1
    for row, count in zip(rows, counts):
        assert row["controls"]["num_needles_configured"] == 3
        assert row["controls"]["num_max_needles"] == 4
        assert row["controls"]["target_num_needles"] == count
        assert row["controls"]["variable_num_needles"] is True
        assert len(row["needles"]) == count
        assert len(row["realized_insertions"]) == count
        assert row["context"].count("[dolphin]") == count


def test_word_level_insertion_samples_with_replacement_when_needed() -> None:
    tok = TokenizerAdapter("simple")
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        num_needles=3,
        word_level_insertion=True,
        randomize_needle_seed=0,
    )
    needles = [
        {
            "needle_id": f"N{i}",
            "inserted_decoded_text": "[dolphin]",
            "inserted_tokens": ["[dolphin]"],
            "token_length": 1,
            "is_control": False,
            "control": None,
        }
        for i in range(1, 4)
    ]

    out, realized = _insert_at_word_boundaries(
        "Alpha Beta",
        needles,
        (True, True, True),
        cfg=cfg,
        tok=tok,
        ex_idx=0,
    )

    assert out == "Alpha [dolphin] [dolphin] [dolphin] Beta"
    assert len(realized) == 3
    assert all(item["word_boundary_sampled_with_replacement"] is True for item in realized)
    assert all(item["word_boundary_candidate_count"] == 1 for item in realized)
    spans = {(item["char_start"], item["char_end"]) for item in realized}
    assert len(spans) == 3
    for insertion in realized:
        assert out[insertion["char_start"] : insertion["char_end"]] == "[dolphin]"
        assert insertion["context_span_start"] < insertion["context_span_end"]
        assert insertion["token_span_verified"] is True


def test_word_level_city_score_allows_context_token_length_change() -> None:
    class SpaceSensitiveTokenizer:
        def encode(self, text):
            out = []
            current = ""
            for ch in text:
                if ch.isspace():
                    if current:
                        out.append(current)
                        current = ""
                    out.append("<space>")
                else:
                    current += ch
            if current:
                out.append(current)
            return out

    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="mock",
        num_examples=1,
        num_needles=1,
        word_level_insertion=True,
    )
    needle = {
        "needle_id": "N1",
        "inserted_decoded_text": "City scored 77.",
        "inserted_tokens": ["City", "scored", "77."],
        "token_length": 3,
        "is_control": False,
        "control": None,
    }

    out, realized = _insert_at_word_boundaries(
        "Alpha Beta",
        [needle],
        (True,),
        cfg=cfg,
        tok=SpaceSensitiveTokenizer(),
        ex_idx=0,
    )

    assert out == "Alpha City scored 77. Beta"
    assert realized[0]["needle_token_length"] == 3
    assert realized[0]["observed_token_length"] != 3
    assert realized[0]["token_length"] == realized[0]["observed_token_length"]
    assert realized[0]["token_span_verified"] is True


def test_literal_text_insertion_verification_can_enforce_uid_context_token_length() -> None:
    class CharacterTokenizer:
        def encode(self, text):
            return list(text)

    with pytest.raises(ValueError, match="token length verification failed"):
        dynamic_niah_v2._verified_text_insertion_metadata(
            tok=CharacterTokenizer(),
            text="Alpha ABC Beta",
            inserted_text="ABC",
            char_start=6,
            char_end=9,
            expected_token_length=1,
        )


def test_text_span_uses_final_context_offset_mapping() -> None:
    class OffsetAwareTokenizer:
        def encode(self, text):
            # Deliberately makes prefix-length subtraction wrong at whitespace
            # boundaries, like byte-level BPE tokenizers with leading-space
            # tokens.
            return list(text)

        def encode_with_offsets(self, text):
            offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
            return list(range(len(offsets))), offsets

    metadata = dynamic_niah_v2._verified_text_insertion_metadata(
        tok=OffsetAwareTokenizer(),
        text="Alpha Needle Beta",
        inserted_text="Needle",
        char_start=6,
        char_end=12,
    )

    assert metadata["context_span_start"] == 1
    assert metadata["context_span_end"] == 2
    assert metadata["observed_token_length"] == 1


def test_text_insertion_modes_are_mutually_exclusive() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        sentence_level_insertion=True,
        word_level_insertion=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_dynamic_niah_dataset_v2(cfg)


def test_count_avg_uncontrolled_prompt_restores_all_needles() -> None:
    cfg = DynamicNiahV2Config(
        task_type="count_avg",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[False, True, False],
        global_random_seed=37,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    controlled_needle = next(n for n in row["needles"] if n["is_control"])

    assert row["gold_answer"]["count"] == 3
    assert row["control_gold_answer"]["count"] == 2
    assert controlled_needle["decoded_text"] not in row["messages"][0]["content"]

    uncontrolled_context = build_uncontrolled_context(row)
    uncontrolled_messages = build_uncontrolled_messages(cfg, row)

    assert controlled_needle["decoded_text"] in uncontrolled_context
    assert controlled_needle["inserted_decoded_text"] not in uncontrolled_context
    assert controlled_needle["decoded_text"] in uncontrolled_messages[0]["content"]
    assert (
        controlled_needle["inserted_decoded_text"]
        not in uncontrolled_messages[0]["content"]
    )


def test_prediction_messages_use_uncontrolled_context_not_saved_prompts() -> None:
    cfg = DynamicNiahV2Config(
        task_type="count_avg",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[False, True, False],
        global_random_seed=43,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    controlled_needle = next(n for n in row["needles"] if n["is_control"])
    row["messages"] = [{"role": "user", "content": "CONTROLLED ONLY"}]
    row["uncontrolled_messages"] = [
        {"role": "user", "content": "STALE CONTROLLED ONLY"}
    ]

    prediction_messages = build_prediction_messages(cfg, row)
    prediction_prompt = prediction_messages[0]["content"]

    assert row["gold_answer"]["count"] == 3
    assert row["control_gold_answer"]["count"] == 2
    assert "CONTROLLED ONLY" not in prediction_prompt
    assert "STALE CONTROLLED ONLY" not in prediction_prompt
    assert controlled_needle["decoded_text"] in prediction_prompt
    assert controlled_needle["inserted_decoded_text"] not in prediction_prompt


def test_control_switch_string_values_are_normalized_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokenizer_name": "simple",
                "num_examples": 1,
                "target_haystack_tokens": 120,
                "num_needles": 2,
                "insertion_positions": [5, 20],
                "haystack_dir": "data/haystacks/paul_graham",
                "control_switch": ["true", "false"],
            }
        ),
        encoding="utf-8",
    )

    from dataset_generation.dynamic_niah_v2 import load_config_file

    assert load_config_file(config_path)["control_switch"] == [True, False]


def test_short_control_switch_is_padded_with_false_for_extra_needles() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=5,
        insertion_positions=(5, 20, 50, 70, 90),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[True, False, False],
        global_random_seed=47,
    )

    with pytest.warns(UserWarning, match="padding missing needle controls with False"):
        row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["controls"]["control_switch"] == [True, False, False, False, False]
    assert [needle["is_control"] for needle in row["needles"]] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_control_switch_mismatch_message_points_to_notebook_overrides() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[True, False, False, False, False],
    )

    try:
        generate_dynamic_niah_dataset_v2(cfg)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected ValueError for mismatched control_switch")

    assert "got 5 control values for num_needles=3" in message
    assert "--num-needles and --positions" in message


def test_match_count_uses_inserted_relevant_records_with_nullable_positions() -> None:
    cfg = DynamicNiahV2Config(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(None, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[False, True, False],
        global_random_seed=53,
        needle_seed=101,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]

    assert row["task_type"] == "match_count"
    assert row["gold_answer"] == {"count": 2}
    assert row["control_gold_answer"] == {"count": 1, "has_answer": True}
    assert len(row["needles"]) == 3
    assert len(row["realized_insertions"]) == 2
    assert row["needles"][0]["is_inserted"] is False
    assert row["needles"][0]["requested_position"] is None
    assert {r["needle_id"] for r in row["relevant_records"]} == {"N2", "N3"}
    assert {r["needle_id"] for r in row["control_relevant_records"]} == {"N3"}
    assert '{"count":0}' in row["messages"][0]["content"]


def test_literal_count_reuses_canary_and_validates_token_occurrences() -> None:
    cfg = DynamicNiahV2Config(
        task_type="literal_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, None, 50),
        haystack_dir="data/haystacks/paul_graham",
        control_switch=[False, False, True],
        global_random_seed=59,
        needle_seed=202,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    canaries = {needle["record"]["canary"] for needle in row["needles"]}
    canary = next(iter(canaries))

    assert row["task_type"] == "literal_count"
    assert len(canaries) == 1
    assert 20 <= len(canary) <= 40
    assert all(needle["token_length"] == 4 for needle in row["needles"])
    assert all(
        needle["record"]["uid_token_length"] == 4 for needle in row["needles"]
    )
    assert f'"{canary}"' in row["query"]
    assert f'The exact literal "{canary}" is inserted' in row["messages"][0]["content"]
    assert "Some exact literal strings are inserted" not in row["messages"][0]["content"]
    assert row["gold_answer"] == {"count": 2}
    assert row["control_gold_answer"] == {"count": 1, "has_answer": True}
    assert row["literal_validation"]["checked"] is True
    assert row["literal_validation"]["validation_basis"] == "literal_text"
    assert row["literal_validation"]["observed_occurrences"] == 2
    assert row["literal_validation"]["observed_literal_text_occurrences"] == 2
    assert row["needles"][1]["is_inserted"] is False


def test_literal_count_word_level_insertion_preserves_uid_token_length(
    tmp_path: Path,
) -> None:
    haystack_dir = tmp_path / "haystacks"
    haystack_dir.mkdir()
    (haystack_dir / "words.txt").write_text(
        "Alpha beta gamma delta epsilon zeta eta theta. " * 400,
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(
        task_type="literal_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=80,
        num_needles=3,
        insertion_positions=(0, 0, 0),
        randomize_needle_insertion=True,
        randomize_needle_seed=7,
        word_level_insertion=True,
        haystack_dir=str(haystack_dir),
        global_random_seed=63,
        needle_seed=204,
        uid_token_length=4,
    )

    row = generate_dynamic_niah_dataset_v2(cfg)[0]
    literal = row["needles"][0]["record"]["literal"]

    assert row["context"].count(literal) == 3
    assert row["literal_validation"]["validated_token_length"] == 4
    for needle in row["needles"]:
        assert needle["token_length"] == 4
        assert needle["record"]["uid_token_length"] == 4
        assert needle["text_insertion_mode"] == "word"
    for insertion in row["realized_insertions"]:
        start = insertion["char_start"]
        end = insertion["char_end"]
        assert row["context"][start:end] == literal
        assert insertion["context_span_end"] - insertion["context_span_start"] == 4
        assert start > 0 and row["context"][start - 1].isspace()
        assert end < len(row["context"]) and row["context"][end].isspace()


def test_literal_count_validation_accepts_token_boundary_mismatch() -> None:
    from dataset_generation.dynamic_niah_v2 import _literal_validation_metadata

    class BoundaryChangingTokenizer:
        backend = "test"

        def encode(self, text: str) -> list[str]:
            if text == "<<<NIAH_CANARY ABC123 >>>":
                return ["FULL_NEEDLE"]
            return ["FULL_NEEDLE", "FULL_NEEDLE"]

    needles = [
        {
            "decoded_text": "<<<NIAH_CANARY ABC123 >>>",
            "tokens": ["FULL_NEEDLE"],
            "record": {
                "literal": "ABC123",
                "delimited_text": "<<<NIAH_CANARY ABC123 >>>",
            },
        }
    ]
    context = "<<<NIAH_CANARY ABC123 >>> x ABC123 y ABC123"

    metadata = _literal_validation_metadata(
        BoundaryChangingTokenizer(),
        context,
        needles,
        inserted_count=3,
    )

    assert metadata["observed_occurrences"] == 3
    assert metadata["observed_delimited_token_occurrences"] == 2
    assert metadata["delimited_token_occurrences_match_expected"] is False


def test_needle_seed_is_independent_of_haystack_seed() -> None:
    base = dict(
        task_type="match_count",
        tokenizer_name="simple",
        num_examples=1,
        target_haystack_tokens=120,
        num_needles=3,
        insertion_positions=(5, 20, 50),
        haystack_dir="data/haystacks/paul_graham",
        global_random_seed=61,
        needle_seed=303,
    )
    row_a = generate_dynamic_niah_dataset_v2(
        DynamicNiahV2Config(**base, haystack_seed=11)
    )[0]
    row_b = generate_dynamic_niah_dataset_v2(
        DynamicNiahV2Config(**base, haystack_seed=19)
    )[0]

    assert [n["record"] for n in row_a["needles"]] == [
        n["record"] for n in row_b["needles"]
    ]
    assert row_a["haystack"]["seed"] != row_b["haystack"]["seed"]


def test_config_file_accepts_null_insertion_position_and_needle_seed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokenizer_name": "simple",
                "num_examples": 1,
                "num_needles": 3,
                "insertion_positions": [None, 20, None],
                "needle_seed": 123,
            }
        ),
        encoding="utf-8",
    )

    from dataset_generation.dynamic_niah_v2 import load_config_file

    loaded = load_config_file(config_path)
    assert loaded["insertion_positions"] == (None, 20, None)
    assert loaded["needle_seed"] == 123


def test_config_kwargs_ignores_analysis_only_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokenizer_name": "simple",
                "num_examples": 1,
                "layers": [4, 8, 12],
                "pca_test_count": 1,
            }
        ),
        encoding="utf-8",
    )

    from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config

    cfg = DynamicNiahV2Config.from_config_file(config_path)
    assert cfg.tokenizer_name == "simple"
    assert cfg.num_examples == 1
    assert not hasattr(cfg, "layers")


def test_random_insertion_sampler_is_sorted_spaced_and_deterministic() -> None:
    positions_a = sample_random_insertion_positions(
        target_haystack_tokens=300,
        num_needles=3,
        seed=42,
        margin=50,
        min_separation=50,
    )
    positions_b = sample_random_insertion_positions(
        target_haystack_tokens=300,
        num_needles=3,
        seed=42,
        margin=50,
        min_separation=50,
    )

    assert positions_a == positions_b
    assert list(positions_a) == sorted(positions_a)
    assert all(50 <= pos < 250 for pos in positions_a)
    assert all(b - a >= 50 for a, b in zip(positions_a, positions_a[1:]))


def test_random_insertion_applies_none_pattern_per_example() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=2,
        target_haystack_tokens=300,
        num_needles=3,
        insertion_positions=(100, None, 200),
        randomize_needle_insertion=True,
        randomize_needle_seed=7,
        haystack_dir="data/haystacks/paul_graham",
    )

    first = insertion_positions_for_example(cfg, cfg.insertion_positions, 0)
    second = insertion_positions_for_example(cfg, cfg.insertion_positions, 1)

    assert first[1] is None
    assert second[1] is None
    assert first != second
    assert first[0] < first[2]
    assert second[0] < second[2]


def test_randomized_generation_records_per_row_positions() -> None:
    cfg = DynamicNiahV2Config(
        tokenizer_name="simple",
        num_examples=2,
        target_haystack_tokens=300,
        num_needles=3,
        insertion_positions=(100, None, 200),
        randomize_needle_insertion=True,
        randomize_needle_seed=11,
        haystack_dir="data/haystacks/paul_graham",
    )

    rows = generate_dynamic_niah_dataset_v2(cfg)
    controls = [row["controls"] for row in rows]

    assert controls[0]["insertion_position_pattern"] == [100, None, 200]
    assert controls[0]["randomize_needle_insertion"] is True
    assert controls[0]["randomize_needle_seed"] == 11
    assert controls[0]["insertion_positions"][1] is None
    assert controls[1]["insertion_positions"][1] is None
    assert controls[0]["insertion_positions"] != controls[1]["insertion_positions"]


def test_random_insertion_sampler_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError, match="at most"):
        sample_random_insertion_positions(
            target_haystack_tokens=130,
            num_needles=3,
            seed=1,
            margin=50,
            min_separation=50,
        )
