from pathlib import Path

import pytest

from counting.analysis import _copy_parent_qk_outlier_tables_if_available
from single_example.single_example_analysis import SingleExamplePaths


def _paths(run_dir: Path) -> SingleExamplePaths:
    return SingleExamplePaths(
        run_name=run_dir.name,
        run_dir=run_dir,
        figures_dir=run_dir / "figures",
        tensors_dir=run_dir / "tensors",
        generate_data_dir=run_dir / "generate_data",
        tables_dir=run_dir / "tables",
        logs_path=run_dir / "logs.txt",
        analyze_config_path=run_dir / "analyze_hidden_states_config.json",
        metadata_path=run_dir / "run_metadata.json",
    )


def test_copy_parent_qk_outlier_tables_if_available_reuses_parent_artifacts(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    example = parent / "ablation_examples" / "example_id_1"
    source = parent / "tables" / "massive_tokens_all.csv"
    source.parent.mkdir(parents=True)
    source.write_text("example_idx,layer,position\n1,24,7\n", encoding="utf-8")
    (parent / "tables" / "attention_sinks_topk.csv").write_text(
        "example_idx,layer,head,position\n1,24,0,7\n", encoding="utf-8"
    )

    copied = _copy_parent_qk_outlier_tables_if_available(
        parent_run_dir=parent,
        paths=_paths(example),
    )

    assert sorted(path.name for path in copied) == [
        "attention_sinks_topk.csv",
        "massive_tokens_all.csv",
    ]
    assert (example / "tables" / "massive_tokens_all.csv").read_text(
        encoding="utf-8"
    ) == source.read_text(encoding="utf-8")


def test_copy_parent_qk_outlier_tables_if_available_does_not_overwrite(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    example = parent / "ablation_examples" / "example_id_1"
    source = parent / "tables" / "massive_tokens_all.csv"
    target = example / "tables" / "massive_tokens_all.csv"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_text("parent\n", encoding="utf-8")
    target.write_text("existing\n", encoding="utf-8")

    copied = _copy_parent_qk_outlier_tables_if_available(
        parent_run_dir=parent,
        paths=_paths(example),
    )

    assert copied == []
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_counting_qk_requirements_detects_enabled_qk_patterns(tmp_path: Path) -> None:
    config = tmp_path / "ablation.json"
    config.write_text(
        '{"critical_token_calc_layer": 12, '
        '"patterns": ["needle_span", "massive_activation_all"]}',
        encoding="utf-8",
    )

    from counting.analysis import _counting_qk_requirements

    layers, patterns = _counting_qk_requirements(
        run_ablation=True,
        ablation_config_path=config,
        num_critical_tokens=None,
        ablation_random_seed=None,
        critical_token_calc_layer=None,
        run_representation_ablation=False,
        representation_config_path=config,
        representation_num_critical_tokens=None,
        randomize_from_top_layer=None,
        run_representation_restore=False,
        representation_restore_config_path=config,
        restore_num_critical_tokens=None,
        restore_randomize_from_top_layer=None,
    )

    assert layers == [12]
    assert patterns == ("massive_activation_all",)


def test_run_one_counting_example_runs_qk_when_required_tables_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    import counting.analysis as analysis
    from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
    from single_example.single_example_analysis import TokenizedPrompt

    paths = _paths(tmp_path / "example_id_1")
    paths.generate_data_dir.mkdir(parents=True)
    paths.tables_dir.mkdir(parents=True)
    config = tmp_path / "ablation.json"
    config.write_text(
        '{"critical_token_calc_layer": 24, "patterns": ["massive_activation_all"]}',
        encoding="utf-8",
    )
    cfg = DynamicNiahV2Config(cache_dir=None)
    qk_calls = []

    class FakeModel:
        pass

    class FakeTokenizer:
        pass

    def fake_tokenize(tokenizer, messages, *, thinking_mode=False):
        del tokenizer, messages, thinking_mode
        return TokenizedPrompt(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            prompt_text="prompt",
            token_offsets=None,
        )

    def fake_hidden(**kwargs):
        assert kwargs["layers"] == [24]
        return {"hidden": paths.tensors_dir / "inputs_1.pt"}

    def fake_qk(**kwargs):
        qk_calls.append(kwargs)
        assert kwargs["layers"] == [24]
        table = paths.tables_dir / "massive_tokens_all.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text(
            "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
            "1,24,0,1,tok,2.0\n",
            encoding="utf-8",
        )
        return {
            "summary_path": str(paths.tables_dir / "qk_outlier_analysis_summary.json")
        }

    monkeypatch.setattr(
        analysis, "_load_existing_dynamic_config", lambda *args, **kwargs: cfg
    )
    monkeypatch.setattr(
        analysis,
        "prepare_single_example_dataset",
        lambda **kwargs: (paths.generate_data_dir / "dynamic_niah_v2.jsonl", cfg),
    )
    monkeypatch.setattr(
        analysis,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: (FakeModel(), FakeTokenizer()),
    )
    monkeypatch.setattr(analysis, "render_and_tokenize_messages", fake_tokenize)
    monkeypatch.setattr(
        analysis,
        "locate_uncontrolled_needle_segments",
        lambda **kwargs: [
            {"needle_id": "n0", "start": 1, "end": 2, "positions": [1]}
        ],
    )
    monkeypatch.setattr(analysis, "compute_single_example_hidden_states", fake_hidden)
    monkeypatch.setattr(analysis, "run_single_example_qk_outlier_analysis", fake_qk)
    monkeypatch.setattr(
        analysis,
        "run_single_example_ablation",
        lambda **kwargs: {"ok": True},
    )

    summary = analysis._run_one_counting_example(
        row={
            "id": "row1",
            "messages": [],
            "uncontrolled_messages": [],
            "needles": [],
        },
        example_id=1,
        dataset_path=tmp_path / "dataset.jsonl",
        example_paths=paths,
        model_name="fake/model",
        config_path=tmp_path / "config.used.json",
        hidden_layers=[],
        run_ablation=True,
        ablation_config_path=config,
        num_critical_tokens=None,
        ablation_random_seed=None,
        critical_token_calc_layer=None,
        run_representation_ablation=False,
        representation_config_path=config,
        representation_num_critical_tokens=None,
        randomize_from_top_layer=None,
        run_representation_restore=False,
        representation_restore_config_path=config,
        restore_num_critical_tokens=None,
        restore_randomize_from_top_layer=None,
        shared_representation_stats_path=tmp_path / "shared.pt",
    )

    assert len(qk_calls) == 1
    assert summary["qk_outlier_summary"] is not None
    assert summary["ablation_summary"] == {"ok": True}


def test_cleanup_counting_archive_artifacts_removes_notebook_intermediates(tmp_path: Path) -> None:
    from counting.analysis import cleanup_counting_archive_artifacts

    run_dir = tmp_path / "run"
    corruption = (
        run_dir
        / "ablation_examples"
        / "example_id_1"
        / "tables"
        / "ablation_representation_restore"
        / "corrupted_needle_tokens.jsonl"
    )
    corruption.parent.mkdir(parents=True)
    corruption.write_text("{}\n", encoding="utf-8")
    attention_input = run_dir / "tensors" / "attention_stats" / "input_0"
    attention_input.mkdir(parents=True)
    (attention_input / "attention_stats_layer_00_head_00.pt").write_bytes(b"stats")
    keep_attention_summary = run_dir / "tensors" / "attention_stats" / "summary.json"
    keep_attention_summary.write_text("{}", encoding="utf-8")
    qk_cache = run_dir / "tensors" / "qk_cache"
    qk_cache.mkdir()
    (qk_cache / "key_layer_00.pt").write_bytes(b"cache")
    large_pt = run_dir / "tensors" / "hidden_inputs_0.pt"
    large_pt.write_bytes(b"123456")
    small_pt = run_dir / "tensors" / "small.pt"
    small_pt.write_bytes(b"12")

    removed = cleanup_counting_archive_artifacts(
        run_dir,
        delete_large_pt=True,
        max_pt_bytes=4,
    )

    assert removed["corrupted_needle_tokens"] == [corruption]
    assert removed["attention_stats_inputs"] == [attention_input]
    assert removed["qk_cache"] == [qk_cache]
    assert removed["large_pt"] == [large_pt]
    assert not corruption.exists()
    assert not attention_input.exists()
    assert keep_attention_summary.exists()
    assert not qk_cache.exists()
    assert not large_pt.exists()
    assert small_pt.exists()


def test_cleanup_counting_archive_artifacts_keeps_large_pt_when_disabled(
    tmp_path: Path,
) -> None:
    from counting.analysis import cleanup_counting_archive_artifacts

    run_dir = tmp_path / "run"
    large_pt = run_dir / "tensors" / "hidden_inputs_0.pt"
    large_pt.parent.mkdir(parents=True)
    large_pt.write_bytes(b"123456")

    removed = cleanup_counting_archive_artifacts(
        run_dir,
        delete_large_pt=False,
        max_pt_bytes=4,
    )

    assert removed["large_pt"] == []
    assert large_pt.exists()


def test_counting_setting_name_includes_cache_identity() -> None:
    from counting.analysis import build_counting_setting_name

    setting = build_counting_setting_name(
        model_name="Qwen/Qwen3-8B",
        task_type="literal_count",
        prompt_style="easier",
        target_haystack_tokens=1000,
        num_examples=20,
        insertion_positions=[200, None, 500],
        global_random_seed=42,
        haystack_seed=123,
        needle_seed=456,
        thinking_mode=False,
        randomize_needle_insertion=True,
        randomize_needle_seed=42,
        sentence_level_insertion=True,
        fact_templates_path="data/templates/niah_fact_single_template.txt",
        uid_token_length=4,
    )

    assert setting.startswith(
        "Qwen3-8B_literal_count_easier_1000_examples_20_needles_200_null_500"
    )
    assert "rand_insrt_seed_42" in setting
    assert "sent_insrt" in setting
    assert "uidtok_4" in setting
    assert "tmpl_niah_fact_single_template" in setting
    assert "gseed_42_hseed_123_nseed_456_thinking_false" in setting


def test_counting_dataset_cache_validation_and_save(tmp_path: Path) -> None:
    import json

    from counting.analysis import (
        save_counting_dataset_cache,
        validate_counting_dataset_cache,
    )

    source = tmp_path / "source"
    source.mkdir()
    dataset = source / "dynamic_niah_v2.jsonl"
    config = source / "config.used.json"
    predictions = source / "predictions.jsonl"
    metrics = source / "metrics.json"
    dataset.write_text('{"id":"x"}\n', encoding="utf-8")
    predictions.write_text('{"id":"x","exact_match":true}\n', encoding="utf-8")
    metrics.write_text('{"accuracy":1.0}\n', encoding="utf-8")
    expected = {
        "task_type": "match_count",
        "tokenizer_name": "Qwen/Qwen3-8B",
        "num_examples": 1,
        "target_haystack_tokens": 1000,
        "num_needles": 3,
        "insertion_positions": [200, None, 500],
        "randomize_needle_insertion": False,
        "randomize_needle_seed": 42,
        "prompt_style": "easier",
        "thinking_mode": False,
        "global_random_seed": 42,
        "haystack_seed": 123,
        "needle_seed": 456,
        "fact_templates_path": "data/templates/niah_fact_single_template.txt",
    }
    config.write_text(json.dumps(expected), encoding="utf-8")

    cache = tmp_path / "cache"
    copied = save_counting_dataset_cache(
        cache_dir=cache,
        dataset_path=dataset,
        config_path=config,
        predictions_path=predictions,
        metrics_path=metrics,
    )
    validated = validate_counting_dataset_cache(cache, expected)

    assert set(copied) == {
        "dynamic_niah_v2.jsonl",
        "config.used.json",
        "predictions.jsonl",
        "metrics.json",
    }
    assert validated["has_predictions"] is True
    assert validated["predictions_path"] == cache / "predictions.jsonl"

    mismatched = dict(expected, num_examples=2)
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_counting_dataset_cache(cache, mismatched)

    mismatched_template = dict(
        expected, fact_templates_path="data/templates/niah_fact_templates.txt"
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_counting_dataset_cache(cache, mismatched_template)


def test_counting_dataset_cache_treats_missing_randomization_as_default(
    tmp_path: Path,
) -> None:
    import json

    from counting.analysis import validate_counting_dataset_cache

    cache = tmp_path / "legacy_cache"
    cache.mkdir()
    (cache / "dynamic_niah_v2.jsonl").write_text('{"id":"x"}\n', encoding="utf-8")
    legacy_config = {
        "task_type": "match_count",
        "tokenizer_name": "Qwen/Qwen3-8B",
        "num_examples": 1,
        "target_haystack_tokens": 1000,
        "num_needles": 3,
        "insertion_positions": [200, None, 500],
        "prompt_style": "easier",
        "thinking_mode": False,
        "global_random_seed": 42,
        "haystack_seed": 123,
        "needle_seed": 456,
    }
    (cache / "config.used.json").write_text(
        json.dumps(legacy_config), encoding="utf-8"
    )

    expected = dict(
        legacy_config,
        randomize_needle_insertion=False,
        randomize_needle_seed=42,
    )

    validated = validate_counting_dataset_cache(cache, expected)
    assert validated["has_predictions"] is False
