import json
import zipfile
from pathlib import Path

import torch

from counting.cot_analysis import (
    _compute_pattern_attention_for_head,
    _cot_attention_patterns,
    build_cot_analysis_run_config,
    cache_mode_outputs,
    collect_analysis_needle_span_eligibility,
    cleanup_cot_mode_artifacts,
    create_cot_run_paths,
    ensure_selected_full_sequence_artifacts,
    restore_mode_outputs_from_cache,
    run_mode_qk_outlier_analysis,
    save_full_sequence_artifact,
    save_selected_hidden_states_for_examples,
    select_analysis_example_ids,
    write_cot_response_checkpoint_archive,
)


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        text = "\n".join(str(message["content"]) for message in messages)
        return text + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        del add_special_tokens
        ids = torch.tensor([_ids(str(text))], dtype=torch.long)
        return {"input_ids": ids} if return_tensors == "pt" else {"input_ids": ids[0].tolist()}

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().reshape(-1).tolist()
        text = "".join(chr(int(i)) for i in ids)
        if skip_special_tokens:
            text = text.replace("\x00", "")
        return text


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_build_cot_analysis_run_config_normalizes_modes_and_rejects_unknown() -> None:
    cfg = build_cot_analysis_run_config(
        {
            "MODEL_NAME": "model",
            "THINKING_MODES": ["non-thinking", "thinking", "thinking"],
            "K": 5,
            "MAX_ANALYSIS_EXAMPLES": 2,
        }
    )

    assert cfg["TOKENIZER_NAME"] == "model"
    assert cfg["THINKING_MODES"] == ["nonthinking", "thinking"]
    assert cfg["K"] == 5
    assert cfg["OUTLIER_RATIO_THRESHOLD"] == 5.0


def test_build_cot_analysis_run_config_accepts_outlier_ratio_threshold() -> None:
    cfg = build_cot_analysis_run_config(
        {
            "MODEL_NAME": "model",
            "OUTLIER_RATIO_THRESHOLD": 7,
        }
    )

    assert cfg["OUTLIER_RATIO_THRESHOLD"] == 7.0


def test_select_analysis_example_ids_balances_count_and_success() -> None:
    rows = [
        {"gold_answer": {"count": 1}, "needles": [{}]},
        {"gold_answer": {"count": 1}, "needles": [{}]},
        {"gold_answer": {"count": 2}, "needles": [{}, {}]},
        {"gold_answer": {"count": 2}, "needles": [{}, {}]},
    ]
    results = {
        "nonthinking": [
            {"exact_match": True},
            {"exact_match": False},
            {"exact_match": True},
            {"exact_match": False},
        ],
        "thinking": [
            {"exact_match": True},
            {"exact_match": False},
            {"exact_match": True},
            {"exact_match": False},
        ],
    }

    selected = select_analysis_example_ids(
        rows=rows,
        results_by_mode=results,
        max_examples=4,
    )

    assert selected == [0, 1, 2, 3]


def test_select_analysis_example_ids_mixes_count_bins_and_failures() -> None:
    rows = [
        {"gold_answer": {"count": 1}, "needles": [{}]},
        {"gold_answer": {"count": 2}, "needles": [{}, {}]},
        {"gold_answer": {"count": 5}, "needles": [{}] * 5},
        {"gold_answer": {"count": 8}, "needles": [{}] * 8},
        {"gold_answer": {"count": 10}, "needles": [{}] * 10},
        {"gold_answer": {"count": 3}, "needles": [{}] * 3},
    ]
    results = {
        "nonthinking": [
            {"exact_match": True},
            {"exact_match": False},
            {"exact_match": True},
            {"exact_match": False},
            {"exact_match": True},
            {"exact_match": False},
        ]
    }

    selected = select_analysis_example_ids(
        rows=rows,
        results_by_mode=results,
        max_examples=4,
    )
    selected_counts = {rows[idx]["gold_answer"]["count"] for idx in selected}
    selected_exact = {bool(results["nonthinking"][idx]["exact_match"]) for idx in selected}

    assert len(selected) == 4
    assert selected_exact == {True, False}
    assert min(selected_counts) <= 2
    assert max(selected_counts) >= 8


def test_select_analysis_example_ids_excludes_unverified_needle_spans(capsys) -> None:
    rows = [
        {"gold_answer": {"count": 1}, "needles": [{}]},
        {"gold_answer": {"count": 1}, "needles": [{}]},
        {"gold_answer": {"count": 1}, "needles": [{}]},
    ]
    results = {"nonthinking": [{"exact_match": True}] * 3}
    eligibility = {
        0: {"eligible": True},
        1: {
            "eligible": False,
            "missing_needle_ids": ["N1"],
            "missing_by_mode": {"nonthinking": ["N1"]},
        },
        2: {"eligible": True},
    }

    selected = select_analysis_example_ids(
        rows=rows,
        results_by_mode=results,
        max_examples=3,
        analysis_eligibility_by_idx=eligibility,
    )

    captured = capsys.readouterr().out
    assert selected == [0, 2]
    assert "fewer than MAX_ANALYSIS_EXAMPLES=3 will be analyzed" in captured
    assert "N1" in captured


def test_collect_analysis_needle_span_eligibility_flags_missing_sequence(
    tmp_path: Path,
) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "MODEL_NAME": "model",
            "TOKENIZER_NAME": "model",
            "THINKING_MODES": ["nonthinking"],
            "RUN_ROOT": str(tmp_path),
            "DATA_CACHE_ROOT": str(tmp_path / "cache"),
        }
    )
    paths = create_cot_run_paths(cfg)
    row = {
        "id": "row0",
        "context": "Alpha needle appears. Count.",
        "uncontrolled_context": "Alpha needle appears. Count.",
        "query": "How many needles are present?",
        "messages": [{"role": "user", "content": "Alpha needle appears. Count."}],
        "uncontrolled_messages": [{"role": "user", "content": "Alpha needle appears. Count."}],
        "realized_insertions": [
            {
                "needle_id": "N1",
                "final_position": 0,
                "tokens": _ids("absent needle"),
                "decoded_text": "absent needle",
            }
        ],
    }

    eligibility = collect_analysis_needle_span_eligibility(
        cfg=cfg,
        paths=paths,
        rows=[row],
        modes=["nonthinking"],
        tokenizer=FakeTokenizer(),
    )

    assert eligibility[0]["eligible"] is False
    assert eligibility[0]["missing_needle_ids"] == ["N1"]


def test_save_full_sequence_artifact_keeps_prompt_and_entire_generation(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    info = save_full_sequence_artifact(
        tensors_dir=tmp_path / "tensors",
        tables_dir=tmp_path / "tables",
        example_id=0,
        row={"id": "row0"},
        prompt_input_ids=torch.tensor([_ids("prompt")], dtype=torch.long),
        generated_ids=torch.tensor(_ids("think</think>{\"count\":4}\x00"), dtype=torch.long),
        tokenizer=tokenizer,
        task_type="match_count",
        max_new_tokens=64,
        prompt_text="prompt",
        mode="thinking",
    )

    payload = torch.load(tmp_path / "tensors" / "inputs_cot_0.pt", map_location="cpu")
    assert payload["schema_version"] == "counting_cot_full_sequence_v1"
    assert payload["prompt_tokens"] == len("prompt")
    assert payload["full_sequence_tokens"] == len("prompt") + len('think</think>{"count":4}')
    assert info["answer_start_full_index"] == len("prompt") + len("think</think>")
    boundaries = (tmp_path / "tables" / "full_sequence_boundaries_0.csv").read_text(
        encoding="utf-8"
    )
    assert "final_answer_start" in boundaries


def test_pattern_attention_ratio_uses_later_queries_and_uniform_baseline() -> None:
    q = torch.zeros((5, 2), dtype=torch.float32)
    k = torch.zeros((5, 2), dtype=torch.float32)
    patterns = [
        {
            "pattern_name": "needle_span_0",
            "pattern_type": "needle_span",
            "pattern_rank": 1,
            "positions": [1],
        }
    ]

    by_pattern = _compute_pattern_attention_for_head(
        q=q,
        k_tensor=k,
        info={"scaling": 1.0},
        patterns=patterns,
        prompt_tokens=3,
        key_padding_mask=None,
        key_block_size=4,
        query_block_size=2,
    )

    ratio = by_pattern["needle_span_0"]["attention_ratio"]
    assert torch.isnan(ratio[0])
    assert torch.isnan(ratio[1])
    assert torch.allclose(ratio[2:], torch.ones(3), atol=1e-6)


def test_cot_attention_patterns_add_prompt_spans_and_filter_weak_outliers(
    tmp_path: Path,
) -> None:
    mode_dir = tmp_path / "mode"
    tables = mode_dir / "tables"
    tables.mkdir(parents=True)
    (tables / "attention_sinks_topk.csv").write_text(
        "example_idx,layer,position,token,received_uniform_ratio\n"
        "0,4,6,strong_sink,9.0\n"
        "0,4,7,weak_sink,4.0\n",
        encoding="utf-8",
    )
    (tables / "massive_tokens_all.csv").write_text(
        "example_idx,layer,position,token,norm_ratio_to_median\n"
        "0,4,8,strong_massive,6.0\n"
        "0,4,9,weak_massive,3.0\n",
        encoding="utf-8",
    )

    patterns = _cot_attention_patterns(
        mode_dir=mode_dir,
        payload={"prompt_tokens": 5, "full_sequence_tokens": 12},
        example_id=0,
        layer=4,
        k=10,
        needle_spans=[{"needle_id": 0, "start": 2, "end": 4}],
        outlier_ratio_threshold=5.0,
    )
    by_name = {pattern["pattern_name"]: pattern for pattern in patterns}

    assert by_name["prompt_span"]["positions"] == [0, 1, 2, 3, 4]
    assert by_name["prompt_span_no_first"]["positions"] == [1, 2, 3, 4]
    assert by_name["needle_span_0"]["positions"] == [2, 3]
    assert by_name["attention_sink_rank_1"]["pattern_display"] == "strong_sink"
    assert by_name["massive_activation_rank_1"]["pattern_display"] == "strong_massive"
    assert "attention_sink_rank_2" not in by_name
    assert "massive_activation_rank_2" not in by_name


def test_ensure_selected_full_sequence_artifacts_rebuilds_from_cached_prediction(
    tmp_path: Path,
) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "RUN_ROOT": str(tmp_path / "runs"),
            "DATA_CACHE_ROOT": str(tmp_path / "cache"),
            "USER_RUN_NAME": "run_materialize",
            "MODEL_NAME": "model",
            "TOKENIZER_NAME": "tokenizer",
            "THINKING_MODES": ["nonthinking"],
            "PROMPT_STYLE": "vanilla",
        }
    )
    paths = create_cot_run_paths(cfg)
    rows = [
        {
            "id": "row0",
            "task_type": "match_count",
            "uncontrolled_context": "needle text",
            "query": "How many cities received a score?",
            "gold_answer": {"count": 1},
            "needles": [],
            "uncontrolled_realized_insertions": [],
        }
    ]
    paths.response_dir("nonthinking").mkdir(parents=True, exist_ok=True)
    paths.predictions_path("nonthinking").write_text(
        json.dumps({"model_output_text": '{"count": 1}'}) + "\n",
        encoding="utf-8",
    )

    materialized = ensure_selected_full_sequence_artifacts(
        cfg=cfg,
        paths=paths,
        mode="nonthinking",
        rows=rows,
        selected_ids=[0],
        tokenizer=FakeTokenizer(),
    )

    assert materialized == [paths.mode_dir("nonthinking") / "tensors" / "inputs_cot_0.pt"]
    payload = torch.load(materialized[0], map_location="cpu")
    assert payload["prompt_tokens"] > 0
    assert payload["generation_tokens"] == len('{"count": 1}')
    assert (paths.mode_dir("nonthinking") / "tables" / "full_sequence_0.txt").exists()


def test_write_cot_response_checkpoint_archive_includes_all_modes(tmp_path: Path) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "RUN_ROOT": str(tmp_path),
            "USER_RUN_NAME": "run_cot_test",
            "MODEL_NAME": "model",
            "THINKING_MODES": ["nonthinking", "thinking"],
        }
    )
    paths = create_cot_run_paths(cfg)
    paths.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    paths.dataset_path.write_text("{}\n", encoding="utf-8")
    for mode in cfg["THINKING_MODES"]:
        paths.predictions_path(mode).write_text("{}\n", encoding="utf-8")
        paths.metrics_path(mode).write_text(json.dumps({"mode": mode}), encoding="utf-8")

    archive = write_cot_response_checkpoint_archive(paths=paths, modes=cfg["THINKING_MODES"])

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "generate_data/dynamic_niah_v2.jsonl" in names
    assert "responses/nonthinking/predictions.jsonl" in names
    assert "responses/thinking/metrics.json" in names


def test_run_mode_qk_outlier_analysis_skips_without_hidden_inputs(tmp_path: Path) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "RUN_ROOT": str(tmp_path),
            "USER_RUN_NAME": "run_no_hidden",
            "MODEL_NAME": "model",
            "THINKING_MODES": ["nonthinking"],
        }
    )
    paths = create_cot_run_paths(cfg)

    summary = run_mode_qk_outlier_analysis(
        cfg=cfg,
        paths=paths,
        mode="nonthinking",
        selected_ids=[0, 1],
    )

    assert summary["status"] == "skipped"
    assert summary["reason"] == "no_hidden_state_files"
    assert summary["missing_hidden_state_example_indices"] == [0, 1]


def test_cleanup_cot_mode_artifacts_removes_intermediate_tensors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tensors = run_dir / "modes" / "thinking" / "tensors"
    (tensors / "massive_activations" / "input_0").mkdir(parents=True)
    (tensors / "hidden_inputs_0.pt").write_bytes(b"hidden")
    (tensors / "inputs_cot_0.pt").write_bytes(b"cot")
    (tensors / "massive_activations" / "input_0" / "hidden_norms_layer_00.pt").write_bytes(
        b"norms"
    )

    removed = cleanup_cot_mode_artifacts(run_dir)

    assert removed["hidden_inputs"]
    assert removed["full_sequence_inputs"]
    assert removed["massive_activation_tensors"]
    assert not (tensors / "hidden_inputs_0.pt").exists()
    assert not (tensors / "inputs_cot_0.pt").exists()
    assert not (tensors / "massive_activations").exists()


def test_restore_mode_outputs_from_cache_copies_predictions_metrics_and_dataset(tmp_path: Path) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "RUN_ROOT": str(tmp_path / "runs"),
            "DATA_CACHE_ROOT": str(tmp_path / "cache"),
            "USER_RUN_NAME": "run_cot_restore",
            "MODEL_NAME": "model",
            "THINKING_MODES": ["thinking"],
        }
    )
    paths = create_cot_run_paths(cfg)
    assert paths.setting_name != paths.run_name
    assert str(paths.cache_dir).startswith(str(tmp_path / "cache"))
    cache = paths.cache_dir / "responses" / "thinking"
    cache.mkdir(parents=True)
    (cache / "predictions.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (cache / "metrics.json").write_text('{"accuracy": 1.0}', encoding="utf-8")
    (cache / "dynamic_niah_v2.jsonl").write_text('{"row": 1}\n', encoding="utf-8")
    paths.dataset_path.unlink(missing_ok=True)

    restored = restore_mode_outputs_from_cache(cfg=cfg, paths=paths, mode="thinking")

    assert restored is True
    assert paths.predictions_path("thinking").read_text(encoding="utf-8") == '{"x": 1}\n'
    assert json.loads(paths.metrics_path("thinking").read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert paths.dataset_path.read_text(encoding="utf-8") == '{"row": 1}\n'


def test_stable_cache_name_is_independent_of_run_name(tmp_path: Path) -> None:
    base = {
        "RUN_ROOT": str(tmp_path / "runs"),
        "DATA_CACHE_ROOT": str(tmp_path / "cache"),
        "MODEL_NAME": "model",
        "THINKING_MODES": ["nonthinking", "thinking"],
    }
    cfg_a = build_cot_analysis_run_config(base | {"USER_RUN_NAME": "run_a"})
    cfg_b = build_cot_analysis_run_config(base | {"USER_RUN_NAME": "run_b"})
    paths_a = create_cot_run_paths(cfg_a)
    paths_b = create_cot_run_paths(cfg_b)

    assert paths_a.run_name == "run_a"
    assert paths_b.run_name == "run_b"
    assert paths_a.setting_name == paths_b.setting_name
    assert paths_a.cache_dir == paths_b.cache_dir


def test_response_cache_metadata_mismatch_prevents_restore(tmp_path: Path) -> None:
    cfg = build_cot_analysis_run_config(
        {
            "RUN_ROOT": str(tmp_path / "runs"),
            "DATA_CACHE_ROOT": str(tmp_path / "cache"),
            "USER_RUN_NAME": "run_cache_metadata",
            "MODEL_NAME": "model",
            "THINKING_MODES": ["nonthinking"],
            "MAX_NEW_TOKENS_NONTHINKING": 64,
        }
    )
    paths = create_cot_run_paths(cfg)
    paths.dataset_path.write_text('{"row": 1}\n', encoding="utf-8")
    paths.predictions_path("nonthinking").write_text('{"x": 1}\n', encoding="utf-8")
    paths.metrics_path("nonthinking").write_text('{"accuracy": 1.0}', encoding="utf-8")
    cache_mode_outputs(cfg=cfg, paths=paths, mode="nonthinking")

    changed_cfg = dict(cfg)
    changed_cfg["MAX_NEW_TOKENS_NONTHINKING"] = 128
    restored = restore_mode_outputs_from_cache(
        cfg=changed_cfg,
        paths=paths,
        mode="nonthinking",
    )

    assert restored is False


def test_save_selected_hidden_states_preserves_original_layer_ids(
    tmp_path: Path, monkeypatch
) -> None:
    import counting.cot_analysis as cot_analysis

    tensors = tmp_path / "mode" / "tensors"
    tensors.mkdir(parents=True)
    torch.save(
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "prompt_tokens": 2,
            "full_sequence_tokens": 3,
        },
        tensors / "inputs_cot_0.pt",
    )

    def fake_capture_selected_hidden_states(**kwargs):
        assert kwargs["layers"] == [4, 8, 12]
        return torch.zeros((3, 3, 2), dtype=torch.float32)

    monkeypatch.setattr(
        cot_analysis,
        "capture_selected_hidden_states",
        fake_capture_selected_hidden_states,
    )

    paths = save_selected_hidden_states_for_examples(
        model=object(),
        mode_dir=tmp_path / "mode",
        example_ids=[0],
        layers=[4, 8, 12],
    )

    payload = torch.load(paths[0], map_location="cpu")
    assert tuple(payload["hidden"].shape) == (3, 3, 2)
    assert payload["stored_layers"] == [4, 8, 12]
    assert payload["layers"] == [4, 8, 12]
