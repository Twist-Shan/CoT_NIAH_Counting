from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

from realistic_niah.drive_sync import build_run_archive
from realistic_niah.runner import (
    EngineConfig,
    _atomic_jsonl,
    _batched,
    _cleanup_checkpoint_parts,
    _decode_generated_text,
    _load_completed_checkpoints,
    _normalized_manifest_engine,
    _resume_from_git_commits,
    _sampling_params_kwargs,
    build_requests,
    decoding_config,
    load_vllm_runtime,
)
from realistic_niah.spec import (
    FORMAL_PROMPT_MODES,
    FULL_MODE_MODEL_LABELS,
    MATCHED_NONTHINKING_CONTROLS,
    MODEL_REVISIONS,
    MODEL_SPECS,
    PASSAGE_LENGTHS,
    PRIMARY_MODEL_LABELS,
    REASONING_ONLY_MODEL_LABELS,
    REASONING_ONLY_PROMPT_MODES,
    NEEDLE_COUNTS,
    QUERY_LAYOUT,
    SEEDS,
    SMOKE_SEEDS,
)


def _stimulus(index: int) -> dict:
    return {
        "stimulus_id": f"T2000_N6_seed{index}",
        "passage": f"passage {index}",
        "seed": index,
    }


def test_atomic_batch_parts_resume_and_cleanup(tmp_path: Path) -> None:
    results = tmp_path / "requests.jsonl"
    parts = tmp_path / "request_parts"
    _atomic_jsonl(parts / "batch_a.jsonl", [{"request_id": "a", "value": 1}])
    _atomic_jsonl(parts / "batch_b.jsonl", [{"request_id": "b", "value": 2}])
    completed = _load_completed_checkpoints(results, parts)
    assert sorted(completed) == ["a", "b"]
    _atomic_jsonl(results, (completed[key] for key in sorted(completed)))
    assert sorted(_load_completed_checkpoints(results, parts)) == ["a", "b"]
    _cleanup_checkpoint_parts(parts)
    assert not parts.exists()


def test_manifest_engine_additive_defaults_are_resume_compatible() -> None:
    legacy = {
        "tensor_parallel_size": 1,
        "max_model_len": 32_768,
    }

    assert _normalized_manifest_engine(legacy) == {
        "tensor_parallel_size": 1,
        "max_model_len": 32_768,
        "enforce_eager": False,
        "disable_custom_all_reduce": False,
    }
    assert legacy == {
        "tensor_parallel_size": 1,
        "max_model_len": 32_768,
    }


def test_resume_git_commits_are_an_explicit_allowlist(monkeypatch) -> None:
    first = "a" * 40
    second = "b" * 40
    monkeypatch.setenv(
        "REALISTIC_NIAH_RESUME_FROM_COMMITS",
        f"{first}:{second}",
    )

    assert _resume_from_git_commits() == {first, second}


def test_qwen_builds_four_formal_requests_per_stimulus() -> None:
    requests = build_requests(
        [_stimulus(index) for index in range(6)],
        model_spec=MODEL_SPECS["Qwen3-8B"],
    )

    assert len(requests) == 24
    assert len({request["request_id"] for request in requests}) == 24
    assert {request["prompt_mode"] for request in requests} == set(
        FORMAL_PROMPT_MODES
    )
    assert all(QUERY_LAYOUT in request["request_id"] for request in requests)


def test_v2_panel_contains_eight_primary_models_and_one_matched_control() -> None:
    assert set(PRIMARY_MODEL_LABELS) == {
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-32B",
        "Gemma4-E4B",
        "Gemma4-12B",
        "DeepSeek-R1-0528-Qwen3-8B",
        "GLM-Z1-9B-0414",
    }
    assert set(MODEL_SPECS) == set(PRIMARY_MODEL_LABELS) | {
        "GLM-4-9B-0414"
    }
    assert MATCHED_NONTHINKING_CONTROLS == {
        "DeepSeek-R1-0528-Qwen3-8B": "Qwen3-8B",
        "GLM-Z1-9B-0414": "GLM-4-9B-0414",
    }


def test_registered_decoding_budgets() -> None:
    qwen = MODEL_SPECS["Qwen3-8B"]

    assert decoding_config(qwen, "direct").max_tokens == 64
    assert decoding_config(qwen, "enumeration_index").max_tokens == 1536
    assert decoding_config(qwen, "enumeration_bullet").max_tokens == 1536
    thinking = decoding_config(qwen, "native_thinking")
    assert thinking.max_tokens == 4096
    assert thinking.temperature == 0.6
    assert EngineConfig().max_model_len == 32_768
    assert EngineConfig().enforce_eager is False
    assert EngineConfig().disable_custom_all_reduce is False
    assert _sampling_params_kwargs(thinking, seed=1234)[
        "skip_special_tokens"
    ] is False


def test_vllm_runtime_forwards_tp2_stability_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return object()

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = object  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "realistic_niah.runner.resolve_model_revision",
        lambda model_id, revision: MODEL_REVISIONS["Qwen3-8B"],
    )

    config = EngineConfig(
        tensor_parallel_size=2,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    runtime = load_vllm_runtime(
        model_spec=MODEL_SPECS["Qwen3-8B"],
        revision=MODEL_REVISIONS["Qwen3-8B"],
        engine_config=config,
        cache_dir=None,
    )

    assert runtime.engine_config == config
    assert captured["tensor_parallel_size"] == 2
    assert captured["gpu_memory_utilization"] == 0.92
    assert captured["enforce_eager"] is True
    assert captured["disable_custom_all_reduce"] is True


def test_deepseek_output_uses_tokenizer_json_decoder() -> None:
    class Decoder:
        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert token_ids == [1, 2, 3]
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is False
            return "<think>\nDone.\n</think>\nTotal: 6"

    decoded, strategy = _decode_generated_text(
        model_spec=MODEL_SPECS["DeepSeek-R1-0528-Qwen3-8B"],
        engine_text="<think>ĊDone.Ċ</think>ĊTotal:Ġ6",
        token_ids=[1, 2, 3],
        token_json_decoder=Decoder(),
    )

    assert decoded == "<think>\nDone.\n</think>\nTotal: 6"
    assert strategy == "tokenizer_json_from_output_token_ids"


def test_other_models_preserve_vllm_output_text() -> None:
    decoded, strategy = _decode_generated_text(
        model_spec=MODEL_SPECS["Qwen3-8B"],
        engine_text="<think>Done.</think>\nTotal: 6",
        token_ids=[1, 2, 3],
        token_json_decoder=None,
    )

    assert decoded == "<think>Done.</think>\nTotal: 6"
    assert strategy == "vllm_output_text"


def test_always_on_reasoning_models_run_native_thinking_only() -> None:
    deepseek = MODEL_SPECS["DeepSeek-R1-0528-Qwen3-8B"]
    glm = MODEL_SPECS["GLM-Z1-9B-0414"]

    assert deepseek.prompt_modes == REASONING_ONLY_PROMPT_MODES
    assert glm.prompt_modes == REASONING_ONLY_PROMPT_MODES

    deepseek_decode = decoding_config(deepseek, "native_thinking")
    assert deepseek_decode.max_tokens == 4096
    assert deepseek_decode.temperature == 0.6
    assert deepseek_decode.top_p == 0.95
    assert deepseek_decode.top_k == -1

    glm_decode = decoding_config(glm, "native_thinking")
    assert glm_decode.max_tokens == 4096
    assert glm_decode.temperature == 0.6
    assert glm_decode.top_p == 0.95
    assert glm_decode.top_k == 40


def test_v2_request_accounting_is_explicit() -> None:
    stimuli_per_model = (
        len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS)
    )

    assert stimuli_per_model == 500
    assert stimuli_per_model * len(FORMAL_PROMPT_MODES) == 2_000
    assert (
        stimuli_per_model
        * len(FORMAL_PROMPT_MODES)
        * len(FULL_MODE_MODEL_LABELS)
        + stimuli_per_model
        * len(REASONING_ONLY_PROMPT_MODES)
        * len(REASONING_ONLY_MODEL_LABELS)
        == 13_000
    )


def test_v2_json_configs_match_registered_python_spec() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    main = json.loads(
        (repo_root / "configs" / "realistic_niah_main.json").read_text(
            encoding="utf-8"
        )
    )
    smoke = json.loads(
        (repo_root / "configs" / "realistic_niah_smoke.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(main["target_passage_tokens"]) == PASSAGE_LENGTHS
    assert tuple(main["needle_counts"]) == NEEDLE_COUNTS
    assert tuple(main["seeds"]) == SEEDS
    assert main["haystack_source_mode"] == "multi_file_no_repeat"
    assert main["haystack_corpus_protocol"] == (
        "ruler_paul_graham_full_url_list_v1"
    )
    assert tuple(main["prompt_modes"]) == FORMAL_PROMPT_MODES
    assert tuple(main["models"]) == PRIMARY_MODEL_LABELS
    assert main["model_revisions"] == MODEL_REVISIONS
    assert main["matched_nonthinking_controls"] == (
        MATCHED_NONTHINKING_CONTROLS
    )
    assert main["expected_stimuli"] == 500
    assert main["reasoning_only_prompt_modes"] == ["native_thinking"]
    assert tuple(main["reasoning_only_models"]) == REASONING_ONLY_MODEL_LABELS
    assert main["expected_requests_per_full_mode_model"] == 2_000
    assert main["expected_requests_per_reasoning_only_model"] == 500
    assert main["expected_requests_total"] == 13_000
    assert main["matched_control_prompt_modes"] == [
        "direct",
        "enumeration_index",
        "enumeration_bullet",
    ]
    assert main["expected_glm4_control_requests"] == 1_500
    assert main["expected_all_planned_requests"] == 14_500
    assert smoke["models"] == [
        "Qwen3-8B",
        "Gemma4-12B",
        "DeepSeek-R1-0528-Qwen3-8B",
        "GLM-Z1-9B-0414",
    ]
    assert smoke["prompt_modes"] == ["native_thinking"]
    assert tuple(smoke["seeds"]) == SMOKE_SEEDS
    assert smoke["haystack_source_mode"] == "multi_file_no_repeat"
    assert smoke["haystack_corpus_protocol"] == (
        "ruler_paul_graham_full_url_list_v1"
    )
    assert smoke["expected_requests_total"] == 48


def test_request_batches_checkpoint_at_bounded_size() -> None:
    rows = [{"index": index} for index in range(7)]

    batches = list(_batched(rows, 3))

    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert [item["index"] for batch in batches for item in batch] == list(range(7))


def test_archive_is_reproducibly_verifiable(tmp_path: Path) -> None:
    source = tmp_path / "run"
    source.mkdir()
    (source / "requests.jsonl").write_text('{"ok": true}\n', encoding="utf-8")

    metadata = build_run_archive(source, tmp_path / "archives" / "run.tar.gz")

    assert Path(metadata.archive_path).exists()
    assert metadata.size_bytes > 0
    assert len(metadata.sha256) == 64
    assert len(metadata.md5) == 32
