from __future__ import annotations

from realistic_niah.smoke_analysis import summarize_guarded_smoke


def _guarded_row(
    *,
    model: str,
    stimulus: str,
    truncated: bool = False,
) -> dict:
    return {
        "request_id": f"{model}/native_thinking/{stimulus}",
        "model_label": model,
        "stimulus_id": stimulus,
        "prompt_mode": "native_thinking",
        "output_tokens": 800 if not truncated else 4096,
        "evaluation": {
            "registered_success": not truncated,
            "exact_count": not truncated,
            "truncated": truncated,
            "parse_status": "ok" if not truncated else "parse_fail",
            "response_format_compliant": not truncated,
            "overthinking_flag": truncated,
        },
    }


def test_guarded_smoke_gate_passes_for_complete_zero_truncation_results() -> None:
    models = ("Qwen3-8B", "Gemma4-12B")
    rows = [
        _guarded_row(model=model, stimulus=stimulus)
        for model in models
        for stimulus in ("s1", "s2")
    ]

    summary = summarize_guarded_smoke(
        rows,
        expected_models=models,
        expected_requests_per_model=2,
    )

    assert summary["gate"]["passed"] is True
    assert summary["gate"]["total_truncations"] == 0
    assert summary["models"]["Qwen3-8B"]["zero_truncations"] is True


def test_guarded_smoke_gate_fails_on_truncation_or_missing_model() -> None:
    rows = [
        _guarded_row(
            model="Qwen3-8B",
            stimulus="s1",
            truncated=True,
        )
    ]

    summary = summarize_guarded_smoke(
        rows,
        expected_models=("Qwen3-8B", "Gemma4-12B"),
        expected_requests_per_model=1,
    )

    assert summary["gate"]["passed"] is False
    assert summary["gate"]["total_truncations"] == 1
    assert summary["gate"]["missing_models"] == ["Gemma4-12B"]
