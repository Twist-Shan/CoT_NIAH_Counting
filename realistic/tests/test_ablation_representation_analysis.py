import json
from pathlib import Path
import pytest
import torch

from single_example.ablation_representation_analysis import (
    RepresentationAblationConfig,
    active_ablation_layers,
    get_profile_indices,
    load_representation_ablation_config,
    sample_replacement_hidden_states,
    select_representation_critical_tokens,
)


class FakeTokenizer:
    eos_token_id = 0

    def convert_ids_to_tokens(self, token_id):
        return f"tok{int(token_id)}"

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(f"tok{int(i)}" for i in ids)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_representation_ablation_config_allows_notebook_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ablation-representation.json"
    path.write_text(
        json.dumps(
            {
                "num_critical_tokens": 3,
                "randomize_from_top_layer": True,
                "critical_token_calc_layer": 9,
                "patterns": ["needle_span"],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_representation_ablation_config(
        path, num_critical_tokens=2, randomize_from_top_layer=False
    )

    assert cfg.num_critical_tokens == 2
    assert cfg.randomize_from_top_layer is False
    assert cfg.critical_token_calc_layer == 9
    assert cfg.patterns == ("needle_span",)


def test_profile_indices_use_latter_half_and_guard_single_example() -> None:
    cfg = RepresentationAblationConfig()

    assert get_profile_indices(5, cfg) == [2, 3, 4]
    with pytest.raises(ValueError, match="at least two examples"):
        get_profile_indices(1, cfg)
    assert get_profile_indices(1, cfg.__class__(profile_allow_single_example=True)) == [0]


def test_select_representation_critical_tokens_keeps_filtered_and_adds_all_position_patterns(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    _write(
        run_dir / "tables" / "massive_tokens_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,2,12,tok12,99\n"  # inside needle: filtered pattern excludes, all pattern includes
        "0,24,4,14,tok14,90\n"  # edge: filtered pattern excludes, all pattern includes
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n",
    )
    _write(
        run_dir / "tables" / "massive_tokens_outside_needles_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,2,12,tok12,99\n"
        "0,24,4,14,tok14,90\n"
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n",
    )
    _write(
        run_dir / "tables" / "attention_sinks_topk.csv",
        "example_idx,layer,head,position,token_id,token,received_uniform_ratio\n"
        "0,24,0,3,13,tok13,100\n"  # inside needle
        "0,24,0,4,14,tok14,50\n"  # edge
        "0,24,1,6,16,tok16,9\n",
    )
    _write(
        run_dir / "tables" / "inputs_0_measurements.csv",
        "position,layer,cosine_similarity\n"
        "1,24,0.01\n"  # inside needle
        "4,24,0.02\n"  # edge
        "6,24,0.20\n"
        "8,24,0.30\n",
    )
    input_ids = list(range(10, 30))
    needle_segments = [{"needle_id": "n0", "start": 1, "end": 4, "positions": [1, 2, 3]}]
    cfg = RepresentationAblationConfig(num_critical_tokens=2, critical_token_calc_layer=24)

    selected = select_representation_critical_tokens(
        run_dir=run_dir,
        example_id=0,
        cfg=cfg,
        input_ids=input_ids,
        needle_segments=needle_segments,
        tokenizer=FakeTokenizer(),
    )

    assert [r["position"] for r in selected["massive_activation"]] == [7, 8]
    assert [r["position"] for r in selected["massive_activation_all"]] == [2, 4]
    assert [r["position"] for r in selected["attention_sink"]] == [6]
    assert [r["position"] for r in selected["attention_sink_all"]] == [3, 4]
    assert [r["position"] for r in selected["needle_sensitive"]] == [6, 8]
    assert [r["position"] for r in selected["needle_sensitive_all"]] == [1, 4]
    assert {
        "massive_activation",
        "attention_sink",
        "needle_sensitive",
        "massive_activation_all",
        "attention_sink_all",
        "needle_sensitive_all",
        "needle_span_0",
        "needle_tail_0",
    } == set(selected)
    # Needle-span is an additional pattern and is not capped by K.
    assert [r["position"] for r in selected["needle_span_0"]] == [1, 2, 3]
    assert [r["position"] for r in selected["needle_tail_0"]] == [4, 5]


def test_active_ablation_layers_top_or_bottom_direction() -> None:
    assert active_ablation_layers(2, 5, randomize_from_top_layer=True) == {2, 3, 4}
    assert active_ablation_layers(2, 5, randomize_from_top_layer=False) == {0, 1, 2}
    with pytest.raises(ValueError, match="out of range"):
        active_ablation_layers(5, 5, randomize_from_top_layer=True)


def test_sample_replacement_warns_and_uses_random_stats_position_for_too_long_position(
    capsys,
) -> None:
    stats = {
        "mean": torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3),
        "std": torch.zeros((1, 2, 3), dtype=torch.float32),
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    with pytest.warns(RuntimeWarning, match="exceeds profiled"):
        replacements = sample_replacement_hidden_states(
            stats=stats,
            layer_idx=0,
            positions=[5],
            device=torch.device("cpu"),
            dtype=torch.float32,
            generator=generator,
        )

    assert 5 in replacements
    assert replacements[5].shape == (3,)
    assert "WARNING: Position 5 exceeds profiled" in capsys.readouterr().out


def test_representation_exports_from_package() -> None:
    from single_example import run_single_example_representation_ablation as exported

    assert callable(exported)
