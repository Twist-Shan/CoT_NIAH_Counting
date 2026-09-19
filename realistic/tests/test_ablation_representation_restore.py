import json
from pathlib import Path

import pytest
import torch

from single_example import run_single_example_representation_restore as exported
from single_example import ablation_representation_analysis_restore as restore_mod
from single_example.ablation_representation_analysis_restore import (
    RepresentationRestoreConfig,
    load_representation_restore_config,
    make_corrupted_needle_input_ids,
    replacement_tokens_for_setting,
    resolve_restore_dataset_path,
    select_representation_restore_critical_tokens,
)
from single_example.single_example_analysis import SingleExamplePaths, cleanup_large_tensor_artifacts


class FakeTokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": [ord(ch) % 50 + 1 for ch in text if not ch.isspace()]}

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


def test_load_representation_restore_config_allows_notebook_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ablation-representation-restore.json"
    path.write_text(
        json.dumps(
            {
                "num_critical_tokens": 3,
                "randomize_from_top_layer": True,
                "critical_token_calc_layer": 9,
                "patterns": ["needle_tail"],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_representation_restore_config(
        path, num_critical_tokens=2, randomize_from_top_layer=False
    )

    assert cfg.num_critical_tokens == 2
    assert cfg.randomize_from_top_layer is False
    assert cfg.critical_token_calc_layer == 9
    assert cfg.patterns == ("needle_tail",)


def test_resolve_restore_dataset_path_requires_named_run(tmp_path: Path) -> None:
    dataset = tmp_path / "run_a" / "dynamic_niah_v2.jsonl"
    _write(dataset, '{"id":"row0"}\n')

    assert resolve_restore_dataset_path("run_a", base_dir=tmp_path) == dataset
    with pytest.raises(ValueError, match="RESTORE_DATASET_RUN_NAME"):
        resolve_restore_dataset_path("", base_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="dynamic_niah_v2.jsonl"):
        resolve_restore_dataset_path("missing", base_dir=tmp_path)


def test_select_restore_critical_tokens_adds_per_needle_tail_patterns(tmp_path: Path) -> None:
    run_dir = tmp_path
    _write(
        run_dir / "tables" / "massive_tokens_outside_needles_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n",
    )
    _write(
        run_dir / "tables" / "massive_tokens_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,2,12,tok12,99\n"
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n",
    )
    _write(
        run_dir / "tables" / "attention_sinks_topk.csv",
        "example_idx,layer,head,position,token_id,token,received_uniform_ratio\n"
        "0,24,0,6,16,tok16,9\n",
    )
    _write(
        run_dir / "tables" / "inputs_0_measurements.csv",
        "position,layer,cosine_similarity\n"
        "6,24,0.20\n"
        "8,24,0.30\n",
    )
    input_ids = list(range(10, 30))
    needle_segments = [
        {"needle_id": "n0", "start": 1, "end": 4, "positions": [1, 2, 3]},
        {"needle_id": "n1", "start": 10, "end": 12, "positions": [10, 11]},
    ]
    cfg = RepresentationRestoreConfig(
        num_critical_tokens=2,
        critical_token_calc_layer=24,
        patterns=("needle_tail", "needle_span", "massive_activation"),
    )

    selected = select_representation_restore_critical_tokens(
        run_dir=run_dir,
        example_id=0,
        cfg=cfg,
        input_ids=input_ids,
        needle_segments=needle_segments,
        tokenizer=FakeTokenizer(),
    )

    assert [r["position"] for r in selected["needle_tail_0"]] == [4, 5]
    assert [r["position"] for r in selected["needle_tail_1"]] == [12, 13]
    assert [r["position"] for r in selected["needle_span_0"]] == [1, 2, 3]
    assert [r["position"] for r in selected["massive_activation"]] == [7, 8]


def test_corrupted_needle_inputs_are_setting_specific() -> None:
    input_ids = list(range(10))
    needle_segments = [{"start": 2, "end": 5, "positions": [2, 3, 4]}]
    pool = list(range(100, 120))

    corrupted_a, records_a = make_corrupted_needle_input_ids(
        input_ids,
        needle_segments=needle_segments,
        replacement_pool=pool,
        seed=1,
        pattern="needle_tail_0",
        layer_idx=0,
    )
    corrupted_b, records_b = make_corrupted_needle_input_ids(
        input_ids,
        needle_segments=needle_segments,
        replacement_pool=pool,
        seed=1,
        pattern="needle_tail_0",
        layer_idx=1,
    )

    assert [r["position"] for r in records_a] == [2, 3, 4]
    assert [r["original_token_id"] for r in records_a] == [2, 3, 4]
    assert corrupted_a.shape == (1, 10)
    assert corrupted_a[0, :2].tolist() == [0, 1]
    assert corrupted_a[0, 5:].tolist() == [5, 6, 7, 8, 9]
    assert corrupted_a[0, 2:5].tolist() != corrupted_b[0, 2:5].tolist()
    assert records_a != records_b


def test_replacement_tokens_for_setting_is_deterministic_by_pattern_and_layer() -> None:
    pool = list(range(100))

    first = replacement_tokens_for_setting(pool=pool, k=4, seed=7, pattern="p", layer_idx=3)
    second = replacement_tokens_for_setting(pool=pool, k=4, seed=7, pattern="p", layer_idx=3)
    other_layer = replacement_tokens_for_setting(pool=pool, k=4, seed=7, pattern="p", layer_idx=4)

    assert first == second
    assert first != other_layer


def test_restore_generation_only_runs_clean_and_restore_outputs(monkeypatch, tmp_path: Path) -> None:
    calls = []

    monkeypatch.setattr(restore_mod, "resolve_decoder_layers", lambda model: [object(), object()])

    def fake_score(row, output):
        del row
        exact_match = output.endswith("restore")
        return {
            "parse_mode": "test",
            "exact_match": exact_match,
            "accuracy": 1.0 if exact_match else 0.0,
        }

    monkeypatch.setattr(restore_mod, "_score_row", fake_score)

    def fake_generate(**kwargs):
        calls.append(
            {
                "pattern": kwargs.get("pattern"),
                "layer_idx": kwargs.get("layer_idx"),
                "restore": kwargs.get("clean_hidden_states") is not None,
            }
        )
        if kwargs.get("clean_hidden_states") is not None:
            return "generated_restore"
        return "generated_clean"

    monkeypatch.setattr(restore_mod, "manual_generate_with_representation_restore", fake_generate)

    predictions, results, corruptions = restore_mod.run_representation_restore_generation(
        model=object(),
        tokenizer=FakeTokenizer(),
        row={"id": "row0"},
        example_id=0,
        input_ids=torch.tensor([[10, 11, 12, 13, 14]]),
        needle_segments=[{"start": 1, "end": 3, "positions": [1, 2]}],
        selected={"needle_tail_0": [{"position": 3}]},
        replacement_pool=[100, 101, 102, 103],
        cfg=RepresentationRestoreConfig(patterns=("needle_tail",), max_new_tokens=1),
        clean_hidden_states=torch.zeros((2, 5, 1)),
        out_dir=tmp_path,
    )

    expected_patterns = ["clean_baseline", "needle_tail_0", "needle_tail_0"]
    assert [row["pattern"] for row in results] == expected_patterns
    assert [row["pattern"] for row in predictions] == expected_patterns
    assert [call["restore"] for call in calls] == [False, True, True]
    assert len(corruptions) == 4
    assert sorted(path.name for path in (tmp_path / "generations").glob("*.txt")) == [
        "clean_baseline.txt",
        "needle_tail_0_layer0_restore.txt",
        "needle_tail_0_layer1_restore.txt",
    ]


def test_cleanup_removes_restore_hidden_and_large_pt_files(tmp_path: Path) -> None:
    paths = SingleExamplePaths(
        run_name="run",
        run_dir=tmp_path,
        figures_dir=tmp_path / "figures",
        tensors_dir=tmp_path / "tensors",
        generate_data_dir=tmp_path / "generate_data",
        tables_dir=tmp_path / "tables",
        logs_path=tmp_path / "logs.json",
        analyze_config_path=tmp_path / "config.json",
        metadata_path=tmp_path / "metadata.json",
    )
    restore_hidden = paths.tensors_dir / "ablation_representation_restore" / "hidden_states_clean_0.pt"
    large_pt = paths.tensors_dir / "misc" / "large.pt"
    small_pt = paths.tensors_dir / "misc" / "small.pt"
    restore_hidden.parent.mkdir(parents=True, exist_ok=True)
    large_pt.parent.mkdir(parents=True, exist_ok=True)
    restore_hidden.write_bytes(b"restore")
    large_pt.write_bytes(b"x" * (2 * 1024 * 1024))
    small_pt.write_bytes(b"small")

    removed = cleanup_large_tensor_artifacts(
        paths,
        example_id=0,
        remove_qk_cache=False,
        max_pt_file_size_mb=1,
    )

    assert restore_hidden in removed
    assert large_pt in removed
    assert not restore_hidden.exists()
    assert not large_pt.exists()
    assert small_pt.exists()


def test_representation_restore_exports_from_package() -> None:
    assert callable(exported)
