import json
from pathlib import Path

import torch

from single_example.ablation_analysis import (
    AblationConfig,
    build_irrelevant_token_pool,
    load_ablation_config,
    make_ablated_input_ids,
    replacement_tokens_for_k,
    run_ablation_generation,
    select_critical_tokens,
    summarize_ablation_results_all,
)


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
        if ids == [901]:
            return '{"city":"Paris","score":91}'
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeEmbeddings:
    def __init__(self):
        self.weight = torch.empty(1, device="cpu")


class FakeModel:
    def get_input_embeddings(self):
        return FakeEmbeddings()

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[901]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_ablation_config_allows_notebook_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps(
            {
                "num_critical_tokens": 3,
                "critical_token_calc_layer": 8,
                "ablation_random_seed": 7,
                "patterns": ["needle_tail"],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_ablation_config(
        path, num_critical_tokens=2, ablation_random_seed=9
    )

    assert cfg.num_critical_tokens == 2
    assert cfg.ablation_random_seed == 9
    assert cfg.critical_token_calc_layer == 8
    assert cfg.patterns == ("needle_tail",)


def test_select_critical_tokens_filters_needles_and_splits_needle_tails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    _write(
        run_dir / "tables" / "massive_tokens_outside_needles_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,2,12,tok12,99\n"  # inside needle; must be ignored despite high score
        "0,24,4,14,tok14,90\n"  # first 5 tokens are excluded
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n"
        "0,24,16,26,tok26,80\n"  # last 5 tokens are excluded
        "0,12,9,19,tok19,100\n",
    )
    _write(
        run_dir / "tables" / "massive_tokens_all.csv",
        "example_idx,layer,position,token_id,token,norm_ratio_to_median\n"
        "0,24,2,12,tok12,99\n"
        "0,24,4,14,tok14,90\n"
        "0,24,7,17,tok17,30\n"
        "0,24,8,18,tok18,20\n"
        "0,24,16,26,tok26,80\n"
        "0,12,9,19,tok19,100\n",
    )
    _write(
        run_dir / "tables" / "attention_sinks_topk.csv",
        "example_idx,layer,head,position,token_id,token,received_uniform_ratio\n"
        "0,24,0,4,14,tok14,50\n"  # first 5 tokens are excluded
        "0,24,0,6,16,tok16,4\n"
        "0,24,1,6,16,tok16,9\n"
        "0,24,0,3,13,tok13,100\n"  # inside needle
        "0,24,0,14,24,tok24,7\n",
    )
    _write(
        run_dir / "tables" / "inputs_0_measurements.csv",
        "position,layer,cosine_similarity\n"
        "0,24,0.7\n"  # first 5 tokens are excluded
        "1,24,0.1\n"  # inside needle
        "5,24,0.2\n"
        "9,24,0.4\n"
        "15,24,0.05\n"  # last 5 tokens are excluded
        "14,24,0.3\n",
    )
    input_ids = list(range(10, 30))
    needle_segments = [
        {"needle_id": "n0", "start": 1, "end": 4, "positions": [1, 2, 3]},
        {"needle_id": "n1", "start": 18, "end": 20, "positions": [18, 19]},
    ]
    cfg = AblationConfig(num_critical_tokens=3, critical_token_calc_layer=24)

    selected = select_critical_tokens(
        run_dir=run_dir,
        example_id=0,
        cfg=cfg,
        input_ids=input_ids,
        needle_segments=needle_segments,
        tokenizer=FakeTokenizer(),
    )

    expected_patterns = {
        "massive_activation",
        "attention_sink",
        "needle_sensitive",
        "massive_activation_all",
        "attention_sink_all",
        "needle_sensitive_all",
        "needle_span_0",
        "needle_span_1",
        "needle_tail_0",
        "needle_tail_1",
    }
    assert set(selected) == expected_patterns
    assert [r["position"] for r in selected["massive_activation"]] == [7, 8]
    assert [r["position"] for r in selected["massive_activation_all"]] == [2, 4, 16]
    assert selected["attention_sink"][0]["position"] == 6
    assert selected["attention_sink"][0]["score"] == 9.0
    assert [r["position"] for r in selected["attention_sink_all"]] == [3, 4, 6]
    assert [r["position"] for r in selected["needle_sensitive"]] == [5, 14, 9]
    assert [r["position"] for r in selected["needle_sensitive_all"]] == [15, 1, 5]
    assert [r["position"] for r in selected["needle_span_0"]] == [1, 2, 3]
    assert [r["position"] for r in selected["needle_span_1"]] == [18, 19]
    assert [r["position"] for r in selected["needle_tail_0"]] == [4, 5, 6]
    # Corner case: second needle ends at sequence length, so its tail is empty.
    assert selected["needle_tail_1"] == []
    assert (run_dir / "tables" / "ablation" / "needle_sensitive_tokens_outside_needles_all.csv").exists()


def test_massive_activation_all_missing_table_returns_empty_selection(
    tmp_path: Path, capsys
) -> None:
    _write(
        tmp_path / "tables" / "inputs_0_measurements.csv",
        "position,layer,cosine_similarity\n5,24,0.2\n",
    )
    cfg = AblationConfig(
        patterns=("massive_activation_all",),
        num_critical_tokens=3,
        critical_token_calc_layer=24,
    )

    selected = select_critical_tokens(
        run_dir=tmp_path,
        example_id=0,
        cfg=cfg,
        input_ids=list(range(10)),
        needle_segments=[],
        tokenizer=FakeTokenizer(),
    )

    assert selected["massive_activation_all"] == []
    assert "Missing selection table for massive_activation_all" in capsys.readouterr().out


def test_replacement_pool_and_ablated_input_are_deterministic(tmp_path: Path) -> None:
    haystack = tmp_path / "hay"
    haystack.mkdir()
    (haystack / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    pool1 = build_irrelevant_token_pool(
        tokenizer=FakeTokenizer(), haystack_dir=haystack, pool_size=5, seed=123
    )
    pool2 = build_irrelevant_token_pool(
        tokenizer=FakeTokenizer(), haystack_dir=haystack, pool_size=5, seed=123
    )

    assert pool1 == pool2
    replacements = replacement_tokens_for_k(pool=pool1, k=2, seed=4, pattern="p")
    ablated = make_ablated_input_ids([10, 11, 12, 13], [1, 3], replacements)

    assert ablated.shape == (1, 4)
    assert ablated[0, 1].item() == replacements[0]
    assert ablated[0, 3].item() == replacements[1]


def test_run_ablation_generation_adds_binary_baseline_accuracy(tmp_path: Path) -> None:
    row = {
        "id": "row0",
        "task_type": "argmax",
        "query": "q",
        "gold_answer": {"city": "Paris", "score": 91},
    }
    selected = {
        "needle_tail_0": [
            {
                "pattern": "needle_tail_0",
                "rank": 1,
                "position": 1,
                "token_id": 11,
                "token": "tok11",
            }
        ]
    }

    predictions, results = run_ablation_generation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        row=row,
        example_id=0,
        input_ids=torch.tensor([[10, 11, 12]]),
        selected=selected,
        replacement_pool=[21, 22, 23],
        cfg=AblationConfig(num_critical_tokens=1),
        out_dir=tmp_path,
    )

    assert results[0]["pattern"] == "baseline"
    assert results[0]["k"] == 0
    assert results[0]["exact_match"] is True
    assert results[0]["accuracy"] == 1.0
    assert results[1]["pattern"] == "needle_tail_0"
    assert results[1]["accuracy"] in {0.0, 1.0}
    assert predictions[0]["model_output_text"] == '{"city":"Paris","score":91}'
    assert (tmp_path / "generations" / "baseline.txt").exists()


def test_summarize_ablation_results_all_is_exported_from_package() -> None:
    from single_example import summarize_ablation_results_all as exported

    assert exported is summarize_ablation_results_all


def test_summarize_ablation_results_all_averages_by_pattern_and_k(tmp_path: Path) -> None:
    header = "example_id,row_id,pattern,k,accuracy\n"
    _write(
        tmp_path / "example_id_0" / "tables" / "ablation" / "ablation_results.csv",
        header + "0,row0,baseline,0,1.0\n0,row0,p,1,0.0\n",
    )
    _write(
        tmp_path / "example_id_1" / "tables" / "ablation" / "ablation_results.csv",
        header + "1,row1,baseline,0,0.0\n1,row1,p,1,1.0\n",
    )

    summary_path = summarize_ablation_results_all(run_dir=tmp_path)

    assert summary_path == tmp_path / "ablation_results_all.csv"
    assert summary_path.read_text(encoding="utf-8").splitlines() == [
        "Pattern,k,accuracy",
        "baseline,0,0.5",
        "p,1,0.5",
    ]
