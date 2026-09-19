from types import SimpleNamespace

import pytest
import torch

from counting.steering import (
    SteeringExample,
    SteeringVector,
    build_needle_span_steering_targets,
    compute_last_token_probe_sigma,
    compute_needle_span_probe_sigma,
    generate_with_needle_span_counting_steering,
    generate_with_counting_steering,
    load_contrastive_success_vector,
    load_counting_steering_vector,
    load_counterfactual_count_vector,
    load_ridge_counting_vector,
    select_steering_examples,
    summarize_steering_results,
)


class Encoded(dict):
    def __init__(self, input_ids):
        super().__init__()
        self.input_ids = input_ids


class TinyTokenizer:
    eos_token_id = 9
    is_fast = False

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        del tokenize, add_generation_prompt, enable_thinking
        return " ".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        del kwargs
        ids = [int(part) for part in text.split()]
        return Encoded(torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(x)) for x in torch.as_tensor(ids).flatten())


class TinyLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class SumToLastLayer(torch.nn.Module):
    def forward(self, hidden):
        out = hidden.clone()
        out[:, -1, 0] = hidden[:, :, 0].sum(dim=1)
        return out


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 2)
        with torch.no_grad():
            self.embed.weight.zero_()
            for idx in range(10):
                self.embed.weight[idx, 0] = float(idx)
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([TinyLayer()]))
        self.lm_head = torch.nn.Linear(2, 10)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.bias.zero_()
            self.lm_head.bias[0] = 0.1
            self.lm_head.weight[1, 0] = 1.0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        if output_hidden_states:
            return SimpleNamespace(logits=logits, hidden_states=tuple(states))
        return SimpleNamespace(logits=logits)


class TinyContextModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([TinyLayer(), SumToLastLayer()])
        )


def test_select_steering_examples_caps_total_and_balances_groups():
    rows = [{"id": str(i), "messages": []} for i in range(8)]
    scored = [
        {"id": str(i), "exact_match": i in {0, 1, 2, 3, 4}}
        for i in range(8)
    ]

    selected, summary = select_steering_examples(rows, scored, max_total=6)

    assert len(selected) == 6
    assert summary["num_selected"] == 6
    assert sum(ex.group == "successful" for ex in selected) == 3
    assert sum(ex.group == "unsuccessful" for ex in selected) == 3


def test_select_steering_examples_fills_from_available_group():
    rows = [{"id": str(i), "messages": []} for i in range(5)]
    scored = [{"id": str(i), "exact_match": True} for i in range(5)]

    selected, summary = select_steering_examples(rows, scored, max_total=4)

    assert len(selected) == 4
    assert summary["num_unsuccessful"] == 0
    assert all(ex.group == "successful" for ex in selected)


def test_load_ridge_counting_vector_normalizes(tmp_path):
    path = tmp_path / "ridge_probe_layer_1.pt"
    torch.save({"coef": torch.tensor([3.0, 4.0])}, path)

    vector = load_ridge_counting_vector(path, layer=1)

    assert vector.norm == pytest.approx(5.0)
    assert vector.vector.tolist() == pytest.approx([0.6, 0.8])




def test_load_ridge_counting_vector_converts_standardized_coef_to_hidden_space(tmp_path):
    path = tmp_path / "ridge_probe_layer_1.pt"
    torch.save(
        {
            "coef": torch.tensor([2.0, 8.0]),
            "feature_scale": torch.tensor([2.0, 4.0]),
            "standardize": True,
        },
        path,
    )

    vector = load_ridge_counting_vector(path, layer=1)

    # Hidden-space direction is coef / scale = [1, 2], then unit-normalized.
    assert vector.source_space == "hidden_from_standardized_ridge"
    assert vector.standardized_probe is True
    assert vector.norm == pytest.approx(5 ** 0.5)
    assert vector.vector.tolist() == pytest.approx([1 / (5 ** 0.5), 2 / (5 ** 0.5)])


def test_load_contrastive_success_vector_uses_saved_direction(tmp_path):
    vector_dir = tmp_path / "contrastive_success"
    vector_dir.mkdir()
    path = vector_dir / "contrastive_success_layer_1.pt"
    torch.save(
        {
            "direction": torch.tensor([0.0, 2.0]),
            "raw_norm": 7.0,
            "method": "raw_mean_difference",
        },
        path,
    )

    vector = load_contrastive_success_vector(path, layer=1)
    via_source = load_counting_steering_vector(
        tmp_path, layer=1, vector_source="contrastive-success"
    )

    assert vector.source_space == "contrastive_success"
    assert vector.norm == pytest.approx(7.0)
    assert vector.vector.tolist() == pytest.approx([0.0, 1.0])
    assert via_source.vector.tolist() == pytest.approx([0.0, 1.0])


def test_load_counterfactual_count_vector_uses_saved_direction(tmp_path):
    vector_dir = tmp_path / "counterfactual"
    vector_dir.mkdir()
    path = vector_dir / "counterfactual_count_layer_1.pt"
    torch.save(
        {
            "direction": torch.tensor([3.0, 0.0]),
            "raw_norm": 9.0,
            "method": "counterfactual_count_difference",
        },
        path,
    )

    vector = load_counterfactual_count_vector(path, layer=1)
    via_source = load_counting_steering_vector(
        tmp_path, layer=1, vector_source="counterfactual"
    )

    assert vector.source_space == "counterfactual_count"
    assert vector.norm == pytest.approx(9.0)
    assert vector.vector.tolist() == pytest.approx([1.0, 0.0])
    assert via_source.vector.tolist() == pytest.approx([1.0, 0.0])


def test_compute_last_token_probe_sigma_uses_requested_hidden_state_layer():
    model = TinyModel()
    tokenizer = TinyTokenizer()
    examples = [
        SteeringExample(0, "a", "successful", {"messages": [{"role": "user", "content": "0 1"}]}),
        SteeringExample(1, "b", "unsuccessful", {"messages": [{"role": "user", "content": "0 3"}]}),
    ]
    vector = SteeringVector(layer=1, vector=torch.tensor([1.0, 0.0]), norm=1.0, source_path="x")

    sigma = compute_last_token_probe_sigma(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        steering_vector=vector,
    )

    assert sigma["projection_values"] == pytest.approx([1.0, 3.0])
    assert sigma["sigma"] == pytest.approx(1.0)


def test_generate_with_counting_steering_patches_current_last_token():
    model = TinyModel()
    tokenizer = TinyTokenizer()
    input_ids = torch.tensor([[0, 0]], dtype=torch.long)

    baseline = generate_with_counting_steering(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        layer=1,
        steering_vector=None,
        beta=0.0,
        sigma=0.0,
        max_new_tokens=1,
    )
    steered = generate_with_counting_steering(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        layer=1,
        steering_vector=torch.tensor([1.0, 0.0]),
        beta=1.0,
        sigma=1.0,
        max_new_tokens=1,
    )

    assert baseline == "0"
    assert steered == "1"


def test_build_needle_span_targets_uses_actual_matching_inserted_needles():
    tokenizer = TinyTokenizer()
    row = {
        "id": "row0",
        "messages": [{"role": "user", "content": "0 3 4 0 5 6 0"}],
        "gold_answer": {"count": 1},
        "relevant_records": [{"needle_id": "N0"}],
        "needles": [
            {"needle_id": "N0", "tokens": [3, 4], "decoded_text": "3 4", "is_inserted": True},
            {"needle_id": "N1", "tokens": [7, 8], "decoded_text": "7 8", "is_inserted": False},
        ],
        "realized_insertions": [
            {"needle_id": "N0", "requested_position": 1, "final_position": 1, "tokens": [3, 4], "decoded_text": "3 4"}
        ],
    }
    examples = [SteeringExample(0, "row0", "successful", row, {"exact_match": True})]

    targets, summary = build_needle_span_steering_targets(tokenizer, examples)

    assert summary["num_targets"] == 1
    assert targets[0].needle_id == "N0"
    assert targets[0].span_start == 1
    assert targets[0].span_end == 3


def test_compute_needle_span_sigma_uses_span_positions():
    model = TinyModel()
    NeedleTarget = SimpleNamespace
    targets = [
        NeedleTarget(input_ids=torch.tensor([[0, 1, 0]]), span_start=1, span_end=2, span_length=1, row_id="a", needle_id="N0", dataset_index=0, needle_ordinal=0),
        NeedleTarget(input_ids=torch.tensor([[0, 3, 0]]), span_start=1, span_end=2, span_length=1, row_id="b", needle_id="N0", dataset_index=1, needle_ordinal=0),
    ]
    vector = SteeringVector(layer=1, vector=torch.tensor([1.0, 0.0]), norm=1.0, source_path="x")

    sigma = compute_needle_span_probe_sigma(
        model=model,
        targets=targets,
        steering_vector=vector,
    )

    assert sigma["num_positions"] == 2
    assert sigma["projection_mean"] == pytest.approx(2.0)
    assert sigma["sigma"] == pytest.approx(1.0)


def test_generate_with_needle_span_steering_patches_prompt_span():
    model = TinyContextModel()
    tokenizer = TinyTokenizer()
    input_ids = torch.tensor([[0, 0]], dtype=torch.long)

    baseline = generate_with_needle_span_counting_steering(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        layer=1,
        span_start=0,
        span_end=1,
        steering_vector=None,
        beta=0.0,
        sigma=0.0,
        max_new_tokens=1,
    )
    steered = generate_with_needle_span_counting_steering(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        layer=1,
        span_start=0,
        span_end=1,
        steering_vector=torch.tensor([1.0, 0.0]),
        beta=1.0,
        sigma=1.0,
        max_new_tokens=1,
    )

    assert baseline == "0"
    assert steered == "1"


def test_summarize_steering_results_groups_accuracy_and_shift():
    rows = [
        {"layer": 1, "beta": 0.5, "example_group": "successful", "baseline_exact_match": True, "steered_exact_match": False, "baseline_prediction_count": 2, "steered_prediction_count": 3},
        {"layer": 1, "beta": 0.5, "example_group": "successful", "baseline_exact_match": True, "steered_exact_match": True, "baseline_prediction_count": 2, "steered_prediction_count": 1},
    ]

    summary = summarize_steering_results(rows)

    assert summary == [
        {
            "layer": 1,
            "beta": 0.5,
            "example_group": "successful",
            "num_examples": 2,
            "baseline_exact_match_count": 2,
            "baseline_accuracy": 1.0,
            "steered_exact_match_count": 1,
            "steered_accuracy": 0.5,
            "mean_predicted_count_shift": 0.0,
            "baseline_parse_failure_rate": 0.0,
            "steered_parse_failure_rate": 0.0,
        }
    ]
