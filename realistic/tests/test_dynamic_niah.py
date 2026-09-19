from dataset_generation.dynamic_niah import DynamicNiahConfig, generate_dynamic_niah_instance


def test_dynamic_niah_reproducible_same_seeds() -> None:
    cfg = DynamicNiahConfig(
        tokenizer_name="simple",
        max_haystack_tokens=64,
        base_seed=7,
        haystack_seed=11,
        needle_global_seed=13,
        insertion_positions=[3, 8, 15],
        num_needles=3,
    )
    a = generate_dynamic_niah_instance(cfg)
    b = generate_dynamic_niah_instance(cfg)
    assert a.final_sequence_tokens == b.final_sequence_tokens
    assert a.gold_answer == b.gold_answer


def test_dynamic_niah_only_needle_1_changes_with_override_seed() -> None:
    base = DynamicNiahConfig(
        tokenizer_name="simple",
        max_haystack_tokens=64,
        base_seed=7,
        haystack_seed=11,
        needle_global_seed=13,
        insertion_positions=[3, 8, 15],
        num_needles=3,
    )
    override = DynamicNiahConfig(
        tokenizer_name="simple",
        max_haystack_tokens=64,
        base_seed=7,
        haystack_seed=11,
        needle_global_seed=13,
        insertion_positions=[3, 8, 15],
        num_needles=3,
        needle_seeds={1: 999},
    )
    a = generate_dynamic_niah_instance(base)
    b = generate_dynamic_niah_instance(override)

    by_id_a = {r["needle_id"]: r for r in a.realized_insertions}
    by_id_b = {r["needle_id"]: r for r in b.realized_insertions}

    assert by_id_a["N1"]["needle_tokens"] == by_id_b["N1"]["needle_tokens"]
    assert by_id_a["N3"]["needle_tokens"] == by_id_b["N3"]["needle_tokens"]
    assert by_id_a["N2"]["needle_tokens"] != by_id_b["N2"]["needle_tokens"]
