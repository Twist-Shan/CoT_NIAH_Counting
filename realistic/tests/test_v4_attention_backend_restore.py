"""Backend switches must not change later forwards on composite models."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from realistic_niah_v4.modeling import _temporary_attention_backend


class CompositeConfig(transformers.PreTrainedConfig):
    sub_configs = {"text_config": transformers.PreTrainedConfig,
                   "vision_config": transformers.PreTrainedConfig}

    def __init__(self):
        super().__init__()
        self.text_config = transformers.PreTrainedConfig()
        self.vision_config = transformers.PreTrainedConfig()

    def get_text_config(self, **kwargs):
        return getattr(self, "text_config", self)


def make_model():
    config = CompositeConfig()
    config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "eager"
    return SimpleNamespace(config=config)


def state(model):
    root = model.config
    return tuple(c._attn_implementation for c in [root, root.text_config, root.vision_config])


def test_composite_backend_restores_all_original_values():
    model = make_model()
    before = state(model)
    with _temporary_attention_backend(model, "eager"):
        assert state(model) == ("eager",) * 3
    assert state(model) == before


def test_nested_backend_and_exception_restore_outer_state():
    model = make_model()
    before = state(model)
    with _temporary_attention_backend(model, "eager"):
        with pytest.raises(RuntimeError, match="query failed"):
            with _temporary_attention_backend(model, "sdpa"):
                assert state(model) == ("sdpa",) * 3
                raise RuntimeError("query failed")
        assert state(model) == ("eager",) * 3
    assert state(model) == before


def test_single_config_restores_without_children():
    model = SimpleNamespace(config=transformers.PreTrainedConfig())
    model.config._attn_implementation = "sdpa"
    with _temporary_attention_backend(model, "eager"):
        assert model.config._attn_implementation == "eager"
    assert model.config._attn_implementation == "sdpa"
