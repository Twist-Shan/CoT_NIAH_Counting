from dataset_generation.chat_templates import apply_generation_chat_template


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def test_apply_generation_chat_template_passes_enable_thinking_false():
    tokenizer = DummyTokenizer()
    messages = [{"role": "user", "content": "answer directly"}]

    rendered = apply_generation_chat_template(
        tokenizer, messages, thinking_mode=False
    )

    assert rendered == "rendered"
    assert tokenizer.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


def test_apply_generation_chat_template_passes_enable_thinking_true():
    tokenizer = DummyTokenizer()
    messages = [{"role": "user", "content": "think first"}]

    apply_generation_chat_template(tokenizer, messages, thinking_mode=True)

    assert tokenizer.calls[0][1]["enable_thinking"] is True
