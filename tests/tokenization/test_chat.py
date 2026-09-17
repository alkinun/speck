import json

import pytest

from speck.tokenization.chat import CHAT_TEMPLATE, ChatTokenizer


class BaseTokenizer:
    vocab_size = 300
    bos_id = 1
    eos_id = 2

    def __init__(self, model_path):
        self.model_path = str(model_path)

    def encode(self, text):
        return [byte + 3 for byte in text.encode()]

    def decode(self, tokens):
        return bytes(token - 3 for token in tokens).decode()

    def fingerprint(self):
        return "base-tokenizer"


def test_chat_template_tokens_and_assistant_mask(tmp_path):
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(model_path))
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    tokens, mask = tokenizer.encode_messages(messages)

    assert tokenizer.vocab_size == 303
    assert tokenizer.role_ids == {"system": 300, "user": 301, "assistant": 302}
    assert tokens[0] == tokenizer.bos_id
    assistant = tokens.index(tokenizer.role_ids["assistant"])
    assert not any(mask[: assistant + 1])
    content_start = assistant + 1 + len(tokenizer.base.encode("\n"))
    assert not any(mask[assistant + 1 : content_start])
    assert all(mask[content_start : assistant + 1 + len(tokenizer.base.encode("\nHi"))])
    assistant_eos = assistant + 1 + len(tokenizer.base.encode("\nHi"))
    assert tokens[assistant_eos] == tokenizer.eos_id and mask[assistant_eos]
    assert tokenizer.render(messages) == (
        "<s><|system|>\nBe concise.</s>\n<|user|>\nHello</s>\n<|assistant|>\nHi</s>\n"
    )


def test_legacy_chat_tokenizer_fingerprint_is_stable_for_prepared_data(tmp_path):
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(model_path), format_version=1)

    assert (
        tokenizer.fingerprint()
        == "43f9e7e14205419cf683d42e12cb8d407f3be73a798e0c7df55fdec025a19d29"
    )


def test_generation_prompt_and_role_validation(tmp_path):
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(model_path))
    user = [{"role": "user", "content": "Hello"}]

    tokens, mask = tokenizer.encode_messages(user, add_generation_prompt=True)

    assert tokens[-len(tokenizer.base.encode("\n")) - 1] == tokenizer.role_ids["assistant"]
    assert not any(mask)
    assert tokenizer.render(user, add_generation_prompt=True).endswith("<|assistant|>\n")
    with pytest.raises(ValueError, match="alternate"):
        tokenizer.encode_messages(user + [{"role": "user", "content": "Again"}])
    with pytest.raises(ValueError, match="final user"):
        tokenizer.encode_messages(
            user + [{"role": "assistant", "content": "Hi"}],
            add_generation_prompt=True,
        )


def test_special_token_text_is_rejected(tmp_path):
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(model_path))

    with pytest.raises(ValueError, match="reserved chat token"):
        tokenizer.encode_messages(
            [{"role": "user", "content": "Print <|assistant|> and </s>."}],
            add_generation_prompt=True,
        )


def test_save_chat_tokenizer_artifact(tmp_path):
    model_path = tmp_path / "base.model"
    model_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(model_path))
    output = tmp_path / "saved"

    tokenizer.save_pretrained(output)

    config = json.loads((output / "tokenizer_config.json").read_text())
    assert config["chat_template"] == CHAT_TEMPLATE
    assert config["split_special_tokens"] is False
    assert set(config["added_tokens_decoder"]) == {"300", "301", "302"}
    assert (output / "chat_template.jinja").read_text() == CHAT_TEMPLATE
    assert (output / "tokenizer.model").read_bytes() == b"sentencepiece"


def test_context_only_assistant_turn_is_never_supervised(tmp_path):
    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Context only", "weight": 0},
        {"role": "user", "content": "Next"},
        {"role": "assistant", "content": "Target", "weight": 1},
    ]
    tokens, mask = tokenizer.encode_messages(messages)
    supervised = [token for token, use in zip(tokens, mask, strict=True) if use]
    assert supervised == tokenizer.base.encode("Target") + [tokenizer.eos_id]
    assert "Context only" in tokenizer.render(messages)
    restored = ChatTokenizer.from_metadata(tokenizer.base, tokenizer.metadata())
    assert restored.encode_messages(messages) == (tokens, mask)
    legacy = ChatTokenizer(tokenizer.base, format_version=1)
    assert legacy.fingerprint() != tokenizer.fingerprint()
    with pytest.raises(ValueError, match="legacy"):
        legacy.encode_messages(messages)


@pytest.mark.parametrize("weight", [None, True, -1, 0.5, "0", float("nan")])
def test_invalid_message_weights_are_rejected(tmp_path, weight):
    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    with pytest.raises(ValueError, match="weight"):
        tokenizer.encode_messages(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "weight": weight},
            ]
        )


@pytest.mark.parametrize(
    "field,value", [("tool_calls", [{"id": "call"}]), ("reasoning_content", "hidden")]
)
def test_unserialized_fields_are_not_silently_dropped(tmp_path, field, value):
    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    with pytest.raises(ValueError):
        tokenizer.encode_messages(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", field: value},
            ]
        )


def test_exported_template_preserves_text_and_weighted_generation_spans(tmp_path):
    pytest.importorskip("transformers")
    from transformers.utils.chat_template_utils import (
        _compile_jinja_template,
        _render_with_assistant_indices,
    )

    tokenizer = ChatTokenizer(BaseTokenizer(tmp_path / "tokenizer.model"))
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Context", "weight": 0},
        {"role": "user", "content": "Next"},
        {"role": "assistant", "content": "Target", "weight": 1},
    ]
    compiled = _compile_jinja_template(tokenizer.chat_template)
    rendered, spans = _render_with_assistant_indices(
        compiled, messages, None, None, False, bos_token="<s>", eos_token="</s>"
    )
    assert rendered == tokenizer.render(messages)
    assert [rendered[start:end] for start, end in spans] == ["Target</s>"]
    with pytest.raises(Exception, match="tool definitions"):
        compiled.render(messages=messages, tools=[{"name": "lookup"}])
