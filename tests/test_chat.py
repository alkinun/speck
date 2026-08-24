import json

import pytest

from speck.chat import CHAT_TEMPLATE, ChatTokenizer


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
