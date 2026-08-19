import json

import speck.tokenizer as tokenizer_module
from speck.tokenizer import Tokenizer


class Processor:
    def __init__(self, model_file):
        self.model_file = model_file

    def vocab_size(self):
        return 32000

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def encode(self, text, out_type=int):
        return [byte + 3 for byte in text.encode()]

    def decode(self, tokens):
        return bytes(token - 3 for token in tokens).decode()


def test_mistral_tokenizer_roundtrip_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenizer_module.sentencepiece, "SentencePieceProcessor", Processor)
    model_path = tmp_path / "tokenizer.model"
    model_path.write_bytes(b"mistral-tokenizer")
    tokenizer = Tokenizer(model_path)
    assert tokenizer.bos_id == 1 and tokenizer.eos_id == 2
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert tokenizer.encode("hello", bos=True, eos=True)[::6] == [1, 2]
    (tmp_path / "tokenizer_metadata.json").write_text(json.dumps({
        "fingerprint": tokenizer.fingerprint()
    }))
    assert Tokenizer.load(tmp_path).fingerprint() == tokenizer.fingerprint()
