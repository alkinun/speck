"""pinned mistral sentencepiece tokenizer."""

import hashlib
import json
import os

import sentencepiece as sentencepiece
from huggingface_hub import hf_hub_download

from speck.common import base_dir


repo = "mistralai/Mistral-7B-v0.1"
revision = "27d67f1b5f57dc0953326b2601d68371d40ea8da"


class Tokenizer:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.processor = sentencepiece.SentencePieceProcessor(model_file=self.model_path)
        if self.vocab_size != 32000 or self.bos_id != 1 or self.eos_id != 2:
            raise ValueError("unexpected mistral tokenizer configuration")

    @classmethod
    def load(cls, directory=None):
        directory = directory or os.path.join(base_dir(), "tokenizer")
        model_path = os.path.join(directory, "tokenizer.model")
        metadata_path = os.path.join(directory, "tokenizer_metadata.json")
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("mistral tokenizer is not prepared; run scripts.tokenizer_prepare")
        tokenizer = cls(model_path)
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["fingerprint"] != tokenizer.fingerprint():
            raise ValueError("tokenizer fingerprint mismatch")
        return tokenizer

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    @property
    def bos_id(self):
        return self.processor.bos_id()

    @property
    def eos_id(self):
        return self.processor.eos_id()

    def encode(self, text, bos=False, eos=False):
        if isinstance(text, str):
            tokens = self.processor.encode(text, out_type=int)
            return ([self.bos_id] if bos else []) + tokens + ([self.eos_id] if eos else [])
        return [self.encode(row, bos, eos) for row in text]

    def decode(self, tokens):
        return self.processor.decode(tokens)

    def fingerprint(self):
        with open(self.model_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()


def prepare(directory=None):
    directory = directory or os.path.join(base_dir(), "tokenizer")
    os.makedirs(directory, exist_ok=True)
    model_path = hf_hub_download(repo, "tokenizer.model", revision=revision, local_dir=directory)
    tokenizer = Tokenizer(model_path)
    metadata = {
        "repo": repo,
        "revision": revision,
        "vocab_size": tokenizer.vocab_size,
        "fingerprint": tokenizer.fingerprint(),
    }
    with open(os.path.join(directory, "tokenizer_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return tokenizer


def get_tokenizer():
    return Tokenizer.load()
