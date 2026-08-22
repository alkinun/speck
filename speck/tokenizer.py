"""Prepare and load the Speck SentencePiece tokenizer."""

import hashlib
import json
import os
from typing import cast

import sentencepiece as sentencepiece
from huggingface_hub import hf_hub_download

from speck.common import base_dir

default_repo = "mistralai/Mistral-7B-v0.1"
default_revision = "27d67f1b5f57dc0953326b2601d68371d40ea8da"


class Tokenizer:
    """Wrap a prepared SentencePiece model for Speck tokenization."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.processor = sentencepiece.SentencePieceProcessor(model_file=self.model_path)

    @classmethod
    def load(cls, directory=None, repo=None, revision=None, filename="tokenizer.model"):
        directory = directory or os.path.join(base_dir(), "tokenizer")
        model_path = os.path.join(directory, filename)
        metadata_path = os.path.join(directory, "tokenizer_metadata.json")
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("tokenizer is not prepared; run scripts.tokenizer_prepare")
        tokenizer = cls(model_path)
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["fingerprint"] != tokenizer.fingerprint():
            raise ValueError("tokenizer fingerprint mismatch")
        if repo is not None and metadata.get("repo") != repo:
            raise ValueError("prepared tokenizer repository does not match the experiment")
        if revision is not None and metadata.get("revision") != revision:
            raise ValueError("prepared tokenizer revision does not match the experiment")
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
        tokens = self.processor.encode(text, out_type=int)
        return ([self.bos_id] if bos else []) + tokens + ([self.eos_id] if eos else [])

    def encode_batch(self, texts, bos=False, eos=False):
        return cast(
            list[list[int]],
            self.processor.encode(
                texts,
                out_type=int,
                add_bos=bos,
                add_eos=eos,
                num_threads=min(8, os.cpu_count() or 1),
            ),
        )

    def decode(self, tokens):
        return self.processor.decode(tokens)

    def fingerprint(self):
        with open(self.model_path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()


def prepare(
    directory=None, repo=default_repo, revision=default_revision, filename="tokenizer.model"
):
    directory = directory or os.path.join(base_dir(), "tokenizer")
    os.makedirs(directory, exist_ok=True)
    model_path = hf_hub_download(repo, filename, revision=revision, local_dir=directory)
    tokenizer = Tokenizer(model_path)
    metadata = {
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "vocab_size": tokenizer.vocab_size,
        "fingerprint": tokenizer.fingerprint(),
    }
    with open(os.path.join(directory, "tokenizer_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return tokenizer


def get_tokenizer(**config):
    return Tokenizer.load(**config)
