import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from speck.tokenizer_pilot_evaluation import (
    document_nll,
    iter_evaluation_documents,
    validate_evaluation_tokenizer,
)


class FakeModel:
    def __init__(self, maximum=32):
        self.config = SimpleNamespace(max_position_embeddings=maximum)
        self.lengths = []
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def state(self, length, device):
        self.state_value = SimpleNamespace(position=0, length=length, device=torch.device(device))
        return self.state_value

    def __call__(self, inputs, targets, *, state, loss_reduction):
        assert loss_reduction == "sum"
        assert inputs.shape == targets.shape
        state.position += inputs.size(1)
        self.lengths.append(inputs.size(1))
        return targets.float().sum()


class NonfiniteModel(FakeModel):
    def __call__(self, inputs, targets, *, state, loss_reduction):
        return torch.tensor(float("nan"))


class FakeTokenizer:
    vocab_size = 10
    bos_id = 1
    eos_id = 2

    def fingerprint(self):
        return "tokenizer-hash"


def test_document_nll_preserves_state_across_chunks():
    model = FakeModel()

    nll = document_nll(model, [1, 4, 5, 2], device="cpu", chunk_tokens=2)

    assert nll == 11
    assert model.lengths == [2, 1]
    assert model.state_value.position == 3


def test_document_nll_rejects_invalid_lengths_and_nonfinite_output():
    with pytest.raises(ValueError, match="at least two"):
        document_nll(FakeModel(), [1], device="cpu")
    with pytest.raises(ValueError, match="exceeds"):
        document_nll(FakeModel(maximum=2), [1, 3, 4, 2], device="cpu")

    with pytest.raises(FloatingPointError, match="non-finite"):
        document_nll(NonfiniteModel(), [1, 3, 2], device="cpu")


def test_evaluation_documents_verify_identity_content_and_totals(tmp_path):
    text = "hello"
    document_id = hashlib.sha256(text.encode()).hexdigest()
    path = tmp_path / "eval-web.jsonl"
    path.write_text(
        json.dumps({"category": "web", "source_content_sha256": document_id, "text": text}) + "\n"
    )
    category = {
        "id": "web",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "documents": 1,
        "utf8_bytes": 5,
    }

    assert list(iter_evaluation_documents(category)) == [
        {"document_id": document_id, "text": text, "utf8_bytes": 5}
    ]

    category["documents"] = 2
    with pytest.raises(ValueError, match="totals mismatch"):
        list(iter_evaluation_documents(category))


def test_evaluation_tokenizer_must_match_run_identity():
    run = {
        "tokenizer": {
            "model": {"sha256": "tokenizer-hash"},
            "vocab_size": 10,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    }
    validate_evaluation_tokenizer(run, FakeTokenizer())
    run["tokenizer"]["vocab_size"] = 11
    with pytest.raises(ValueError, match="differs"):
        validate_evaluation_tokenizer(run, FakeTokenizer())
