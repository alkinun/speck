import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts import trained_decode_sentinel as sentinel

root = Path(__file__).parents[1]
contract_path = root / "research" / "paper-1" / "cuda_decode_trained_sentinel.json"


def test_checked_trained_decode_sentinel_is_frozen_and_pinned():
    _, contract = sentinel.load_contract(contract_path)

    assert contract["data"]["prefix_lengths"] == [8, 64, 512]
    assert contract["greedy_generation_tokens"] == 32
    assert [value["seed"] for value in contract["checkpoints"]] == [42, 42, 43, 44]


def test_trained_decode_sentinel_rejects_tolerance_change(tmp_path):
    copied = tmp_path / "research" / "paper-1"
    copied.mkdir(parents=True)
    value = deepcopy(json.loads(contract_path.read_text(encoding="utf-8")))
    value["trigger_result"] = str(root / value["trigger_result"])
    value["numerical_contract"]["absolute_tolerance"] = 0.1
    path = copied / contract_path.name
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="tolerance"):
        sentinel.load_contract(path)


class ToyState:
    def __init__(self):
        self.tokens = None


class ToyModel:
    def state(self, **kwargs):
        return ToyState()

    def __call__(self, tokens, state=None, last_token_only=False):
        if state is not None:
            state.tokens = tokens if state.tokens is None else torch.cat((state.tokens, tokens), 1)
            tokens = state.tokens
        total = tokens.sum(dim=1) % 5
        logits = torch.zeros(tokens.size(0), 1, 5)
        logits[torch.arange(tokens.size(0)), 0, total] = 1
        return logits


def test_generation_comparisons_preserve_common_and_free_histories():
    model = ToyModel()
    prompt = torch.tensor([[1, 2, 3]])

    common = sentinel.compare_common_history(model, prompt, 4, 0, 0)
    free = sentinel.compare_free_running(model, prompt, 4)

    assert common["argmax_agreement"] == 1
    assert common["first_argmax_disagreement_step"] is None
    assert free["complete_sequence_match"]
    assert free["first_argmax_disagreement_step"] is None
