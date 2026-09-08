from types import SimpleNamespace

import pytest
import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
)
from speck.generation import generate_tokens
from speck.model import SpeckForCausalLM


class ScriptedModel:
    config = SimpleNamespace(max_position_embeddings=16)

    def __init__(self, predictions):
        self.predictions = predictions
        self.inputs = []
        self.states = []

    def state(self, **kwargs):
        state = SimpleNamespace(**kwargs)
        self.states.append(state)
        return state

    def __call__(self, inputs, *, state, last_token_only):
        assert torch.is_inference_mode_enabled()
        assert last_token_only
        assert state is self.states[0]
        predicted = self.predictions[len(self.inputs)]
        self.inputs.append(inputs.tolist())
        logits = torch.full((1, 1, 8), -10.0)
        logits[0, 0, predicted] = 10.0
        return logits


@pytest.mark.parametrize("temperature", (0.0, 0.8))
def test_generation_prefills_once_and_stops_at_token_budget(temperature):
    model = ScriptedModel([3, 4, 5])
    generated = generate_tokens(
        model, [1, 6], max_tokens=3, eos_token_id=2, device="cpu", temperature=temperature, top_k=1
    )
    assert generated == [3, 4, 5]
    assert model.inputs == [[[1, 6]], [[3]], [[4]]]
    assert len(model.states) == 1


@pytest.mark.parametrize("predictions,expected", (([2], []), ([3, 2], [3])))
def test_generation_stops_at_eos_without_emitting_or_feeding_it(predictions, expected):
    model = ScriptedModel(predictions)
    assert generate_tokens(model, [1], max_tokens=5, eos_token_id=2, device="cpu") == expected
    assert len(model.inputs) == len(predictions)


def test_generation_clamps_top_k_to_vocabulary_size():
    model = ScriptedModel([3])
    assert generate_tokens(
        model, [1], max_tokens=1, eos_token_id=2, device="cpu", temperature=0.01, top_k=100
    ) == [3]


@pytest.mark.parametrize(
    "overrides",
    (
        {"tokens": []},
        {"tokens": list(range(16))},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": float("nan")},
        {"temperature": -1.0},
        {"top_k": 0},
    ),
)
def test_invalid_generation_fails_before_allocating_cache(overrides):
    model = ScriptedModel([])
    arguments = {"tokens": [1], "max_tokens": 2, "eos_token_id": 2, "device": "cpu"}
    with pytest.raises(ValueError):
        generate_tokens(model, **(arguments | overrides))
    assert model.states == []


def test_cached_generation_matches_full_prefix_greedy_reference():
    torch.manual_seed(42)
    config = ArchitectureConfig(
        blocks=(BlockGroup(BlockConfig(8, (StageConfig((AttentionSpec(4, 1),)),))),),
        embedding_size=8,
        vocab_size=8,
        max_position_embeddings=16,
    )
    model = SpeckForCausalLM(config).eval()
    model.init_weights()
    prompt = [1, 3, 4]
    reference = []
    with torch.inference_mode():
        for _ in range(4):
            logits = model(torch.tensor([prompt + reference]))
            token = logits[0, -1].argmax().item()
            if token == config.eos_token_id:
                break
            reference.append(token)
    assert (
        generate_tokens(model, prompt, max_tokens=4, eos_token_id=config.eos_token_id, device="cpu")
        == reference
    )
