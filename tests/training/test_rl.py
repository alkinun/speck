import json
import shutil

import torch

from speck.model import SpeckForCausalLM
from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    KimiDeltaAttentionSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.model.generation import sample_group
from speck.tokenization.chat import ChatTokenizer
from speck.training import rl
from speck.training.checkpoint import load_model, save
from tests.training.test_sft import BaseTokenizer


def tiny_model(vocab_size):
    torch.manual_seed(0)
    stages = (StageConfig((KimiDeltaAttentionSpec(4, 4, 1, 1),)), StageConfig((SwiGLUSpec(16),)))
    attention = (StageConfig((AttentionSpec(4, 1, rope_dim=0),)), StageConfig((SwiGLUSpec(16),)))
    config = ArchitectureConfig(
        (BlockGroup(BlockConfig(8, stages)), BlockGroup(BlockConfig(8, attention))),
        8,
        vocab_size=vocab_size,
        max_position_embeddings=64,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    return model


def test_group_advantages_are_centered_and_constant_groups_carry_none():
    assert torch.equal(rl.group_advantages([1.0, 1.0, 1.0]), torch.zeros(3))
    assert torch.equal(rl.group_advantages([1.0, 0.0, 0.0, 1.0]), torch.tensor([1.0, -1, -1, 1]))


def test_policy_loss_raises_the_likelihood_of_advantaged_completions():
    model = tiny_model(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    prompt, good, bad = [1, 5, 6], [7, 8, 11], [9, 10, 12]

    def nll(completion):
        with torch.no_grad():
            loss, _ = rl.policy_loss(model, prompt, [completion], torch.ones(1))
        return float(loss)

    before = nll(good), nll(bad)
    loss, tokens = rl.policy_loss(model, prompt, [good, bad], rl.group_advantages([1.0, 0.0]))
    assert tokens == 6
    (loss / tokens).backward()
    optimizer.step()
    assert nll(good) < before[0] and nll(bad) > before[1]


def test_sample_group_stops_each_completion_at_eos_and_is_seeded():
    model = tiny_model(16).eval()

    def draw():
        generator = torch.Generator().manual_seed(3)
        return sample_group(
            model, [1, 4], group=4, max_tokens=12, eos_token_id=2, generator=generator
        )

    completions = draw()
    assert completions == draw() and len(completions) == 4
    for completion in completions:
        assert 1 <= len(completion) <= 12
        assert 2 not in completion[:-1]


class AnyIdTokenizer(BaseTokenizer):
    """Decode every sampled ID, as a real tokenizer does; a random policy samples any row."""

    def decode(self, tokens):
        return "".join(chr(32 + token % 90) for token in tokens)


def parent_checkpoint(tmp_path, tokenizer):
    model = tiny_model(tokenizer.vocab_size)
    directory = tmp_path / "parent"
    metadata = {
        "training_phase": "sft",
        "step": 1,
        "config": model.config.settings(),
        "resolved": {"tokenizer": tokenizer.metadata()},
    }
    save(directory, 1, model.state_dict(), {}, metadata)
    return {
        "checkpoint_dir": str(directory),
        "step": 1,
        "model_sha256": rl._sha256(directory / "model_000001.pt"),
        "metadata_sha256": rl._sha256(directory / "metadata_000001.json"),
    }


def test_trainer_runs_steps_and_resumes_exactly(tmp_path, monkeypatch):
    tokenizer = ChatTokenizer(AnyIdTokenizer(tmp_path / "tokenizer.model"))
    monkeypatch.setattr(rl, "get_chat_tokenizer", lambda **config: tokenizer)
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"domain": "Math", "query": f"{index}+1", "ground_truth": "1"} for index in range(4)]
    rows.append({"domain": "Knowledge", "query": "why", "ground_truth": "because"})
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    settings = {
        "prompt_files": [str(prompts)],
        "group_size": 4,
        "activation_checkpointing": False,
        "deterministic": False,
        "prompts_per_step": 2,
        "max_prompt_tokens": 32,
        "max_new_tokens": 8,
        "temperature": 1.0,
        "steps": 2,
        "lr": 1e-2,
        "min_lr": 0.1,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "save_every": 1,
        "seed": 5,
        "parent": parent_checkpoint(tmp_path, tokenizer),
        "output_dir": str(tmp_path / "run"),
    }

    def reward(prompt, text):
        return float(len(text) % 2)

    trainer = rl.RLTrainer({"tokenizer": {}, "rl": settings}, reward=reward)
    assert trainer.skipped == {"unverifiable_domain": 1, "prompt_too_long": 0}
    trainer.run()
    metrics = [
        json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in metrics] == [1, 2]
    assert all(row["completion_tokens"] > 0 for row in metrics)

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    for name in ("model", "optimizer", "metadata"):
        suffix = "json" if name == "metadata" else "pt"
        shutil.copy2(tmp_path / f"run/{name}_000001.{suffix}", resumed)
    shutil.copy2(tmp_path / "run/complete_000001", resumed)
    trainer = rl.RLTrainer(
        {"tokenizer": {}, "rl": {**settings, "output_dir": str(resumed)}}, reward=reward
    )
    assert trainer.start == 1
    trainer.run()
    expected, actual = load_model(tmp_path / "run", 2, "cpu"), load_model(resumed, 2, "cpu")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
