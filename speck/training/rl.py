"""Train an assistant with group-relative policy gradients on verifiable rewards (stage 5).

Each step samples a group of completions per prompt from the current policy, scores them with a
checked verifier, normalizes rewards within the group, and takes one on-policy gradient step on
the completion tokens. Groups whose completions all score the same carry no signal and are
reported, not hidden. This is a minimal single-process trainer; it performs no KL penalty,
reference model or multi-epoch clipping.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from speck.config import load_experiment
from speck.evaluation.verifiers import code_reward, math_reward
from speck.model import SpeckForCausalLM
from speck.model.architecture import ArchitectureConfig
from speck.model.generation import sample_group
from speck.operations.runtime import configure_determinism, print0
from speck.tokenization.chat import get_chat_tokenizer
from speck.training.checkpoint import latest, load, load_metadata, load_model, save
from speck.training.step import assert_finite, lr_scale, set_optimizer_lr

SETTINGS = {
    "prompt_files",
    "group_size",
    "prompts_per_step",
    "max_prompt_tokens",
    "max_new_tokens",
    "temperature",
    "steps",
    "lr",
    "min_lr",
    "warmup_steps",
    "weight_decay",
    "grad_clip",
    "optimizer",
    "activation_checkpointing",
    "deterministic",
    "save_every",
    "seed",
    "parent",
    "output_dir",
}
REWARDS = {"Math": math_reward, "Code": code_reward}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_settings(settings):
    if set(settings) != SETTINGS:
        raise ValueError(f"rl settings must contain exactly: {', '.join(sorted(SETTINGS))}")
    for key in ("group_size", "prompts_per_step", "max_prompt_tokens", "max_new_tokens", "steps"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["group_size"] < 2:
        raise ValueError("group-relative advantages need at least two completions per prompt")
    if not isinstance(settings["temperature"], (int, float)) or settings["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if type(settings["activation_checkpointing"]) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if type(settings["deterministic"]) is not bool:
        raise ValueError("deterministic must be boolean")
    if set(settings["parent"]) != {"checkpoint_dir", "step", "model_sha256", "metadata_sha256"}:
        raise ValueError("parent must name a native checkpoint and its sha256 digests")
    return settings


def load_prompts(files, tokenizer, max_prompt_tokens):
    """Read verifiable prompts; unverifiable domains and over-long prompts are counted, not kept."""

    prompts, skipped = [], {"unverifiable_domain": 0, "prompt_too_long": 0}
    for path in files:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row["domain"] not in REWARDS:
                skipped["unverifiable_domain"] += 1
                continue
            messages = [{"role": "user", "content": row["query"]}]
            tokens, _ = tokenizer.encode_messages(messages, add_generation_prompt=True)
            if len(tokens) > max_prompt_tokens:
                skipped["prompt_too_long"] += 1
                continue
            prompts.append(dict(row, tokens=tokens))
    if not prompts:
        raise ValueError("no verifiable prompts")
    return prompts, skipped


def group_advantages(rewards):
    """Center rewards within a group and scale by their spread; a constant group has none."""

    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    spread = rewards.std(unbiased=False)
    if spread == 0:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / spread


def policy_loss(model, prompt, completions, advantages):
    """Return the advantage-weighted completion-token NLL sum and its token count."""

    device = next(model.parameters()).device
    width = len(prompt) + max(len(completion) for completion in completions)
    tokens = torch.zeros((len(completions), width), dtype=torch.long, device=device)
    targets = torch.full_like(tokens, -100)
    for row, completion in enumerate(completions):
        sequence = torch.tensor(prompt + completion, device=device)
        tokens[row, : len(sequence)] = sequence
        targets[row, len(prompt) - 1 : len(sequence) - 1] = sequence[len(prompt) :]
    logits = model(tokens[:, :-1]).float()
    nll = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), targets[:, :-1], ignore_index=-100, reduction="none"
    )
    weights = advantages.to(device)[:, None]
    return (nll * weights).sum(), int((targets[:, :-1] != -100).sum())


def load_parent(parent, tokenizer):
    directory = Path(parent["checkpoint_dir"])
    step = parent["step"]
    if (
        _sha256(directory / f"model_{step:06d}.pt") != parent["model_sha256"]
        or _sha256(directory / f"metadata_{step:06d}.json") != parent["metadata_sha256"]
    ):
        raise ValueError("parent checkpoint does not match its recorded digests")
    metadata = load_metadata(directory, step)
    if metadata["resolved"]["tokenizer"] != tokenizer.metadata():
        raise ValueError("parent checkpoint was trained with a different chat tokenizer")
    model = SpeckForCausalLM(ArchitectureConfig.from_dict(metadata["config"]))
    model.load_state_dict(load_model(directory, step, "cpu"))
    return model


class RLTrainer:
    def __init__(self, configs, device="cpu", reward=None):
        self.settings = validate_settings(dict(configs["rl"]))
        configure_determinism(self.settings["deterministic"])
        self.device = torch.device(device)
        self.tokenizer = get_chat_tokenizer(**configs["tokenizer"])
        self.prompts, self.skipped = load_prompts(
            self.settings["prompt_files"], self.tokenizer, self.settings["max_prompt_tokens"]
        )
        random.Random(self.settings["seed"]).shuffle(self.prompts)
        self.reward = reward or (
            lambda prompt, text: REWARDS[prompt["domain"]](text, prompt["ground_truth"])
        )
        self.output = Path(self.settings["output_dir"])
        self.model = load_parent(self.settings["parent"], self.tokenizer).to(self.device)
        self.model.set_gradient_checkpointing(self.settings["activation_checkpointing"])
        self.optimizer = self.model.optimizer(
            self.settings["lr"], self.settings["weight_decay"], self.settings["optimizer"]
        )
        self.start = 0
        step = latest(self.output) if self.output.exists() else None
        if step is not None:
            model, optimizer, metadata = load(self.output, step, "cpu")
            # The run location is operational; every scientific setting must match.
            recorded = {**metadata["settings"], "output_dir": self.settings["output_dir"]}
            if recorded != self.settings:
                raise ValueError("cannot resume RL with different settings")
            self.model.load_state_dict(model)
            self.optimizer.load_state_dict(optimizer)
            self.start = step

    def _batch(self, step):
        count = self.settings["prompts_per_step"]
        return [self.prompts[(step * count + index) % len(self.prompts)] for index in range(count)]

    def step(self, step):
        settings = self.settings
        # Sampling depends only on (seed, step), so a resumed run draws the same completions.
        generator = torch.Generator(self.device).manual_seed(settings["seed"] * 1_000_003 + step)
        self.model.eval()
        groups = []
        for prompt in self._batch(step):
            completions = sample_group(
                self.model,
                prompt["tokens"],
                group=settings["group_size"],
                max_tokens=settings["max_new_tokens"],
                eos_token_id=self.tokenizer.eos_id,
                temperature=settings["temperature"],
                generator=generator,
            )
            texts = [
                self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in completions
            ]
            rewards = [self.reward(prompt, text) for text in texts]
            groups.append((prompt, completions, rewards))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        tokens = sum(len(completion) for _, completions, _ in groups for completion in completions)
        loss_sum = torch.zeros((), device=self.device)
        for prompt, completions, rewards in groups:
            advantages = group_advantages(rewards)
            # One completion per backward bounds activation memory by the longest sequence,
            # not the group; the summed gradient is the same.
            for completion, advantage in zip(completions, advantages):
                if not advantage:
                    continue
                loss, _ = policy_loss(self.model, prompt["tokens"], [completion], advantage[None])
                (loss / tokens).backward()
                loss_sum += loss.detach()
        assert_finite(loss_sum, "non-finite RL loss")
        scale = lr_scale(
            step, settings["steps"], settings["warmup_steps"], settings["min_lr"], "cosine"
        )
        set_optimizer_lr(self.optimizer, settings["lr"] * scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings["grad_clip"])
        assert_finite(grad_norm, "non-finite RL gradients")
        self.optimizer.step()
        rewards = [reward for _, _, group in groups for reward in group]
        return {
            "step": step + 1,
            "reward_mean": sum(rewards) / len(rewards),
            "solved_prompts": sum(any(group) for _, _, group in groups),
            "zero_signal_groups": sum(len(set(group)) == 1 for _, _, group in groups),
            "completion_tokens": tokens,
            "truncated_completions": sum(
                completion[-1] != self.tokenizer.eos_id
                for _, completions, _ in groups
                for completion in completions
            ),
            "loss": float(loss_sum) / tokens,
            "grad_norm": float(grad_norm),
        }

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "metrics.jsonl").open("a") as log:
            for step in range(self.start, self.settings["steps"]):
                started = time.perf_counter()
                metrics = {**self.step(step), "seconds": time.perf_counter() - started}
                log.write(json.dumps(metrics, sort_keys=True) + "\n")
                log.flush()
                print0(json.dumps(metrics, sort_keys=True))
                completed = step + 1
                if (
                    completed % self.settings["save_every"] == 0
                    or completed == self.settings["steps"]
                ):
                    self._checkpoint(completed)

    def _checkpoint(self, step):
        metadata = {
            "format_version": 1,
            "training_phase": "rl",
            "step": step,
            "config": self.model.config.settings(),
            "settings": self.settings,
            "resolved": {"tokenizer": self.tokenizer.metadata(), "skipped_prompts": self.skipped},
        }
        save(self.output, step, self.model.state_dict(), self.optimizer.state_dict(), metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment directory with rl.json, model and tokenizer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configs = load_experiment(args.experiment, "tokenizer", "rl")
    RLTrainer(configs, args.device).run()


if __name__ == "__main__":
    main()
