"""Generate text from a Speck training checkpoint with state caching."""

import argparse
import os

import torch

from speck.architecture import ArchitectureConfig
from speck.chat import ChatTokenizer
from speck.checkpoint import latest, load_metadata, load_model
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import SpeckForCausalLM
from speck.tokenizer import get_tokenizer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to continue")
    parser.add_argument(
        "--experiment",
        default="experiments/Speck1-140M",
        help="experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="checkpoint directory; defaults to the experiment output directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="checkpoint step; defaults to the latest available step",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="maximum number of tokens to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature; use 0 for greedy decoding (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="number of highest-probability tokens considered during sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="inference device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--system", default=None, help="optional system prompt for an SFT checkpoint"
    )
    return parser.parse_args(argv)


def load_checkpoint_model(checkpoint_dir, step, device):
    """Load only inference state, releasing the CPU state before moving the model."""

    metadata = load_metadata(checkpoint_dir, step)
    model_state = load_model(checkpoint_dir, step, "cpu")
    model = SpeckForCausalLM(ArchitectureConfig.from_dict(metadata["config"]))
    model.load_state_dict(model_state)
    del model_state
    return model.to(device).eval(), metadata


def main(argv=None):
    args = arguments(argv)
    configs = load_experiment(args.experiment, "tokenizer", "train")
    checkpoint_dir = (
        args.checkpoint_dir
        or configs["train"].get("output_dir")
        or os.path.join(base_dir(), "checkpoints", configs["train"]["run"])
    )
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(checkpoint_dir, step, device)
    tokenizer = get_tokenizer(**configs["tokenizer"])
    if metadata.get("training_phase") == "sft":
        tokenizer = ChatTokenizer(tokenizer)
        if metadata.get("resolved", {}).get("tokenizer") != tokenizer.metadata():
            raise ValueError("SFT checkpoint and tokenizer do not match")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        tokens, _ = tokenizer.encode_messages(messages, add_generation_prompt=True)
    elif args.system:
        raise ValueError("system prompts require an SFT checkpoint")
    else:
        tokens = tokenizer.encode(args.prompt, bos=True)
    if len(tokens) + args.max_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt and generated tokens exceed the model context")

    with torch.inference_mode():
        state = model.state(length=len(tokens) + args.max_tokens)
        logits = model(
            torch.tensor([tokens], device=device),
            state=state,
            last_token_only=True,
        )[:, -1]
        generated = []
        for _ in range(args.max_tokens):
            if args.temperature == 0:
                token = logits.argmax(dim=-1)
            else:
                values, indices = torch.topk(logits, min(args.top_k, logits.size(-1)))
                probabilities = torch.softmax(values / args.temperature, dim=-1)
                token = indices.gather(-1, torch.multinomial(probabilities, 1)).squeeze(-1)
            token_id = token.item()
            if token_id == tokenizer.eos_id:
                break
            generated.append(token_id)
            logits = model(token[:, None], state=state, last_token_only=True)[:, -1]

    if isinstance(tokenizer, ChatTokenizer):
        print(tokenizer.decode(generated, skip_special_tokens=True))
    else:
        print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
