"""minimal kv-cached text generation from a training checkpoint."""

import argparse
import os

import torch

from speck.checkpoint import latest, load
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import Config, Llama
from speck.tokenizer import get_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("prompt")
parser.add_argument("--experiment", default="experiments/speck-50m")
parser.add_argument("--checkpoint-dir", default=None)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

configs = load_experiment(args.experiment, "tokenizer", "train")
args.checkpoint_dir = args.checkpoint_dir or configs["train"].get("output_dir") or os.path.join(
    base_dir(), "checkpoints", configs["train"]["run"]
)
step = args.step if args.step is not None else latest(args.checkpoint_dir)
if step is None:
    raise FileNotFoundError(f"no checkpoint found in {args.checkpoint_dir}")
device = torch.device(args.device)
model_state, _, metadata = load(args.checkpoint_dir, step, device)
model = Llama(Config(**metadata["config"])).to(device)
model.load_state_dict(model_state)
model.eval()
tokenizer = get_tokenizer(**configs["tokenizer"])
tokens = tokenizer.encode(args.prompt, bos=True)
if len(tokens) + args.max_tokens > model.config.max_position_embeddings:
    raise ValueError("prompt and generated tokens exceed the model context")

with torch.inference_mode():
    cache = model.cache(length=len(tokens) + args.max_tokens)
    logits = model(torch.tensor([tokens], device=device), cache=cache)[:, -1]
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
        logits = model(token[:, None], cache=cache)[:, -1]

print(tokenizer.decode(generated))
