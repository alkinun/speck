"""Write the model configuration of every ladder rung from one shape rule.

PYTHONPATH=. python experiments/ladder/shapes.py [--check]

A rung is a width w and G groups of three KDA layers followed by one NoPE GQA layer. Head
dimension is 128; KDA has w/256 key and w/128 value heads, GQA max(1, w/512) key-value heads,
and SwiGLU is 2.5w wide. Embeddings are tied and the tokenizer is the frozen Mistral 32K. The
rule reproduces the 1.2B reference exactly, so every rung differs from it only in scale.
`--check` verifies the written configurations instead of rewriting them.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speck.model import build_model  # noqa: E402

VOCAB = 32003
RUNGS = {"50m": (512, 3), "130m": (768, 4), "410m": (1280, 5)}
REFERENCE = (2048, 6)


def shape(width, groups):
    kda = {
        "conv_kernel_size": 4,
        "key_head_dim": 128,
        "kind": "kimi_delta_attention",
        "num_key_heads": width // 256,
        "num_value_heads": width // 128,
        "output_gate_activation": "sigmoid",
        "value_head_dim": 128,
    }
    gqa = {
        "head_dim": 128,
        "kind": "attention",
        "num_key_value_heads": max(1, width // 512),
        "rope_dim": 0,
    }
    swiglu = {"intermediate_size": width * 5 // 2, "kind": "swiglu"}
    blocks = []
    for _ in range(groups):
        for mixer, repeat in ((kda, 3), (gqa, 1)):
            stages = [{"branches": [mixer]}, {"branches": [swiglu]}]
            blocks.append({"block": {"hidden_size": width, "stages": stages}, "repeat": repeat})
    config = {
        "blocks": blocks,
        "embedding_size": width,
        "initializer_range": 0.02,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-05,
        "rope_scaling_factor": 1.0,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "vocab_size": VOCAB,
    }
    with torch.device("meta"):
        parameters = build_model(config, VOCAB).parameter_count()
    return config | {
        "expected_active_parameters": parameters,
        "expected_parameters": parameters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reference = json.loads((ROOT / "experiments/qualification/model.json").read_text())
    if shape(*REFERENCE) != reference:
        raise ValueError("the shape rule no longer reproduces the 1.2B reference")
    for rung, dimensions in RUNGS.items():
        path = ROOT / "experiments/ladder" / rung / "model.json"
        config = shape(*dimensions)
        if args.check:
            if json.loads(path.read_text()) != config:
                raise ValueError(f"{path} differs from the shape rule")
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(rung, f"{config['expected_parameters']:,}")


if __name__ == "__main__":
    main()
