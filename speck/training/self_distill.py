"""Build the stage-6 self-distillation SFT dataset from verified samples of a promoted parent.

For each prompt the parent samples several completions; only completions that end at EOS and pass
their verifier are kept, deduplicated, up to a per-prompt limit. Hash-pinned anchor conversations
are mixed in unchanged. The output is a local SFT source directory (one train and one validation
Parquet file) plus a receipt, so the existing SFT trainer runs the final self-SFT unmodified.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from speck.config import load_experiment
from speck.model.generation import sample_group
from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.chat import get_chat_tokenizer
from speck.training.rl import REWARDS, load_parent, load_prompts

SETTINGS = {
    "prompt_files",
    "samples_per_prompt",
    "keep_per_prompt",
    "max_prompt_tokens",
    "max_new_tokens",
    "temperature",
    "seed",
    "validation_fraction",
    "parent",
    "anchor_files",
    "output_dir",
}


def _validation(messages, fraction):
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def _anchor_rows(anchor_files):
    rows = []
    for anchor in anchor_files:
        if file_sha256(anchor["path"]) != anchor["sha256"]:
            raise ValueError(f"anchor file does not match its digest: {anchor['path']}")
        for row in pq.read_table(anchor["path"]).to_pylist():
            rows.append({"messages": row["messages"], "tools": [], "source": row["source"]})
    return rows


def build(configs, device="cpu", reward=None):
    settings = dict(configs["self_distill"])
    if set(settings) != SETTINGS:
        raise ValueError(
            f"self_distill settings must contain exactly: {', '.join(sorted(SETTINGS))}"
        )
    if not 1 <= settings["keep_per_prompt"] <= settings["samples_per_prompt"]:
        raise ValueError("keep_per_prompt must be between one and samples_per_prompt")
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = get_chat_tokenizer(**configs["tokenizer"])
    model = load_parent(settings["parent"], tokenizer).to(device).eval()
    prompts, skipped = load_prompts(
        settings["prompt_files"], tokenizer, settings["max_prompt_tokens"]
    )
    reward = reward or (
        lambda prompt, text: REWARDS[prompt["domain"]](text, prompt["ground_truth"])
    )
    generator = torch.Generator(device).manual_seed(settings["seed"])
    counts = Counter()
    rows = []
    for prompt in prompts:
        completions = sample_group(
            model,
            prompt["tokens"],
            group=settings["samples_per_prompt"],
            max_tokens=settings["max_new_tokens"],
            eos_token_id=tokenizer.eos_id,
            temperature=settings["temperature"],
            generator=generator,
        )
        accepted = []
        for completion in completions:
            counts["samples"] += 1
            if completion[-1] != tokenizer.eos_id:
                counts["truncated"] += 1
                continue
            text = tokenizer.decode(completion, skip_special_tokens=True)
            if not reward(prompt, text):
                counts["failed_verification"] += 1
            elif text in accepted:
                counts["duplicates"] += 1
            else:
                accepted.append(text)
        counts["prompts"] += 1
        counts["prompts_with_accepted"] += bool(accepted)
        for text in accepted[: settings["keep_per_prompt"]]:
            messages = [
                {"role": "user", "content": prompt["query"]},
                {"role": "assistant", "content": text},
            ]
            rows.append(
                {"messages": messages, "tools": [], "source": f"self_distill:{prompt['domain']}"}
            )
    counts["self_distilled_conversations"] = len(rows)
    anchors = _anchor_rows(settings["anchor_files"])
    counts["anchor_conversations"] = len(anchors)
    splits = {"train": [], "val": []}
    for row in rows + anchors:
        split = "val" if _validation(row["messages"], settings["validation_fraction"]) else "train"
        splits[split].append(row)
    if not splits["train"] or not splits["val"]:
        raise ValueError(f"too few conversations for both splits: {dict(counts)}")
    files = []
    for split, split_rows in splits.items():
        path = output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(split_rows), path)
        files.append({"filename": path.name, "split": split, "sha256": file_sha256(path)})
    dataset = {
        "format": "messages_v1",
        "name": output.name,
        "files": files,
        "expected_samples": len(rows) + len(anchors),
        "validation_samples": len(splits["val"]),
    }
    receipt = {
        "format": "speck_self_distillation_dataset",
        "format_version": 1,
        "settings": settings,
        "skipped_prompts": skipped,
        "counts": dict(sorted(counts.items())),
        "dataset": dataset,
        "training_admitted": False,
    }
    atomic_json(output / "self_distill.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", help="experiment directory with self_distill.json and tokenizer"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    receipt = build(load_experiment(args.experiment, "tokenizer", "self_distill"), args.device)
    print(json.dumps(receipt["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
