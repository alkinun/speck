# Training and Inference

Speck supports base pretraining, full-model supervised fine-tuning, explicit checkpoint resume, and
local checkpoint generation. Run commands from the repository root.

## Base Training

Prepare the tokenizer and packed corpus first. See [Data preparation](data.md).

Authenticate Weights & Biases and start a single-GPU run:

```bash
wandb login
uv run --extra gpu python -m scripts.base_train experiments/Speck1-140M
```

Use `experiments/Speck1.5-140M` to train the same architecture and optimization recipe against the
isolated Speck1.5 corpus.

Weights & Biases logging is enabled unless `train.run` is `dummy`. Checkpoints remain local under
`~/.cache/speck/checkpoints/<train.run>` unless `train.output_dir` overrides the path. Training does
not upload checkpoints to Hugging Face.

Launch distributed data-parallel training with `torchrun`:

```bash
uv run --extra gpu torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- experiments/Speck1-140M
```

The configured 65,536-token optimizer batch is divisible by
`device_batch_size * sequence_length * world_size` for world sizes 1, 2, 4, and 8. Because 5B is
not batch-aligned, base training performs 76,294 optimizer steps and consumes 5,000,003,584 tokens.
Mixture phases are selected from each global microbatch's starting token position, so a microbatch
that begins before a phase boundary stays in that phase even when it crosses the boundary.

Despite its historical name, `train.json`'s `min_lr` is a multiplier of the peak `lr`, not an
absolute learning rate. A value of `0.1` ends the schedule at 10% of the peak rate.

## Resume Contracts

Existing checkpoints are never resumed implicitly. A run fails rather than overwriting them unless
an exact completed checkpoint step is supplied:

```bash
uv run --extra gpu python -m scripts.base_train experiments/Speck1-140M \
  --resume <checkpoint-step>
```

Resume validates the architecture, packed-data manifest, optimizer settings, batch geometry,
training horizon, world size, and next loader offset. It restores the optimizer, data position,
elapsed time, and W&B run identity.

## Supervised Fine-Tuning

SFT configurations live in experiment directories containing `model.json`, `tokenizer.json`, and
`sft.json`. Prepared conversations use the Speck chat template and an assistant-only loss mask.

### SpeckChat1

Prepare the pinned `specklabs/SpeckChat1` dataset:

```bash
uv run --extra cpu python -m scripts.sft_prepare experiments/Speck1-140M-Instruct
```

The configuration uses `<|system|>`, `<|user|>`, and `<|assistant|>` as token IDs 32000-32002,
preserves pretrained BOS/EOS tokens, and holds out 1,000 conversations for validation.
Conversations are isolated in 256-, 512-, 1,024-, or 2,048-token buckets. Per-device batch sizes of
32, 16, 8, and 4 give every microbatch the same 8,192-token compute budget.

Train one epoch from the pinned `specklabs/Speck1-140M` release:

```bash
uv run --extra gpu python -m scripts.sft_train experiments/Speck1-140M-Instruct
```

SFT checkpoints and a Hugging Face-compatible tokenizer are written under
`~/.cache/speck/checkpoints/Speck1-140M-Instruct`. Use `torchrun` for multiple GPUs and pass
`--resume <checkpoint-step>` to resume an exact SFT checkpoint.

### SpeckChat2

The maintainer dataset builder reconstructs and optionally publishes the 500,000-row
`specklabs/SpeckChat2` training split:

```bash
uv run --extra cpu --group dataset-build python -m scripts.speckchat2_prepare \
  --output-dir <path> \
  --no-push
```

The mixture contains 200K LMSYS DeepSeek conversations, 130K Magpie Llama 3.1 multi-turn
conversations, 85K Hermes, 65K UltraChat, 10K Magpie Reasoning, 8K No Robots, and 2K Everyday
Conversations. It uses source training splits and intentionally publishes no validation or test
split. Source revisions, quality filters, exact prompt deduplication, and tokenizer length checks
are fixed in the builder.

Prepare and train the canonical one-epoch SpeckChat2 experiment:

```bash
uv run --extra cpu python -m scripts.sft_prepare experiments/Speck1.1-140M-Instruct
uv run --extra gpu python -m scripts.sft_train experiments/Speck1.1-140M-Instruct
```

Prepared data is written under `~/.cache/speck/data/SpeckChat2-v3`; checkpoints use
`~/.cache/speck/checkpoints/Speck1.1-140M-Instruct`.

The retained two-epoch variant consumes the same prepared data:

```bash
uv run --extra gpu python -m scripts.sft_train experiments/Speck1.1-140M-Instruct-2ep
```

Its checkpoints use `~/.cache/speck/checkpoints/Speck1.1-140M-Instruct-2ep`.

## Checkpoint Inference

Generate from the latest base checkpoint, or select an exact step with `--step`:

```bash
uv run --extra gpu python -m scripts.infer "The meaning of life is" \
  --experiment experiments/Speck1-140M
```

Select an instruction checkpoint directory to apply its saved chat metadata. A plain prompt is
rendered as a user message:

```bash
uv run --extra gpu python -m scripts.infer "Explain why the sky is blue." \
  --checkpoint-dir ~/.cache/speck/checkpoints/Speck1-140M-Instruct
```

Useful controls include `--max-tokens`, `--temperature`, `--top-k`, `--device`, `--step`,
`--checkpoint-dir`, and `--system`.
