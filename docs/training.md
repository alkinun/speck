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

Use `experiments/Speck2-140M` for the pinned 20B-token curriculum. It retains the architecture,
tokenizer, optimizer, batch geometry, and peak learning rate while scaling warmup and checkpoint
cadences with the four-times-longer horizon:

```bash
uv run --extra gpu python -m scripts.base_train experiments/Speck2-140M
```

Weights & Biases logging is enabled unless `train.run` is `dummy`. Checkpoints remain local under
`~/.cache/speck/checkpoints/<train.run>` unless `train.output_dir` overrides the path. Training does
not upload checkpoints to Hugging Face.

Launch distributed data-parallel training with `torchrun`:

```bash
uv run --extra gpu torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train -- experiments/Speck1-140M
```

`device_batch_size` is a per-device ceiling. The configured ceiling of 16 resolves to device batches
16, 16, 8, and 4 for world sizes 1, 2, 4, and 8, keeping the optimizer batch fixed at 65,536 tokens
while minimizing gradient accumulation. Use `--device-batch-size` to set a lower ceiling for a
resume or constrained GPU, or to test 32 on a larger-memory GPU. Because 5B is not batch-aligned,
base training performs 76,294 optimizer steps and consumes 5,000,003,584 tokens.
Mixture phases are selected from each global microbatch's starting token position, so a microbatch
that begins before a phase boundary stays in that phase even when it crosses the boundary.

Compiled base training also batches same-shaped Muon matrices for the Newton-Schulz update and
compiles that optimizer step independently. The fixed-width causal convolution branches use a
compiler-fused depthwise stencil instead of launching a generic grouped convolution. CUDA AdamW
parameter groups use the fused kernel.
Distributed runs wrap the model in DDP before compilation so gradient communication can overlap the
partitioned backward graph, and gradients reuse DDP bucket storage. CUDA finite checks and
performance timing synchronize only at reporting or artifact boundaries; `--no-compile` disables
both model and optimizer compilation.

Speck2 performs 305,176 optimizer steps and consumes 20,000,014,336 tokens. It evaluates every
1,952 steps, saves approximately every 1B tokens at 15,260-step intervals, and warms up for 2,048
steps. Its cosine schedule ends at 5% of the `0.0015` peak learning rate, or `0.000075`.

Base recipes use the explicit `cosine` learning-rate schedule. Schedule type is part of the
checkpoint contract; isolated constant-LR branches are described below.

Use `--save-every` or `--eval-every` to override artifact cadence without changing the optimization
contract. Intervals are optimizer steps; zero disables that periodic action, while final and
token-milestone artifacts still run. For example, `--save-every 1526` is approximately every 100M
tokens at the 65,536-token optimizer batch.

Validation reports the equal-batch aggregate and each source separately. Checkpoints retain the
latest per-source losses alongside the aggregate so later data decisions do not depend on W&B.

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

Start an isolated branch from a complete checkpoint with a separate experiment whose `run`, output
directory, and local `train_tokens` describe the branch:

```bash
uv run --extra gpu python -m scripts.base_train experiments/branch \
  --branch-from ~/.cache/speck/checkpoints/Speck2-140M --branch-step <step>
```

Branches currently preserve the parent model, packed-data manifest, optimizer, batch geometry, data
cursor, and learning-rate schedule. They create a new W&B run and record model, optimizer, and
metadata hashes for the parent. Changed data and implicit schedule changes are rejected rather than
silently weakening the comparison.

Pass `--branch-schedule new` to start the schedule in the branch experiment at local step zero.
Only `lr`, `warmup_steps`, `min_lr`, and `lr_schedule` may then differ; optimizer state and every
non-schedule training setting remain inherited and validated. A constant-LR tail uses
`lr_schedule: "constant"`, zero warmup, and `min_lr: 1.0` with the desired absolute `lr`.

Prepare matched inherited-schedule and constant-LR experiments without manually calculating the
parent LR:

```bash
uv run --extra cpu python -m scripts.tail_pair_prepare \
  experiments/Speck2-140M experiments/Speck2-140M-TailPair \
  --checkpoint-dir ~/.cache/speck/checkpoints/Speck2-140M \
  --step <step> --train-tokens <tokens> --save-every 1526 --eval-every 1526
```

The command verifies architecture, packed data, and fixed training settings, preserves the parent's
actual device batch, and atomically writes `control/`, `constant/`, and a hashed `pair.json` lineage
record. Launch both arms at the reported world size against the same parent checkpoint; add
`--branch-schedule new` only for `constant/`.

The planner rejects tails that do not fit in the control's remaining inherited schedule. Evaluate
the two completed final checkpoints through the existing pinned benchmark paths before adding any
checkpoint-averaging experiment.

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

Train the same one-epoch SpeckChat2 recipe from the pinned Speck1.5 base release:

```bash
uv run --extra cpu python -m scripts.sft_prepare experiments/Speck1.5-140M-Instruct
uv run --extra gpu python -m scripts.sft_train experiments/Speck1.5-140M-Instruct
```

This run writes checkpoints under `~/.cache/speck/checkpoints/Speck1.5-140M-Instruct`.

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
