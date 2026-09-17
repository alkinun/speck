# Training and inference

`make setup` installs the CPU environment. CUDA uses `uv sync --extra gpu --extra linear`;
the allocation's arm64/CUDA environment must be checked on site.

## Offline baseline

```bash
make smoke
# Retain configs, data, checkpoints, and report in a new directory:
uv run --no-sync python -m scripts.smoke --output-dir /tmp/speck-smoke-run
```

This exercises local tokenization, packing, training, exact interrupted/resumed parameter equality,
and held-out loss. It requires no corpus downloads or W&B account. It is a software check.

## First GPU check

Follow [qualification](../experiments/qualification/README.md). It uses the starting 1.2B model and
synthetic 4K inputs, checks optimization and fresh-process restart, and records resource use.
Actual corpus training has separate data and sustained-throughput requirements.

## Corpus training

A complete experiment supplies `model.json`, `tokenizer.json`, `data.json`, and `train.json`.
The pilot's data, token endpoint, learning rate, and batch are still to be frozen; the qualification
directory is not a pretraining recipe.

```bash
uv run --no-sync python -m scripts.base_train PATH_TO_EXPERIMENT
uv run --no-sync torchrun --standalone --nproc-per-node=4 \
  -m scripts.base_train PATH_TO_EXPERIMENT
```

Use `--device cpu --no-compile` for small CPU experiments. Run name `dummy` disables W&B.
Resume explicitly with `--resume STEP`; use `--branch-from DIRECTORY --branch-step STEP` for a
new branch. Resume checks the original model, data cursor, optimizer, tokenizer, and schedule.
A changed recipe is a new run, not an edited resume. See `--help` for branch options and
[Slurm](slurm.md) for scheduler interruption/requeue.

## Assistant training and generation

SFT requires its own `sft.json`, prepared assistant-masked data, and an explicit parent checkpoint:

```bash
uv run --no-sync python -m scripts.sft_prepare PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.sft_train PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.infer "Explain this result:" \
  --experiment PATH_TO_EXPERIMENT --checkpoint-dir CHECKPOINT_DIRECTORY --max-tokens 128
```

The current chat implementation is not yet a qualified tool-calling or reasoning protocol. Reconcile
the separate post-training work, parser/template, loss masks, and output limits before making those
claims. Keep base and assistant checkpoints separately identifiable.

Chat format v2 preserves `weight: 0` assistant turns as context and supervises only `weight: 1`
(default) turns, including their EOS. Its fingerprint differs from v1, so old prepared masks must
not be reused. Set `chat_format_version: 1` in an SFT tokenizer configuration only to reproduce
a historical unweighted run. Inference restores the version recorded in checkpoint metadata.
The current format rejects tool fields and separate `reasoning_content` instead of dropping them.

Audit local post-training stock before selecting a recipe:

```bash
uv run --no-sync python -m scripts.sft_audit /external/cache/generator-train-*.arrow \
  --tokenizer /external/tokenizer/tokenizer.model --lengths 4096 8192 16384 \
  --output /external/reports/sft-audit.json
```

The audit counts every row and tokenizes a deterministic sample per source/subset. It records file
hashes, tokenizer identity, rejection reasons, and complete-conversation fit without truncation.
Pass conversation shards only: Hugging Face `cache-*.arrow` files can contain shuffle indices.
A sample's fit percentage is an estimate, not a prepared training count or quality score.

SFT can initialize directly from a completed native base checkpoint. Bind its model and metadata
hashes with `speck.export.pretrained.native_pretrained_source(directory, step)` and use the returned
object as `sft.json`'s `pretrained` setting. This reads local weights without exporting or uploading
an unfinished model. Changed parent bytes fail before loading; SFT resume retains the original
parent identity and does not require the base checkpoint to remain locally available.

Local SFT preparation also accepts `dataset.format: "messages_v1"` with the same pinned train/val
Parquet file declarations as `prompt_completion_v1`. It decodes `List(Json())` messages, retains
assistant weights, and rejects unsupported tools and overlength conversations. No local record is
truncated. For a Hub dataset, set `long_sequences: "reject"` explicitly for the same length policy;
the absent-field default remains the historical truncation behavior. Rejected counts stay in the
manifest, and an empty accepted split is an error.
