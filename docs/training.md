# Training and inference

The [program lifecycle](program.md#training-lifecycle) defines pretraining, capability/context
mid-training and post-training on the fixed model. This guide describes implemented training paths;
working stage plans are not launch configurations.

Set `deterministic: true` in base or SFT settings when qualifying reproducible CUDA restart.
This enables deterministic PyTorch algorithms and a reproducible cuBLAS workspace before training.
The pilot and hardware probe enable it after a local attention-backward diagnostic showed gradient
variation without it. The setting is immutable on resume. Older recipes default to `false` for
compatibility; deterministic execution and its throughput cost still require checks on the target
hardware, distributed geometry, and compiled path.

New base and SFT checkpoints include every rank's Python, NumPy, CPU, and local CUDA RNG state
inside the atomic metadata publication. Resume restores those generators after runtime/loader
initialization; CUDA resumes warm the eager forward/backward kernels before loading saved tensors.
Older checkpoints without RNG metadata remain readable but cannot establish full RNG continuity.
The synthetic probe and the production trainer have separate restart checks. The initial GH200
rehearsal uses `--no-compile`; compiled and multi-GPU continuation require measured qualification.

`python -m scripts.training_replay EXPERIMENT --output /external/new-attempt --seconds 1800`
exercises the production base trainer with four diagnostic steps and a restart after two. It
compares every model/optimizer tensor and exact loader/RNG state. `--phase sft` uses a supplied
finite SFT recipe whose computed step count must match `--steps`. Declare all allocated GPUs with
`--allocated-gpus`; failed runs retain logs and checkpoints. See the [rental runbook](gh200.md)
for the complete transferable workflow.

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
The [pilot](../experiments/pilot/README.md) freezes those settings for 104,857,600 tokens and has a
verified local packed corpus; that engineering run and its backups are complete. Reuse its
configuration as an identified reference, not an instruction to repeat it. A new training run needs
its own frozen recipe, hardware/scheduler checks and enforced cost ceiling; these raw commands
do not enforce cumulative budgets. The historical H100 pilot used its bounded rental supervisor.

```bash
uv run --no-sync python -m scripts.base_train PATH_TO_EXPERIMENT
uv run --no-sync torchrun --standalone --nproc-per-node=4 \
  -m scripts.base_train PATH_TO_EXPERIMENT
```

Use `--device cpu --no-compile` for small CPU experiments. Run name `dummy` disables W&B.
Resume explicitly with `--resume STEP`; use `--branch-from DIRECTORY --branch-step STEP` for a
new branch. Resume checks the original model, data cursor, optimizer, tokenizer, and schedule.
A changed recipe requires an explicitly supported new run, not an edited resume. See `--help` for branch options and
[Slurm](slurm.md) for scheduler interruption/requeue.

The pretraining data study uses separate fresh runs with paired initialization seeds; verify initial
model tensor identities within each pair. It precedes main pretraining and does not use checkpoint
branching. Freeze manifests, schedules and evaluation inputs for each arm; study tokens and cost
remain separate from the production run. The [research design](../experiments/main-data/README.md#runtime-and-launch-requirements)
records proposed run counts, finite-epoch SFT exposure matching and remaining runtime/scoring work.
These design rules are not executable launch manifests.

The [capability mid-training efficiency packet](../experiments/main-data/mid-training-study-packet.json)
adds a design-only comparison of replay/source-only, targeted raw and grounded data from one useful
parent. It measures downstream SFT convergence and early RL adaptation under fixed recipes; executable
trajectories remain conditional on environment and verifier qualification.

The [post-training data-study packet](../experiments/main-data/post-training-study-packet.json) fixes
two paired-seed SFT selection arms and a fixed-policy RL prompt/verifier slot. It keeps structural
format checks, independent outcome verification and environment success as separate gates; no policy
update is implied by the research packet.

## Mid-training readiness

The base loader supports token-endpoint mixture phases declared in one immutable packed manifest.
This can express a preplanned curriculum using the existing next-token objective. Keep source supply,
phase boundaries, replay and aggregate token exposure explicit. It does not make a new dataset
compatible with an ordinary checkpoint branch.

`--branch-kind same` requires the parent's model and data manifest, and inherits optimizer/data
state. `--branch-kind context` allows a changed manifest and context configuration under
`training_phase: context_extension`, resets the data cursor and retains optimizer state. Qualify
actual long-context workloads separately. A dedicated changed-data continuation contract remains
to be implemented/qualified for capability mid-training and any later continuation-data arms; do not
mislabel those as context runs. Any new masked repair objective also requires a validated adapter.

Before executing either part of mid-training, freeze parent identity, objective, data, schedule,
optimizer policy and cost, and qualify resume. Capability continuation consumes a portion of the
base horizon; context extension uses its separately declared exposure. No production RL trainer
exists yet; its data, rollout and verifier requirements are in [the program](program.md#conditional-rl).

## Assistant training and generation

SFT requires its own `sft.json`, prepared assistant-masked data, and an explicit parent checkpoint:

```bash
uv run --no-sync python -m scripts.sft_prepare PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.sft_train PATH_TO_SFT_EXPERIMENT
uv run --no-sync python -m scripts.infer "Explain this result:" \
  --experiment PATH_TO_EXPERIMENT --checkpoint-dir CHECKPOINT_DIRECTORY --max-tokens 128
```

The [assistant rehearsal contract](assistant.md) now defines tool envelopes, reasoning serialization,
loss masks, and a deterministic tool environment. Actual learned tool/reasoning capability remains
unmeasured. Keep base and assistant checkpoints separately identifiable. SFT supports explicit
`activation_checkpointing` and `loss_backend` settings; its defaults preserve historical behavior,
and changing them on resume is rejected.

Chat format v2 preserves `weight: 0` assistant turns as context and supervises only `weight: 1`
(default) turns, including their EOS. Its fingerprint differs from v1, so old prepared masks must
not be reused. Set `chat_format_version: 1` in an SFT tokenizer configuration only to reproduce
a historical unweighted run. Inference restores the version recorded in checkpoint metadata.
The chat tokenizer itself rejects raw tool fields and separate `reasoning_content` instead of
dropping them. Apply the existing `speck_tools_v1` adapter first for supported structured records;
see the assistant contract for loss masks, accepted fields and rejection rules.

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

The offline smoke now also prepares weighted local SFT examples, initializes from its native base
checkpoint, trains two assistant steps, and verifies exact parameter equality after SFT resume.
This exercises the reserved-vocabulary path used by the flagship. It remains a tiny CPU fixture.
