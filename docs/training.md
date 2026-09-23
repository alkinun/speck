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

## Throughput settings

The [performance plan](performance.md) defines timing boundaries, MFU accounting and optimization
priorities. The RTX 3090 proxy selected checkpointing off, compiled execution with
`max-autotune-no-cudagraphs`, deterministic kernels and Liger loss. These are candidates for the
flagship; its optimized rate and memory headroom remain unmeasured. Training and the benchmark
compile with `COMPILE_OPTIONS` from `speck/operations/runtime.py`. Unlike Torch's mode of the same
name, the benchmark's `max-autotune-no-cudagraphs` omits coordinate-descent tuning, which broke
compiled restart parity.

Freeze microbatch, activation checkpointing and determinism on GH200 before production; these
settings are immutable on resume. Microbatch affects loader scheduling even at constant global
batch. Base training compiles both the model and BatchedMuon; the benchmark's compile toggle
therefore measures their combined gain. The production call also requests typed loss diagnostics.

Use the [H100 rental](throughput-rental.md) to measure single-GPU implementation deltas. Then
qualify compiled four-worker DDP with `scripts.training_replay --compile` on the grant hardware.
Persist `TORCHINDUCTOR_CACHE_DIR` and its Triton cache across requeues and include warmup in wave costs.
Runtime setup fixes `TRITON_CACHE_DIR` before eager warmup or compilation: an explicit setting wins;
otherwise it uses `triton/` under `TORCHINDUCTOR_CACHE_DIR`, or under the Speck cache directory when
no Inductor directory is declared. Different cached FLA choices can change numerical results;
preserving the cache is necessary for this comparison but does not by itself qualify recovery. Benchmark
`end-to-end` mode includes packed loading, but excludes validation and saves; a sustained trainer
run is needed to update the full-trainer overhead ratio.

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
`--allocated-gpus`; failed runs retain logs and checkpoints. See the
[GH200 access qualification runbook](compute-qualification.md) for the complete transferable
workflow.

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
defines a design-only four-arm proxy screen: replay, source-only repository data, grounded workflows
and executable trajectories. It also separates selective loss/packing and context-transition studies,
then confirms at most one candidate at 1.2B. Production mid-training is staged at 4K, 16K and 32K for
repository reasoning and long-horizon agentic coding. Executable trajectories require pinned
environments, tool schemas, hidden tests, outcome receipts and recovery labels.

The [post-training data-study packet](../experiments/main-data/post-training-study-packet.json) fixes
two paired-seed SFT selection arms, a fixed-policy RL prompt/verifier slot, and a bounded final
self-SFT pilot. It keeps structural format checks, independent outcome verification, environment
success and self-distillation lineage as separate gates. No policy update or production self-SFT is
implied by the research packet.

## Mid-training readiness

The base loader supports token-endpoint mixture phases declared in one immutable packed manifest.
This can express a preplanned curriculum using the existing next-token objective. Keep source supply,
phase boundaries, replay and aggregate token exposure explicit. It does not make a new dataset
compatible with an ordinary checkpoint branch.

`--branch-kind same` requires the parent's model and data manifest, and inherits optimizer/data
state. `--branch-kind context` allows a changed manifest and context configuration under
`training_phase: context_extension`, resets the data cursor and retains optimizer state. The explicit
`--branch-kind data` path allows a changed manifest under `training_phase: data_continuation`; it
requires the same architecture, sequence length, optimizer semantics, schedule and world-size
contract as the parent, resets only the data cursor, and retains the parent optimizer state. Data
branches require `--branch-schedule inherit`; context branches remain the only path that changes
sequence capacity. Qualify 4K, 16K and 32K workloads separately.

A data source can be row-packed with `packing: {row_tokens, open_rows}`. Preparation places whole
records best-fit into rows of exactly the training sequence length, never truncates (over-long and
invalid records are counted in `rejected_records`), and writes a parallel loss mask hashed with the
shards. `record_format: messages` encodes chat and tool trajectories so that only assistant
content is supervised. Records sharing a row are not isolated from one another. The loader reads a
row-packed source only at its row length, and every optimizer step is normalized by its supervised
tokens. `scripts.smoke` branches a base checkpoint onto masked chat rows and repeats it with exact
parameter parity. Still to qualify: 16K/32K memory and throughput on GH200, and supervised counts
per real trajectory source.

Before executing any mid-training stage, freeze parent identity, objective, data, schedule, optimizer
policy and cost, and qualify resume. Capability, repository and agentic continuation use the combined
600-hour mid-training production reservation; 16K/32K context changes are part of that reservation.
## Reward training

`python -m scripts.rl_train EXPERIMENT` trains stage 5 from a hash-pinned SFT or RL checkpoint named
in `rl.json`. For each prompt it samples a group of completions from the current policy over the
full vocabulary, scores them with the checked-math or sandboxed stdin/stdout verifier in
`speck/evaluation/verifiers.py`, normalizes rewards within the group, and takes one on-policy
step on completion tokens. Groups whose rewards are all equal carry no signal and are counted;
unverifiable domains and over-long prompts are counted and skipped. Sampling is seeded per step,
so resume is exact. The trainer is single-process, with no KL penalty, reference model or
clipping; distributed rollout, long-context RL and measured benefit remain to qualify. The smoke
run checks the plumbing only: its tiny model earns no reward. Data and verifier requirements are in
[the program](program.md#conditional-rl).

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

After structural adaptation, run the separate outcome gate in
`speck.training.sft_verify.verify_sft_outcomes`. Rows must declare one of `exact_text`, `code`,
or `tool` under `verification`; rows without a declaration remain `unverified`. Exact answers are
compared literally after trimming, code rows execute the frozen tests through the fail-closed
sandbox, and tool rows replay the declared calls against the deterministic tool implementation
before checking the recorded results and final answer. Validate the resulting receipt with
`verify_sft_outcome_receipt` and retain its input fingerprint. This receipt is a selection and
analysis artifact, not a replacement for held-out evaluation or structural SFT validation.
Use `select_verified_sft_rows(rows, receipt)` for downstream selection; it fails if the row order
or contents differ from the receipt's input fingerprint.

The same gate is available for local shards:

```bash
uv run --no-sync python -m scripts.sft_verify /external/cache/sft-train.parquet \
  --output /external/reports/sft-outcomes.json
```

The [post-training audit protocol](../experiments/main-data/post-training-audit-protocol.json) extends
this structural audit with fixed outcome strata, tool-trajectory checks, deterministic environment
controls, a reasoning-mode measurement panel and a bounded fixed-policy RL panel. The RL panel records
correctness-first length-efficiency curves and truncation/shortcut controls before any policy update.
It must close before a post-training study arm is selected.

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
