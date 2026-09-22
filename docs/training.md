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

The measurement definitions, current bottleneck evidence, H100 protocol, and optimization order
are maintained in the [performance plan](performance.md). This section records implementation
defaults and their qualification boundaries.

The frozen pilot recipe is not an efficient configuration. A bounded RTX 3090 pass measured
**2.084x** against it, taking model FLOPs utilization from 25.0% to 52.2%. The receipts are in
[`experiments/qualification/throughput-3090/`](../experiments/qualification/throughput-3090/sweep.json).
Three settings account for nearly all of it, and all three are immutable on resume, so they must be
frozen before production starts.

**That 2.084x is a 318M proxy number and is not the flagship's speedup.** The proxy is a separate
24-block, width-1024 model in [`experiments/throughput-proxy`](../experiments/throughput-proxy); it
is geometry-matched to the reference layout, not parameter-matched to the 1.2B. Its baseline sits at
25.0% utilization. The only flagship measurement in the whole sweep is the eager, checkpointed
baseline, and that one is already at **34.6%**. If the flagship reaches the proxy's 52.2% ceiling
its speedup is **1.51x**, because it started from the more efficient of the two baselines. No
flagship configuration with checkpointing off was ever measured at all: two attempts reached 23.03
and 22.91 GiB before exhausting the 24 GiB card.

Expect the checkpointing-off component (1.248x) to transfer, since it removes recompute FLOPs
arithmetically, and expect the compile component to shrink, since it is overhead and fusion and the
flagship's GEMMs are larger. Quote no flagship speedup until it is measured on the 1.2B reference.
Two packets do that, both benchmarking `experiments/pilot`:
[`throughput-h100.json`](../experiments/qualification/throughput-h100.json) on rented time before
grant access, at zero grant cost, and
[`throughput-gh200.json`](../experiments/qualification/throughput-gh200.json) on the grant hardware,
which is where `device_batch_size`, activation checkpointing and determinism are actually frozen.

| Setting | Selected | Why |
| --- | --- | --- |
| `activation_checkpointing` | `false` | Recomputation costs about a quarter of executed FLOPs and buys memory that a 96 GiB card does not need. It is also what blocks compilation. |
| compile | on, `max-autotune-no-cudagraphs` | With checkpointing on, `torch.compile` returns 1.008x, because FLA's `chunk_kda` carries `torch.compiler.disable` and Dynamo cannot tolerate a graph break inside a checkpointed region. With it off the same compile returns 1.612x. |
| `deterministic` | `true`, retained | Determinism costs 1.086x, but raising the microbatch from one to two returns 1.104x. Reproducibility is affordable and the replay receipts depend on it. |
| `loss_backend` | `liger` | The `torch` backend materializes the full vocabulary logits twice, once in float32. |
| `device_batch_size` | choose on GH200 | Larger is better until memory binds. Changing it changes `global_stride` and therefore the data schedule, so it is not resume compatible. |

CUDA graphs were measured and rejected: they cannot capture across the eager KDA islands and ran
0.7% slower. The input pipeline was measured and cleared: end-to-end mode is 0.6% faster than
synthetic compute, inside noise. Six graph breaks remain and are not cheaply removable, because
bypassing the disable decorator still breaks on a lock context manager inside the library.

Two consequences of selecting a compiled recipe are not yet qualified, and both are distributed:

- **Compile under DistributedDataParallel has never been exercised.** `base.py` compiles the
  DDP-wrapped module, and DDPOptimizer splits the graph at bucket boundaries, which is exactly
  where the six remaining breaks could interact badly. The throughput sweep is single-GPU and the
  four-worker replay used to hardcode eager. `scripts.training_replay --compile` now exists for
  this; run it at four workers before any production wave.
- **max-autotune warmup is not free on requeue.** The Slurm renderer exports a per-wave
  `TORCHINDUCTOR_CACHE_DIR` so a preempted job reuses its compilation. Benchmark tokens/s also
  excludes that warmup, so amortize it per wave when costing, not per step.

Benchmark rates are not trainer rates. `scripts.benchmark --mode compute` excludes startup, inline
validation and checkpoint saves; the plan's anchor includes them. The pilot measured that gap at
0.9459, recorded as `compute.throughput_reanchoring_rule.overhead_derate`. Apply it before turning
any measured rate into horizon hours, and remeasure it on GH200 at production save cadence.

Confirm all of this on the real hardware with
[`experiments/qualification/throughput-gh200.json`](../experiments/qualification/throughput-gh200.json)
before freezing a production recipe. Ampere rankings do not transfer directly: the library's
autotune configurations are Hopper-tuned, and the same code reached 34.6% utilization on a 3090
against 9.8% on an H100, which indicates a bandwidth-bound rather than compute-bound step.

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
sequence capacity. Qualify 4K, 16K and 32K workloads separately. Any output/action-only trajectory
loss or masked repair objective also requires a validated adapter, mask fingerprint and resume test.

Before executing any mid-training stage, freeze parent identity, objective, data, schedule, optimizer
policy and cost, and qualify resume. Capability, repository and agentic continuation use the combined
600-hour mid-training production reservation; 16K/32K context changes are part of that reservation.
No production RL trainer exists yet; its data, rollout and verifier requirements are in [the
program](program.md#conditional-rl).

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
