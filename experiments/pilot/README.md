# First real-data pilot

This is a bounded engineering experiment for the existing 1.2B KDA/GQA candidate at 4K.
The full 800-step pilot has not run. A separate H100 timing experiment completed 48 production
steps on real pilot data with the 800-step learning-rate schedule. This is an engineering learning
and timing check, not a model-quality result or an architecture comparison.

The [local preparation receipt](preparation.json) records a completed corpus and full one- and
four-rank CPU loader scans, with exact fresh-process replay of eight saved microbatches per rank.
The packed corpus contains 105,652,323 training tokens including reserve and whole-document
overshoot, plus 799,536 validation tokens. Both loader geometries consume exactly 104,857,600
training tokens in the declared mixture without repetition. The
[H100 rehearsal](../qualification/h100-result.json) now verifies single-worker CUDA execution and
production model/optimizer/loader/RNG recovery. GH200, collectives, distributed restart, and scheduler
recovery remain checks on the actual allocation.

## Recipe

The endpoint is **104,857,600 training tokens**, 800 optimizer steps of 131,072 tokens.
The initial four-worker geometry is one 4K sequence per worker and eight accumulation steps.
Use the frozen tokenizer, FP32 parameter/optimizer storage, BF16 CUDA activations, activation
checkpointing, Liger loss, and Muon/AdamW. Deterministic PyTorch kernels support strict checkpoint
replay. The peak learning rate is 3e-4, with 40 warmup steps and cosine decay to 10%.
These are conservative starting settings, not optimized hyperparameters.
The model inherits the qualification geometry, including its three reserved embedding rows.
Launch the initial pilot with `--no-compile` to match qualification. Bind that flag in the Slurm wave;
the compiled path requires its own check before use.

| Source | Training weight |
| --- | ---: |
| FineWeb-Edu | 50% |
| Stack-Edu | 15% |
| FineMath 4+ | 15% |
| Cosmopedia v2 | 10% |
| peS2o v3 | 5% |
| FineWiki English | 5% |

This mixture preserves broad text while exercising math and code. It is a preparation choice,
not a quality-selected mixture. Each source has separate validation and shards. Candidate selection
uses finite source-order prefixes with 2x training headroom plus validation allowance, with no planned
repetition. Code candidate shares retain eleven languages from the earlier prepared supply;
post-filter and packed language shares are measured, not assumed equal to candidate shares.

## Prepare on the retained-data machine

`inputs.json` binds existing stock manifests and historical acquisition records. Code payloads are
read from their recorded local acquisition paths; stock text/index identities are reopened. Large
artifacts stay outside Git. Set `speck_base_dir` before starting Python so packing and training
resolve the same volume. The tokenizer directory in `tokenizer.json` is the current machine's path.
Prefer SSD scratch for the preparation work directory: joint exclusion performs random SQLite
index access. Preserve the completed outputs and hashes when moving them to durable storage.

```bash
export speck_base_dir=/mnt/speck-data/speck
uv run --no-sync python -m scripts.evaluation_prepare experiments/pilot/evaluation.json \
  --output "$speck_base_dir/flagship-preparation-20260917/evaluation.json"
uv run --no-sync python -m scripts.pilot_prepare experiments/pilot --stage select \
  --runtime-root "$speck_base_dir" \
  --work-dir "$speck_base_dir/flagship-preparation-20260917/pilot" \
  --evaluation "$speck_base_dir/flagship-preparation-20260917/evaluation.json"
uv run --no-sync python -m scripts.pilot_prepare experiments/pilot --stage exclude \
  --work-dir "$speck_base_dir/flagship-preparation-20260917/pilot"
uv run --no-sync python -m scripts.pilot_prepare experiments/pilot --stage pack \
  --work-dir "$speck_base_dir/flagship-preparation-20260917/pilot"
```

Selection excludes both development and final public benchmark material using exact fields and
informative n-grams. The exclusion stage gives the twelve retained reference partitions precedence,
then applies global exact/near deduplication across all pilot candidates. Packing shuffles each selected source by a seeded content hash, then assigns validation
by an independent seeded content hash and rechecks exact duplicates globally. The shuffle prevents
acquisition language groups from determining which code examples reach the finite token endpoint. Supply shortfalls fail explicitly;
they do not trigger repetition or source reweighting. Completed source selections can be reused;
a partial source selection is preserved for inspection and requires a new work directory.
The existing joint-exclusion stage supports checkpointed resume with the same configuration.

## Before GPU execution

For the current work order, follow [PLAN.md](../../PLAN.md#immediate-order-of-work). The next H100
rental should run the full pilot once its one-worker launch and cumulative cost are bound. The
measured projection is 2.20 hours for training/validation/checkpoints plus approximately 2.15 hours
for one development backend pass. Reserve six single-H100 hours including margin; grading, setup,
transfer and failures still need accounting. This reservation is not an executable billing guard.
Keep the full run in a fresh output directory and preserve the earlier timing experiment separately.
The main-corpus research and expanded coding protocol do not alter this frozen engineering run.

Follow the [hardware qualification](../qualification/README.md), then test numerical/cache parity,
the production loader, and scheduler recovery. Transfer the completed packed corpus and frozen
tokenizer; preserve hashes and update machine-specific paths before binding the scheduler wave.

The pilot ceiling is **50 allocated GPU-hours including failed attempts and overhead**. A four-GPU
job's initial wall limit must be at most 12 hours (48 GPU-hours), reduced for prior usage. The
remaining margin covers shutdown/accounting uncertainty. `budget.json` describes that envelope;
it is not an executable budget guard. Bind it through the existing Slurm accounting workflow before
launch. These four-GPU Slurm settings are not a ready-to-run one-H100 rental binding. A raw
`speck train` command does not enforce a cumulative GPU-hour limit.

Inspect source-wise loss, gradient health, checkpoint replay, samples, actual mixture exposures,
and end-to-end throughput. Retain intermediate checkpoints. Public capability results at this short
endpoint are diagnostic; do not use this run to claim release quality or architectural superiority.
Choose any larger horizon only after inspecting these measurements.

After packing, scan and replay the real loader in separate processes:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=4 -m scripts.loader_check \
  experiments/pilot --batches 6400 --mode scan --output-dir /external/pilot/loader-four
uv run --no-sync torchrun --standalone --nproc-per-node=4 -m scripts.loader_check \
  experiments/pilot --batches 6400 --mode replay --output-dir /external/pilot/loader-four
```

The four ranks collectively read the complete 105M-token schedule on CPU. Reports record source
exposures and exact replay of eight saved microbatches per rank. This validates rank slicing and
fresh-process cursor replay; CUDA transfers, NCCL, optimizer restart, and scheduler behavior remain
separate hardware checks. Use 25,600 batches for a complete one-worker scan into another directory.
