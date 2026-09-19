# First real-data pilot

This is a bounded engineering experiment for the existing 1.2B KDA/GQA candidate at 4K.
Training completed all 800 steps / 104,857,600 tokens in **2.27 hours**. Final validation loss was
**3.379**, steady throughput **13,595 tokens/s**, and peak allocated memory **19.3 GiB**.
Export and tokenizer checks passed after inspected recovery. The remote endpoint subsequently
became unavailable; access was restored on September 19 and the disk survived. The interrupted
evaluation had only 84 GSM8K outputs and no complete result. A fresh, separate development
evaluation is now running from the same export, alongside resumed checkpoint backups.
Complete checkpoints 100, 200, 300 and 800 are verified locally, including the final model and optimizer.
The final export has been reconstructed and verified on CPU from retained inputs. The
[execution receipt](h100-run.json) retains measurements and the original export failure.
The original eight-hour watcher ended at **2026-09-19 04:11 UTC** without observing restored access.
Its last successful remote observation before the outage was September 18 at 20:57 UTC.
Terminal states/logs and backup receipts are preserved in a hashed local evidence snapshot.
Checkpoints 400–700, the remote export backup and complete capability outputs remain unconfirmed;
the CPU reconstruction is a separate verified artifact. The original execution deadline has
expired. After the user restarted one H100, a **new four-hour evaluation reservation** was bound
through **09:59:06 UTC on September 19**; the original reservation was not reused. No training or
export regeneration is needed. The evaluator has no resume support, so the prior 84 rows remain
separate and the complete frozen partition is rerun. Existing backup/grading services and the
bounded milestone watcher were resumed without duplicates. Provider billing remains unknown;
21 prior conservative hours + 6 original hours + 4 restored-session hours are reservation records,
not the actual bill. [h100-run.json](h100-run.json) binds the preserved files and new launch.
A separate H100 timing experiment completed 48 production
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

## Local qualitative preview — 2026-09-19

The user-requested [completion preview](completion-preview.json) ran eight newly written plain
completion prompts on the local RTX 3090 using the reconstructed final export. All 18 export
file hashes were reverified before loading. BF16 greedy decoding with a 64-token limit completed
in 11.98 seconds including identity checks, loading and cold startup; peak allocated memory was
2.55 GiB. These short local completions do not forecast the full rental evaluation's runtime.

All eight outputs reach the cap and show repetition, including repeated blank lines. The factual
prompt does not name France's capital, both simple arithmetic prompts fail, and the explanatory
prose is unreliable. The Python addition completion starts correctly with `a + b`, then repeats
function definitions. No generated code was executed. The current base does not demonstrate
useful general completion ability in this preview. Its 105M-token training exposure is small;
these observations alone do not isolate training scale, data, decoding or implementation effects.

Complete prompts, token IDs, unedited outputs, runner and hashes are retained under
`/mnt/speck-data/speck/h100-pilot-execution-20260918/local-completion-preview-20260919`;
open `samples.md` for all eight prompt/completion pairs. Preserve this packet as a qualitative
development baseline, not a benchmark accuracy score or final-test set. The frozen H100 evaluation,
training recipe and live backup/grading jobs were not changed.

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

For the current work order, follow [PLAN.md](../../PLAN.md#immediate-order-of-work). The
[one-H100 launch packet](../../docs/pilot-rental.md) is built and CPU-validated, with a six-hour
supervisor and cumulative budget reservations. The active H100 passed host preflight. The
measured projection is 2.20 hours for training/validation/checkpoints plus approximately 2.15 hours
for one development backend pass. Reserve six single-H100 hours including margin; grading, setup,
transfer and failures still need accounting. The supervisor bounds execution; it does not stop
provider billing. [rental-readiness.json](rental-readiness.json) records the archive and local checks.
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
