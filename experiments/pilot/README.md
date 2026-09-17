# First real-data pilot

This is a bounded engineering experiment for the existing 1.2B KDA/GQA candidate at 4K.
No GPU training has run. It is not a model-quality result or an architecture comparison.

## Recipe

The endpoint is **104,857,600 training tokens**, 800 optimizer steps of 131,072 tokens.
The initial four-worker geometry is one 4K sequence per worker and eight accumulation steps.
Use the frozen tokenizer, FP32 parameter/optimizer storage, BF16 CUDA activations, activation
checkpointing, Liger loss, and Muon/AdamW. The peak learning rate is 3e-4, with 40 warmup steps
and cosine decay to 10%. These are conservative starting settings, not optimized hyperparameters.
The model inherits the qualification geometry, including its three reserved embedding rows.

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

Follow the [hardware qualification](../qualification/README.md), then test numerical/cache parity,
the production loader, and scheduler recovery. Transfer the completed packed corpus and frozen
tokenizer; preserve hashes and update machine-specific paths before binding the scheduler wave.

The pilot ceiling is **50 allocated GPU-hours including failed attempts and overhead**. A four-GPU
job's initial wall limit must be at most 12 hours (48 GPU-hours), reduced for prior usage. The
remaining margin covers shutdown/accounting uncertainty. `budget.json` describes that envelope;
it is not an executable budget guard. Bind it through the existing Slurm accounting workflow before
launch. A raw `speck train` command does not enforce a cumulative GPU-hour limit.

Inspect source-wise loss, gradient health, checkpoint replay, samples, actual mixture exposures,
and end-to-end throughput. Retain intermediate checkpoints. Public capability results at this short
endpoint are diagnostic; do not use this run to claim release quality or architectural superiority.
Choose any larger horizon only after inspecting these measurements.
