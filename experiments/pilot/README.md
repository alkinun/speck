# First real-data pilot

A bounded engineering run of the 1.2B KDA/GQA reference at 4K on one H100. **It is closed and is
not to be repeated**; follow [PLAN.md](../../PLAN.md#work-order) for current work.
The receipts own the detail:

| Result | Receipt |
| --- | --- |
| 800 steps / 104,857,600 tokens in 2.27 hours; final validation loss 3.379; 13,595 steady tokens/s; 19.3 GiB peak allocated | [Execution receipt](h100-run.json) |
| 2,619 development tasks in 2.12 evaluation hours, with protocol/model identities and export hashes verified | [Development receipt](development-result.json) |
| All eight model/optimizer checkpoints, both exports, grading outputs and recovery evidence verified locally | [Backup receipt](backup-result.json) |
| Corpus built; one- and four-rank CPU loader scans with exact fresh-process replay | [Preparation receipt](preparation.json) |
| Single-worker CUDA execution and model/optimizer/loader/RNG recovery | [H100 rehearsal](../qualification/h100-result.json) |
| Eight plain local completions from the final export | [Completion preview](completion-preview.json) |

| Development metric | Correct / evaluated | Result |
| --- | ---: | ---: |
| GSM8K strict exact match | 0 / 253 | 0.00% |
| GSM8K flexible extraction | 1 / 253 | 0.40% |
| IFEval strict prompt accuracy | 12 / 101 | 11.88% |
| IFEval strict instruction accuracy | 37 / 152 | 24.34% |
| HumanEval+ custom compiled tests, pass@1 | 0 / 33 | 0.00% |
| ARC-Challenge normalized accuracy | 44 / 222 | 19.82% |
| HellaSwag normalized accuracy | 530 / 2,010 | 26.37% |

These are frozen custom development subsets scored with BF16 greedy decoding and no chat/SFT, not
official leaderboard scores; no final-test partition was evaluated. The base is weak: all 33 code
executions failed, and the local preview completions repeat and fail simple facts and arithmetic.
At 105M tokens these results do not isolate architecture, corpus or training-scale effects.

## Recipe

The endpoint is **104,857,600 training tokens**: 800 optimizer steps of 131,072 tokens. Use the
frozen tokenizer, FP32 parameter/optimizer storage, BF16 CUDA activations, activation checkpointing,
Liger loss and Muon/AdamW with deterministic kernels. Peak learning rate 3e-4, 40 warmup steps,
cosine decay to 10%, launched with `--no-compile`. These are conservative starting settings, not
optimized hyperparameters.

| Source | Training weight |
| --- | ---: |
| FineWeb-Edu | 50% |
| Stack-Edu | 15% |
| FineMath 4+ | 15% |
| Cosmopedia v2 | 10% |
| peS2o v3 | 5% |
| FineWiki English | 5% |

This is a preparation choice, not a quality-selected mixture. Each source has separate validation
and shards. The packed corpus holds 105,652,323 training tokens including loader reserve and
whole-document overshoot, plus 799,536 validation tokens; both loader geometries consume exactly
104,857,600 training tokens without repetition.

## Reproduce the preparation

`inputs.json` binds the existing stock manifests; large artifacts stay outside Git. Set
`speck_base_dir` before starting Python so packing and training resolve the same volume.

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

Selection excludes development and final benchmark material by exact fields and informative
n-grams; the exclusion stage then applies global exact/near deduplication. Packing shuffles each
source by a seeded content hash and assigns validation by an independent hash. Supply shortfalls fail
explicitly rather than repeating or reweighting. Scan and replay the real loader in separate
processes:

```bash
uv run --no-sync torchrun --standalone --nproc-per-node=4 -m scripts.loader_check \
  experiments/pilot --batches 6400 --mode scan --output-dir /external/pilot/loader-four
uv run --no-sync torchrun --standalone --nproc-per-node=4 -m scripts.loader_check \
  experiments/pilot --batches 6400 --mode replay --output-dir /external/pilot/loader-four
```

## Lessons for later rentals

`scripts.pilot_rental` ran this workflow and was removed after closeout; see
`git log -- speck/operations/pilot_rental.py`. The execution receipt preserves the original export
and evaluation failures alongside the recovery.

- **A trainer stop is not a billing stop.** No tool in this repository deletes an instance.
- **Supervise long transfers and deferred grading with persistent services**, not task-bound workers.
- **Some hosts deny user namespaces**, which the code grader requires. `--defer-code-grading` keeps
  generation on the GPU host and lets `scripts.code_grade` finalize locally against the same
  protocol.
- **Verify backups by hash**, from the root of the local copy, before reclaiming remote checkpoints.
- **Never reuse or reset a budget ledger** to unblock an attempt.
- **Compare exports on matching precision paths.** Native versus exported BF16 logits, then
  cached/full semantics in FP32 on the same weights, separate rounding from wrapper drift.
