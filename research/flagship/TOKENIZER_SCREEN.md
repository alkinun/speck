# Measured tokenizer-screen handoff

Use this handoff after all three frozen seed-42 arms complete. The command implements the v10
screen ranking and 30-local-GPU-hour check. It produces a review result; confirmation still needs
qualified execution manifests. The D5 audit stays unopened.

## Required inputs

- The selected `tokenizer_pilot_plan_v10.json` and the real static nomination output.
- The three completed `run-summary.json` files. Each summary pins its full `run-result.json`.
- An accounting record with measured spending across **all attempts**, including losing arms,
  interruptions, and work replayed after checkpoint recovery.
- Timing evidence supporting the accounting, pinned by path and SHA-256.

The existing trainer's `active_seconds` includes training, in-loop evaluation, and earlier checkpoint
saves. It excludes setup before its session timer, final checkpoint/result publication, and work lost
after the resumed checkpoint. It is a lower bound on expenditure, not a complete GPU-hour ledger.
Do not add evaluation overhead a second time. Time while a local GPU job is stopped is not GPU use.

## Accounting record

Create a new runtime JSON with this schema. Replace every `null` with measured hours, repeat the run
entry for each of the three actual tokenizer IDs, and pin the timing evidence. Keep `complete: false`
until the totals are reconciled. Missing historical timing must remain explicit rather than becoming
zero. A conservative documented upper bound may cover a timing gap.

```json
{
  "format": "speck_tokenizer_pilot_screen_accounting",
  "format_version": 1,
  "complete": false,
  "other_spent_gpu_hours": null,
  "runs": [
    {
      "tokenizer_id": "mistral-32k",
      "spent_gpu_hours": null,
      "confirmation_run_gpu_hours": null
    }
  ],
  "evidence": [
    {"path": "timing-evidence.json", "sha256": "<actual SHA-256>"}
  ]
}
```

`spent_gpu_hours` is the total already consumed by that screen arm across attempts.
`confirmation_run_gpu_hours` is a full per-run estimate from the completed screen, including observed
setup/evaluation/checkpoint overhead. It must cover at least the measured active duration. Explain any
difference between spending and the estimate in the timing evidence. `other_spent_gpu_hours` covers
pilot preflight/qualification and other charged pilot work not already in those three totals; record
the included jobs explicitly so nothing is counted twice.

The projection is:

```text
already spent = other spending + spending on all three screen arms
remaining = 2 × (baseline full-run estimate + selected-custom full-run estimate)
projected total = already spent + remaining
```

The custom finalist is chosen by the frozen fixed-document macro BPB, then active seconds, vocabulary
size, and ID. A cheaper losing custom cannot replace a winner whose confirmation would exceed budget.
Exactly 30 hours passes; exceeding 30 retains Mistral and leaves D5 unopened.

## Review manifest and command

Create a runtime review manifest. All identities use exactly `path` and `sha256`; relative paths are
resolved against the containing JSON file. Include all three summary identities.

```json
{
  "format": "speck_tokenizer_pilot_screen_review",
  "format_version": 1,
  "plan": {"path": "<selected v10 plan>", "sha256": "<actual SHA-256>"},
  "nominations": {"path": "<real nomination output>", "sha256": "<actual SHA-256>"},
  "accounting": {"path": "accounting.json", "sha256": "<actual SHA-256>"},
  "summaries": [
    {"path": "<Mistral run-summary.json>", "sha256": "<actual SHA-256>"},
    {"path": "<40960 run-summary.json>", "sha256": "<actual SHA-256>"},
    {"path": "<32000 run-summary.json>", "sha256": "<actual SHA-256>"}
  ]
}
```

Run the review from the maintained checkout:

```bash
uv run --no-sync python -m scripts.tokenizer_pilot_screen \
  /path/to/review.json /path/to/new-screen-analysis.json
```

The CPU-only analysis verifies summary/result hashes, paired evaluation documents, common backbone
and training stream, matching FLOP targets, and complete seed-42 coverage. It records input and analysis
implementation hashes and refuses an existing output path. Incomplete accounting produces a blocked
review with no budget projection. A passing projection does not open the audit or launch training.

Record the completed review under `results/` and update `research/status.json` only when real inputs
exist. The scientific runs continue in their frozen checkout as described in [TOKENIZER.md](TOKENIZER.md).

## Terminal-report recovery

A completed checkpoint and all checkpoint-bound evaluation boundaries can recover a report whose
publication failed. `tokenizer_pilot_recover` verifies the frozen execution record, terminal state,
model/optimizer hashes, finite model tensors, evaluation identities, and accounting input. It performs
no training or evaluation replay. The recovered run/summary use schema v2: the unpersisted peak GPU
allocation is `null`, with an explicit missing-measurement marker. Quality, compute, timing, and
selection rules retain their original values. A missing measurement is never replaced by zero,
process RAM, a later GPU observation, or a preflight estimate.

The screen reader accepts this explicit recovery schema. The original analyzer fixture remains
evidence for its original revision; tests and the checked recovery qualify the additive metadata
successor. Schema-v1 runs still require a finite memory measurement.
