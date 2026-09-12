# 2026-09-12 — Pre-compute Instruct data pilot

## Context

Completed a local RTX 3090 pilot to exercise dataset compilation, native SFT, and evaluation before
the compute grant. Speck2 is the 140M model; these results do not select the flagship mixture or
establish transfer to larger models. Work tracked in SPE-177, SPE-178, and SPE-179.

## Work performed

- Built a configurable 17-source English prompt/completion compiler with pinned sources, exact
  quotas, filtering, global prompt/family deduplication, exclusions, and resumable source staging.
- Added native local-data preparation with explicit splits and final-response-only masking.
- Trained the same pinned Speck2 base on 100K conversations for one and two epochs, and on a
  capacity-adjusted 500K mixture for one epoch. Native preparation rejects over-4K rows.
- Evaluated final checkpoints against the release using fixed diagnostics, common validation,
  BananaMind, and the pinned Open SLM protocol. Fixed Transformers export cache configuration.

## Decisions

| Metric | Released | 100K, 1 epoch | 100K, 2 epochs | 500K, 1 epoch |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic correct / 215 | 10 | 20 | 20 | 35 |
| Exact-format answers / 215 | 7 | 1 | 5 | 19 |
| Common validation loss | 1.2671 | 1.2488 | 1.2348 | 1.1481 |
| BananaMind correct / 350 | 157 | 163 | 167 | 166 |
| Open SLM average | 45.10 | 44.79 | 45.02 | 44.02 |
| Open SLM Intelligence Index | 21.33 | 21.02 | 21.10 | 19.65 |

The larger mixture improved the small instruction diagnostic and held-out fit, but Open SLM
regressed and qualitative failures remained. Keep the released model as default. This pilot is
finished; the reusable compiler and SFT path are its main engineering outcome.

The 500K build required owner-approved source rebalancing and additional benchmark/validation
exclusions. Data allocations, masking, context limits, token exposure, and schedule horizons differ
across recipes. These comparisons do not isolate dataset quality or unique-sample count. The
diagnostic is not official IFEval, and exact/near matching does not prove semantic decontamination.

## Evidence and links

- [Consolidated result and artifact hashes](../../results/Speck2-Instruct-Data-Pilot/summary.json)
- [100K compiler recipe](../../experiments/Speck-Instruct-Starter/README.md) and
  [capacity-adjusted 500K recipe](../../experiments/Speck-Instruct-Starter500K/README.md)
- [SFT reproduction](../../experiments/Speck2-140M-Instruct-Starter100K/README.md)
- Detailed logs, predictions, checkpoints, and dataset identities remain under
  `/mnt/speck-data/speck/`; runtime directories are identified in the result.
- Uncommitted working notes and intermediate result snapshots were consolidated at closeout;
  their hash-checked copies remain in
  `/mnt/speck-data/speck/evaluations/Speck2-Instruct-Data-Pilot/working-history-20260912/`.

## Open questions

How well the mixture transfers to a stronger base, and which source/finishing changes improve
instruction adherence without reducing general capability, remain untested.

## Next actions

Resume post-training development within the [flagship plan](../flagship/POST_TRAINING.md) when
compute is available. No further local training or release promotion is scheduled by this pilot.
