# 2026-09-12 — Tokenizer pilot screen materialization

## Context

Can the already authorized three-run D5 screen be reduced to immutable executable identities without
opening the sealed D5 audit or granting flagship training authority?

## Work performed

- `tokenizer_pilot_plan_v10.json`, which authorizes only the seed-42 Mistral, compression-endpoint,
  and compact-endpoint screen.
- The completed 1,200,007,273-reference-token fixed document stream and shared post-endpoint
  continuation.
- The unopened six-category tokenizer evaluation partition.
- The exact `ladder-60m` model geometry and corrected RTX 3090 batch-four throughput preflight.

Implemented a dedicated screen materializer and froze
`tokenizer_pilot_runs_v1/plan.json`. Every run binds the Git revision, materializer implementation,
tokenizer model, model/backbone geometry, fixed and continuation shard manifests, evaluation documents,
optimizer settings, and both stopping views. Publication is exclusive and leaves every run in
`materialized_not_started` state.

## Decisions

The optimizer inherits the already qualified 65,536-token Muon recipe: learning rate 0.0015, weight
decay 0.1, clip 1.0, Torch loss, and cosine decay to a 0.1 scale. The 120-step warmup preserves the
approximately 0.65% inherited warmup fraction at the 18,311-step reference horizon. Learning-curve
evaluation is scheduled every 916 steps (approximately 5%) and checkpoints every 1,831 steps
(approximately 10%); fixed-document and final fixed-FLOP boundaries remain mandatory regardless of
those intervals.

## Evidence and links

- [Run materialization plan](../flagship/tokenizer_pilot_runs_v1/plan.json)
- [Authorizing pilot successor](../flagship/tokenizer_pilot_plan_v10.json)
- [Tokenizer protocol](../flagship/TOKENIZER.md)

## Open questions

A trainer and document-level evaluator must still consume these manifests exactly. Real training,
evaluation, and checkpoint overhead remain unmeasured, so confirmation execution cannot yet be
authorized.

## Next actions

This change materializes no model output, runs no training, opens no audit, selects no tokenizer, and
grants no flagship training authority. Confirmation seeds remain blocked until all three screen runs
complete and their measured training, evaluation, and checkpoint overhead is reconciled against the
30-hour ceiling. Commit the frozen plan, materialize its three run records from a clean tree, then
implement and fixture-qualify the exact manifest consumer before starting the screen.
