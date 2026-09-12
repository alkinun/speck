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
- [Checked materialization result](../../results/data/tokenizer-pilot-screen-materialization-20260912.json)
- [Authorizing pilot successor](../flagship/tokenizer_pilot_plan_v10.json)
- [Tokenizer protocol](../flagship/TOKENIZER.md)

## Open questions

The three seed-42 records were materialized from clean commit `29c8067` under
`/mnt/speck-data/speck/tokenizer-pilot-runs-v1`; no model output was created. A trainer and
document-level evaluator must still consume these manifests exactly. Real training,
evaluation, and checkpoint overhead remain unmeasured, so confirmation execution cannot yet be
authorized.

Pre-trainer review then found that v10's fixed-FLOP section uses 415,552,512 reference FLOPs/token,
while its bound batch-four preflight and current accounting use 415,543,296. The corrected 40,960
exact token stop is 1,125,458,190 rather than 1,125,459,741. Both values map to the same 1,125,515,264
optimizer boundary, but the exact scientific view must still be corrected. The v1 records are therefore
preserved as a pre-output defect and denied execution.

The corrected FLOP equation and all three exact stops are now frozen separately. A distinct v2
materializer independently checks the scale equations against the bound preflight result and published
three corrected records from clean commit `3d7a95b`. These records remain execution-blocked; they do not
retroactively repair or overwrite v1.

A subsequent real-input qualification hashed all 63 packed shards plus tokenizer, parent, correction,
and evaluation inputs (76 unique files total), parsed all 12,470 evaluation documents and 60,035,301
UTF-8 bytes, and exercised initial, resumed, and final-batch cursors for every tokenizer. All gates
passed without constructing a model. CUDA execution and checkpointed training remain blocked.

The first instrumented exact-shape CUDA checkpoint/replay qualification then passed checkpoint
publication/load, second-step loss equality, and data-cursor equality. Model and optimizer states each
differed by a maximum `6.103515625e-05` (`2^-14`), so the preregistered exact-equivalence gate failed.
The checkpoint is retained only as failed qualification evidence. No tolerance is applied
retroactively; a successor must freeze it before a fresh retry.

A successor froze exact loss and cursor plus `2^-13` model/optimizer absolute-error limits before a
fresh retry. The retry passed: model error remained `2^-14`, optimizer error was
`7.62939453125e-05`, and both were below the frozen limit. This qualifies eager exact-shape checkpoint
resume mechanics only; compiled training, document NLL, and screen execution remain blocked.

The separately frozen CUDA document-NLL policy then passed on the first immutable document from all
six categories. Five shorter documents matched exactly between one chunk and production 4K chunks;
the 19,629-token science document differed by `3.8806e-05` nats/token, below the frozen `1e-4` limit.
Peak allocated memory was 5.24 GB. These are qualification outputs, not tokenizer-quality evidence.

## Next actions

This change materializes no model output, runs no training, opens no audit, selects no tokenizer, and
grants no flagship training authority. Confirmation seeds remain blocked until all three screen runs
complete and their measured training, evaluation, and checkpoint overhead is reconciled against the
30-hour ceiling. Freeze a corrected accounting successor, make its materializer independently verify
the FLOP equations, publish distinct v2 records, and fixture-qualify the exact manifest consumer before
starting the screen.
