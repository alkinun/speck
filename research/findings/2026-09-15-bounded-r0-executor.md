# Bounded R0 executor: local qualification

The [local receipt](../../results/systems/r0-executor-local-qualification-20260915.json) at `f59bc70`
records 12 no-execution request bindings for the six checked exact shapes at one/four workers and
23 focused engineering tests. Tiny dense and KDA CPU models pass optimization plus model/optimizer
next-step parity after checkpoint reload. Two actual Gloo ranks pass accumulation, checkpoint replay
and final-weight agreement. These are CPU fixtures, not GH200 performance or model-quality evidence.

The executor preserves effective vocabulary 32,003 while generating inputs below 32,000. It uses
the existing optimization step and atomic checkpoint implementation. Actual execution records FP32
parameter/optimizer storage and BF16 CUDA activations; the earlier shape report's BF16 weight-byte
estimate is not promoted into an observed training-memory claim.

Process inspection showed that the installed torchrun launcher creates separate sessions for rank
workers. The supervisor therefore starts and tracks each rank directly, with an attempt-specific file
rendezvous, a wall deadline and termination grace. A real four-process test uses workers that ignore
SIGTERM and checks that all four are killed. Interrupt, incomplete-rank, changed-evidence and budget
failure tests also pass. Every attempt retains its request, progress, results, logs and checkpoints;
unresolved attempts block further reservations in the shared ledger.

The 70-hour ceiling is enforced against external prior hours plus conservative reservations. Observed
cost charges every declared allocated GPU, including unused GPUs in a single-worker attempt. It covers
supervisor launch through termination; scheduler startup/idle/cleanup and other R0 work require
separate reconciliation. Passing a short synthetic step benchmark does not establish sustained
production throughput or a complete training-cost forecast.

The preserved first focused qualification directory contains 22 passes and one test failure caused by
a missing test import. The corrected successor has 23 passes; original files/logs remain in the hashed
artifact inventory. Receipt assembly also rejected duplicate pytest symlink aliases before publication;
only physical rank results are counted. Full quality passed 1,267 tests, 10 skipped, 125 deselected,
plus formatting, lint, catalog, archive and source-pin checks. Checked local artifacts reopened by hash.

See [R0_EXECUTOR.md](../flagship/R0_EXECUTOR.md) for the review-only command, explicit site invocation,
settings and timing/recovery boundaries. No GPU qualification, flagship training or new source download
was launched here. Fresh-process RNG/checkpoint restart, independent numerical-reference and cached
inference parity, scheduler/requeue integration, production-loader throughput, and actual GH200/arm64/
four-GPU NCCL qualification remain open. The proposal is still under evaluation with no confirmed
access date. Source preparation, source-use approval and scientific model-launch authority remain distinct.

The maintained status update is an [explicit checked successor](../../results/systems/r0-executor-status-successor-20260915.json):
only `lc_hardware` changes. Its previous bytes match the earlier coherence audit and are preserved in
research history. Current cross-contract checks pass. The source-pin diff against the predecessor
correctly flags this intentional status change; the earlier implementation-validation pass does not
hide that later evidence-bound update.
