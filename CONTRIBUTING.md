# Contributing

Read [PLAN.md](PLAN.md) for current decisions and work order; [the program overview](docs/program.md)
connects the stages, and [main-data plan.json](experiments/main-data/plan.json) owns working numeric
targets. Keep one configuration per runnable experiment.
A new idea belongs in a short discussion or Git issue until it becomes the next measured experiment.
The first release centers on data across all six training stages. The backbone is already fixed by
declaration, not selected over a control, so go straight to data experiments before each production
stage. Keep training/inference claims tied to measurements and make no architecture superiority or
parity claim. Reserve the deferred architecture/efficiency comparison, MoE, attention residuals and
broad architecture/size searches for later programs.

## Development

```bash
make setup
make quality
make plan-check
make smoke
```

`make quality` checks formatting, lint, portable behavior tests, and historical snapshot integrity.
`make plan-check` is offline design arithmetic: it reconciles the reservation table, the study
packets, record references, and the GPU-hour figures written in prose against
[`plan.json`](experiments/main-data/plan.json). Both run in CI. If a budget moves, change the plan
and the documents in one commit; `check_documents.py` fails the build when a prose figure is left
behind. `make evidence-test` additionally verifies frozen historical inputs. Accelerator-specific
tests skip when their dependencies are unavailable. CPU success does not qualify CUDA kernels,
GH200 throughput, NCCL, or the scheduler.

Set `TMPDIR` to a path with real disk space; the distributed tests exhaust a small `/tmp` tmpfs and
report the exhaustion as a test failure.

The `speck` package owns behavior; `scripts` provides command entry points. Package code must not
import command scripts. Keep data order, checkpoint tensor names, optimizer state, and resume
semantics stable unless a behavioral change is explicit and tested. The runtime implements only
what the reference model and its preserved control use: global attention with optional RoPE, KDA
and SwiGLU. Earlier variants live in Git history.
Remove unreferenced helpers and superseded active documentation rather than adding compatibility
wrappers or parallel plans. Check callers, export dependencies and historical fixtures before
removing runtime functionality. Preserve completed evidence in its original form.

## Records and history

Update the status/next step in PLAN.md after a meaningful transition. Store exact runnable model,
data, tokenizer, and training settings with the experiment. Keep a small result summary containing
Git revision, input identities, metrics, costs, failures, and external output locations. Do not create
another catalog, claim registry, or chain of successor documents for routine engineering changes.
When a decision changes, update the overview, status and affected numeric fields together. Recheck
mixture/budget sums and local links; distinguish selected choices, proposed recipes and measured
outcomes. Historical result receipts retain their original bytes.

Each fact has one owning record; others reference it rather than copy it. Per-source identity,
inventory, gates and blockers live in
[`source-readiness.json`](experiments/main-data/source-readiness.json), bank-to-source assignment in
the [source registry](experiments/main-data/source-registry.json), and numeric targets in
`plan.json`. A design record refers to another repository file by `path` alone, because Git already
versions it and a copied digest or status goes stale on the next edit. Files outside Git carry a
`sha256`. Result receipts and signed records keep the digests they were written with.
`speck.provenance.io.check_reference` enforces this in `make plan-check`.

Large corpora, checkpoints, caches, and logs stay outside Git. Preserve historical result bytes and
expensive artifacts. The [archive guide](archive/README.md) restores old workflows at their original
revision. Some behavioral tests use their frozen configurations as fixtures while importing the
current implementation; they do not execute the retired experiment suite.

Before committing, inspect `git diff --check` and `git status`. Keep changes on a `codex/` or feature
branch, and retain the pre-simplification snapshot when publishing history. Do not force-push or
rewrite recorded experiment commits.
