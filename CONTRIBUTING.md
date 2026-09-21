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
make smoke
```

`make quality` checks formatting, lint, portable behavior tests, and historical snapshot integrity.
`make evidence-test` additionally verifies frozen historical inputs. Accelerator-specific tests skip
when their dependencies are unavailable. CPU success does not qualify CUDA kernels, GH200
throughput, NCCL, or the scheduler.

The `speck` package owns behavior; `scripts` provides command entry points. Package code must not
import command scripts. Keep data order, checkpoint tensor names, optimizer state, and resume
semantics stable unless a behavioral change is explicit and tested. Model variants retained in the
runtime support checkpoint compatibility; they do not imply active architecture searches.
Remove unreferenced helpers and superseded active documentation rather than adding compatibility
wrappers or parallel plans. Check callers, export dependencies and historical fixtures before
removing runtime functionality. Preserve completed evidence in its original form.

## Records and history

Update the status/next step in PLAN.md after a meaningful transition. Store exact runnable model,
data, tokenizer, and training settings with the experiment. Keep a small result summary containing
Git revision, input identities, metrics, costs, failures, and external output locations. Do not create
another catalog, claim registry, or chain of successor documents for routine engineering changes.
When a decision changes, update the overview, status and affected numeric fields together. Recheck
mixture/budget sums, receipt identities and local links; distinguish selected choices, proposed
recipes and measured outcomes. Historical result receipts retain their original bytes.

Large corpora, checkpoints, caches, and logs stay outside Git. Preserve historical result bytes and
expensive artifacts. The [archive guide](archive/README.md) restores old workflows at their original
revision. Some behavioral tests use their frozen configurations as fixtures while importing the
current implementation; they do not execute the retired experiment suite.

Before committing, inspect `git diff --check` and `git status`. Keep changes on a `codex/` or feature
branch, and retain the pre-simplification snapshot when publishing history. Do not force-push or
rewrite recorded experiment commits.
