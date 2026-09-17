# Contributing

Read [PLAN.md](PLAN.md) for scope. Keep one current plan and one configuration per runnable experiment.
A new idea belongs in a short discussion or Git issue until it becomes the next measured experiment.

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

## Records and history

Update the status/next step in PLAN.md after a meaningful transition. Store exact runnable model,
data, tokenizer, and training settings with the experiment. Keep a small result summary containing
Git revision, input identities, metrics, costs, failures, and external output locations. Do not create
another catalog, claim registry, or chain of successor documents for routine engineering changes.

Large corpora, checkpoints, caches, and logs stay outside Git. Preserve historical result bytes and
expensive artifacts. The [archive guide](archive/README.md) restores old workflows at their original
revision. Some behavioral tests use their frozen configurations as fixtures while importing the
current implementation; they do not execute the retired experiment suite.

Before committing, inspect `git diff --check` and `git status`. Keep changes on a `codex/` or feature
branch, and retain the pre-simplification snapshot when publishing history. Do not force-push or
rewrite recorded experiment commits.
