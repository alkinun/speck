# Contributing

[PLAN.md](PLAN.md) gives status and the work order, the [program design](docs/program.md) the
method and experiments, and [plan.json](experiments/main-data/plan.json) the numbers.

## Development

```bash
make setup
make quality      # formatting, lint, tests
make plan-check   # plan arithmetic, ladder configs, readiness, manifests, throughput packet, doc links
make smoke
```

CI runs all of them. Set `TMPDIR` to a
path with real disk space; the distributed tests fill a small `/tmp` tmpfs and report it as a test
failure. CPU success does not qualify CUDA kernels, GH200 throughput, NCCL or the scheduler.

`speck/` owns behavior; `scripts/` are thin entry points, and package code never imports them.
Keep data order, checkpoint tensor names, optimizer state and resume semantics stable unless a
change is explicit and tested. Files that exports copy by path (`speck/model`,
`speck/training/optimizers.py`, `speck/transformers_*.py`) stay where they are.

## Keeping the repository small

- **One owner per fact.** plan.json owns numbers, source-readiness.json per-source gates, the source
  registry bank assignments, and PLAN.md status. Other documents link to them rather than restating
  figures; the only tables that render plan figures are checked by `make plan-check`.
- **Delete, don't archive in place.** Remove superseded documents, records, scripts and helpers.
  [Git history](README.md#history) keeps them. No
  compatibility wrappers, parallel plans, dated narratives or successor chains.
- **Ideas are not plans.** A new idea stays in an issue until it becomes a predeclared record under
  [experiments/ladder/records](experiments/ladder/records).
- **Receipts keep their bytes.** A result receipt or signed record is never edited. Design records
  refer to repository files by `path` alone; files outside Git carry a `sha256`
  (`speck.provenance.io.check_reference` enforces this).
- When a decision changes, update the design, PLAN.md and the affected plan.json fields in the same
  change.

Each run stores its exact model, data, tokenizer and training settings, and keeps a small result
receipt: Git revision, input identities, metrics, costs, failures and output locations. Large
corpora, checkpoints and logs stay outside Git.

Work on a feature branch; inspect `git diff --check` and `git status` before committing.
Do not force-push or rewrite recorded experiment commits.
