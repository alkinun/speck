# Contributing

## Development

```bash
make setup                 # Locked CPU development environment
make quality               # Format, lint, portable tests, catalog, archive integrity
make evidence-test         # Frozen research record checks
make integration-test      # Checks requiring local artifacts or hardware
```

Run focused tests while developing, followed by the complete appropriate gate. GPU kernel,
distributed resume, and scheduler behavior require explicit hardware qualification.

## Code organization

The `speck` package owns reusable behavior. `scripts` provides command entry points. The package's
main areas are `model`, `training`, `data`, `tokenization`, `evaluation`, `export`, `operations`, and
`provenance`. New library code should not import command scripts.

Keep checkpoint parameter names, optimizer state, data order, and report serialization stable during
refactors. Changes to those contracts need targeted behavioral tests and explicit new run identities.
Use `speck.provenance.repository.repository_root` when a checkout root is needed; avoid directory-depth
assumptions. File hashes and report publication belong in `speck.provenance.io`.

## Current code and historical evidence

The [archive manifest](archive/manifest.json) binds the pre-cleanup tree and every preserved artifact.
Historical verification checks those original bytes and source revisions. It does not require the
maintained implementation to remain byte-identical to a completed experiment.

```bash
python -m scripts.archive check
python -m scripts.source_pin_check --base HEAD
```

The source-pin check reports the effect of changes on active and archived evidence separately.
Never change an old result's hash to claim it was produced by new code. Current execution must bind
and qualify the implementation it actually uses. Existing tokenizer-pilot authorization belongs to
the frozen pre-cleanup checkout; use the archive restore command to continue that execution.

## Research records

- Edit draft plans normally in Git. Freeze exact inputs before consequential experiment outputs.
- Preserve protocol amendments and failed attempts with their original identities.
- Keep one coherent result/history per experiment and one finding per durable conclusion.
- Record current state in `research/status.json`; keep the catalog's selected contracts explicit.
- Put literature in `research/literature`, current conclusions in `research/findings`, and paper
  claims in `paper/claims.json`.
- Move completed collections into the archive with an identity manifest.
- Keep datasets, checkpoints, full predictions, and logs in the runtime store.

See the [research workflow](research/WORKFLOW.md) and [artifact policy](research/DATA_MANAGEMENT.md).
Operational documentation describes maintained commands; historical commands remain revision-bound
inside the archive.
