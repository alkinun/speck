# Contributing

Speck is a research codebase with reproducibility-sensitive data, training, and evaluation paths.
Keep changes focused, preserve artifact contracts deliberately, and include tests for behavioral
changes.

Consequential research follows the [research workflow](research/WORKFLOW.md). The central
[research catalog](research/catalog.json) defines collection roles and authority; validate it with
`uv run --extra cpu python -m scripts.research_catalog`.

The [2026-09-08 cleanup audit](docs/code_cleanup_2026-09-08.md) records source-pinned cleanup
follow-ups. Source-pinned files that require evidence requalification are explicitly excluded from
formatting, while remaining part of lint and test checks.

## Development Setup

Run commands from the repository root. Install the CPU environment and development tools for the
default test suite:

```bash
uv sync --extra cpu --group dev
```

Use `--extra gpu` instead of `--extra cpu` when exercising CUDA-specific behavior. The GPU extra
targets the CUDA 12.8 PyTorch index.

## Quality Checks

Run the complete local gate before submitting a change:

```bash
uv run --extra cpu --group dev ruff format --check --config ruff-format.toml .
uv run --extra cpu --group dev ruff check .
uv run --extra cpu --group dev pytest -q
uv run --extra cpu python -m scripts.research_catalog
```

`make quality` runs the same four checks, and `make setup` installs the locked CPU development
environment.

Apply the formatter with:

```bash
uv run --extra cpu --group dev ruff format .
```

During development, run the narrowest relevant test file first, then run the complete suite before
finishing. Tests must not depend on network access, maintainer-local cache contents, or a GPU unless
they are explicitly isolated as integration tests.

## Change Guidelines

- Keep bug fixes, refactors, generated artifacts, and documentation reorganizations in separate
  commits when they can be reviewed independently.
- Add or update focused tests before changing reproducibility-sensitive behavior.
- Treat experiment JSON, packed-data manifests, checkpoint metadata, and evaluation results as
  versioned contracts. Document intentional schema changes.
- Keep generated datasets, checkpoints, model exports, logs, and benchmark working directories
  outside the repository. Runtime artifacts belong under `~/.cache/speck` by default.
- Pin remote revisions and checksums when a workflow claims to be reproducible.
- Write commands in documentation so they run from a clean checkout at the repository root.
- Update nearby documentation when changing a CLI, configuration key, artifact path, or runtime
  prerequisite.
- Record consequential work context in a dated `research/notebook/` entry, then promote stable
  conclusions into `findings/` only when checked evidence exists.
- Keep external-paper claims in `papers/`, Speck conclusions in `findings/`, and manuscript claims in
  `paper/claims.json`; do not collapse these evidence levels.
- Treat Linear as the work queue and W&B as a monitoring mirror. Neither may be the only record of a
  scientific result, failed gate, or artifact identity.
- Assign retention and backup requirements before producing expensive or irreplaceable artifacts;
  follow [the data and artifact management plan](research/DATA_MANAGEMENT.md).

## Documentation

The README is the task-oriented entry point. Put detailed operational guidance in `docs/` and link
to it from the relevant README section. Use repository-relative links for local files and verify
that every documented command matches the corresponding CLI help.
