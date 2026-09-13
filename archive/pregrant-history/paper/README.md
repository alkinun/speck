# Flagship paper workspace

This directory is the publication layer of the Speck research compendium. The scientific scope and
failure rules are frozen in [`research/flagship/PAPER.md`](../research/flagship/PAPER.md); the
machine-readable status of each paper claim lives in [`claims.json`](claims.json).

```text
paper/
  claims.json   Claim status and evidence map.
  analysis/     Scripts that generate paper figures and tables from checked results.
  figures/      Generated figures and source manifests.
  tables/       Generated tables and source manifests.
  manuscript/   Manuscript and supplement source.
  references/   Bibliography source and citation audit.
```

This workspace is intentionally skeletal before grant outputs. It must not accumulate speculative
prose or manually copied metrics that later become difficult to audit.

## Rules

- Manuscript claims cite IDs from `claims.json`.
- Figures and tables are generated from checked `results/` records by scripts under `analysis/`.
- Every generated output records source paths/hashes, analysis revision, and command.
- W&B, Linear, notebook entries, and terminal output are not valid sole sources for paper numbers.
- Failed, refuted, and unresolved claims stay visible in the registry.
- The abstract and headline are written only after their required claims become supported.
- Final paper and supplement builds run from a clean checkout under a pinned environment.
- A clean-room reviewer regenerates the final outputs and audits every supported claim before release.

Validate the current catalog and claims with:

```bash
uv run --extra cpu python -m scripts.research_catalog
```
