# 2026-09-09 — Research compendium organization

## Context

The repository grew to more than 1,200 tracked research plans, experiment configs, literature notes,
findings, and result records before the flagship allocation began. The material was mostly separated by
type, but there was no single machine-readable catalog, chronological notebook convention, artifact
retention policy, or paper claim registry.

## Work performed

Reviewed research-compendium guidance, FAIR principles, RO-Crate, ENCORE, reproducible-ML guidance,
and the existing Speck path/hash relationships. Inventoried the repository collections and storage
sizes. Added a central catalog and validator, research lifecycle, data/artifact management plan,
notebook convention, paper workspace with claim-level evidence tracking, one-command local quality
gate, and commit-pinned clean-run CI workflow.

## Decisions

- Keep the repository as one research compendium containing code, contracts, small evidence, findings,
  and manuscript sources.
- Keep large datasets, checkpoints, complete logs, and traces outside Git with checked identities.
- Preserve existing experiment, result, finding, paper-note, and archived-contract paths because their
  original manifests cite them. A visual mass move would damage provenance without improving science.
- Treat Linear as the work queue and W&B as a monitoring mirror, never as sole scientific records.
- Require the lifecycle question → literature → preregistration → implementation → execution → analysis
  → finding → claim → release for consequential work.

## Evidence and links

- [`research/catalog.json`](../catalog.json)
- [`research/WORKFLOW.md`](../WORKFLOW.md)
- [`research/DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md)
- [`paper/claims.json`](../../paper/claims.json)
- [`.github/workflows/quality.yml`](../../.github/workflows/quality.yml)
- [The Turing Way research compendia](https://book.the-turing-way.org/reproducible-research/compendia/)
- [FAIR Guiding Principles](https://doi.org/10.1038/sdata.2016.18)
- [RO-Crate](https://doi.org/10.3233/DS-210053)
- [ENCORE](https://doi.org/10.1038/s41467-024-52446-8)

## Open questions

- Which DOI-minting archival repository will hold the final compendium?
- Where will the independent second copy of R3 checkpoints and sealed authority records live?
- Will the paper build use LaTeX directly, Quarto, or another reproducible manuscript tool?
- Which full per-example outputs are small enough to retain in Git versus external storage?

## Next actions

- Keep `paper/claims.json` synchronized as grant evidence arrives.
- Create per-arm manifests and paper-output IDs before each flagship experiment.
- Select and rehearse the independent backup and publication archive before the first irreplaceable
  flagship checkpoint.
- Add the manuscript build only after the venue/toolchain decision; do not hand-copy result values.
- Complete SPE-172's independent backup/restore and archive selection before the first irreplaceable
  flagship checkpoint, then complete SPE-171's clean-room claim audit before release.
