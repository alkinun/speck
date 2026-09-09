# Research compendium

This repository is both research software and the working compendium for the first Speck flagship.
It preserves the question, literature, preregistrations, implementation, run identities, results,
interpretation, and paper claims needed to inspect the work without relying on chat history, W&B, or
Linear.

Run the catalog check from the repository root:

```bash
uv run --extra cpu python -m scripts.research_catalog
```

The machine-readable [`catalog.json`](catalog.json) is the directory and authority map. It identifies
which collections are active, evidentiary, external, historical, operational, or publication output.
It also validates the research notebook and [`paper/claims.json`](../paper/claims.json).

## Start here

| Need | Source |
| --- | --- |
| Current model, scope, and experiments | [`flagship/README.md`](flagship/README.md) |
| Paper question and claim standard | [`flagship/PAPER.md`](flagship/PAPER.md) |
| Readiness before allocated compute | [`flagship/PREGRANT.md`](flagship/PREGRANT.md) |
| Exact phase and GPU-hour plan | [`flagship/plan.json`](flagship/plan.json) |
| Paper claim status | [`paper/claims.json`](../paper/claims.json) |
| Current Speck evidence | [`findings/README.md`](../findings/README.md) |
| Complete historical narrative | [`findings/ARCHIVE.md`](../findings/ARCHIVE.md) |
| External literature | [`papers/README.md`](../papers/README.md) |
| Experiment status | [`experiments/README.md`](../experiments/README.md) |
| Checked result status | [`results/README.md`](../results/README.md) |
| Research process | [`WORKFLOW.md`](WORKFLOW.md) |
| Data and artifact policy | [`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md) |
| Chronological research notebook | [`notebook/`](notebook/) |

## Authority boundaries

- **Active authority:** only `research/flagship/` and its explicitly named successors define work for
  grant 1.
- **Scientific evidence:** checked result JSON plus a finding that interprets it within its declared
  limitations. A config, notebook entry, W&B chart, or Linear comment is not a result.
- **External evidence:** `papers/` records what other researchers report. It is never silently promoted
  to a Speck finding.
- **Historical evidence:** `research/paper-1/` and retired experiment/result families remain available
  for provenance but cannot authorize work.
- **Operational state:** Linear coordinates work and W&B monitors runs. Both may disappear without
  changing the durable scientific record.
- **Publication claims:** `paper/claims.json` is the bridge between evidence and manuscript language.
  A claim cannot become supported merely because prose was written.

## Collections

### Active flagship

[`flagship/`](flagship/) holds the active charter, paper contract, data and architecture protocols,
integration checks, tokenizer decision, execution plan, and readiness gate. Planning targets are not
runnable experiments until their data, hardware, analysis, and launch receipts exist.

### Reusable protocol

[`architecture-promotion-v1/`](architecture-promotion-v1/) retains the statistical and evaluation
infrastructure reused by the flagship: paired non-inferiority, power, cost envelopes, internal
retrieval/composition protocols, RULER disposition, and historical evidence mapping. It is governance,
not an invitation to reactivate every archived axis.

Validate it separately with:

```bash
uv run --extra cpu python -m scripts.research_contract_validate \
  research/architecture-promotion-v1 \
  --tokenizer-experiment experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope
```

### Retired Paper 1 program

[`paper-1/`](paper-1/) contains the retired tri-axis/finalist contracts. Checked results hash and cite
them, so they remain immutable history. They have no launch or paper-scope authority.

### Notebook, findings, and paper

The [`notebook/`](notebook/) is chronological process memory: attempts, discussions, unresolved
questions, and handoffs. [`findings/`](../findings/) is distilled scientific memory: conclusions,
uncertainty, limitations, and decisions linked to exact evidence. [`paper/`](../paper/) contains the
claim registry and generated manuscript workspace. These are deliberately separate.

## Why old paths stay in place

The repository already contains hundreds of cryptographically linked plans and results. Moving them
would not improve reproducibility; it would invalidate the names recorded by the original run. The
catalog supplies a clean conceptual structure while preserving those research objects exactly where
their manifests say they are. New work follows the lifecycle and naming rules in `WORKFLOW.md`.

## External standards used

This organization follows the common research-compendium separation of data, method, and output; the
FAIR emphasis on findability, access metadata, interoperability, reuse, and provenance; and the
Research Object idea that a paper should package its code, inputs, outputs, workflows, and narrative.
See:

- [The Turing Way: Research Compendia](https://book.the-turing-way.org/reproducible-research/compendia/)
- [FAIR Guiding Principles](https://doi.org/10.1038/sdata.2016.18)
- [Packaging research artefacts with RO-Crate](https://doi.org/10.3233/DS-210053)
- [ENCORE reproducibility framework](https://doi.org/10.1038/s41467-024-52446-8)
