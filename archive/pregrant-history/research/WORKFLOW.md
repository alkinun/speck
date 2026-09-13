# Research workflow

This is the default lifecycle for consequential Speck research. The goal is not paperwork for its own
sake. The goal is to make it possible, months later, to determine what was asked, what ran, what failed,
what the evidence supports, and exactly which public sentence came from it.

## 1. Ask one decision-relevant question

Start with a dated notebook entry and a Linear issue. State:

- the uncertainty;
- why resolving it changes the model, data, evaluation, systems claim, or release;
- existing Speck evidence and relevant external literature;
- what happens if the result is null, adverse, or operationally impossible.

Exploration may begin from a broad question. Confirmatory training may not.

## 2. Separate literature from local evidence

External paper notes belong under `papers/` and record the exact version, authors' claim, quantitative
evidence, systems context, limitations, code/data availability, and Speck interpretation. Language such
as “the paper reports” remains distinct from “Speck reproduces.”

Local conclusions belong under `findings/` only after checked evidence exists. A finding cites the
exact experiment and result rather than relying on a paper note, W&B URL, or recollection.

## 3. Preregister consequential comparisons

Before output, freeze:

- hypothesis and estimand;
- treatment, competent control, and shared parent;
- parameters, tokens, data order, seeds, and hardware class;
- primary and secondary metrics;
- uncertainty calculation and multiplicity handling;
- source/task guardrails and failure criteria;
- stopping rule, run count, GPU-hour ceiling, and reserve boundary;
- promotion, fallback, and “do not interpret” rules;
- expected output artifacts and paper claim IDs.

Changing any of these after output requires a named successor. Preserve the predecessor and explain why
the change was necessary. Exploratory output may motivate a successor but cannot acquire retroactive
confirmatory authority.

## 4. Qualify implementation separately

Correctness is not quality, and a fixture is not production evidence. Before training:

1. establish a readable reference implementation;
2. compare optimized/reference forward, backward, state, and cache behavior;
3. verify parameter, FLOP, state, and optimizer accounting;
4. inject failures into checkpoint, resume, and publication boundaries;
5. run the exact-shape hardware preflight;
6. label every result as fixture, local diagnostic, hardware qualification, or scientific run.

An implementation preflight can authorize execution. It cannot promote a scientific hypothesis.

## 5. Execute from one immutable run manifest

Every scientific run must identify:

- Git commit and clean tracked tree;
- experiment, model, tokenizer, data, and analysis-contract digests;
- source revisions and packed-shard identities;
- seed and actual data order;
- hardware, driver, CUDA, libraries, precision, compile settings, and environment lock;
- requested and observed resources;
- checkpoint parent and exact resume lineage;
- W&B run only as a secondary pointer;
- raw artifact locations, byte counts, and digests.

Raw corpora, checkpoints, full predictions, traces, and logs live in the runtime artifact store. Git
stores their manifests and the bounded records needed to audit conclusions.

### Monitoring and peeking

During confirmatory runs, operators may inspect mechanical health: process state, finite loss,
throughput, memory, temperature, checkpoint publication, and scheduler behavior. They must not add or
remove arms, replace an inconvenient seed, extend only a favorable run, change the primary endpoint, or
stop for scientific success unless that rule was preregistered. Scientific comparison begins only when
the declared analysis set is complete or a frozen missingness/failure rule applies.

Mechanical retries reuse the identical manifest and are recorded. Application, numerical, data, and
scientific failures remain outcomes; relabeling one as a mechanical retry requires a predeclared
classification rule. Reserve use cannot erase a failed mandatory run from the analysis.

## 6. Analyze without changing the experiment

Analysis code and decision rules are frozen before reading confirmatory output. Reports retain:

- every arm and seed, including failures;
- point estimates and declared intervals/bounds;
- per-source, per-category, and per-task views;
- fixed-token, fixed-FLOP, and fixed-wall-clock views where applicable;
- missingness, censoring, numerical errors, and hardware anomalies;
- raw artifact and analysis-code identities.

Do not pool away a failed seed or guardrail. An interval spanning zero is unresolved, not proof of no
effect. A benchmark aggregate cannot replace task-level output.

## 7. Promote analysis to a finding

A finding is a concise durable interpretation, not a chronological diary. It contains:

1. question and frozen design;
2. result with uncertainty and costs;
3. failed and passed gates;
4. supported claim and explicit non-claims;
5. limitations and transfer boundary;
6. model/data/program decision;
7. exact contracts, results, run IDs, and hashes.

Corrections create successor findings and result records. Do not rewrite an old record to look as if it
was produced under new code or a new interpretation.

## 8. Update the claim registry

`paper/claims.json` contains the paper-level claims. A claim moves through:

```text
planned → prior_evidence_only → partially_supported → supported
                                           └──────→ refuted
```

`retired` is for a claim removed before decisive output or made irrelevant by scope. Supported claims
must cite the result and finding that support the exact wording. Refuted or unresolved claims remain in
the registry and become negative results or limitations when scientifically relevant.

## 9. Generate the paper from evidence

Figures and tables are outputs, not notebooks. Their scripts read checked result records and emit files
under `paper/figures/` and `paper/tables/`. Manuscript prose references claim IDs. Important values are
not copied manually from W&B, terminal output, or Linear.

Before release, a paper build must verify:

- every claim ID resolves;
- every public number resolves to a checked artifact;
- every figure/table records its generating command and source hashes;
- failed gates are not represented as passes;
- comparator versions and token budgets are explicit;
- model/data licenses, attribution, intended use, and limitations are complete.

An independent reviewer who did not author the final analysis should then start from a clean checkout,
resolve the public or authorized inputs, regenerate the paper tables and figures, sample-check raw
records against summaries, and challenge each supported claim against its non-claims and failed gates.
Reviewer findings and dispositions are retained with the release package.

## 10. Archive a research object

The public release should contain or point to the manuscript, code snapshot, environment, contracts,
source/data metadata, analysis scripts, small results, figure/table sources, model cards, and immutable
large artifacts. Use a DOI-minting archive for the compendium snapshot and Hugging Face for model
artifacts. Restricted bytes may remain closed, but their metadata, provenance, access conditions, and
reason for restriction remain public where permitted.

## Allocation operating cadence

### Before each wave

- confirm the question, claim IDs, dependencies, exact manifests, analysis, budget charge, expected
  outputs, retention, and fallback;
- pass clean-tree, data-authority, hardware, storage, and scheduler preflight;
- write the notebook handoff and require a second review for flagship-scale or sealed-audit work.

### During each wave

- monitor only under the peeking policy above;
- preserve scheduler/accounting events and failed attempts;
- checkpoint and requeue only at qualified boundaries;
- never make a promotion decision while sibling arms are incomplete.

### At wave close

- collect actual GPU-hours, energy, failures, and artifact identities;
- run the frozen analysis once over the declared set;
- write or update the finding and claim registry;
- reconcile Linear to the checked evidence rather than the reverse;
- verify required backups before deleting or starting dependent work.

### Weekly during the allocation

- run `make quality` from a clean checkout;
- audit mandatory and reserve GPU-hours against Slurm accounting;
- review disk headroom, R3/R4 backup verification, and checkpoint restoreability;
- list incomplete result/finding/claim transitions and assign owners;
- regenerate the paper evidence map and remove unsupported wording, not inconvenient evidence;
- record scope amendments as successors before the next affected output.

## Status tools

Validate the central catalog, notebook, and paper claims with:

```bash
uv run --extra cpu python -m scripts.research_catalog
```

This complements—not replaces—the program-specific validators, test suite, Slurm preflight, and data
launch receipt.
