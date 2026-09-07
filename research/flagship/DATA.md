# Flagship data experiment protocol

Status: planning contract, 2026-09-06. This document defines how the English-first flagship corpus
is selected. [`data_plan.json`](data_plan.json) is the machine-checked run and mixture contract.
Dataset revisions, licenses, filters, hashes, and exact arm weights become immutable experiment
manifests before any result from that experiment is inspected.

The broad source search and pinned candidate revisions are in [`SOURCES.md`](SOURCES.md) and
[`source_registry.json`](source_registry.json). The registry is discovery evidence, not permission to
train; only source-card-qualified entries can enter the experiments below.

[`source_rights_acceptance_template.json`](source_rights_acceptance_template.json) consolidates the
30 selected sources and six evidence packets for a named human authority. Validation checks source
coverage, evidence hashes, intended scope, attribution, redistribution, removal policy, and signature
completeness, but makes no decision. Pending or rejected sources cannot produce the all-approved
record required by production firewall construction; rejected selected sources require a versioned
replacement and requalification.

There is no universally perfect corpus. The target is the best reproducible mixture for this model,
token budget, and capability profile: a Pareto winner that improves the equal-domain objective
without buying its average by silently sacrificing code, math, science, reference, or general
English. A single training-weighted validation loss never selects the corpus.

## 1. Scope and starting prior

Natural language is English-only for the first flagship. Programming-language syntax is not passed
through an English detector, but code comments, documentation, notebooks, and surrounding text are
English-filtered. Multilingual modeling is a later, separately budgeted program.

The stable-phase search has six disjoint reporting categories. The starting prior is a complete
mixture, not the expected winner:

| Category | Prior | Search range | Candidate sources or treatments |
| --- | ---: | ---: | --- |
| Web | 55% | 45–65% | Ultra-FineWeb English v1.4/L1-HQ, DCLM, FineWeb-Edu, blend |
| Code | 15% | 10–22% | restricted Stack v3.1, Stack-Edu, qualified blend |
| Math | 10% | 8–18% | FineMath-4+, MegaMath Web-Pro, qualified blend |
| Synthetic | 10% | 8–18% | Cosmopedia v2, Ultra-FineWeb-L3 Multi-Style, qualified blend |
| Science | 5% | 3–10% | peS2o v3, qualified FinePDFs/Common Pile supplements |
| Reference | 5% | 3–10% | FineWiki English, qualified open books/technical discussion |

Every generated E2 arm sums to 100% and stays inside those bounds. Science and reference use the
already integrated incumbents after fresh qualification; their proportions still vary in E2. Code
is the largest missing implementation dependency. The newly released Stack v3.1 is now the primary
raw-code candidate because it includes contents, preserves repository structure, and fixes v3.0's
reported duplicate leak. It is not approved merely because it is public: exclude no-license and
vendored files, retain per-file provenance and attribution, rescan secrets/PII and contamination,
and qualify a repository-aware adapter. The code-source decision remains a rights and provenance
gate, not just a loss comparison.

The 32K/128K extension corpus is separate from these weights. It uses complete books, papers, and
repository trees rather than unrelated packed fragments. It cannot leak into the base-mixture
selection set.

## 2. Source qualification before GPU experiments

Each candidate must produce a signed-off source card and a small packed sample. Failure on a hard
gate eliminates it without a training run.

1. **Identity and rights:** repository, immutable revision, split, upstream provenance, license or
   terms, attribution obligations, redistribution decision, and contact/removal process.
2. **Content:** schema, document unit, English policy, code-language policy, quality-score meaning,
   safety and PII handling, document-length distribution, and parsing failure rate.
3. **Leakage:** train/evaluation separation, repository-level splitting for code, benchmark
   decontamination, exact deduplication, and a measured near-duplicate policy within and across
   sources. The current pipeline implements global exact deduplication; near-duplicate support is a
   pre-grant deliverable, not an assumed capability.
4. **Operations:** usable-token yield, download and packing throughput, transient and final storage,
   restart behavior, stable shuffle, shard checksum, and projected unique-token capacity.
5. **Tokenizer:** fertility and byte fallback by category on a tokenizer-training-disjoint sample.
   All GPU arms use one frozen tokenizer so tokenization cannot confound the data comparison.

The 20B-token rehearsal exercises the exact winning or fallback implementations, including cleanup
of raw shards. No source can be launch-critical if its full-scale acquisition path or legal status is
still conditional.

[`production_data_plan.json`](production_data_plan.json) freezes the pre-rehearsal operations order.
The fixture-qualified disk-backed preprocessor applies human-reviewed deny entries, global exact and
verified near deduplication, redacted removal records, record-level checkpoints, and post-publication
cleanup receipts before the existing exact-dedup/tokenization packer. It does not replace the 20B
rehearsal: production throughput, memory, storage, unique yield, interruption recovery, and cleanup
must still be measured on the frozen real source path before an operations authority record exists.

## 3. Evaluation firewall

Three disjoint data partitions are frozen before training:

- **Tokenizer sample:** balanced across the six categories and used only to train or evaluate
  tokenizer candidates.
- **Selection held-out:** equal bytes per category, with held-out domains and an unseen source in
  each category where feasible. It is used by E1–E5 but never mixed into training.
- **Sealed audit:** separately hashed, never inspected during selection, and opened once after the E2
  finalists are ranked. Evaluate all three finalists in that one opening; choose the highest-ranked
  candidate that passes the same category guardrail, falling back to the balanced prior if none pass.
  A failed audit does not invite hand-tuning.

All benchmark prompts and reference answers are also decontaminated from training candidates before
packing. Contamination checks and removals are recorded per source and per benchmark.

The first executable firewall contract is
[`web_contamination_v1.json`](web_contamination_v1.json). It freezes 20 short-context, math, and code
payloads (63,652 tasks), independent normalized exact-field matching, task-unique 13-gram critical
matching, and 10-gram sensitivity disclosure. Its bounded web successors are technical evidence, not
training authority; source rights, production deduplication, and cleanup/resume still gate use.

[`firewall_plan.json`](firewall_plan.json) freezes the executable partition and consumer contract
without choosing previously unspecified real held-out/audit byte sizes. The fixture-qualified builder
requires explicit equal-category targets, globally disjoint content hashes, unseen-source bytes,
held-out domains, and distinct `D5_tokenizer`/`E2_mixture` seeds. Generic consumers cannot read sealed
files; an opening claim is durably recorded before all required finalist payloads become readable, so
a failed opening cannot be retried. Fixture outputs cannot authorize real consumers. Production
construction additionally requires hash-bound human rights acceptance covering every source and a
production-operations record passing global exact/near deduplication, cleanup, and resume gates.

The primary reported unit is bits per UTF-8 byte (BPB), which avoids rewarding a mixture merely
because the tokenizer fragments one domain differently. For each category, report paired BPB delta
against the frozen balanced prior, bootstrap confidence intervals over documents, and seed
variation. The primary mixture score is the unweighted macro-average of the six category deltas.
Training-weighted BPB, perplexity, source-level BPB, and small-scale benchmarks are secondary.

An E2 candidate is eligible only when the upper paired 95% confidence bound on regression is no
worse than +0.01 BPB in every category. Among eligible candidates, choose the lowest macro BPB. If
finalists are statistically tied, prefer the mixture with more secure rights and supply, then the
one closest to the prior. This order is frozen before outputs.

## 4. Experiment funnel

| Stage | Purpose | Scale and tokens | New runs | GPU-h |
| --- | --- | --- | ---: | ---: |
| E1W | Four web treatments; one seed each, then two more seeds for the top two | 350M, 8B | 8 | 133 |
| E1S | Code, math, and synthetic source/treatment screens; three arms per category, then one extra seed for each category's top two | 150M, 2B | 15 | 27 |
| E2a | Space-filling stable-mixture search inside the six-category bounds | 60M, 1.2B | 24 | 10 |
| E2b | Retrain the six predicted Pareto candidates at a useful proxy scale | 150M, 3B | 6 | 16 |
| E2c | Confirm the top three mixtures at full proxy scale and three seeds | 350M, 12B | 9 | 225 |
| E3 | One, two, and four effective epochs at matched tokens, two seeds | 150M, 6B | 6 | 32 |
| E4 | Four decay mixtures branched from the stable winner, two seeds | 350M, +3B | 8 | 50 |
| E5 | Within-source quality order and two-phase curriculum, two seeds; reuse the E2c winner as the uniform control | 350M, 12B | 4 | 100 |
| | | | **80** | **593** |

E2a uses a deterministic maximin/space-filling design over the constrained simplex; it is not a
hand-picked set of stories such as “web-heavy.” Fit per-category response surfaces with uncertainty,
publish diagnostics and leave-one-arm-out prediction error, then advance diverse Pareto candidates.
The fitted optimum is a nomination, never evidence by itself. E2b and E2c retrain candidates from
scratch and make the decision.

E3 starts in parallel with E1 on the current high-quality incumbent mixture because its result
determines whether corpus preparation needs roughly 500B unique tokens or can safely use about 150B
with repetition. E4 branches from the same E2c stable checkpoint family. E5 counts only four new
runs because the two-seed uniform control is reused from the selected E2c mixture.

## 5. Promotion and transfer

- E1W promotes two web treatments after three total seeds each. E1S promotes one treatment per
  specialist category after two total seeds for each finalist.
- E2a advances six candidates using the preregistered model plus diversity constraints; E2b advances
  three. E2c freezes one stable mixture using the eligibility and tie-break rules above.
- E3, E4, and E5 use paired two-seed comparisons because their expected effects are larger. A result
  that is indistinguishable from noise retains the simpler incumbent.
- The stable winner receives a 750M transfer check already budgeted in the scale ladder. Code and
  math task conclusions are deferred to that scale and the flagship; chance-level 350M benchmark
  scores cannot veto held-out evidence.
- The flagship reports the same six category losses at the pre-decay and final checkpoints. Any
  failure to transfer is a result in the paper, not grounds to rewrite the experiment history.

## 6. Flexibility without post-hoc search

Candidate source names may change before their source cards are frozen, and exact E2 weights are
generated only after E1. The category definitions, bounds, total run ceilings, held-out partitions,
selection statistic, guardrail, seed counts, and advancement counts are binding.

If a source fails rights, capacity, or operational qualification, replace it with the documented
fallback before its first GPU output. If E2a response surfaces are unstable, promote the best six
observed diverse arms rather than spending more runs. If compute is cut, remove E5 first as specified
in the grant plan; never remove the sealed audit, domain guardrails, or E2c replication.

## 7. Artifacts required for the paper

- Source cards and immutable manifests, including rejected sources and reasons.
- Token counts before and after every filter, deduplication, and decontamination stage.
- Exact arm weights, seeds, tokenizer hash, code revision, packed-shard hashes, and runtime cost.
- Per-source and per-category BPB with document bootstrap intervals and seed dispersion.
- E2 response-surface predictions, diagnostics, advancement record, and all losing arms.
- Stable, decay, repetition, and curriculum decisions plus the sealed-audit result.
- The exact stable and decay manifests used by every released checkpoint.

Primary dataset references: [The Stack v3 dataset card](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train),
[the Common Pile](https://huggingface.co/common-pile), and
[peS2o dataset card](https://huggingface.co/datasets/allenai/peS2o).
