# Data

This document owns the data methods for the 5,000 GPU-hour program: the shared pipeline, where
natural, derived and synthetic records may enter, counting rules and the code route. The
[program overview](program.md) owns stage boundaries and budgets, [PLAN.md](../PLAN.md) the work
order, the [source-readiness matrix](../experiments/main-data/source-readiness.json) per-source
gates, and the [main data plan](../experiments/main-data/README.md#base-mixture) the mixture.
No document here authorizes a training run.

## Pipeline

Pin source revisions and permitted use → acquire/filter → deduplicate and exclude evaluation
material → partition → tokenize/pack → verify → train. Source-separated shards allow later mixture
changes without retokenizing. Recheck tokenizer fingerprints and hashes whenever reusing stock.

Every record follows the same auditable path, at every stage:

1. identify the source, licence or use constraint, family, acquisition receipt and transformation history;
2. extract with a source-specific parser, retaining the original text or artifact identity;
3. remove exact duplicates, near duplicates, template copies and semantic repeats, while preserving
   a family graph for held-out splits;
4. apply quality, language, safety, benchmark-contamination and source-coverage filters;
5. attach lineage, quality dimensions, derivation cost and accepted-token accounting;
6. partition by source family before tokenization and packing;
7. verify manifests, token counts, packing, sampling weights, checksums and resumability;
8. train only from the hash-bound manifest, reporting exposure, replay, discarded data and
   evaluation results separately.

AI-assisted filtering may recommend keep/remove decisions, but pretraining retains the original
source and records the model, prompt, version and decision. Generated text is a derived record and
must never be indistinguishable from source text in a manifest.

**Where derived text may enter.** The broad production pretraining mixture is natural and
source-traceable. Late pretraining decay is the *only* initial pretraining experiment that can admit
derived text, and it must stay separately identified by lineage, teacher/generator, verification
result and cost. Synthetic material is most acceptable in post-training, where it is generated
inside pinned, testable environments or used for self-distillation; it remains separately attributed
and earns its place on held-out transfer, not training reward. Derived data never compensates for
missing natural supply.

**Admission gates.** Before compute is committed, each stage must have a pinned manifest, a
family-disjoint evaluation split, a contamination report, accepted-token or example accounting,
reproducible checksums, measured throughput, a retry/recovery policy and a declared stop rule.

**Counting.** Keep raw stock, accepted unique tokens, exposure/replay and rejected material as
separate counts, and count accepted tokens only after family, overlap and extraction review.
Candidate flags stay review-only until a rule is adopted. Count overlapping banks once. Repetition
is not an approved way to fill a supply gap. Include complete assistant examples in each length band
(<=4K, 4–16K, 16–32K and 32–128K), and keep training stock, supervised tokens and exposure separate.

| Responsibility | Code / command |
| --- | --- |
| Source readers, configuration, packing, resume | `speck/data/{acquisition,configuration,packing,dataset}.py` |
| Global disk-backed deduplication | `scripts.production_data_preprocess` (batched MinHash is the default; `--per-shingle-minhash` is a slower bitwise-identical fallback) |
| Secret filtering, near duplicates, contamination | `scripts.text_gitleaks_filter`, `text_near_duplicates`, `text_contamination` |
| Cross-source family graph and partitions | `scripts.joint_family_graph` |
| Source-use review | `scripts.source_rights_review` (validates a pending human template; never makes an approval decision) |
| Checked retained-stock tokenization | `scripts.tokenize_stock` |
| Distributed loading | `speck/data/loader.py` |

For a complete experiment directory containing `tokenizer.json` and `data.json`:

```bash
uv run --no-sync python -m scripts.tokenizer_prepare PATH_TO_EXPERIMENT
uv run --no-sync python -m scripts.data_prepare PATH_TO_EXPERIMENT
```

These can download substantial data. Use `make smoke` for the offline fixture. The old finite
source-acquisition jobs run only from the [historical checkout](../archive/README.md).

## Retained material

The runtime store is `/mnt/speck-data/speck` on the maintainer's machine; portable code defaults to
`~/.cache/speck`, overridden by `speck_base_dir`. The frozen tokenizer is under
`tokenizer-final-mistral-v1`; its model SHA-256 is
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`.

The [supply gap](../experiments/main-data/supply-gap.json) derives retained stock per bank; the
[pilot supply receipt](../experiments/pilot/supply.json) verifies the retained token stocks, and the
[pilot](../experiments/pilot/README.md) records its finite packed corpus. Retained stock is not
eligible supply: reopen manifests, preserve source-use conditions and verify cross-source and
validation separation before reuse.

At uint16 storage, each billion packed tokens requires about 2 GB for token IDs, before indexes,
validation, preparation intermediates and duplicate databases. Budget raw acquisition, exclusion
outputs, packed data and recovery checkpoints separately. Choose the main horizon from measured
all-in throughput and eligible supply, then run the same joint-exclusion, partition, pack and
full-loader checks at that scale.

## Recipe direction — 2026-09-19

The flagship target is always-thinking agentic coding, general coding, math reasoning and tool
use. High-quality code and reasoning must be present during pretraining, not deferred entirely to
SFT. Prioritize these forms in the main corpus:

- Natural implementation code with useful tests, documentation, API usage and project context;
  preserve Python and other target languages, including JavaScript/TypeScript.
- Correct worked math, derivations and scientific explanations spanning elementary foundations
  through harder problems; inspect intermediate reasoning as well as final answers.
- Checked code explanations, algorithm derivations, debugging/repair examples and exercises with
  independent tests. Preserve mistakes only when clearly identified and followed by valid correction.
- Coherent repository/document bundles for context extension. Complete observed tool trajectories
  belong in the separately serialized reasoning/agent training stage; planning prose alone does
  not establish an agent's ability to inspect, edit, test and recover.

Selected natural web and reference material provide language, knowledge and task diversity.
Preserve everyday, nontechnical topics and varied prose as well as difficult educational material.
Do not equate educational score, reasoning length or a passing generated test with quality. Retain
source-family identity across original pages, rewrites, Q&A and instruction derivatives; several
dataset names can represent the same underlying information. Pilot shares are engineering
settings, not quotas.

Web and math review rules carried from the audits in the
[corpus-audit record](../experiments/corpus-audit/README.md):

- Score labels are not interchangeable cutoffs. Name the exact field and operator for any stricter
  predicate; configuration labels (FineMath 4+, InfiWebMath 3+) are not equivalent continuous-score
  thresholds.
- High-scoring pages can still carry extraction defects; review flags stay separate from the
  production filter until validated against archived captures.
- Arithmetic or answer-consistency parsers are triage, not correctness certificates; symbolic,
  unit-bearing and multi-line reasoning need independent checks.
- Exact normalized benchmark matches are a floor, not a contamination clearance; derived variants
  and semantic overlap remain separate checks.

The [source registry](../experiments/main-data/source-registry.json) fixes the selected sources;
[recipe-review.json](../experiments/corpus-audit/recipe-review.json) keeps the reviewed public
cards for later revised freezes. The assistant stock and its gaps are in the
[assistant recipe](assistant.md#main-assistant-data-direction--2026-09-19).

Design records for this evidence:

- The [data-design contract](../experiments/main-data/data-design-contract.json) fixes the
  manifest fields every stage records: stage, source family, transformation, quality, coverage,
  dependency, contamination and lineage.
- The [source-mapping receipt](../experiments/main-data/frontier-data-source-mapping.json) indexes
  which audits support each reviewed hypothesis; it closes no gate.
- The [natural-web](../experiments/main-data/natural-web-candidate-manifest.json),
  [natural-code](../experiments/main-data/natural-code-candidate-manifest.json),
  [math](../experiments/main-data/math-candidate-manifest.json) and
  [post-training](../experiments/main-data/post-training-candidate-manifest.json) candidate
  manifests add domain evidence and comparison contracts for the readiness sources.
- The [post-training research synthesis](../experiments/main-data/post-training-research.json)
  keeps general SFT, reasoning SFT, agent trajectories, RL prompts, preference/critique data and
  on-policy teacher feedback as separate banks. Correctness remains primary in RL; any efficiency
  preference is delayed, soft and task-conditioned.

Preserve source-family identity and exclusions across every stage, including derived exercises,
teacher traces and RL prompts. Long-context qualification must measure retrieval across positions,
cross-document reasoning, sustained generation and short-task retention, alongside memory/runtime;
a configured maximum alone does not establish usable context.

## Code priority and qualification

Code quality is a pretraining requirement, not only a post-training concern: natural code, tests,
documentation and correct worked explanations should establish useful foundations before
reasoning-SFT. Preserve practical API use, debugging and repository relationships alongside
algorithmic exercises. Long-context preparation retains coherent repository units and dependencies
for 16K/32K qualification; an arbitrary concatenation of unrelated files is not repository-level
supervision. Split original repositories and derived tasks together to protect held-out repair and
agent evaluations.

**Qualification rules.** Natural source code needs immutable identity, applicable source-use
evidence, intact useful content, family/benchmark exclusions and joint deduplication. It does *not*
need to pass invented tests to be natural-code material. Verified exercises additionally require
clear specifications, same-revision implementation/test linkage, bound dependencies, independent
oracles and deliberately wrong controls; tests generated alongside a solution establish
self-consistency only. Keep natural-code and verified-exercise outcomes separate throughout
preparation and experiments. Preserve original bytes and notices, upstream content IDs,
repository/commit/file identities and consumed-text hashes. Encoding changes, redaction and notebook
cleanup require explicit provenance; never disable exact-byte checks to accommodate an unexplained
mismatch. Related originals, forks, rewrites, patches and exercises belong to the same
exclusion/partition family. A clean bounded screen or a publisher licence label alone authorizes
nothing; notice text governs, not a scanner label.

**Supply position.** Natural code is the bank where the whole horizon binds; the
[supply gap](../experiments/main-data/supply-gap.json) owns the figures. The
[qualification packet](../experiments/main-data/QUALIFICATION.md) owns cohort holds and origin/notice
recovery.

**First comparison to prepare.** The
[research design](../experiments/main-data/README.md#research-before-the-main-run) owns the
screening/confirmation matrix. The code candidate is the natural-source contrast (Stack-Edu versus
Stack v3). Keep total code share, non-intervened language coverage, non-code banks and
serialization fixed. Checked code is not a declared bank, so do not add it back as an extra arm or
fill a code quota by repeating a tiny bank.

**Evidence for a coding claim.** Keep the pilot's compiled HumanEval+ metric as a continuity check;
33 development tasks cannot establish broad coding strength. Prepare a separate pinned protocol
before admitting new data:

| Dimension | Next evidence |
| --- | --- |
| Python generation | HumanEval+ continuity plus MBPP+ through a declared [EvalPlus](https://github.com/evalplus/evalplus) protocol |
| Language coverage | Selected [MultiPL-E](https://github.com/nuprl/MultiPL-E) languages, reporting each separately |
| Harder generation and repair | A fixed release/date window of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), with explicit test variant and output limits |
| Practical use | Bounded held-out tasks for library use, debugging and repair, scored against independent hidden tests |
| General usefulness | Existing math, instruction following, knowledge/reasoning checks, and per-source held-out loss |

These are planned, not implemented or scored. Freeze task-family partitions and exclusion identities
before data selection, fit prompts and output within the context budget, pin Python/library
versions, and re-run reference models under the same protocol. Report denominators, uncertainty,
execution failures, output tokens and runtime; do not tune against the final partition or equate our
compiled metric with the official leaderboard.

**Repository change data.** Qualify 8–16 repository repair cases before any bulk route. Retain
origin, licence evidence, immutable parent/fix commits, issue specification, changed files, patch
and environment/dependency identities. Exclude benchmark/task families before acquisition,
deduplicate commits that also occur in pull requests, and group related files by repository for
partitioning. Prevent post-fix state or hidden-test answers from leaking into task inputs. Run
checks only in the existing sandbox: at least one relevant test must fail before the fix and pass
after it, while declared regression tests keep passing. Verify empty-patch failure and
reference-patch success repeatedly; reject flaky or underspecified cases, and record
environment/setup failures separately from incorrect solutions. Before training on change examples,
specify how pre-change context is loss-masked and patches are supervised — the current plain
pretraining pack does not implement that objective.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts.
