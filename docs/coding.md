# Coding priority

Decision updated 2026-09-19. Agentic coding, general coding and math reasoning are primary
first-release targets for the always-thinking assistant. The paper studies their data and training
across pretraining, mid-training and post-training after a bounded architecture study and backbone freeze. General usefulness supports these targets.
Code-data selection is a candidate contrast for the pretraining recipe study, which must precede
main pretraining. Freeze its question after source/supply audits; the frozen engineering pilot and
its evaluation protocol remain intact.

Code quality is a pretraining requirement as well as a post-training concern. Natural code,
tests, documentation and correct worked explanations should establish useful foundations before
reasoning-SFT. Preserve practical API use, debugging and repository relationships alongside
algorithmic exercises. The pilot's code weight is an engineering setting, not the final target.
Long-context preparation should retain coherent repository units and dependencies for 16K/32K
qualification, with 128K as a stretch;
an arbitrary concatenation of unrelated files is not repository-level supervision. Split original
repositories and derived tasks together to protect held-out repair and agent evaluations.

## What we are taking from OpenBMB

OpenBMB is a primary research reference for data selection, synthesis, and small-model development.
Its [UltraData-Code release](https://huggingface.co/datasets/openbmb/UltraData-Code) reports matched
10B-token continuation experiments on a 1B model: multilingual L2 improves EvalPlus by 4.37 points
over Stack-Edu; replacing half the L2 tokens with L3 adds 8.42 points over L2 alone. These are
publisher results, not Speck measurements. They motivate testing selected natural code plus
task-oriented examples; they do not establish the best source or ratio for our model.

The separate [Ultra-FineWeb-L3 inspection](../experiments/corpus-audit/README.md#follow-up-decisions--2026-09-18)
does not establish superiority to Cosmopedia. Keep it as a candidate and test its generated answers
against their source passages. General synthetic prose and executable code need different checks.

## Qualification rules

Natural source code needs immutable identity, applicable source-use evidence, intact useful content,
family/benchmark exclusions and joint deduplication. It does not need to pass invented tests to be
natural-code material. Verified exercises additionally require clear specifications, same-revision
implementation/test linkage, bound dependencies, independent oracles and deliberately wrong controls.
Tests generated alongside a solution establish self-consistency only. Keep natural-code and
verified-exercise outcomes separate throughout preparation and experiments.

Preserve original bytes and notices, upstream content IDs, repository/commit/file identities and
consumed-text hashes. Encoding changes, redaction and notebook cleanup require explicit provenance;
never disable exact-byte checks to accommodate an unexplained mismatch. Related originals, forks,
rewrites, patches and exercises belong to the same exclusion/partition family. A clean bounded
screen or a publisher license label alone does not authorize training.

## Completed code qualification

These bounded checks establish methods and known limitations. Detailed artifacts and historical
results remain in the linked receipts; none admits a new corpus or establishes full-release yield.

| Evidence | Finding and practical limit |
| --- | --- |
| [UltraData-Code preview](../experiments/corpus-audit/code-preview.json), [execution](../experiments/corpus-audit/code-execution.json) | Sixteen L3 solutions parse; seven pass supplied tests, nine fail. Passing self-generated tests is not independent correctness. One contiguous window cannot estimate release-wide rates. |
| [L3 provenance](../experiments/corpus-audit/code-provenance.json) | All sixteen candidates remain held: released UUIDs do not establish the missing L2/original-source mapping, commit and source-use evidence. Preserve duplicate-source families. |
| [Natural-code cohort](../experiments/corpus-audit/natural-code-cohort.json), [static review](../experiments/corpus-audit/natural-code-review.json) | Sixteen immutable byte matches, thirteen ancestor-notice matches, one conservative benchmark flag. Matching revisions need not be the original crawl revisions; all families remain outside training. |
| [Practical CPU checks](../experiments/corpus-audit/practical-code-checks.json) | Python and JS/TS checks distinguish ordinary behavior, demonstrated bugs and additional robustness policies. The 13,236-file coverage sample precedes the full census below. Test adapters are not complete upstream builds. |
| [Bundle linkage](../experiments/corpus-audit/code-bundles.json) | Nineteen linked files at four revisions; family holds affect fifteen. Co-present test filenames and weak assertions do not establish independently verified exercises. |
| [Expansion inventory](../experiments/corpus-audit/code-expansion.json) | Upstream inventories pinned; thirteen length-matched files / 9,541 tokens in the bounded new-shard probe. This does not estimate scalable eligible yield. |
| [Application origins](../experiments/corpus-audit/code-application-origins.json) | Four revisions, complete trees and MIT notices recovered; the one direct test link is stale. No independently verified example follows. |

UltraData-Code L2 remains a candidate only with resolved source lineage; L3 remains held. Retain
exact source terms and restrictions separately from dataset-level declarations. Reopen lineage
work when an authoritative mapping or independently verified origin is available, rather than
acquiring a large corpus to guess an undocumented join.

## Retained supply census — 2026-09-19

The [complete retained-stock census](../experiments/corpus-audit/code-supply.json) reopens both
finite Stack-Edu acquisition batches: **1,999 archives / 12,383,211,520 archive bytes**. All archive
hashes verify; every retained file was retokenized with the frozen Mistral tokenizer, including
BOS/EOS. Document, byte and token totals reconcile with every original unit. This took 140 seconds
with four local CPU workers and executed no corpus code. The earlier 13,236-file coverage sample
covered only the first batch; these are complete finite-stock counts, not full-release estimates.

| Language | Retained files | Tokens before joint eligibility |
| --- | ---: | ---: |
| C | 37,344 | 21,493,845 |
| C++ | 113,007 | 77,198,333 |
| Go | 33,145 | 20,663,202 |
| Java | 80,057 | 46,994,537 |
| JavaScript | 117,696 | 69,119,927 |
| Markdown | 17,791 | 17,964,030 |
| Python | 209,564 | 123,490,220 |
| Rust | 32,949 | 44,985,199 |
| SQL | 14,766 | 11,997,503 |
| Shell | 14,710 | 11,986,906 |
| TypeScript | 43,340 | 30,881,145 |
| **Total** | **714,369** | **476,774,847** |

Every record has a named repository/path, upstream license labels and a matching SHA1 content ID.
Every `commit_id` field is empty. Content identity does not establish a repository version or
source-use acceptance. There are 259,257 named repositories, no repeated source rows and no exact
text duplicates within this stock; fork relationships and near-duplicates remain unresolved.
Seventeen texts match the previously reviewed 37-file cohort, whose evidence remains tied to its
recorded origins and versions. No new training admission follows from that overlap.

| Mutually exclusive path hint | Files | Tokens |
| --- | ---: | ---: |
| Test | 18,197 | 14,597,279 |
| Documentation/example | 27,899 | 23,988,599 |
| Vendor | 175 | 210,470 |
| Other/unclassified | 668,098 | 437,978,499 |

The script fixes priority as vendor, test, documentation/example, then other. These are filename
hints, not semantic role labels: `other` can include algorithm exercises, and test naming conventions
are incompletely covered. Generated-header wording appears in 2,836 files; it does not establish
LLM generation. Under the same repository name, 5,127 repositories have test and other files,
3,507 have documentation/example and other files, and **326 have all three**, totaling 6,019,590
tokens. These are candidates for linkage work, not complete or runnable same-commit bundles.
Only four files exceed 32K tokens; this stock alone does not establish long-repository supply.

All acquisition units bind the original 20-lane content screen. The newer 22-lane screen was run
on the separate 37-file cohort, not this whole stock. The twelve benchmark repository names match
84 stock files / 174,977 tokens. Adding previously recorded content-held components and aliases
raises the known hold to **123 files / 209,751 tokens**; six previously content-flagged texts are
present. This propagates existing evidence, without claiming a new complete content scan. Removing
only these files would not establish a qualified remainder or complete family separation.

At the working 100B base horizon, natural code needs 30B exposure / 37.5B eligible unique
preparation, separately from 5B checked-code exposure. The retained stock covers 1.59% of natural
code exposure before joint eligibility. Repetition is not an approved way to fill the gap.
The original census retains its historical planning denominator; current targets live in
[the numeric plan](../experiments/main-data/plan.json).

## Stratified retained-code audit — 2026-09-19

The [frozen protocol](../experiments/corpus-audit/code-yield-plan.json) samples the complete finite
census by language, existing path role and <=4K/>4K tokens. Two lowest hash-ranked records per
nonempty group (or the sole member) produce **138 files / 459,615 tokens**, across **11 languages
and 72 strata**. Known holds and previously reviewed files stay in the frame; no failed selection
is replaced. Preserve each group's population and inclusion probability. Neither equal per-group
counts nor this retained acquisition are a representative sample of the entire upstream release.

The [first-pass result](../experiments/corpus-audit/code-yield-result.json) reopens 129 archives,
verifies full selected bytes and Mistral token counts, and applies the existing 22-lane exclusion
screen. **30 files trigger conservative content matches; 31 are held after named-family propagation.**
The other 107 remain unresolved. Combining prior and new named-family evidence holds 5,325 files /
3,858,264 tokens in retained stock; this is propagation of known flags, not a full-stock content
scan or proof of semantic leakage. Exact sample, source and screen replay succeeds; a changed
source hash is rejected. The first pass took 43.25 seconds locally, replay 35.79; no GPU or network
was used and no corpus code executed.

An exploratory assistant reading of all eight short `other` files in Python/JavaScript/TypeScript/Go
finds three algorithm exercises, a language tutorial, a browser tutorial, a scientific script, a
library implementation and an application module. It also records possible defects and missing
context. This is static reading, not independent annotation or a practical-role/correctness rate.

**Next:** complete source/origin/use and semantic-quality assessment on this frozen sample, review
conservative matches and record family links and cost. Missing tests do not alone reject ordinary
natural code. Verified exercises additionally require specifications, dependency/test linkage,
independent oracles and deliberately wrong controls. Keep unassessed/unresolved/held/passed gates
separate. Only completed assessments can support weighted natural-code yield or checked-exercise
feasibility estimates; no training data is admitted by this first pass.

## Stack v3 feasibility — 2026-09-19

The [Marin review](research.md#marin-corpus-review--2026-09-19) led to a bounded
[revision/schema probe](../experiments/corpus-audit/stack-v3-feasibility.json). Use corrected
revision `8f3f25d86e44fd691428131efd17af75d4716499` for further qualification: its inventory has
8,192 shards / 3.546 TB compressed and differs from Marin's older pin. Publisher sizes and token
estimates do not establish eligible Mistral supply.

The initial two-group probe inspected 2,699 repository rows / 47,826 files and exposed repository
concentration and transformed identities. The [origin follow-up](../experiments/corpus-audit/stack-v3-origins.json)
linked twelve originals to pinned Git blobs, characterizing unchanged, decoded, redacted and
notebook-transformed text. That diagnostic explains those selected cases only; notice applicability
and intended-use decisions remain pending. Labels are screening metadata, not source-use approval.

The [broader protocol](../experiments/corpus-audit/stack-v3-sampling.json) is now
[acquired and screened](../experiments/corpus-audit/stack-v3-broader.json):

| Measure | Observed result |
| --- | --- |
| Acquisition | All 16 frozen row groups, one per shard; four shards per size band; 510.02 MiB, no failed groups or retries |
| Reconciled frame | 29,347 repository rows / 379,942 file entries; no duplicate repository/commit keys or declared-count mismatches |
| Fixed review cohort | 44 repositories / 80 files across 11 nonempty strata and 24 language labels |
| Token/identity preflight | 133,211 Mistral tokens with BOS/EOS; three files exceed 4K; ten supplied-text/content-ID mismatches, with follow-up below |
| Exclusion screen | 22 benchmark lanes; four content-flagged files; eight files held through known repository/fork families |
| Reproducibility | Exact network-disabled replay; a deliberately corrupted cached range is rejected |

The 48-repository / 96-file review caps were maxima: one mixed-license stratum is structurally
empty and single-file repositories contribute one file. Original denominators and unequal selection
factors are retained. All 80 files fit the review byte cap. The remaining 72 files are unresolved,
not eligible. Neither cohort fractions nor the frame's 369,422 `no_license` / 10,520 `permissive`
labels establish population yield. Whole-shard publisher hashes were not verified; pinned range
boundaries and local byte hashes were checked. No upstream or corpus code ran.

Keep upstream content IDs, original Git/blob identities, decoded-text identities and consumed-text
SHA256 separate. Do not explain arbitrary mismatches as redaction or equate transformation
equivalence with useful semantics. Artifacts, selection, acquisition/replay driver and hashes remain
under `/mnt/speck-data/speck/data-qualification-20260919/stack-v3-broader`; the receipt binds their
manifest. Earlier probe/origin artifacts and the frozen retained-Stack-Edu audit remain unchanged.

## Common cohort review

The [review receipt](../experiments/corpus-audit/code-cohort-review.json) binds all 218 original
records and their sampling factors. A frozen first batch selects two currently unheld, <=4K files
per cohort in Java, JavaScript, Markdown, Python, Rust and Shell: **24 full texts / 18,609 tokens**.
This length/language/hold-restricted assistant review is not independent annotation or a yield sample.

| Follow-up | Stack-Edu | Stack v3 |
| --- | ---: | ---: |
| Full supplied texts reviewed | 12 | 12 |
| Originals requested, including the separate transformation check | 12 | 18 |
| Originals verified against pinned Git blobs | 11 | 17 |
| Files with verified ancestor-notice evidence | 10 | 11 |

One Stack-Edu repository lookup and one pinned Stack v3 file returned 404; neither was replaced.
A matching Stack-Edu revision need not be its original crawl revision. Notice presence is not an
intended-use decision. Original/consumed identities, notices, failures and all review observations
remain external, bound by the receipt; cached replay is byte-identical and copied-cache corruption
is rejected.

Nine of the ten Stack v3 mismatches are explained by placeholder substitutions. In
`opipoy/file-locker: locker.py`, the original parses under CPython 3.14.6 but the supplied text
fails at line 49 because redaction changed executable expressions. Keep that representation on hold;
do not silently restore redacted originals. Explanation of a transformation does not establish
semantic preservation. The tenth original remains unavailable.

Both batches contain practical modules, tests and documentation needing repository or environment
context. Static observations also include instructional/edge-case defects, incomplete test coverage,
a proprietary/confidential-looking template header requiring notice review, a link directory and a
project stub. These are individual findings, not comparable corpus defect rates or a source ranking.
No code ran, verified exercise was established or training data admitted.

**Next:** review remaining unheld records without replacing held selections, resolve notices and family
exclusions, and validate a transformation-aware content rule on fixed controls before bulk packing.
Keep natural-code eligibility separate from independent exercise verification. The retained 0.477B
code stock supports at most **1.59B total one-pass tokens at 30% natural code**, before exclusions,
validation and other bank constraints. Qualified supply must set the experiment horizon alongside
runtime cost; longer confirmation runs need more eligible baseline data.

## First comparison to prepare

The [research design](../experiments/main-data/README.md#research-before-the-main-run) owns the
screening/confirmation matrix, shared controls, cost caps and decision rules before main pretraining.
On the frozen backbone, screening permits one code-bank candidate alongside one natural-web candidate,
each against the same qualified baseline. Confirm one selected intervention on fresh paired seeds.

Use the fixed Stack-Edu and Stack v3 audits to choose a feasible natural-source contrast.
Natural code plus independently checked exercises remains an alternative if qualified supply
supports it. Keep total code share, non-intervened language coverage, non-code banks and serialization
fixed. A declared Python substitution can change its selected source without silently changing
the whole mixture. Do not add an extra checked-code arm or fill its quota by repeating a tiny bank.

The contrast tests its declared source/selection recipe. Checked-code substitution does not isolate
synthesis, selection and verification separately; a source swap does not test repository packing.
Freeze the chosen contrast and eligibility before training, preserve inconclusive results, and keep
final benchmarks untouched. Mid-training and SFT questions use useful parent checkpoints under their
separate caps; they do not replace the from-scratch pretraining comparison.

## Evidence for a coding claim

Keep the pilot's compiled HumanEval+ metric as a continuity check; 33 development tasks cannot
establish broad coding strength. Prepare a separate, pinned protocol before admitting new data:

| Dimension | Next evidence |
| --- | --- |
| Python generation | HumanEval+ continuity plus MBPP+ through a declared [EvalPlus](https://github.com/evalplus/evalplus) protocol |
| Language coverage | Selected [MultiPL-E](https://github.com/nuprl/MultiPL-E) languages, reporting each separately |
| Harder generation and repair | A fixed release/date window of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), with explicit test variant and output limits |
| Practical use | Bounded held-out tasks for library use, debugging, and repair, scored against independent hidden tests |
| General usefulness | Existing math, instruction following, knowledge/reasoning checks, and per-source held-out loss |

These additions are planned, not implemented or scored. Freeze task-family partitions and exclusion
identities before data selection. Fit prompts and generated output within our context budget,
report any eligibility exclusions, and pin Python/library versions. Re-run reference models under
the same protocol. Report denominators, uncertainty, execution failures, output tokens, and runtime;
do not tune against the final partition or equate our compiled metric with the official leaderboard.

## Research review cadence

At each data-recipe freeze, review the [MiniCPM releases](https://github.com/OpenBMB/MiniCPM),
[UltraData framework](https://arxiv.org/abs/2602.09003), source cards, and released classifiers.
Record exact revisions, matched ablations, verification methods, token definitions, source coverage,
and processing cost; check errata and negative results. Review relevant new releases when surfaced
during project work. This is a project review practice, not an automatic background monitor.
Keep independent baselines from other teams and promote a recipe only through our own evidence.

## Repository change data

The [MAI review](research.md#mai-thinking-1-review--2026-09-19) motivates a bounded follow-up to
our existing 16-file provenance cohort: qualify 8–16 repository repair cases before any bulk route.
This is a planned cohort, not admitted training data or a new running job.

Retain origin, license evidence, immutable parent/fix commits, issue specification, changed files,
patch and environment/dependency identities. Exclude benchmark/task families before acquisition.
Deduplicate commits that also occur in pull requests and group related files/changes by repository
for partitioning. Prevent post-fix state or hidden-test answers from leaking into task inputs.

Run checks only in the existing sandbox: at least one relevant test must fail before the fix and
pass after it, while declared regression tests remain passing. Verify empty-patch failure and
reference-patch success repeatedly; reject flaky or underspecified cases. Record environment/setup
failures separately from incorrect solutions. Natural code remains distinct from verified exercises.

Before training on change examples, specify how pre-change context is loss-masked and patches or
other targets are supervised. Audit processed-context and supervised-token totals independently;
the current plain pretraining pack does not automatically implement that objective. Reuse compatible
masking infrastructure only after adapter validation. Freeze any bank-share or objective changes
explicitly, within the 100B working base horizon and its measured-cost gate.
