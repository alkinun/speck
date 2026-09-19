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

The fixed cohorts retain **218 original records**, their sampling factors and observed tokens.
Common full-text review now covers **176 files / 412,974 tokens**. There are **44 family-held records**
and **no currently unheld records outside full-read coverage**. Review completion does not establish
source use, correctness, eligible yield or a source ranking; the two sampling frames remain distinct.
All 174 currently unheld records are read. Two previously read records are now held, so reading
and hold counts overlap.

| Evidence | Completed scope | Practical implication |
| --- | --- | --- |
| [Initial common review](../experiments/corpus-audit/code-cohort-review.json) | 24 texts / 18,609 tokens across six shared languages; 28 of 30 requested originals Git-verified | Length/language-restricted reading; origin failures remain in denominators |
| [Python follow-up](../experiments/corpus-audit/python-cohort-validation.json) | 12 more texts / 20,218 tokens; all 16 Python files unheld at selection time read | Redaction checks need original/consumed pairs and contextual review |
| [Family/provenance follow-up](../experiments/corpus-audit/code-family-provenance.json) | Known links assessed across all 218 records; three dependency edges | Established 42 holds before the later metadata-only hold |
| [Go/Rust follow-up](../experiments/corpus-audit/go-rust-cohort-review.json) | 13 more texts / 28,634 tokens; all eight unheld Go and nine unheld Rust files now read | Static concerns and useful coverage recorded; origins/notices remain unresolved |
| [JS/TSX/Vue follow-up](../experiments/corpus-audit/javascript-cohort-review.json) | 14 more texts / 29,940 tokens; all unheld records carrying these labels read | One metadata-only hold added; language/dialect observations remain separate from source labels |
| [TypeScript/C#/Kotlin follow-up](../experiments/corpus-audit/typed-language-cohort-review.json) | 14 more texts / 25,281 tokens; all unheld records carrying these labels read | Declarations, translated code, tests and teaching scaffolds distinguished; no new origins or holds |
| [Shell/build/config follow-up](../experiments/corpus-audit/build-config-cohort-review.json) | 24 more texts / 36,832 tokens; all unheld Shell/CMake/Git Config/JSON/YAML/Dockerfile/Makefile records read | 12 scripts, eight authored configs, two generated build files and two localization files; no new origins or holds |
| [C/C++ follow-up](../experiments/corpus-audit/c-cpp-cohort-review.json) | 22 more texts / 54,222 tokens; all nine unheld C and thirteen unheld C++ records read | Test-harness limitations and template/SDK context recorded; one metadata-only hold added |
| [Java follow-up](../experiments/corpus-audit/java-cohort-review.json) | 15 more texts / 27,176 tokens; all nineteen unheld Java records read | Source-notice applicability and caller/test context recorded; one prior verified original reused, no new holds |
| [SQL-labelled follow-up](../experiments/corpus-audit/sql-cohort-review.json) | 11 more texts / 37,430 tokens; all unheld records carrying the SQL label read | Schemas, notebooks, dumps and test roles distinguished; one CQL fixture; no new origins or holds |
| [MATLAB/Objective-C/PHP follow-up](../experiments/corpus-audit/application-cohort-review.json) | Six more texts / 5,546 tokens; all unheld records carrying these labels read | Signal-validation assumptions, framework/configuration dependencies and notice context recorded; no new origins or holds |
| [Markdown follow-up](../experiments/corpus-audit/markdown-cohort-review.json) | 18 more texts / 58,611 tokens; all 22 unheld Markdown records read | Tutorials, reference/test excerpts and personal/project prose distinguished; no new origins or holds |
| [Stylesheet closeout](../experiments/corpus-audit/stylesheet-cohort-review.json) | Three more texts / 70,475 tokens; all currently unheld records now read | Page, template and component roles distinguished; no new origins or holds |

All batches preserve held selections rather than replacing them. Assistant observations are recorded
replay inputs, not independent annotations. Offline replay and corruption/selection controls pass;
no corpus code, tests, examples or package setup ran. Historical receipts retain their original bytes.

**Transformations and behavior.** Nine of ten Stack v3 identity mismatches are explained by
placeholder substitutions; one original is unavailable. In `opipoy/file-locker: locker.py`, redaction
changes executable expressions so the original parses and consumed text fails. Fourteen synthetic
controls pass on each of CPython 3.10.20 and 3.14.6. Require a verified parseable original and exact
placeholder-only changes before assigning that diagnosis. Parsing cannot validate runtime behavior,
embedded doctests, redacted assertion meaning or contamination. No bulk filter is adopted, and
redacted originals must not be silently restored.

Static readings record implementation defects separately from missing context. Examples include
mutation before validation, unsynchronized shared state, missing SQL delimiters and contradictory
notebook execution order. MATLAB feature selection ORs its proximity bounds, rejecting an entire
channel when granularity exceeds one; validation resets prediction offsets across files, whose effect
depends on the preprocessor ordering contract. None of these observations is a measured execution result.

Test evidence also needs context: some examples only log output, bypass the behavior under test,
check stale results, compare values with themselves or mislabel timings. Regression inputs and
query snapshots need their harness and expected outputs; printed accuracy needs a documented data
split. Test presence and printed metrics alone establish neither independent correctness nor
efficiency. Batch receipts retain the specific findings and origin-check outcomes.

**Language and execution context.** Preserve original labels and sampling weights while recording
observed language, dialect and role separately. A JavaScript-labelled `.pde` file uses Processing
Java mode, while a SQL-labelled `.cql` file contains Clinical Quality Language test definitions.
Extended syntax, translated algorithms and Unicode cases need relevant toolchains and attribution;
ordinary-language parser failure alone does not establish corruption.

Declarations, tutorials, generated build maps, localized strings and configuration are not equivalent
to standalone implementations. C++ templates need relevant instantiations; UIKit entrypoints and PHP
adapters need surrounding implementations and contracts. TODO comments can accompany implemented
methods. Database dumps embed application data, including identifiers whose synthetic/real status
and suitability remain unresolved. Record those distinctions without automatic rejection, repair,
relabelling or splitting. Markdown includes reference notes with defective examples, intentional
security demonstrations, profiles and six minimal project descriptions. Preserve teaching intent and
document boundaries: an article’s algorithm table is not a set of independently verified exercises.
Historical setup instructions and author-reported outputs require their original context.
Stylesheets span a short page, a 69,312-token eCommerce template and component SCSS. Shared selectors
and vendor prefixes do not prove generated provenance or independent examples. Browser behavior,
keyboard focus, nested-style compilation and asset dependencies need their surrounding context.

**Families and package attribution.** A previously unflagged GoLLIE test contains evaluation examples;
both sampled GoLLIE records remain held. Those examples are exposed material, not blind evaluation
evidence. The vendored Pylint file matches an upstream Git blob from an already-held family; its
installed version and host revision remain unresolved. Dependency parents are distinct from rename
aliases. No exact consumed-byte duplicates were found in the fixed cohorts, which says nothing about
undiscovered near-duplicates or original/transformed copies. The graph remains partial.
Repository/path metadata additionally identifies a LeetCode solution collection; its one sampled
record is conservatively held for lineage/evaluation-coverage review. Its solution text was not
displayed or semantically reviewed, and no exact benchmark match is claimed. A later metadata-only
hold covers the one sampled `belyaev-mikhail/borealis` record under `test/testcases/svcomp/`, bringing
held records to 44. Its task text was also not displayed or semantically reviewed; correspondence
with the intended scoring suite remains unresolved. The Java review additionally records restrictive
notice wording in a `toothlou/nature` file sampled as `no_license`. Placeholder organization text does
not settle applicability or permission; source-use review remains open and family partitions are unchanged.
The [notice follow-up](../experiments/corpus-audit/code-notice-provenance.json) verifies all eight
selected files across four repositories against pinned host bytes and Git blobs: seven new
origins and one reconfirmed origin. It recovers the Objective-C repository's Apache license alongside
its existing copyright headers. The complete Java tree yields no matching ancestor notices under
the declared filename rule; both Java files retain restrictive wording and unresolved source use.
The Ororus stylesheet and sibling Vue component match their host revision, but template attribution
and applicable notices remain unresolved; a license in the separate VvvebJs subtree is not applied
to these records. Eclipse's recovered NOTICE distinguishes Apache-2.0 for code from CC-BY-4.0 for
non-code, consistent with the sampled TSX/SCSS headers. Its referenced dependency context still
needs review; repository labels do not establish dual licensing of each file or training admission.

The [pinned-origin follow-up](../experiments/corpus-audit/pinned-code-origins.json) attempts all 44
remaining unheld Stack v3 records, verifying 42 new origins. Thirty-eight identities use complete
trees; four use smaller path-specific metadata at the same pinned commits after tree responses
exceed the cap. Those four still lack complete ancestor-notice searches. Two pinned source files
return 404 and remain unresolved, without revision substitution. Fifteen distinct notice files are
recovered for 22 records; availability alone is not a notice-applicability decision.

The [retained-origin follow-up](../experiments/corpus-audit/retained-code-origins.json) requests bounded
path histories for all 85 previously unattempted unheld Stack-Edu records. Dataset commit IDs are
absent; ten C++ files match exact retained bytes and verified Git blobs at recovered revisions, with
ten ancestor notices. Those revisions do not reconstruct the original dataset acquisition snapshot.
GitHub quota exhaustion blocks 74 history lookups and one tree verification after exact bytes match.
The successful C++ prefix reflects request order and available quota, not language/source quality.
A fresh-request quota stop was added after acquisition; all actual responses remain recorded.

Current linked origin coverage is **28/104 unheld Stack-Edu** and **68/70 unheld Stack v3**. Stack-Edu's
76 unresolved records comprise 75 quota-blocked outcomes and one earlier failed attempt; Stack v3's
two pinned 404s remain unresolved. No unheld record lacks an attempted outcome, but 74 new requests
returned no history data. Resume those cases only after quota recovery or an available authenticated
route, without repeating the unchanged prior failure. Advance independent web/math work meanwhile.
All 218 assessment records, sampling weights, 176 reads and 44 holds remain unchanged; host evidence
and observed-token counts do not establish eligible supply, notice applicability or copied-source provenance.

Installed metadata and `RECORD` hashes identify Arcade 2.5.7 and stringutils 0.3.0. Arcade's package
notice is recovered. The publisher-hash-verified stringutils wheel contains the sampled file, while
its same-version source archive omits it. A package/version label or ancestor notice is insufficient.
Stringutils notice applicability and both upstream commits remain unresolved; GitHub rate-limit
responses and the oversized Arcade wheel remain recorded acquisition limits.

**Next:** use the existing [qualification packet](../experiments/main-data/QUALIFICATION.md#next-bounded-data-packet)
to resolve origins/notices, complete broader family/near-duplicate discovery and establish finite
eligible inventories. The fixed-cohort reading pass is complete. Natural-code eligibility is
separate from independent exercise verification. The retained 0.477B code stock supports at most
**1.59B total one-pass tokens at 30% natural code**, before exclusions, validation and other bank
constraints. Qualified supply and runtime cost jointly set the experiment horizon; longer confirmation
runs need more eligible baseline data.

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
