# Coding priority

Decision updated 2026-09-19. Agentic coding, general coding and math reasoning are primary
first-release targets for the always-thinking assistant. The paper studies their data and training
across pretraining, mid-training and post-training on the fixed backbone. General usefulness supports these targets.
The first substantive data comparison should test a code-data intervention. This is preparation
for a future experiment; the frozen engineering pilot and its evaluation protocol remain intact.

Code quality is a pretraining requirement as well as a post-training concern. Natural code,
tests, documentation and correct worked explanations should establish useful foundations before
reasoning-SFT. Preserve practical API use, debugging and repository relationships alongside
algorithmic exercises. The pilot's code weight is an engineering setting, not the final target.
Long-context preparation should retain coherent repository units and dependencies toward 128K;
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

## Bounded code inspection

The [inspection receipt](../experiments/corpus-audit/code-preview.json) binds dataset revision
`85182d829f2ce7ea07cca72ebfc509deea1d9f5f`, four revision-checked responses, and local artifacts.
Two seeded, contiguous 16-row Python windows were downloaded, plus one schema row per tier.
This is a schema and feasibility inspection, not a representative quality estimate. Six sampled
records received head-excerpt inspection; that static inspection executed no corpus code.

- L2 has repository/path identifiers and quality scores, but no commit or license columns.
  Five of the 16 sampled paths explicitly indicate vendored or installed packages. This motivates
  checking duplicate families and original provenance; it does not prove those files are bad.
- L3 includes raw code, task, analysis, solution, and test fields but no repository/path columns.
  Establish a verified link to the original source before admission; shared UUID semantics and
  a usable cross-tier join have not been established by this inspection.
- All 16 L3 solutions and tests parse under the local Python runtime. This is only syntax checking;
  the subsequent execution check below tests self-consistency, with independent correctness and
  useful test coverage still open.
- With our frozen tokenizer and BOS/EOS, all 16 L3 `content` fields fit 4K. Their combined size is
  15,685 tokens, versus 34,274 for `full_content`; one full record exceeds 4K. Choose the serialized
  fields deliberately and measure coverage at larger scale. Packing preserves tokens but can split
  a task across context windows. Do not blindly concatenate every representation of the same example.

### Isolated execution follow-up

[code-execution.json](../experiments/corpus-audit/code-execution.json) binds the next CPU check on
the same 16 L3 records. Under Python 3.10.20 and the existing non-root namespace sandbox, **seven
solutions pass their supplied tests and nine fail** (eight assertion failures and one type error).
Every record contains immediate top-level assertions; this adapter is not a general pytest collector.
Known-good, known-wrong, early-exit, deadline, hidden-host-path and unavailable-network controls pass.
All seven passing originals reject a version with explicit return values replaced by `None`.
That gross-bug check is not a measurement of test coverage or independent correctness.

Two failed records were then inspected in full. Row 11273901's required empty dictionary construction
calls an update method that immediately raises on zero arguments. Row 11273888 includes a generated
test comment disputing its own expected result while leaving the assertion active. Both solution
and test defects need review; a failing suite cannot automatically assign fault to the solution.
The 16 rows form one contiguous window, so these counts do not estimate corpus-wide pass rates.
Keep all records outside training for now, including the seven that passed. Resolve provenance,
independent checks, benchmark exclusion and broader coverage before admission.

### Source lineage follow-up — 2026-09-19

The [offline provenance receipt](../experiments/corpus-audit/code-provenance.json) verifies the
four retained responses and the pinned release card/license. All **16 L3 candidates remain on
hold** for missing verified origin, source revision and license evidence. This is an admission
decision for the preview, not a finding that the entire release is unusable.

| Evidence | What it establishes | What remains missing |
| --- | --- | --- |
| Dataset revision `85182d8…` | Identity of the acquired release | Original repository revision |
| L2 `repo_name`, `relative_path`, `uuid` | Named repository and file path in every sampled L2 row | Source commit and license columns |
| L3 `uuid` and generated representations | Identity within the release; retained exercise content | Repository/path, source commit, license columns and verified L2 mapping |
| Card's L0 provenance claim | Publisher describes retaining provenance internally | A usable link from these released L3 records to that archive |

The [pinned card](https://huggingface.co/datasets/openbmb/UltraData-Code/blob/85182d829f2ce7ea07cca72ebfc509deea1d9f5f/README.md)
does not explicitly specify shared UUID semantics, join cardinality or a lookup procedure.
There are no UUID or exact L2-content/L3-raw-content matches across our 17 retained rows per tier
(including schema probes). These unrelated small windows cannot disprove a global join. A root
listing contains no mapping file; we did not scan every shard or recursively prove absence.
Even a future UUID match must bind the original bytes and repository revision, then license evidence.

The card describes `raw_content` as a generated record before serialization, while the inspected
values contain implementation-like Python. Do not assume this field is an authenticated original.
No `copyright`, `license` or `SPDX` marker appears in these 16 raw fields; that limited text scan
is not license identification. Rows 11273894 and 11273898 have identical raw bytes but different
UUIDs, reinforcing the need for content/source-family deduplication.

The project Apache declaration does not resolve source eligibility: the same card calls for
source-repository terms and adds an unchanged-redistribution restriction. Retain this evidence
and keep raw samples outside Git; this audit makes no legal acceptance decision.

**Next route: qualify a small cohort from retained natural code, then derive checked exercises.**
The first archived Python Stack-Edu unit contains 354 retained records with repository/path,
content identities and detected-license metadata, but zero populated `commit_id` values. Thus
existing stock is more traceable, not automatically qualified for the new intervention.
As a concrete starting point, ordinal 29 (`stroxler/tdxutil`, `tdxutil/exceptions.py`) exactly matches
[upstream source at commit `8d21cb9…`](https://github.com/stroxler/tdxutil/blob/8d21cb9c9a489da4c138dd05dd578ffea7765c2a/tdxutil/exceptions.py).
The full SHA256 and plain-file SHA1 match the retained text/content ID. This identifies a matching
immutable revision, not necessarily the revision originally crawled. The
[license notice at that revision](https://github.com/stroxler/tdxutil/blob/8d21cb9c9a489da4c138dd05dd578ffea7765c2a/LICENSE)
is saved with its hash. The dataset's MIT label is less specific than the full notice, which also
contains a restriction on promotional use of the author's name; preserve the exact notice.

### Natural-code cohort qualification — 2026-09-19

The [16-file cohort receipt](../experiments/corpus-audit/natural-code-cohort.json) now extends that
single-source result. We froze a purposeful practical-code selection from the first 125 retained
rows before new upstream lookups; this is a feasibility cohort, not a representative quality sample.
Each lookup considered at most five recent commits affecting the named path. All 16 retained files
match immutable upstream bytes, with SHA256, plain-file SHA1 and Git tree/blob checks. These are
matching revisions, not proof of the original crawl commits.

| Gate | Result |
| --- | --- |
| Exact source/revision match | 16/16 |
| Matching revision with ancestor license-notice evidence | 13/16 |
| Python 3.10 syntax | 14/16 |
| Existing frozen benchmark exclusion | One conservative overlap flag |
| Ready for independent quality review after those gates | 10 files / 8,702 BOS/EOS-inclusive tokens |
| Admitted to the new training intervention | Zero |

The full cohort contains 22,976 candidate tokens under the frozen Mistral tokenizer. Three files
have no conventionally named license file in the inspected revision's ancestor scope: Tesserect's
entry point, navtools' location encoder and minimal-text-diffusion's progress utility. This does
not prove absence of license terms elsewhere; their notices remain unresolved. Exact notices for
the other 13 are retained, including Flytesnacks' NOTICE. Presence is not automatic legal acceptance.

The book-model file has malformed Python; the game-level file uses Python 2 print syntax. Keep
runtime incompatibility separate from a quality judgment about Python 2 code. The compiler allocator
triggers the existing exact/fragment benchmark filter; treat it as held, not proven contamination.
That filter uses the existing five pinned benchmark inputs, both partitions, solely for exclusion.
No final tasks were scored or exposed for manual inspection. The broader planned coding benchmarks
and repository repair families still need pinned exclusions before synthesis/admission.

GitHub metadata resolves the two forks to `windelbouwman/ppci` and `fooof-tools/fooof`, and the
renamed game project to `0xd3adcafe/basinboa`. Preserve those family keys across original, fork and
derived records. All 16 families remain in quarantine; no cross-corpus near-deduplication or final
train/evaluation family split is claimed. There are no exact duplicate files within this cohort.

[Static quality notes](../experiments/corpus-audit/natural-code-review.json) cover eight full files.
For example, one test module casts elements before checking their types and retains only the last
membership result; its tests alone would not establish correct outputs. The exception helper indexes
the first exception argument, motivating a zero-argument exception test. These observations reinforce
that traceable natural code still needs quality checks. Subsequent isolated execution is recorded below.

### Practical CPU checks and broader coverage — 2026-09-19

[practical-code-checks.json](../experiments/corpus-audit/practical-code-checks.json) binds replay
scripts and logs outside Git. The three Python candidates have now been checked in isolation:
the exception helper masks a zero-argument exception; dice passes its 14 documented-behavior checks;
entropy passes ordinary inputs but overflows/underflows at extreme positive standard deviations.
Unspecified invalid-input policies are reported separately from demonstrated contract failures.

A deterministic 36-unit archive sample covers 13,236 files / 9,958,026 Mistral tokens across eight
programming languages and Markdown. All have repository/path and license-label metadata, none a
populated source commit; 185 exceed 4K tokens and none exceed 32K. No exact duplicates occur within
this sample. Archive units are stratified by language and position; unweighted counts are not
population estimates, and path-role hints do not establish executable tests or useful content.

Three JS/TS files match immutable upstream revisions and exact license notices. The `frans` example
passes four supplied tests through a small adapter and eight independent checks, including 1,456
dense-array oracle comparisons; all four intentionally broken variants fail. The task queue passes
six async checks and rejects three broken variants. Synchronous task throws leave it stuck under an
additional proposed robustness policy. Node-stories remains a lower-priority historical-runtime
example. These checks use Node 26.4.0 type stripping and bubblewrap isolation, not the upstream Jest
installation, a TypeScript type check, or a full repository build. The pinned dependency closure is
preserved; no checked example is admitted and no repair exercise has been generated.

**Next: finish the [qualification packet](../experiments/main-data/QUALIFICATION.md), then qualify
supply across the main data banks.** Broader inputs, LiveCodeBench release-v6 public text and family
rules are pinned; the full source-family graph and final scoring coverage remain open. Repeating more tiny helpers is not the next
milestone. Preserve practical behavior, repository context and exact notices. These examples do not
establish scalable code supply.

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

At the working 320B horizon, natural code proposes **96B exposure / 120B eligible unique-token
preparation**, separate from 16B checked-code exposure. Retained stock covers only **0.49664%**
of natural-code exposure before further exclusions. Filling that exposure from this stock would
require about 201 passes; that is arithmetic, not an approved repetition recipe. Even if every
retained token qualified, the preparation gap would be 119,523,225,153 tokens. This audit establishes
no checked-exercise supply and makes no claim about total upstream availability.

Replay aggregate accounting from the verified external document index:

```bash
PYTHONPATH=. .venv/bin/python experiments/corpus-audit/audit_code_supply.py summarize \
  /mnt/speck-data/speck/data-qualification-20260919/code-supply/plan.json \
  /mnt/speck-data/speck/data-qualification-20260919/code-supply/census-v2
```

The summary replays byte-for-byte. Use `scan` with a fresh external output directory to reopen and
retokenize all archives. Raw text, paths and the detailed inventory remain outside Git.

## Immutable bundle follow-up — 2026-09-19

The [bounded receipt](../experiments/corpus-audit/code-bundles.json) follows four new repositories
from the census, one each for Python, JavaScript, TypeScript and Go. Selection used frozen hash
ranks and 100–3,000-token role candidates, excluding known held and previously reviewed families.
At each recovered test revision, only seven of twelve original role-selected files match retained
bytes; only the Python triplet matches completely, and its files concern different exercises.
The search covered at most five test-path commits per repository, not every possible revision.
Repository co-presence is therefore insufficient evidence of a coherent bundle.

Following test imports and related documentation recovers **19 files / 5,650 Mistral tokens**,
including BOS/EOS, at four immutable revisions. Complete Git trees bind their blob identities.
All four repositories have retained MIT notices; that does not complete source-use approval.
GitHub fork metadata connects the TypeScript repository to `trungvose/typescript-data-structures`.
The existing 22-lane exclusion screen flags four files; family propagation quarantines 15.
These are conservative matches, not proof of semantic contamination.

| Language | Payload files | Static finding | Exclusion outcome |
| --- | ---: | --- | --- |
| Python | 3 | Four positive-number tests use `assertTrue(actual, expected)`, which checks truthiness and treats the second argument as a failure message | Whole family held |
| JavaScript | 6 | A balanced-tree fixture misses unguarded child dereferences for single-node/sparse trees | Whole family held |
| TypeScript | 6 | Queue exercise with six test cases and a resolved local type import; no independent correctness result | Whole family held |
| Go | 4 | Supplied test expects `MinInt(1, 0) == 1`, contradicting the implementation's return of 0 | No content flag; still unqualified |

No corpus code ran, no dependencies were installed, and **nothing is admitted**. The Go
`train` candidate partition is only a splitter output, not an eligibility decision. All four
examples are educational exercises; this small selection cannot estimate corpus defect rates or
establish practical API/debugging/repair coverage. Dependency closure and independent tests remain
open. Initial unrelated role samples and license notices are provenance evidence, outside the
proposed payload screen. Raw files, complete responses, review and byte-identical offline replay
remain under `/mnt/speck-data/speck/data-qualification-20260919/code-bundles`.

## Data work to do next

1. Freeze broader code-evaluation exclusions and family separation before exercise derivation;
   the bounded independent checks above are complete. The retained L3 preview remains on hold;
   reopen it when an authoritative mapping or independently verified origin becomes available.
   Do not spend on a bulk L3 acquisition to infer an undocumented join. Neither existing natural
   stock nor passing generated tests alone grants eligibility for the new intervention.
2. Pin expansion routes and measure eligible yield by language and practical role before larger
   acquisition or freezing the main horizon. The bounded bundle follow-up is complete: require
   explicit module/import/behavioral linkage and independent test oracles, not just co-present
   filenames. Prioritize library use, debugging and repository repair over more tiny algorithm
   exercises; apply known family holds before selection. Algorithm puzzles alone do not cover
   practical coding.
3. Validate the Python lane first, using the existing non-root, isolated, resource-limited runner.
   Check task/solution consistency, dependencies, test discovery, empty or vacuous tests, and
   deliberately incorrect solutions. Add independently checked edge cases: generated solutions
   passing their own generated tests demonstrate self-consistency, not independent correctness.
4. Deduplicate natural code and derived exercises by source family before splitting. Exclude the
   next evaluation protocol's tasks and related solutions from every training representation,
   including raw code, reasoning, tests, and documentation. Count eligible Mistral tokens by
   language, serialized field, and verification outcome, including rejected material and CPU cost.
5. Retain multilingual coverage. The published code release has no separate TypeScript split;
   preserve our existing TypeScript source. JavaScript/TypeScript is the next practical verification
   lane after Python. Do not treat a Python-only audit as multilingual qualification.

## First comparison to prepare

Prepare **retained natural code versus natural code plus checked exercises**, starting from the
same useful base checkpoint. Use the flagship's working 35% total code share as the starting
proposal for both arms, with all non-code sources fixed; freeze actual shares after supply checks.
The pilot's 15% code share is not the main recipe. Keep non-Python code unchanged. In the treatment,
replace half of the Python code tokens with eligible, checked exercises; the control uses natural
Python throughout. The 50% replacement is a testable starting hypothesis, not a chosen main recipe.
This tests the complete checked-exercise intervention, not the separate effects of synthesis,
selection, or verification. L2 natural-code replacement and a higher total code share are later
questions, avoiding multiple simultaneous changes in the first comparison.

Before launch, freeze source manifests, no-overlap partitions, language shares, serialized fields,
repetition, optimizer and token schedule, development endpoints, general-capability regression
tolerances, and the total cost ceiling. Both arms use the same code-token budget and inference
protocol. If checked supply is insufficient, reduce the experimental horizon before launch rather
than silently repeat examples. The 105M engineering pilot alone is not evidence of a useful base.

At the recorded 12,859 full-trainer tokens/s, 1B total mixture tokens per arm projects to about 43.2
single-H100 trainer hours for both arms at the pilot's overhead rate. Add common-base training,
evaluation and data generation/verification to the appropriate ledgers. Both arms and their
evaluations must fit the 91-hour comparison reservation. This is a planning scenario, not a frozen
run length or GH200 forecast. Qualify the [changed-data continuation path](training.md#mid-training-readiness)
before executing either arm; an ordinary same-manifest branch cannot run this experiment.

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
explicitly, within the existing desired 320–400B scale and its unresolved compute feasibility gate.
