# Coding priority

Decision updated 2026-09-18. Coding is a first-release priority alongside broad usefulness.
The first substantive data comparison should test a code-data intervention. This is preparation
for a future experiment; the frozen engineering pilot and its evaluation protocol remain intact.

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

Next qualify 16 practical Python source files with this origin/revision/byte/license evidence,
retaining attribution and rejecting unresolved or vendored origins. Establish source-family
partitions and benchmark exclusions before synthesis, record each exercise's parent hash and
generation recipe, and run independent edge-case checks in the existing sandbox. Exception
handling offers a concrete practical topic, but this single source is neither sufficient supply
nor an admitted exercise. Count eligible tokens only after the remaining gates pass.

## Data work to do next

1. Follow the natural-code qualification route above. The retained L3 preview remains on hold;
   reopen it when an authoritative mapping or independently verified origin becomes available.
   Do not spend on a bulk L3 acquisition to infer an undocumented join. Neither existing natural
   stock nor passing generated tests alone grants eligibility for the new intervention.
2. Expand inspection across languages, file roles, lengths, and upstream quality scores. Keep
   implementation, tests, documentation, practical library use, and repair examples visible in the
   inventory. Algorithm puzzles alone do not cover practical coding.
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
same useful base checkpoint. For a bounded first experiment, keep the current 15% total code share
in both arms and all non-code sources fixed. Keep non-Python code unchanged. In the treatment,
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

At the measured 13,615 tokens/s, 1B total mixture tokens per arm would cost approximately 40.8
single-H100 optimization hours for both arms. That excludes common-base training, evaluation,
verification, checkpointing, and startup. It is a planning scenario, not an approved run length or
a GH200 forecast. Cost a smaller development experiment first if pilot learning curves support it.

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
