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

## Data work to do next

1. Establish source eligibility and provenance for a bounded L3 candidate. The dataset's project
   license does not override original repository terms. If lineage cannot be established, use
   an eligible alternative or derive checked exercises from our traceable natural-code stock.
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
