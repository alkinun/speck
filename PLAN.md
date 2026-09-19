# SpeckLabs: first from-scratch model program

Updated 2026-09-19. This is the single current plan. Change it in place as evidence arrives;
Git retains earlier decisions. Concrete run settings belong beside their experiment.

## Goal

The user clarified on 2026-09-19 that this is SpeckLabs' first from-scratch foundation-model
program. Use the 5,000-total-GPU-hour allowance to ship a useful, competitive and efficient model
and establish the data, training and evaluation capability needed for a future 50,000-GH200-hour
program and beyond. Future compute is an ambition, not confirmed access or authorization to spend.
Pretrain our own weights; adapting an external pretrained base is not the first-release route.
Published models remain comparators and eligible teachers, with their contributions disclosed.

Build an English-first thinking model primarily for agentic coding, general coding, mathematical
reasoning and tool-mediated tasks. The final assistant always uses the reasoning protocol; a
non-thinking switch or purely instruct release is not a product objective. Reasoning effort may
vary with task difficulty while preserving thinking behavior. General language and knowledge
support these priorities and remain regression checks.

Coding is a first-release priority. The [coding plan](docs/coding.md) makes checked code exercises
the first substantive data-comparison candidate, with practical repair and multilingual evaluation.
OpenBMB is a primary data-research reference; source adoption still requires our own evidence.
Data quality, diversity and learnability are the main development priority while retaining the
current architecture. The [data recipe direction](docs/data.md#recipe-direction--2026-09-19)
and [assistant inventory](docs/assistant.md#main-assistant-data-direction--2026-09-19) cover both
pretraining and post-training, including the newly confirmed approximately 128K context target.
The [main data working plan](experiments/main-data/README.md) now sets concrete preparation targets:
320B desired / 400B stretch base tokens (35% code, 25% math), with 100B as the current
measurement-based budget-fit scenario; a provisional 8B context extension; and 1.5M unique
thinking/agent conversations within a 1–2M range. The existing 500K conversations are starting stock.
The [competitive strategy](docs/competitive.md) separates this budget fit from release quality:
retain 1.2B as the first from-scratch candidate, measure runtime headroom, and require matched
task/efficiency evidence before a competitive flagship claim. Model size and token horizon remain
subject to this evidence; pretrained adaptation is excluded by the clarified program objective.

The immediate deliverable is a reproducible training baseline. The first flagship release includes
identified base/assistant checkpoints and a companion [technical report](docs/report.md). The previous
long-context paper, mixture-selection funnels, tokenizer competition, and architecture sweeps are
historical. The planned context extension toward 128K needs a new measured training/evaluation
contract; old long-context experiments do not qualify the current checkpoint. A focused research
claim still requires its own evidence.

## Starting point

- Keep the existing 1.2B-class, 24-layer KDA/GQA hybrid as the first candidate: 18 recurrent layers,
  six global-attention layers, width 2048, tied embeddings, SwiGLU, sigmoid KDA gates, and NoPE globals.
- Reuse the frozen Mistral 32K tokenizer. The model reserves three extra chat-role IDs (32,003 rows).
  Check its fingerprint when reusing cached data; do not retrain a tokenizer.
- Begin at 4K context. Use BF16 activations with existing FP32 parameter/optimizer storage and
  Muon/AdamW. The hardware probe's learning rate is diagnostic, not a chosen training recipe.
- Extend toward approximately 128K after establishing the base, then support short and long SFT
  examples. Retain complete long records now; 4K fit only controls eligibility for the initial
  phase. Freeze extension stages and cost after measuring memory, throughput and context quality.
- Preserve base and assistant checkpoints separately for training provenance. The final assistant
  reasons in `<think>...</think>` before answers or tool actions, with brief or deeper reasoning
  appropriate to the task. The [rehearsal contract](docs/assistant.md) defines its
  initial serialization and tool protocol; reasoning budgets and learned behavior still need an
  evaluated recipe. The sibling project's retained stock supplies the finite rehearsal.

These are starting choices that reuse working code. None proves an advantage over a dense model.
Change one only when a measured failure or capability comparison justifies the work.

## Immediate order of work

The single-H100 rehearsal and planning measurements are complete. The frozen engineering pilot
completed 800 steps on the migrated H100; its [execution receipt](experiments/pilot/h100-run.json)
records 2.27 training hours, final validation loss 3.379 and 19.3 GiB peak allocated memory.
The restored complete development evaluation and isolated local code grading are now finished:
2,619 tasks in 2.12 evaluation hours. Results remain weak: GSM8K strict 0/253, compiled code
pass@1 0/33, IFEval strict prompts 12/101, ARC normalized 44/222 and HellaSwag normalized
530/2,010. These frozen development subsets establish a scored engineering endpoint, not a useful
flagship. All eight checkpoints, the original remote export, CPU reconstruction and final recovery
logs are verified locally; [backup closeout](experiments/pilot/backup-result.json) is complete.
The Runpod instance can be stopped for this pilot. Provider billing remains unknown and has not
been stopped by the agent. The 16-file natural-code audit is now complete: ten files clear lineage,
syntax and existing benchmark checks for independent quality review. Next test the three bounded
practical-code candidates; no new GPU run is authorized by pilot completion.

| Work | Current state | Concrete next deliverable |
| --- | --- | --- |
| Runtime qualification | H100 single-worker rehearsal and timing complete | On changed hardware, verify environment/input identities and relevant recovery checks; GH200 and four-worker qualification remain separate |
| Code data | Natural-code cohort audited: 16 exact upstream matches, 13 with notice evidence, ten preliminary review candidates; L3 remains held | Independently test exception handling, dice parsing and entropy candidates; freeze broader benchmark exclusions/family splits before derivation |
| Web data | Matched published evidence favors natural Ultra-FineWeb; now the leading candidate to qualify | Audit the pinned English subset against retained FineWeb-Edu, checking selection threshold, coverage, overlap and eligible supply; keep DCLM as an independent comparator |
| Assistant data and context | 500K retained conversations inventoried; finite tool-aware rehearsal qualified; always-thinking target at approximately 128K | Audit reasoning quality, code/math correctness and agent outcomes; balance brief/deep reasoning, retain long trajectories and recover missing source tails in a separately qualified acquisition |
| Rental launch | Migrated H100 passed host preflight; locked environment and frozen payloads verified | Preserve launch provenance, shared deadline and cumulative accounting |
| Current paid experiment | Training, export, all 2,619 development scores and complete backups verified | Pilot closed; Runpod can be stopped. Continue CPU data qualification |
| Main training | Desired 320–400B mixture recorded; current 5,000-GPU-hour envelope needs higher measured throughput or more compute | Qualify new supply against the 35% code / 25% math targets; freeze measured manifests and runtime before launch |

The [executable launch packet](docs/pilot-rental.md) completed training. Inspected recovery fixed
offline template lookup, separated BF16 wrapper identity from FP32 cache consistency, and removed
singleton rank variables before evaluation. All failed attempts are retained; export checks pass and
development evaluation was running within the remaining original deadline when SSH became
unavailable. Restored access confirmed that evaluation was interrupted; its stale running result and
84 partial rows are retained separately. The final export has been reconstructed on CPU, with source identities, model/tokenizer checks
and all 18 output-file hashes verified locally. The migrated container blocks user namespaces, so generated code is graded locally.
The supervisor enforces a shared six-hour
execution deadline and conservatively reserves six GPU-hours against the 50-hour pilot ceiling.
Record cumulative external usage before execution; provider billing is separate. Start the pilot
in a fresh run directory; preserve the 48-step timing experiment as separate evidence. Stop for
nonfinite loss/gradients, data or checkpoint failures, or the bound budget; inspect before restarting.
Back up the resulting evidence before the user stops/deletes the rental. Provider billing continues
until the instance is stopped/deleted; process deadlines do not stop billing.

The immediate data work prioritizes code verification. Broad synthetic-source replacement and
main mixture weights remain open. Freeze new evaluation identities before producing candidate
training packs. A passing generated test suite alone is not sufficient for data admission.

## Sequence

1. **Qualify the runtime.** Run the offline smoke workflow, then the bounded 4K hardware probe on
   the actual allocation: one worker first, four workers after review. Check memory, kernels,
   optimizer, checkpoint/RNG restart, and data-independent throughput. Complete production-loader,
   cache/numerical parity, and scheduler recovery checks before a paid corpus run.
2. **Prepare one pilot.** Reopen retained data, check exclusions and tokenizer identities, and
   materialize a small source-separated corpus with separate validation. Use broad text plus math
   and code. Freeze actual source weights, repetition, data order, learning rate, batch, token
   endpoint, evaluation schedule, and maximum cost in that experiment before launching it.
3. **Train and inspect.** Run the frozen 104,857,600-token pilot within its 50 GPU-hour ceiling.
   If measured runtime cannot fit, freeze a smaller experiment before launching. Inspect losses by
   source, gradient health, samples, checkpoint recovery, and cost. This is an engineering baseline,
   not a causal comparison or claim about architecture quality.
4. **Establish the main data recipe.** Complete the [corpus quality and coverage work](experiments/corpus-audit/README.md)
   before committing the main training budget. Preserve the engineering pilot, inspect retained
   text, validate candidate sources and eligible supply, and freeze one affordable data comparison.
   Quality labels, mixture weights, repetition, and staged use of refined material need evidence;
   correct packing and upstream dataset branding do not supply it.
   Prioritize the [code-data comparison](docs/coding.md#first-comparison-to-prepare), keeping general
   coverage fixed. In web preparation, prioritize natural Ultra-FineWeb qualification based on the
   [matched paper review](docs/research.md#openbmb-web-data-review--2026-09-19), retaining FineWeb-Edu
   as the control and DCLM as an independent comparator. Ultra-FineWeb-L3 remains a separate
   candidate, not an established Cosmopedia replacement. No new GPU experiment is launched here.
   The [L3 provenance audit](experiments/corpus-audit/code-provenance.json) holds all 16 preview
   records outside training: source revision/license and a supported L2 join remain unresolved.
   Next qualify 16 practical Python files from retained natural-code stock; one exact upstream
   commit/content match and license notice are now retained. Complete origin, eligibility,
   source-family exclusion and independent-test checks before deriving an admitted exercise set.
5. **Extend context and develop useful behavior.** Once the base learns reliably, qualify extension
   toward approximately 128K with coherent long records and short replay. Measure runtime/memory,
   retrieval across positions, cross-document reasoning and short-task retention before claiming
   support. Preserve full long reasoning/tool trajectories for the later SFT stages; initial 4K
   compatibility is not a permanent data filter. Prepare a costed SFT baseline with
   verified reasoning for math/code solutions, structured tool interactions and supporting ordinary
   assistance. Reuse the separate post-training work where compatible. Audit source/answer quality,
   measure reasoning/final-answer/tool-context tokens and balance brief versus deep reasoning under
   one always-thinking inference protocol. Do not train a separate non-thinking response mode.
   Score model-driven tool completion separately from scripted environment checks. Consider a 4K
   capability-focused continuation with broad replay only if pilot measurements justify its cost.
   Add distillation or RL only after a clear baseline,
   working graders, and an affordable experiment exist.
6. **Scale what works.** Decide the main token horizon and stage budgets from measured data supply,
   learning curves, and all-in runtime. Prepare the [320B desired / 400B stretch mixture](experiments/main-data/README.md).
   The 100B scenario fits the current H100-rate base reservation; it is not a replacement flagship
   target. Resolve the throughput or funding gap before launch while preserving context extension,
   post-training and final evaluation. Count four allocated GPUs as four GPU-hours per wall hour;
   confirm the provider's allowance units and measure GH200 per-device speed and distributed scaling.

## Compute

The application is recorded as under evaluation; access and site details are unconfirmed.
The requested envelope is four GH200s, 5,000 GPU-hours, roughly 90 calendar days.
Retain 889 hours as protected recovery/evaluation reserve. Four allocated GPUs cost four GPU-hours
per wall hour even when some are idle. Qualification is capped at 70 hours; the initial pilot is
capped at 50. The working allocation reserves 2,300 hours for base pretraining, 800 for context
extension, 800 for post-training and 91 for a bounded data comparison, alongside the protected
889-hour reserve. These sum to 5,000; confirm actual hardware rates and usable supply before launch.
No GH200 jobs have been launched.

The [measured H100 planning inputs](experiments/qualification/timing-result.json) put the original
one-worker pilot at about **2.20 GPU-hours** for training, validation and checkpoints; the completed
800-step run measured **2.27 hours** including cold startup. One full
development capability pass adds roughly **2.15 GPU-hours** at the full output caps, before grading
and operational overhead. Use **6 single-H100 GPU-hours as a working reservation** for that combined
workflow, including margin; the current run uses this reservation within the frozen 50-hour
ceiling. A full final capability pass is a roughly **9-hour backend scenario per checkpoint**, not a
measured full-suite runtime. Account for every allocated GPU if evaluation leaves others idle.
GH200 rates and four-worker scaling still require measurements on the allocation.

## What success means

| Capability | Evidence to collect |
| --- | --- |
| Broad usefulness | Held-out loss by source, general knowledge, writing, and conversation samples |
| Math | Checked final answers; report easy and harder problems separately |
| Coding | Execution-based correctness on held-out tests; include repair tasks |
| Agentic coding | Held-out repository tasks: inspect, edit, test and repair; measure actual completion and regressions |
| Tools | Valid arguments, correct tool selection, correct use of results, and task completion |
| Thinking behavior | Reasoning-protocol adherence, correctness versus reasoning budget, useful correction and bounded tool loops |
| Reliability | Instruction/format compliance, appropriate abstention, correction after errors, and grounded answers |
| Efficiency | End-to-end GPU-hours, output-token budget, latency, and memory |

Use a fixed development set while improving a recipe, and a separately held-out final test.
Pin task/scorer versions before use; report failures and denominators. A tool-call-shaped string is
not evidence of successful tool use. Match decoding budgets when comparing models.

## Current state and immediate work

- Model/training/data/recovery implementations and CPU tests exist. The full 1.2B model passed
  [one-worker synthetic optimization and fresh-process restart on an RTX 3090](experiments/qualification/local-result.json),
  with deterministic kernels and no compilation. GH200 feasibility and four-worker execution are unmeasured.
- The first hardware configuration is [ready for dry-run binding](experiments/qualification/README.md).
- FineWeb's completed receipt reports 2,306,703,052 cached tokens with a passing reopen check.
  Stack-Edu reports 476,774,847 tokens before full exclusion; it is not final training stock and is
  not interchangeable with restricted Stack v3. Both finite preparation services exited successfully.
- Other retained sources include FineMath, Cosmopedia, peS2o, FineWiki, and approved natural
  UltraData-Math. Existing counts are not a globally deduplicated training union.
- The [corpus audit follow-up](experiments/corpus-audit/README.md#follow-up-decisions--2026-09-18)
  keeps FineMath and prepares a narrow topic-directory exclusion candidate (18.78M source tokens).
  Numeric normalization did not reveal material strict-template repetition. A bounded English L3
  inspection found answer/refinement defects; prioritize Q&A consistency checks before admitting
  new synthetic data. Main mixture weights remain open and the frozen pilot is unchanged.
- The [105M-token pilot](experiments/pilot/README.md) fixes initial source weights, optimization,
  checkpoint milestones, and a 50-hour cost ceiling. Selection, joint exclusion, packing, and reopen
  checks are complete. Full one- and four-rank CPU loader scans consumed the planned tokens without
  repetition and passed fresh-process replay. The [preparation receipt](experiments/pilot/preparation.json)
  records actual source/language exposures and artifact hashes. H100 single-worker recovery is now
  qualified; the [one-H100 launcher](docs/pilot-rental.md) now binds a shared deadline and cumulative
  reservations. Host preflight and allocation-specific distributed checks remain open.
- Five public evaluation inputs and scorer revisions are pinned; development/final task identities
  are materialized. Golden grader checks pass, including all 33 development code canonical solutions
  in an isolated runner and five scripted tool episodes. A pinned Qwen3-0.6B reference completed
  eight development examples per benchmark. This validates the pipeline, not comparative quality.
- A full 500,000-row post-training census and 256-row-per-subset length sample are complete.
  The text-only 4K fit estimate is approximately 249,000 rows; 110,000 tool-bearing rows require
  a tool-aware format. The versioned adapter preserves explicit calls/results and assistant weights.
  A complete, benchmark-filtered rehearsal contains 64 training and 16 validation conversations,
  balanced between text and tools. The final assistant mixture and teacher correctness remain open.
- The tiny offline base-to-assistant workflow passes exact resume for both stages. Native/export
  tokenizer and generation checks pass for both tiny checkpoints. The portable suite passes 726
  tests; separate local CUDA tests and the full-size synthetic probe are recorded above.
- Full-size production base training on actual pilot data now passes fresh-process model/optimizer
  replay at the original CUDA tolerance, with exact loader/RNG state. A resumed Muon allocation
  failure exposed avoidable temporary-tensor retention; the fix passed the same check. This four-step
  diagnostic does not replace the 800-step pilot or establish model quality.
- The full-size model also completes the finite assistant rehearsal and fresh-process SFT recovery
  with exact loader/RNG state and the same model/optimizer tolerance. Masked sum-loss kernels are
  warmed before SFT restore. The [readiness receipt](experiments/qualification/readiness.json)
  records failures, source identities, checks, and the portable transfer archive.
- The [GH200 rental runbook](docs/gh200.md) has a verified portable bundle, locked ARM64 dependency
  resolution, relocated inputs, and a bounded loader/kernel/base/SFT/export sequence. The rebuilt
  bundle includes the phase-specific assistant exporter and stricter recovery-counter checks.
- A rented [single H100 SXM passed the full rehearsal](experiments/qualification/h100-result.json):
  loader/replay, KDA numerics/gradients/recurrence, full-size base/SFT fresh-process recovery, native
  CUDA generation, and CPU native/Transformers export parity. Each recovery check compared 321 model
  and 588 optimizer tensors, with exact RNG/loader state. Additional metadata checks pass too.
  A separate four-step probe at the 131,072-token pilot batch (32 accumulated 4K microbatches) completed
  524,288 tokens, averaging 13,724 tokens/s over its last three logged steps with 19.3 GiB peak allocated
  GPU memory. This is a short timing diagnostic, not sustained throughput or model-quality evidence.
  The original supplemental verifier error and its corrected check are retained. All 113 evidence
  files are verified locally; ARM64 GH200 and four-worker execution remain unqualified.
- Longer H100 measurements cover 48 production steps (6.29M tokens), a fresh-process restart,
  four full 786K-token validation passes, three durable saves, SFT and inference timing, and 70
  development evaluation requests. The 28 steady base steps average **13,615 tokens/s**, using
  **19.3 GiB** peak allocated memory; validation takes **14.2 seconds**, a save **11 seconds / 9.16 GiB**,
  and fresh-process resume overhead about **17 seconds**. The [timing receipt](experiments/qualification/timing-result.json)
  retains exact configurations, failed attempts, hashes and projection formulas. Four sequences per
  microbatch reached **16,398 tokens/s (+20%)** in six steady observations; qualify its restart and
  longer behavior before changing the frozen pilot. SFT reached 13.5K padded positions/s, but only
  23% of positions in this rehearsal are supervised. Measure the final corpus and existing length
  buckets before assigning an SFT budget. Batched native decode reached 428 aggregate tokens/s at
  batch eight versus 55 at batch one; the capability runner remains serial and needs separate parity
  checks before adopting batching. These measurements establish engineering costs, not model quality.
- Real evaluator execution exposed and fixed missing full-sequence likelihood logits, empty EOS
  decoding that silently stopped generation after one token, and base generation of undecodable
  reserved assistant IDs. The local-export evaluator and refreshed transfer bundle contain the fixes.
  All 138 new evidence files are verified locally; historical exports/results remain unchanged.
- The [pilot launch receipt](experiments/pilot/rental-readiness.json) binds a private 152 MiB archive
  containing source, frozen data, benchmark inputs and offline export/grading assets. Extraction,
  fresh Git clone, relocated launch preview and all golden graders pass locally, including 33
  canonical code tasks. The new supervisor has CPU tests for budget, tampering, failure and timeout
  behavior. Training completed all 800 steps; export checks pass after inspected recovery and
  capability evaluation was interrupted by remote access loss, then completed in the restored session.
  All eight checkpoints, both exports, graded development outputs and recovery records are verified.
  The [execution receipt](experiments/pilot/h100-run.json) preserves the failures and completed
  engineering outcome; [development scores](experiments/pilot/development-result.json) remain weak.
- The [ZGCM-1 review](docs/research.md#zgcm-1-review--2026-09-18) prioritizes verified assistant
  supervision, response-budget measurements, and learned tool evaluation after the pilot. It does
  not change the frozen pilot, tokenizer, or rental bundle. No ZGCM data has been admitted.

Follow the immediate order above. On the eventual allocation, qualify ARM64 GH200, four-worker
execution and scheduler recovery, and measure its pilot batch before running there. Main-corpus
expansion and joint exclusions remain necessary: the retained code-language mixture has a
pre-exclusion single-pass ceiling of about 2.76B total mixture tokens, and post-exclusion eligibility
will be lower. Main training scale and model-quality claims require the pilot's measurements.
Preparation procedures and artifact locations are in [data](docs/data.md); supporting evidence is in
[research notes](docs/research.md). Historical result bytes remain in [Git](archive/README.md).
