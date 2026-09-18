# Speck: first useful baseline

Updated 2026-09-18. This is the single current plan. Change it in place as evidence arrives;
Git retains earlier decisions. Concrete run settings belong beside their experiment.

## Goal

Build an English-first, broadly useful small model with strengths in math, coding, tool use, and
reliable instruction following. General knowledge, writing, and conversation remain requirements.
"Good at everything" is an ambition; each capability needs its own evidence.

Coding is a first-release priority. The [coding plan](docs/coding.md) makes checked code exercises
the first substantive data-comparison candidate, with practical repair and multilingual evaluation.
OpenBMB is a primary data-research reference; source adoption still requires our own evidence.

The immediate deliverable is a reproducible training baseline. The first flagship release includes
identified base/assistant checkpoints and a companion [technical report](docs/report.md). The previous
long-context paper, mixture-selection funnels, tokenizer competition, and architecture sweeps are
historical. A focused research claim or longer context requires its own measured justification.

## Starting point

- Keep the existing 1.2B-class, 24-layer KDA/GQA hybrid as the first candidate: 18 recurrent layers,
  six global-attention layers, width 2048, tied embeddings, SwiGLU, sigmoid KDA gates, and NoPE globals.
- Reuse the frozen Mistral 32K tokenizer. The model reserves three extra chat-role IDs (32,003 rows).
  Check its fingerprint when reusing cached data; do not retrain a tokenizer.
- Begin at 4K context. Use BF16 activations with existing FP32 parameter/optimizer storage and
  Muon/AdamW. The hardware probe's learning rate is diagnostic, not a chosen training recipe.
- Preserve base and assistant checkpoints separately. The intended assistant can reason in
  `<think>...</think>` before its answer. The [rehearsal contract](docs/assistant.md) defines its
  initial serialization and tool protocol; reasoning budgets and learned behavior still need an
  evaluated recipe. The sibling project's retained stock supplies the finite rehearsal.

These are starting choices that reuse working code. None proves an advantage over a dense model.
Change one only when a measured failure or capability comparison justifies the work.

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
   coverage fixed. Ultra-FineWeb-L3 remains a candidate, not an established Cosmopedia replacement.
5. **Develop useful behavior.** Once the base learns reliably, prepare a costed SFT baseline with
   verified math/code solutions, ordinary assistance, and structured tool interactions. Reuse the
   separate post-training work where compatible. Audit source/answer quality, measure direct versus
   reasoning supervision by tokens, and interleave general/tool examples under one inference protocol.
   Score model-driven tool completion separately from scripted environment checks. Consider a 4K
   capability-focused continuation with broad replay only if pilot measurements justify its cost.
   Add distillation or RL only after a clear baseline,
   working graders, and an affordable experiment exist.
6. **Scale what works.** Decide the main token horizon and stage budgets from measured data supply,
   learning curves, and all-in runtime. Keep time for post-training and final evaluation. The old
   320B/400B targets are historical estimates, not current commitments.

## Compute

The application is recorded as under evaluation; access and site details are unconfirmed.
The requested envelope is four GH200s, 5,000 GPU-hours, roughly 90 calendar days.
Retain 889 hours as protected recovery/evaluation reserve. Four allocated GPUs cost four GPU-hours
per wall hour even when some are idle. Qualification is capped at 70 hours; the initial pilot is
capped at 50. Allocate the remaining work after those measurements, rather than
maintaining speculative budgets for multiple research programs. No GH200 jobs have been launched.

The [measured H100 planning inputs](experiments/qualification/timing-result.json) put the original
one-worker pilot at about **2.20 GPU-hours** for training, validation and checkpoints. One full
development capability pass adds roughly **2.15 GPU-hours** at the full output caps, before grading
and operational overhead. Use **6 single-H100 GPU-hours as a working reservation** for that combined
workflow, including margin; this is not a launched job or a replacement for the frozen 50-hour
ceiling. A full final capability pass is a roughly **9-hour backend scenario per checkpoint**, not a
measured full-suite runtime. Account for every allocated GPU if evaluation leaves others idle.
GH200 rates and four-worker scaling still require measurements on the allocation.

## What success means

| Capability | Evidence to collect |
| --- | --- |
| Broad usefulness | Held-out loss by source, general knowledge, writing, and conversation samples |
| Math | Checked final answers; report easy and harder problems separately |
| Coding | Execution-based correctness on held-out tests; include repair tasks |
| Tools | Valid arguments, correct tool selection, correct use of results, and task completion |
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
  records actual source/language exposures and artifact hashes. GPU launch qualification remains open.
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
  tokenizer and generation checks pass for both tiny checkpoints. The portable suite passes 713
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
- The [ZGCM-1 review](docs/research.md#zgcm-1-review--2026-09-18) prioritizes verified assistant
  supervision, response-budget measurements, and learned tool evaluation after the pilot. It does
  not change the frozen pilot, tokenizer, or rental bundle. No ZGCM data has been admitted.

**Next:** on the eventual allocation, repeat the rehearsal on ARM64 GH200, qualify four-worker
execution and scheduler recovery, and measure its pilot batch before the bounded pilot. If continuing
on rented H100 hardware, bind a one-worker launch and its cumulative budget using the measurements
above; the timing run stopped at 48 of 800 steps. Expand and jointly exclude
main-corpus supply before choosing its token
horizon: the retained code-language mixture has a pre-exclusion single-pass ceiling of about 2.76B
total mixture tokens, and post-exclusion eligibility will be lower. Main training scale and
model-quality claims require the pilot's measurements.
Preparation procedures and artifact locations are in [data](docs/data.md); supporting evidence is in
[research notes](docs/research.md). Historical result bytes remain in [Git](archive/README.md).
