# Speck: first useful baseline

Updated 2026-09-18. This is the single current plan. Change it in place as evidence arrives;
Git retains earlier decisions. Concrete run settings belong beside their experiment.

## Goal

Build an English-first, broadly useful small model with strengths in math, coding, tool use, and
reliable instruction following. General knowledge, writing, and conversation remain requirements.
"Good at everything" is an ambition; each capability needs its own evidence.

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
  `<think>...</think>` before its answer; training format, reasoning budget, and tool protocol still
  need an evaluated recipe. Existing post-training work in the sibling project must be reconciled.

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
4. **Develop useful behavior.** Once the base learns reliably, prepare a costed SFT baseline with
   verified math/code solutions, ordinary assistance, and structured tool interactions. Reuse the
   separate post-training work where compatible. Add distillation or RL only after a clear baseline,
   working graders, and an affordable experiment exist.
5. **Scale what works.** Decide the main token horizon and stage budgets from measured data supply,
   learning curves, and all-in runtime. Keep time for post-training and final evaluation. The old
   320B/400B targets are historical estimates, not current commitments.

## Compute

The application is recorded as under evaluation; access and site details are unconfirmed.
The requested envelope is four GH200s, 5,000 GPU-hours, roughly 90 calendar days.
Retain 889 hours as protected recovery/evaluation reserve. Four allocated GPUs cost four GPU-hours
per wall hour even when some are idle. Qualification is capped at 70 hours; the initial pilot is
capped at 50. Allocate the remaining work after those measurements, rather than
maintaining speculative budgets for multiple research programs. No GH200 jobs have been launched.

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
- The [105M-token pilot](experiments/pilot/README.md) fixes initial source weights, optimization,
  checkpoint milestones, and a 50-hour cost ceiling. Selection, joint exclusion, packing, and reopen
  checks are complete. Full one- and four-rank CPU loader scans consumed the planned tokens without
  repetition and passed fresh-process replay. The [preparation receipt](experiments/pilot/preparation.json)
  records actual source/language exposures and artifact hashes. GPU launch qualification remains open.
- Five public evaluation inputs and scorer revisions are pinned; development/final task identities
  are materialized. Grader execution, tools, public comparators, and site settings remain open.
- A full 500,000-row post-training census and 256-row-per-subset length sample are complete.
  The text-only 4K fit estimate is approximately 249,000 rows; 110,000 tool-bearing rows require
  a tool-aware format. New SFT masks preserve context-only assistant turns; local parent loading
  and complete-conversation packing exist. The actual assistant recipe is still to be evaluated.
- The tiny offline base-to-assistant workflow passes exact resume for both stages. Native/export
  tokenizer and generation checks pass for both tiny checkpoints. The portable suite passes 668
  tests; separate local CUDA tests and the full-size synthetic probe are recorded above.

**Next:** qualify grader execution and integrate the assistant/tool protocol. Once access arrives,
repeat hardware qualification on one and four GH200 workers, check the production CUDA training
loop and scheduler recovery, then run the bounded pilot. Main training scale and model-quality
claims require its measurements.
Preparation procedures and artifact locations are in [data](docs/data.md); supporting evidence is in
[research notes](docs/research.md). Historical result bytes remain in [Git](archive/README.md).
