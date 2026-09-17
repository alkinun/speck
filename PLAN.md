# Speck: first useful baseline

Updated 2026-09-17. This is the single current plan. Change it in place as evidence arrives;
Git retains earlier decisions. Concrete run settings belong beside their experiment.

## Goal

Build an English-first, broadly useful small model with strengths in math, coding, tool use, and
reliable instruction following. General knowledge, writing, and conversation remain requirements.
"Good at everything" is an ambition; each capability needs its own evidence.

The next deliverable is a reproducible training baseline and a capability report. The previous
long-context paper, mixture-selection funnels, tokenizer competition, and architecture sweeps are
historical. Longer context and a paper can follow measured needs and results.

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
3. **Train and inspect.** Start with one bounded pilot, proposed ceiling 50 GPU-hours and at most
   1B tokens. Use measured throughput to choose an attainable endpoint within both bounds. Inspect
   losses by source, gradient health, samples, checkpoint recovery, and cost. This is an engineering
   baseline, not a causal comparison or claim about architecture quality.
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
proposed at no more than 50. Allocate the remaining work after those measurements, rather than
maintaining speculative budgets for multiple research programs. No jobs are launched by this cleanup.

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

- Model/training/data/recovery implementations and CPU tests exist. GH200 feasibility is unmeasured.
- The first hardware configuration is [ready for dry-run binding](experiments/qualification/README.md).
- FineWeb's completed receipt reports 2,306,703,052 cached tokens with a passing reopen check.
  Stack-Edu reports 476,774,847 tokens before full exclusion; it is not final training stock and is
  not interchangeable with restricted Stack v3. Both finite preparation services exited successfully.
- Other retained sources include FineMath, Cosmopedia, peS2o, FineWiki, and approved natural
  UltraData-Math. Existing counts are not a globally deduplicated training union.
- Exact pilot data/weights, benchmark versions, site settings, and post-training recipe remain open.

**Next:** reconcile the retained source manifests into one pilot data configuration and pin a compact
capability evaluation set. Hardware qualification can proceed on synthetic inputs once access arrives.
Preparation procedures and artifact locations are in [data](docs/data.md); supporting evidence is in
[research notes](docs/research.md). Historical result bytes remain in [Git](archive/README.md).
