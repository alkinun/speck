# Evaluation

The [method](program.md#method) sets what is measured at each scale. Ladder runs are scored
mainly by held-out loss per source and domain on fixed family-disjoint validation packs, plus an
early-signal suite of log-likelihood multiple choice and arithmetic (`scripts.open_slm_eval`:
HellaSwag, ARC-Easy, ARC-Challenge, PIQA and ArithMark). Parent branches add generative benchmarks
after the [SFT probe](program.md#sft-probe-100-gpu-hours), whose assistant is scored as a probe of
its base checkpoint. Each rung's seed-to-seed spread sets the smallest effect it can report.
Stage-to-stage changes describe progression; attributing a gain to data requires a predeclared
contrast with the model, training exposure and other recipe settings held fixed. Keep cost
alongside quality.

## Available checks

```bash
uv run --no-sync python -m scripts.checkpoint_loss_eval PATH_TO_EXPERIMENT \
  --checkpoint-dir CHECKPOINT_DIRECTORY --eval-tokens 65536 --no-compile
```

This reports held-out loss, with per-source diagnostics.
`instruct_eval` supplies small local assistant diagnostics; these are not a complete benchmark suite.
`open_slm_eval` integrates pinned external benchmark tooling in a separate `open-slm` environment.
Every script accepts `--help`.

Use `benchmark` for optimization cost, `inference_benchmark` for prefill/decode measurements,
`evaluation_server` for a local export endpoint, and `logprob_parity` for backend comparison.
Report hardware, precision, batch, sequence/output lengths, startup, steady throughput, and memory.

## Quality and cost evidence

Report source-wise loss and capability against processed tokens and all-in GPU-hours, with unique
data and replay disclosed. Compare cost to reach a declared quality target and show absolute quality.
Equal-token, equal-FLOP and equal-wall-time comparisons answer different questions; do not treat
accuracy divided by cost as a universal efficiency score.

For FLOPs, declare the counting convention and included operations. The coarse 6*N*D reference
does not fully account for global attention, recurrent/chunk work, vocabulary projection or
recomputation. Separate model FLOP estimates from hardware time and measured kernel work.

For inference, separate prefill and decode, count reasoning and final-answer tokens, and disclose
context length, output caps and stopping policy. SFT supervised tokens, processed context, padding
and generated tokens are distinct quantities. Report teacher/verification costs separately.

Each family's [predeclared record](program.md#method) fixes its primary metric, validation packs,
minimum useful effect and decision rule. Family-level uncertainty and fixed-suffix context scoring
need qualified retained outputs; the aggregate loss evaluator does not supply those analyses.

## Benchmark protocol

For each experiment family, pin evaluation inputs and graders before training. Keep development and
final-test data separate from training and from each other. Verify final math answers and execute
code in an isolated resource-limited runner. Report correctness and failures with denominators,
output budgets, latency and cost. Compare base with base and assistant with assistant. Re-run public
baselines under the same declared protocol; published leaderboard numbers are context, not directly
comparable measurements.

The benchmark dashboard is the early-signal suite on every run plus generative math and coding
benchmarks on parent branches after the probe; this document owns its contents.

## Coding evaluations

The pilot's compiled HumanEval+ metric is a continuity check; 33 development tasks cannot establish
broad coding strength. These are candidate generative evaluations for parent branches after the
probe; none is implemented or scored:

| Dimension | Candidate evidence |
| --- | --- |
| Python generation | HumanEval+ continuity plus MBPP+ through a declared [EvalPlus](https://github.com/evalplus/evalplus) protocol |
| Language coverage | Selected [MultiPL-E](https://github.com/nuprl/MultiPL-E) languages, reporting each separately |
| Harder generation | A fixed release/date window of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), with explicit test variant and output limits |
| Practical use | Bounded held-out tasks for library use and debugging, scored against independent hidden tests |

Freeze task-family partitions and exclusion identities before data selection, fit prompts and output
within the context budget, pin Python/library versions, and re-run reference models under the same
protocol. Report denominators, uncertainty, execution failures, output tokens and runtime; do not
tune against the final partition or equate the compiled metric with the official leaderboard.

## Pilot protocol

The engineering pilot's frozen protocol pins GSM8K, IFEval, HumanEval+, ARC-Challenge and HellaSwag
subsets with a seeded development/final split and a sandboxed code grader. Its preparation,
qualification and runner commands are in the [pilot record](../experiments/pilot/README.md#evaluation-protocol).
