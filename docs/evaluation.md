# Evaluation

The [method](program.md#method) sets what is measured at each scale. Ladder runs are scored by
held-out loss per source on fixed family-disjoint validation packs and by an early-signal suite of
log-likelihood multiple choice and arithmetic. Parent branches add generative benchmarks after the
[SFT probe](program.md#sft-probe-100-gpu-hours). Each rung's seed-to-seed spread sets the smallest
effect it can report. Each family's [record](../experiments/ladder/records) fixes its primary
metric, validation packs, minimum effect and decision rule before it runs. Final test partitions
are scored once, at the end.

## Held-out loss and early signal

```bash
uv run --no-sync python -m scripts.checkpoint_loss_eval PATH_TO_EXPERIMENT \
  --checkpoint-dir CHECKPOINT_DIRECTORY --eval-tokens 65536 --no-compile
```

This reports held-out loss with per-source diagnostics. `scripts.open_slm_eval` runs the
early-signal suite (HellaSwag, ARC-Easy, ARC-Challenge, PIQA and ArithMark) through pinned external
tooling in a separate `open-slm` environment.

## Generative evaluation

`scripts.evaluation_prepare` and `scripts.capability_eval` run a pinned generative protocol: dataset
revisions, file hashes and a seeded development/final split by normalized prompt. The pilot's
[evaluation.json](../experiments/pilot/evaluation.json) pins GSM8K, IFEval, HumanEval+, ARC-Challenge
and HellaSwag. Grading uses the `capability` dependency group in its own environment
(`UV_PROJECT_ENVIRONMENT=.venv-capability uv sync --extra gpu --group capability`).

```bash
python -m scripts.capability_eval PROTOCOL PREPARED --qualify --output /external/grader-check
python -m scripts.capability_eval PROTOCOL PREPARED --local-export EXPORT_DIR --limit 0 \
  --output /external/scores
```

`--qualify` checks every grader against canonical and deliberately wrong answers, timeouts and
sandbox isolation. Code runs under Bubblewrap with no network and a non-root account, with no
unsandboxed fallback. `--local-export` requires a passing export parity receipt. `--partition final`
is an explicit held-out action.

The parent-branch suite is still to be frozen. Candidates are GSM8K and other checked math, and for
code, HumanEval+ and MBPP+ through a declared [EvalPlus](https://github.com/evalplus/evalplus)
protocol, selected [MultiPL-E](https://github.com/nuprl/MultiPL-E) languages, and a fixed
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) window. Freeze task partitions and
exclusion identities before data selection, and re-run reference models under the same protocol.

## Protocol rules

- Keep development and final data separate from training and from each other.
- Report correctness with denominators, output budgets, execution failures, latency and cost.
- Compare base with base and assistant with assistant. Published leaderboard numbers are context,
  not comparable measurements.
- Report quality against processed tokens and all-in GPU-hours, with unique data and replay
  disclosed. Declare the FLOP convention; 6ND omits attention, recurrence and recomputation.
- For inference, separate prefill and decode and count reasoning and answer tokens.

`scripts.benchmark` measures training cost and `scripts.inference_benchmark` prefill and decode.
