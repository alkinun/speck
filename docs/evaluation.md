# Evaluation

Use the capability table in [PLAN.md](../PLAN.md) as the reporting outline. Measure math, coding,
tools, reliability, and broad usefulness separately; keep cost alongside quality.

## Available checks

```bash
uv run --no-sync python -m scripts.checkpoint_loss_eval PATH_TO_EXPERIMENT \
  --checkpoint-dir CHECKPOINT_DIRECTORY --eval-tokens 65536 --no-compile
```

This reports held-out loss, with per-source diagnostics. `speck evaluate` is equivalent.
`instruct_eval` supplies small local assistant diagnostics; these are not a complete benchmark suite.
`open_slm_eval` integrates pinned external benchmark tooling in a separate `open-slm` environment.
Every script accepts `--help`.

Use `benchmark` for optimization cost, `inference_benchmark` for prefill/decode measurements,
`evaluation_server` for a local export endpoint, and `logprob_parity` for backend comparison.
Report hardware, precision, batch, sequence/output lengths, startup, steady throughput, and memory.

## Before the pilot

Pin a compact evaluation set and graders before training. Keep development and final-test data
separate from training and from each other. Verify final math answers, execute code in an isolated
resource-limited runner, and evaluate tools in a deterministic environment. Include missing
information, malformed calls, tool failures, corrections, and cases where no tool should be called.

Report correctness and failures with denominators, output budgets, latency, and cost. Compare base
with base and assistant with assistant. Re-run public baselines under the same declared protocol;
published leaderboard numbers are context, not directly comparable measurements.

The full math/code/tool/reliability dashboard remains work to do. Candidate references and the reasons
for them are in [research notes](research.md). Historical retrieval/long-context experiments are in
[Git](../archive/README.md), outside the current experiment path.

The first pilot pins GSM8K, IFEval, HumanEval+, ARC-Challenge, and HellaSwag in
[its protocol](../experiments/pilot/evaluation.json), including dataset revisions and file hashes.
Prepare the exact inputs with:

```bash
uv run --no-sync python -m scripts.evaluation_prepare experiments/pilot/evaluation.json \
  --output /external/pilot/evaluation.json
```

This command verifies task counts and assigns approximately 20% to development and 80% to final
by a seeded hash of the normalized prompt. Identical prompts share a partition. It produces task
identities, not evaluation scores. These are custom subsets; full-benchmark leaderboard scores are
not directly comparable. Near-duplicate task families across the two partitions remain a limitation.
All benchmark inputs, including both partitions, are excluded from the pilot candidates using the
existing exact-field and informative n-gram scanner. Sensitivity matches are also removed.
The protocol pins lm-evaluation-harness and EvalPlus source revisions. Grader execution, a deterministic
tool environment, compatible comparator revisions, and output-budget qualification still need work.
