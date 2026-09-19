# Evaluation

Use the capability table in [PLAN.md](../PLAN.md#what-success-means) as the reporting outline. Measure math, coding,
tools, reliability, and broad usefulness separately; keep cost alongside quality.
Evaluate the [training lifecycle](program.md#training-lifecycle) with identified pretraining,
mid-training, SFT and any RL checkpoints. Stage-to-stage changes describe progression; attributing
a gain to data requires a controlled comparison with the model, training exposure and other recipe
settings held fixed. The primary planned control is the
[pretraining recipe study](coding.md#first-comparison-to-prepare), before main pretraining;
code-data substitution is one candidate contrast after source qualification.

The final assistant target always uses the thinking protocol for coding, math and agent tasks.
Evaluate brief/deep reasoning budgets within that protocol, including cap exhaustion, correctness,
tool loops and end-to-end task completion. A concise final answer does not mean reasoning is off.
This future assistant contract does not change the frozen base pilot or external reference-model
protocols below. A thinking tag alone is not evidence of useful reasoning.

Coding is a first-release priority. The [coding evaluation roadmap](coding.md#evidence-for-a-coding-claim)
adds Python breadth, multilingual checks, and practical repair to prepare after the engineering
pilot. Those additions need their own frozen protocol and training exclusions; they are not yet
implemented, and the pilot's 33 development code tasks do not establish broad coding strength.

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

## Quality and cost evidence

Report source-wise loss and capability against processed tokens and all-in GPU-hours, with unique
data and replay disclosed. Compare cost to reach a declared quality target and show absolute quality.
Equal-token, equal-FLOP and equal-wall-time comparisons answer different questions; do not treat
accuracy divided by cost as a universal efficiency score.

For FLOPs, declare the counting convention and included operations. The coarse 6*N*D reference
does not fully account for global attention, recurrent/chunk work, vocabulary projection or
recomputation. Separate model FLOP estimates from hardware time and measured kernel work.

For inference, separate prefill and decode, count reasoning and final-answer tokens, and disclose
context length, tool access, output caps and stopping policy. Agent cost per success includes failed
attempts and environment costs alongside success rate. SFT supervised tokens, processed context,
padding and RL rollout tokens are distinct quantities. Report teacher/verification costs separately.
Attention/size analysis describes the selected model's behavior; architectural superiority and
parameter scaling are outside the claims supported by this fixed-model program.

## Frozen pilot protocol and future evaluations

The pilot development evaluation and isolated code grading are complete: [results](../experiments/pilot/development-result.json)
cover 2,619 tasks; final tests remain untouched. The preparation commands below describe the frozen
protocol, not unfinished pilot work. For each new training study, pin evaluation inputs and graders
before training. Keep development and final-test data
separate from training and from each other. Verify final math answers, execute code in an isolated
resource-limited runner, and evaluate tools in a deterministic environment. Include missing
information, malformed calls, tool failures, corrections, and cases where no tool should be called.

Report correctness and failures with denominators, output budgets, latency, and cost. Compare base
with base and assistant with assistant. Re-run public baselines under the same declared protocol;
published leaderboard numbers are context, not directly comparable measurements.

The broader math/code/tool/reliability dashboard remains work to do. Candidate references and the reasons
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
The protocol pins lm-evaluation-harness and EvalPlus source revisions. The `capability` dependency
group installs those exact commits plus IFEval's optional dependencies. Keep this environment separate
from training (`UV_PROJECT_ENVIRONMENT=.venv-capability uv sync --extra gpu --group capability`).
IFEval requires NLTK's `punkt_tab` resource; acquire it before an offline run. Bubblewrap (`bwrap`) and
working Linux user namespaces and a non-root grading account are required for code execution.
The process-count limit does not apply to root, so root execution is rejected. There is no
unsandboxed fallback.

```bash
python -m scripts.capability_eval experiments/pilot/evaluation.json /external/pilot/evaluation.json \
  --qualify --output /external/grader-check
python -m scripts.capability_eval experiments/pilot/evaluation.json /external/pilot/evaluation.json \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --chat --limit 8 --output /external/reference-smoke
python -m scripts.capability_eval experiments/pilot/evaluation.json /external/pilot/evaluation.json \
  --local-export /external/pilot/base-export --limit 8 --output /external/pilot-base-smoke
```

`--local-export` replaces the Hub reference, requires a passing native/Transformers parity receipt,
hashes the complete export, and loads its explicitly selected local model/tokenizer code offline.
Use only an export whose code you intend to execute. Add `--chat` for an assistant export. The same
frozen tasks, output caps, non-root code sandbox, and partition rules apply to local checkpoints.
Local likelihood scoring disables the model's default generation cache so all continuation logits
are returned; HFLM explicitly enables caching for generation. The runner rejects an empty decoded
EOS stop string. Re-export older checkpoints with the current tokenizer: control tokens must retain
their spelling when special-token skipping is disabled. Base generation suppresses the model's
reserved assistant rows, which have no base-tokenizer pieces; assistant exports retain their roles.

Qualification checks the pinned GSM8K strict/flexible extraction, IFEval constraints, both
multiple-choice scorers, all 33 development code tasks' canonical solutions, deliberate wrong
answers, timeout/early-exit handling, and filesystem/network isolation. Five scripted
[tool episodes](assistant.md) check the deterministic tool environment separately.

The model runner reuses pinned task prompts, filters, scoring, aggregation, and continuation
log-likelihoods. It verifies frozen input hashes, selects only the declared partition, records raw
responses and task IDs, and rejects prompts that exceed the 4K context plus output budget.
Greedy output caps are 1,024 tokens for GSM8K/code and 512 for IFEval. `--chat` applies the model's
chat template with `enable_thinking=False`; omit it for a base comparator. Scores from chat and
plain completion protocols must be labeled separately. The frozen `enable_thinking=False` chat
path is not the future always-thinking assistant evaluation protocol. Prepare and qualify an explicit
thinking/tool-aware evaluator before assistant comparisons; preserve this historical base/reference
protocol and its results. `--limit 0` runs the full selected partition;
the default eight examples per benchmark is only an integration check. `--partition final` is an
explicit held-out evaluation action, not part of development qualification.

The frozen Hugging Face HumanEval+ file contains compiled `check(candidate)` programs, rather
than EvalPlus's `base_input`/`plus_input` representation. The runner uses the pinned EvalPlus code
sanitizer and executes those exact compiled tests with a 15-second wall deadline, 10-second CPU
limit, 4 GiB address-space limit, read-only Python/runtime mounts, no host home/corpus mounts, and
no network namespace access. It reports `compiled_plus_pass@1`, retaining execution failures in
the denominator. This is not the adaptive per-test timing protocol of the standard EvalPlus CLI;
do not compare it directly with published leaderboard scores. OS isolation reduces the consequences
of faulty generated programs; these are correctness benchmarks, not an adversarial grading system.
