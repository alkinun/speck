# Evaluation and Benchmarking

Speck includes model-quality and systems-performance harnesses. Checked evaluation configurations
pin runner code, datasets, model identities, and expected checksums where the upstream interface
allows it.

## Parser-independent held-out contract

The flagship's bounded pre-results contract is
[`heldout_evaluation_plan_v1.json`](../research/flagship/heldout_evaluation_plan_v1.json). It applies
Magic's pretraining lesson that formatting and parser behavior are part of the measured distribution:
the exact production-formatted firewall output remains one view, while a separately stored alternate
extraction with a different hash-bound parser/extractor is a second view. The views are paired by
source ID and source-document SHA-256, scored separately, and never pooled. Parser-independent
equal-category BPB is the ranking view, while all six category guardrails must pass in both views.

Build a pre-results manifest with:

```bash
uv run --extra cpu python -m scripts.heldout_evaluation_build <config.json>
```

After separately producing complete baseline and candidate logprob reports, apply the frozen BPB
analysis without opening an audit:

```bash
uv run --extra cpu python -m scripts.heldout_evaluation_analyze \
  <manifest.json> <baseline-scores.json> <candidate-scores.json> --output <analysis.json>
```

The builder requires all six formal categories and reports subdomains beneath each category without
giving them separate decision weight. It binds production and alternate view hashes, parser artifact
identities, source/document hashes, and hash-only ledgers for aggregate training,
`selection_heldout`, `D5_tokenizer`, and `E2_mixture`. Those ledgers must be globally disjoint by
source-derived identity, source-document hash, and normalized-content hash; selection/audit ledgers
must reproduce the existing firewall's ordered content commitments without opening sealed payloads.

Two additional leakage channels fail closed: every sliding 96-character window after
NFKC/lower/whitespace normalization, and exhaustive exact token-shingle Jaccard verification inside
the declared pair bound. They do not replace, relax, or reinterpret the existing firewall, whose
global disjointness, equal-category, unopened-audit, and consumer checks must already pass.

Score reports bind model and backend identities and contain NLL plus the exact byte count for every
document in each view. Analysis preserves the existing unweighted six-category macro BPB and +0.01
BPB upper-95%-CI category guardrail. Subdomain values are diagnostic only. An optional helper checks
identical external-model logprobs across two distinct backend identities against a predeclared
per-document NLL tolerance.

Fixture manifests and score reports always retain false consumer, selection, training, and audit
opening authority. Real sizes, parsers, data, and model runs remain blocked pending a production
successor; fixture files cannot be passed to real consumers. This repository-local contract records
the operational lesson only; it does not add or reinterpret external evidence.

Run commands from the repository root. Model-quality evaluations require network access and may
require gated-dataset acceptance and Hugging Face authentication.

## Long-context curves

Long-context experiments contain `long_context.json`, which fixes evaluated lengths, needle
depths, samples per depth, and the effective-length threshold. Run the complete quality and systems
curve for a local checkpoint with:

```bash
uv run --extra gpu python -m scripts.long_context_eval \
  experiments/SpeckLC-150M-GDN --step <step>
```

The diagnostic creates prompt-plus-answer cases at exact total token lengths, scores the answer autoregressively without
retaining sequence-wide vocabulary logits. The access code is selected from ten single-token
candidates, so reports include controlled-choice accuracy, rank, probability, and margin alongside
open-vocabulary exact match, answer log probability, prefill/decode throughput, peak CUDA
allocation, and a state-memory split between attention KV and fixed recurrent state. It reports
effective length against both exact match and controlled-choice accuracy. A zero short-context
baseline produces no effective-length claim for that metric.

Use `--counterfactual` for retrieval claims. It pairs every case with an otherwise identical prompt
whose needle answer is changed, then tests whether the relative candidate scores move in the
matching direction. This removes stable answer-token preferences that can look like retrieval in
ordinary multiple-choice accuracy. A contrastive effective length is reported only when the
shortest-context directional result exceeds a one-sided binomial chance test at `p < 0.05`.

`kv_cache_dtype` may be `bfloat16`, `float16`, `float32`, or `int8`. The INT8 reference cache uses
per-token, per-head K/V scales and includes their bytes in the memory report. It is intended for
quality and capacity experiments; backend-native quantized attention kernels are still required
for maximum decode speed because the portable path dequantizes values before SDPA.

This built-in passkey task is a systems and literal-retrieval qualification, not evidence of robust
long-context reasoning. Release evaluation must additionally run the pinned upstream RULER suite.
Report its upstream revision and raw outputs separately rather than relabeling the built-in
diagnostic as that benchmark.

For every headline length, publish four distinct values: model allocation ceiling, maximum training
length, measured effective length, and maximum usable length under a named latency/memory contract.

Use explicit overrides for a cheap regression pilot before launching a complete configured curve:

```bash
uv run --extra gpu --extra linear python -m scripts.long_context_eval \
  experiments/SpeckLC-150M-GDN --step <step> \
  --lengths 4096,32768 --depths 0.5 --samples-per-depth 1 --warmup-each-length
```

Default outputs are namespaced by training run so evaluating multiple variants cannot overwrite
earlier reports. Reports also state whether global-attention layers are evaluated beyond their
training positions without RoPE scaling; treat such results as diagnostics, not fair comparisons.
Use the per-length warm-up when comparing prefill latency so one-time kernel compilation is not
charged to the first architecture or sample.

### Reusable architecture-promotion retrieval protocols

The reusable [architecture-promotion contract](../research/architecture-promotion-v1/) freezes two
200-case internal protocols. Pass one with `--protocol` to both
`scripts.structured_retrieval_adapt` and `scripts.structured_retrieval_eval`. Protocol mode overrides
all scientific CLI settings, verifies the protocol path and SHA-256 against the active evaluation
manifest, qualifies its answer and route vocabulary against the prepared tokenizer, and embeds the
identity in every checkpoint and report.

`structured_retrieval_v2` separates two- and eight-record load and uses disjoint two-token training
and validation answer sets. `symbolic_composition_v2` reports route, payload, and direct composition
separately over 100 tokenizer-qualified intermediate nodes. Protocol evaluation requires exactly one
`--protocol-length`: run 4K first, then 32K and 128K only after the paired current-length capability
gate passes.

### External long-context suites

The reusable manifest pins suite-specific contracts under
[`research/architecture-promotion-v1/external`](../research/architecture-promotion-v1/external/).
`scripts.external_suite_qualify` verifies the exact upstream commit and every required source-file hash
without installing the suite or downloading its data.

- RULER v2 uses the qualified `rulerv1-ns` source bundle and deterministic 4K–128K case matrices.
  The active disposition gives contaminated `qa_1` and `qa_2` zero primary weight; reports must retain
  their task-level outputs rather than silently dropping them.
- NoLiMa and HELMET contracts remain pinned for reference but are not part of the current evaluation
  plan: NoLiMa's license is academic-only and HELMET's runtime data, tokenizer, and judge dependencies
  could not be qualified. See the findings ledger for the audits.

Source qualification is not evaluation qualification. A suite contributes no promotion evidence until
its data, licenses, model adapter, raw outputs, and official scorer all pass and are hashed.

### Local external-evaluation endpoint

`scripts.evaluation_server` exposes an attested local Transformers export through the non-streaming
OpenAI chat and text-completion endpoints used for evaluation. It is a serialized correctness adapter,
not a production serving or throughput claim. The loader stays offline, refuses exports without a
successful `speck_parity.json`, and rejects decoding options that would be silently ignored.

Chat requests follow the tokenizer's serialization contract: an optional initial system message,
alternating user/assistant turns beginning and ending with a user, and no reserved chat tokens in
message content. Invalid requests return a client error before generation.

Seeds must be integers in PyTorch's supported range, `[-2**63, 2**64 - 1]`. Request fields use
their declared types: `n` must be the integer `1`, `stream` must be `false` or `null`, and
`logprobs` must be `false` or `null`. Boolean counts and numeric stand-ins for booleans are rejected.

Stop strings are applied to decoded output at the earliest matching position, independent of their
order in the request. `usage.completion_tokens` counts token IDs actually generated, including EOS
and tokens removed by output trimming. It does not re-tokenize the displayed text; stop-string
trimming currently happens after generation completes.

Export an instruction checkpoint and start the endpoint with:

```bash
uv run --extra cpu --group transformers python -m scripts.model_publish \
  --checkpoint-dir ~/.cache/speck/checkpoints/Speck2-140M-Instruct \
  --step <step> --repo specklabs/Speck2-140M-Instruct \
  --output-dir ~/.cache/speck/releases/Speck2-140M-Instruct-eval \
  --no-upload
uv run --extra cpu --group transformers python -m scripts.evaluation_server \
  ~/.cache/speck/releases/Speck2-140M-Instruct-eval \
  --device cpu --dtype bfloat16 --port 8000
```

Both base and SFT exporters attest full-sequence logits, incremental native-cache logits, parameter
count, and a Transformers `generate()` smoke. This gate specifically guards derived RoPE buffers and
Speck's nonstandard recurrent cache from generic Transformers loading behavior. The endpoint enforces
the export's configured context ceiling; qualifying the API at 4K does not qualify a candidate at 32K
or 128K.

### Offline cross-backend log-probability parity

Before using an optimized backend for R13/SPE-116 comparator measurements, compare backend-native
per-token log-probability exports with a declared trusted native or Hugging Face reference:

```bash
uv run --extra cpu python -m scripts.logprob_parity \
  research/flagship/logprob_parity_plan.json \
  <trusted-reference.json> <optimized-backend.json> [<optimized-backend.json> ...] \
  --output <new-report.json>
```

This is a local file comparator, not an evaluator or collector. It never contacts a server, installs
a backend, loads a model, or changes the evaluation endpoint above (which continues to reject
`logprobs=true`). Generate each input using a separately reviewed backend-native adapter. Inputs bind
an evaluation manifest, model, tokenizer, canonical request payloads, ordered case IDs, and scored
token IDs. Every payload and complete record set carries a SHA-256. Missing, non-finite, unhashed, or
misaligned values fail validation rather than becoming threshold failures.

Record schema version 1 is:

```text
format, format_version
producer: role, backend, revision, dtype, environment_sha256, synthetic_fixture
identity:
  evaluation: id, manifest_sha256
  model/tokenizer: id, revision, artifact_sha256
records[]:
  case_id, payload, payload_sha256
  tokens[]: position, token_id, logprob
records_sha256, payload_token_ids_sha256
```

Reports retain input hashes and give maximum and mean absolute error plus both provisional threshold
checks for every backend/dtype pair. Output paths are create-only to avoid replacing earlier evidence.
The checked plan and fixtures are pre-access diagnostics only: they do not qualify vLLM, SGLang,
generation/cache behavior, performance, model quality, or any external evaluation suite. Any real
comparison requires a successor plan frozen before its outputs, with exact backend revisions,
hardware/software identities, payload selection, and independently reviewed dtype thresholds.

### Local instruction-question scoring

`scripts.instruct_eval` evaluates the fixed 15-question instruction set with greedy decoding.
New reports include `method.scoring_version: 2`. For `final_number` questions, the scorer extracts
the last complete numeric token and compares its value using decimal arithmetic. It preserves
signs (including the Unicode minus), decimals, exponents, and comma-grouped thousands. For example,
`-45` and `4.45` do not match `45`, while `45.0` and `4.5e1` do.

The numeric exact-format tie-breaker additionally requires the response to contain only that
numeric token, apart from surrounding whitespace. Text questions retain their existing normalized
word-boundary or exact-text scoring. Reports without a scoring version use the historical scorer;
their numeric results must be rescored or rerun before comparing them with version 2.

## Open SLM Leaderboard

Run every stage in the checked Open SLM configuration:

```bash
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all
```

`experiments/Speck1-140M/open_slm.json` records the leaderboard formula provenance, model revision,
lm-eval harness revision, standard-task dataset revisions, and official ArithMark repositories and
file checksums. Output defaults to `~/.cache/speck/evaluations/open-slm/Speck1-140M`.

Run `lm-eval`, `arithmark-2`, `arithmark-3`, or `summary` separately to diagnose or resume stages.
Use `--limit 2` only with `lm-eval` for a smoke test. The evaluator refuses to run if a configured
Hub drift guard no longer matches the repository head; review and update the pin rather than
silently evaluating changed inputs. Full lm-eval reruns update a checksummed
`lm-eval/selected-result.json` pointer so the summary remains bound to one exact result.

Every stage also accepts `--local-model <export>`. Local exports must contain the successful
native/Transformers parity attestation produced by `scripts.base_checkpoint_export`. The default
output directory combines the export name and directory hash; explicit output directories are
also permanently bound to that hash. ArithMark and lm-eval selections, plus the final summary, are
therefore keyed to the local directory identity instead of a Hub revision.

ArithMark 2.0's verified official runner right-pads without disabling the model cache. The wrapper
leaves its scoring code unchanged and sets `model.config.use_cache=False` after loading.

Evaluate the public instruction releases against the same raw-continuation tasks without a chat
template:

```bash
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all \
  --config experiments/Speck1-140M-Instruct/open_slm.json
uv run --extra gpu --group open-slm python -m scripts.open_slm_eval all \
  --config experiments/Speck1.1-140M-Instruct/open_slm.json
```

The model-specific configurations inherit benchmark and dataset pins from the base config. Their
outputs use separate directories named after each Hub repository.

Pinned zero-shot results are checked in under `results/<model>/open_slm.json`:

| Model | HellaSwag | ARC-Easy | ARC-Challenge | PIQA | ArithMark 3.0 | Intelligence Index | ArithMark 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speck1-140M | 35.03 | 46.68 | 25.94 | 63.87 | 36.60 | 18.15 | 31.52 |
| Speck1-140M-Instruct | 35.22 | 45.66 | 25.85 | 63.60 | 36.10 | 17.75 | 33.64 |
| Speck1.1-140M-Instruct | 35.64 | 46.93 | 26.02 | 64.15 | 33.70 | 17.90 | 32.44 |

Do not update checked results without a complete pinned run and its provenance.

## Optimization Benchmark

Measure compiled optimization steps with synthetic input:

```bash
uv run --extra gpu python -m scripts.benchmark experiments/Speck1-140M \
  --mode compute \
  --output benchmark.json
```

Use `--mode end-to-end --data-dir <packed-data>` to include packed-data loading. Warmup is reported
separately. `--peak-tflops` reports model FLOPs utilization, and `--no-compile` measures eager
execution.

The optimization benchmark honors `train.json`'s `activation_checkpointing` setting and records the
resolved value. Use `--activation-checkpointing` or `--no-activation-checkpointing` for an explicit
paired runtime comparison without editing the experiment contract.

## BananaMind Base Bench 1.1

Accept the dataset gate and authenticate with Hugging Face before running BananaMind Base Bench
1.1:

```bash
uv run --extra gpu --group transformers python -m scripts.bananamind_bench \
  --model experiments/Speck1-140M \
  --speck-checkpoint-step 76294 \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 32
```

The wrapper verifies the official runner and data checksums, pins checkpoint and tokenizer hashes
in the report, and rejects resume when checkpoint or numerical settings change. Scoring remains in
the official runner. The `transformers` group is included because the runner requires
`transformers==5.1.0`.

## Inference Performance

Compare normalized prompt-prefill and cached-decoding speed:

```bash
uv run --extra cpu --group transformers python -m scripts.inference_benchmark \
  --model speck --device cpu
uv run --extra gpu --group transformers python -m scripts.inference_benchmark \
  --model speck --device cuda
```

Model aliases resolve as follows:

| Alias | Model |
| --- | --- |
| `speck` | `Speck1-140M` from a local experiment checkpoint. |
| `supra` | `SupraLabs/Supra2-100M-Base`. |
| `gptx` | `AxiomicLabs/GPT-X2.5-135M`. |
| `banana` | `BananaMind/BananaMind-2-Pro`. |
| `smol` | `HuggingFaceTB/SmolLM2-135M`. |

CPU defaults to FP32 batch 1; CUDA defaults to BF16 batches 1 and 32. The benchmark excludes
tokenization, uses eager SDPA, returns only the final-position logit, and records synchronized raw
timings. The Speck case uses the local configured checkpoint; external models must already exist in
the Hugging Face cache because loading is offline-only.

Use `--output <path>` to preserve the JSON report. Record the exact local Speck checkpoint with the
report because the current benchmark identifies it by experiment and step rather than a public
model snapshot.
