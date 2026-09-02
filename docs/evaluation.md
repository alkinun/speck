# Evaluation and Benchmarking

Speck includes model-quality and systems-performance harnesses. Checked evaluation configurations
pin runner code, datasets, model identities, and expected checksums where the upstream interface
allows it.

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
long-context reasoning. Release evaluation must additionally run pinned upstream RULER, NoLiMa,
and HELMET suites. Report their upstream revisions and raw outputs separately rather than relabeling
the built-in diagnostic as one of those benchmarks.

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
