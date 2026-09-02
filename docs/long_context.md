# Long-Context Research

SpeckLabs treats context length as a measured capability, not a configuration value. The research
stack is organized around four separately reported ceilings:

- **Allocated:** the largest state the runtime can reserve.
- **Trained:** the largest sequence present during pretraining or post-training.
- **Effective:** the longest evaluated sequence retaining the configured fraction of short-context
  quality.
- **Usable:** the longest sequence satisfying a named hardware, memory, and latency contract.

Never substitute one ceiling for another in a model card or benchmark comparison.

## Architecture

The primary candidate is a dense 3:1 Gated DeltaNet/GQA hybrid. Gated DeltaNet supplies a fixed
recurrent matrix per value head and an error-correcting delta update; a short causal convolution
provides local order. Periodic GQA layers remain an experimental variable because they improve
direct retrieval but make global prefill and training quadratic.

The implementation has three deliberately distinct paths:

1. `torch_gated_delta_rule` is the readable numerical reference and CPU path.
2. FLA chunkwise Gated DeltaNet is the required CUDA training/prefill path.
3. FLA fused recurrent Gated DeltaNet is the single-token decode path.

Run `scripts.gdn_kernel_qualify` on every new GPU/software combination. Kernel availability is not
treated as kernel correctness.

The checked [RTX 3090 qualification](../results/hardware/rtx3090-gdn-fla-0.5.0.json) binds FLA 0.5.0,
PyTorch 2.9.1+cu128, driver 610.43.03, tensor geometry, raw timings, tolerances, and clean Git source.
It passes output/final-state parity at 64, 512, and 4,096 tokens, all five input gradients, and
bitwise repeatability. This attests only that exact software/hardware path, not future cluster GPUs.

Attention supports full, partial, or zero RoPE dimensions. RoPE frequencies are retained, but
position tables are generated only for the active chunk. Global cached prefill uses a nonmaterialized
causal bias. CUDA sliding prefill uses FlexAttention with block-level mask metadata; at 128K and a
2K window the mask metadata occupies about 16 MiB instead of constructing a 16+ GiB token mask.
Sliding decode retains the bounded ring-buffer path. In mixed models, global and sliding layers use
separate rotary modules so global RoPE scaling does not compress local-window distances.

## Reference models

`SpeckLC-150M-GDN` is the inexpensive architecture and curriculum proxy: 20 layers, 15 Gated
DeltaNet mixers, five GQA layers, partial RoPE, 152,916,468 parameters, 4K core training, and a 128K
allocation ceiling.

`SpeckLC-1.2B` is a materialized target, not a claim of a trained release: 24 layers, 18 Gated
DeltaNet mixers, six two-KV-head GQA layers, partial RoPE, 1,218,451,776 parameters, and a 1M
allocation ceiling. At 1,048,576 tokens its portable INT8 KV state—including scales—is about 3.05
GiB; 4-bit weights plus all recurrent/KV state are below 4 GiB before runtime workspace. Global
attention compute, not resident state, is the dominant 1M risk.

## Experimental sequence

1. Materialize the mixer family with `scripts.mixer_ablation_prepare`.
2. Train token-matched short-context proxies and report the compute-matched view alongside them.
3. Qualify Torch/FLA outputs, states, gradients, determinism, and speed on the target hardware.
4. Run the built-in exact-length curve to catch positional, memory, retrieval, and latency failures.
5. Promote only candidates that survive independent upstream RULER, NoLiMa, and HELMET runs.
6. Prepare each progressive length stage with `scripts.context_stage_prepare`; never edit a resume
   contract to force a new length or dataset through it.
7. Re-run short quality evaluations at every promotion so context specialization cannot silently
   erase the model's basic language capability.

The built-in passkey diagnostic is intentionally labeled weak evidence. Its purpose is fast
regression detection and systems measurement. Effective-length claims intended for publication must
come from the harder upstream suites and include full per-length curves.

## 150M global-layer frontier

The checked 32K pilot continues every point from the same 131M-token `gdn-local` checkpoint on 32M
tokens of complete FineMath, peS2o, and Wikipedia documents of at least 16K tokens. Retrieval uses
paired counterfactual needles to cancel answer-token preferences. These are internal diagnostics,
not publication-grade RULER, NoLiMa, or HELMET results.

| Global layers | Placement | 32K val loss | 4K loss change | Effective retrieval | Detectable retrieval | BF16 / INT8 state @128K |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | none | 2.68892 | -0.00268 | 4K | 4K | 8.97 / 5.34 MiB |
| 1 | final (19) | 2.68697 | +0.00234 | 4K | 32K | 103.47 / 54.07 MiB |
| 1 | middle (11) | 2.65777 | +0.00396 | none | none | 103.47 / 54.07 MiB |
| 2 | middle + final | 2.65660 | +0.00903 | 4K | 32K | 197.97 / 102.79 MiB |
| 5 | every fourth layer | 2.63490 | +0.01568 | 16K | 32K | 481.47 / 248.97 MiB |

One middle global layer captures nearly all of the two-layer language-modeling gain but does not
surface retrieval at the output. One final global layer surfaces retrieval but barely improves
long-document loss. The two placements therefore play different roles. Five global layers retain
retrieval most robustly, at a steep state and short-context cost. The raw frontier is recorded in
`results/SpeckLC-150M-GlobalCount32K`.

## Data

Core pretraining may retain the existing quality mixture. Context-extension data should instead
contain dependencies that span the intended lengths: complete papers and books, deterministic
repository trees, hyperlink-connected pages, and synthetic retrieval/aggregation tasks. Use source
token-length filters to prevent a nominal long-data source from being dominated by short documents.

The loader retains BOS/EOS document boundaries but otherwise consumes a flat source stream.
Therefore, adjacency must be meaningful before packing. Concatenating unrelated web documents is a
useful stress condition, not sufficient long-context supervision.

## Known boundaries

- The 3:1 ratio and GDN are hypotheses to ablate, not settled SpeckLabs doctrine.
- Full global-attention layers remain quadratic. Use the budget report before every length stage and
  compare a local-attention hybrid where full prefill becomes uneconomic.
- INT8 KV currently dequantizes into the SDPA compute dtype. It proves capacity and measures quality,
  but a backend-native quantized kernel is needed for maximum decode throughput.
- Transformers export is supported through vendored native code and parity tests. GDN GGUF is not;
  the legacy converter deliberately accepts only the original conv hybrid.
- The trainer provides DDP and activation checkpointing. Context parallelism and production FP8 are
  hardware-stack projects and must not be claimed until their distributed checkpoint and numerical
  parity contracts exist.
