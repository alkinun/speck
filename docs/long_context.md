# Long-Context Tooling

The experiment ledger, including negative results, checkpoint identities, and decisions, is in
[findings/README.md](../findings/README.md). Checked JSON under `results/` is the machine-readable
source of truth. The current research scope is the [flagship document](../research/flagship/README.md).
This page describes the long-context machinery the repository provides.

SpeckLabs treats context length as a measured capability, not a configuration value. Report four
ceilings separately and never substitute one for another in a model card or comparison:

- **Allocated:** the largest state the runtime can reserve.
- **Trained:** the largest sequence present during pretraining or post-training.
- **Effective:** the longest evaluated sequence retaining the configured fraction of short-context
  quality.
- **Usable:** the longest sequence satisfying a named hardware, memory, and latency contract.

## Mixers

The recurrent mixers each have three deliberately distinct paths:

1. A readable Torch recurrence as the numerical reference and CPU path.
2. FLA chunkwise kernels as the CUDA training and prefill path.
3. FLA fused recurrent kernels as the single-token decode path.

Run `scripts.gdn_kernel_qualify` and `scripts.kda_kernel_qualify` on every new GPU and software
combination. Kernel availability is not kernel correctness. The checked
[RTX 3090 qualification](../results/hardware/rtx3090-gdn-fla-0.5.0.json) binds FLA 0.5.0, PyTorch
2.9.1+cu128, driver 610.43.03, tensor geometry, raw timings, tolerances, and clean Git source; it
attests only that exact path, not other hardware.

Gated DeltaNet uses scalar per-head decay. Kimi Delta Attention uses channel-wise decay and is the
lead recurrent mixer: on calibrated MQAR it matches SiLU-gated GDN at length 1,024 and passes 3/3
seeds against 1/3 at length 2,048 (findings [13](../findings/13_synthetic_mqar.md) and
[14](../findings/14_mqar_length_scaling.md)). The sigmoid output gate improves the 131M-token
language loss by 0.037 nats over SiLU (finding [16](../findings/16_kimi_transfer_131m.md)).

## Attention

Attention supports full, partial, or zero RoPE dimensions. RoPE frequencies are retained, but
position tables are generated only for the active chunk. Global cached prefill uses a nonmaterialized
causal bias. CUDA sliding prefill uses FlexAttention with block-level mask metadata; at 128K and a
2K window the mask metadata occupies about 16 MiB instead of a 16+ GiB token mask. Sliding decode
uses a bounded ring buffer. In mixed models, global and sliding layers use separate rotary modules so
global RoPE scaling does not compress local-window distances.

NoPE global layers trained from the base stage retain a directional needle signal through 128K
after 4K training, where RoPE layers fail at 4K (findings [16](../findings/16_kimi_transfer_131m.md)
to [18](../findings/18_kimi_context32k.md)). Converting a trained RoPE checkpoint to NoPE late
damages both long and short loss (finding [12](../findings/12_nope_context_activation.md)); train
NoPE from the start.

Speck Reader Attention lets query-only reader layers share a writer layer's key-value cache. The
grammar is `memory` and `memory_role` on the attention spec, validated across the execution plan.
The mechanism is implemented and tested but not promoted: see
[finding 24](../findings/24_reader_attention.md).

## Global-layer roles

From the same-parent 32K frontier (finding [08](../findings/08_global_attention_frontier.md)): a
middle global layer lowers long-document loss but does not surface retrieval at the output, a final
global layer surfaces retrieval but barely improves loss, and five distributed global layers give the
best loss and retention at the largest state cost. Global KV cache is about 99.7% of the 128K
resident state of a five-layer 150M hybrid, which is why the number and representation of global
caches is the state axis that matters.

## Progressive context training

The flagship's approximate source mixtures, coherent-window preparation, token targets, and 200-hour
continuation envelope are in [the context-extension plan](../research/flagship/CONTEXT_EXTENSION.md).
The plan is not evidence that long-document data or exact-shape GH200 training is already qualified.

Prepare each length stage with `scripts.context_stage_prepare`. A stage binds an exact parent
checkpoint and packed long-document dataset by hash; never edit a resume contract to force a new
length or dataset through it. Re-run the original short-context evaluation at every stage so context
specialization cannot silently erase basic language capability.

Extension data must contain dependencies that span the intended lengths: complete papers and books,
deterministic repository trees, connected pages, and synthetic retrieval or aggregation tasks. Use
source token-length filters so a nominal long source is not dominated by short documents. The loader
retains BOS/EOS boundaries but consumes a flat stream, so adjacency must be meaningful before packing;
concatenating unrelated documents is a stress condition, not long-context supervision.

## Evaluation

- `scripts.long_context_eval` runs the built-in exact-length passkey curve. It is a fast regression
  and systems diagnostic, not a capability claim.
- `scripts.position_loss_eval` reports position-binned and trailing-token loss.
- `scripts.structured_retrieval_adapt` and `scripts.structured_retrieval_eval` run the frozen
  200-case internal protocols under `research/architecture-promotion-v1/internal`.
- RULER v2 through `scripts.ruler_source_prepare`, `scripts.ruler_case_prepare`, and the local
  evaluation endpoint. See [Evaluation](evaluation.md).

Effective-length claims intended for publication come from RULER and the internal protocols with
full per-length curves, plus original-4K retention.

## Known boundaries

- Full global attention remains quadratic. Use `scripts.context_budget` before every length stage.
- INT8 KV currently dequantizes into the SDPA compute dtype. It proves capacity and measures
  quality; a backend-native quantized kernel is needed for maximum decode throughput.
- Transformers export is supported through vendored native code and parity tests. Recurrent-mixer
  GGUF export is not; the legacy converter accepts only the original conv hybrid.
- The trainer provides DDP and activation checkpointing. Context parallelism and production FP8 are
  hardware-stack projects and must not be claimed until their parity contracts exist.
