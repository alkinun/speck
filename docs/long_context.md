# Long-context tooling

Report allocated, trained, effective, and usable context separately. Reserving state or completing a
passkey diagnostic does not establish robust long-document capability.

## Runtime

The model supports global/sliding GQA, full/partial/zero RoPE, recurrent GDN/KDA, and shared-cache
Reader Attention. Layers and reference recurrences live in `speck.model.layers`; persistent cache
state lives in `speck.model.state`.

Recurrent execution has readable Torch reference paths and optional FLA chunkwise/recurrent CUDA
paths. Run `scripts.gdn_kernel_qualify` and `scripts.kda_kernel_qualify` for each new hardware/software
combination. Prior [findings](../research/findings/README.md) refer to their original qualified code.

Global attention is quadratic. Sliding CUDA prefill uses FlexAttention block metadata, while decode
uses bounded state. INT8 KV storage currently dequantizes to the SDPA compute dtype.

## Progressive continuation

`scripts.context_stage_prepare` binds a new length stage to an exact parent checkpoint and data
manifest. `scripts.context_budget` reports analytic cost and state before a stage is materialized.
Evaluate original-4K retention after each continuation.

Use coherent books, papers, repository trees, or connected documents. The packed loader consumes a
flat stream; BOS/EOS markers do not themselves reset recurrence or isolate attention between examples.
The [context protocol](../research/flagship/CONTEXT_EXTENSION.md) specifies the planned source mixture
and stage budgets.

## Measurement and export

Use position/trailing loss, controlled retrieval/composition, RULER curves, and measured prefill/decode
cost together; see [evaluation](evaluation.md). Reader Attention is retained and tested, with its
negative result preserved in the archive.

Native/Transformers export is supported through a bundled native implementation and parity checks.
Recurrent GGUF, context parallelism, and production FP8 require additional implementation or hardware
qualification. See [current status](../research/status.json).
