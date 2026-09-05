# 61 — Conditional five-cache GQA3/MQA1/MLA design

## Question

If the recurrent/global backbone eventually qualifies, can its five independent exact global
memories be compressed without repeating Reader Attention's route-edge failure or trading away
quality for cache savings?

## Controlled arms

The conditional design keeps all 15 KDA layers, five global positions (layers 3, 7, 11, 15, and 19),
five independent memories, twelve query heads, 64-dimensional heads, NoPE, standard residuals, and
dense SwiGLU fixed.

- **GQA3 control:** 153,958,938 parameters and 3,840 BF16 cache bytes per token across five memories.
- **MQA1:** one KV head per memory. Removing 983,040 attention parameters and adding 21 units to every
  FFN restores 967,680, leaving only 15,360 fewer parameters (0.00998%). State falls to 1,280 bytes per
  token, a 66.7% reduction, using the existing attention implementation.
- **NoPE MLA128:** one shared 128-element KV latent per memory, conventional uncompressed queries, and
  twelve expanded content heads. The latent dimension matches MQA1's BF16 state. Its 768×128 down
  projection plus two 128×768 up projections also exactly matches GQA3's K/V projection-weight count;
  replacing the 64-element key norm with a 128-element latent norm leaves a preliminary 320-parameter
  excess. Exact counts remain blocked on implementation.

At 128K context and batch one, GQA3's five caches occupy 503,316,480 BF16 bytes versus 167,772,160 for
MQA1 or MLA128. At 1M, the corresponding values are 4,026,531,840 and 1,342,177,280 bytes. These are
analytic state counts, not realized throughput evidence.

## Why MLA128 and NoPE

The dimension is chosen from Speck's geometry, not copied from DeepSeek: it creates a state-matched
MQA comparison and projection-parameter-matched GQA comparison simultaneously. NoPE avoids coupling
the cache representation question to MLA's decoupled positional-key path. Query compression is also
excluded so the first MLA experiment changes only KV representation.

MLA requires both an expansion-based reference and an optimized absorbed-projection path. Forward,
gradient, state, prefill, incremental decode, generation, checkpoint, export, workspace, and exact
cache accounting must pass before training. The optimized state may not retain expanded K/V tensors.

## Decision boundary

Training is not authorized. The current whole-architecture proxy and the ordered sequence-backbone
finalist must pass first. Later confirmation tests MQA1 and MLA128 against the shared GQA3 control with
Holm correction. Quality and correctness are constraints: MQA1 must realize at least a 10% primary
systems improvement, MLA at least 20%, and both at least 25% state reduction. Analytic savings cannot
promote either arm.

## Artifact

- [Conditional cache-representation design](../research/paper-1/sequence_cache_representation_v1.json)
