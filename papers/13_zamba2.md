# The Zamba2 Suite

- **Paper:** [arXiv:2411.15242](https://arxiv.org/pdf/2411.15242)
- **Version reviewed:** v1, 22 November 2024
- **Primary topic:** shared attention blocks inside a Mamba-2 backbone

## Central claim

Zamba2 obtains most of the quality benefit of a Mamba/attention hybrid without allocating a distinct
attention block at every invocation. One or two Transformer blocks are reused repeatedly through depth;
small per-call LoRA adapters let the shared block specialize to its location.

This is a parameter-efficiency technique, not a KV-cache sharing technique. Every invocation still
produces layer-specific activations that must be cached for autoregressive decoding unless an additional
mechanism shares them.

## Architecture

The backbone uses Mamba-2. The 2.7B and 7.4B models alternate between two shared attention blocks,
reducing the correlation and optimization burden observed when one block is reused everywhere. The 1.2B
model keeps one shared block because a second offered less value at its smaller number of invocations.

Non-shared LoRA adapters can be attached to the shared attention, shared MLP, or both. The released 1.2B
model includes LoRA on both; the larger training runs began before that ablation was complete and do not.

The models train at base length 4,096. Zamba2-1.2B and 2.7B see 3T tokens, while 7.4B sees 2T. The paper
uses a 100B-token annealing phase with 60% replay of phase-one data to reduce forgetting.

## Evidence

- The architecture search reports Mamba-2 at materially higher throughput than Mamba-1 at matched
  performance, creating budget for periodic attention.
- Alternating two shared blocks improves over repeatedly using one at the larger sizes. Per-invocation
  LoRA recovers depth-specific expressivity at low parameter cost.
- The paper describes an invocation ratio of roughly one attention call per six Mamba-2 layers, hence an
  approximately `6×` KV-cache reduction relative to a comparable all-attention stack. This assumes the
  same per-call KV geometry and should be converted to actual bytes for Speck.
- Zamba2-1.2B reports MMLU `43.1` and competitive commonsense scores against larger small-model
  baselines; 2.7B and 7B variants likewise perform strongly in their comparison sets. Training data and
  token counts differ across external baselines.
- Without additional training, loss remains useful beyond 4K and the authors report an effective region
  around 17K after RoPE scaling. A curriculum that doubles length every 100 steps from 4K to 65,536
  produces accurate passkey retrieval to 65K.

## What matters for Speck

At small scale, global attention parameters can be a significant fraction of the model even if cache is the
main long-context cost. Shared global blocks with call-specific LoRA provide a clean way to spend more
depth on KDA/GDN or MLP capacity while retaining several retrieval opportunities.

A Speck experiment should distinguish:

- one shared global block with no adapter;
- one shared block with per-call LoRA;
- two alternating shared blocks;
- independent global blocks at the same invocation positions.

Match both total parameters and active FLOPs. Report whether cache bytes actually change; weight sharing
alone should not be credited with a per-token cache reduction beyond the small change in architecture.

## Limitations and cautions

- The larger released models omit the LoRA improvement discovered later, so the suite is not a perfectly
  crossed ablation.
- Shared weights can reduce model capacity or cause conflicting gradients even when per-call adapters are
  present.
- The headline long-context evidence relies heavily on passkey and loss; it does not establish complex
  65K reasoning.
- The paper's `6×` cache statement follows fewer attention invocations, not reuse of one KV tensor across
  all invocations.

## Bottom line

Zamba2 is a useful parameter-saving hybrid pattern: reuse expensive global-attention weights while
letting each depth specialize through a small adapter. For Speck it is most attractive if global-layer
parameters, rather than only KV state, constrain the 1.2B design.
