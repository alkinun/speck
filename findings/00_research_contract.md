# 00 — Research contract

## Objective

The research question is not “which sequence mixer wins at 4K?” It is:

> How little global attention is required to preserve content-addressable retrieval, and what
> resident-state and prefill-compute price does each global layer impose?

The working architectural hypothesis is that Gated DeltaNet and sliding attention should perform
cheap local and recurrent processing, while a small number of strategically placed global layers
perform integration and retrieval.

## Four separate context ceilings

Every report must distinguish:

1. **Allocated:** the maximum context for which the runtime can allocate state.
2. **Trained:** the maximum sequence length actually used during training.
3. **Effective:** the longest independently evaluated length that retains the declared quality
   threshold.
4. **Usable:** the longest length satisfying an explicit latency and memory contract on named
   hardware.

Configuring `max_position_embeddings=131072` proves only allocation permission. It does not prove
training, retrieval, reasoning, or usable latency at 128K.

## Evidence hierarchy

From weakest to strongest:

1. Analytic FLOPs and state accounting.
2. Synthetic kernel and systems preflight.
3. Built-in literal or counterfactual passkey regression tests.
4. Held-out long-document language-modeling loss.
5. Original short-context validation after promotion.
6. Independent RULER, NoLiMa, and HELMET evaluation.
7. Replication over seeds and data orders.

No publication-grade effective-length claim is allowed from levels 1–4 alone. The current work
reaches levels 4–5 and an internal counterfactual diagnostic; it deliberately stops before a 128K
promotion claim.

## Promotion gates

- Establish seed noise before interpreting close one-seed loss differences.
- Require exact parent checkpoint and packed-data hashes for every continuation.
- Apply RoPE scaling to global attention beyond trained positions; do not pretend the resulting
  short-context trade-off is free.
- Re-run original 4K loss after every context extension.
- Use complete long documents or explicit long-dependency tasks. Concatenated unrelated web pages
  are a stress condition, not long-context supervision.
- Run independent RULER, NoLiMa, and HELMET before promoting a model for release.
- Stop a length promotion when data does not contain dependencies at the target length.

## Fixed hardware and software

- GPU: NVIDIA GeForce RTX 3090, 24 GiB
- Driver: 610.43.03
- PyTorch: 2.9.1+cu128
- CUDA runtime: 12.8
- Triton: 3.5.1
- flash-linear-attention: 0.5.0
- Main training dtype: bfloat16
- Optimizer: Muon plus AdamW parameter roles
- Default context-extension loss: Liger fused linear cross entropy

The FLA Gated DeltaNet qualification is checked at
[results/hardware/rtx3090-gdn-fla-0.5.0.json](../results/hardware/rtx3090-gdn-fla-0.5.0.json).

## Current terminal boundary

The completed work establishes a 32K, same-parent global-layer frontier. It does not start 128K
training because:

- the derived corpus contains documents down to 16K and up to 86K, not enough genuine 128K
  dependencies;
- all frontier points are near the counterfactual noise floor by 128K;
- open-vocabulary passkey exact match is zero;
- independent upstream promotion suites have not passed.

This is a scientific stop, not an infrastructure failure.
