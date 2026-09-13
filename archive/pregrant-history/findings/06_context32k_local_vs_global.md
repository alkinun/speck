# 06 — Matched 32K local/global continuation

## Question

After actual 32K continuation on complete long documents, does the short-context local/global tie
survive, and what capability is bought by five global attention layers?

## Recipe

- Parent checkpoints: completed 131M-token `gdn-local` and `gdn-global` runs
- Sequence length: 32,768
- Continuation request: 32,000,000 tokens
- Consumed continuation: 32,047,104 tokens, 489 optimizer steps
- Batch tokens: 65,536, device batch 1, accumulation 2
- LR: `1e-4`, 25-step warm-up, cosine decay to 10%
- Loss: Liger fused linear cross entropy
- Activation checkpointing: enabled
- Data: `SpeckLC-LongDocs-32M`, exact manifest `0a14833…`
- Local RoPE: 1×
- Global RoPE: fixed 8× linear scaling
- Seed: 42

Both compiled preflights passed at about 4.64 GiB peak allocation. Synthetic preflight throughput
was 18.6k tok/s local and 15.3k tok/s global.

## Training results

| Variant | Initial 32K long-doc loss | Final loss | Change | Actual tok/s | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gdn-local` | 2.80261 | 2.688916 | -0.11369 | 25,861.4 | 0.3446 |
| `gdn-global` | 2.82779 | 2.639523 | -0.18827 | 19,011.6 | 0.4644 |

Final held-out source losses:

| Variant | FineMath 4+ | peS2o | Wikimedia |
| --- | ---: | ---: | ---: |
| `gdn-local` | 2.50886 | 3.01752 | 2.60039 |
| `gdn-global` | 2.49958 | 2.90573 | 2.55990 |

Global finishes `0.04939` nats ahead overall and improves on every held-out long-document source.
This gap is about five times the measured seed range, although the pair began from different
short-context parent checkpoints.

## Original 4K regression gate

Each promoted checkpoint was evaluated over 19,988,480 tokens from the original 11-source corpus.

| Variant | Parent 4K loss | Promoted 4K loss | Change |
| --- | ---: | ---: | ---: |
| `gdn-local` | 2.810548 | 2.807867 | -0.00268 |
| `gdn-global` | 2.819378 | 2.835424 | +0.01605 |

Local passes with a small improvement. Global loses more than the measured seed range.

## RoPE diagnostic

| Checkpoint | Evaluation scale | 4K loss |
| --- | ---: | ---: |
| Parent global | native 1× | 2.819378 |
| Parent global | forced 8× | 2.876823 |
| Promoted global | trained 8× | 2.835424 |
| Promoted global | forced back to 1× | 2.865541 |

Changing the unextended parent to 8× immediately costs `0.05745` nats. Training recovers about 72%
of that penalty, but the learned weights adapt to compressed positions and no longer work well when
simply switched back to 1×. RoPE scaling is a training/deployment contract, not a free evaluation
toggle.

## Counterfactual retrieval

| Length | Local directional accuracy / score | Global directional accuracy / score |
| ---: | ---: | ---: |
| 4K | 76.7% / 0.0902 | 100.0% / 0.2629 |
| 8K | 50.0% / 0.0164 | 100.0% / 0.1456 |
| 16K | 43.3% / -0.0008 | 93.3% / 0.0628 |
| 32K | 46.7% / -0.0016 | 70.0% / 0.0192 |
| 64K | 36.7% / -0.0012 | 70.0% / 0.0207 |
| 128K | 30.0% / -0.0008 | 63.3% / 0.0032 |

- Local effective retrieval: 4K.
- Global effective retrieval: 16K.
- Global remains directionally detectable through 64K in this different-parent comparison, but the
  score is small and non-monotonic by then.
- Neither model supports a 128K capability claim.

At 128K, local prefill is 2.37× faster and its resident state is 53.66× smaller.

## Checkpoints and artifacts

- Local model SHA-256: `4af4f6023ea6bfbb5e1771e62dc18846b32492d5c73aafc82ce8d666cdaf2ebc`
- Global model SHA-256: `f9eff7288b755a8308fddb94e179f16cf1d8707745bcdc1d1c76b5bf09437dd9`
- Checkpoints: `~/.cache/speck/checkpoints/SpeckLC-150M-Context32K-gdn-{local,global}`
- Experiment: [experiments/SpeckLC-150M-Context32K](../experiments/SpeckLC-150M-Context32K)
- Checked results: [results/SpeckLC-150M-Context32K](../results/SpeckLC-150M-Context32K)
- Consolidated summary:
  [results/SpeckLC-150M-Context32K/summary.json](../results/SpeckLC-150M-Context32K/summary.json)
