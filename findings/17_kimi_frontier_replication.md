# 17 — Three-seed Kimi-frontier replication

## Question

Does the seed-42 tie between GDN/sigmoid/RoPE and KDA/sigmoid/NoPE survive initialization
variance, and is KDA's 128K counterfactual sensitivity reproducible?

Finding [16](16_kimi_transfer_131m.md) selected two points from the one-intervention staircase:

- GDN with FLA timescales, sigmoid output gate, and partial RoPE: best short-context loss control;
- KDA with channel-wise decay, sigmoid output gate, and NoPE: position-free treatment.

Seeds 43 and 44 change only model/training randomness, the run name, and W&B group. Architecture,
packed-data order, optimizer, schedule, batch geometry, sequence length, and 131,072,000-token
horizon are identical. Each final loss uses the same 19,988,480-token validation sample.

## Final language-model loss

| Seed | GDN/sigmoid/RoPE | KDA/sigmoid/NoPE | KDA minus GDN | Old 0.00965 tie gate |
| ---: | ---: | ---: | ---: | --- |
| 42 | 2.790629 | 2.795380 | +0.004751 | pass |
| 43 | 2.789735 | 2.793794 | +0.004060 | pass |
| 44 | 2.789039 | 2.800956 | +0.011917 | **fail** |
| mean | **2.789801** | 2.796710 | +0.006909 | — |

The RoPE control is exceptionally stable: range 0.001589 and population standard deviation
0.000651. KDA's range is 0.007162 and population standard deviation 0.003071, both still below the
old baseline's 0.00965 range but materially wider than the new control.

The mean paired penalty is 0.006909 nats. With only three pairs, its 95% Student-t interval is
`[-0.003898, 0.017716]`, spanning zero. Aggregate loss therefore cannot rank the models, but the
predeclared strict per-seed tie gate passes only 2/3 because seed 44 is 0.011917 worse. We preserve
both statements; averaging must not erase the failed seed.

KDA also shows higher early optimization variance. At 31.98M tokens its validation losses are
3.69230, 3.76750, and 3.76819. Seeds 43 and 44 converge toward seed 42 by halfway, but seed 44 keeps
a small final deficit. The architecture is stable—no NaNs or throughput collapse—but less
initialization-insensitive than GDN/RoPE under this recipe.

## Replicated retrieval endpoints

All seeds use the same 30 paired factual/counterfactual cases per endpoint across depths 0.1, 0.5,
and 0.9. Seed 42 comes from the full six-length curve; seeds 43 and 44 repeat the 4K and 128K
endpoints selected before observing their results.

### Contrastive score

| Model | Seed | 4K | 128K | 128K directions |
| --- | ---: | ---: | ---: | ---: |
| GDN/RoPE | 42 | −0.0047 | 0.0049 | 15/30 |
| GDN/RoPE | 43 | 0.0525 | 0.0070 | 19/30 |
| GDN/RoPE | 44 | 0.0309 | −0.0023 | 9/30 |
| KDA/NoPE | 42 | 1.7065 | 0.1448 | **30/30** |
| KDA/NoPE | 43 | 0.3018 | 0.1426 | **30/30** |
| KDA/NoPE | 44 | 0.6801 | 0.1534 | **30/30** |

KDA's 4K score varies substantially, but the 128K scores are remarkably stable: mean 0.146918,
population standard deviation 0.004661, and range 0.010807. All three KDA checkpoints pass the
directional 128K gate. None of the RoPE checkpoints does; their mean 128K score is 0.003212.

Seed 43 is a useful control against a simplistic “RoPE never learned the needle” story. Its 4K
directional result is significant (29/30, score 0.0525), yet it falls to 19/30 and score 0.0070 at
128K. KDA seed 43 instead remains 30/30 with score 0.1426.

Exact match remains zero for every checkpoint and endpoint. Candidate accuracy is 23.3% throughout,
except KDA seed 44 at 4K reaches 26.7%. These differences do not establish usable retrieval. The
replicated result is narrower: NoPE KDA preserves a robust causal signal at 128K that RoPE GDN does
not.

## Systems replication

Full-training throughput is stable by architecture:

| Model | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| GDN/RoPE tok/s | 47,802 | 47,825 | 47,784 |
| KDA/NoPE tok/s | 44,384 | 44,314 | 44,322 |

KDA costs about 7.3% realized training throughput and roughly 0.06 GPU-hours per 131M-token run.
Both still use exactly 504,860,160 bytes (481.47 MiB) of resident BF16 state at 128K because
channel-wise decay changes parameters and updates, not the recurrent state shape or global KV
geometry.

## Decision

Advance the pair to one controlled 32K context-activation stage on the seed-42 checkpoints:

- GDN/sigmoid/RoPE remains the capability control;
- KDA/sigmoid/NoPE is the long-context treatment.

This is a research promotion, not a release selection. The justification is the combination of an
unresolved mean loss difference and a 3/3 replicated 128K sensitivity advantage. The seed-44 loss
miss means the treatment must retain mandatory original-4K evaluation and cannot be called
loss-equivalent without qualification.

Use the same long-document data and token horizon as the prior 32K stages. Do not add RoPE scaling
to the NoPE treatment. Rank the continuation on long-document loss, original-4K regression,
retrieval score, prefill cost, and resident state—not long loss alone.

## Artifacts

- Experiment contracts:
  [experiments/SpeckLC-150M-KimiReplication131M](../experiments/SpeckLC-150M-KimiReplication131M)
- Machine-readable result:
  [results/SpeckLC-150M-KimiReplication131M/summary.json](../results/SpeckLC-150M-KimiReplication131M/summary.json)
- Endpoint records:
  [results/SpeckLC-150M-KimiReplication131M/retrieval](../results/SpeckLC-150M-KimiReplication131M/retrieval)
- Repeat W&B runs: `5v91iscd`, `k0h3vww8`, `aetzonew`, and `not9lyf2`
