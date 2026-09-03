# 08 — Global-layer count and placement frontier

## Question

How little global attention buys long-document modeling and content-addressable retrieval, and does
layer placement matter independently of count?

## Controlled design

Every point starts from the exact same `gdn-local` parent checkpoint:

- Model SHA-256: `ae692777c0f3603261a935c7c56c5a05be8d9768e85eb4eddd444236e5fe46e1`
- Parent step: 2,000 at 131,072,000 tokens
- Parent 4K validation loss: `2.8105483`

Five sliding-attention positions exist at logical layers 3, 7, 11, 15, and 19. Selected layers are
promoted from sliding to global without changing parameter shapes. Global layers use 8× RoPE;
remaining sliding layers retain 1× local RoPE. All points use the same 32M-token long-document data,
schedule, seed, optimizer state, Liger loss, activation checkpointing, and batch geometry described
in [05](05_long_document_dataset.md) and [06](06_context32k_local_vs_global.md).

## Runtime qualification

Every mixed graph completed compiled forward, backward, gradient clipping, and Muon update with a
finite loss before training began.

| Variant | Synthetic tok/s | Peak allocated |
| --- | ---: | ---: |
| `global-1` final | 18,247.5 | 4.64 GiB |
| `global-1-mid` | 14,851.4 | 4.64 GiB |
| `global-2` | 16,692.7 | 4.64 GiB |
| `global-5` | 15,446.5 | 4.64 GiB |

These very short preflights prove execution and memory, not stable throughput. After the compiler
cache was cleared, the first one-step `global-1` measurement spilled compilation into the measured
step and reported only 3,233 tok/s; a two-step rerun produced 18,247.5 tok/s. Actual full-run
throughput, reported below, is authoritative. The middle-only preflight was not rerun, which
explains its artificially low value relative to its later full run.

## Training and short-context results

| Variant | Global layers | Initial 32K loss | Final 32K loss | Change | tok/s | GPU-h | Δ original 4K loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global-0` | none | 2.80261 | 2.688916 | -0.11369 | 25,861 | 0.3446 | -0.00268 |
| `global-1` | 19 | 2.81241 | 2.686968 | -0.12544 | 24,160 | 0.3722 | +0.00234 |
| `global-1-mid` | 11 | 2.79487 | 2.657770 | -0.13710 | 24,117 | 0.3654 | +0.00396 |
| `global-2` | 11, 19 | 2.80954 | 2.656605 | -0.15294 | 22,636 | 0.3898 | +0.00903 |
| `global-5` | 3, 7, 11, 15, 19 | 2.85583 | 2.634904 | -0.22093 | 19,043 | 0.4621 | +0.01568 |

The initial values are measured immediately after changing attention scope and RoPE, before the
first continuation update. They are not comparable as pretrained model rankings; the final values
after identical continuation are the experimental endpoint.

Using the `0.00965`-nat seed range:

- `global-0` versus final-only `global-1` differs by `0.00195`: unresolved.
- middle-only `global-1-mid` beats `global-0` by `0.03115`: larger than noise.
- `global-2` beats middle-only by only `0.00117`: unresolved on loss.
- `global-5` beats `global-2` by `0.02170`: larger than the measured seed range, but still a
  one-seed frontier result.

## Retrieval results

Directional accuracy and mean paired score:

| Variant | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `global-0` | 76.7% / 0.0902 | 50.0% / 0.0164 | 43.3% / -0.0008 | 46.7% / -0.0016 | 36.7% / -0.0012 | 30.0% / -0.0008 |
| `global-1` final | 96.7% / 0.5938 | 76.7% / 0.2156 | 66.7% / 0.0931 | 73.3% / 0.0174 | 56.7% / 0.0031 | 50.0% / 0.0003 |
| `global-1-mid` | 56.7% / 0.0469 | 46.7% / 0.0034 | 33.3% / -0.0012 | 26.7% / -0.0017 | 30.0% / -0.0043 | 36.7% / -0.0031 |
| `global-2` | 100.0% / 0.3742 | 76.7% / 0.1060 | 63.3% / 0.0354 | 66.7% / 0.0118 | 53.3% / 0.0020 | 50.0% / 0.0000 |
| `global-5` | 93.3% / 0.1777 | 86.7% / 0.0538 | 83.3% / 0.0281 | 70.0% / 0.0134 | 50.0% / 0.0029 | 63.3% / 0.0040 |

| Variant | 4K directional p-value | Effective retrieval | Longest detectable retrieval |
| --- | ---: | ---: | ---: |
| `global-0` | 0.00261 | 4K | 4K |
| `global-1` final | 2.89e-8 | 4K | 32K |
| `global-1-mid` | 0.292 | none | none |
| `global-2` | 9.31e-10 | 4K | 32K |
| `global-5` | 4.34e-7 | 16K | 32K |

The final-only layer has the largest 4K retrieval score but loses relative strength rapidly. The
middle-only layer improves next-token loss but does not create significant output-side retrieval.
The two-layer model combines the two roles. Five distributed global layers give the only 16K
85%-retention result.

## State and latency frontier

| Variant | BF16 state @128K | INT8 KV state @128K | 128K prefill |
| --- | ---: | ---: | ---: |
| `global-0` | 8.97 MiB | 5.34 MiB | 1.375 s |
| `global-1` final | 103.47 MiB | 54.07 MiB | 1.755 s |
| `global-1-mid` | 103.47 MiB | 54.07 MiB | 1.756 s |
| `global-2` | 197.97 MiB | 102.79 MiB | 2.131 s |
| `global-5` | 481.47 MiB | 248.97 MiB | 3.262 s |

INT8 figures include per-token, per-head K/V scales. They are allocation results, not a quantized
decode-quality result. The current counterfactual answer is scored from prefill logits and cannot
validate multi-token INT8 cache decoding.

## Interpretation

Global attention layers are not interchangeable:

- **Middle global layer:** integrates long-document information and lowers language-modeling loss.
- **Final global layer:** makes distant content directly accessible to the output logits.
- **Middle + final:** the smallest architecture containing both observed roles.
- **Distributed five-layer pattern:** best loss and strongest retention, with the largest state,
  compute, and short-context penalty.

Practical Pareto choices depend on the objective:

- Cheapest detectable 32K retrieval: one final global layer, about 54 MiB INT8 state at 128K.
- Cheapest strong 32K language modeling: one middle global layer, same state, but no demonstrated
  retrieval.
- Smallest model combining both roles: two global layers, about 103 MiB INT8 state at 128K.
- Best measured loss and retention: five global layers, about 249 MiB INT8 state at 128K.

## Checkpoint identities

| Variant | Model SHA-256 | W&B run |
| --- | --- | --- |
| `global-0` | `4af4f6023ea6bfbb5e1771e62dc18846b32492d5c73aafc82ce8d666cdaf2ebc` | `ssxkmv23` |
| `global-1` | `e83b069ca0e088df3656dbc8ced689210522aee3bd5a1d73b5f5d3d45da19f08` | `3mmvxyar` |
| `global-1-mid` | `a945428a0076d657f2a6610a439094837ec4c779bbb14a037cbd1cb57b9fe47d` | `1mx2q4w3` |
| `global-2` | `c74fb56c5405d967783f60040b8933680b54f7255568cfba66d35e84dfd2cf18` | `eqlm4zzw` |
| `global-5` | `ce1862f63373f5a47f323c9a619288454d2d8aa3cd3425fd0949cf0669bd1efb` | `9cstb9l6` |

Full model, optimizer, and metadata hashes are preserved in the checked summary.

## Artifacts

- Experiment contracts:
  [experiments/SpeckLC-150M-GlobalCount32K](../experiments/SpeckLC-150M-GlobalCount32K)
- Consolidated result:
  [results/SpeckLC-150M-GlobalCount32K/summary.json](../results/SpeckLC-150M-GlobalCount32K/summary.json)
- Raw preflights, short-loss evaluations, and 720 new paired long-context cases:
  [results/SpeckLC-150M-GlobalCount32K](../results/SpeckLC-150M-GlobalCount32K)
- The 180 `global-0` pairs are stored with the first 32K comparison:
  [results/SpeckLC-150M-Context32K/retrieval/gdn-local.json](../results/SpeckLC-150M-Context32K/retrieval/gdn-local.json)
