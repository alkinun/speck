# 15 — Palindrome and 64-stack mixer qualification

## Question

Does KDA's advantage generalize beyond associative lookup to exact sequence copying and mutable
state tracking?

## Controlled protocol

Both tasks use the two-layer, two-head, head-dimension-128 synthetic architecture from
[13](13_synthetic_mqar.md), sequence length 1,024, vocabulary 8,192, batch size 8, fixed 256-example
validation, seed-42 LR grid `{5e-5, 1e-4, 5e-4, 1e-3}`, 20K-step ceiling, and 99% top-1 gate.

Compared variants:

- KDA with channel-wise decay and sigmoid output gate;
- scalar GDN with FLA-style timescale initialization and SiLU output gate, the strongest MQAR
  control.

Palindrome inputs contain 512 random tokens, separator token 0, then their exact reversal. Only the
512 reversed tokens are supervised. Stack inputs interleave PUSH and valid POP operations across 64
independent stacks; only each POP answer is supervised. Every generated Stack trace is replayed in
unit tests to prove LIFO correctness.

## Palindrome discovery grid

| Variant | `5e-5` | `1e-4` | `5e-4` | `1e-3` |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 6.07% | 99.004% at 16,500 | **99.005% at 10,750** | 88.60% |
| GDN-SiLU | 0.22% | **93.05%** | 0.018% | 0.60% |

KDA has two passing rates; GDN has none. The LR response is again non-monotonic. KDA at `1e-3`
learns a low-loss partial solution but plateaus near 88.6%, while `5e-4` crosses the exact-copy gate.

Palindrome gradients are much noisier than MQAR. Pre-clipping norms occasionally reach hundreds,
and validation accuracy can temporarily regress before recovering. Gradient clipping remains fixed
at 1.0 for all variants and rates.

### Three-seed confirmation

KDA's fastest rate, `5e-4`, was repeated. Because KDA then missed one seed, the near-gate GDN
`1e-4` control was also repeated symmetrically rather than comparing 2/3 against 0/1.

| Variant | Seed 42 | Seed 43 | Seed 44 | Passes |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 10,750 | 9,000 | fail, best 98.15% | 2/3 |
| GDN-SiLU | fail, best 93.05% | fail, best 73.67% | fail, best 0.22% | 0/3 |

KDA is decisively stronger, but it does not satisfy a strict 3/3 exact-copy gate. The seed-44 KDA
run learns a strong solution and plateaus just below threshold; the gate is retained rather than
relaxed after observing the result.

## 64-stack discovery grid

First step reaching 99%:

| Variant | `5e-5` | `1e-4` | `5e-4` | `1e-3` |
| --- | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 6,750 | 2,500 | 750 | **500** |
| GDN-SiLU | 7,250 | 3,250 | **500** | **500** |

Both mixers solve Stack across the entire LR grid. Following the declared higher-LR tie break,
`1e-3` was selected for both.

### Three-seed confirmation

| Variant | Seed 42 | Seed 43 | Seed 44 | Passes | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| KDA-sigmoid | 500 | 500 | 250 | 3/3 | 500 |
| GDN-SiLU | 500 | 500 | 500 | 3/3 | 500 |

There is no meaningful architecture advantage at the 250-step evaluation resolution. Mutable LIFO
state is easy for both delta-rule mixers; KDA's extra decay granularity is not required here.

## Combined interpretation

The three synthetic families separate different capabilities:

| Task | KDA result | Strongest GDN result | Interpretation |
| --- | --- | --- | --- |
| MQAR 1,024/32 | 3/3 | 3/3 SiLU; 0/3 sigmoid | similar median, KDA lower tail variance and rescues sigmoid |
| MQAR 2,048 endpoints | 3/3 on both | 1/3 on both | KDA substantially more reliable under length/load stress |
| Palindrome | 2/3 | 0/3 | KDA stronger at exact copying, but not fully reliable |
| 64-stack | 3/3 | 3/3 | tied; scalar GDN is sufficient |

KDA is not a universal replacement that wins every task and seed. Its advantage concentrates where
finite associative memory must retain many identities or reconstruct a long ordered sequence. That
is precisely the capability relevant to our long-context research question. The Stack tie is a
useful negative result because it rules out a generic optimization-speed explanation.

## Decision

Promote KDA to a **language-model discovery staircase**, alongside the existing GDN baseline. Do
not call it the selected release mixer yet:

- it has robust MQAR evidence and a strong Palindrome advantage;
- it misses one of three strict Palindrome seeds;
- its RTX 3090 training kernel is slower than GDN despite similar analytic FLOPs;
- synthetic tasks do not measure natural-language generalization.

The first language-model staircase must isolate recurrent initialization, output gate, global NoPE,
and KDA rather than bundle them into a single comparison. Only effects larger than the measured
4K seed floor should advance to repeat seeds and 32K context training.

## Artifacts

- Palindrome runs:
  [results/SpeckLC-SyntheticMemory/palindrome-1024](../results/SpeckLC-SyntheticMemory/palindrome-1024)
- Stack runs:
  [results/SpeckLC-SyntheticMemory/stack64-1024](../results/SpeckLC-SyntheticMemory/stack64-1024)
- Consolidated synthetic result:
  [results/SpeckLC-SyntheticMemory/summary.json](../results/SpeckLC-SyntheticMemory/summary.json)
