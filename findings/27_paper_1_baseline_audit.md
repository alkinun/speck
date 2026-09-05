# 27 — Paper 1 matched-baseline audit and launch contract

## Question

Can the existing 131M-token runs serve as the conventional dense and conservative hybrid baselines
for Paper 1, and what exactly must be rerun before any sequence, depth, or width component is credited?

## Historical evidence audit

Five completed seed-42 checkpoints were rehashed from model, optimizer, and metadata files and checked
against their experiment configurations, packed-data manifest, tokenizer revision, training geometry,
validation loss, and mixer topology. Together they occupy 6,605,141,453 bytes.

| Arm | Parameters | FLOP/token at 4K | Final validation loss | Valid use |
| --- | ---: | ---: | ---: | --- |
| Dense global GQA | 157,156,608 | 1.3203B | 2.826207 | historical whole-architecture context |
| SWA, window 2,048 | 157,156,608 | 1.2259B | 2.822668 | historical bounded-attention context |
| GDN/SiLU/RoPE + 5 global GQA | 152,916,468 | 1.0194B | 2.819378 | historical mixer-screen control |
| GDN/sigmoid/NoPE + 5 global GQA | 152,916,468 | 1.0194B | 2.814996 | one-seed KDA predecessor |
| KDA/sigmoid/NoPE + 5 global GQA | 153,958,938 | 1.0216B | 2.795380 | conservative hybrid context |

The historical dense-minus-KDA loss difference is `0.030827` nats, but dense attention uses 2.08%
more parameters and 29.24% more analytic FLOPs per 4K token, while positional treatment and mixer
layout also differ. It is therefore not a component estimate. The closer GDN/NoPE-to-KDA/NoPE step
improves loss by `0.019616` nats with 0.68% more parameters and 0.21% more FLOPs, but it still uses only
one seed and one packed-stream order. The later three-seed replication held that order fixed. None of
these results has promotion authority under `architecture-promotion-v1`.

## New primary baseline pair

The checked matrix materializes two whole-architecture controls:

| Arm | Geometry | Parameters | FLOP/token at 4K |
| --- | --- | ---: | ---: |
| `dense_global_param_match` | 20 global-GQA layers, partial RoPE, uniform SwiGLU width 2,235 | 153,977,088 | 1.3012B |
| `five_cache_kda_gqa` | 15 KDA layers, five global GQA layers, sigmoid/NoPE | 153,958,938 | 1.0216B |

The dense model's FFN width is reduced uniformly from 2,304 to 2,235. Its 18,150-parameter excess is
0.0118%, below the preregistered 0.025% tolerance; uniform width was preferred over an artificial mix
of layer-specific FFN widths that could get 282 parameters closer.

This pair deliberately estimates a whole-architecture baseline difference. It cannot assign the
effect to KDA, NoPE, mixer ratio, or attention count. Those claims require the separate single-axis
sequence experiments in the Paper 1 program.

## Replication and matching views

The proxy confirmation contains three paired cells. Candidate and control share every packed-data
window, seed, tokenizer, optimizer, global batch, schedule, and evaluation sample within a cell:

| Pair | Initialization seed | Packed-stream start | Packed-stream end |
| ---: | ---: | ---: | ---: |
| 0 | 42 | 0 | 131,072,000 |
| 1 | 43 | 536,870,912 | 667,942,912 |
| 2 | 44 | 1,073,741,824 | 1,204,813,824 |

All offsets are batch-aligned, all windows are disjoint, and all remain in mixture phase zero. Base
training now records `data_token_offset` as immutable checkpoint state and reconstructs the exact
loader cursor, making these data-order cells executable and resumable.

Results must be reported four ways:

1. token matched at 131,072,000 tokens per arm;
2. parameter matched under the 0.025% tolerance;
3. compute matched at 131,072,000 KDA tokens versus 102,891,520 dense tokens;
4. validation loss against optimizer time, including right-censored time-to-quality thresholds.

The three pairs estimate paired variance but remain a non-promotion proxy stage. A later finalist stage
crosses three seeds with two data orders and trains every arm for 1,539,833,856 tokens, just over ten
tokens per parameter. That stage is not materialized until proxy confirmation passes.

## Storage audit

At audit time the research filesystem had 7,085,318,144 bytes (6.60GiB) free. The proxy launch floor
is 16GiB, leaving a 10,094,551,040-byte (9.40GiB) deficit. Six retained proxy checkpoints are estimated
at 8.4GB. The future twelve-run finalist design is estimated at 16.8GB before exports and evaluation
artifacts.

The largest current consumers are checkpoints (185.4GB), packed data (21.4GB), releases (4.5GB),
search artifacts (4.4GB), evaluations (4.2GB), and GGUF artifacts (3.2GB). No file was deleted. The
contract forbids automatic cleanup; archival targets require explicit review because checkpoints and
optimizer state are part of the evidence chain.

## Decision

The historical evidence qualifies as reproducible discovery context only. The new primary pair and
three paired seed/data-order cells are materialized but **not authorized to launch**. Remaining gates
are:

- close the SPE-58 evaluation-manifest dependency for this stage;
- provision at least 16GiB free storage;
- run paired compiled forward/backward/optimizer, export, and incremental-generation preflights on the
  named RTX 3090 environment; and
- freeze the paired analysis and stopping implementation before observing results.

No baseline ranking, architecture promotion, component contribution, or Paper 1 performance claim is
made by this audit.

## Artifacts

- [Baseline matrix](../research/paper-1/baseline_matrix.json)
- [Materialized experiments](../experiments/Speck-Paper1-Baselines-131M)
- [Machine-readable audit](../results/Speck-Paper1/baseline-audit.json)
- [Architecture promotion policy](../research/architecture-promotion-v1/policy.json)
