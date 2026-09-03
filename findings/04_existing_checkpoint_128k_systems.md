# 04 — Existing-checkpoint 128K systems frontier

## Question

Can all six 4K-trained checkpoints execute exact total lengths through 128K, and how do prefill
latency, resident state, and peak allocation scale before any context extension?

This was a systems and regression pilot, not a capability comparison. Each length used one midpoint
needle after an unmeasured per-length warm-up. Global models used their unscaled 4K-trained RoPE at
longer positions, so their quality outputs are positionally confounded.

## Evaluator corrections required first

- Default result paths were namespaced by run; previously all variants overwrote `2000.json`.
- “Length” was defined as exact prompt-plus-scored-answer length so a 128K case did not overrun a
  128K state during answer decoding.
- CLI pilot overrides were added for lengths, depths, and sample counts.
- Per-length warm-up separated compilation from steady prefill time.
- Reports identify global RoPE extrapolation.

## Results

| Variant | 4K prefill | 32K prefill | 128K prefill | State @128K | Peak @128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure-gdn` | 0.0472 s | 0.3075 s | 1.3829 s | 1.96 MiB | 2,901.80 MiB |
| `gdn-local` | 0.0486 s | 0.3137 s | 1.3666 s | 8.97 MiB | 2,937.11 MiB |
| `full-local` | 0.0526 s | 0.3271 s | 1.3031 s | 30.00 MiB | 2,959.80 MiB |
| `conv-global` | 0.0399 s | 0.3450 s | 2.8822 s | 480.03 MiB | 3,389.88 MiB |
| `gdn-global` | 0.0487 s | 0.4124 s | 3.2011 s | 481.47 MiB | 3,391.94 MiB |
| `full-global` | 0.0527 s | 0.7239 s | 8.6513 s | 1,920.00 MiB | 4,832.77 MiB |

At 4K the matched local/global models are effectively identical. At 128K:

- `gdn-local` is 2.34× faster than `gdn-global` and uses 53.66× less resident state.
- `full-local` is 6.64× faster than `full-global` and uses 64× less resident state.
- Peak allocation differs less dramatically because weights, activations, and common runtime
  workspaces dominate the single-prefill measurement.

## Capability result

None. The initial random six-digit answer required seven tokens and yielded zero exact match. A
single-token controlled-choice revision also had no significant 4K signal in this small pilot.
Therefore the run established execution and systems scaling only. Retrieval methodology was fixed
later; see [07 — Counterfactual retrieval](07_counterfactual_retrieval.md).

## Artifacts

- Checked summary:
  [results/SpeckLC-150M-MixerScreen-131M/long-context-systems/summary.json](../results/SpeckLC-150M-MixerScreen-131M/long-context-systems/summary.json)
- The same directory contains one raw JSON report per variant.
- Evaluator hardening commits: `e878582`, `332bc60`
- Block-mask scalability and boundary commits: `f015c0d`, `e88ee32`
- Result commit: `84bcc2f`
