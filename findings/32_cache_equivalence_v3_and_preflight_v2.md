# 32 — Powered cache equivalence v3 and baseline preflight v2

## Question

Can a prospectively powered, disjoint common-history study resolve the inconclusive/adverse v2 cells,
and does the resulting evidence qualify the exact-shape Paper 1 baseline hardware path without erasing
the failed v1 and v2 records?

## Power and endpoint authority

V2 case-level paired variance was converted into a one-sided 5% alpha, 90% power plan under equal true
candidate/control behavior. Required source-balanced cases were:

- common-history token disagreement: at most 66;
- margin-at-least-0.1 disagreement: at most 88;
- JS divergence, top-10 overlap, and log relative RMS: 11; and
- five-point exact free-running divergence: up to 1,683.

V3 selects 88 cases per length, eight from each of 11 validation sources. This supplies 2,816 nested
common-history decisions per seed/length and gives a zero-event one-sided 95% upper rate of 0.00106.
All token windows are disjoint within v3 and from v2.

Free-running exact identity remains mandatory to report but loses pass/fail authority in the new
version. This is not because KDA failed v2. The estimand is unsuitable as the primary numerical gate:
one low-margin tie-break changes all subsequent conditioning, dense itself diverged in 15–27% of v2
cases, and adequate five-point non-inferiority would cost `19.125×` the powered common-history design.
Common-history comparison instead holds the model history fixed while measuring distributions,
rankings, and confident token decisions.

V2 remains failed and unchanged.

## Control-first execution

The new provenance-only case stream and power analysis were committed before the v3 contract. The
dense checkpoint then ran alone and was sealed in a new control lock before any v3 KDA result existed.

Dense behavior on the disjoint stream:

| Prompt | Token disagreement | Mean JS | Mean relative RMS | Mean top-10 overlap | Free divergence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.78% | 5.11e-5 | 0.00472 | 0.9900 | 18.18% |
| 64 | 0.43% | 4.82e-5 | 0.00474 | 0.9912 | 15.91% |
| 512 | 0.28% | 3.73e-5 | 0.00427 | 0.9925 | 11.36% |
| 4,096 | 0.57% | 4.05e-5 | 0.00440 | 0.9921 | 21.59% |

Dense has two margin-at-least-0.1 changes at length 8 and zero margin-at-least-0.5 changes.

## KDA v3 result

All three KDA checkpoints pass every primary endpoint at every length and the hard guardrail: 60/60
seed/length/endpoint cells and 12/12 length-level hard-guardrail cells.

Worst one-sided bounds across all KDA cells:

| Endpoint | Worst bound | Frozen threshold |
| --- | ---: | ---: |
| Token-disagreement increase | 0.00604 | ≤ 0.01000 |
| ≥0.1-margin disagreement increase | 0.00114 | ≤ 0.00200 |
| JS-divergence increase | 2.42e-6 | ≤ 1.00e-4 |
| Top-10 overlap decrease | 0.00249 | ≤ 0.02000 |
| Relative-RMS ratio | 1.0342 | ≤ 1.5000 |
| ≥0.5-margin changes | 0 | 0 |

Observed free-running rates remain visible without decision authority. Across cells, KDA spans
11.36–21.59% and dense spans 11.36–21.59%; the metric does not reveal a uniform KDA penalty on the
disjoint stream.

## Baseline preflight v2

The failed v1 preflight is preserved verbatim. A new preflight version requires:

- the original passing isolated KDA operator/state/gradient qualification;
- qualified cache-equivalence v3;
- exact-shape compiled forward, backward, clipped-gradient, and Muon update;
- the 16GiB peak-allocation envelope; and
- temporary CPU Transformers full/incremental/generation export parity.

Both exact materialized Paper 1 arms pass. Dense peaks at 10.24GiB and KDA at 14.14GiB. Both exports
pass. The old random-weight elementwise full-model result remains attached as a diagnostic and still
fails; it has no v2 pass/fail authority because the powered trained-topology contract replaced that
specific invalid endpoint, not the strict operator checks.

## Decision

**Cache-equivalence v3 qualifies and Paper 1 baseline preflight v2 passes.** This removes the CUDA
decode/preflight blocker from the baseline launch audit.

It does not rank the architectures, credit KDA, establish language quality, or authorize Paper 1
training by itself. The historical dense control has one seed and 2.08% more parameters; it is valid
for runtime calibration, not the planned quality comparison. SPE-58 evaluation-manifest closure and
SPE-104 storage-provenance closeout remain blockers before the six proxy runs.

## Artifacts

- [V3 power analysis](../results/Speck-Paper1/cache-equivalence-v3-power.json)
- [Disjoint v3 case stream](../results/Speck-Paper1/cache-equivalence-v3-cases.json)
- [V3 contract](../research/paper-1/cache_equivalence_v3.json)
- [V3 control lock](../results/Speck-Paper1/cache-equivalence-v3-control-lock.json)
- [V3 analysis](../results/Speck-Paper1/cache-equivalence-v3-analysis.json)
- [Baseline preflight v2](../results/Speck-Paper1/baseline-preflight-v2.json)
