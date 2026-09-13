# 31 — Control-first CUDA cache-equivalence v2

## Question

Can a behavior-linked, source-balanced contract qualify the historical five-cache KDA/GQA CUDA
decode path relative to a conventional trained dense Transformer without weakening strict operator
correctness or choosing thresholds after seeing expanded candidate results?

## Why v2 was required

Finding [30](30_cuda_decode_failure_classification.md) showed that the old full-model elementwise logit
threshold fails the trained dense control as well as KDA. It also found three real KDA greedy
divergences, so the threshold could neither remain the sole gate nor be waived.

V2 retains every isolated operator, state, gradient, checkpoint, and export prerequisite. It changes
only the full-model BF16 behavior layer, with a new version and explicit disclosure of the known v1
results.

## Case stream and phase order

The provenance-only stream contains no token contents. It hashes exact loader coordinates and token
arrays from the frozen validation manifest:

- 33 linked 8/64/512-token cases: three per each of 11 validation sources;
- 11 independent 4,096-token cases: one per source;
- 44 base cases and 110 evaluated case/length cells total.

The dense checkpoint executed and was committed first. Its report hash, all endpoints, all margins,
10,000 paired case-bootstrap resamples, and deterministic seed were then sealed in a control-lock
artifact. Only after that lock existed were KDA seeds 42/43/44 executed.

## Frozen endpoints

Every seed and length had to pass all endpoints:

- common-history token-disagreement increase at most 1 percentage point;
- disagreement increase among full-path margins of at least 0.1 logits at most 0.2 points;
- Jensen-Shannon divergence increase at most 0.0001;
- top-10 overlap decrease at most 2 points;
- relative-RMS ratio at most 1.5;
- early free-running divergence increase at most 5 points;
- zero disagreements when the full-path top-1/top-2 margin is at least 0.5 logits.

The case, not the 32 nested decode steps, is the bootstrap unit. Each KDA seed must pass separately.

## Dense control envelope

Across 3,520 common-history decode comparisons:

| Prompt | Token disagreement | Mean JS | Mean relative RMS | Mean top-10 overlap | Early free divergence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.47% | 5.24e-5 | 0.00515 | 0.9901 | 27.27% |
| 64 | 0.57% | 5.17e-5 | 0.00462 | 0.9901 | 15.15% |
| 512 | 0.95% | 4.23e-5 | 0.00435 | 0.9918 | 15.15% |
| 4,096 | 0.57% | 4.16e-5 | 0.00423 | 0.9935 | 18.18% |

Dense has zero token changes at full-path margins of at least 0.1 logits. Exact 32-token free-running
identity is already unstable for 15–27% of source-balanced prompts, confirming that it is a high-
variance behavioral endpoint rather than a bitwise correctness test.

## KDA measurements

All twelve KDA seed/length cells pass Jensen-Shannon divergence, relative RMS, top-10 overlap, and the
0.5-logit hard guardrail. Eleven of twelve pass common-history token disagreement. Ten of twelve pass
the 0.1-logit-margin endpoint.

Observed common-history metrics are generally as good as or better than dense:

- mean JS is 3.38e-5 to 4.94e-5;
- mean relative RMS is 0.00412 to 0.00458;
- mean top-10 overlap is 0.9892 to 0.9932;
- token disagreement is 0.19% to 0.95%; and
- there are no disagreements at margins of at least 0.5 logits.

Specific misses remain:

- seed 42 at 64 tokens: token-disagreement upper bound 0.01042, just above the 0.01 margin;
- seed 44 at 8 and 512 tokens: one margin-at-least-0.1 disagreement in each cell, producing upper
  bounds of 0.00313 versus the 0.002 margin.

## Free-running endpoint and power

Early free-running divergence passes only 2 of 12 seed/length cells. This makes V2 fail. It does not
mean KDA is consistently worse: raw candidate rates range from 6.06% to 27.27%, versus dense rates of
15.15% to 27.27%, and several failed cells have equal or lower observed candidate rates. Their
one-sided bounds remain too wide.

The predeclared five-point margin is underpowered at the checked sample sizes. Using the observed
paired discordance, the approximate case counts needed for a one-sided normal half-width of 0.05 range
from 221 to 483 per seed/length cell. V2 has 33 short cases and 11 4K cases. At 4K, even seed 44's raw
rate of 9.09% versus dense's 18.18% yields an upper bound of 0.1818 and fails.

This power finding is post-result and cannot rescue V2. It shows why the failure must be read as a
failed qualification gate containing both adverse and inconclusive cells, not as proof that KDA
free-running behavior is uniformly worse.

## Decision

**Cache-equivalence v2 fails.** No preflight is rerun and no baseline training begins.

The stable distribution, top-k, relative-RMS, and high-margin results argue against a gross KDA cache
implementation error. The remaining decision requires a new version, not a reinterpretation of V2:

- either remediate CUDA numerical sensitivity and rerun V2 unchanged; or
- power a v3 common-history equivalence study adequately and treat exact free-running divergence as a
  separately reported chaotic-path risk rather than an underpowered five-point primary endpoint.

The latter would still have to resolve the seed-42 64-token and seed-44 margin-conditioned misses. V2
has runtime-qualification authority only and does not establish language-model quality or promote KDA.

## Artifacts

- [V2 contract](../research/paper-1/cache_equivalence_v2.json)
- [Case stream](../results/Speck-Paper1/cache-equivalence-cases.json)
- [Dense control lock](../results/Speck-Paper1/cache-equivalence-v2-control-lock.json)
- [V2 analysis](../results/Speck-Paper1/cache-equivalence-v2-analysis.json)
