# Evaluate useful context and its cost

Selected [evaluation](evaluation_plan_v2.json) and [comparator](comparator_plan_v2.json) contracts.
All candidate benchmark/model identities require pinned releases, local compatibility and exclusion
qualification before use. No benchmark run or capability result is claimed by this plan.

## Quality

Primary families: document/multi-document reasoning and conversation-history/update understanding.
Give the two families equal weight in the primary macro; show every task/family, source type and seed.
Supporting checks cover technical/code comprehension, ordinary instructions/writing/assistance, math,
code and the six-category short BPB dashboard. Freeze scorer normalization and weights before R2.

Measure 4K, 32K, 64K and 128K where supported. Useful context requires both primary-family absolute
quality floors and retained short/general quality at a frozen output budget. Set numerical floors and
non-inferiority margins from disjoint pilots before confirmation; do not choose them from final scores.
Oracle-evidence and short versions expose tasks the small model cannot solve even without distractors.

Candidate external evaluations: HELMET application categories, LongBench Pro in the declared English
length scope, LongMemEval with exact version/payload, and RULER as a retrieval diagnostic. Subsets,
modified prompts and length bins are labeled; do not present a subset as an official full-suite score.
Independent controlled tasks supplement natural benchmarks, never replace all external evidence.

Record answer correctness, grounding, abstention/coverage, explicit update handling, output tokens,
truncation and latency. For judges, pin model/settings and calibrate against a blinded human sample;
account for judge/API cost. At natural benchmark lengths, use their populations without covert
truncation of required evidence. When controllable, compare the same underlying questions across lengths.

## Controls, public models and retrieval

- Matched dense/hybrid R2 controls establish architecture-package effects.
- Pin at least two public models: one near total footprint and one stronger/larger reference. Candidates
  include MiniCPM5-1B/2B, LFM2.5-1.2B and SmolLM3-3B; candidates are not yet qualified comparisons.
- Practical baselines: BM25 retrieval plus the same answer model, and BM25 plus one larger public
  short-context answer model. Use the same source texts, questions and output budgets. Freeze a bounded
  chunk/top-k development search. No new reranker pipeline is required.

Count retrieval/indexing separately (one-time and amortized), memory, prompt/output tokens and complete
latency. Report where retrieval wins. Distinguish near-size, equal-footprint, equal-cost and broader
reference comparisons; parameters alone do not establish equivalent training or deployment cost.
Separate base versus instruct comparisons and teacher-assisted development.

## Systems

Primary platform: allocated GH200. Measure batch one and one declared throughput workload with at
least five randomized/interleaved warmed blocks, actual backend/precision, cold/warm cache behavior,
startup, prefill/TTFT, decode/TPOT, end-to-end task latency, resident state, weights plus state and peak
memory including workspace. Energy requires calibrated available telemetry; missing telemetry is
unavailable, not zero. Report output budget/length with every quality-cost operating point.

Compare matched backends/precision where possible and separately report each model's best qualified
runtime. Reference-path disadvantages cannot establish universal speedup. Secondary RTX 3090 and
optional CPU results require their own qualification. State reduction alone is not total-memory savings.

## Budget and final audit

R4=60 hours owns core/transfer formal evaluations; L6 owns stage diagnostics. V1=110 hours owns final
quality/public/RAG comparisons; V2=90 owns serving; V3=36 owns parity and demonstrations. Count shared
artifacts once. A bounded baseline implementation may be prepared early without consuming final tests.

Freeze final weights, prompts, decoding, task and cost budgets and primary claims before the one-opening
sealed final audit. A failed audit narrows claims and preserves failures; it does not reopen tuning on
the same test. Historical D5/E2 audits remain unopened exclusion inputs.
