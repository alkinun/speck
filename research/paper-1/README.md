# Speck Paper 1 research program

## Working thesis

> A language model can improve quality per training and serving cost by controlling information flow
> independently across sequence, depth, and width, then composing only mechanisms whose isolated
> benefits survive replication, scale transfer, and realized runtime measurement.

The three axes are:

- **Sequence:** local raw context, fixed-state recurrence, compressed global coverage, and selective
  precise retrieval.
- **Depth:** content-dependent access to earlier representations without unbounded residual dilution.
- **Width:** sparse expert activation with explicit stability, balance, memory, and communication
  contracts.

This is a research hypothesis, not the name of a finished architecture. Candidate mechanisms include
SWA, KDA, GQA/MQA/MLA, HCA, CSA, Block Attention Residuals, and Stable LatentMoE. A component enters the
final architecture only after passing the active
[`architecture-promotion-v1`](../architecture-promotion-v1/) policy.

## Novelty gate

Combining published mechanisms is not sufficient novelty. Before any paper-scale pretraining run,
Speck must establish at least one of the following under a matched and replicated design:

1. a new sequence, depth, or width mechanism with a reproducible advantage;
2. a new composition rule that predicts when known mechanisms cooperate or interfere and survives a
   held-out interaction test;
3. a generalizable empirical law connecting an internal mechanism diagnostic to quality and realized
   cost across at least three scales; or
4. a systems method that changes the feasible quality-cost frontier and is inseparable from the model
   design.

An implementation contribution alone may support the paper, but it cannot satisfy this gate without a
scientific claim and evidence.

## Reference-paper standard

The program adopts the coverage—not the conclusions—of three primary references:

- [Kimi Linear](https://arxiv.org/abs/2510.26692) develops an operator and chunkwise algorithm, tests
  synthetic mechanisms, isolates key components and layer ratios, fits scaling laws, compares matched
  large models through pretraining/SFT/RL, and reports long-context and serving efficiency.
- [Kimi K3](https://arxiv.org/abs/2607.24653) organizes architecture around information flow across
  sequence, depth, and width, then covers data, scaling, training, long-context extension,
  infrastructure, serving, broad evaluation, and cost efficiency.
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348) specifies compressed/sparse attention mathematically,
  analyzes cache and FLOPs, documents training and inference systems, reports data/training/stability
  details, and separates base, post-trained, real-world, and limitation sections.

Their reports also illustrate a limitation Speck should improve upon: whole-system gains do not isolate
every component. Paper 1 therefore requires explicit component, removal, and interaction evidence before
the final combined model.

## Files

- [`claims.json`](claims.json) is the claim and falsification ledger.
- [`baseline_matrix.json`](baseline_matrix.json) audits historical controls and freezes the first
  parameter-matched dense/KDA paired design, data orders, matching views, storage gate, and non-claims.
- [`baseline_analysis.json`](baseline_analysis.json) freezes the paired estimand, confidence bound,
  source guardrails, interpolation rules, control-only time-to-quality lock, censoring, and fixed-sample
  stopping rule before any new baseline result exists.
- [`cuda_decode_diagnostic.json`](cuda_decode_diagnostic.json) and
  [`cuda_decode_trained_sentinel.json`](cuda_decode_trained_sentinel.json) freeze the multi-seed
  failure-classification matrix and its immutable trained-checkpoint follow-up.
- [`cache_equivalence_v2.json`](cache_equivalence_v2.json) freezes the source-balanced, control-first
  behavioral cache contract. Its checked decision is a failed qualification, not permission to relax
  the original gate.
- [`cache_equivalence_v3.json`](cache_equivalence_v3.json) freezes the powered, disjoint successor.
  V3 qualifies common-history CUDA cache behavior while preserving free-running risk and the failed v2
  decision.
- [`baseline-audit.json`](../../results/Speck-Paper1/baseline-audit.json) rehashes the five historical
  checkpoints, verifies the materialized pair/data windows, and records the live storage deficit.
- [`baseline-storage-volume-qualified.json`](../../results/Speck-Paper1/baseline-storage-volume-qualified.json)
  binds all six proxy checkpoint paths to a dedicated physical filesystem without moving or deleting
  prior evidence.
- [`contamination_v1.json`](contamination_v1.json) freezes exact-token probes over the three proxy
  training windows. Its checked result fails the answer-anchor gate without changing the threshold.
- [`contamination_disposition_v1.json`](contamination_disposition_v1.json) reconstructs every matched
  hash reference, quarantines the implicated tasks, and requires a new manifest version.
- [`evaluation_manifest.json`](../architecture-promotion-v1/evaluation_manifest.json) is now the v2
  successor: eleven official synthetic RULER tasks are primary, both QA tasks have zero primary
  weight, and HELMET RAG/long-QA is the separately gated source-document guardrail.
- [`helmet_runtime_dependencies_v1.json`](../architecture-promotion-v1/helmet_runtime_dependencies_v1.json)
  proves that 50 of HELMET's 105 entries are not supplied by the main archive and freezes the external
  dataset, tokenizer, runtime-compatibility, rights, and model-judge blockers.
- [`helmet_materializer_preflight_v1.json`](../architecture-promotion-v1/helmet_materializer_preflight_v1.json)
  qualifies a hash-locked legacy-to-Parquet bridge for Banking77 and NLU Evaluation Data with complete
  row, order, value, feature, replay, and current-offline-reader parity.
- [`helmet_clinc_source_v1.json`](../architecture-promotion-v1/helmet_clinc_source_v1.json) pins the
  exact CLINC150 `plus` train/validation snapshots, upstream row provenance, feature identity, and
  retained CC-BY-3.0 attribution.
- [`helmet_trec_rights_v2.json`](../architecture-promotion-v1/helmet_trec_rights_v2.json) preserves a
  failed raw-transport v1, canonicalizes only volatile delivery wrappers, and records the absence of
  affirmative authority without downloading TREC payloads.
- [`helmet_multilexsum_decision_v1.json`](../architecture-promotion-v1/helmet_multilexsum_decision_v1.json)
  separates database and summary rights and proves that noncommercial summaries enter two-shot prompts
  through an unseeded selection path.
- [`experiment_program.json`](experiment_program.json) freezes baselines, stages, scales, axes, and the
  paper-scale pretraining gate.
- [`paper_outline.md`](paper_outline.md) defines the manuscript structure and required evidence in each
  section.
- [`reference_audit.md`](reference_audit.md) maps the evidence depth of Kimi Linear, Kimi K3, and
  DeepSeek-V4 to explicit Speck requirements.
- [`reporting_checklist.md`](reporting_checklist.md) is the release-grade completeness checklist.

Validate the program with:

```bash
uv run --extra cpu python -m scripts.paper_program_validate research/paper-1
uv run --extra cpu python -m scripts.paper_baseline_prepare \
  research/paper-1/baseline_matrix.json --check
```

Each completed baseline checkpoint is normalized with `scripts.paper_baseline_analyze collect`. Collect
the three dense controls first, run `lock-target`, and only then collect and analyze the hybrid results.
The checked analysis plan contains the exact CLI input and result contracts.

## Current state

The five historical sequence controls are now identity-audited as discovery evidence only. A new
153.96M-parameter dense/KDA baseline pair is materialized across three paired initialization/data-order
cells. Its analysis and stopping rule are frozen, and the versioned RTX 3090 training/export/cache
preflight now passes. The dedicated volume passes both proxy and finalist floors. The frozen RULER
contamination audit, however, detects 28 answer-anchored patterns in `qa_1`/`qa_2`; v1 remains failed.
The v2 successor is frozen before model outputs with the other eleven tasks as its primary matrix and
HELMET RAG/long-QA as the source-document guardrail. NoLiMa and HELMET contamination checks remain
blocked on their separate data/legal qualifications. HELMET's runtime audit additionally shows that
archive completion alone cannot qualify 50 externally loaded entries or the proprietary-judge metrics.
Its isolated materialization strategy now qualifies for two permissive ICL sources, but no broader
HELMET execution authority follows. CLINC150's separate data-only snapshot also qualifies, leaving
TREC blocked on written authority or pre-results removal, followed by final prompt/contamination work,
before the ICL category can qualify.
Multi-LexSum independently blocks summarization on organizational scope, unseeded demonstrations, the
truncation tokenizer, and the model judge.
The project otherwise has strong evidence for GDN/KDA
trade-offs, the need for some global attention, a global-cache sharing failure frontier, and rigorous
promotion infrastructure. It does **not** yet have:

- a promoted sequence architecture;
- a local implementation or isolation of HCA/CSA, AttnRes, or Stable LatentMoE;
- a demonstrated Speck-specific architectural novelty;
- medium-scale transfer;
- independent long-context results; or
- a production serving runtime.

Accordingly, the paper is in **thesis and experiment-design**, not model-training or manuscript-claim,
status.
