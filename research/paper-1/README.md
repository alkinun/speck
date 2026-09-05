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
- [`baseline_collection_v2.json`](baseline_collection_v2.json) preserves the first collector failure
  and freezes the complete-validation-batch arithmetic correction after dense control 0 but before
  any candidate checkpoint or result exists.
- [`baseline_automation_v1.json`](baseline_automation_v1.json) freezes one-shot, process-event-driven
  continuation through the remaining control, target lock, three candidate runs, and final proxy
  analysis, with no periodic polling or quality-dependent branching.
- [`proxy_disposition_v1.json`](proxy_disposition_v1.json) freezes the integrity, aggregate-failure,
  source-failure, and quality-pass branches before candidate 0, including the exact limits on finalist
  eligibility and non-claims.
- [`sequence_cache_representation_v1.json`](sequence_cache_representation_v1.json) conditionally
  freezes the five-memory GQA3/MQA1/NoPE-MLA128 isolation, analytic geometry, implementation gates,
  multiplicity, and realized systems thresholds without authorizing training.
- [`hca_readiness_v1.json`](hca_readiness_v1.json) records why HCA is not implementation-ready and
  freezes the causal tail/prefix semantics, compressor-isolation requirement, conditional rate grid,
  accounting, and realized-cost gates needed before it can become an experiment.
- [`csa_readiness_v1.json`](csa_readiness_v1.json) keeps sparse precision separate from HCA/local/cache
  changes and freezes oracle-first block selection, selector diagnostics, conditional geometry,
  causal state, complexity accounting, and sparse-prefill/sparse-decode gates.
- [`proxy_launch_v1.json`](proxy_launch_v1.json) freezes the boundary between checkpoint-producing
  proxy execution and checkpoint-consuming release evaluation, including control-first ordering and
  explicit non-claims.
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
- [`proxy-launch-qualified.json`](../../results/Speck-Paper1/proxy-launch-qualified.json) rehashes the
  complete freeze and records the clean live GPU, host-memory, output-absence, and storage gate that
  authorizes proxy training while preserving blocked release claims.
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
- [`helmet_narrativeqa_decision_v1.json`](../architecture-promotion-v1/helmet_narrativeqa_decision_v1.json)
  separates Apache metadata from 1,572 embedded external works and freezes independent rights,
  unseeded-demo, tokenizer, and judge blockers.
- [`helmet_infinitebench_decision_v1.json`](../architecture-promotion-v1/helmet_infinitebench_decision_v1.json)
  freezes the minimal three-file payload, web-derived work boundary, seeded prompt selection, local
  long-QA metrics, gated truncation, and HELMET-only summarization-judge change.
- [`helmet_seeded_demos_v1.json`](../architecture-promotion-v1/helmet_seeded_demos_v1.json) qualifies
  a two-line seed repair for NarrativeQA and Multi-LexSum across independent fixture processes without
  activating either rights-blocked dataset.
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
NarrativeQA independently blocks the source-document guardrail on embedded-work rights, another
unseeded two-shot path, the same truncation tokenizer, and its proprietary judge.
All seven HELMET runtime sources now have explicit dispositions: three immutable paths qualify and
four sources are blocked rather than unresolved.
The known unseeded two-shot defect now has a qualified but inactive repair; a successor manifest and
real-data replay remain mandatory.
The launch boundary now distinguishes those checkpoint-consuming release gates from the
checkpoint-producing language-model proxy. The matrix, fixed-sample analysis, control-only target
lock, materialization, behavioral/hardware preflight, storage, evaluation definitions, contamination
disposition, thresholds, and missing-suite failure rule were frozen and rehashed before output. The
clean live gate passed, so the proxy is authorized in strict control-first order. RULER, NoLiMa, and
HELMET remain required failed release gates until they are independently qualified and executed.
All three dense controls are complete and qualified at 2.833646, 2.839127, and 2.832686 final
validation loss. Their mean is 2.835153, their range is 0.006440 nats, and their steady training times
span only 0.135% of the mean despite different initialization/data cells. The preregistered
worst-control rule has locked the time-to-quality target at 2.839127 before any candidate output.
Control 0's first collection
attempt exposed a tooling-only mismatch between the requested 20M-token final validation budget and
the runtime's 19,988,480 complete-batch count. The frozen pre-candidate correction derives the latter
from the existing 4-by-4,096 single-GPU geometry and changes no loss, threshold, sample, or decision.
The control phase is complete; candidate pair 0 is the next fixed cell after cooldown and live checks.
The project otherwise has strong evidence for GDN/KDA
trade-offs, the need for some global attention, a global-cache sharing failure frontier, and rigorous
promotion infrastructure. It does **not** yet have:

- a promoted sequence architecture;
- a local implementation or isolation of HCA/CSA, AttnRes, or Stable LatentMoE;
- a demonstrated Speck-specific architectural novelty;
- medium-scale transfer;
- independent long-context results; or
- a production serving runtime.

Accordingly, the paper has entered **matched proxy execution**, not architecture-promotion,
paper-scale-training, or manuscript-claim status.
