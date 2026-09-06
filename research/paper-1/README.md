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
- [`raw_local_readiness_v1.json`](raw_local_readiness_v1.json) conditionally fixes a five-slot,
  shared-query, single-softmax raw-local formulation and freezes its deduplication, causal ring,
  window-selection, quality, state, and systems gates without authorizing implementation.
- [`ratio_placement_readiness_v1.json`](ratio_placement_readiness_v1.json) corrects the future 20-layer
  ratio grid to exact integer counts, freezes a shared quantile placement rule, and requires count
  selection before a fixed-count integration/readout placement successor.
- [`attnres_readiness_v1.json`](attnres_readiness_v1.json) fixes the 40-module residual source graph,
  adds the static-depth mechanism control, and freezes Full/Block equations, correctness, activation,
  block-count, depth/width, mechanistic, and efficiency gates before implementation.
- [`stable_latentmoe_readiness_v1.json`](stable_latentmoe_readiness_v1.json) records the missing primary
  specification and freezes conventional-MoE, latent, normalization, activation, balancing, and expert-
  geometry stages plus routing, stability, rescue, memory, and hardware gates.
- [`interaction_readiness_v1.json`](interaction_readiness_v1.json) freezes the axis-bundle unit, complete
  2³ cube, paired contrasts, multiplicity, retuned absence controls, subcomponent removals, and
  cross-scale retention rules without selecting a combined architecture.
- [`scaling_readiness_v1.json`](scaling_readiness_v1.json) separates transfer from frontier claims and
  freezes five fit scales, allocation pilots, joint/compute models, full-refit uncertainty, horizon
  interaction, and a held-out 1.2B prediction sentinel without authorizing runs.
- [`systems_cost_readiness_v1.json`](systems_cost_readiness_v1.json) preserves the failed 1.0-hour dense
  proxy envelope and freezes the analytic/operator/model/serving/monetary evidence hierarchy, energy,
  memory, online load, datacenter, and prospective-envelope requirements.
- [`novelty_landscape_v1.json`](novelty_landscape_v1.json) pins recent direct prior-art overlaps,
  rejects several easy novelty stories, and retains two falsifiable hypotheses without claiming either
  is novel before full-text/code/citation and held-out causal review.
- [`novelty_landscape_v2.json`](novelty_landscape_v2.json) adds HeadKV-R2 and routing-aware KV
  compression, concedes reasoning-aware allocation and retention/accessibility novelty, and narrows N2
  to conjunctive necessary-source survival with incremental prediction and causal restoration.
- [`novelty_landscape_v3.json`](novelty_landscape_v3.json) adds BRIEF, BRIEF-Pro, and IterCOMP, rejects
  novelty of the necessary-source conjunction and missing-hop recovery themselves, and leaves only an
  unestablished cheap internal-state predictive/causal law for possible later review.
- [`novelty_landscape_v4.json`](novelty_landscape_v4.json) classifies STEC's structured candidate-
  evidence verification as a mandatory adjacent baseline while preserving the N2 concept rejection.
- [`novelty_landscape_v5.json`](novelty_landscape_v5.json) adds the strongest N1 role/scaling, 72-model
  ratio/mixer, and greedy conversion baselines; it rejects the broad role claim and leaves only an
  unestablished prospective non-uniform from-scratch placement law.
- [`novelty_landscape_v6.json`](novelty_landscape_v6.json) adds a direct from-scratch ratio/placement
  study, rejects novelty of non-uniform placement and early/middle role recipes, and leaves prospective
  unseen-layout prediction only as an unestablished incremental question with no experiment authority.
- [`novelty_landscape_v7.json`](novelty_landscape_v7.json) adds controlled activation/placement
  evidence: PAS/ISP and gate-role diagnostics are occupied, while saturated PAS alignment fails to rank
  early/middle/late retrieval, lowering the prior on N1 and making retirement an explicit review option.
- [`novelty_landscape_v8.json`](novelty_landscape_v8.json) adds HALO's hidden-state-aligned,
  recall/commonsense-guided layer selection and control suite, leaving only a procedural held-out
  from-scratch interaction predictor for independent retention-or-retirement review.
- [`novelty_landscape_v9.json`](novelty_landscape_v9.json) adds generic-KL to held-out-task,
  cross-scale, cross-mixer, clustering, spacing, and early-stop selection evidence, leaving only a
  teacher-free from-scratch procedural residual and making N1 retirement evidence-favored.
- [`novelty_claim_overlap_v1.json`](novelty_claim_overlap_v1.json) maps fourteen candidate claims to
  their strongest overlap, keeps N1 as the only primary architecture hypothesis, and defers the residual
  N2 diagnostic before any experimental budget is spent.
- [`novelty_claim_overlap_v2.json`](novelty_claim_overlap_v2.json) supersedes that priority after the
  N1 role audit: zero architecture-novelty candidates are established, both residual laws lack experiment
  authority, and only the already-frozen baseline sequence continues.
- [`novelty_claim_overlap_v3.json`](novelty_claim_overlap_v3.json) adds direct from-scratch placement
  and early/middle role overlap, leaving prospective arbitrary-layout prediction only as a technical
  residual whose incremental scientific value must pass independent review before any protocol.
- [`novelty_claim_overlap_v4.json`](novelty_claim_overlap_v4.json) adds PAS/ISP overlap and saturated
  diagnostic counterevidence, making N1 a low-prior multidiagnostic question that requires explicit
  independent retention-or-retirement review before any further budget.
- [`novelty_claim_overlap_v5.json`](novelty_claim_overlap_v5.json) adds HALO task-guided layer
  selection and the HypeNet composition bundle, leaving only a procedural held-out from-scratch
  interaction-prediction residual for independent retention-or-retirement review.
- [`novelty_claim_overlap_v6.json`](novelty_claim_overlap_v6.json) adds generic-KL cross-task,
  cross-scale, cross-mixer, clustering, and spacing overlap, reducing N1 to a teacher-free procedural
  residual and making retirement evidence-favored pending independent review.
- [`novelty_code_availability_v1.json`](novelty_code_availability_v1.json) pins official repository
  revisions, tree/code counts, and license scopes without executing third-party code; only Ada-KV
  currently has a qualified code-plus-root-rights path for deeper audit.
- [`novelty_code_availability_v2.json`](novelty_code_availability_v2.json) extends that inventory to
  HeadKV-R2 and the routing study: HeadKV has no root license or reproducible environment/data/profile
  chain, while the routing paper declares no dedicated immutable artifact.
- [`novelty_code_availability_v3.json`](novelty_code_availability_v3.json) covers all twelve audited
  overlaps: the evolving combined BRIEF tree lacks root rights and paper-specific revisions, while
  IterCOMP and STEC declare no dedicated immutable implementation.
- [`novelty_code_availability_v4.json`](novelty_code_availability_v4.json) extends coverage to fifteen
  sources: the hybrid-role repository has root MIT analysis code but lacks training/constraint/full-data
  artifacts, the systematic study exposes moving model repositories, and DtR declares no code.
- [`novelty_code_availability_v5.json`](novelty_code_availability_v5.json) records that the direct
  systematic placement study declares no official code, models, exact configs, or immutable data order;
  it is a conceptual baseline, not an authorized reproduction path.
- [`novelty_code_availability_v6.json`](novelty_code_availability_v6.json) adds the pinned MIT
  massive-activation analysis tree and Apache-2.0 checkpoint aggregate, while withholding execution and
  full reproduction for missing training, gated-FA, exact-input, and parity evidence.
- [`novelty_code_availability_v7.json`](novelty_code_availability_v7.json) adds HALO's pinned
  training/selection tree and result logs, but root-rights, semantic, environment, data, checkpoint,
  execution, reuse, and reproduction gates all remain blocked.
- [`novelty_code_availability_v8.json`](novelty_code_availability_v8.json) adds the KL-selector's
  pinned source/config tree while blocking absent root rights/tests, held-out-KL and early-stop parity,
  environment/data/log/checkpoint identity, execution, reuse, and reproduction.
- [`adakv_code_audit_v1.json`](adakv_code_audit_v1.json) statically audits that pinned path and records
  its environment, native-build, monkeypatch, cache, remote-code/data, and test blockers before any
  clean-room reference or upstream execution.
- [`headkv_code_audit_v1.json`](headkv_code_audit_v1.json) pins HeadKV's tree, selected source hashes,
  narrow CUDA license, embedded data/profiles, environment, model, rounding, monkeypatch, native-binary,
  and missing-test blockers without importing or executing the repository.
- [`brief_code_audit_v1.json`](brief_code_audit_v1.json) separates the old 49-file BRIEF subtree from
  BRIEF-Pro and its 492-file vendored Axolotl tree, preserving root-rights, historical-revision,
  environment, model/data, credential, and held-out-contamination blockers.
- [`rethinking_hybrid_code_audit_v1.json`](rethinking_hybrid_code_audit_v1.json) pins the 20-file MIT
  analysis tree while recording absent training configs, receptive-field implementation, complete
  scaling observations, immutable checkpoint/data identities, environment pins, and tests.
- [`massive_hla_code_audit_v1.json`](massive_hla_code_audit_v1.json) pins the official MIT analysis
  tree and Apache-2.0 aggregate checkpoint tree without checkout or execution, qualifying identities
  while blocking missing training code, gated-FA behavior, exact inputs, and full reproduction.
- [`halo_code_audit_v1.json`](halo_code_audit_v1.json) pins HALO's training/selection source and 206
  evaluation logs, while blocking absent root rights/tests, paper-code epsilon drift, unpinned
  environment/data/checkpoints, execution, reuse, and full reproduction.
- [`kl_selection_code_audit_v1.json`](kl_selection_code_audit_v1.json) pins the official 203-file
  selector/config tree while blocking absent root rights/tests, held-out-KL versus training-loss drift,
  missing early-stop code, external state, data/log/checkpoint identity, and reproduction.
- [`n1_independent_review_packet_v1.json`](n1_independent_review_packet_v1.json) freezes the residual
  N1 claim, mandatory evidence, reviewer independence, questions, outputs, and retirement/preregistration
  dispositions without treating Speck's own audit as independent or authorizing execution.
- [`adaptive_cache_budget_v1.json`](adaptive_cache_budget_v1.json) freezes and qualifies a clean-room,
  within-layer physical-head allocation reference against exhaustive small cases, while keeping GQA
  reduction, model integration, training, novelty, and architecture claims blocked.
- [`adaptive_cache_gqa_v1.json`](adaptive_cache_gqa_v1.json) proves and exhaustively checks equal-group
  mean reduction onto physical GQA heads, rejects max as theorem-equivalent, and records a safeguard
  conservation failure without authorizing model integration or training.
- [`adaptive_cache_safeguard_v1.json`](adaptive_cache_safeguard_v1.json) defines an exact-rational,
  largest-remainder integer control that conserves physical slots and resolves neither the upstream
  coefficient ambiguity nor its unmeasured quality tradeoff.
- [`adaptive_cache_salience_readiness_v1.json`](adaptive_cache_salience_readiness_v1.json) freezes
  attention-probe parity, question-visible/context-only/hindsight modes, pooling and geometry isolation,
  cache lifecycle, source-completeness diagnostics, and variable-length runtime blockers.
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
- [`baseline-analysis.json`](../../results/Speck-Paper1/baseline-analysis.json) completes the frozen
  three-pair proxy: aggregate and all source quality bounds pass, time-to-quality is uncensored, and
  only separate finalist materialization/qualification becomes eligible.
- [`finalist_analysis_v1.json`](finalist_analysis_v1.json) freezes the exact six-pair, 23,496-step
  longer-horizon analysis, df=5 bounds, unchanged margins, control-only target, and zero-interim stopping
  rule before finalist materialization, while explicitly withholding training authority.
- [`finalist_analysis_v2.json`](finalist_analysis_v2.json) supersedes v1 before any result: two fixed
  data-order strata each use three-seed df=2 bounds plus hard cell/source guards; pooled df=5 output is
  descriptive only, and inference to arbitrary data orders is forbidden.
- [`finalist_materialization_v1.json`](finalist_materialization_v1.json) pins both proxy parent arms,
  the six-pair crossing, new config/checkpoint roots, exact longer-horizon changes, fail-on-overwrite
  behavior, and data/storage/runtime/release qualification gates without authorizing training.
- [`finalist_materialization.json`](../../experiments/Speck-Paper1-Finalist-131M/finalist_materialization.json)
  hashes all 84 generated configs across the two arm templates and twelve unique runs; materialization
  is complete but data/storage/runtime/analysis qualification and training remain blocked.
- [`finalist-qualification-v1.json`](../../results/Speck-Paper1/finalist-qualification-v1.json)
  qualifies all config hashes, twelve absent outputs, two disjoint data windows with ten exact
  direct/resume replay points, and the 5.64-TB-free dedicated volume; runtime, analysis, release gates,
  and training remain blocked.
- [`finalist-analysis-qualified-v1.json`](../../results/Speck-Paper1/finalist-analysis-qualified-v1.json)
  pins and qualifies the twelve-cell collector, six-control target lock, df=5 paired analysis,
  censoring, and stopping-rule implementation; runtime, release gates, and training remain blocked.
- [`finalist-analysis-qualified-v2.json`](../../results/Speck-Paper1/finalist-analysis-qualified-v2.json)
  supersedes that inference before results and qualifies two order-stratum df=2 bounds, hard cell/source
  guards, descriptive-only pooling, and the unchanged collector/stopping geometry.
- [`finalist-preflight-v1.json`](../../results/Speck-Paper1/finalist-preflight-v1.json) qualifies
  exact-config compiled CUDA steps, temporary Transformers export, and the powered trained-topology
  cache reference. Random-weight native drift remains explicit; release gates and training stay blocked.
- [`finalist_launch_v1.json`](finalist_launch_v1.json) separates checkpoint production from mandatory
  checkpoint-consuming release suites, freezes the 121.23-hour control-first event sequence and live
  gates, and authorizes automation implementation but not the initial launch.
- [`finalist_launch_v2.json`](finalist_launch_v2.json) supersedes v1 before output, binding that
  unchanged sequence and release boundary to the corrected two-order, three-seed analysis and
  withholding launch until automation v2 and a post-HELMET live gate exist.
- [`finalist_automation_v1.json`](finalist_automation_v1.json) pins the event-driven twelve-run runner,
  empty program state, all-control target lock, per-result commits, live gates, cooldown, and fail-closed
  no-retry behavior; a post-HELMET live qualification is still required before launch.
- [`finalist_automation_v2.json`](finalist_automation_v2.json) supersedes v1 before output and binds
  the same event state machine exclusively to the corrected two-order analysis and final status; live
  qualification and training remain blocked while HELMET acquisition is active.
- [`finalist-launch-qualified-v2.json`](../../results/Speck-Paper1/finalist-launch-qualified-v2.json)
  records the clean empty v2 state, inactive HELMET/finalist services, and passing live GPU, memory,
  mount, storage, and output-absence gates; only the initial dense control and event successors qualify.
- [`finalist-failed-attempt-control-pair0-v1.json`](../../results/Speck-Paper1/finalist-failed-attempt-control-pair0-v1.json)
  preserves the mistaken operator stop after one step, proves v2 predated launch, hashes local logs, and
  records zero checkpoint/result files; an identical preregistered restart and fresh gate are required.
- [`finalist_rerun_v1.json`](finalist_rerun_v1.json) freezes attempt 2 as the identical dense pair-0
  step-0 restart, lists known ineligible observations, changes no scientific field, and requires a new
  live gate without enabling automatic retry.
- [`finalist-launch-qualified-v3.json`](../../results/Speck-Paper1/finalist-launch-qualified-v3.json)
  requalifies the empty v2 ledger with its failed-attempt/rerun pins and passing live GPU, memory,
  storage, output-absence, unit, and HELMET gates for the identical attempt-2 restart.
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
- [`helmet-data-acquisition.json`](../../results/Speck-Architecture-Promotion-v1/helmet-data-acquisition.json)
  records the complete 11.27-GB revision-pinned archive and independently checked SHA-256; extraction,
  runtime-loaded sources, rights, tokenizer, judges, and execution remain blocked.
- [`helmet-archive-inspection.json`](../../results/Speck-Architecture-Promotion-v1/helmet-archive-inspection.json)
  reproduces a safe 175-member/35.63-GB inventory with all 52 local paths, but finds no license metadata;
  component rights, extraction, 50 external entries, tokenizer, judges, and execution remain blocked.
- [`helmet-archive-local-rights-audit.json`](../../results/Speck-Architecture-Promotion-v1/helmet-archive-local-rights-audit.json)
  accounts for all 52 local paths as RULER 15, JSON-KV 5, KILT RAG 24, MS MARCO 6, and ALCE 2.
  Zero families qualify for extraction because exact derivation/attribution is absent and component
  terms remain mixed or restricted. This append-only result does not modify the frozen v2 manifest.
- [`helmet-synthetic-reconstruction-readiness.json`](../../results/Speck-Architecture-Promotion-v1/helmet-synthetic-reconstruction-readiness.json)
  proves that Speck-tokenized RULER cases cannot substitute for HELMET's Llama-2-tokenized cases and
  that the paper's JSON-KV concept omits the exact generator, seeds, schema, sampling unit, and hashes.
  Neither path can currently produce an official HELMET result; no substitute is activated.
- [`helmet-real-data-reconstruction-readiness.json`](../../results/Speck-Architecture-Promotion-v1/helmet-real-data-reconstruction-readiness.json)
  accounts for the other 32 archive-local paths and separates deterministic loading from undocumented
  construction. RAG lacks pinned source/retriever/index/seed identities, reranking lacks its exact
  TREC/MS MARCO construction, and HELMET's ALCE top-2000 files have no released generation path.
- [`helmet-truncation-tokenizer-readiness.json`](../../results/Speck-Architecture-Promotion-v1/helmet-truncation-tokenizer-readiness.json)
  maps the gated Llama-2 tokenizer to 25 runtime entries and 15 RULER-generation cells. Exact BOS,
  token, offset, Unicode-cut, suffix, and retokenization behavior makes fixture-only replacement
  insufficient; authorized oracle bytes and real-data parity are required before any successor.
- [`helmet-model-judge-readiness.json`](../../results/Speck-Architecture-Promotion-v1/helmet-model-judge-readiness.json)
  corrects the historical seed record to 42, then shows why best-effort seed sampling is insufficient:
  fingerprints are discarded, parse failures change denominators, schemas are unchecked, the exact
  snapshot is deprecated, and Batch data retention/deletion/cost authority is absent.
- [`finalist-automation-transition-audit-v1.json`](../../results/Speck-Paper1/finalist-automation-transition-audit-v1.json)
  executes all 12 frozen state transitions on temporary paths: 12 collections/commits, 11 schedules,
  one control-only target lock, and one final analysis. It preserves failure history and records two
  post-sequence hardening items without changing the live runner or contract.
- [`finalist-crossed-views-audit-v1.json`](../../results/Speck-Paper1/finalist-crossed-views-audit-v1.json)
  verifies that endpoint, source, fixed-FLOP, fixed-time, and uncensored time-to-quality outputs all
  preserve the two fixed data-order strata; one censored pair suppresses every time bound. Secondary
  pooled blocks remain descriptive under the plan despite a deferred redundant inline-label gap.
- [`finalist-systems-measurement-audit-v1.json`](../../results/Speck-Paper1/finalist-systems-measurement-audit-v1.json)
  separates preserved synchronized timing/peak allocation from absent energy and hardware telemetry.
  Control-first target integrity aliases architecture with multi-day calendar block, so finalist timing
  remains descriptive until a thermally interleaved, telemetry-rich systems protocol is run.
- [`finalist-source-stability-audit-v1.json`](../../results/Speck-Paper1/finalist-source-stability-audit-v1.json)
  proves per-step loss/gradient fail-fast behavior and positive 11-source validation coverage, but also
  reproduces the frozen analyzer's consistent-omission gap. An append-only sidecar now requires every
  expected source and finite source loss before any collected result is interpreted.
- [`finalist-result-acceptance-qualified-v1.json`](../../results/Speck-Paper1/finalist-result-acceptance-qualified-v1.json)
  qualifies one post-event verifier for program/result/source/transition/target/analysis consistency.
  It passes the empty ledger and must pass after every automatic commit before interpretation or Linear
  completion; it intentionally remains outside the hash-frozen finalizer.
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
The acquired archive's local-path audit now resolves its five-family inventory without opening payload
content. None is extraction-qualified: permissive code licenses do not establish the derivative data
chain, and MS MARCO adds an explicit noncommercial/acceptance boundary. The active v2 evaluation
manifest remains byte-identical; any future subset or clean-room regeneration requires a pre-results
successor rather than an in-place edit.
The synthetic-reconstruction audit also rejects a tempting shortcut: the existing RULER cases use a
different generation tokenizer from HELMET, and the cited JSON-KV ancestor is structurally different
from HELMET's undocumented six-depth variant. A clean-room diagnostic would need a separate name and
cannot discharge the official-suite gate.
The same boundary holds for all real-data files: the paper describes sensible construction policies,
but not executable, versioned pipelines. Runtime determinism cannot establish source, retrieval,
ordering, or byte parity, so independent rebuilds remain separately named diagnostics rather than
HELMET substitutes.
The gated tokenizer is likewise not replaceable by token-count resemblance: it defines row filtering
and exact Unicode cut positions, so an oracle comparison requires the authorized Llama-2 tokenizer and
qualified real documents. Until then every dependent entry stays failed rather than approximated.
The judge audit similarly prevents a superficial fix: seed 42 is present, but official determinism is
best-effort and HELMET drops fingerprints and failed parses. Full prompt uploads also require explicit
retention/deletion and dataset-transmission authority. Neither a newer API model nor a local judge can
inherit the original metric without a versioned, repeated, human-calibrated replacement protocol.
The full temporary finalist simulation additionally verifies every control/candidate transition and
the exact lock/analyze ordering. Terminal service provenance and an explicit finalizer `next_run`
recheck remain future hardening; applying them mid-sequence would invalidate the frozen runner, so v2
continues byte-identically under artifact-first collection.
The blind crossed-views audit confirms that the v2 correction propagates beyond endpoint loss: every
secondary view preserves order strata, any censor suppresses time bounds, and none controls the final
language pass. Only redundant inline labels on secondary pooled summaries are deferred.
The systems audit further limits interpretation: same-GPU/start-temperature launch gates do not remove
the control-versus-candidate calendar-block confound over five days. Analytic FLOPs and physical timing
may be reported separately, but energy, dollars, causal speedup, serving, and systems promotion require
new thermally interleaved measurements after the language sequence.
Training non-finiteness already stops before a final event, but complete source identity needs a
redundant sidecar: deterministic validation covers all 11 sources, while the frozen analyzer alone
would accept the same omitted source in every result. Missing/non-finite sources now mandate rejection,
never imputation or silent deletion.
The result-acceptance verifier composes those gates after each event: it binds every accepted result to
the program and transition, checks complete source coverage, validates the exact successor edge, and
preserves the failed-attempt/rerun history. Failure blocks interpretation without rewriting evidence.
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
All three candidate pairs are complete and qualified at 2.794477, 2.796465, and 2.795503 final loss.
Their paired fixed-token differences are -0.039170, -0.042662, and -0.037184 nats. The mean is
-0.039672 and its upper one-sided 95% bound is -0.034996, below the +0.01 non-inferiority margin. All
eleven source bounds pass, all arms reach the locked target, and the mean time-to-quality improvement is
19.48% with a 16.83% lower bound. At the endpoint, candidate steady time is 10.01% shorter and analytic
FLOPs/token are 21.49% lower on average, while peak allocation is 20.77% higher. The whole-architecture
proxy quality screen passes. Under the frozen disposition this authorizes only materializing and
separately qualifying the exact six-pair finalist; it grants no component attribution, architecture
promotion, novelty, release claim, finalist training, or paper-scale authority.
The project otherwise has strong evidence for GDN/KDA
trade-offs, the need for some global attention, a global-cache sharing failure frontier, and rigorous
promotion infrastructure. It does **not** yet have:

- a promoted sequence architecture;
- a local implementation or isolation of HCA/CSA, AttnRes, or Stable LatentMoE;
- a demonstrated Speck-specific architectural novelty;
- medium-scale transfer;
- independent long-context results; or
- a production serving runtime.

Accordingly, the paper has completed its **matched proxy quality screen**, finalist config
materialization, and data/storage, crossed-factor collector/analysis v2, and exact-runtime
qualification. V1 pooled inference is superseded. HELMET acquisition is complete and inactive, but v2
live qualification was consumed by an operator-interrupted step-1 attempt. No result exists; an
identical restart contract and new live gate were recorded, and attempt 2 is now running under the
final-summary event chain. The paper has entered finalist control execution, not architecture-promotion,
paper-scale-training, or manuscript-claim status.
