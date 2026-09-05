# Speck long-context research ledger

This directory is the narrative index for the long-context experiments. Checked JSON under
`results/` remains the machine-readable source of truth; these files preserve the experimental
question, controls, failures, decisions, and interpretation around those artifacts.

Read in order:

1. [00 — Research contract](00_research_contract.md)
2. [01 — 131M-token mixer screen](01_mixer_screen_131m.md)
3. [02 — Correctness and kernel calibration](02_correctness_and_kernel_calibration.md)
4. [03 — Seed noise floor](03_seed_noise_floor.md)
5. [04 — Existing-checkpoint 128K systems frontier](04_existing_checkpoint_128k_systems.md)
6. [05 — Long-document dataset](05_long_document_dataset.md)
7. [06 — Matched 32K local/global continuation](06_context32k_local_vs_global.md)
8. [07 — Counterfactual retrieval diagnostic](07_counterfactual_retrieval.md)
9. [08 — Global-layer count and placement frontier](08_global_attention_frontier.md)
10. [09 — Decisions, open questions, and change log](09_decisions_and_change_log.md)
11. [10 — Kimi Linear transfer review and revised experiment order](10_kimi_linear_transfer_review.md)
12. [11 — KDA implementation and kernel qualification](11_kda_implementation_and_qualification.md)
13. [12 — Same-parent NoPE context activation](12_nope_context_activation.md)
14. [13 — Synthetic MQAR calibration and mixer comparison](13_synthetic_mqar.md)
15. [14 — MQAR distance and load scaling](14_mqar_length_scaling.md)
16. [15 — Palindrome and 64-stack mixer qualification](15_palindrome_and_stack.md)
17. [16 — Kimi-transfer language-model staircase](16_kimi_transfer_131m.md)
18. [17 — Three-seed Kimi-frontier replication](17_kimi_frontier_replication.md)
19. [18 — Matched 32K KDA/NoPE context activation](18_kimi_context32k.md)
20. [19 — Retrieval specificity, exact completion, and language replay](19_retrieval_specificity_and_replay.md)
21. [20 — K3 diagnostics and global attention gating](20_k3_diagnostics_and_attention_gating.md)
22. [21 — Retrieval answer transfer and template failure](21_retrieval_template_transfer.md)
23. [22 — Template-diverse retrieval adaptation](22_template_diverse_retrieval_adaptation.md)
24. [23 — Symbolic two-hop composition](23_symbolic_two_hop_composition.md)
25. [24 — Speck Reader Attention and the global cache-count staircase](24_reader_attention.md)
26. [25 — Architecture promotion policy and cost envelopes](25_architecture_promotion_policy.md)
27. [26 — Paper 1 research program](26_paper_1_research_program.md)
28. [27 — Paper 1 matched-baseline audit and launch contract](27_paper_1_baseline_audit.md)
29. [28 — Paper 1 baseline analysis and stopping contract](28_paper_1_baseline_analysis.md)
30. [29 — Paper 1 baseline hardware preflight failure](29_paper_1_baseline_preflight.md)
31. [30 — Full-depth CUDA decode failure classification](30_cuda_decode_failure_classification.md)
32. [31 — Control-first CUDA cache-equivalence v2](31_cache_equivalence_v2.md)
33. [32 — Powered cache equivalence v3 and baseline preflight v2](32_cache_equivalence_v3_and_preflight_v2.md)
34. [33 — RULERv1 offline source-bundle qualification](33_ruler_offline_source_bundle.md)
35. [34 — RULERv1 4K deterministic case qualification](34_ruler_4k_case_qualification.md)
36. [35 — RULERv1 8K deterministic case qualification](35_ruler_8k_case_qualification.md)
37. [36 — RULERv1 16K deterministic case qualification](36_ruler_16k_case_qualification.md)
38. [37 — RULERv1 32K deterministic case qualification](37_ruler_32k_case_qualification.md)
39. [38 — RULERv1 64K deterministic case qualification](38_ruler_64k_case_qualification.md)
40. [39 — RULERv1 128K and all-length data qualification](39_ruler_128k_and_data_completion.md)
41. [40 — HELMET native Speck adapter qualification](40_helmet_native_adapter_qualification.md)
42. [41 — HELMET offline native scorer runtime](41_helmet_offline_scorer_runtime.md)
43. [42 — NoLiMa license and metadata decision gate](42_nolima_license_decision.md)
44. [43 — HELMET data metadata and storage plan](43_helmet_data_metadata_and_storage_plan.md)
45. [44 — Paper 1 dedicated checkpoint-volume qualification](44_paper_1_dedicated_checkpoint_volume.md)
46. [45 — Paper 1 RULER contamination audit and v1 failure](45_paper_1_ruler_contamination.md)
47. [46 — Post-contamination RULER v2 successor manifest](46_ruler_v2_successor_manifest.md)
48. [47 — HELMET archive-external runtime and scorer boundary](47_helmet_runtime_dependency_boundary.md)
49. [48 — HELMET two-family offline materializer preflight](48_helmet_two_family_materializer_preflight.md)
50. [49 — HELMET CLINC150 source qualification](49_helmet_clinc_source_qualification.md)
51. [50 — HELMET TREC rights and provenance decision](50_helmet_trec_rights_decision.md)
52. [51 — HELMET Multi-LexSum rights and prompt-determinism decision](51_helmet_multilexsum_decision.md)
53. [52 — HELMET NarrativeQA embedded-work and prompt-path decision](52_helmet_narrativeqa_decision.md)
54. [53 — HELMET InfiniteBench embedded-work and metric decision](53_helmet_infinitebench_decision.md)
55. [54 — HELMET seeded-demonstration repair qualification](54_helmet_seeded_demo_repair.md)
56. [55 — Paper 1 proxy-launch and release-claim boundary](55_paper_1_proxy_launch_boundary.md)
57. [56 — Paper 1 dense control 0 and collection correction](56_paper_1_dense_control_0.md)
58. [57 — Paper 1 dense control 1](57_paper_1_dense_control_1.md)
59. [58 — Event-driven Paper 1 baseline continuation](58_paper_1_event_continuation.md)
60. [59 — Paper 1 dense controls complete and target locked](59_paper_1_dense_controls_and_target_lock.md)
61. [60 — Paper 1 proxy disposition frozen before candidates](60_paper_1_proxy_disposition.md)
62. [61 — Conditional five-cache GQA3/MQA1/MLA design](61_five_cache_representation_design.md)
63. [62 — HCA readiness gate before implementation](62_hca_readiness_gate.md)
64. [63 — CSA selector-readiness gate before implementation](63_csa_readiness_gate.md)
65. [64 — Raw-local branch readiness gate](64_raw_local_readiness_gate.md)
66. [65 — Recurrent/global ratio and placement readiness gate](65_ratio_placement_readiness_gate.md)
67. [66 — Attention Residuals readiness gate](66_attnres_readiness_gate.md)
68. [67 — Stable LatentMoE readiness gate](67_stable_latentmoe_readiness_gate.md)
69. [68 — Tri-axis interaction and removal readiness gate](68_interaction_readiness_gate.md)
70. [69 — Scaling and held-out prediction readiness gate](69_scaling_readiness_gate.md)
71. [70 — Systems-cost readiness and proxy envelope failure](70_systems_cost_readiness_gate.md)
72. [71 — Recent novelty landscape and surviving hypotheses](71_novelty_landscape_audit.md)
73. [72 — Novelty-baseline code and license availability](72_novelty_code_availability.md)
74. [73 — Ada-KV immutable static code audit](73_adakv_static_code_audit.md)
75. [74 — Paper 1 KDA/GQA candidate pair 0](74_paper_1_candidate_0.md)
76. [75 — Adaptive cache budget clean-room reference](75_adaptive_cache_budget_reference.md)
77. [76 — Adaptive cache GQA reduction reference](76_adaptive_cache_gqa_reference.md)
78. [77 — Adaptive cache safeguard apportionment reference](77_adaptive_cache_safeguard_reference.md)
79. [78 — Adaptive cache salience-acquisition readiness](78_adaptive_cache_salience_readiness.md)
80. [79 — N2 direct-overlap audit and scope reduction](79_n2_direct_overlap_audit.md)
81. [80 — HeadKV and routing-artifact availability audit](80_n2_code_availability.md)
82. [81 — N2 concept rejection after evidence-compression audit](81_n2_concept_rejection.md)
83. [82 — STEC adjacent evidence-verification baseline](82_stec_baseline.md)

Conventions:

- Losses are natural-log cross entropy (“nats”) unless stated otherwise.
- `K` means 1,024 tokens in context lengths; token budgets are written exactly.
- Resident state is model state needed across decoding steps. Peak allocation includes runtime
  workspaces and temporary tensors.
- “Effective retrieval” is the longest tested length retaining at least 85% of a statistically
  significant 4K counterfactual directional baseline.
- “Detectable retrieval” is the longest tested length with a one-sided binomial directional test
  at `p < 0.05`, even if it fails the 85% retention rule.
- Internal passkey diagnostics are not RULER, NoLiMa, or HELMET results.
- A difference below the measured `0.00965`-nat seed range is treated as unresolved on one seed.

Last consolidated state: KDA/sigmoid/NoPE with five global GQA caches remains the conservative
research control, not a release selection. Its target-specific association signal transfers across
held-out wording and load, but exact eight-record decoding and symbolic two-hop composition remain
fragile. Independent suites and genuine 64K–128K dependency data are still required before any
long-length promotion.

Finding 24 closes the Speck Reader Attention frontier. Sharing is a real systems trade, not a free
win: three caches cut persistent 128K state `1.66×` and reproduce a thermally controlled `1.31×`
eager decode gain at 524,288 cached slots, while prefill is unchanged between cache-count arms.
Reader-to-writer distance four fails retrieval at fixed fan-out one, establishing the need for
periodic nonlinear refresh. The three-cache candidate fixes the retention cliff and preserves
specificity on 3/3 seeds, but strict paired loss and candidate promotion each pass only 2/3 seeds,
and the symbolic diagnostic exposes a route-edge failure (`0.53` versus `1.00` for five caches).
It is therefore retained as a paper mechanism and research candidate but not promoted into the lead
architecture. MQA/MLA interaction work remains gated.

Finding 25 replaces informal equivalence reasoning with the versioned
[`architecture-promotion-v1`](../research/architecture-promotion-v1/) contract. It defines paired
one-sided non-inferiority, staged replication, realized systems thresholds, named RTX 3090 cost
envelopes, frozen evaluation revisions, and a live evidence matrix. External long-context integrations,
200-case internal confirmation suites, and a production hardware profile remain explicit blockers.

Finding 26 establishes the first-paper program without selecting an architecture. It organizes
candidate research across sequence, depth, and width; requires a Speck-specific novelty rather than a
bundle of published mechanisms; mandates isolated, interaction, scaling, systems, and mechanistic
evidence; and blocks paper-scale pretraining until ten checked prerequisites pass.

Finding 27 audits five historical sequence controls without granting them retrospective promotion
authority, materializes a 0.0118%-parameter-matched dense/KDA baseline pair across three disjoint
seed/data-order cells, and records a 9.40GiB storage deficit against the proxy launch floor. Training
remains blocked pending evaluation-manifest closure, storage-provisioning provenance closeout, and
paired GPU/export preflight; a later audit passes the numeric 16GiB capacity floor.

Finding 28 freezes the paired baseline analysis before results. It makes the three seed/data-order
cells the statistical units, fixes one-sided language-loss and per-source guardrails, separates
fixed-token/FLOP/steady-time views, locks time-to-quality from controls only, preserves right-censoring,
and forbids interim quality stopping. The proxy still has no architecture-promotion authority.

Finding 29 runs the exact-shape RTX 3090 baseline preflight. Both arms complete finite compiled
training steps within the 16GiB allocation envelope, and CPU Transformers exports pass, but the native
CUDA full-versus-cached decode gate fails. Dense attention has one tail-logit miss with unchanged
argmax; KDA has 8,180/256,000 mismatched logits and only 0.75 token argmax agreement. Training remains
blocked pending a layerwise CUDA decode diagnosis; the observed tolerance is not relaxed post hoc.

Finding 30 classifies the failure across three seeds, three lengths, CUDA/CPU, FLA/Torch KDA, removed
convolution history, and four trained checkpoints. CPU state semantics pass; CUDA BF16 shape-dependent
differences are amplified by depth, about 2.2–3.1× more in random-weight KDA than dense. FLA and
convolution history are not sole causes. Training damps relative error, but KDA still produces three
greedy-generation counterexamples across nine seed/length sentinels. The old full-model elementwise
gate also fails dense and must be replaced only through a new versioned, behavior-linked contract.

Finding 31 executes that replacement control-first on 44 source-balanced validation cases. All KDA
cells pass JS divergence, relative RMS, top-10 overlap, and the 0.5-logit guardrail; narrow token and
0.1-margin misses remain. The five-point exact free-running endpoint passes only 2/12 cells and is
severely underpowered at 33/11 cases. V2 therefore fails without proving uniform KDA inferiority, and
training stays blocked pending numerical remediation or a separately powered v3 contract.

Finding 32 powers v3 from v2 variance, freezes 88 new disjoint cases per length, and makes exact
free-running identity a mandatory risk report rather than a chaotic-path primary. All 60 primary KDA
cells and 12 hard-guardrail cells pass. A new exact-shape baseline preflight consequently passes while
preserving failed v1/v2 artifacts. Evaluation-manifest and storage-provenance gates still block runs.

Finding 33 audits RULERv1's hidden network/package dependencies and replaces them with a 99MB retained
offline bundle: 218 essay sources, the consolidated essay artifact, SQuAD, HotpotQA, the English-word
LFS payload, Wonderwords assets, and NLTK archives. Mixed-rights payloads stay outside Git. Transitive
sources now qualify, while official task/length case generation and candidate execution remain blocked.

Finding 34 qualifies the complete 4K RULER matrix: 13 tasks, 100 cases each, and two byte-identical
offline generations. It also exposes a pinned upstream `qa_2` non-termination at HotpotQA example 7
and applies a hashed, control-flow-only repair that preserves prompts, answers, sources, seed, and
scorer. The 4K cases qualify; five longer matrices and all model capability runs remain blocked.

Finding 35 carries the identical gate through 8K. All task hashes reproduce, every accounted case
fits, and retained storage scales from 17MB to 34MB. The 8K cases qualify; four longer matrices and all
model capability runs remain blocked.

Finding 36 carries the gate through 16K with the same deterministic and network-denied result. The
retained artifact is 69MB and every accounted case fits. Three longer matrices and all model
capability runs remain blocked.

Finding 37 carries the gate through 32K. All reproducibility controls pass, every accounted case fits,
and retained storage remains approximately linear at 140MB. The 64K and 128K matrices and all model
capability runs remain blocked.

Finding 38 carries the gate through 64K with the same result. The retained artifact is 282MB and the
storage projection remains safe. The 128K matrix and all model capability runs remain blocked.

Finding 39 completes the 128K stage and the RULER data-generation matrix. All 78 task/length cells
qualify across two full offline generations per length, representing 7,800 retained and 15,600 checked
cases in a 1.1GB local cache. Candidate-specific execution and scoring remain blocked, so no capability
claim is made.

Finding 40 qualifies HELMET's pinned native Hugging Face path on a real parity-attested Speck export
in a locked CPU/eager environment. Load, truncation, deterministic generation, raw-output shape,
RULER/QA post-processing, and network denial pass. The 34GB data volume, component licenses,
reproducible reranking dependency, candidate exports, and capability execution remain blocked.

Finding 41 closes the reranking dependency with a platform-specific local runtime. Exact pytrec/NIST
sources produce identical wheels across two offline rebuilds, and pinned HELMET retrieval metrics pass.
The 34GB volume, component licenses, dataset-bound processing, candidate exports, and execution remain
blocked.

Finding 42 freezes the NoLiMa legal decision boundary without authorizing use. Adobe's terms are limited
to academic research and teaching and expressly exclude commercial product development or gain. The
empty worktree fetched nothing during audit, but its pre-existing object cache already holds all 16
restricted blobs. Authorized entity acceptance and post-acceptance payload hashing remain blocked.

Finding 43 pins HELMET's required 11.27GB archive, excludes the unused v2 archive, and qualifies all
14 active configs and a 64GiB storage floor without downloading data. The 5.5TB candidate partition
passes capacity but is unmounted; component licenses, download, extraction, loader checks, and execution
remain blocked.

Finding 44 uses that separately mounted physical filesystem to close Paper 1 checkpoint capacity with
positive provenance: six unique proxy paths, both proxy/finalist floors passed, unchanged packed data,
and zero historical evidence moved or deleted. SPE-58 remains the only proxy launch blocker.

Finding 45 executes the frozen exact-token contamination audit over 393,216,000 proxy training tokens
and all 7,800 RULER cases. The full-prompt gate passes, but 28 answer-anchored patterns match and fail
the critical gate; complete reference reconstruction localizes them to `qa_1` and `qa_2`. RULER v1 is
preserved as failed, both tasks are quarantined, and a new manifest version is required before any
candidate execution or uncontaminated capability claim.

Finding 46 freezes the successor before any model output exists. RULER v2 retains 6,600 cases over the
eleven untouched official synthetic tasks and gives the quarantined QA tasks zero primary weight.
HELMET RAG/long-QA becomes the explicit source-document guardrail after its own data, rights, and
contamination qualification. The failed v1 identity remains pinned and external execution remains
blocked.

Finding 47 proves that HELMET's 11GB archive is not a complete execution bundle. Of 105 active entries,
55 are archive-local and 50 load seven external dataset families at runtime without revision pins.
Four source families are incompatible with the pinned `datasets==5.0.1`; long QA and summarization also
need a gated Llama 2 tokenizer, and NarrativeQA/summarization depend on unqualified proprietary judges.
The inventory is now frozen, but archive completion alone cannot authorize HELMET execution.

Finding 48 qualifies the proposed compatibility bridge on two CC-BY-4.0 families. A hash-locked
datasets-3.6 materializer produces deterministic Banking77 Parquet and exactly matches the official NLU
conversion across all 25,715 rows and feature labels. The datasets-5 reader reproduces all retained
identities offline. This qualifies the mechanism for those two sources only; blocked licenses,
tokenizers, judges, and the rest of HELMET are unchanged.

Finding 49 pins the exact CLINC150 `plus` train/validation Parquet, upstream JSON, and CC-BY-3.0
license. All 18,350 rows map exactly and in order to the declared upstream parts, all 151 intent labels
match, and offline datasets-5 reload passes. Three of seven archive-external source families now have
a qualified immutable path, but HELMET ICL remains blocked on TREC and final prompt/contamination
qualification.

Finding 50 audits TREC without acquiring its payload. CogComp distributes the mixed-source labeled
collection but states no license or terms; the Hugging Face card says `unknown`, and NIST's general
download guidance does not grant rights to CogComp's derivative annotations. A raw-HTML v1 failure is
preserved, while v2 proves stable visible-text evidence across volatile Cloudflare wrappers. TREC and
therefore HELMET ICL remain blocked pending written authority or a pre-results manifest replacement.

Finding 51 audits Multi-LexSum without acquiring its 836MB payload. ODC-By covers the database, but
expert summaries/metadata are CC BY-NC and HELMET inserts two training summaries into every prompt and
uses a short summary as the reference. The same path has an unseeded demonstration shuffle. Both scope
authority and prompt determinism fail, independently of the still-blocked tokenizer and judge.

Finding 52 traces NarrativeQA's 1,572 linked works into the 3.23GB embedded Hugging Face conversion:
783 Gutenberg books and 789 movie scripts, most from educational-only script sites. Apache metadata
licensing does not qualify those full texts; non-U.S. Gutenberg status is work-specific. HELMET also
uses an unseeded two-shot demo selection, gated Llama 2 truncation, and a proprietary judge. Zero
payloads were acquired and the RULER-v2 source-document guardrail remains blocked.

Finding 53 completes disposition of the seventh HELMET runtime source. InfiniteBench needs only three
pinned files totaling 562.8MB; its ten long-QA cells have seeded prompts and local metrics, while five
summarization cells replace upstream ROUGE-L-Sum with a proprietary judge. Web-derived novel/summary
rights and Llama 2 truncation remain blocked. Across all runtime sources, three immutable paths qualify
and four sources now have explicit blocked decisions.

Finding 54 qualifies the exact two-line repair for NarrativeQA and Multi-LexSum's unseeded two-shot
selection. Frozen fixtures produce identical full prompts across independent processes and different
Python hash seeds, while changing the declared loader seed changes prompt identity. The repair is not
activated: both datasets' rights, payload, tokenizer, judge, and real-case gates remain blocked.

Finding 55 separates checkpoint-producing proxy execution from checkpoint-consuming release
evaluation. Every statistical, data, hardware, storage, contamination, threshold, and missing-suite
rule is frozen; all six outputs are absent; and the live RTX 3090/storage/host gate passes. The paired
proxy is authorized control-first, while external suite failures, long-context claims, component
attribution, architecture promotion, and paper-scale training remain blocked.

Finding 56 completes and qualifies the first dense control at seed 42/order zero. Its final loss is
2.833646 nats after 131,072,000 tokens. Collection exposed an unattainable raw-token equality: the
runtime evaluates complete 16,384-token batches, yielding 19,988,480 of the requested 20M final
tokens. A pre-candidate correction now derives and validates that exact count without changing any
loss, threshold, stopping rule, or execution order. Two dense controls remain before target lock.

Finding 57 completes dense control 1 at seed 43 and packed offset 536,870,912. Final validation loss is
2.839127 nats, steady training time is 3,260.22 seconds, and all source/identity checks pass. Two of
three control observations are now qualified; target lock and candidate records remain forbidden until
control 2 completes.

Finding 58 freezes event-driven continuation before control 2 completes. Successful final summaries
trigger one-shot collectors that disable themselves and wait on process-exit events rather than poll.
Every result is validated and committed before a fixed successor is scheduled after a single cooldown;
any failure stops the chain. No loss-dependent branch or promotion authority is introduced.

Finding 59 completes the three-control phase. Dense control 2 finishes at 2.832686 nats; the three
controls average 2.835153 with a 0.006440-nat range, while steady-time spread is only 0.135%. The
predeclared worst-control rule locks the candidate time-to-quality target at 2.839127 before any
candidate output exists. Candidate pair 0 is now the next fixed cell.

Finding 60 freezes proxy disposition before candidate 0. Integrity failure, aggregate failure,
source-guardrail failure, and complete quality pass have explicit non-overlapping consequences.
Efficiency views cannot rescue failed quality, censoring stays explicit, and only a complete quality
pass may authorize finalist materialization. No branch grants component attribution or promotion.

Finding 61 freezes the conditional exact-cache representation isolation without authorizing it.
Five independent memories and the KDA/NoPE backbone stay fixed. MQA1 and a Speck-derived MLA128 both
reduce BF16 state by 66.7%; uniform FFN compensation matches MQA1 within 0.00998%, while MLA128 matches
GQA3 projection weights and MQA1 state. Correctness, realized systems thresholds, backbone selection,
and multiplicity-controlled quality remain mandatory.

Finding 62 shows HCA is not yet an implementable architecture choice. The parent representation and
compressor are unselected, and causal partial-block state is unspecified. The new gate requires a
three-arm compressor isolation before a 32/64/128/256 rate curve, exact tail/prefix/resume semantics,
complete state/compute accounting, and realized 20% systems plus 25% state thresholds. No training is
authorized.

Finding 63 separates CSA from HCA, local coverage, and cache representation before implementation.
Block-mass oracle feasibility precedes mean-key and learned selectors; captured probability and
all-required-source recall replace index overlap as primary diagnostics. A 2/4/8 compression,
32/64/128 block, and 512/2,048/8,192 budget grid is conditional. Dense index scan remains `O(L²/m)`,
and token-level routing needs a successor contract. No training is authorized.

Finding 64 freezes a conditional raw-local formulation without authorizing it. Exact local rings occur
only at the five global slots and share one causal, deduplicated softmax with CSA's precise path; HCA and
KDA stay fixed. Windows 64/128/256/512 are selected by local-specific floors, not copied. Local state is
an explicit added cost, and a retained branch must clear the 10% systems threshold without erasing its
parent's benefit.

Finding 65 corrects the impossible 7:1 label in a 20-layer model before ratio results. Exact 1:1, 3:1,
and 9:1 arms use 10, 5, and 2 global layers under one quantile placement rule. Count is selected before
placement; integration/readout layouts require a fixed-count successor. Operator definitions, quality
constraints, parameter/FLOP views, mechanistic roles, and realized systems gates remain explicit, and
no ratio training is authorized.

Finding 66 corrects the AttnRes unit of analysis from 20 logical blocks to 40 ordered residual modules.
The initial four-arm isolation adds a static-depth control beside PreNorm, Full, and eight-block
AttnRes; the bounded arm uses eight five-module blocks and at most nine sources. Exact equations,
activation/recomputation, block-count, three-by-three depth/width, content-dependence, and 10% efficiency
gates are frozen. No implementation or training is authorized.

Finding 67 records that the local K3 review is not an implementation-complete Stable LatentMoE
specification. Width work is frozen as six separate stages: conventional MoE, latent factorization,
normalization, bounded activation, balancing, then geometry. Dropless routing semantics, stability and
rescue evidence, total/active memory, single-device and expert-parallel systems gates are explicit.
Primary-source equations, parents, hardware, implementation, and training remain blocked.

Finding 68 freezes the tri-axis interaction unit as three selected bundles, not their hidden
subcomponents. The complete 2³ discovery cube is 24 fixed runs across three paired cells with explicit
difference-in-differences, three-way, conditional-removal, Holm, aggregate/source, and retuned absence
controls. Every retained subcomponent still needs a final removal, and simpler quality-passing ties are
deleted. No cube or combined architecture is authorized.

Finding 69 separates scale transfer from a fitted scaling-efficiency claim. It freezes five 30M–600M
fit points, symmetric N/D allocation pilots, constrained joint and compute fits, complete-refit paired
bootstrap uncertainty, horizon interaction, and a truly held-out 1.2B/20B sentinel. The current
sub-one-token-per-parameter proxy cannot support scaling claims. Architecture, hardware, budgets, fits,
sentinel, and paper-scale execution remain blocked.

Finding 70 preserves a negative cost result: all three dense controls pass 40K tok/s, 16 GiB, and
finite-state limits but exceed the frozen 1.0 total GPU-hour envelope at 1.0455/1.0129/1.0114 hours. The
threshold is not widened or relabeled. Analytic, operator, model, serving, energy/memory, and monetary
evidence are separated; online and datacenter profiles plus price inputs remain blocked. This does not
alter the fixed proxy quality experiment.

Finding 71 audits recent direct novelty overlaps from pinned primary arXiv records. Joint hybrid-layer
selection, adaptive cache budgeting, sparse recurrent prefix checkpoints, budget-conditioned attention,
and alternating local/global latent sparse attention are not novel alone. Only a role-grounded
from-scratch placement law and an all-required-source composition predictor remain provisionally
distinct; both still require full landscape and held-out causal evidence. No novelty is claimed.

Finding 72 checks immutable code trees and rights without execution. FlashMorph's declared repository
has only a README/images; Sparse Prefix and Budgeted Attention declare no code; ASA links only the base
NSA kernel; SqueezeAttention has code but no root license. Ada-KV alone has a pinned code tree and root/
CUDA MIT licenses, making it eligible for deeper non-executing audit. No reproduction or novelty status
changes.

Finding 73 completes that deeper Ada-KV static audit without cloning/execution. Its environment is not
portable: local-path/older-SSH locks, Transformers contract mismatch, native install side effects, global
monkeypatching, dynamic CUDA cache code, unpinned remote model/data, and absent tests all block execution.
Only a clean-room exhaustive allocation reference is authorized next; upstream reuse remains blocked.

Finding 74 completes candidate pair 0 at 2.794477 nats, a descriptive -0.039170 paired loss difference;
all eleven sources favor the candidate. Steady/active time is 10.09%/11.13% lower and analytic FLOPs are
21.49% lower, while peak allocation is 20.84% higher. One pair has no decision authority. The event chain
committed it and scheduled pair 1 without branching on quality.

Finding 75 qualifies a clean-room Ada-KV allocation reference without running upstream code. Across 200
normalized matrices, all 1,700 budgets, 28,240 exhaustive feasible allocations, and 84 adversarial
budget cases, global top-salience allocation conserves capacity, is deterministically optimal, dominates
the quotient/remainder uniform control, and makes the paper bound monotone within `8.88e-16` numerical
noise. This is reference evidence only; GQA reduction, model integration, training, novelty, and
architecture promotion remain blocked.

Finding 76 qualifies arithmetic-mean GQA reduction only under equal-size groups. Across 600 tensors,
4,200 budgets, and 22,000 physical allocation oracles—including 500 Speck GQA3 cases—mean and sum have
identical allocation identities and attain maximal original query-head retained mass. Max loses a
frozen counterexample by 0.1 mass, and code-like safeguard rounding loses one of six slots. Model
integration, safeguard use, training, novelty, and architecture promotion remain blocked.

Finding 77 replaces an ambiguous fractional safeguard with a precisely named uniform fraction and exact
largest-remainder apportionment. All 14,120 cases and 952,660 feasible-allocation comparisons pass with
zero conservation error, exact L1/L2 optimality, deterministic ties, valid endpoints, and a uniform
lower floor. This is only a control primitive: upstream equivalence, quality benefit, primary use,
model integration, training, novelty, and architecture promotion remain blocked.

Finding 78 blocks observation-window implementation after separating instrumentation from the cache
mechanism. Speck exposes no attention weights, uses a fixed rectangular chronological ring, and lacks
variable-length per-head kernels. A successor must first pass an offline attention-probe parity gate,
then isolate question-visible, reusable context-only, and hindsight modes; pooling; window/kernel;
allocation; cache lifecycle; source-completeness prediction; causal restoration; and realized systems
cost. No implementation or evaluation is authorized before the three-pair parent decision.

Finding 79 adds two direct N2 overlaps and narrows the surviving claim. HeadKV-R2 already profiles
retrieval-plus-reasoning heads and globally allocates cache; a 2026 routing study already separates
retention, accessibility, and utilization with GER, consensus, token graphs, and multi-hop probes. N2
now concedes all of that and survives only as conjunctive availability of every independently necessary
route/payload group, with incremental held-out prediction and fixed-budget single-source restoration.
No novelty is established.

Finding 80 pins HeadKV at `0862a095` and records 426 files, 37 source files, 376 data-like files, no
root license, an unpinned/misnamed environment, global monkeypatches, native binaries, unqualified
profiles/data, and no assertion suite. The routing paper declares no dedicated repository/data or
immutable KVPress/Expected-Attention revisions. Neither reproduction path is authorized; novelty status
does not change.

Finding 81 rejects novelty of the N2 source-conjunction concept. BRIEF already requires a helpful
proposition for every hop from distinct documents; BRIEF-Pro scales source-aware compression beyond 10K
words with budgets and one-missing-hop analysis; IterCOMP explicitly tests Full versus Partial hop
evidence, judges sufficiency, identifies the missing hop, and retrieves it iteratively. Only an
unestablished cheap internal-state predictor with incremental held-out and equal-budget causal evidence
remains reviewable; no novelty is established.

Finding 82 classifies STEC as an adjacent mandatory baseline: it compresses multi-trajectory supporting
and conflicting evidence plus reasoning paths for constrained candidate verification, but does not
measure pre-output internal cache-source survival or equal-state interventions. It strengthens the
external comparison without changing the N2 concept rejection or establishing the residual empirical
law.
