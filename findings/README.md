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
84. [83 — BRIEF-family artifact availability audit](83_brief_code_availability.md)
85. [84 — Claim-granular novelty overlap and priority](84_novelty_claim_overlap.md)
86. [85 — Paper 1 KDA/GQA candidate pair 1](85_paper_1_candidate_1.md)
87. [86 — N1 role-overlap audit and scope reduction](86_n1_role_overlap.md)
88. [87 — N1 released-artifact availability audit](87_n1_code_availability.md)
89. [88 — Novelty priority correction after N1 audit](88_novelty_priority_correction.md)
90. [89 — Systematic from-scratch hybrid placement overlap](89_systematic_hybrid_placement_overlap.md)
91. [90 — Systematic hybrid-study artifact availability](90_systematic_hybrid_artifact_availability.md)
92. [91 — N1 claim-table correction after placement overlap](91_n1_claim_table_v3.md)
93. [92 — Hybrid massive-activation and placement overlap](92_hybrid_massive_activation_overlap.md)
94. [93 — Hybrid massive-activation artifact audit](93_hybrid_massive_activation_code_audit.md)
95. [94 — Novelty artifact inventory after massive-activation audit](94_novelty_code_availability_v6.md)
96. [95 — N1 claim table after activation-diagnostic counterevidence](95_n1_claim_table_v4.md)
97. [96 — HALO source and result-log audit](96_halo_code_audit.md)
98. [97 — HALO task-guided layer-selection overlap](97_halo_layer_selection_overlap.md)
99. [98 — Novelty artifact inventory after HALO audit](98_novelty_code_availability_v7.md)
100. [99 — N1 claim table after HALO](99_n1_claim_table_v5.md)
101. [100 — Cross-task, cross-mixer KL-guided placement overlap](100_kl_guided_selection_overlap.md)
102. [101 — KL-guided selector source/config audit](101_kl_selection_code_audit.md)
103. [102 — Novelty artifact inventory after KL-selector audit](102_novelty_code_availability_v8.md)
104. [103 — N1 claim table after cross-task KL selection](103_n1_claim_table_v6.md)
105. [104 — Frozen N1 independent-review packet](104_n1_independent_review_packet.md)
106. [105 — Paper 1 three-pair proxy quality screen](105_paper_1_paired_proxy_analysis.md)
107. [106 — Six-pair finalist analysis freeze](106_finalist_analysis_freeze.md)
108. [107 — Finalist materialization boundary](107_finalist_materialization_freeze.md)
109. [108 — Six-pair finalist configs materialized](108_finalist_materialized.md)
110. [109 — Finalist materialization, data, and storage qualification](109_finalist_data_storage_qualified.md)
111. [110 — Finalist collector and analysis implementation qualified](110_finalist_analysis_qualified.md)
112. [111 — Exact finalist CUDA runtime preflight](111_finalist_runtime_preflight.md)
113. [112 — Multi-day finalist launch boundary](112_finalist_launch_boundary.md)
114. [113 — Event-driven finalist automation frozen](113_finalist_automation_frozen.md)
115. [114 — Finalist crossed-factor inference correction](114_finalist_crossed_factor_correction.md)
116. [115 — Crossed-factor finalist analysis v2 qualified](115_finalist_analysis_v2_qualified.md)
117. [116 — Finalist launch v2 after crossed-factor correction](116_finalist_launch_v2.md)
118. [117 — Crossed-factor finalist automation v2](117_finalist_automation_v2.md)
119. [118 — HELMET archive acquired and hash-qualified](118_helmet_archive_acquired.md)
120. [119 — HELMET archive path inventory qualified](119_helmet_archive_inspected.md)
121. [120 — Finalist v2 live launch qualification](120_finalist_live_launch_qualified.md)
122. [121 — Finalist control 0 operator interruption](121_finalist_control0_operator_interruption.md)
123. [122 — Identical finalist control 0 restart frozen](122_finalist_control0_rerun_frozen.md)
124. [123 — Fresh live gate for finalist control 0 attempt 2](123_finalist_rerun_live_gate.md)
125. [124 — HELMET archive-local rights and provenance audit](124_helmet_archive_local_rights_audit.md)
126. [125 — HELMET synthetic-recall reconstruction boundary](125_helmet_synthetic_reconstruction_readiness.md)
127. [126 — HELMET real-data reconstruction boundary](126_helmet_real_data_reconstruction_readiness.md)
128. [127 — HELMET truncation-tokenizer replacement gate](127_helmet_truncation_tokenizer_readiness.md)
129. [128 — HELMET model-judge reproducibility and data boundary](128_helmet_model_judge_readiness.md)
130. [129 — Full finalist automation transition simulation](129_finalist_automation_transition_audit.md)
131. [130 — Crossed-factor audit of every finalist analysis view](130_finalist_crossed_views_audit.md)
132. [131 — Finalist systems-measurement and temporal-confounding boundary](131_finalist_systems_measurement_boundary.md)
133. [132 — Finalist finiteness and complete-source boundary](132_finalist_source_stability_boundary.md)
134. [133 — Append-only finalist result-acceptance gate](133_finalist_result_acceptance_gate.md)
135. [134 — Complete finite source coverage moved before commit](134_finalist_program_source_gate.md)
136. [135 — One-control integration proof for the pre-commit source gate](135_finalist_program_source_gate_integration.md)
137. [136 — Retained-checkpoint replay closes the result-provenance gap](136_finalist_checkpoint_provenance_replay.md)
138. [137 — Exact append-only Git provenance for every finalist event](137_finalist_append_only_git_provenance.md)
139. [138 — Post-language systems protocol frozen before finalist results](138_finalist_systems_protocol_frozen.md)
140. [139 — Fail-closed systems analysis qualified offline](139_finalist_systems_analysis_qualified.md)
141. [140 — Conservative systems telemetry integration qualified synthetically](140_finalist_systems_telemetry_integrator.md)
142. [141 — Systems sampler implementation qualified without a live query](141_finalist_systems_sampler_mock_qualified.md)
143. [142 — Exact systems workload plan and persistent-mutation detector](142_finalist_systems_workload_plan.md)
144. [143 — Disposable kernel read-only isolation qualifies](143_finalist_systems_sandbox_qualified.md)

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

Finding 83 pins the combined BRIEF repository at `07794332`: 601 files, of which 492 are vendored
Axolotl. The old BRIEF subtree has 49 files; BRIEF-Pro adds about 31 non-vendored files. No root license
or paper-specific historical revisions exist, environments conflict and include a machine-local prefix,
and model/data derivations remain unpinned. Across twelve sources, only the prior Ada-KV tree has root
rights and no new reproduction path is authorized.

Finding 84 maps fourteen candidate claims to their strongest prior-art overlaps. Twelve are direct or
adjacent baselines and cannot be claimed; N1's prospective from-scratch placement law is the only
primary architecture hypothesis left. N2's concept is rejected and its possible internal-state
empirical law is deferred before protocol or experiment budget pending independent expert review. No
novelty or architecture is established.

Finding 85 completes candidate pair 1 at 2.796465 nats, a descriptive -0.042662 paired loss difference;
all eleven sources again favor the candidate. Steady/active time is 10.12%/10.16% shorter and analytic
FLOPs are 21.49% lower, while peak allocation is 20.74% higher. Two pairs still have no decision
authority. The one-shot finalizer committed the result and scheduled pair 2 after the fixed cooldown.

Finding 86 rejects N1's broad role novelty. A five-scale from-scratch study already shows middle full
attention carries retrieval while efficient mixers shape its learning trajectory; a 72-model study
crosses mixer and uniform ratio; DtR greedily selects non-uniform conversion layouts and shows static
probes miss interactions. Only prospective prediction of unseen non-uniform from-scratch layouts may
remain, as an unestablished empirical law. No placement experiment or architecture freeze is authorized.

Finding 87 pins the hybrid-role repository at `feaad089`: 20 MIT-licensed files with analysis scripts,
but no training/config pipeline, referenced receptive-field implementation, complete scaling tables,
immutable checkpoint/data manifests, pinned environment, or tests. The 72-model collection is visible
but unaudited, and DtR declares no code. Fifteen sources now yield two root-rights paths and zero new
full reproductions; N1 remains conceptual only.

Finding 88 corrects the priority table: there are zero established architecture-novelty candidates.
N1 and N2 survive only as unestablished empirical-law questions, with no experiment authority; N2 stays
deferred and N1 receives landscape/artifact/expert review only. The fixed Paper 1 proxy continues because
it estimates the conservative parent independently of novelty, but even a proxy pass cannot satisfy the
separate novelty gate.

Finding 89 adds the closest direct N1 overlap: a 60B-token from-scratch study crosses Transformer/Mamba
ratios and early/middle/late placements at 350M and 1B, finds a consistent front-attention penalty, and
links it to early uniform attention versus Mamba locality. Non-uniform placement and the middle/later
recipe are prior art. Only prospective prediction of arbitrary unseen layouts remains technically
distinct, with no demonstrated value and no experiment authority.

Finding 90 records that the systematic placement paper declares no official code, model, exact-config,
or immutable data-order artifact. It remains a mandatory conceptual baseline but adds no qualified
reproduction path. Across sixteen sources, only two repositories have root code rights; no N1
execution or placement protocol is authorized.

Finding 91 updates the claim table after direct placement overlap. From-scratch non-uniform placement
and the avoid-early-attention/middle-later recipe are explicit do-not-claim rows. Prospective prediction
of arbitrary unseen layouts is only a technical residual with no demonstrated feasibility or scientific
value; independent review may retire it, and no placement experiment is authorized.

Finding 92 adds controlled massive-activation evidence across hybrid depth. Pre-attention spikes and
inter-spike plateaus recur across five mixers and large public hybrids; matched early/middle/late GDN
placements all reach nearly perfect spike alignment despite large retrieval gaps. The diagnostic is
occupied and does not rank placement quality, further lowering N1's prior. Independent review may
retire the residual; no placement experiment is authorized.

Finding 93 pins the official massive-activation analysis tree and aggregate checkpoint tree without
checkout or execution. MIT code rights, Apache-2.0 model-card rights, ten named checkpoint directories,
package pins, and tests are visible. Training code, gated-full-attention behavior, exact sampled inputs,
and table parity remain missing, so neither upstream execution nor full reproduction is authorized.

Finding 94 updates the artifact inventory to seventeen sources, eight immutable code/checkpoint
snapshots, six repositories with code, and three root-licensed code paths. The massive-activation
release may support bounded analysis later, but no new full reproduction is qualified and no download,
execution, or N1 protocol is authorized before independent review.

Finding 95 updates N1's claim rows with PAS/ISP direct overlap and saturated diagnostic
counterevidence. The remaining multidiagnostic prospective-prediction question has a low prior and is
not an architecture contribution. It now requires independent retention-or-retirement review before
any further artifact execution, protocol, experiment, or training budget.

Finding 96 pins HALO's 271-file source tree, including training/selection code and 206 layer-sweep
logs. The config resolves HypeNet-2B as seven attention plus twenty-one recurrent layers, but the code's
epsilon/clipping differs from the paper, and root rights, tests, dependency/data/checkpoint identity,
and behavior are missing. No upstream execution, reuse, or reproduction is authorized.

Finding 97 adds HALO as a direct N1 baseline: each recurrent substitute is hidden-state aligned, every
single replacement is scored on recall versus commonsense outcomes, and the top quarter of attention
layers are retained. The only remaining distinction is held-out, from-scratch, interaction-aware
prediction—a procedural, low-prior residual pending independent retention-or-retirement review. No new
placement work is authorized.

Finding 98 updates the artifact inventory to eighteen sources, nine immutable source/checkpoint
snapshots, seven repositories with code, and three root-licensed code paths. HALO's selection logs are
useful inspected evidence, but rights and behavioral identity block execution/reuse/reproduction. No
new full reproduction path exists, and independent N1 review precedes remediation.

Finding 99 updates the claim table with HALO's task-guided conversion selection and HypeNet's
composition bundle. The only formal N1 distinction is held-out prediction of jointly trained
from-scratch layout interactions—a procedural residual with no established architecture value. N1 has
no experiment authority and requires independent retention-or-retirement review.

Finding 100 adds generic-text KL selection with held-out recall, cross-scale Qwen evidence, GDN-to-GLA
probe transfer, teacher-dependent clustering, spacing interventions, and a ranking-stability early stop.
Those N1 distinctions are occupied. Only teacher-free prediction of interacting from-scratch layouts
remains, and retirement is evidence-favored pending independent review. No placement work is authorized.

Finding 101 pins the KL-selector's 203-file tree with 24 Python files and 171 experiment configs. The
paper specifies held-out KL, but released ranking code consumes W&B training loss; the early-stop rule,
root rights, tests, deterministic data, checkpoint/log identity, and full parity are absent. No upstream
execution, reuse, reproduction, or N1 protocol is authorized.

Finding 102 updates the artifact inventory to nineteen sources, ten immutable source/checkpoint
snapshots, eight repositories with code, and three root-licensed code paths. The KL-selector tree adds
config evidence but no qualified reproduction path; execution, reuse, and behavior remain blocked.

Finding 103 updates N1's claim table after cross-task KL selection. Held-out-task, cross-scale,
cross-mixer, clustered-layout, and spacing-intervention stories are occupied. Only teacher-free
prediction of jointly trained from-scratch interactions remains—a procedural residual with retirement
evidence-favored pending independent review. No N1 experiment is authorized.

Finding 104 freezes an independent-review packet for N1 before any future outcome. Two separate domain-
qualified reviews must identify a non-procedural causal claim, unique prediction, falsifier, baselines,
replication, and evidence cost. Only unanimous support permits a preregistration draft; disagreement
requires a third review. No review, execution, protocol, architecture freeze, or training is authorized.

Finding 105 completes the frozen three-pair proxy. Mean candidate-minus-control loss is -0.039672 nats
with upper one-sided 95% bound -0.034996; all eleven source bounds pass and no time-to-quality pair is
censored. Mean time-to-quality improves 19.48% (lower bound 16.83%); endpoint steady time is 10.01%
shorter and analytic FLOPs/token 21.49% lower, while peak allocation is 20.77% higher. The quality screen
passes and authorizes finalist materialization/qualification only—not component attribution, promotion,
novelty, release claims, finalist training, or paper-scale execution.

Finding 106 freezes the eligible finalist analysis before config creation: six crossed seed/data-order
pairs, twelve 1.540B-token runs, exact quartile validation, df=5 one-sided bounds, unchanged aggregate/
source margins, six controls before target lock, and zero interim looks. Materialization is authorized;
training, component attribution, promotion, novelty, and scale/release claims remain blocked.

Finding 107 freezes finalist materialization before implementation: SHA-pinned proxy parents, a new
family/root, six seed-by-order pairs, twelve 1.540B-token runs, final-only checkpoints, exact quartile
validation, fail-on-overwrite behavior, disjoint/replayable data windows, and fresh storage/runtime/
analysis qualification. Config work is authorized; training and promotion remain blocked.

Finding 108 materializes exactly 84 hashed configs plus one manifest for the six-pair, twelve-run
finalist. Every run resolves the frozen 1.540B-token horizon and a unique absent checkpoint path;
materialization revalidation passes. No checkpoint/result path exists and the manifest keeps training
blocked pending data, storage, runtime, collector/analysis, and release-dependency qualification.

Finding 109 qualifies finalist configs, output absence, data, and storage. Both 1.540B-token windows
are disjoint, crossed with all three seeds, and replay byte-identically at ten start/quartile/end points.
All twelve outputs remain absent; the dedicated device has 5.64 TB free versus a 25.77 GB floor with no
deletions. Runtime, collector/analysis, release suites, training, and automatic launch remain blocked.

Finding 110 qualifies the finalist collector, six-control target lock, df=5 paired analysis, censoring,
and stopping-rule implementation through six focused fixtures and lint. All implementation and contract
files are pinned. Exact CUDA runtime and release dependencies remain blocked; no training, automatic
launch, attribution, promotion, or paper-scale work is authorized.

Finding 111 qualifies exact finalist compiled CUDA steps, temporary Transformers exports, and the
powered trained-topology cache reference. Dense/candidate peaks are 10.24/14.14 GiB. The random-weight
native diagnostic still fails elementwise, and candidate argmax agreement is only 87.5%; this has no v2
pass/fail authority but remains a mandatory risk. Release gates, training, and automatic launch stay
blocked.

Finding 112 freezes the multi-day finalist launch boundary. Missing release suites remain failed
promotion gates but do not circularly block checkpoint production. The 121.23 projected steady GPU-
hour sequence is six controls, target lock, six candidates, and final analysis via one-shot events,
15-minute gaps, no polling/branch/retry, and per-result commits. Automation may be built; launch remains
blocked while its source/hash is absent and HELMET acquisition is active.

Finding 113 pins and qualifies the finalist event runner before output. Five tests enforce six controls,
target lock, six candidates, final analysis, and invalid-state rejection. Every successor uses one path
event plus live repository/GPU/memory/storage/HELMET gates, 15-minute cooldown, no polling/branch/retry,
and per-result commits. Training waits for a post-HELMET live qualification.

Finding 114 catches a crossed-factor inference flaw before finalist output. Six seed-by-order cells are
not six independent replicates. V2 requires separate df=2 bounds over three seeds within each fixed data
order, both order strata and every cell/source guard to pass, and makes pooled df=5 output descriptive
only. Configs do not change; v1 analysis/automation authority is superseded and training stays blocked.

Finding 115 qualifies the v2 collector and crossed-factor analyzer through seven fixtures. Both fixed
orders receive separate three-seed df=2 bounds plus every-cell/source guards; pooling is descriptive.
An adversarial pooled-zero case fails when one order is +0.02. Configs and runtime remain unchanged;
automation v1 is invalid and training waits for automation v2 plus post-HELMET live qualification.

Finding 116 supersedes the launch boundary before output and binds the unchanged twelve-run, 121.23-
hour event sequence to analysis v2. V1 launch/automation authority is false. Config, runtime, release,
and failure contracts are unchanged; automation v2 and a post-HELMET live qualification are required
before training.

Finding 117 pins automation v2 to the corrected analysis, qualification, launch boundary, and final
status. Five state tests and seven analysis tests pass; the same twelve-run no-poll/no-branch/no-retry
chain remains. V1 is unauthorized, all outputs are absent, and training still requires a post-HELMET
live qualification.

Finding 118 completes the resumable HELMET archive acquisition in 2,622.97 seconds. The 11,271,916,108-
byte file matches the frozen SHA-256, the isolated volume retains 5.63 TB free, and no credential or
discard occurred. Extraction/execution remain blocked pending safe inventory and license review; the
inactive transfer now permits a separate finalist v2 live qualification.

Finding 119 reproduces the HELMET tar inventory twice without extraction. All 175 members are safe,
154 files total 35.63 GB uncompressed, all 52 declared local paths exist, and 50 entries remain external.
The archive contains zero license/README metadata, so component rights and extraction remain blocked,
along with runtime datasets, tokenizer, judges, and candidate execution.

Finding 120 records a clean v2 live gate: the RTX 3090 is idle at 45°C, 23.70 GB host memory and 5.63
TB volume space are available, required mount options hold, all 48 finalist units and unfinished outputs
are absent, and HELMET is inactive. The exact first dense control and event successors are authorized;
concurrency, branching, retry, attribution, promotion, novelty, release, and paper-scale work are not.

Finding 121 preserves an operator-interrupted first dense attempt. V2 was frozen before launch, but the
service was mistakenly stopped after step 1. Dense initial/step loss and all local W&B hashes are
reported; no checkpoint/result path exists and the attempt cannot enter analysis. No silent retry is
allowed: an identical-cell restart contract and fresh live gate are required.

Finding 122 freezes attempt 2 as an identical step-0 restart of dense pair 0. Every config, analysis,
margin, seed, order, and horizon stays fixed; attempt-1 observations are listed and ineligible. This is
a manual preregistered recovery, not automatic retry. Training remains false until the failed attempt,
rerun contract, and a fresh live gate are registered.

Finding 123 records the fresh attempt-2 gate after recovery registration. The failed attempt and rerun
are present in the empty ledger; all units/outputs remain absent, the GPU is idle at 45°C, host memory
and 5.63 TB storage pass, and HELMET is inactive. The identical restart is authorized without granting
automatic retry, branching, attribution, promotion, novelty, release, or paper-scale authority.
