# Findings relevant to the active flagship

The current model, paper, and compute plan live under
[`research/flagship/`](../research/flagship/). This page routes to the evidence that constrains those
defaults without presenting the retired 169-entry program as current work.

Checked JSON under `results/` remains the machine-readable source of truth. The
[`complete historical index`](ARCHIVE.md) preserves every finding, including failed and retired gates.
Chronological attempts and discussion belong in [`research/notebook/`](../research/notebook/); a
finding is added here only after evidence supports a stable conclusion or decision. Paper-level claim
status is tracked in [`paper/claims.json`](../paper/claims.json).

## Foundation

| Evidence | Why it matters now |
| --- | --- |
| [Research contract](00_research_contract.md) | Reproducibility, controls, and claim boundaries |
| [Mixer screen](01_mixer_screen_131m.md) | Initial recurrent/global architecture comparison |
| [Correctness and kernel calibration](02_correctness_and_kernel_calibration.md) | Reference/optimized parity boundary |
| [Seed resolution](03_seed_noise_floor.md) | One-seed differences below 0.00965 nats are unresolved |
| [Existing 128K systems frontier](04_existing_checkpoint_128k_systems.md) | State and latency accounting |
| [Long-document dataset](05_long_document_dataset.md) | Complete-document extension data |
| [Global-layer frontier](08_global_attention_frontier.md) | Middle integration and final readout roles |

## KDA and NoPE selection

| Evidence | Why it matters now |
| --- | --- |
| [KDA implementation and qualification](11_kda_implementation_and_qualification.md) | Current recurrent operator and kernel contract |
| [Same-parent NoPE activation](12_nope_context_activation.md) | Correct positional-treatment protocol |
| [Synthetic MQAR](13_synthetic_mqar.md) | Associative-memory calibration |
| [Distance/load scaling](14_mqar_length_scaling.md) | Capacity limits across length and load |
| [Palindrome and stack](15_palindrome_and_stack.md) | Non-retrieval recurrent-state tests |
| [Kimi-transfer staircase](16_kimi_transfer_131m.md) | Isolated gate, position, and decay changes |
| [Three-seed KDA/NoPE replication](17_kimi_frontier_replication.md) | Short-loss uncertainty and replicated 128K signal |
| [Matched 32K activation](18_kimi_context32k.md) | Extension quality and 4K retention |
| [Selectable KDA output gate](198_kda_output_gate_successor.md) | Default-sigmoid compatibility and explicit SiLU successor |

## Evaluation and negative results

| Evidence | Why it matters now |
| --- | --- |
| [Retrieval specificity and replay](19_retrieval_specificity_and_replay.md) | Separates sensitivity from usable retrieval |
| [Attention output gating](20_k3_diagnostics_and_attention_gating.md) | Rejected marginal complexity |
| [Answer transfer failure](21_retrieval_template_transfer.md) | Held-out answer/template requirement |
| [Template-diverse adaptation](22_template_diverse_retrieval_adaptation.md) | Adaptation protocol evidence |
| [Symbolic composition](23_symbolic_two_hop_composition.md) | Retrieval does not imply reasoning composition |
| [Reader Attention frontier](24_reader_attention.md) | Preserved negative architecture result |
| [Three-pair dense/KDA proxy](105_paper_1_paired_proxy_analysis.md) | Whole-architecture quality, FLOP, and time signal |
| [Program transition](168_finalist_stopped_flagship_plan.md) | Why the retired finalist work no longer governs |

## Flagship data qualification

| Evidence | Why it matters now |
| --- | --- |
| [Stack v3 bounded qualification](169_stack_v3_bounded_qualification.md) | Pinned repository-aware source adapter and yield boundary |
| [Stack v3 refinement](170_stack_v3_security_language_refinement.md) | Secret, language, license, and partition gates |
| [Stack v3 benchmark screen](171_stack_v3_code_decontamination.md) | Frozen code-contamination policy and cleaned source |
| [Stack-Edu acquisition](172_stack_edu_bounded_swh_sample.md) | SWH identity, missing-blob, and metadata-length evidence |
| [Code-source overlap](173_code_source_overlap_and_precedence.md) | Bounded exact/fuzzy comparison and blend precedence |
| [Code contamination successors](174_code_contamination_successors.md) | Stack-Edu/Common Pile benchmark cleanup and final bounded code inputs |
| [Code tokenizer supplements](175_code_tokenizer_supplements.md) | Python-Edu/PEP qualification and the complete five-source bounded code slice |
| [Web tokenizer sources](176_web_tokenizer_bounded_sources.md) | Four-source bounded sampling, security, partition, and overlap evidence |
| [Web evaluation firewall](177_web_evaluation_firewall.md) | Frozen flagship payloads and decontaminated bounded web successors |
| [Web rights review packet](178_web_rights_review_packet.md) | Hashed terms, unresolved rights chain, and required human acceptance record |
| [Math tokenizer sources](179_math_tokenizer_technical_qualification.md) | Six-source bounded technical qualification and firewall result |
| [Math rights review packet](180_math_rights_review_packet.md) | Hashed terms, code-license chain, and required human acceptance record |
| [Synthetic tokenizer sources](181_synthetic_tokenizer_technical_qualification.md) | Corrected lineage, synthetic quality gates, and firewall result |
| [Synthetic rights review](182_synthetic_rights_review_packet.md) | Generator, seed-source, and redistribution decision packet |
| [Science rights review](183_science_rights_review_packet.md) | Paper/PDF license metadata gaps and required human decision |
| [Science tokenizer sources](184_science_tokenizer_technical_qualification.md) | Five-source bounded qualification and fail-closed firewall successor |
| [Reference tokenizer sources](185_reference_tokenizer_technical_qualification.md) | Six-source bounded qualification and corrected source contracts |
| [Reference rights review](186_reference_rights_review_packet.md) | Attribution, share-alike, public-domain, and removal decisions |
| [Three-partition firewall](187_three_partition_firewall_tooling.md) | Fixture-qualified partition, consumer, and one-opening enforcement |
| [Production data tooling](188_production_data_tooling.md) | Disk-backed dedup, removal, cleanup, and resume fixture evidence |
| [Source-rights decision readiness](189_source_rights_decision_readiness.md) | Consolidated 30-source human acceptance contract |
| [Data launch gate](190_data_launch_gate.md) | Exact authority binding before flagship model construction |
| [Data rehearsal orchestration](191_data_rehearsal_orchestration.md) | Durable six-stage 20B runner qualified only on fixtures |
| [Full-size tokenizer fixture](192_tokenizer_fullsize_fixture.md) | Three custom sizes and pinned Mistral pass non-selecting plumbing checks |
| [Tokenizer static nomination](193_tokenizer_static_nomination_policy.md) | Pre-results Pareto endpoint rule for two LM-pilot finalists |
| [Tokenizer pilot analysis](194_tokenizer_pilot_analysis.md) | Seven-run BPB, compute, guardrail, and D5 handoff contract |
| [Local tokenizer data study](195_local_tokenizer_data_study.md) | Whitespace-piece support reverses the bounded custom-tokenizer deficit |
| [Runtime cleanup successors](196_runtime_cleanup_successors.md) | Failure-safe checkpoint replacement and production handle reuse |
| [Tokenizer v2 contract](197_tokenizer_v2_contract.md) | Corrected exact-32K/32,768/40,960 pre-results candidate set |
| [Comparator parity pre-access contract](199_logprob_parity_contract.md) | Offline hash-bound correctness gate before R13 backend measurements |
| [Parser-independent held-out contract](200_parser_independent_heldout_contract.md) | Separate parser views, global identities, and additive leakage gates |
| [Tied embedding/head contract](201_tied_embedding_head_contract.md) | Inherited physical sharing is now explicit across accounting and export |
| [Pre-access training and Slurm hardening](202_preaccess_training_and_slurm_hardening.md) | Safe-point requeue, completion, finite-value, identity, and budget gates |
| [Integrated flagship evidence plan](203_flagship_integrated_evidence_plan.md) | E5 retirement, I1 data transfer, I2 assembled-recipe confirmation, and the paper claim spine |
| [Release and source-use policy](204_release_and_source_use_policy.md) | MIT code, Apache-2.0 weights, no corpus redistribution, guarded source approval, and remaining production gates |
| [Production MinHash batch successor](206_production_minhash_batch_successor.md) | Exact signature equivalence and faster resume for the 2B global-dedup calibration |
| [Logical SQLite resume equivalence](207_sqlite_logical_resume_equivalence.md) | Exact scientific-state equality despite noncanonical physical database pages |
| [2B production-data calibration](208_production_data_calibration_2b.md) | All gates pass; 150B fits conditionally while the current 500B path does not |
| [Production calibration fallback](209_production_calibration_fallback_decision.md) | R3 closes through 150B; 20B is not repeated and 500B remains blocked |
| [Production data firewall](210_production_data_firewall.md) | Equal-category partitions pass; audits stay sealed and model training stays blocked |
| [Formal tokenizer static nomination](211_formal_tokenizer_static_nomination.md) | 40,960 and exact-32K advance to the matched LM pilot |
| [Independent R3 backup/restore](212_independent_r3_backup_restore.md) | A 5.08 GB representative tree restores exactly from a second physical device |
| [Tokenizer-pilot corpus capacity](213_tokenizer_pilot_corpus_capacity.md) | Conservative firewall exclusion leaves 2.056B reference tokens |
| [Tokenizer-pilot fixed stream](214_tokenizer_pilot_fixed_stream.md) | One document stream is packed under all three finalist tokenizers |
| [Allocation-thesis program](205_allocation_thesis_program.md) | Four linked paper claims, fixed 1.2B target, five-seed I1, mature-horizon S2, and V4.1 systems taxonomy |

## Conventions

- Losses are natural-log cross entropy (“nats”).
- `K` means 1,024 tokens in context lengths; token budgets are written exactly.
- Resident state persists across decoding steps; peak allocation also includes temporary workspace.
- Internal directional diagnostics are not claims of exact retrieval or RULER performance.
- Findings 25–167 are historical unless the active flagship plan cites them explicitly.
