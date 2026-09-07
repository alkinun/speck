# Findings relevant to the active flagship

The current model, paper, and compute plan live under
[`research/flagship/`](../research/flagship/). This page routes to the evidence that constrains those
defaults without presenting the retired 169-entry program as current work.

Checked JSON under `results/` remains the machine-readable source of truth. The
[`complete historical index`](ARCHIVE.md) preserves every finding, including failed and retired gates.

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

## Conventions

- Losses are natural-log cross entropy (“nats”).
- `K` means 1,024 tokens in context lengths; token budgets are written exactly.
- Resident state persists across decoding steps; peak allocation also includes temporary workspace.
- Internal directional diagnostics are not claims of exact retrieval or RULER performance.
- Findings 25–167 are historical unless the active flagship plan cites them explicitly.
