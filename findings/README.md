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
