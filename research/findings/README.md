# Current conclusions

These are entry points into the current and archived evidence that constrains the flagship. Each
finding states its measurement boundary; links to prior evidence do not broaden historical claims.

| Topic | Prior evidence and boundary |
| --- | --- |
| Recurrent/global backbone | [Kimi transfer](../../archive/pregrant-history/findings/16_kimi_transfer_131m.md), [replication](../../archive/pregrant-history/findings/17_kimi_frontier_replication.md), and [paired dense/KDA proxy](../../archive/pregrant-history/findings/105_paper_1_paired_proxy_analysis.md) motivate the default; flagship-scale transfer remains unmeasured. |
| Context and resident state | [128K systems](../../archive/pregrant-history/findings/04_existing_checkpoint_128k_systems.md) and [32K continuation](../../archive/pregrant-history/findings/18_kimi_context32k.md) distinguish allocation from useful context. |
| Negative architecture result | [Reader Attention](../../archive/pregrant-history/findings/24_reader_attention.md) remains relevant evidence rather than an active run family. |
| Data preparation | [2B calibration](../../archive/pregrant-history/findings/208_production_data_calibration_2b.md), [150B fallback](../../archive/pregrant-history/findings/209_production_calibration_fallback_decision.md), and [firewall](../../archive/pregrant-history/findings/210_production_data_firewall.md) support preparation boundaries, not data-quality claims. |
| Tokenizer | [Static nomination](../../archive/pregrant-history/findings/211_formal_tokenizer_static_nomination.md) and the [Mistral screen](../../archive/pregrant-history/results/data/tokenizer-pilot-mistral-seed42-20260913.json) leave the custom-tokenizer comparison and D5 decision open. |
| Software correctness | [Checkpoint/data cleanup](../../archive/pregrant-history/findings/196_runtime_cleanup_successors.md) and [training/Slurm hardening](../../archive/pregrant-history/findings/202_preaccess_training_and_slurm_hardening.md) describe the original qualified revisions. Reorganized code is checked separately. |
| Current data readiness | [Critical path and streaming resume](2026-09-13-data-preparation-readiness.md) identifies the unresolved day-21 data schedule and qualifies bounded-memory resume-chain verification; full production throughput remains unmeasured on the successor. |
| Source-bank mechanics | [Six-category retained-source rehearsal](2026-09-13-source-bank-rehearsal.md) measures byte-quota selection and reference packing with identical interrupted/uninterrupted payloads; acquisition/global-dedup costs are inherited rather than remeasured. |
| Upstream unit mechanics | [Raw acquisition and cohort dedup](2026-09-13-acquisition-unit-rehearsal.md) executes pinned source/security filters, independent Parquet/gzip recovery, and global dedup over a bounded cohort; complete exclusion is checked separately below. |
| Complete exclusion integration | [Full reference pass and bank handoff](2026-09-13-firewall-integration.md) preserves all twelve reference views, removes superset matches, verifies full-reference checkpoint recovery, and prepares six excluded category banks; training-scale capacity and rates remain open. |
| Screen data capacity | [Conditional E1/E3 supply review](2026-09-13-screen-capacity.md) verifies the existing reference-token bank and quantifies category deficits; exact treatment recipes, final-tokenizer counts, and other-source capacity remain open. |
| Preparation cost | [Phase-separated checkpoint replay](2026-09-13-dedup-phase-timing.md) preserves the qualified outputs and attributes about 85% of measured continuation time to SQLite commit; the later WAL comparison below tests a bounded policy change. |
| Bounded WAL policy | [Durable policy comparison and hard-crash recovery](2026-09-13-sqlite-wal-policy.md) measures 43.9–53.2% lower complete continuation time with the 65,536-page trigger, full parity, and committed-WAL recovery; production-scale transfer remains open. |
| Explicit execution settings | [Config-bound SQLite qualification](2026-09-13-bound-sqlite-policy.md) carries the policy in config/checkpoint identities and actual runtime metadata; ordinary-CLI hard-crash recovery and reopen pass. |
| Tokenizer progress | [Terminal report recovery](2026-09-13-tokenizer-report-recovery.md) preserves the completed 40,960 screen and its missing-memory disclosure; the last compact screen is separately launched before budget/ranking review. |

Use the [complete ledger](../../archive/pregrant-history/findings/ARCHIVE.md) for the full history,
[current status](../status.json) for actions, and [paper claims](../../paper/claims.json) for publication
eligibility. Add a new finding when checked evidence supports a durable conclusion.
