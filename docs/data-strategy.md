# Data strategy for the 5,000 GPU-hour program

This document is the policy layer for the data program. Numeric ceilings and launch receipts remain in [`experiments/main-data/plan.json`](../experiments/main-data/plan.json), while executable study checks live beside each study packet. No document here authorizes a training run.

## One pipeline, three stage policies

Every record follows the same auditable path:

1. identify the source, license or use constraint, family, acquisition receipt and transformation history;
2. extract with a source-specific parser, retaining the original text or artifact identity;
3. remove exact duplicates, near duplicates, template copies and semantic repeats, while preserving a family graph for held-out splits;
4. apply quality, language, safety, benchmark-contamination and source-coverage filters;
5. attach lineage, quality dimensions, derivation cost and accepted-token accounting;
6. partition by source family before tokenization and packing;
7. verify manifests, token counts, packing, sampling weights, checksums and resumability;
8. train only from the hash-bound manifest and report exposure, replay, discarded data and evaluation results separately.

AI-assisted filtering may recommend keep/remove decisions, but pretraining retains the original source and records the model, prompt, version and decision. Generated text is a derived record and must never be indistinguishable from source text in a manifest.

### Pretraining and decay

The broad production pretraining mixture is natural and source-traceable. The first study has one broad baseline, one code contrast and one web-coverage contrast; all arms use the same tokenizer, objective, packing, exposure and held-out families. Synthetic or derived text is never silently mixed into a natural bank.

Late pretraining decay is a separate three-way comparison from one matched stable checkpoint: natural decay, curated natural capability decay, and curated decay with a small share of source-grounded or independently checked derived data. This is the only initial pretraining experiment that can admit derived text, and it must remain separately identified by lineage, teacher/generator, verification result and cost. The decision criterion is capability density per token and GPU-hour with general-retention and style guardrails—not training loss alone.

The primary work is boring infrastructure: source-specific extraction, exact SHA deduplication, MinHash/fuzzy and template deduplication, semantic-family checks, benchmark exclusion, quality filtering and contamination audits. Derived data is not used to compensate for missing natural supply. The current inventory remains a constraint: eligible-token and code-supply shortfalls must be reported before the grant run is scheduled.

### Mid-training

Mid-training is a continuation from one useful 4K parent with a qualified changed-data continuation contract. The 150-hour research envelope is split into a 70-hour capability comparison, a 50-hour context comparison and 30 hours of shared support. The capability comparison pairs replay with one selected candidate: raw, source-only targeted data is preferred; grounded augmentation is conditional on lineage and correctness gates; executable trajectories are a replacement branch, never an undeclared extra arm.

The context comparison is separate from capability-data claims. It tests unchanged-domain repacking against the long-context candidate mix at 16K, measuring fixed-suffix loss, related-prefix benefit, position and multi-file retrieval, short-task retention and source-family reuse. The 32K step remains a production gate inside the separate 300-hour context reservation.

### Post-training

SFT begins with structurally valid, outcome-verified candidate selection. Tool conversations must pass the repository's tool-aware adapter and schema checks; structural serializability alone is not correctness. The research comparison is candidate-pool control versus outcome-verified selection with paired seeds.

The release baseline remains the qualified always-thinking protocol. A direct-response mode may be evaluated as an SFT/control variant, but it cannot replace the release contract without explicit latency, quality, tool-use and long-task evidence.

RL is conditional. First qualify the trainer, rollout isolation, deterministic environments, verifiers, failure/recovery cases and held-out tests in a fixed-policy feasibility study. Only then can verified-reward RL enter production. Synthetic data is most acceptable here when it is generated inside pinned, testable environments or used for self-distillation; it remains separately attributed and must earn its place on held-out transfer, not training reward alone.

## Admission gates and grant order

Before compute is committed, each stage must have: a pinned manifest, family-disjoint evaluation split, contamination report, accepted-token or example accounting, reproducible checksums, measured throughput, retry/recovery policy and a declared stop rule. The order is:

1. close source and manifest readiness, including the current natural-code supply gap;
2. run the pretraining data screen and freeze one recipe;
3. qualify the changed-data continuation and run the bounded mid-training comparisons;
4. qualify SFT outcome checks and RL infrastructure;
5. launch production only from the selected, hash-bound recipes.

This staged data policy combines MAI's source-grounded, aggressively deduplicated pretraining discipline with MiniCPM and GLM-style late-stage enrichment. It is a transfer hypothesis, not evidence that derived data is universally harmful or universally beneficial.
