# Data strategy for the 5,000 GPU-hour program

This document is the policy layer for the data program. Numeric ceilings and launch receipts remain in [`experiments/main-data/plan.json`](../experiments/main-data/plan.json), while executable study checks live beside each study packet. No document here authorizes a training run.

## One pipeline, four stage policies

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

Mid-training is a continuation from one useful 4K parent with explicit data and context continuation contracts. The 360-hour research envelope screens four data arms, selective loss/packing, and the 4K-to-16K transition before one 1.2B confirmation. The 600-hour production reservation contains three stages: a 4K capability bridge, 16K repository reasoning, and 32K long-horizon agentic coding. The 4K stage uses repository structure, repair, tool schemas, short workflows and replay; the 16K stage adds multi-file repositories and issue/PR/commit context; the 32K stage adds repeated tool calls, test failures, recovery, summaries and state tracking.

Repository records retain file/dependency relationships, issues, reviews, pull requests, commits, diffs and tests. Grounded workflows retain parent context and teacher lineage. Executable trajectories require pinned repository snapshots, tool schemas, environment images, turn order, loss masks, verifier outcomes and recovery labels. Best-fit packing is required for complete repository, reasoning and trajectory records. Environment observations remain context; model actions and valid reasoning are supervised only after the mask adapter is qualified. A changed data manifest uses the data-continuation branch; a length change uses the context branch.

The research controls are replay, source-only repository data, grounded workflows and executable trajectories. A separate objective/packing comparison tests all-token next-token loss against output/action-only masking. Context comparisons measure fixed-suffix loss, dependency-distance retrieval, multi-file repair, tool-state continuation and short-task retention. 64K/128K remains future work until 32K benefit and cost are measured.

### Post-training

SFT begins with structurally valid, outcome-verified candidate selection. Tool conversations must pass the repository's tool-aware adapter and schema checks; structural serializability alone is not correctness. The research comparison is candidate-pool control versus outcome-verified selection with paired seeds.

The release baseline remains the qualified always-thinking protocol. An alternative direct-response
behavior may be evaluated as a research control, but it is not a release mode unless a later
decision records explicit latency, quality, tool-use and long-task evidence.

RL is conditional. First qualify the trainer, rollout isolation, deterministic environments, verifiers, failure/recovery cases and held-out tests in a fixed-policy feasibility study. Only then can verified-reward RL enter production. Synthetic data is most acceptable here when it is generated inside pinned, testable environments or used for self-distillation; it remains separately attributed and must earn its place on held-out transfer, not training reward alone.

Final self-SFT is a distinct stage after SFT and optional RL. Starting from the promoted RL checkpoint, or the SFT checkpoint if RL is not promoted, freeze a prompt pool and generate tool-aware responses in pinned environments. Verify, deduplicate and filter those responses, mix them with a verified anchor set, and train with the same assistant masks and tool protocol. Compare anchor-only continuation against self-distillation plus anchor replay. Preserve the pre-self-SFT parent and promote the final checkpoint only for held-out transfer without general, protocol, length or tool-call regressions.

## Admission gates and grant order

Before compute is committed, each stage must have: a pinned manifest, family-disjoint evaluation split, contamination report, accepted-token or example accounting, reproducible checksums, measured throughput, retry/recovery policy and a declared stop rule. The order is:

1. close source and manifest readiness, including the current natural-code supply gap;
2. run the pretraining data screen and freeze one recipe;
3. qualify the changed-data continuation and run the bounded mid-training comparisons;
4. qualify SFT outcome checks and RL infrastructure;
5. launch production only from the selected, hash-bound recipes.

This staged data policy combines MAI's source-grounded, aggressively deduplicated pretraining discipline with MiniCPM and GLM-style late-stage enrichment. It is a transfer hypothesis, not evidence that derived data is universally harmful or universally beneficial.
