# search v3

search v3 is a calibrated, budgeted architecture search system. it is being built alongside the completed version two system so historical studies keep their original meaning and remain reproducible.

version three does not promise a universally best architecture. it produces posterior hardware-specific frontiers and states exactly what evidence supports each recommendation.

## current status

the repository currently contains the version three foundations:

- independent architecture, configuration, study, scheduler, worker, artifact, and report versions
- immutable training protocols, objective sets, and seed bundles
- content-addressed artifacts and lineage manifests
- document-to-token indexes for new packed datasets
- deterministic document-aligned train, monitor, promotion, audit, and final segment plans
- resumable quality checkpoints with model, optimizer, data cursor, parent, and rng state
- a versioned hybrid block grammar with lossless version two conversion
- global and sliding grouped-query attention
- gated depthwise causal convolution
- optional and parallel swiglu stages
- immediate block repetition and weight sharing with occurrence-specific state
- exact shared-weight parameter and logical-state accounting
- a backend-neutral profiling contract and native torch backend
- raw nearest-rank p50 and p95 latency summaries
- rank, frontier, and bootstrap calibration reports
- joint posterior pareto estimates
- a grouped bootstrap ridge surrogate with cross-fitted predictions
- a normalized transactional version three study store
- leased worker actions and append-only decision events
- budgeted posterior action planning without product thresholds

the following work is intentionally not presented as complete:

- no production version three command line runner exists yet
- no calibration panel has been executed yet
- no search recommendation is calibrated to 100m or 1b tokens yet
- no actual q4 cpu backend has been selected or implemented
- no arm or mobile backend exists yet
- no full-bandwidth feedback operator is part of the search
- the checked-in experiment remains a version two model and search configuration

## trust contract

every production recommendation must identify:

- architecture schema and digest
- training protocol and token horizon
- dataset, tokenizer, and segment plan digests
- initialization, data-order, and numerical seed identities
- objective-set revision
- calibration artifact and out-of-sample report
- profiling backend, artifact, device, and scenario
- posterior frontier probability and uncertainty
- untouched audit status

a result must not claim calibration beyond the longest measured calibration anchors.

## scientific study boundaries

version three uses three separate studies.

### calibration

an uncensored, structurally diverse panel is trained through fixed token checkpoints. the panel measures noise sources, fidelity rank correlation, learning-curve prediction, and frontier recall.

calibration data are not selected through successive halving. structurally weak and unusual architectures remain in the panel because the calibrator needs negative and out-of-distribution examples.

### production search

the production search uses a frozen calibration artifact. it can dynamically spend budget on:

- a new architecture
- continuation to another token checkpoint
- another initialization or data-order seed
- a gpu profile
- a cpu profile
- a novelty or random sentinel

selection uses posterior pareto samples and budgeted information value. there are no fixed quality, latency, ram, cache, or model-size product cutoffs.

### audit

audit segments and metrics are unavailable to generation, surrogate fitting, and promotion. after the production frontier is frozen, the audit measures rank reversal, frontier recall, regret, and interval coverage.

using audit results to change the search creates a new study revision.

## data protocol

new packed datasets use format two and write `documents.jsonl`. every record includes:

- content hash
- train or validation split
- packed token start and end
- source
- score when available

format one datasets remain readable for historical runs, but they cannot provide document-aligned version three segment plans.

segment plans assign complete documents to disjoint partitions. training order is deterministic for each data seed. validation documents cannot appear in more than one of monitor, promotion, audit, or final evaluation.

## seed protocol

seed bundles separate:

- initialization randomness
- data-order randomness
- numerical repeat randomness

higher-fidelity continuation keeps the same initialization and data seed. a numerical repeat changes only the numerical seed identity.

## architecture grammar

an architecture contains explicit block groups. every group defines:

- residual width
- ordered stages
- logical repeat count
- weight-sharing policy

a stage contains one or more parallel branches. supported branch kinds are:

- `attention`
- `gated_causal_conv`
- `swiglu`

attention defines global or sliding scope, head dimension, kv heads, and an optional sliding window. convolution defines inner width and kernel size. swiglu defines intermediate width.

logical occurrences, unique parameter blocks, and state occurrences are separate identities. shared weights execute multiple times, while every logical occurrence keeps independent attention and convolution state.

## architecture accounting

parameter accounting counts:

- tied embeddings once
- occurrence-specific width adapters
- each unique shared or unshared block core once
- operation norms
- final norm and output adapter

state accounting counts every logical occurrence:

- global attention grows with context
- sliding attention is capped by its window
- convolution keeps a fixed kernel history

compute accounting must count every execution, including repeated shared blocks.

## profiling

profiling is backend and scenario specific. analytical estimates and measured metrics use different names and cannot substitute for each other.

the native torch backend uses resident model dtype. it allocates a fresh request state, performs prefill, and advances a real growing decode sequence.

an authoritative profile will require:

- fixed token fixtures
- fresh process repetitions
- compilation and allocation warmup
- at least 100 measured requests for p95 decisions
- raw latency samples
- separate prefill, first decode, steady decode, and whole-request timing
- process rss or vram measurements
- actual artifact and state bytes
- full backend and device provenance

the current native backend is a correctness and initial performance reference. q4 size remains an estimate until a backend packs and executes those weights.

## calibration reports

calibration reports include:

- spearman correlation
- kendall tau-b
- pairwise concordance
- top-k recall
- mean absolute error
- bootstrap confidence intervals
- predicted and observed pareto frontiers
- frontier recall and precision

surrogate predictions used in reports must be out of sample. grouped cross-fitting keeps observations from the same architecture out of both training and validation folds.

## posterior selection

candidate objective uncertainty is represented by a joint mean and covariance matrix. posterior pareto sampling preserves objective direction and covariance and reports:

- probability of being nondominated
- expected pareto rank

the planner receives actions with frontier probability, expected information, novelty, and estimated cost. each planning event uses a seeded random scalarization of those criteria, then selects actions that fit the available compute budget.

the planning seed, eligible proposal digests, sampled weights, priorities, selected actions, and decision digest are persisted for replay.

## storage and recovery

the version three study store is separate from the version two schema. it normalizes:

- objective sets
- architectures
- quality runs
- observations
- worker actions
- append-only events
- artifacts and lineage edges

worker actions use transactional claims, owner identities, random claim tokens, leases, and heartbeats. stale workers cannot complete actions after their lease has been released and reclaimed.

quality checkpoints are content addressed and include exact rng and data state. promoted work resumes a checkpoint instead of restarting from token zero.

## implementation map

| area | path |
| --- | --- |
| block grammar | `speck/architecture.py` |
| hybrid runtime | `speck/model_v3.py` |
| v3 search space | `speck/search/architecture_v3.py` |
| protocols | `speck/search/protocol.py` |
| document segments | `speck/search/segments.py` |
| quality checkpoints | `speck/search/checkpoints.py` |
| artifact store | `speck/search/artifacts.py` |
| calibration | `speck/search/calibration.py` |
| posterior pareto | `speck/search/posterior.py` |
| surrogate | `speck/search/surrogate.py` |
| budget planner | `speck/search/planner.py` |
| v3 study store | `speck/search/study_v3.py` |
| profiling | `speck/profile/` |

## next implementation sequence

the segment-plan preparation command is available:

```bash
python -m scripts.segment_plan ~/.cache/speck/ultra_fineweb/packed \
  ~/.cache/speck/search/segments-v3.json \
  --train-tokens 1000000000 \
  --monitor-tokens 5000000 \
  --promotion-tokens 20000000 \
  --audit-tokens 20000000 \
  --final-tokens 20000000
```

the packed dataset must contain the format-two document index.

`experiments/speck00-200m/search-v3.example.json` records the initial moderate-panel configuration. it is an example rather than an active experiment until its segment-plan digest is frozen and the worker workflow is complete.

remaining implementation sequence:

1. add a version three experiment and search configuration schema
2. add a resumable quality-run worker using the shared training loop
3. add isolated gpu and cpu profiling workers
4. add the event-driven coordinator and worker command line interface
5. add surrogate shadow-mode proposal generation
6. execute the noise-decomposition calibration study
7. execute the broad 100m panel and long-horizon anchors
8. freeze the first calibration artifact
9. start the first production version three search
