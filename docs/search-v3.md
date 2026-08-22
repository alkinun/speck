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
- a read-only dashboard with objective-set, token-horizon, run, action, checkpoint, profile, and posterior views
- a portable calibration launcher with safe interruption, resumption, worker draining, and dashboard lifecycle management

the following work is intentionally not presented as complete:

- no integrated production version three command line runner exists yet
- no complete broad and long-horizon calibration panel has been executed yet
- no search recommendation is calibrated to 100m or 1b tokens yet
- no actual q4 cpu backend has been selected or implemented
- no arm or mobile backend exists yet
- no full-bandwidth feedback operator is part of the search
- the checked-in baseline model remains version two and is converted losslessly during v3 initialization

## changes from version two

version three is a separate study protocol rather than an in-place database upgrade. version two studies remain readable and reproducible under their original rules.

### architecture representation

version two represents a model as a linear list of layers. every layer has a hidden width and swiglu width, may contain global grouped-query attention, and uses one global attention head dimension.

version three represents a model as block groups:

- a group controls logical repetition and optional immediate weight sharing
- a block controls residual width and ordered stages
- a stage may contain parallel branches
- branches may be global or sliding grouped-query attention, gated causal convolution, or swiglu
- attention head dimensions, kv-head counts, and windows may vary by block
- swiglu is optional, so mixer-only blocks are representable

logical depth, unique parameter blocks, and sequence-state occurrences are separate. sharing a repeated block reduces parameter bytes but does not merge its occurrence-specific attention or convolution state. version two models convert losslessly to this grammar; architectures using convolution, sliding attention, parallel branches, heterogeneous head dimensions, missing swiglu, or shared repetition generally cannot convert back to version two.

the v3 runtime supports parallel branches, while the current bootstrap generator intentionally samples a narrower singleton-stage form. grammar support and currently generated search proposals are therefore not identical.

### search policy and fidelity

version two is an integrated evolutionary successive-halving search. it generates mutations and crossovers, trains independent trials at fixed rungs, promotes selected architectures, and emits quality, efficiency, balanced, and frontier recommendations. promotion starts a new trial from token zero with the next rung's training geometry.

the current version three coordinator is a calibration bootstrap:

- it creates an uncensored deterministic broad panel rather than an evolutionary population
- it crosses initialization, data-order, and numerical seeds for explicit noise decomposition
- it trains through one immutable sequence and batch geometry
- it resumes exact checkpoints across token horizons instead of restarting
- it evaluates every checkpoint before allowing continuation
- it fits grouped, cross-validated surrogate and calibration reports only after the configured evidence is complete
- it uses joint posterior pareto samples and cost-aware random scalarization to choose long-horizon anchors

mutation and crossover operators exist for the v3 grammar, but a production proposal loop using a frozen calibration artifact remains future work. consequently, an `anchor_complete` calibration study is not the same thing as a final production-search recommendation.

### objective semantics

version two has one fixed objective tuple and assumes every objective is minimized. quality, latency, cache, memory, and estimated q4 size are aggregated together at a selected rung.

version three has named objective sets. every objective declares:

- minimize or maximize direction
- quality, efficiency, safety, or reporting role
- whether it is required for selection

the dashboard and posterior code preserve these directions. reporting-only values such as `quality.procedural_score` cannot make an architecture incomplete or alter its pareto rank. quality comparisons use one exact training-token horizon; profile repetitions are aggregated only within the selected objective set and scenario.

### quality and data isolation

version two reads token slices directly and computes validation inside the trial worker. its validation limit can omit a final partial batch.

version three requires a verified document index and a frozen document-aligned segment plan with disjoint `train`, `monitor`, `promotion`, `audit`, and `final` partitions. training order is deterministic per data seed. `quality.target_nll` is produced by a separate worker that evaluates every next-token target in the selected partition, including the final partial batch. checkpoint training loss is reporting metadata and is never substituted for target nll.

### seeds and resumability

version two derives one trial seed that combines initialization and execution randomness. it stores trial results but no resumable optimizer/data checkpoint.

version three records independent initialization, data-order, and numerical seed identities. checkpoints contain model, optimizer, data cursor, Python, NumPy, Torch, and CUDA rng state plus parent and protocol identities. continuation must advance to the immediate next configured checkpoint and atomically fence the expected parent.

### profiling and accounting

version two profiles inside each trial and primarily exposes p50 GPU timing, analytical kv-cache bytes, peak memory, and estimated q4 bytes.

version three profiles independently by backend, device, dtype, request geometry, and isolated process repetition. the native backend records raw samples and nearest-rank p50/p95 summaries for prefill, first decode, steady decode, and whole requests, plus resident weight bytes, allocated state bytes, and peak rss or vram. CPU and GPU profiles are separate evidence. q4 remains an analytical estimate until an executable packed backend exists.

version three accounting counts unique shared parameters separately from logical execution and occurrence state. sliding attention state is window bounded and convolution history is kernel bounded.

### workers, storage, and recovery

version two's coordinator launches local subprocesses, tracks pid/process identity, ingests one result document, and retries according to fixed timeout settings.

version three workers claim normalized actions transactionally using owner identities, random claim tokens, expiring leases, and heartbeats. stale workers cannot publish after reclamation. training, evaluation, and profiling use separate action kinds and atomic result commits.

version three stores immutable objects by content digest and records artifact lineage, normalized observations, checkpoint ancestry, planning decisions, posterior reports, and append-only events. evaluated checkpoint payloads may be pruned while their hashes, metadata, observations, and lineage remain. archived runs are retained evidence, not failures.

### operation and dashboard differences

version two has one integrated `run` command and stores studies below `~/.cache/speck/search/`. version three stores studies below `~/.cache/speck/search-v3/`. `scripts/run_search_v3.sh` orchestrates the current calibration workflow; the underlying coordinator, `worker`, `evaluation-worker`, and `profile-worker` commands remain independently available.

the shared dashboard detects all three study formats. for v3 it adds:

- objective-set and exact token-horizon selectors
- direction-aware observed pareto ranks
- checkpoint-horizon coverage
- raw run and action status, including archived runs
- native block-group geometry
- seed bundles and resumable checkpoint lineage
- whole-monitor quality curves
- isolated profile repetitions
- posterior anchor, probability, expected-rank, and calibration metadata when available

the dashboard is read only. it never claims actions, changes study status, or reads pruned checkpoint payloads.

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
| v3 configuration | `speck/search/spec_v3.py` |
| study initialization | `speck/search/initialize_v3.py` |
| quality worker | `speck/search/quality_worker.py` |
| evaluation worker | `speck/search/evaluation_worker.py` |
| profile worker | `speck/search/profile_worker.py` |
| bootstrap coordinator | `speck/search/coordinator_v3.py` |
| posterior shadow report | `speck/search/posterior_v3.py` |
| profiling | `speck/profile/` |
| v1/v2/v3 dashboard | `scripts/search_dashboard.py` and `scripts/search_dashboard.html` |

## next implementation sequence

the segment-plan preparation command is available:

```bash
python -m scripts.segment_plan ~/.cache/speck/ultra_fineweb/packed \
  ~/.cache/speck/search/segments-v3.json \
  --train-tokens 1010000000 \
  --monitor-tokens 1000000 \
  --promotion-tokens 5000000 \
  --audit-tokens 5000000 \
  --final-tokens 5000000
```

the packed dataset must contain the format-two document index.

an existing format-one packed corpus can be upgraded without rewriting token shards:

```bash
python -m scripts.upgrade_document_index ~/.cache/speck/ultra_fineweb/packed
```

the upgrader verifies every shard while recovering reserved bos/eos boundaries, preserves `manifest.v1.json`, derives document identities from the verified original dataset plus split token ranges, and atomically publishes `documents.jsonl` before the format-two manifest.

`experiments/speck00-200m/search-v3.json` records the active moderate-panel configuration and pins the checked segment-plan digest.

run or resume the complete calibration workflow from any shell, including fish:

```bash
./scripts/run_search_v3.sh --study calibration-v3
```

the launcher initializes the study idempotently, prepares the cuda environment, starts the read-only dashboard at `http://127.0.0.1:8000`, coordinates bounded action batches, drains quality and evaluation work, runs every gpu and cpu profile repetition in a fresh process, and stops at `anchor_complete`. `Ctrl-C` stops the dashboard child and exits; committed checkpoints and observations remain resumable by running the same command again.

the checked-in planner budget is 1,800,000 wall-seconds. this covers the approximately 1,368,000-second minimum implied by the measured default rates, required cpu/gpu profile repetitions, broad panel, and ten long-horizon anchors, while retaining headroom for variance and retries. the launcher exits with a diagnostic instead of spinning if the coordinator has no active or schedulable work before `anchor_complete`.

use `--no-dashboard`, `--host`, `--port`, `--experiment`, or `--config` when needed. calibrated scheduling defaults can be overridden without editing the script:

```bash
SPECK_QUALITY_TOKENS_PER_COST=10000 \
SPECK_EVALUATION_TOKENS_PER_COST=30000 \
SPECK_PROFILE_COST=600 \
./scripts/run_search_v3.sh --study calibration-v3
```

`./scripts/run_search_v3.sh --help` lists every launcher option. the lower-level commands below are retained for debugging and manual operation.

after an active configuration records the emitted segment-plan digest, initialize and inspect a study with:

```bash
python -m scripts.architecture_search_v3 init experiments/speck00-200m \
  --study calibration-v3 \
  --config experiments/speck00-200m/search-v3.json
python -m scripts.architecture_search_v3 status calibration-v3
```

initialization verifies all packed shards, the tokenizer identity, every selected document span, partition coverage, the full quality horizon, baseline context limits, and exact parameter accounting before atomically registering the study bundle.

launch the read-only dashboard by study name:

```bash
PYTHONPATH=. uv run --extra gpu python -m scripts.search_dashboard calibration-v3 \
  --host 127.0.0.1 --port 8000
```

then open `http://127.0.0.1:8000`. a v3 study name resolves below `~/.cache/speck/search-v3/`; v1 and v2 names continue to resolve below `~/.cache/speck/search/`. if the same name exists in both roots, pass the explicit `study.sqlite3` path to avoid ambiguity.

schedule and execute one checkpoint continuation with:

```bash
python -m scripts.architecture_search_v3 schedule-quality calibration-v3 \
  --seed-index 0 \
  --estimated-cost 3600
python -m scripts.architecture_search_v3 worker calibration-v3 \
  --device cuda \
  --once
```

each scheduling call targets only the immediate next protocol checkpoint. repeating the call with the same seed index resumes the same run from its latest immutable checkpoint. the segment-plan seed fixes partition membership; the run data seed fixes a deterministic document order within that membership.

the worker heartbeats through model construction, training, and checkpoint serialization. it writes the content-addressed checkpoint first, then atomically fences the expected parent, registers lineage, advances the run, and completes the action. expired or reclaimed workers cannot publish progress. the initial worker is single-device, requires `compile_model` to be false, and records training state only; quality evaluation remains a separate action so training loss cannot be mislabeled as target nll.

evaluate a completed checkpoint with:

```bash
python -m scripts.architecture_search_v3 schedule-evaluation calibration-v3 \
  --run 1 \
  --estimated-cost 300
python -m scripts.architecture_search_v3 evaluation-worker calibration-v3 \
  --device cuda
```

the evaluation protocol covers every next-token target in the frozen `monitor` partition exactly once, including a final partial batch. its partition and batch geometry are part of the immutable training protocol. `quality.procedural_score` is reporting-only and excluded from selection until a pinned benchmark and scoring artifact exist.

schedule the configured independent profile repetitions with:

```bash
python -m scripts.architecture_search_v3 schedule-profile calibration-v3 \
  --profile gpu_short \
  --estimated-cost 60
python -m scripts.architecture_search_v3 profile-worker calibration-v3 \
  --backend torch_native \
  --device cuda
```

`profile-worker` claims exactly one capability-matched action and then exits, so each configured process repetition starts in a fresh process. profile artifacts preserve every raw request and decode sample, the exact backend identity, peak rss or vram, resident weight bytes, and allocated sequence-state bytes. result registration, normalized observations, and action completion share one lease-fenced transaction.

the bootstrap coordinator creates the deterministic broad panel, crosses the configured initialization, data-order, and numerical seeds for the noise subset, and keeps at most `max_actions_per_event` work items active:

```bash
python -m scripts.architecture_search_v3 coordinate calibration-v3 \
  --quality-tokens-per-cost 10000 \
  --evaluation-tokens-per-cost 30000 \
  --profile-cost 600
```

rates and costs use the configuration's declared `cost_unit`; with `wall_seconds`, the two rates are measured tokens per second and profile cost is seconds per isolated repetition. continuation cost is derived from its exact token delta, while evaluation cost covers the complete monitor partition. repeated coordinator ticks are idempotent: architecture, run, profile-repetition, and checkpoint-evaluation identities prevent duplicate work after interruption. every checkpoint is evaluated before that run can continue.

after the noise and broad trajectories plus all configured profiles complete, the coordinator writes an immutable shadow report. it contains the raw observation identities, initialization/data-order/numerical variance decomposition, stable architecture features, per-objective normalization, grouped bootstrap surrogate states, cross-fitted rank and frontier calibration, joint posterior pareto probabilities, and the exact budgeted random-scalarization decision. measured seed variance is added to posterior quality covariance. objective normalization prevents units such as bytes from dominating information value. no quality, latency, memory, cache, or size threshold is introduced.

the report's selected canonical runs are the only architectures advanced from the broad horizon to `anchor_tokens`. each intermediate anchor checkpoint still requires whole-monitor evaluation before continuation. the coordinator reports `anchor_complete` only after all selected long-horizon checkpoints are evaluated.

checkpoint retention is evidence gated. a payload is never pruned before its whole-monitor evaluation commits. superseded parent payloads are removed only after child checkpoint metadata exists; noncanonical noise payloads are archived after their configured horizon; unselected broad payloads remain available until the posterior report is immutable; selected anchors retain only their latest resumable payload until final evaluation. hashes, lineage, run metadata, evaluation artifacts, and explicit pruning events remain in the study after payload removal.

remaining implementation sequence:

1. execute the noise-decomposition calibration study
2. execute the broad 100m panel and long-horizon anchors
3. review and freeze the first calibration artifact
4. add production mutation and crossover proposals
5. start the first production version three search
