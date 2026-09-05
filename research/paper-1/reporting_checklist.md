# Paper 1 reporting checklist

The paper is not submission-ready until every applicable item is checked and linked to a repository
artifact. “Not applicable” requires a written reason.

## Claim integrity

- [ ] One central claim and each supporting claim have an explicit estimand.
- [ ] At least one Speck-specific contribution passes the novelty gate.
- [ ] Every numeric abstract/conclusion statement links to a checked result.
- [ ] Hypotheses, exploratory findings, confirmatory results, and post-hoc analyses are labeled.
- [ ] Negative, inconclusive, interrupted, and replaced runs are reported.
- [ ] Non-claims and scale/context/runtime boundaries appear in the paper.

## Architecture specification

- [ ] Complete canonical model configuration is published.
- [ ] Every operator has equations, tensor shapes, initialization, normalization, and pseudocode.
- [ ] Layer ordering, ratios, placement, sharing, routing, and positional treatment are exact.
- [ ] Total, active, embedding, expert, recurrent, residual-summary, and cache parameters are separated.
- [ ] Train/prefill/decode state transitions and cache eviction/resume semantics are specified.
- [ ] The difference from every directly inherited method is explicit.

## Experimental design

- [ ] Candidate and control use matched tokenizer, data, tokens, schedule, and evaluation.
- [ ] Fixed-token, fixed-FLOP, fixed-active, fixed-total, and time-to-quality views are reported.
- [ ] Seeds and packed-data orders meet the promotion policy.
- [ ] Non-inferiority margins, capability floors, alpha, confidence bounds, and multiplicity are frozen.
- [ ] Hyperparameter search spaces and selection budgets are equal and retained.
- [ ] Sample-size/power rationale is stated.
- [ ] Stopping, replacement, checkpoint-selection, and missing-data rules are followed.

## Component evidence

- [ ] Sequence components are isolated before their final combination.
- [ ] Standard, Full, and Block AttnRes plus a depth/width cross-sweep are complete.
- [ ] Dense, conventional MoE, latent projection, normalization, bounded activation, and balancing are
      separated.
- [ ] All promoted pairwise interactions are measured.
- [ ] The final `2^3` presence/absence cube is complete.
- [ ] Every retained component passes a final removal ablation.

## Scaling

- [ ] At least five compute-optimal points exist per architecture for a scaling claim.
- [ ] Fitting function, optimization, uncertainty, and residuals are published.
- [ ] A held-out scale or compute point tests the fit.
- [ ] Architecture-by-training-horizon interaction is measured.
- [ ] Proxy, medium, target-sentinel, and paper-scale roles are not conflated.

## Data

- [ ] Dataset names, revisions, licenses, mixtures, and token counts are published.
- [ ] Filtering, quality scoring, language detection, exact/near deduplication, and packing are specified.
- [ ] Train/validation/test and synthetic generator/template/entity/answer splits are isolated.
- [ ] Benchmark contamination checks and their limitations are reported.
- [ ] Tokenizer model, revision, fingerprint, fertility, and domain behavior are reported.
- [ ] Long-context data contains measured dependencies at the claimed lengths.

## Quality evaluation

- [ ] Aggregate and per-source language loss trajectories are published.
- [ ] General, knowledge, code, math, and reasoning suites are pinned and complete.
- [ ] Structured retrieval reports exact, token, candidate, target-direction, and specificity metrics.
- [ ] Symbolic route and payload edges qualify before direct composition is interpreted.
- [ ] RULER, NoLiMa, and HELMET run from pinned, qualified integrations.
- [ ] Long-context results are curves, not one endpoint.
- [ ] Raw per-example outputs and parser failures are retained.
- [ ] Post-training comparisons use the identical recipe if architecture transfer is claimed.

## Mechanistic analysis

- [ ] KDA decay, write strength, timescale, overwrite, and state-norm distributions are measured.
- [ ] Compression fidelity and selected-versus-dense attention support are measured.
- [ ] Selector recall is connected to retrieval/composition failures.
- [ ] AttnRes weights, entropy, source age, hidden RMS, update fraction, and gradients are measured.
- [ ] Expert load, routing margins, specialization, redundancy, overflow, and outliers are measured.
- [ ] Diagnostics are tested for predictive validity across seeds/scales, including counterexamples.

## Correctness and systems

- [ ] Torch/reference and optimized outputs, states, and all gradients pass declared tolerances.
- [ ] Full-sequence and incremental decode agree.
- [ ] Checkpoint, distributed resume, Transformers, and production-runtime exports pass parity.
- [ ] Training reports time/energy to fixed quality and achieved utilization.
- [ ] Serving reports p50/p90/p99 TTFT, TPOT/ITL, throughput, and maximum resident batch.
- [ ] Weight, KV, recurrent, residual, expert, workspace, fragmentation, and peak memory are separated.
- [ ] Consumer and datacenter hardware, software, precision, temperatures, and power are named.
- [ ] Compilation/startup is separated from steady state.
- [ ] Custom-kernel gains survive end-to-end scheduling and communication.

## Reproducibility and release

- [ ] Repository revision and clean/dirty state are recorded for every run.
- [ ] Config, data, tokenizer, parent, model, optimizer, metadata, and result hashes resolve.
- [ ] Environment lockfiles and container/driver/runtime versions are published.
- [ ] Commands and orchestration manifests reproduce every table and figure.
- [ ] Table/figure values are generated from machine-readable results.
- [ ] Code, kernels, model weights, data recipes, and evaluation outputs have compatible licenses.
- [ ] Independent rerun or artifact audit is complete.
- [ ] Known limitations, compute/energy use, and failure cases are prominent.

