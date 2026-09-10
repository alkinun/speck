# Pre-grant readiness checklist

This list is the P0 gate for the 5,000-GH200-hour plan. “Before grant” means before starting the paid
three-month allocation, not necessarily before an award notice. Status reflects the repository and
research machine on 2026-09-10.

## Critical path

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R1 | Restore root storage headroom | Complete: root is 72% used with about 125 GiB free; `/mnt/speck-data` has about 5.1 TiB available | Root stays below 80%; relocated checkpoint paths remain valid; all new work uses the data volume | SPE-24 |
| R2 | Add and qualify a code source | Complete for source use: all five bounded inputs pass technical gates and project-owner guarded-use approval records residual metadata risks, attribution, no corpus redistribution, removal, and rebuild policy; production execution remains R3 | Restricted Stack v3.1, Stack-Edu, and blend treatments pass rights, provenance, English-text, repository-split, dedup, contamination, secret/PII, yield, and operational gates | SPE-114 |
| R3 | Calibrate production data operations | Full 20B attempt safely paused; 2B acquisition, global exact/near dedup, and 2.000008B packing pass; logical SQLite resume equivalence is qualified and final resume/firewall/projection publication remains | The resumed 2B path completes cleanup, interruption equivalence, and disjointness; measured 20B and 150B/500B projections receive an explicit fallback decision or the full rehearsal resumes | SPE-174, SPE-173 |
| R4 | Build the evaluation firewall | Benchmark firewall complete for all six bounded categories; three-partition construction/consumer tooling fixture-qualified; real targets, authority, and partitions pending | Tokenizer sample, equal-byte selection set, unseen-source slices, and sealed audit are immutable and decontaminated before outputs | SPE-114 |
| R5 | Prepare flagship corpora | All six bounded categories technically qualified and human source use approved; production/firewall/launch tooling fixture-qualified; real 20B rehearsal and corpus preparation pending | Six-category stable candidates are launch-ready; decay candidates can finish before E4; unique-token risk is explicit | SPE-114 |
| R6 | Prepare long-document data | Not started | Complete books, papers, and repositories pass provenance, split, dedup, and length checks | SPE-52 |
| R7 | Qualify the four-GH200 stack | Local checkpoint replacement and CPU launch-risk failure injection are qualified: tracked-tree binding, distributed tuple/manifest checks, optimizer-boundary requeue, non-final artifact separation, finite-value publication gates, and resume timing are implemented; GH200/DDP/Slurm qualification not started | arm64, DDP, KDA, checkpoints, resume, storage, and achieved TFLOP/s pass a recorded rehearsal | SPE-115, SPE-165 |

The award can arrive while R2–R7 are running; the paid node must not start until their launch-critical
parts pass or have a written fallback that consumes no undeclared GPU work.

## Model and training readiness

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R8 | WSD schedule support | CPU implementation complete; GH200 qualification pending | Warmup/stable/cosine-decay behavior, resume identity, and D4 integration pass | SPE-115 |
| R9 | Optional FP8 | Not started; bf16 is the fallback | Numerical, quality, throughput, memory, and resume gates pass on GH200, or bf16 is frozen | SPE-83 |
| R10 | Decide tokenizer D5 | Active v5 binds human-approved sources and preserves exact-32K/32,768/40,960 candidates plus v2 Pareto policy; real operations/firewall, formal inputs/runs, and D5 audit remain pending | Materialize the authorized source-balanced sample, train corrected candidates, run static report and matched 60M pilot, open the audit once, then freeze artifact/hash and repacking branch before E1/C0 | SPE-117 |
| R11 | Materialize scale targets | Complete as non-launchable v2 successor geometry: seven targets pass independent and instantiated parameter/FLOP/state/optimizer accounting; D5 remains open and the inherited physically tied head is frozen by contract | 60M–1.2B configs pass exact parameter/FLOP/state accounting | SPE-60, SPE-166 |
| R12 | Freeze analysis contracts | Four paper claims and active data/architecture/integration/execution successors are specified; per-arm materializers, analyzers, figure schemas, and manifests are missing | E1W/E1S/E2–E4, C0/D2/D3/D4/D6/D7/D8, five-seed I1, I2/I3, scale, S2 mature horizon, and paper-output analyses are immutable and executable before outputs | SPE-70 |
| R17 | Implement selectable KDA output gating | Default-compatible implementation, CPU and export gates complete; full-layer FLA integration cases await R7 CUDA execution | Missing/explicit sigmoid are identical for old configs and checkpoints; SiLU and sigmoid pass Torch/FLA, geometry, export, and strict-load tests | SPE-128 |

[`targets/shape-a`](targets/shape-a/) is deliberately non-launchable. A real experiment appears only
after its data, training, hardware, and analysis contracts are complete.

[`DATA.md`](DATA.md) and [`data_plan_v2.json`](data_plan_v2.json) freeze the data categories, bounds,
experiment funnel, held-out firewall, statistic, guardrails, seed counts, and GPU-hour ceiling. They
do not substitute for source cards, prepared data, or per-arm manifests.

[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`architecture_plan_v2.json`](architecture_plan_v2.json) freeze
the shared-control identity, D2–D8 estimands, promotion rules, scale transfer, systems measurements,
and architecture GPU-hour ceiling. They do not authorize training before D8 support, scale targets,
and per-arm manifests pass.

[`integration_plan_v2.json`](integration_plan_v2.json) separately freezes five-seed I1's
data-by-architecture transfer, I2's assembled-recipe confirmation and complete-C0 fallback, and I3's
analysis inside 122 GPU-hours.
[`PAPER.md`](PAPER.md) maps those results and the existing programs to one claim ladder. Neither may
introduce a new mechanism or reopen a completed selection after outputs.

## Evaluation and release readiness

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R13 | Comparator baseline table | Pre-access offline log-probability parity contract is fixture-qualified; backend access, revisions, evaluations, and baseline table are pending | Named comparator revisions, token budgets, quality results, and 3090/CPU serving profiles are recorded | SPE-116 |
| R14 | RULER v2 and internal protocols | Sources/cases are qualified | Launch manifests pin the exact active tasks, lengths, prompts, and scoring | SPE-63 |
| R15 | Export rehearsal | KDA Transformers support exists; recurrent-mixer GGUF is missing | One representative KDA/NoPE checkpoint passes native/Transformers/GGUF identity and generation parity | SPE-76 |
| R16 | Release destinations, backup, and credentials | Compendium metadata and retention policy exist; independent backup, restore rehearsal, DOI archive, destinations, and credentials are not complete | Model, code, artifact, and paper destinations plus credentials/licenses pass; a representative R3 tree restores from an independent verified copy | SPE-91, SPE-172 |
| R18 | Research compendium and claim governance | Complete locally: catalog, notebook, workflow, data-management plan, paper workspace, claim registry, validator, and CI workflow exist | Catalog/claims resolve from a clean checkout; ongoing work follows the notebook→result→finding→claim boundary | SPE-70 |

HELMET, NoLiMa, MoE, depth routing, and sparse/compressed attention are not readiness work for this
grant. Their absence cannot block the flagship.

## Administrative readiness

- Confirm the allocation start/end dates, job wall-time limit, filesystem quotas, network policy, and
  checkpoint retention policy.
- Confirm whether the four GH200 GPUs are allocated as one exclusive node and whether concurrent
  single-GPU jobs are supported.
- Record support contacts and the escalation path for scheduler, storage, and node failures.
- Keep at least two independent copies of the launch manifest and every irreplaceable checkpoint.
- Complete SPE-172's hash-verified backup/restore rehearsal before the first irreplaceable flagship checkpoint.
- Decide who can authorize reserve use; no automatic process may spend it.

## P0 exit record

Before the first allocated job, create one dated record that contains:

- completion or fallback for R1–R18;
- exact Git revision and dirty-worktree check;
- exact tracked Git tree identity (untracked harness state is outside the contract unless declared);
- hardware/software and dataset manifest hashes;
- measured throughput class from [`plan_v2.json`](plan_v2.json);
- selected contingency branch and remaining reserve;
- the first four launch commands and their expected outputs.
