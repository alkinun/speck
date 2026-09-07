# Pre-grant readiness checklist

This list is the P0 gate for the 5,000-GH200-hour plan. “Before grant” means before starting the paid
three-month allocation, not necessarily before an award notice. Status reflects the repository and
research machine on 2026-09-07.

## Critical path

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R1 | Restore root storage headroom | Complete: root is 78% used with about 98 GiB free | Root stays below 80%; relocated checkpoint paths remain valid; all new work uses the data volume | SPE-24 |
| R2 | Add and qualify a code source | Bounded Stack v3.1, Stack-Edu, and Common Pile samples pass provenance, filtering, secrets, pinned benchmark contamination, partition yield, and cross-source overlap; training remains blocked on human rights/attribution acceptance and production cleanup/resume/global dedup | Restricted Stack v3.1, Stack-Edu, and blend treatments pass rights, provenance, English-text, repository-split, dedup, contamination, secret/PII, yield, and operational gates | SPE-114 |
| R3 | Run the 20B data rehearsal | Protocol complete; run not started | Download, filtering, exact/near dedup memory, packing throughput, storage, cleanup, and resume are measured | SPE-114 |
| R4 | Build the evaluation firewall | Protocol complete; data not built | Tokenizer sample, equal-byte selection set, unseen-source slices, and sealed audit are immutable and decontaminated before outputs | SPE-114 |
| R5 | Prepare flagship corpora | Experiment contract complete; data not started | Six-category stable candidates are launch-ready; decay candidates can finish before E4; unique-token risk is explicit | SPE-114 |
| R6 | Prepare long-document data | Not started | Complete books, papers, and repositories pass provenance, split, dedup, and length checks | SPE-52 |
| R7 | Qualify the four-GH200 stack | Not started | arm64, DDP, KDA, checkpoints, resume, storage, and achieved TFLOP/s pass a recorded rehearsal | SPE-115 |

The award can arrive while R2–R7 are running; the paid node must not start until their launch-critical
parts pass or have a written fallback that consumes no undeclared GPU work.

## Model and training readiness

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R8 | WSD schedule support | CPU implementation complete; GH200 qualification pending | Warmup/stable/cosine-decay behavior, resume identity, and D4 integration pass | SPE-115 |
| R9 | Optional FP8 | Not started; bf16 is the fallback | Numerical, quality, throughput, memory, and resume gates pass on GH200, or bf16 is frozen | SPE-83 |
| R10 | Decide tokenizer D5 | Deterministic trainer/evaluator and protocol complete; six-category inputs and runs pending; Mistral 32K fallback | Freeze source-balanced sample, three custom candidates, static report, matched 60M pilot, one-opening audit, selected artifact/hash, and repacking branch before E1/C0 | SPE-117 |
| R11 | Materialize scale targets | Shape-A planning target exists; shape B and ladder are missing | 60M–1.2B configs pass exact parameter/FLOP/state accounting | SPE-60 |
| R12 | Freeze analysis contracts | Data and architecture programs are specified; per-arm manifests are missing | E1W/E1S/E2–E5, C0/D2/D3/D4/D6/D7/D8, and scale analyses are immutable before outputs | SPE-70 |
| R17 | Implement selectable KDA output gating | KDA hardcodes sigmoid | Missing/explicit sigmoid are identical for old configs and checkpoints; SiLU and sigmoid pass Torch/FLA, geometry, export, and strict-load tests | SPE-128 |

[`targets/shape-a`](targets/shape-a/) is deliberately non-launchable. A real experiment appears only
after its data, training, hardware, and analysis contracts are complete.

[`DATA.md`](DATA.md) and [`data_plan.json`](data_plan.json) freeze the data categories, bounds,
experiment funnel, held-out firewall, statistic, guardrails, seed counts, and GPU-hour ceiling. They
do not substitute for source cards, prepared data, or per-arm manifests.

[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`architecture_plan.json`](architecture_plan.json) freeze
the shared-control identity, D2–D8 estimands, promotion rules, scale transfer, systems measurements,
and architecture GPU-hour ceiling. They do not authorize training before D8 support, scale targets,
and per-arm manifests pass.

## Evaluation and release readiness

| ID | Work | Current state | Done when | Linear |
| --- | --- | --- | --- | --- |
| R13 | Comparator baseline table | Not started | Named comparator revisions, token budgets, quality results, and 3090/CPU serving profiles are recorded | SPE-116 |
| R14 | RULER v2 and internal protocols | Sources/cases are qualified | Launch manifests pin the exact active tasks, lengths, prompts, and scoring | SPE-63 |
| R15 | Export rehearsal | KDA Transformers support exists; recurrent-mixer GGUF is missing | One representative KDA/NoPE checkpoint passes native/Transformers/GGUF identity and generation parity | SPE-76 |
| R16 | Release destinations and credentials | Not audited | Model, code, artifact, and paper destinations plus credentials and licenses are verified | SPE-91 |

HELMET, NoLiMa, MoE, depth routing, and sparse/compressed attention are not readiness work for this
grant. Their absence cannot block the flagship.

## Administrative readiness

- Confirm the allocation start/end dates, job wall-time limit, filesystem quotas, network policy, and
  checkpoint retention policy.
- Confirm whether the four GH200 GPUs are allocated as one exclusive node and whether concurrent
  single-GPU jobs are supported.
- Record support contacts and the escalation path for scheduler, storage, and node failures.
- Keep at least two independent copies of the launch manifest and every irreplaceable checkpoint.
- Decide who can authorize reserve use; no automatic process may spend it.

## P0 exit record

Before the first allocated job, create one dated record that contains:

- completion or fallback for R1–R17;
- exact Git revision and dirty-worktree check;
- hardware/software and dataset manifest hashes;
- measured throughput class from [`plan.json`](plan.json);
- selected contingency branch and remaining reserve;
- the first four launch commands and their expected outputs.
