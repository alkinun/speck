# Flagship execution plan

This is the human operating view of [`plan.json`](plan.json). The JSON freezes budgets,
dependencies, exit gates, fallback rules, and scope. Calendar ranges are targets rather than reasons
to bypass a gate.

## Budget and calendar

The grant supplies 5,000 GPU-hours on one four-GH200 node over approximately 90 days. That is 52.1
fully occupied node-days. Mandatory work consumes 4,111 GPU-hours, or 42.8 node-days; the remaining
889 hours are protected reserve. Data plus architecture decisions and scale transfer consume 1,236
GPU-hours, or 12.9 fully occupied node-days, before the day-21 freeze.

| Phase | Target days | GPU-h | Output |
| --- | ---: | ---: | --- |
| P0 — Pre-grant readiness | before day 1 | 0 | Data, hardware, storage, targets, and contracts ready |
| P1 — Calibration and screens | 1–4 | 230 | Web and specialist sources, repetition policy, LR/batch region |
| P2 — Mixture and dense architecture | 4–11 | 566 | Mixture, position, ratio, recurrent operator, and output gate |
| P3 — Decay, curriculum, and scale | 11–20 | 440 | Decay recipe, scale fit, flagship size |
| P4 — Configuration freeze | 20–21 | 0 | One hash-bound launch manifest |
| P5 — Flagship pretraining | 21–50 | 2,425 | Pre-decay and final base checkpoints |
| P6 — Extension through release candidates | 51–65 | 450 | 128K, annealed, instruct, evaluated exports |
| P7 — Protected reserve | 1–80 | 889 | Triggered recovery or continuation only |
| P8 — Paper and release buffer | 66–90 | 0 planned | Audited paper and public artifacts |

Independent single-GPU arms should be packed four at a time. The flagship alone owns all four GPUs.

## Phase order

### P0 — Before the allocation

Complete [`PREGRANT.md`](PREGRANT.md). Do not start a paid node while corpus, storage, launch targets,
selectable KDA gating, or resume behavior are unresolved.

Future flagship train configs must set `requires_data_launch_authority=true`. The fixture-qualified
[`data_launch_plan.json`](data_launch_plan.json) requires an immutable receipt binding Git, experiment
files, human rights, production operations, deny ledger, firewall, selected tokenizer, and packed
data. The trainer revalidates those artifacts after shard verification and before model construction.
No real receipt exists while the P0 data gates remain open.

### P1 — First answers first

Run these independent workstreams in parallel:

- E1W web source/filter screen: 133 GPU-hours.
- E1S code, math, and synthetic source screens: 27 GPU-hours.
- E3 repetition policy: 32 GPU-hours.
- D4 LR/global batch: 33 GPU-hours.
- D6 LR/width anchors: 5 GPU-hours.

D4 and D6 constrain every later training recipe. E3 determines whether the data pipeline needs near
500B unique tokens or can safely repeat roughly 150B. E1W and E1S supply qualified treatments for
E2. Source identity, rights, language, deduplication, contamination, yield, and operational gates are
CPU-only P0 work defined in [`DATA.md`](DATA.md).

### P2 — Select the stable model and data recipe

- E2 stable-mixture funnel: 251 GPU-hours—24 tiny space-filling arms, six medium-scale candidates,
  then three finalists at 350M/12B and three seeds.
- C0 three-seed shared dense control: 63 GPU-hours.
- D2 NoPE versus partial RoPE: 63 GPU-hours.
- D3 3:1 versus 5:1 KDA/global ratio: 63 GPU-hours.
- D7 KDA versus matched scalar-decay GDN under sigmoid/NoPE: 63 GPU-hours.
- D8 KDA sigmoid versus SiLU output gating: 63 GPU-hours.

C0 launches only after E2c freezes the stable mixture, then is trained once per seed and reused by
D2, D3, D7, and D8. Architecture decisions use paired one-sided 95% bounds, the 0.01-nat aggregate
margin, the 0.02-nat source guardrail, and mandatory 32K/128K and original-4K gates. Shared-control
reuse requires exact identity of parent, data, training, seed, and analysis. Non-inferiority only
establishes eligibility; each arm must produce its declared quality, state, or systems benefit. Data
decisions use equal-domain bits per UTF-8 byte, paired bounds, per-category guardrails, and a sealed
audit. The fitted E2 response surface nominates candidates; only retrained, replicated arms decide.
[`ARCHITECTURE.md`](ARCHITECTURE.md) freezes the complete contract.

### P3 — Transfer before committing the flagship

- E4 decay mixture: 50 GPU-hours, branched from the E2 stable winner.
- E5 curriculum shape: 100 GPU-hours; reuse the selected E2 stable mixture as the uniform control.
- Dense scale ladder and 750M reversal check: 290 GPU-hours.

Resolve the 1.2B/400B default versus the 600M/800B alternative by day 18. The fitted scale curve must
exclude the eventual flagship point, which remains a held-out check.

### P4 — Day-21 freeze

Freeze model, tokenizer, stable/decay data, optimizer, LR, global batch, WSD tail, precision,
checkpoints, context stages, evaluation, and release paths. An unresolved decision takes its declared
default. No new axis may enter after day 1, and no new experiment may delay the freeze.

### P5 — Flagship

The selected shape receives 2,425 GPU-hours. The default is 1.2B parameters over 400B tokens:

- approximately 320B stable-phase tokens;
- publish the pre-decay checkpoint;
- approximately 80B decay-phase tokens;
- publish the final base checkpoint.

The pre-decay checkpoint is the primary recovery and continuation asset. Optional throughput gains do
not buy extra tokens until it is secure.

### P6 — Context, post-training, evaluation, and release

Working envelope, adjustable without changing the 450-hour phase ceiling:

| Work | Initial GPU-h envelope |
| --- | ---: |
| 32K and 128K context extension | 200 |
| Three annealing branches and merge | 80 |
| Supervised fine-tuning | 50 |
| Quality and long-context evaluation | 60 |
| Serving and export qualification | 60 |

Each context stage must retain original-4K quality before the next length begins. Preference tuning is
optional and cannot displace evaluation, serving, or release parity.

### P7/P8 — Recover, then publish

Reserve pays for invalid mandatory work first, then flagship throughput shortfall, then extension or
post-training repair. Only after every mandatory artifact is secure may it replicate an existing
decision or continue flagship training. It never funds a new architecture axis.

The final calendar window is for paper generation, audits, model cards, artifact upload, and release
verification. Ideally it consumes no training hours.

## Flexibility without drift

Flexible:

- Exact dates inside a phase.
- Packing independent jobs across GPUs.
- Post-training sub-budget allocation.
- Using a documented default when an experiment misses its deadline.
- Applying the frozen throughput contingency and cut order.

Not flexible:

- The 5,000-hour ceiling and 889-hour reserve.
- Dense-width flagship scope.
- Dependency and exit-gate order.
- Neutral held-out evaluation and seed confirmations.
- Day-21 freeze.
- Original-4K regression, checkpoint/resume, and release-parity requirements.

When throughput misses plan, cut E5 first, then scale contingency runs, then nonmandatory scale points,
then reduce the flagship from 400B to 320B tokens. Never save compute by dropping evaluation or
replication while retaining the associated claim.
