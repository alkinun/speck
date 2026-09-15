# Long-context allocation execution plan

**Selected successor:** [plan_v4.json](plan_v4.json), replacing plan_v3 under the owner's explicit
2026-09-15 pivot. Phase caps are binding. Per-run endpoints require measured cost and immutable inputs.

## Budget

| Work | GPU-hours | Share |
| --- | ---: | ---: |
| R0-R4: qualification and focused research | 750 | 15.00% |
| B0: broad base pretraining | 2,425 | 48.50% |
| L0: context and instruction capability development | 700 | 14.00% |
| V1-V3: final evaluation, serving and release verification | 236 | 4.72% |
| Protected reserve | 889 | 17.78% |
| **Total** | **5,000** | **100%** |

Mandatory work remains 4,111 hours. One GPU-hour is one allocated GPU for one hour; four GPUs cost
four hours per node-hour. The allocation buys 52.08 fully occupied node-days, not 90 continuous
four-GPU days. Failed runs, startup, checkpoints, evaluation and debugging are charged once to an
identified phase. External teachers/judges, CPU preparation, storage and founder time have separate
ledgers; they are not described as zero-cost or hidden inside analytic FLOPs.

## Research: 750 hours

| ID | GPU-h | Output |
| --- | ---: | --- |
| R0 | 70 | Exact 1.2B 4K/32K/128K fit, backward, recovery, four-GPU and end-to-end throughput qualification |
| R1 | 80 | Broad source capacity/repetition, bounded optimization calibration, task difficulty and fully costed study freeze |
| R2 | 360 | Six paired architecture parents and twelve supervision branches, three complete paired seeds |
| R3 | 180 | One larger-scale or longer-horizon transfer check, selected before R2 outputs |
| R4 | 60 | Core/transfer evaluation, paired analysis and standard-versus-targeted recipe decision |

The [study](STUDY.md) defines controls and inference. R1 replaces the old source/mixture search with a
bounded recipe qualification: at most two source-exposure schedules on the same broad mixture over two
paired seeds, plus bounded LR checks for dense/hybrid. Its exact runs, cost and horizon must fit 80
hours. Failed capacity does not select a scientifically superior source. Materialization failure
requires a pre-results supply successor and re-cost; no old E1/E2 search resumes automatically.

## Dependency order and calendar

P0 pre-access preparation -> R0 hardware -> R1 pilot/freeze -> R2 core -> R3 transfer -> R4 decision
-> F0 verified production data/launch freeze -> B0 base -> L0 capability -> V1/V2 -> V3 -> publication.
R4 evaluation may consume its allowance as R2/R3 checkpoints arrive; final decision waits for both.
Independent CPU preparation and disjoint pilot work proceed early. A full-node base run owns all four
GPUs; overlapping target dates do not authorize double-booking.

Aim for research/launch freeze around days 21-25, base completion around day 55, capability by day 76,
verification by day 85 and publication assets by day 90. Reforecast at every gate. The historical
150B preparation path projected 18.93 serial days after inputs were ready; no day-21 readiness is
claimed. Source-separated preparation and site delivery need an explicit independent critical path.

Bounded R0 hardware tests may use qualified synthetic or retained inputs before the production corpus
exists. That does not permit R1/R2 or base training on unqualified data.

## B0: base training

Keep the 1.2B hybrid fixed. Default to 320B tokens and a 20% WSD decay; preserve the pre-decay
continuation asset. Select 400B before launch only if it fits 2,425 hours and the calendar without
borrowing capability, evaluation or reserve. Required average per-GPU throughput is approximately
36,655 tokens/s for 320B or 45,819 for 400B, including all time charged to B0. These are requirements,
not measurements. No TFLOP/s threshold overrides a measured end-to-end forecast.

If 320B does not fit, stop before B0 for a costed successor. A late shortage cannot quietly remove
capability work. Unspent base allowance remains headroom; optional continuation needs a recorded,
pre-results endpoint change and secure mandatory artifacts.

## L0: capability development

[capability_plan_v1.json](capability_plan_v1.json) partitions 700 hours into 140 for 32K continuation,
180 for 128K/length fallback, 120 for broad SFT, 160 for mixed-length finishing, 60 for one optional
preference/distillation intervention, and 40 for stage integration/diagnostics. Exact-shape measurements
determine tokens. Optional L5 cannot displace evaluation; no new RL infrastructure is assumed.

## V1-V3: release verification

- V1: 110 hours for final quality, public models and retrieval comparisons.
- V2: 90 hours for systems and one accelerated serving path.
- V3: 36 hours for parity, reproducible artifacts and two held-out demonstrations.

R0 covers early fit/parity probes; V2 covers final serving economics. Intermediate stage diagnostics
use L6. R4 evaluates the controlled study. This prevents double charging or an unbudgeted comparator
fleet. Pin at least two public models, one near footprint and one stronger/larger reference, plus the
practical retrieval baselines. CPU/GGUF is optional.

## Cuts and reserve

Cut optional preference work, CPU/GGUF and extra demonstrations first; choose 320B over the 400B
stretch before launch; preserve a qualified 32K/64K release if 128K fails. Do not preserve a statistical
claim while dropping its paired seeds or missing controls.

Reserve repairs existing interrupted/invalid work, repairs a frozen capability stage, completes
mandatory verification, and only then replicates a consequential decision or continues the flagship.
Every use records trigger, remaining hours and calendar. Reserve never silently funds a new axis.

## Execution boundary

The selected design does not grant training authority. Each run binds exact Git/config/data/tokenizer,
parent checkpoints, analysis, endpoints, costs and a compatible launch receipt. Existing rights,
firewall and bounded operations evidence remain in force; changed partitions and removed E2 decisions
need explicit tooling qualification before receipts. See [PREGRANT.md](PREGRANT.md).
