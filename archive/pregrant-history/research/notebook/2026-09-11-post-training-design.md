# 2026-09-11 — Flagship post-training design discussion

## Context

The owner requested a fresh, diverse post-training pipeline for grant 1, with strong benchmarks and
real-world utility. Early Speck1/1.5/2 models are pilots rather than inherited flagship recipes. The
owner prefers one mode per checkpoint, remains open to Instruct/Think alternatives, has 5090/3090
resources, and reports sufficient staffing. Exact local hardware and teacher budgets remain unknown.

## Work performed

Reviewed the existing SFT implementation and P6 budget, primary SmolLM/Qwen/DeepSeek/OLMo/Liquid
reports, and component-level dataset cards. Created a discussion proposal with a Base + Instruct
default, balanced and grounded SFT, preference learning from actual student errors, and bounded RLVR
against an additional verified-SFT control. Recorded tokenizer, recurrent packing, rollout, evaluation,
and source-lineage constraints.

## Decisions

No recipe, mode, dataset, budget, or training authority was frozen. The proposed 130-hour post-training
envelope replaces P6's 80-hour annealing commitment while preserving its 450-hour total and the
889-hour overall reserve. This requires an explicit successor to the active plan before execution.
Null/adverse continuations retain the earlier qualified checkpoint and narrow scientific claims.

## Evidence and links

- [Current successor to the initial discussion draft](../flagship/POST_TRAINING.md)
- [SPE-176](https://linear.app/openspecklabs/issue/SPE-176)
- [Active execution plan](../flagship/EXECUTION.md)
- [Active machine-readable budget](../flagship/plan_v2.json)
- [Paper contract](../flagship/PAPER.md)

## Open questions

- Is Base + Instruct the final grant-1 release target, or should a dedicated Think screen replace it?
- Does measured 1.2B throughput support the proposed stage lengths within 130 hours?
- Which candidate data components pass provenance, verification, and independent-family evaluation?
- Which local teachers maximize accepted useful examples per GPU-hour?

## Next actions

- Settle release mode and budget substitution with the owner.
- Audit candidate samples and local teacher throughput.
- Freeze exact experiment and evaluation successors before outputs.
- Qualify local-checkpoint SFT and hybrid generation before paid scientific post-training runs.

## Pre-commit cleanup

The uncommitted full proposal was removed after the owner chose the simpler three-stage Instruct
recipe. This entry retains the initial discussion and budget context; the current plan is linked above.
