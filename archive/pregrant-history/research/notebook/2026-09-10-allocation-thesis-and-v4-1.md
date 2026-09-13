# 2026-09-10 — Allocation thesis and DeepSeek-V4.1-Flash review

## Context

The active flagship plan already combined controlled data selection, recurrent/global architecture,
integration, scale, one held-out model, and hardware measurement. Review exposed two risks: the paper
could read as a large collection of ablations, and the near-constant-tokens-per-parameter scale ladder
could not select between the proposed 600M/800B and 1.2B/400B endpoints. DeepSeek-V4.1-Flash was
released during this pre-results period with a new asymmetric-compute and compressed-cache design.

## Work performed

Audited the official X announcement, release page, MIT-licensed Hugging Face repository at revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, 51-page technical report, model configuration, reference
inference implementation, quantized kernels, Engram implementation, prompt encoding, and benchmark
reproduction instructions. Compared its CED, CSA2, hierarchical indexer, FP4 cache, bounded replay,
mHC, Engram, DSpark, optimization, training, and post-training claims with Speck's implementation and
completed Reader Attention/KDA evidence.

## Decisions

- Frame grant 1 around allocation of training data and exact sequence memory, not a catalogue of model
  components.
- Reduce the paper to four primary claims: data allocation, exact-memory allocation, transfer and
  composition, and held-out systems realization.
- Freeze the flagship target to 1.2B/400B with a 1.2B/320B throughput fallback. The 600M planning
  geometry remains historical/future evidence but is no longer a grant-1 selection endpoint.
- Increase I1 from three to five paired seeds by transferring 22 hours from the scale program's
  generic 107-hour rerun contingency, and use the remaining 85 hours for one predeclared matched 350M
  dense/hybrid mature-horizon continuation. This preserves the qualified 4,111/889 Slurm boundary.
- Add no CED, CSA2, Reader Attention, FP4, mHC, Engram, DSpark, multimodal, or MoE launch decision to
  grant 1.
- Expand systems reporting to separate prefill, decode, runtime state, persistent state, and generated
  token cost. V4.1 is an analytic reference, not a locally measured comparator.
- Use cheap state/cache interventions in I3 to test mechanism without adding training arms.

## Evidence and links

- [`papers/48_deepseek_v4_1_flash.md`](../../papers/48_deepseek_v4_1_flash.md)
- [`research/DIRECTION.md`](../DIRECTION.md)
- [`research/flagship/PAPER.md`](../flagship/PAPER.md)
- [`research/flagship/architecture_plan_v2.json`](../flagship/architecture_plan_v2.json)
- [`research/flagship/integration_plan_v2.json`](../flagship/integration_plan_v2.json)
- [`research/flagship/plan_v2.json`](../flagship/plan_v2.json)

## Open questions

- What exact 350M mature-horizon endpoint fits after measured GH200 dense/hybrid throughput?
- How many long-context tokens are required before 32K and 128K capability plateaus?
- Which persistent-cache metrics are meaningful for a locally deployable 1.2B model?
- Can an independent reviewer be named before confirmatory output begins?

## Next actions

- Materialize per-arm experiment and analysis contracts before grant output.
- Finish production-data calibration and decide its explicit full-scale fallback.
- Qualify the four-GH200 stack and reconcile the analytic/observed flagship budget.
- Generate paper figures and tables only from checked results under the four-claim registry.
