# Complete reference exclusion and bank handoff

The [integration contract](firewall_integration_v1.json) connects the measured acquisition-unit and
source-bank paths using the complete twelve-view firewall reference superset. It is an engineering
qualification with a pinned reference tokenizer, not final corpus or model-training authority.

## Inputs and fixed method

- The [larger acquisition plan](acquisition_unit_rehearsal_v2.json) uses `[0, 2048)` and `[2048, 4096)`
  physical rows from each of the same six pinned raw files. It reuses the retained raw cache, expands
  the declared row coverage eightfold, and checkpoints acquisition every 256 reader-yielded rows.
- The selected v4 firewall contract binds both the prepared input manifest and the final firewall
  manifest. The twelve complete reference views are derived from those commitments, in unseen-then-
  primary order across the six categories. Final firewall input paths and hashes must match the
  prepared views exactly. Audit payloads are never opened by this procedure.
- The existing reference superset contains **288,872 records** and approximately **1.49 GB**. All
  references take precedence over every candidate. The pass must preserve each reference output
  byte-for-byte, preventing dropped reference records from silently weakening exclusion.
- Completed acquisition units are verified and concatenated by category in their fixed unit order.
  Every original JSONL record, including its source row and metadata, is preserved.
- Global dedup uses the existing normalized exact / MinHash-candidate / verified-Jaccard policy and
  the intended **10,000-record checkpoint interval**. This is not exhaustive all-pairs near matching.

## Controls and recovery

Two lower-precedence controls are materialized before the exclusion pass. The first eligible
`web_unseen` reference with at least 100 lexical tokens and an uncapped tail supplies an exact replay
and a one-token appended near replay. Eligibility also requires the declared near threshold and a
shared MinHash band. This constructs an engineering positive control; it selects no training recipe.
Both control outputs must be empty, with the expected exact/near removal reason and a reference owner.

The run deliberately interrupts on the first candidate record after all references have been
processed. It must recover from the complete reference checkpoint with source index 12 and the exact
reference record count. The interrupted metadata is retained independently before resume advances it.
Resume therefore checks the full reference index, rather than only a tiny initial checkpoint.

This is one full reference build plus injected recovery. Small-fixture tests compare uninterrupted
and resumed outputs; a second full uninterrupted reference build is not claimed. Final verification
checks reference preservation, control removals, and zero retained candidate/reference exact-hash
overlap using the completed SQLite index. Per-category exact/near removals are retained separately.

## Bank handoff and capacity outcome

The source-bank v2 schema additionally binds the firewall plan and verifies the parent's complete
reference identities, order, policy, and preserved outputs. Only the corresponding
`acquired_train__<category>` source groups can supply this handoff. V1 remains the frozen retained-pilot
path.

The fixed handoff quota is **16,384 UTF-8 text bytes per category**, with whole-document overshoot and
Mistral-reference packing. This small quota tests the complete pipeline, not E1/E3 training supply.
If any excluded category cannot supply it, record the capacity failure and retain the exclusion
results. Do not search alternative windows or reduce the quota after the result.

## Execution and artifacts

From a clean implementation commit, with enough command time for full reference indexing:

```bash
uv run --no-sync python -m scripts.firewall_integrate \
  research/flagship/firewall_integration_v1.json \
  results/systems/firewall-integration-20260913.json
```

The runtime directory is frozen in the plan. The driver preserves acquisition reports, grouped-source
identities, control inputs, exclusion config/results, interruption metadata, analysis, bank plan/results,
and stage progress. `--resume` continues the same runtime only from its original clean Git revision
and plan identity. It preserves completed stage reports. Costs lost before an external driver stop
remain explicitly outside any complete timing claim.

The final result binds its implementation commit, inputs, parent manifest, checkpoint snapshot, and
bank identities. Stage durations separate acquisition, reference build through interruption, candidate
continuation/recovery, and bank preparation. Reference setup, hash verification, cache state, six-source
coverage, and the bounded token horizon must remain explicit when interpreting rates. A successful
integration alone does not establish 150B throughput, full E1/E3 treatment capacity, or final D5 selection.
