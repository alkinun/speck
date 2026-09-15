> **Scope update, 2026-09-15:** the [long-context pivot](PIVOT.md) supersedes old experiment budgets, quotas and selection dependencies in this note. Retained mechanics and measurements remain evidence at their recorded identities. Use [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md) and [FIRST_WAVE.md](FIRST_WAVE.md) for new preparation; do not automatically expand the retired wave.

# Admitted natural Math L2 preparation

The owner explicitly approved the pinned **UltraData-Math L2-preview** source for guarded internal
training and public model-weight release under the existing scope. The
[source-use extension](ultradata_math_l2_source_use_v1.json) records that decision, parent acceptance,
evidence, and the acknowledged absence of original URLs/parent IDs. It adds no Code or L3 approval.

This additive record does not change the frozen tokenizer sample or rewrite its existing source-use
record. The new preparation binds the parent and extension together; later launch receipts must also
carry the relevant source-use lineage.

## First real stock preparation

The [preparation plan](ultradata_math_l2_preparation_v1.json) binds the first two L2-preview shards
(415.4 MB and 343.0 MB compressed) at the admitted revision and exact LFS hashes. It processes the
first 100,000 physical rows in each, preserving content, quality labels, and dataset/shard/row identity.
The source-size and 200M reference-token target are fixed before this execution.

The existing acquisition-unit path applies:

- the inherited 200–100,000-character reader limits, benchmark checks, PII/repeated-line policy,
  and redacted Gitleaks scan;
- the existing math-prose English extractor: formulas/code fences/URLs are excluded from the language
  diagnostic, requiring at least 80 alphabetic characters and English probability ≥0.8;
- no additional `quality_label` cutoff;
- record-boundary recovery, with the source-use extension and math-English settings included in the
  unit configuration identity.

Original web URLs are unavailable. Preserve dataset revision, file, row, content hash and available
metadata; do not invent URL/domain or semantic-parent coverage.

## Complete exclusion and operating evidence

The run reuses a verified private copy of the complete 288,872-reference checkpoint. All six candidate
slots are explicitly rebound: the math slot contains the newly acquired data and the others are empty.
Reference inputs/order, dedup policy, and exact/near controls remain fixed. Old candidate rows are
removed from the private index before processing; none of the old candidate corpus is silently reused.

The config-bound FULL-sync 65,536-page WAL policy is applied with 10,000-record checkpoints. This run
extends observation to a larger within-source workload. Its declared observed-WAL envelope is 2 GiB;
the report also records whether the earlier 512 MiB observation still holds. Neither is a hard SQLite
WAL cap or evidence of a production-scale speedup.

All reference outputs must remain unchanged, controls must pass, and retained candidates must have zero
exact overlap with references under the existing exact/MinHash-candidate/verified-near policy.

## Capacity and use

The run counts Mistral-reference tokens in the excluded math text, including BOS/EOS. It does not pack
training shards or select D5. The measured count answers whether these two files can supply the nominal
200M E1 math-component requirement under that reference tokenizer. Final-tokenizer counts, packing
headroom and launch manifests remain separate requirements.

```bash
uv run --no-sync python -m scripts.prepare_admitted_math \
  research/flagship/ultradata_math_l2_preparation_v1.json \
  results/data/ultradata-math-l2-preparation-20260913.json
```

Use a clean implementation revision and new destinations. `--resume` continues that exact recorded
revision/plan and completed stages. Raw inputs, acquisition reports, grouped text, restored checkpoint,
bound preprocessing config, exclusion result, reference count and progress are retained. A shortfall
or exceeded operating envelope is recorded rather than silently expanding the run.

## Completed first stock

The [checked result](../../results/data/ultradata-math-l2-preparation-20260913.json) retains 169,058
documents and **384,788,210 Mistral-reference tokens**, covering the nominal 200M requirement.
Reference preservation, exact/near controls and zero final exact overlap pass. Observed WAL peaks at
455.00 MiB while processing larger within-source transactions. The
[finding](../findings/2026-09-13-admitted-math-stock.md) records counts, timing boundaries and remaining
final-tokenizer/experiment-view work.

The [Mistral budget-fallback decision](tokenizer_decision_v1.json) now freezes the same tokenizer
bytes used for this stock's count. The 384,788,210 tokens therefore also hold under the selected base
tokenizer. Joint experiment-view eligibility and packed-shard manifests remain to be materialized.
