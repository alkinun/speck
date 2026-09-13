# The admitted Math L2 source now has a measured text stock

The owner approved natural UltraData-Math L2-preview under the
[additive source-use record](../flagship/ultradata_math_l2_source_use_v1.json), accepting the documented
missing URL/parent identifiers within the existing guarded-use scope. Code and L3 are not covered.

The [preparation result](../../results/data/ultradata-math-l2-preparation-20260913.json) was produced by
clean revision `988fe5158684717f41287ee2a6adc3dc51f00e98` under the
[fixed two-shard plan](../flagship/ultradata_math_l2_preparation_v1.json). Both complete downloaded shards
match their pinned LFS hashes and each contains 100,000 physical rows.

## Prepared stock

| Stage | Records |
| --- | ---: |
| Physical source rows | 200,000 |
| Rows yielded after reader length/content checks | 199,998 |
| After benchmark, PII, repetition, math-English and Gitleaks filters | 174,090 |
| After complete reference exclusion and candidate deduplication | **169,058** |

The final math text contains **1,285,393,547 UTF-8 bytes** and **384,788,210 Mistral-reference tokens**,
including BOS/EOS. That exceeds the nominal **200M-token** math-challenger requirement, with reference-
token headroom. The tokenizer is explicitly a counting reference; final D5 packing remains pending.
This stock is tracked independently rather than added to potentially overlapping earlier banks.

Acquisition removes 10,461 records matched by the frozen benchmark-contamination policy, 7,798 raw
email/IP records, 3,390 repeated-line records, 4,117 failing the math-prose English diagnostic, 136
with insufficient prose, and six Gitleaks-affected records. These reasons are sequential and follow
the reader filters. Benchmark matches in this released source do not establish contamination in the
authors' trained models, which may have used their own downstream filtering.

Exclusion removes 660 exact and 4,100 verified-near duplicates within the new candidate stream, plus
272 verified-near matches to the complete firewall reference superset. The exact and near positive
controls also pass. Every reference output remains unchanged and final exact candidate/reference
overlap is zero. The other five candidate slots are empty, so no old candidate corpus is silently
counted as new math supply.

## Operating evidence

- Acquisition/filtering: **1,148.98 seconds**.
- Private reference-prefix restoration: **91.14 seconds**.
- Exclusion invocation: **969.78 seconds**, including **609.81 seconds** in SQLite commit.
- Final index: **567,472,128 bytes**, with 457,930 accepted records including the reference prefix.
- Observed WAL peak: **477,104,272 bytes / 455.00 MiB**, inside both the previous 512 MiB observation
  and this run's separately declared 2 GiB observed-space envelope.
- Main-process peak RSS: **3,268,927,488 bytes**, including source-filter setup and reference counting.

This exercises 10,000-record checkpoints within a substantially larger single-source candidate stream.
It is not a paired speed comparison or a production-scale rate. The configured FULL-sync policy,
source-use extension, and math-English rules are bound into the relevant configuration identities.

## Remaining work

The prepared text can supply the proposed challenger once final-tokenizer counts, arm-specific
background/treatment membership, packing headroom, and launch manifests are qualified. It is not a
complete balanced E1/E3 corpus and does not resolve the incumbent science/reference capacity gaps.
No model-training authority is issued by this preparation result.

For Code, the pinned repository root and `data` directory expose L2/L3 assets but no metadata mapping
at those locations. The named `UltraData-Code-L0` and separate Math L2/L3 repository aliases were not
accessible with the current client; this does not prove that no private or differently named resource
exists. Code's repository/file-license and lineage requirements therefore remain open, with the
previously specified approved-source fallback available before source-screen outputs.
