# Complete firewall-reference exclusion connects acquisition to source banks

The [checked integration](../../results/systems/firewall-integration-20260913.json) passes from clean
revision `396a6fa831a442282f0a9f3e533c1321a5970333`, under the frozen
[integration contract](../flagship/firewall_integration_v1.json). It connects larger raw acquisition
units, complete reference-superset exclusion, full-reference checkpoint recovery, and reference-tokenizer
bank preparation. The preceding cohort-only rehearsal remains a separate result.

## Complete reference pass

The v4 firewall commitments bind twelve input views containing 288,872 records and approximately
1.49 GB. The implementation verifies the prepared-view and final-firewall input identities and places
every reference before the candidates. All twelve output hashes equal their original input hashes;
no reference was removed or changed. The sealed audit payloads are not opened by this procedure.

The candidates come from `[0, 2048)` and `[2048, 4096)` physical rows in each of the same six pinned raw
files, eight times the earlier row coverage. Acquisition reuses the raw cache, runs the inherited
reader/security/benchmark filters, and checkpoints every 256 reader-yielded rows. The resulting units
are concatenated in fixed order within each category without changing their JSONL records or metadata.

| Category | Candidate records before exclusion | Exact matches to reference superset removed | Retained records | Retained UTF-8 bytes |
| --- | ---: | ---: | ---: | ---: |
| Web | 4,063 | 127 | 3,936 | 14,908,851 |
| Code | 1,199 | 204 | 995 | 5,466,599 |
| Math | 3,788 | 1,010 | 2,778 | 12,620,199 |
| Synthetic | 4,061 | 717 | 3,344 | 12,390,587 |
| Science | 3,983 | 1,877 | 2,106 | 63,283,435 |
| Reference | 3,676 | 1,700 | 1,976 | 5,735,990 |
| **Total** | **20,770** | **5,635** | **15,135** | **114,405,661** |

These are matches to the **reference superset**, not a count of examples in a particular held-out or
sealed partition. No additional natural-candidate exact or near removals occur in this bounded pass.
The completed index has zero exact content-hash overlap between retained candidates and references.

## Positive controls and recovery

An exact replay and one-token near replay of an eligible `web_unseen` reference are placed after all
candidates. The exact control is removed as an exact duplicate. The near control is removed by the
verified-near path with Jaccard **0.9986667**. Both identify a reference owner and leave empty outputs.
Near matching uses the declared MinHash candidate policy; this is not an exhaustive all-pairs guarantee.

The run checkpoints every **10,000 records** and at source boundaries. An injected interruption on the
first candidate recovers from source index 12 with exactly 288,872 processed and accepted references.
The reference checkpoint metadata and its ordered index chain are retained independently. Resume
verifies the entire reference index before candidate processing. This is one full reference build
with recovery; the result does not claim a second full uninterrupted comparison. Fixture tests cover
uninterrupted/resumed parity and rejected reference-identity/order/policy changes.

## Bank handoff

All six excluded categories satisfy the frozen **16,384-byte-per-category** engineering quota.
The bank v2 materializer validates the parent's complete reference identities, order, policy, preserved
outputs, and category/source mapping before selecting and packing. Its whole-document output contains
22 documents, **117,816 UTF-8 bytes**, and **33,577 Mistral-reference tokens**. Every selected-text and
packed-shard hash was independently reverified after qualification.

## Measured costs and scope

| Stage | Seconds |
| --- | ---: |
| Larger acquisition invocation, with retained raw cache | 175.884 |
| Complete reference build through injected interruption | 1,173.879 |
| Resume verification, candidate/control processing, and final publication/verification | 282.311 |
| Bank selection and reference-token packing | 1.017 |

No raw files were downloaded again. The final index is 376,786,944 bytes and contains 304,007 accepted
records: 288,872 references plus 15,135 candidates. Main-process peak RSS is 2,580,062,208 bytes, including
acquisition/filter setup. Stage times above have different boundaries and are not isolated steady-state
rates; the continuation time includes rebuilding the resume identity and verifying final artifacts.

This closes the bounded **acquisition → complete exclusion → source-bank** integration gap. Full E1/E3
source-treatment coverage, final D5 tokenizer selection, training-scale supply, and a qualified 150B
production rate remain open. The fixed reference setup, cache state, small index, and source mixture
prevent using these timings as a new production forecast. Model-training and production-operations
authority remain false.
