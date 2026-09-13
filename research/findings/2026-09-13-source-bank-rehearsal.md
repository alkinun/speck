# Bounded source banks preserve text, packing, and recovery identities

The [source-bank qualification](../../results/systems/source-bank-rehearsal-20260913.json) completes
from clean revision `e5635aae65bc91f627ac9778aedd8bc2f7288cec`, using the frozen
[v2 destination plan](../flagship/source_bank_rehearsal_v2.json). The preceding
[external-timeout attempt](../../results/systems/source-bank-rehearsal-interruption-20260913.json)
and its partial outputs remain preserved.

## Materialized output

The six retained pilot-training sources had already passed global deduplication with all twelve
firewall candidate views taking precedence. This execution verifies their pinned source-file hashes
and selects the shortest whole-document prefixes reaching 2 MB of UTF-8 text per category. Original
JSONL records and metadata are copied verbatim. The pinned Mistral tokenizer is used only to measure
and pack the selected reference-token views; selection itself uses bytes.

| Category | Documents | Selected UTF-8 bytes | Reference tokens, including BOS/EOS |
| --- | ---: | ---: | ---: |
| Web | 518 | 2,002,270 | 491,438 |
| Code | 362 | 2,008,343 | 609,083 |
| Math | 428 | 2,005,461 | 645,438 |
| Synthetic | 541 | 2,002,997 | 431,066 |
| Science | 63 | 2,017,939 | 564,114 |
| Reference | 729 | 2,012,513 | 562,125 |
| **Total** | **2,641** | **12,049,523** | **3,303,264** |

The recovery branch interrupts code selection after record 17, resumes from the durable record-16
checkpoint, and matches the uninterrupted branch's selection counters, selected-text hashes, token
counts, and every packed-shard hash for all six sources. Reopening the clean published bank also
retains its manifest identity. Portable tests additionally cover torn uncommitted writes, invalid
quota counters, retained-input/selected-text/shard corruption, packing failures, source exhaustion,
and tokenizer-independent byte selection.

## Measured operations

| Clean invocation stage | Seconds |
| --- | ---: |
| Original input-file identity verification | 48.512 |
| Whole-document selection and durable checkpoints | 13.575 |
| Reference-tokenizer packing | 2.591 |
| Entire clean invocation, including setup | 64.770 |

The injected-interruption invocation takes 39.739 seconds; recovery takes 23.717 seconds; published
reopen takes 5.206 seconds. These are separate invocation costs. Cached input verification can make
later invocations much faster, so the recovery/reopen times are not production restart forecasts.
The entire qualification process peaks at 301,322,240 RSS bytes. Both successful runtime branches and
metadata total 38,956,274 bytes. Free-space endpoints are recorded separately from that tree size.

## Evidence boundary and next work

This qualifies bounded retained-source selection, reference packing, and recovery. It inherits rather
than reruns acquisition, filtering, global deduplication, and firewall exclusion. It does not establish
all E1 source treatments, 6B/8B/12B experiment supply, a final D5 tokenizer, production throughput,
long-document reconstruction, or model-training authority.

The next engineering step is a bounded acquisition/global-dedup successor with independently resumable
input units, preserved metadata, fixed source precedence, and measured stage/resource costs. The
qualified bank mechanics provide its downstream selection and packing reference. Keep the historical
150B preparation forecast until that end-to-end path has measured evidence.
