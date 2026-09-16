# Existing token stocks have very little intact long-document capacity

The [checked index census](../../results/data/document-length-census-v2-20260916.json) covers five existing source-specific caches. It verifies every document-index hash, ordinal, span and total against its bound cache manifest. It does not join or repack documents, reopen token-shard payloads, or qualify family separation.

| Source | Longest document, tokens | Documents ≥32K+1 | Documents ≥64K+1 | Documents ≥128K+1 |
| --- | ---: | ---: | ---: | ---: |
| finewiki_en | 61,037 | 41 | 0 | 0 |
| pes2o_v3 | 31,302 | 0 | 0 | 0 |
| finemath_4plus | 71,218 | 112 | 2 | 0 |
| cosmopedia_v2 | 2,508 | 0 | 0 | 0 |
| ultradata_math_l2_preview | 24,591 | 0 | 0 | 0 |

Counts include existing BOS/EOS and require one extra token inside the same document for next-token lookahead. At 32K there are 41 FineWiki and 114 FineMath disjoint input windows, with overlapping lookahead permitted; these are length candidates, not selected training units. The FineMath count comes from 112 documents. Across these separate stocks, no document reaches 128K+1. This is a statement about the checked snapshots, not the full upstream corpora.

The source preparation plans inherit a 100,000-character common maximum from the archived production plan. Other source-specific filters also constrain their outputs. The current caches were prepared for the earlier short-context experiments; their completion cannot satisfy the new 2–4B coherent-long-unit or 0.5–1B coherent-128K planning envelopes. In particular, arbitrary concatenation would change the unit of evidence and cannot be reported as natural long-document supply.

Every retained FineMath and natural UltraData-Math index row in this census has a null upstream content ID. The original content digest and source-bound ordinal remain available; the census preserves the null instead of inventing an upstream or family identity. FineWiki, peS2o and Cosmopedia have recorded content IDs, but those alone do not establish complete edition/fork/task-family separation.

The first census invocation stopped because its new checker incorrectly required a nonempty string content ID. Its log is preserved at `/mnt/speck-data/speck/document-length-census-20260916.log`; no complete result was published. The corrected invocation preserves nullable/string/integer upstream IDs and checks their digest format while verifying the index bytes. Seven focused tests cover lookahead, non-concatenation, spans, totals, corruption and null-ID preservation. The successful log is `/mnt/speck-data/speck/document-length-census-v2-20260916.log`.

Next, inspect pinned existing raw files for genuine long-unit candidates before selecting any new acquisition. Any larger-document intake requires an explicit preparation successor with bounded memory, retained source-use/security rules, provenance, source-family review and actual post-exclusion yields. Do not silently edit the short-bank filters or treat raw candidate lengths as eligible stock. Keep useful-32K and 128K claims conditional on the required data and capability gates.
