# Retained-document engineering windows preserve boundaries and finite exposure

The [checked real-data receipt](../../results/data/finewiki-document-windows-20260916.json) records
eight 32K input windows from eight existing, filtered FineWiki documents. Each read includes its
within-document lookahead token, for 262,144 input positions. The explicit longest-first order is
bound in the [preparation plan](../flagship/finewiki_document_window_preparation_v1.json); no random
recipe selection, raw long-candidate intake, concatenation or new source acquisition occurred.

The new engineering reader verifies the stock index and token payloads, validates each window against
its original document span, reads physical shard crossings, and stops at the finite end. A serialized
middle cursor was reopened using a new reader in the same process and produced identical tokens,
document identities and next cursor. Per-window payload hashes are retained in the receipt. Eight
fixture tests additionally cover rank-disjoint assignment, incompatible resume geometry, insufficient
lookahead, overlapping inputs, absent documents, changed payloads and changed view spans.

This is a useful input-mechanics step, not a production-loader or training qualification. No model,
optimizer, GPU or distributed process was run; actual model reset behavior remains unqualified.
Original content IDs/digests are retained, but edition/fork/family grouping and train/evaluation
partitions remain unqualified. These deliberately selected mechanics inputs are not a quality test.
Reader initialization currently reopens complete stock payloads, so multi-rank startup cost still
needs a production design. See [DOCUMENT_WINDOWS.md](../flagship/DOCUMENT_WINDOWS.md) for the boundary.

The service `speck-document-window-check-20260916.service` exited successfully. Its log is retained at
`/mnt/speck-data/speck/document-window-check-20260916.log`; existing stock bytes are unchanged.
