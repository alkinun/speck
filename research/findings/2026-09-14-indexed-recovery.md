# An unindexed foreign-key cascade caused the FineMath recovery stall

The owner's observation of one busy CPU core prompted live inspection. The resumed worker used
approximately 97% CPU, while durable progress stayed at checkpoint 72 / candidate row 360,000.
Its process I/O counters accumulated over 2.6 TB of logical reads with about 5 GB of physical reads;
these were largely repeated cached accesses, not newly downloaded data.

The preserved database contained **7,329 committed documents beyond the file checkpoint**, requiring
the existing rollback cleanup. The `bands` table had an index on `(band, band_hash)` but none on its
foreign key `doc_seq`. With `ON DELETE CASCADE`, the recovery delete's query plan included **SCAN bands**
for each removed parent document, over roughly 10.4 million band rows. The initial resumed service was
stopped after **33m 58.265s wall / 33m 13.523s CPU** without a new durable checkpoint.

## Verified narrow repair

On a private diagnostic copy, adding `bands_doc_seq ON bands(doc_seq)` took 1.30 seconds and checkpoint
cleanup took 0.53 seconds. The resulting accepted-document chain exactly matched the preserved
checkpoint; 10,279,968 surviving band rows and foreign-key checks passed. This copy was on the root
filesystem, so these times are not a controlled same-storage speedup claim.

With the live worker stopped, the [recorded migration](../../results/systems/finemath-recovery-index-20260914.json)
added only the physical lookup index. Building it on the data drive took **102.65 seconds**. The
651,694-document checksum chain and 10,397,232 band-row count were unchanged. The query plan switched
to **SEARCH bands USING COVERING INDEX bands_doc_seq (doc_seq=?)**.

The original frozen worker then resumed under `speck-finemath-indexed-resume-20260914.service` and
advanced to **checkpoint 73 / candidate row 370,000**. Final exclusion/capacity publication remains
pending. The source text, filters, tokenizer, checkpoint counters and scientific plan were not changed
by index creation. The migration receipt is an additional operating dependency of this frozen resume.

## Preventing recurrence

New maintained source-stock executions install and verify this complete nonunique index at database
open, before the preserved checkpoint-cleanup logic. The observation is included in the exclusion
report. Tests compare full retained document/band rows against legacy cleanup, check foreign keys,
require the indexed query plan, and reject partial or incompatible named indices. The preserved
`production_data` implementation and original scientific checkout remain unchanged.

The extra index has storage and write costs that future measurements must include. It removes this
pathological cleanup scan; it does not make a single SQLite writer multicore. Independent document
filtering/signature computation remains an opportunity for ordered parallel execution. Thread-count
environment variables alone cannot parallelize a SQLite statement.
