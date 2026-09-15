# Inspect code metadata before downloading complete text shards

The [one-file qualification](../../results/systems/stack-v3-metadata-range-qualification-20260915.json)
passes on a pinned Stack v3 shard already present as a complete, SHA-256-verified local file.
The remote path fetched **16,149,873 payload bytes in eleven range attempts**, compared with
**272,393,747 bytes** for the complete file: **5.93% of its size**. This is a byte-volume comparison,
not a measured download-speed ratio. Range acquisition took 17.02 seconds and the subsequent
projection/parity/reopen checks brought this invocation to 18.49 seconds. Initial complete-file
hashing and HTTP/TLS/redirect overhead are outside these respective timing/byte boundaries.

Every retrieved byte range matches the complete verified local file. Decoded metadata agrees
exactly over **21,014 repositories / 231,664 physical files**, and reopening the saved ranges
reproduces the projection without network access. The reader retrieves compressed metadata
columns, coalescing small gaps. Footer reads and merged gaps can include adjacent unselected
bytes; no source-content column is decoded. The source text remains in the original complete file.

The implementation requires HTTP 206, exact Content-Range, identity encoding and exact payload
length. A whole-file response fails rather than becoming an implicit bulk download. Attempt
receipts, partial bodies and failures remain preserved; resume requires identical contracts and
valid completed payload hashes. Byte reservations include failed attempts and one-byte overrun
detection. Tests cover malformed responses, corruption, interrupted-body preservation, concurrent
budget enforcement and metadata-only reopen on actual Parquet fixtures.

This removes the need to download every source text shard merely to inspect license/language
supply. It does not establish token capacity or permit training from a partial file. The
[bound successor](../flagship/stack_v3_metadata_discovery_v1.json) inspects sixteen new file
identities from a preserved pinned tree page. Their complete files total **6,697,520,579 bytes**;
the metadata inspection permits at most **1 GiB** in conservative range reservations across
sixteen files (64 MiB each). This deterministic file-order tranche is not a random or representative
sample of the full source. Its purpose is to make the next complete-file acquisition reviewable.

The successor keeps the approved source and existing license/vendor/fork/size/path/language
filters. It first checks its census implementation against the earlier complete-file result.
New-file observations remain provisional metadata: declared whole-file hashes are not verified
by partial reads. Full-file hashing, content/prose/security/tokenizer checks and joint exclusion
remain necessary for usable stock. No language shares, shared background, source-use approval,
training authority or scientific experiment slots change.
