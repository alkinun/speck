# 188 — Production data operations qualify on fixtures, not at 20B

An additive preprocessor now closes the tooling gap ahead of the existing packer. It accepts immutable
precedence-ordered lineage JSONL, performs global normalized exact deduplication, generates MinHash
candidates through a disk-backed SQLite band index, and verifies every near removal with exact
token-shingle Jaccard. The retained SQLite index, outputs, and redacted removal records are hash-bound.

A human-reviewed deny ledger supports content hash, URL, exact/subdomain, and blob identifiers. Every
entry requires a reason, authority, and timestamp. Removal records contain no removed text. Cleanup is
limited to individually named and hashed transient files; source inputs, the ledger, output, and
staging are forbidden targets. Cleanup starts only after manifest publication and produces a receipt,
allowing an interruption between publication, unlink, and receipt to resume safely.

Record checkpoints bind each newly written output slice plus a logical accepted-document hash chain
committed in SQLite. Fixture crashes prove that uncommitted output/database divergence is rolled back,
resumed outputs equal uninterrupted outputs, and corruption inside committed output, SQLite, published
files, or cleanup receipts fails closed.

This preprocessor feeds the unchanged evidence-pinned packer, which already has revision-pinned file
discovery, global exact deduplication, checksummed source-separated uint16 shards, durable file-boundary
resume, and raw cleanup. Thirty-eight focused preprocessor/packer tests pass.

This is not a 20B rehearsal result. It measures no production throughput, memory, storage, or unique
yield; reads no real qualified source; issues no production authority record; and authorizes no
training. Human rights and a frozen real rehearsal manifest remain prerequisites.

Artifacts: [checked tooling result](../results/data/production-data-tooling-20260907.json) and
[production operations plan](../research/flagship/production_data_plan.json).
