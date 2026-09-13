# Config-bound SQLite policy works through normal preparation and recovery

The [binding qualification](../../results/systems/bound-sqlite-policy-20260913.json) passes from clean
revision `b725f235edfaeb07b93fc0b9bab56177d863a299`. The selected
[policy artifact](../flagship/sqlite_preparation_policy_v1.json) links the earlier measured WAL
comparison. This step moves the qualified setting into the ordinary preparation configuration path.

## Identity and runtime behavior

Config schema v2 requires an explicit SQLite declaration. Its normalized fingerprint includes that
declaration, and checkpoints bind the resulting contract. Creation/resume applies and verifies the
declared settings. The published v2 result records both the request and actual runtime observations:

| Setting | Observed |
| --- | --- |
| Journal mode | WAL |
| Synchronization | FULL (`2`) |
| WAL autocheckpoint | 65,536 pages |
| Page size | 4,096 bytes |
| Cache setting | −2,000 KiB |
| SQLite version | 3.53.1 |

Tests reject changed policies when resuming a checkpoint or reopening a published result, partial
declarations, weaker/unqualified settings, and altered runtime observations. Valid schema-v1 config
fingerprints and result behavior retain the legacy path. An experiment observer cannot override a
different bound declaration.

## Real-data normal-CLI qualification

A private copy of the qualified reference checkpoint is restored with the v2 declaration. The hard-exit
observer reports `policy_source=bound_config`: it observes the setting applied by the ordinary database
constructor rather than imposing an experiment override. At exit, 292,808 documents are committed,
while the main file alone contains 288,872. Committed WAL is therefore required to recover 3,936
candidate documents.

The ordinary command `scripts.production_data_preprocess_batched` resumes from that state, discards
the uncommitted SQL update/output tail, and publishes the expected v2 result. All output hashes,
removal records, counts, and logical SQLite tables match the original qualified dataset. A second
ordinary-CLI invocation reopens it with an identical manifest.

The interrupted invocation takes 8.821 seconds, normal-CLI recovery takes 141.895 seconds, and reopen
takes 6.270 seconds. These are binding/recovery measurements, not another speed comparison. The
earlier paired WAL comparison supplies the bounded local speed/space evidence.

## Operational use and remaining work

[`preprocess_bind_sqlite`](../flagship/PREPARATION_CONFIG.md) generates a new v2 config and a binding
receipt connecting the parent config, checked policy/qualification, new config hash, and fingerprint.
The generated config runs through the existing preparation commands. The CLI binder was also exercised
against the retained integration config, producing a loadable successor and matching receipt.

Larger transactions/indexes and site storage still need qualification. The source-recipe discussion
requested by the owner remains open; this execution makes no E1/E3 source or blend selection and adds
no model-training authority.
