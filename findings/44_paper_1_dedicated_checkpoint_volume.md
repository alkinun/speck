# 44 — Paper 1 dedicated checkpoint-volume qualification

## Question

Can the six-run Paper 1 proxy and later finalist stage retain complete checkpoint evidence without
depending on undocumented cleanup of the crowded root filesystem?

## Provisioning

The newly mounted ext4 filesystem has UUID `b64b59d1-ea2c-4206-9171-b7cd739f3eff`, is physically
distinct from the root filesystem, and is mounted with `rw,nosuid,nodev,noexec`. A new private `0700`
directory at `/mnt/speck-data/speck/paper1-baselines-131m` was empty at qualification time. No existing
checkpoint, optimizer, dataset, result, or export was moved or deleted, and no cleanup is counted as
capacity.

The volume had approximately 5.65TB free. It passes both the 17,179,869,184-byte proxy launch floor
and 25,769,803,776-byte finalist floor by orders of magnitude. The frozen estimates are 8.4GB for six
proxy checkpoints and 16.8GB for twelve finalist checkpoints.

## Operational binding

All six materialized seed/data-order/architecture experiments map to unique checkpoint directories on
the volume. The original packed dataset remains at its existing path and retains fingerprint
`b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e`.

`scripts.base_train` now accepts an operational `--output-dir` override. This changes only checkpoint
location; optimizer, data, tokenizer, architecture, schedule, seed, token count, and every other
scientific value remain config-bound. The qualification report freezes exact launch and result-collect
commands for each run.

## Decision

Paper 1 proxy and finalist checkpoint storage qualify. This closes SPE-104 using positive provisioning
provenance rather than inventing details about the earlier cleanup. The mount is not persistent in
`/etc/fstab`, so its UUID/options/free space must be rechecked after reboot or remount. Storage alone
does not authorize training: SPE-58 remains open.

## Artifacts

- [Storage qualification](../results/Speck-Paper1/baseline-storage-volume-qualified.json)
- [Volume qualifier](../scripts/paper_baseline_storage_qualify.py)
- [Frozen baseline matrix](../research/paper-1/baseline_matrix.json)
