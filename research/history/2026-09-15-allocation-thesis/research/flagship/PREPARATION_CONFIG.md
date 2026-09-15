# Explicit preparation execution policy

Preparation config **schema v2** binds SQLite settings to the normal preprocessing path. The selected
[local policy](sqlite_preparation_policy_v1.json) cites the checked
[WAL comparison](SQLITE_WAL.md) and declares:

```json
"sqlite": {
  "journal_mode": "WAL",
  "synchronous": "FULL",
  "wal_autocheckpoint_pages": 65536,
  "page_size": 4096,
  "cache_size_kib": 2000
}
```

These settings enter the normalized config fingerprint and therefore the checkpoint contract.
Creation and resume apply and verify the settings through the ordinary database constructor. The
published v2 manifest contains both the requested declaration and actual SQLite values/version.
Reopen verifies the declaration and recorded observations. Changing a policy cannot reuse an old
checkpoint or published output under the same identity.

V1 config normalization, fingerprints, and result schema remain the legacy path. Schema v2 requires
the entire declaration; a partial declaration, weaker synchronization, or an unqualified value is
rejected. The two previously measured page thresholds, 1,000 and 65,536, are accepted by the schema;
the selected policy binder uses the recommended 65,536-page setting and verifies its qualification.

## Bind a new execution config

```bash
uv run --no-sync python -m scripts.preprocess_bind_sqlite \
  /path/to/input-preprocess.json \
  research/flagship/sqlite_preparation_policy_v1.json \
  /path/to/new-preprocess.json \
  --output-directory /path/to/new-prepared-output
```

The binder preserves source identities, order, dedup policy, and checkpoint interval. It creates a new
destination config plus a `.binding.json` receipt linking the parent config, selected policy,
qualification, new config hash, and fingerprint. Existing config/receipt/output paths are not reused.
Binding a setting does not authorize a new corpus or change a source recipe.

Run or resume through the ordinary entry point:

```bash
uv run --no-sync python -m scripts.production_data_preprocess_batched /path/to/new-preprocess.json
```

The schema also works through `production_data_preprocess`; the MinHash implementation choice remains
part of the separately recorded execution. No experiment-scoped WAL override is needed. A WAL observer
used for a recovery probe must agree with the bound declaration and cannot override it.

## Config-bound qualification

From a clean implementation revision, with new destinations:

```bash
uv run --no-sync python -m scripts.qualify_bound_sqlite \
  research/flagship/sqlite_preparation_policy_v1.json \
  /mnt/speck-data/speck/bound-sqlite-policy-v1 \
  results/systems/bound-sqlite-policy-20260913.json
```

The qualification restores the known reference checkpoint into a v2 config, observes the configured
settings during an abrupt committed-WAL crash, and resumes with the ordinary preparation CLI. It
requires original output/removal/count/logical-index parity, a v2 manifest with the correct actual
settings, and an identical ordinary-CLI reopen. This is interface/binding qualification; the earlier
comparison supplies the local speed/space envelope. Larger transactions/indexes and site storage still
need qualification before extrapolating the policy to production scale.

## Completed binding qualification

The [checked result](../../results/systems/bound-sqlite-policy-20260913.json) passes config-bound
committed-WAL recovery, ordinary-CLI publication, complete output/logical-index parity, and ordinary-CLI
reopen. Actual observations match the declaration, including FULL sync and 65,536 pages. See the
[finding](../findings/2026-09-13-bound-sqlite-policy.md) for the interface and evidence boundary.
