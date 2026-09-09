# Data and artifact management plan

This is the repository-wide storage, retention, backup, and publication policy. Program-specific data
selection and rights decisions remain under `research/flagship/`.

## Storage classes

| Class | Examples | Canonical location | Git policy |
| --- | --- | --- | --- |
| Source and contract | Code, configs, policies, source cards, manifests, notebook, claims | Git | Track |
| Small evidence | Summary JSON, bounded task output, failure record, final figure/table source | Git | Track append-only |
| Raw input | Downloaded corpora, upstream archives, benchmark payloads | Runtime store | Manifest only |
| Derived data | Filtered text, dedup indexes, packed shards, held-out partitions | Runtime store | Manifest only |
| Training state | Model, optimizer, RNG/data state, partial and milestone checkpoints | Runtime store | Manifest only |
| Full evaluation | Per-example generations/logprobs, traces, profiler output, complete logs | Runtime store | Summary and manifest only |
| Publication | Final checkpoints, exports, model cards, paper package | Hugging Face plus archival store | Identity record and small metadata |
| Convenience mirror | W&B charts and metrics, Linear discussion | External service | Never sole copy |

`speck_base_dir` selects the runtime store. Flagship work uses `/mnt/speck-data/speck`; code must not
hard-code that maintainer-local path into portable experiment semantics.

## Retention levels

| Level | Meaning | Required handling |
| --- | --- | --- |
| R0 — Ephemeral | Compiler caches, incomplete temporary downloads, disposable staging | May be removed after the owning process or failed attempt is recorded |
| R1 — Reproducible | Intermediate material that can be regenerated from retained inputs | Keep while active; deletion requires a valid manifest and reproducible command |
| R2 — Evidence | Raw outputs needed to audit a finding or paper result | Keep through publication and independent artifact audit |
| R3 — Irreplaceable | Unique checkpoints, signed authority, sealed partitions, expensive source snapshots | Two verified copies before dependent work proceeds |
| R4 — Release | Published model/data/paper/code snapshot | Permanent checksummed archive and public metadata |

Every launch manifest assigns retention to outputs before execution. “Large” and “old” are not deletion
criteria. A cleanup report lists removed paths, bytes, identities, regeneration commands, and retained
parents.

## Required artifact metadata

Each external artifact or tree must be represented by a checked record containing, where applicable:

- stable artifact ID and role;
- producing run and Git revision;
- source/input/parent identities;
- URI or storage-relative path;
- bytes and SHA-256, or an ordered tree manifest;
- media/schema/format version;
- creation time and responsible operator;
- retention level and deletion condition;
- license, attribution, redistribution, and access restrictions;
- backup locations and latest verification time.

A path alone is not identity. W&B run IDs and scheduler job IDs are pointers, not artifact digests.

## Backup and integrity

- R3 and R4 artifacts require at least two independent copies. Copies on the same filesystem do not
  count as independent.
- Verify hashes after copy, before deleting a source, and periodically while the project is active.
- Keep data-launch authority, sealed-audit commitments, and irreplaceable checkpoint metadata in Git
  and in an independent administrative backup.
- Checkpoints are complete only after all payloads, metadata, completion markers, and lineage checks
  pass. Partial/requeue checkpoints never become release artifacts.
- Record filesystem corruption, missing files, and failed restores as evidence rather than silently
  reacquiring a moving source.

## Data lifecycle

1. Pin source identity, rights, provenance, and acquisition method.
2. Acquire into a source-specific immutable area.
3. Filter and emit counts/reasons without editing raw input.
4. Deduplicate and decontaminate through versioned transformations.
5. Partition tokenizer, selection, audit, training, and evaluation identities before consumption.
6. Pack with the selected tokenizer and emit shard/tree hashes.
7. Train only through a data-launch receipt that binds all preceding records.
8. Retain the exact stable/decay/long-document manifests used by each checkpoint.
9. Publish metadata and permitted artifacts; execute removal/deny-ledger obligations.

## Publication and FAIR boundary

Research objects should be findable, accessible under stated authorization, interoperable through
documented formats, and reusable with provenance and licenses. FAIR does not mean every byte must be
open. For restricted or non-redistributable data, publish the metadata, source citation, transformation
description, checksums where lawful, access conditions, and exact reason the bytes are unavailable.

Before paper release, choose the DOI-minting archive, verify its size and versioning limits, and create
an archival manifest connecting the Git commit, paper, model release, data metadata, and external
artifacts.
