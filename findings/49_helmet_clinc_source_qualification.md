# 49 — HELMET CLINC150 source qualification

## Question

Can the exact CLINC150 `plus` inputs used by HELMET's ICL category be retained as immutable,
rights-attributed, data-only artifacts and traced completely to the pinned upstream dataset?

## Source and rights

The Hugging Face snapshot is pinned at `155b9c710419136e17307b80d0a13e68cd46b4ec` and declares
CC-BY-3.0. The original `clinc/oos-eval` source is independently pinned at
`828f8093932c8fe6ca7936c3d2e52903b1c523de`. Its 19,467-byte CC-BY-3.0 license and
2,509,789-byte `data_oos_plus.json` are retained beside the Parquet payloads with exact hashes.

Only the splits HELMET actually reads were acquired. The 312,096-byte train Parquet has SHA-256
`30188119…`; the 77,789-byte validation Parquet has SHA-256 `fbd545b4…`. The unused test split was not
downloaded.

## Full provenance parity

The `plus` train split contains 15,250 rows and is exactly the upstream `train` list followed by
`oos_train`. Validation contains 3,100 rows and is exactly `val` followed by `oos_val`. Every text,
intent label, and row position agrees. Both Parquet splits also share the frozen feature identity for
all 151 intent names, including the out-of-scope label.

The retained snapshots reload in the current datasets-5 runtime with network access disabled and
reproduce the frozen row and feature hashes. This avoids both mutable Hub `main` resolution and legacy
remote code.

## Decision

The CLINC150 source snapshot required by HELMET ICL qualifies. Together with Finding 48, three of the
seven archive-external source families now have a technically qualified immutable path: Banking77,
NLU Evaluation Data, and CLINC150.

This does not qualify the ICL category. TREC remains rights-blocked, the final five-dataset prompt
matrix has not been materialized, and no contamination scan or model execution has occurred.
Attribution and license retention remain required.

## Artifacts

- [Frozen CLINC protocol](../research/architecture-promotion-v1/helmet_clinc_source_v1.json)
- [Source qualification](../results/Speck-Architecture-Promotion-v1/helmet-clinc-source-qualified.json)
- [Qualification runner](../scripts/helmet_data_only_source_qualify.py)
- Private source directory: `/mnt/speck-data/speck/helmet-runtime-sources-v1/clinc_oos`
