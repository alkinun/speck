# 48 — HELMET two-family offline materializer preflight

## Question

Can legacy Hugging Face loader-script data be converted once in an isolated runtime, retained as
immutable data-only snapshots, and read by the current HELMET environment without changing examples,
row order, values, or label semantics?

## Frozen scope

The preflight deliberately covers only Banking77 and NLU Evaluation Data. Both source repositories
carry CC-BY-4.0 license files, and their exact Hub, upstream Git, raw payload, and license identities are
pinned. TREC, Multi-LexSum, Llama 2, NarrativeQA, InfiniteBench, and model judges are outside this
authorization and remain blocked.

The materializer uses Python 3.10.20 with `datasets==3.6.0`, `huggingface-hub==0.36.0`, and
`pyarrow==25.0.1`. A generated requirements file pins all 33 transitive distributions and accepted
artifact hashes. It runs in an isolated environment separate from the model evaluator. The reader
remains the qualified `datasets==5.0.1` environment.

## Banking77

The legacy loader fetched the exact 839,073-byte train CSV and 239,961-byte test CSV expected from the
pinned upstream source. It materialized 10,003 train and 3,080 test rows. Canonical row hashes and the
complete 77-label feature identity matched the frozen values.

Each split was independently written twice. Train Parquet was byte-identical at 293,553 bytes with
SHA-256 `01aa7c72…`; test was byte-identical at 92,409 bytes with SHA-256 `8b8373cf…`.

## NLU Evaluation Data

The legacy loader consumed the exact 5,867,439-byte pinned CSV and produced 25,715 rows. The exact
546,704-byte Parquet artifact from conversion revision `e71b2fe9…` produced the same canonical row hash
`dd8f8699…` and the same complete 68-label feature identity. This checks every row, its order, all
values, and label-name mapping—not only counts or a sample.

## Cross-runtime result

With Hub and Datasets offline modes enabled, the current datasets-5 runtime reloaded all three retained
Parquet snapshots and reproduced every frozen row and feature hash. Dataset-library internal
fingerprints are not treated as cross-version identities; content and semantic feature identities are.

The strategy therefore qualifies for Banking77 and NLU Evaluation Data. It demonstrates that a legacy
materializer can be isolated from the model runtime and replaced by immutable, remote-code-free inputs.
It does not qualify HELMET execution or authorize extension to a source with unresolved terms.
Finding 49 separately qualifies the already data-only CLINC150 source without extending this legacy
materializer result.

## Storage and provenance

The retained snapshots and isolated cache live in a private directory on the dedicated ext4 volume
with UUID `b64b59d1-ea2c-4206-9171-b7cd739f3eff`. Only the compact identity report is committed. No
model, checkpoint, existing evaluation evidence, or HELMET archive partial was moved or deleted.

## Artifacts

- [Preflight protocol](../research/architecture-promotion-v1/helmet_materializer_preflight_v1.json)
- [Hash-locked legacy requirements](../research/architecture-promotion-v1/helmet_materializer_requirements_v1.txt)
- [Preflight result](../results/Speck-Architecture-Promotion-v1/helmet-materializer-preflight.json)
- [Materializer/checker](../scripts/helmet_materializer_preflight.py)
- Private snapshots: `/mnt/speck-data/speck/helmet-runtime-materializer-v1/snapshots`
