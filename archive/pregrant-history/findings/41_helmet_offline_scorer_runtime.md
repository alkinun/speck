# 41 — HELMET offline native scorer runtime

## Question

Can HELMET's `pytrec_eval` retrieval/reranking metric path be built and executed reproducibly without
allowing its package setup to download mutable nested source?

## Transitive source audit

`pytrec_eval==0.5` is available only as a source distribution. Its `setup.py` downloads NIST
`trec_eval` v9.0.8 from GitHub when a local `trec_eval/` tree is absent, but it applies no checksum.
The new builder pins both layers:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| `pytrec_eval` 0.5 sdist | 15,248 | `d9eb4616e7d6b73bf1b5ba4c2c4916e88124e790b53fd61610c30158999e7bde` |
| NIST `trec_eval` v9.0.8 archive | 189,343 | `c3994a73103ec842e12df693749584a45814c35c36dcc15f38984bd463566ba1` |

The builder safely extracts both archives, injects the exact NIST tree before setup runs, and blocks
IPv4/IPv6 socket operations throughout compilation. `pytrec_eval` is MIT. The NIST archive contains
copyright notices but no repository license file, so sources and binaries remain in the local research
cache rather than being redistributed by Speck.

## Compiler qualification

The first diagnostic build failed because GCC 16 defaults to C23, where the old empty-parameter
declarations in `trec_eval` conflict with their later typed definitions. The qualified build pins GNU
C17. It also sets a fixed source-date epoch, remaps debug paths to `/usr/src/pytrec_eval`, and disables
the linker build ID.

Two complete rebuilds from fresh extractions produce the same 306,583-byte wheel with SHA-256
`48b873ad0eb3e887dde0d90b88ffc3a71f3a5d211c6f063fd7d3cbe34af64a9f`. The same wheel hash was first
observed under a different absolute test bundle, confirming the debug-path remapping works. The runtime
identity is `7af9f92b93cb5d4ba6a1069247afc30d9b162dfeeaee311fd195bb4db52a5a0a` for CPython 3.10.20,
Linux x86-64, glibc 2.43, and GCC 16.1.1.

## Metric check

The built native extension is imported through pinned HELMET `utils.py`. A two-query synthetic ranking
case returns the frozen expected values, including precision@1 `0.5`, recall@3 `1.0`, MRR `0.75`,
MAP@3 `0.66667`, and NDCG@3 `0.77533`. Both builds and the metric process record zero network attempts
after a positive denial self-test.

## Decision

The platform-specific `pytrec_eval` runtime qualifies for local HELMET retrieval/reranking evaluation.
This removes the nested-download blocker but does not qualify other dataset-bound category processing,
the 34GB dataset, a candidate export, or any model score.

## Artifacts

- [Scorer runtime manifest](../results/Speck-Architecture-Promotion-v1/helmet-scorer-runtime-manifest.json)
- [Offline builder](../scripts/helmet_scorer_prepare.py)
- [HELMET contract](../research/architecture-promotion-v1/external/helmet.json)
