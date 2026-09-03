# 05 — Long-document dataset

## Motivation

The existing 5B-token packed mixture is a flat source stream. Packing unrelated web documents into
32K sequences would expose the runtime to long inputs but would not supervise long dependencies.
The existing manifest and per-source document indexes made it possible to derive a small dataset of
complete long documents without redownloading or retokenizing the source corpus.

## Candidate supply in the parent corpus

Documents at least 16,384 tokens long:

| Source | Documents | Tokens | Longest document |
| --- | ---: | ---: | ---: |
| peS2o | 1,092 | 21,552,626 | 36,807 |
| Wikimedia | 215 | 4,524,834 | 41,772 |
| FineMath 4+ | 1,097 | 25,703,494 | 86,349 |
| **Total** | **2,404** | **51,780,954** | **86,349** |

The parent packed-data manifest is
`b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e`.

## Derived recipe

- Dataset: `SpeckLC-LongDocs-32M`
- Requested training tokens: 32,000,000
- Minimum complete-document length: 16,384 tokens
- Mixture: 50% FineMath 4+, 40% peS2o, 10% Wikimedia
- Validation target: 100,000 tokens per source
- Training reserve: 131,072 tokens per source
- Output shard size: 16,000,000 tokens
- On-disk size: approximately 64 MiB
- Derived manifest:
  `0a14833ad84d0f240fd7787e542c47c2f77f40d73427c207cc7ae6b2a95f9da0`

The derivation copies exact token spans using the original document index, preserves BOS/EOS
boundaries, rewrites offsets, and rebuilds shard, document-index, and deduplication hashes. It is
atomic and validates the complete new manifest before publication.

## Produced data

| Source | Train docs | Train tokens | Median / max train length | Val docs | Val tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| FineMath 4+ | 688 | 16,146,125 | 20,986.5 / 86,349 | 6 | 115,620 |
| peS2o | 657 | 12,937,178 | 18,957 / 36,807 | 5 | 109,301 |
| Wikimedia | 157 | 3,345,840 | 19,767 / 41,772 | 5 | 113,790 |
| **Total** | **1,502** | **32,429,143** | — | **16** | **338,711** |

Every selected document is at least 16K tokens. A 32K training sequence therefore contains one or
two complete long documents rather than dozens of unrelated short web pages.

## Command and artifacts

```bash
uv run --extra gpu --extra linear python -m scripts.long_document_data_prepare \
  experiments/SpeckLC-LongDocs-32M
```

- Recipe: [experiments/SpeckLC-LongDocs-32M](../experiments/SpeckLC-LongDocs-32M)
- Derivation implementation: [speck/long_data.py](../speck/long_data.py)
- Entry point: [scripts/long_document_data_prepare.py](../scripts/long_document_data_prepare.py)
- Runtime dataset: `~/.cache/speck/data/SpeckLC-LongDocs-32M`
- Implementation commit: `bbef983`

## Limitations

- The validation set has only 16 complete documents; its absolute loss has nontrivial sampling
  uncertainty.
- This is long-document next-token supervision, not explicit long-range retrieval or aggregation
  supervision.
- The maximum document is 86,349 tokens and only a small fraction approach that length. This corpus
  does not justify 128K context training.
- The three source domains are narrower than the original 11-source quality mixture, so original
  4K validation must be rerun after every promotion.
