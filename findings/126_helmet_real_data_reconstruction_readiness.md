# 126 — HELMET real-data reconstruction boundary

## Complete archive-local real-data accounting

The remaining 32 configured archive-local paths partition exactly into 24 KILT-derived RAG files,
6 MS MARCO reranking files, and 2 ALCE citation files. The pinned runtime loaders are deterministic
given exact payloads and seed 42. Payload construction is a separate, unqualified process.

## Paper descriptions leave executable gaps

For RAG, the paper names KILT's 2019-08-01 Wikipedia source, 100-word passages,
`Alibaba-NLP/gte-large-en-v1.5`, hard negatives, gold insertion, and six/three depth variants. It does
not pin source payloads, chunking, the retriever revision/runtime/index, answer matching, gold fallback,
length-to-k computation, insertion/permutation seeds, or serialization.

For reranking, it names TREC/MS MARCO, BM25 passages, three relevance levels, balanced sampling, and
three permutations. The exact TREC release and split, qrels, corpus/run identities, balance algorithm,
ID mapping, and seed schedule are absent. Microsoft terms independently keep use authority blocked.

For citation, ALCE's pinned source releases top-100 GTR results from an unversioned data ref, whereas
HELMET consumes undocumented top-2000 files. Neither repository releases that construction. The exact
ASQA/QAMPARI/corpus identities, GTR revision and index, retrieval ordering, and full rights chain remain
unresolved.

## Decision

All 32 paths are technically and legally dispositioned, but none can be exactly reconstructed or
replaced from current evidence. Deterministic loading does not rescue undocumented construction.
No archive extraction, source acquisition, retriever execution, candidate evaluation, clean-room
substitution, or frozen-manifest change is authorized.

## Artifact

- [Real-data reconstruction readiness](../results/Speck-Architecture-Promotion-v1/helmet-real-data-reconstruction-readiness.json)

Validate it offline with:

```bash
python -m scripts.helmet_real_data_reconstruction_validate \
  --readiness results/Speck-Architecture-Promotion-v1/helmet-real-data-reconstruction-readiness.json \
  --rights-audit results/Speck-Architecture-Promotion-v1/helmet-archive-local-rights-audit.json \
  --helmet-contract research/architecture-promotion-v1/external/helmet.json
```
