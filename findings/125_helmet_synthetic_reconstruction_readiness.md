# 125 — HELMET synthetic-recall reconstruction boundary

## RULER is not interchangeable across tokenizers

HELMET's v3 paper source states that its RULER inputs were generated with the original scripts and
the Llama-2 tokenizer. Speck's 1,500 already-qualified overlapping cases use the same currently pinned
RULER generator revision, but deliberately use the Speck endpoint tokenizer. They are deterministic,
offline, and valid independent RULER evidence; they are not HELMET cases. The paper does not identify
the exact RULER code revision, tokenizer payload hash, commands, or output hashes used for the archive.

## JSON-KV is described, not specified

The paper says each JSON dictionary uses random UUID keys and values and constructs six queries at
evenly spaced depths. The cited Lost-in-the-Middle implementation is only an ancestor: it releases
75/140/300-key data, makes one query per dictionary, uses unseeded `uuid4` plus unseeded gold selection,
and has a different schema. HELMET instead names 105/220/440/900/1800-key files, consumes two demos,
and expects additional fields. Neither the pinned tree nor the paper supplies its generator, seeds,
serialization, depth expansion, length fitting, row counts, or output hashes. The config selects at
most 100 rows while the paper reports 600 JSON-KV examples, leaving the sampling unit unresolved.

## Decision

Exact official-HELMET reconstruction is blocked for both branches. A separately named deterministic
JSON-KV diagnostic could be designed, but it would not satisfy the HELMET release gate and is therefore
not a justified substitute. No generation, extraction, evaluation-manifest edit, or model execution is
authorized. Obtain the original generation materials and tokenizer identity, or retain the Speck cases
only under their existing independent RULER label.

## Artifact

- [Synthetic reconstruction readiness](../results/Speck-Architecture-Promotion-v1/helmet-synthetic-reconstruction-readiness.json)

Validate it offline with:

```bash
python -m scripts.helmet_synthetic_reconstruction_validate \
  --readiness results/Speck-Architecture-Promotion-v1/helmet-synthetic-reconstruction-readiness.json \
  --rights-audit results/Speck-Architecture-Promotion-v1/helmet-archive-local-rights-audit.json \
  --helmet-contract research/architecture-promotion-v1/external/helmet.json
```
