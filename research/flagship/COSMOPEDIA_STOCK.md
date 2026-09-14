# Cosmopedia v2 incumbent stock

The [preparation plan](cosmopedia_stock_preparation_v1.json) selects five complete Parquet files
at the already-approved SmolLM-Corpus revision. The [shard manifest](../../results/data/cosmopedia-shard-manifest-20260914.json)
pins 1,881,445 physical rows / 5,879,393,188 compressed bytes. Remote footers establish schema/row
expectations; acquisition verifies complete local hashes and physical counts before accepting units.
The fixed target is **960M selected-Mistral tokens**, covering 800M nominal plus 20% headroom.
Five files are a preparation allocation, not evidence that the target passes.

## Generated text and lineage

The corrected [prior qualification](../../archive/pregrant-history/research/flagship/synthetic_cosmopedia_v2_v2.json)
is the source of per-document filters and declared generator/seed lineage. Released `text` is the
candidate training document. `seed_data` is a source label, not the seed text or an original seed ID.
`prompt` embeds the source excerpt and instruction; generated-text/prompt 5-shingle Jaccard at or
above 0.8 is rejected. Prompt and seed hashes both identify that full released prompt. They do not
establish original seed ancestry or independence. Raw prompts stay in the preserved raw files and
are not copied into derived training records.

Retained records preserve the declared Mixtral generator family, explicitly undisclosed generator
and seed-source revisions, transformation, source label, style, audience, prompt hashes, overlap,
repetition statistic and template-prefix hash. A content locator combines source ID, complete raw-file
SHA and physical row because this release has no original per-document ID. Source revision/file/row
and generated-content SHA remain separately available. Nonempty prompt, style and seed label are
required; missing lineage is counted as a rejection rather than inferred.

Qualified 256–1,000,000-byte size, English probability ≥0.8, duplicate-line ratio ≤0.5, repeated
5-gram ratio ≤0.35, model-identity phrases, PII and secret checks apply. Common acquisition character
limits, benchmark and Gitleaks filtering also apply. Full reference exclusion and exact/verified-near
candidate deduplication follow acquisition. Neither these checks nor source labels establish factual
or answer correctness.

## Explicit full-stock selection policy

Under pre-results preparation delegation, the tokenizer sampler's 2MB-per-template-prefix and
2MB-per-seed-domain quotas are not imposed as full-corpus quotas. Document repetition/overlap
thresholds remain unchanged. The larger view retains natural postfilter proportions and reports
style and seed-source-label byte shares, hashed template concentration and HHI, and multiplicity of
identical full-prompt hashes. The released schema supplies no seed domains; source labels must not
be interpreted as domains. This policy choice is declared before model outputs and does not change
human source-use approval or confer model-launch authority.

The older conditional Cosmopedia bank is overlapping evidence, not additional supply. Source-specific
capacity still needs joint background/treatment exclusion, whole-document allocation, order/seed,
exact quotas, splits and launch contracts. Qualified text can later be cached with a separate plan
binding the actual result hash and frozen tokenizer; no speculative cache result is declared here.

```bash
uv run --no-sync python -m scripts.prepare_cosmopedia_stock \
  research/flagship/cosmopedia_stock_preparation_v1.json \
  results/data/cosmopedia-stock-preparation-20260914.json
```

Run from a clean frozen checkout. `--resume` requires the original execution revision and absolute
plan identity; preserve partial state and attempt history. Concurrent local preparation timings
include contention and are not isolated throughput or GH200 forecasts.
