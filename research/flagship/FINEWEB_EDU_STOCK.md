> **Scope update, 2026-09-15:** the [long-context pivot](PIVOT.md) supersedes old experiment budgets, quotas and selection dependencies in this note. Retained mechanics and measurements remain evidence at their recorded identities. Use [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md) and [FIRST_WAVE.md](FIRST_WAVE.md) for new preparation; do not automatically expand the retired wave.

# FineWeb-Edu incumbent stock

The [stock plan](fineweb_edu_stock_preparation_v1.json) pins all fourteen complete files of the
released `sample/10BT` view at the already approved revision. The
[manifest](../../results/data/fineweb-edu-shard-manifest-20260914.json) records LFS SHA-256/byte
identities and retained remote Parquet footer identities, schemas and physical row counts.
The original qualified 182,101-row file is `013_00000.parquet`, included in this view rather
than additional supply. Original inputs and review records remain preserved.

The upstream 10BT label does not establish usable selected-tokenizer capacity. The actual target
is **5.28B frozen-Mistral tokens** including BOS/EOS: 4.4B nominal plus 20% preparation headroom
after security checks and full reference/candidate exclusion. The resulting stock is still not
jointly eligible background/treatment membership or a final training manifest. Any capacity
shortfall requires an explicit successor, not a changed target or source substitution.

Preserve the original score >=3, released English language/confidence, independent py3langid
English probability >=0.8, document size, alphabetic ratio, duplicate-line, excluded-host, PII,
secret and Gitleaks rules. Explicit metadata validity guards require a nonempty released document
ID, integer grade in [3,5] and finite released language confidence in [0.8,1]. Retain original
URLs, IDs, crawl/file lineage and exact acquisition row indices. No publication of source text or
new source-use approval follows from this preparation plan.

The old tokenizer-sampler 2MB-per-host limit is explicitly omitted from full stock. Retain natural
postfilter host proportions and report concentration after exclusion. Apply this corpus-level
host policy consistently to the remaining web comparisons before outputs; do not silently add
source-specific caps. This is an agent preparation choice before model outputs, not evidence of
optimal domain proportions or separate human recipe approval.

Raw-file acquisition has an independent durable execution identity and per-file receipts. Resume
requires the same frozen revision/plan and verifies completed payloads again; partial downloads
and failed attempts remain preserved. Each completed file must pass full SHA-256, bytes, physical
rows and required schema. Raw acquisition performs no text exclusion or token counting, allowing
the download stage to proceed separately from the existing source-processing workload.

```bash
uv run --no-sync python -m scripts.prepare_fineweb_edu_stock \
  research/flagship/fineweb_edu_stock_preparation_v1.json \
  results/data/fineweb-edu-raw-acquisition-20260914.json --raw-only
```

After raw-file completion, launch the same stock plan without `--raw-only` into its separate text
output directory and a new text-result path. It uses the common verified acquisition/exclusion
machinery with the explicit web policy. Actual text processing and token-cache launch remain
separate operational steps; raw receipts must not populate paper token-capacity tables.

## First E1S tranche

The [E1S successor](fineweb_edu_e1s_stock_preparation_v2.json) selects complete files 0–2
from the same already-verified fourteen-file intake. It targets **1.32B selected-Mistral tokens**
(1.1B nominal web background for a 2B run plus 20% headroom). This is a smaller operational
milestone; the original 5.28B full-stock requirement remains in force for subsequent preparation.
The three-file yield has not yet been measured. If it falls short, pin additional complete files
in another successor before execution; do not lower the target or relax filters.

The loader requires the completed raw result, preserves every original policy/input and exact
acquisition-unit configuration, and rejects reordered/partial file selections or overlapping
output directories. Both views share the verified raw cache. Their text stocks overlap and must
never be added. Further full-stock preparation can reuse configuration-identical completed
acquisition units, then rerun combined exclusion.

This text plan is conditionally queued in the [finite followup sequence](LOCAL_PREPARATION_FOLLOWUPS.md).
It starts after successful FineMath/Cosmopedia completion and checked caches, avoiding another
large concurrent HDD exclusion workload. No recipe, source-use
approval or model-launch authority changes.
