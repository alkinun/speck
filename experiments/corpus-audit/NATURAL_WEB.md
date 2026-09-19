# Bounded natural-web inspection

2026-09-19. Prioritize **UltraFineWeb's newer L1-derived English HQ route** for further
qualification, with retained FineWeb-Edu as the control. This decision rests on recoverable
source identity, not a measured quality win. The [receipt](natural-web-inspection.json) binds
the acquisition, analysis, samples and review notes. No documents are admitted to training.

## What was inspected

UltraFineWeb is pinned to `02c85641e3d19a854be2e09139c25adaa9518063`. Its
[card](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/blob/02c85641e3d19a854be2e09139c25adaa9518063/README.md)
distinguishes the older FineWeb-derived English split from newer Common-Crawl/L1-derived HQ.
The latter is selected L2 material despite `l1` in its directory name. Synthetic L3 is separate.

| Route | Inspected population | Review sample | Origin evidence |
| --- | --- | --- | --- |
| Default `ultrafineweb_en` | Four seeded 12-row viewer windows, with matching revision headers | All 48 rows | Only `content`, `score`, `source`; no original URL, crawl or WARC ID |
| New `ultrafineweb_l1_en_hq` | Six complete shards, one from each 2025 crawl group; 144,876 documents | 64 documents, 16 per byte-length band | All inspected rows contain URL, WARC record ID/date/file, language and classifier scores in JSON metadata |
| Retained FineWeb-Edu | Reused existing audit of 2,097,270 retained documents | Existing 64 documents, 16 per byte-length band | All sample rows retain URL, crawl/WARC identity and pinned acquisition coordinates |

HQ files have `uid`, `content`, `meta`, `dataset_index`; the default schema does **not** apply.
The HQ WARC filename is a basename, not a complete retrieval coordinate with byte offsets.
Local UID/index fields are preserved but not treated as a demonstrated join to another release.
The publisher's project license does not establish individual page permissions: its card explicitly
directs users to check component dataset licenses. No per-document license field was observed.

Acquisition took about 112 seconds for 476.6 MB, including metadata and viewer responses.
All six Parquet files match pinned publisher LFS hashes. Their selection was seeded **within
the first 100 listed files per crawl**; the remaining listing pages were not enumerated. These
are a bounded inspection population, not a representative estimate of the full HQ release.
Do not extrapolate full storage, eligible supply or topic proportions from it.
The [subsequent complete inventory](WEB_INVENTORY_DCLM.md) establishes 6,000 shards and
477.97 GB compressed from file metadata; it does not change this inspection's sampling limits.

## Findings that change preparation

- **Prefer HQ for the next source bank qualification.** Its page/WARC evidence supports source
  tracing, attribution review, family grouping and recovery of damaged extraction. The older
  default split needs a demonstrated join to original FineWeb before it can meet the same standard.
  A broad `source=FineWeb` label alone does not establish that join.
- **Keep score and content checks separate.** A reviewed HQ scientific article scored 0.9557
  but omitted displayed equations and numerical expressions that its prose referenced. Reject
  that unchanged serialization for the technical bank; try origin-based repair without dropping
  the scientific topic. A policy template scored 0.9971, and a default-split workshop agenda
  scored 0.9733. High scores do not guarantee technical depth or extraction completeness.
- **Do not adopt a stricter cutoff yet.** At `score >= 0.8`, 82,166 of 144,876 HQ documents remain:
  56.7% document retention. This is not retained-token yield or a quality improvement. Inspect
  lower-score material and preserve useful domain coverage before choosing a threshold.
- **Preserve broadness with explicit controls.** The small review includes research overviews,
  religious history, business policy, public safety, financial explanation and scientific prose.
  Keep useful nontechnical material; measure template repetition and density instead of treating
  a technical-only corpus as inherently better. These examples are not topic-frequency estimates.
- **Keep deduplication and extraction work open.** The six HQ shards contain 38 duplicate copies
  beyond first occurrences under exact content SHA256, and 111,416 distinct URL hostnames.
  Hostnames are not independent source families. No exact cross-source match was found in the
  inspected scopes; that does not establish disjoint parent corpora or near-duplicate cleanliness.
  A FineWeb-Edu example also contains a long navigation tail requiring boundary cleanup.

The assistant reviewed 11 deterministically selected examples: six full short texts and five
head/tail excerpts. Each has an exact identity, content hash, scope, disposition and limitations
in the receipt. Dispositions concern further inspection or extraction; none authorizes training.
No independent factual review, quality-rate estimate or model benchmark result is implied.

## Comparability and validation

The existing four UTF-8 byte bands are reused: <=2,048; 2,049–8,192; 8,193–32,768; >32,768.
HQ and FineWeb-Edu samples each have 16 records per band. The 48 default rows contain no record
in the longest band. Equal quotas are not source proportions. Saved conditional weights refer
only to the six HQ shards or retained FineWeb stock; default cluster population weights are
not estimated. Scores from the different classifiers are not mutually calibrated.

All 176 sample documents were counted with unchanged text and the frozen Mistral tokenizer,
including BOS/EOS: 259,134 HQ tokens, 34,334 default tokens and 251,210 control tokens. These
544,678 tokens are **inspection material**, not eligible stock. All 64 control counts match
their retained document-index counts. Each 64-document length-balanced HQ/control sample has
19 documents over 4K tokens and none over 32K; this does not qualify 128K source supply.

Offline replay reproduced identical sample bytes and census, metadata and token results.
To replay from the external artifact store into a fresh output directory:

```bash
PYTHONPATH=. .venv/bin/python \
  /mnt/speck-data/speck/data-qualification-20260919/web-comparison/analyze.py replay-new
```

The receipt hashes the acquisition/replay scripts, their inputs, review packet and observations.
Raw corpus text remains outside Git. The original FineWeb stock and frozen pilot are unchanged.

## Next bounded decision

The [follow-up](WEB_INVENTORY_DCLM.md) closes the HQ inventory and now records the completed
12-shard acquisition: 290,761 documents, a 192-record stratified sample and 24 reviewed texts/excerpts.
It also adds partial-viewer DCLM previews. The [extraction follow-up](web-filter-validation.json)
now recovers all three matching archived captures and compares frozen flags on 32 previously
unreviewed documents. False alarms and missed defects keep those flags review-only. The retained
code census is now [complete](code-supply.json); immutable bundle linkage and expansion yield
are the next code steps. Web source-use evidence, parent/duplicate families and source-aware repair
remain open while preserving technical and everyday topic coverage.
Only then count accepted unique tokens. DCLM needs comparable source-file evidence for any
concrete coverage/eligibility decision before freezing allocations. The working 25% selected-web
share remains a hypothesis; this inspection neither changes it nor establishes its 100B-token
eligible-bank target.
