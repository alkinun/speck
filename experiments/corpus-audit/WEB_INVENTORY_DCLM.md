# Natural-web qualification: HQ sample and DCLM preparation

2026-09-19. The pinned twelve-shard UltraFineWeb HQ acquisition, census and bounded content review
are complete. The [new receipt](web-hq-stratified.json) binds the results and replayable
[audit script](audit_web_hq.py). The earlier [inventory/DCLM receipt](web-inventory-dclm.json)
preserves the complete file inventory, partial-viewer DCLM previews and original acquisition plan.
The [source recovery and review-flag comparison](web-filter-validation.json) below completes
the bounded extraction follow-up. This extends the [initial web inspection](NATURAL_WEB.md);
no training data is admitted.

## UltraFineWeb HQ inventory is complete

At revision `02c85641e3d19a854be2e09139c25adaa9518063`, the
[`ultrafineweb_l1_en_hq` directory](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/02c85641e3d19a854be2e09139c25adaa9518063/data/ultrafineweb_l1_en_hq)
contains **6,000 Parquet shards, 477,974,475,357 compressed bytes** (477.97 GB, about 445.15 GiB).
Each crawl contains exactly numbered parts 1–1,000, with terminal listings and distinct LFS hashes.

| Crawl | Shards | Compressed GB |
| --- | ---: | ---: |
| CC-MAIN-2025-30 | 1,000 | 73.072 |
| CC-MAIN-2025-33 | 1,000 | 75.814 |
| CC-MAIN-2025-38 | 1,000 | 78.099 |
| CC-MAIN-2025-43 | 1,000 | 91.815 |
| CC-MAIN-2025-47 | 1,000 | 81.254 |
| CC-MAIN-2025-51 | 1,000 | 77.921 |

All six previously acquired shard identities match this inventory. They cover every crawl but
only 0.1% of files, selected within the first 100 filenames per crawl. Inventory completion
does not retroactively make that sample representative. File sizes do not establish release
document counts, Mistral tokens, eligible supply or the size of a decoded/tokenized working copy.

## Twelve-shard acquisition and stratified review

Two deterministic hash-ranked files per crawl were acquired from the complete inventory:
**959,301,706 compressed bytes**, within the frozen 1.2 GB cap. All twelve match publisher LFS
SHA256 and size; none is one of the earlier six files. Acquisition took 175.8 seconds. The
old inventory receipt's unacquired status describes that earlier point and remains unchanged.

The full acquired population contains **290,761 documents / 1,614,709,043 UTF-8 content bytes**.
All records have URL, WARC record/date/file, language and classifier metadata. Upstream language
labels are all English; independent language verification remains open. There are **162 exact
and 162 normalized duplicate copies** beyond first occurrences, **109 repeated URL copies**, and
no repeated local UID or WARC record ID. This is within-sample equality checking, not a
near-duplicate screen, a cross-source comparison or an eligible unique-token count.

| Candidate predicate | Documents retained | Document share | UTF-8 content bytes retained | Byte share |
| --- | ---: | ---: | ---: | ---: |
| `pred_score >= 0.65` | 227,501 | 78.24% | 1,283,028,582 | 79.46% |
| `pred_score >= 0.8` | 166,027 | 57.10% | 949,786,370 | 58.82% |
| `pred_score >= 0.95` | 82,128 | 28.25% | 480,898,747 | 29.78% |

These exact counts apply only to the twelve acquired shards. They are not token-retention rates,
full-release yield or evidence that a higher cutoff improves learning. No cutoff is selected.

All **96 crawl × score × byte-length cells** are populated. The score bands are [0.5,0.65),
[0.65,0.8), [0.8,0.95), [0.95,1.0]; byte boundaries are 2,048 / 8,192 / 32,768. Two bottom-hash
records per cell give **192 complete sample documents**, containing **682,970 Mistral tokens**
with BOS/EOS. Of these, 57 exceed 4K tokens and none exceeds 32K. These token counts describe
inspection material, not training supply or proof of the long-context target.

The population has 200,262 URL hostnames; the largest is PubMed with 482 records (0.166%).
Concentration can be higher inside small cells: one long/high-score cell has 11 of 164 records
from the same hostname (6.71%). Hostnames are not source families; subdomains and mirrored articles
still need grouping. Cell quotas overrepresent rare lengths, and the stored weights apply only
within the acquired population. Twelve selected file clusters do not establish corpus quality rates.

A separately frozen reading panel covers every crawl/score pair, rotating the length band:
**24 assistant-reviewed documents, eight full texts and sixteen head/tail excerpts**. The other
168 samples are available for later review, not implicitly annotated. Observations are qualitative;
no independent factual verification, corpus-wide topic proportions or defect rates are claimed.

- **High scores do not fix extraction.** A tutorial at 0.99998 requires configuration values in
  an absent graphic. An explainer at 0.99079 introduces several lists/examples with missing content.
  A study page at 0.92344 mixes lesson text with upload UI and unrelated study-guide teasers.
- **Lower scores still contain useful coverage candidates.** The panel includes an applied-AI
  engineering article at 0.59586, a specialist research abstract at 0.75755, an appliance specification
  list at 0.52180 and an economics explainer at 0.67954. Their factual accuracy remains unverified.
- **Density and purpose need separate checks.** Generic repetition, promotional copy, an unfilled
  issue template and unsupported product claims occur across score bands. Distinguish templates,
  opinion, reference material and checked solutions without rejecting entire nontechnical topics.

**Decision:** retain HQ as a qualification candidate with FineWeb-Edu as control; do not adopt a
stricter cutoff or freeze source weights from this packet. The source recovery and review-flag
comparison below completes the immediate follow-up. Source-family/near-duplicate exclusions and
eligibility remain necessary before counting usable tokens. A low exact-duplicate count alone
does not establish novel information.

Replay offline into a fresh external directory:

```bash
PYTHONPATH=. .venv/bin/python experiments/corpus-audit/audit_web_hq.py analyze \
  /mnt/speck-data/speck/data-qualification-20260919/web-inventory/hq-next-sample-plan.json \
  /mnt/speck-data/speck/data-qualification-20260919/web-hq-stratified \
  /external/fresh-hq-review
```

Offline replay reproduced identical sample and reading-packet bytes and all summary fields except
output paths. Ten focused tests cover corrupt/oversized/truncated downloads and score boundaries.
Raw corpus/review text stays outside Git; the receipt identifies each reviewed document and scope.

## Archived source recovery and frozen review flags

All three original Common Crawl records were recovered with **147,943 compressed range bytes**.
Exact URL, capture date and WARC record ID match the corpus metadata; WARC block and payload
digests also verify. The dataset's WARC filenames carry an extra eight-hex suffix, so the join
uses record identity rather than a filename guess. One CDX request returned HTTP 503; one retry
succeeded. Live pages were saved as supplementary context; the findings use the archived captures.

| Case | Finding against the matching capture | Consequence |
| --- | --- | --- |
| FlowWright translation tutorial | Four article images have no nonempty alt text; six of thirteen list items, covering inputs/returns, are also absent from corpus text | Restore missing text separately from graphical values; HTML re-extraction alone cannot resolve the image dependency |
| Fasting explainer | All fourteen article list items are absent, although surrounding prose remains | Confirmed extraction omission; preserving list structure is a concrete repair target |
| Contract-law study page | HTML separates related-guide teasers and upload UI; the corpus merges them into the document | Recover main-content boundaries; do not treat related material as part of the lesson |

No images were recovered/OCRed and no subject-matter claims were independently verified. Hold
these three serializations for repair/review, without inferring a whole-domain blacklist.

Three simple candidates were frozen before reading a new panel: visual-reference wording,
consecutive nonempty lines ending in colons, and upload-interface phrases. The
[replay script](audit_web_filters.py) keeps them separate from production diagnostics. Evaluation
uses **16 previously unreviewed HQ documents**, one per score/byte-length cell, and **16 fresh
FineWeb-Edu documents**, four per byte-length band. The control comes from a new seed over the
2,097,270-document retained stock, with its original manifest/index/text hashes verified. Exact
text overlap with earlier samples is excluded; all 32 token counts were recomputed with the
frozen Mistral tokenizer. Review covers sixteen full texts and sixteen head/tail excerpts plus
flag-match contexts, not full semantic inspection of every document.

| Source | Documents hypothetically retained if any candidate flag caused rejection | Sample tokens retained, including BOS/EOS | Token share |
| --- | ---: | ---: | ---: |
| HQ | 15 / 16 | 46,347 / 52,216 | 88.76% |
| FineWeb-Edu | 15 / 16 | 51,438 / 73,237 | 70.23% |

Only the colon flag triggers in this evaluation panel. In HQ it flags a heading followed by
a paragraph introducing an intact list: a false alarm for missing lists. In the control it finds
context gaps in a multi-post blog, but also consecutive introductions before intact quotations.
That does not establish a reliable missing-list rule or justify dropping the whole document.
The other two flags have **zero evaluation hits**, so their usefulness remains unestablished.
Each catches its corresponding development case; those three cases are not validation results.

Unflagged control excerpts also mix unrelated articles/catalog UI, while a full article introduces
a poem absent from its text. Existing diagnostics flag none of these 32 records. Thus passing
either flag set is not evidence of clean extraction. These small, length-balanced samples do not
estimate corpus defect rates, full-corpus token retention, classifier precision/recall or learning
quality. The complete 32 observations and artifact identities are in the
[receipt](web-filter-validation.json); raw text remains outside Git.

**Decision:** keep all candidates review-only, with no stricter HQ score cutoff. Prefer source-aware
list preservation and boundary repair, then measure useful-content retention on fresh material.
The bounded web diagnosis and subsequent [natural-code supply census](code-supply.json) are
complete; next qualify immutable code bundles and expansion yield. Web eligibility, near-duplicate
work and usable token counts remain launch gates, not a reason for an indefinite preview loop.

```bash
PYTHONPATH=. .venv/bin/python experiments/corpus-audit/audit_web_filters.py \
  /mnt/speck-data/speck/data-qualification-20260919/web-filter-validation/plan.json \
  /external/fresh-web-filter-comparison
```

Offline replay reproduces identical sample bytes and summary metrics. Negative checks reject
changed input hashes, a changed frozen script and control overlap before creating output.

## DCLM previews: useful fields, limited sampling scope

Both viewers have **partial indexes**. Their reported indexed row counts are not release totals.
Four seeded 12-row windows per source provide 48 untruncated records each, with revision headers
checked against the pins below. They support schema/content reconnaissance only. We did not
download or hash-verify the original Parquet payloads for these previews.

| Source | Revision prefix | Fields observed in all 48 rows | Sample Mistral tokens, including BOS/EOS |
| --- | --- | --- | ---: |
| DCLM baseline Parquet | `817d6752765f` | Text, URL, document ID, language, language score, fastText score | 53,468 |
| DCLM-Edu | `dbad8ad71224` | The same fields plus integer and continuous educational scores | 74,226 |

The baseline preview has 45 distinct hostnames; Edu has 48. No repeated document IDs or exact
text duplicates appear within either preview. No parent/child ID match or exact match to the
earlier 176 web samples appears in these small sets. **This does not demonstrate independent
source families or disjoint supply.** DCLM-Edu is a filtered derivative of DCLM; shared parent
content must be counted once. Original URLs and IDs improve tracing, but these fields do not
provide per-page license clearance or prove a completed join to an immutable upstream capture.

The [pinned Edu card](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/blob/dbad8ad71224482740cd9c9d353591adbf62fe04/README.md)
recommends the explicit integer predicate `edu_int_score >= 3` for its stricter candidate and
retains score-2 material for diversity. In our 48-row preview, that predicate keeps **16** rows;
`edu_score >= 3` keeps only **six**. These are different rules. Neither preview retention rate
estimates the full release. Preserve both original fields; do not substitute one for the other
or reuse UltraFineWeb thresholds for DCLM's differently defined scores.

The same card reports improved knowledge/reasoning results for its 360M experiment, but says
mid-training gains were inconsistent at 1.7B. That supports a model-specific comparison; it does
not establish the best source for our 1.2B model. Its performance-section wording and executable
filter example also differ at the boundary, so our preparation must name an exact predicate.

Both cards label their datasets CC-BY-4.0. The
[baseline card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/blob/817d6752765f6a41261085171dd546b104f60626/README.md)
describes a research-oriented intended use. Record both statements for source-use and release
suitability review; this inspection makes no legal determination or automatic admission.

## Content observations and next step

Seven deterministically selected examples were reviewed under the existing rubric: four full
texts and three head/tail excerpts. The receipt records exact identities and limitations.

- Baseline includes coherent historical news and substantive chemistry research, alongside
  repetitive product-promotion prose. A low fastText score alone is not a demonstrated rejection rule.
- Edu includes a long philosophical discussion at integer score 2. Preserve potentially useful
  broad-domain material when testing stricter filters; do not equate educational score with depth.
- A short API explanation contains a JSON fragment with missing braces. Other examples have
  mixed article/teaser boundaries or an unresolved opening reference. Source recovery and extraction
  checks remain necessary even for educationally selected material.

These are assistant inspection observations, not independent factual checks or source quality
rates. Both cards were checked against pinned Git blob IDs. Offline replay reproduced exact
sample/review bytes, metadata statistics and all 127,694 sample tokens.

The HQ sample and source-recovery comparison above complete the bounded web diagnosis. Prepare
a comparable source-file-based DCLM packet only for a concrete remaining coverage/eligibility
question after source-use review. Exact bounded DCLM
file routes are pinned externally: approximately 224 MB for baseline and 2.91 GB for Edu
(3,129,916,683 bytes combined), before decoding overhead. Those selected files are convenience
samples, not a complete DCLM inventory or representative comparison design. No bulk acquisition
or GPU experiment follows automatically from these routes. Keep FineWeb-Edu as control and
freeze source weights only after source-family exclusions and usable token counts are established.
