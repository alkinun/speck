# Web inventory and DCLM comparison preparation

2026-09-19. The complete pinned UltraFineWeb HQ file inventory is now available, and DCLM
baseline/Edu have separate schema and content previews. The [receipt](web-inventory-dclm.json)
binds metadata, external samples, analysis scripts, review notes and the next acquisition plan.
This extends the [initial web inspection](NATURAL_WEB.md); no training data is admitted.

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

The next bounded acquisition is pinned: two deterministic hash-ranked files per crawl drawn
from **all** 1,000 files. These 12 files total **959,301,706 bytes**, under a 1.2 GB download cap;
none duplicates the retained six. They have not been acquired. Within that next sample, census
and review should distinguish four classifier bands, four byte-length bands and six crawls.
Select two records per nonempty cell, at most 192, and report host concentration separately.
Any supplementary host-balanced review must retain its different sampling interpretation.
No full-release download is needed to perform this next check.

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

The next work is the 959 MB HQ acquisition and score/crawl/length review, followed by a comparable
source-file-based DCLM packet if its source-use review supports proceeding. Exact bounded DCLM
file routes are pinned externally: approximately 224 MB for baseline and 2.91 GB for Edu
(3,129,916,683 bytes combined), before decoding overhead. Those selected files are convenience
samples, not a complete DCLM inventory or representative comparison design. No bulk acquisition
or GPU experiment follows automatically from these routes. Keep FineWeb-Edu as control and
freeze source weights only after source-family exclusions and usable token counts are established.
