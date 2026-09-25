# Corpus audits

Completed audits of candidate sources. Each receipt holds its own detail and keeps its original
bytes. Per-source gates and blockers are in
[source-readiness.json](../main-data/source-readiness.json), which cites these receipts, and the
open qualification work is in the [main-data record](../main-data/README.md#qualification).
Superseded audits and their replay scripts are in Git history (tag `pre-cleanup-2026-09-25`).

| Area | Receipts |
| --- | --- |
| Retained stock | [result.json](result.json) (first quality audit of the pilot stock), [followup.json](followup.json), [data-readiness.json](data-readiness.json) (bank token/overlap closeout, assistant format census) |
| Selected web | [ultrafineweb-hq-listing.json](ultrafineweb-hq-listing.json) (HQ files per crawl), [web-variants.json](web-variants.json), [natural-web-inspection.json](natural-web-inspection.json), [web-hq-stratified.json](web-hq-stratified.json), [web-filter-validation.json](web-filter-validation.json) |
| Math | [finemath-config-sample.json](finemath-config-sample.json), [infiwebmath-4plus-arithmetic.json](infiwebmath-4plus-arithmetic.json) and its [review](infiwebmath-4plus-arithmetic-review.json), [openmathinstruct2-sample.json](openmathinstruct2-sample.json) |
| Stack-Edu | [stack-edu-metadata-census.json](stack-edu-metadata-census.json) (all 42 files), [stack-edu-acquisition.json](stack-edu-acquisition.json) (`int_score` 4+ and fetched tranches), [stack-edu-yield-probe.json](stack-edu-yield-probe.json) (`int_score` 3 projection) |
| Retained code | [code-supply.json](code-supply.json), [code-yield-result.json](code-yield-result.json), [natural-code-cohort.json](natural-code-cohort.json), [code-provenance.json](code-provenance.json) |
| Stack v3 | [stack-v3-feasibility.json](stack-v3-feasibility.json), [stack-v3-sampling.json](stack-v3-sampling.json), [stack-v3-broader.json](stack-v3-broader.json), [stack-v3-origins.json](stack-v3-origins.json) |
| Code origins and notices | [retained-code-origins.json](retained-code-origins.json), [code-origin-recovery-20260923.json](code-origin-recovery-20260923.json); pinned by the [code rights review](../main-data/code-rights-review.json): [code-application-origins.json](code-application-origins.json), [code-family-provenance.json](code-family-provenance.json), [code-notice-provenance.json](code-notice-provenance.json), [code-provenance-inputs.json](code-provenance-inputs.json) |
| Assistant stock | [recipe-review.json](recipe-review.json) |

Rules carried from these audits into the pipeline are in [Data](../../docs/data.md#source-quality).
