# Stack-Edu E1S ordered acquisition

The [explicit acquisition successor](../flagship/stack_edu_e1s_ordered_acquisition_v1.json)
binds the selected 300M nominal / 360M headroom Stack-Edu treatment to all 28 complete metadata
files and their 1,536,206 original-order eligible rows. It preserves the selected eleven-language
shares, original source-use lineage and source-specific metadata/content/security policies.
The original 1.44B stock worker remains paused, with its inputs and cache retained.

Java and TypeScript run first because the completed feasibility analysis identified them as the
scarcest languages. Within each language, acquisition preserves metadata-file and row order.
Candidate whole-document prefixes target twice the nominal language quota before Gitleaks and
full exclusion. Available metadata is exhausted if necessary. If a completed language cannot
meet its 20% headroom even before full exclusion, the worker records that shortfall and stops
subsequent languages; it does not reweight, borrow another language's surplus or repeat data.

The producer uses the qualified 32-worker / 64-window ordered fetcher with fixed units of 512
eligible records. The sample wrapper's repeated whole-cache accounting scan is replaced by a
single-owner cache that scans once at startup, reserves worst-case retry space and charges
actual retained attempts. The new cache is limited to 12 GiB, its working tree to 24 GiB, and
free NVMe space to at least 64 GiB. Four preserved caches are verified read-only fallbacks.
Storage limits fail explicitly while retaining completed and interrupted work.

Each completed unit retains its original metadata, source row, eligible ordinal, content hash,
license/repository/path/encoding fields, fetch receipts, content-filter outcomes and Gitleaks
report. The complete fixed fetch batch is preserved even if a whole-document prefix stops early.
A sequential tar containing the unit and its referenced blobs is written to the data drive and
all payload hashes are reopened before the next batch. Earlier interrupted attempts remain intact.
The full producer requires a clean frozen checkout, identical plan/revision on resume and an
exclusive working-directory lock. No full exclusion pass competes with the active FineWeb job.

Validation: **1,196 tests passed**, 10 skipped and 125 deselected, plus formatting, lint,
research catalog, archive and source-pin checks. New tests cover deterministic prefixes and
prefetched tails, interrupted attempts with incomplete journal tails, byte-identical replay,
existing category-group handoff, changed configuration rejection, archive corruption,
read-only fallback reuse, reservation limits, failed-attempt accounting, 404 treatment,
source-index order and stopping dependent languages after a shortfall.

This is preparation under delegation. Final source capacity requires complete combined
reference/candidate exclusion and frozen-tokenizer recounts per language; joint experimental
eligibility, training manifests and model-launch qualification remain separate work.

The [first real unit qualification](../../results/systems/stack-edu-e1s-ordered-qualification-20260915.json)
passed after all 28 metadata files and eligible indices were verified. The first 512 eligible
Java rows retained **300 documents / 171,342 Mistral tokens / 516,715 UTF-8 bytes**, with the
original fetch receipts, text, attribution and positions independently reopened. Both unscanned
and post-Gitleaks token totals were recomputed from original SWH bytes. Every archived payload
was independently checked, and reopening the completed unit required no new fetch. The checks
reuse the actual production prefix; these counts are included in the future acquisition result.

The [finite background service](../../results/systems/stack-edu-e1s-ordered-launch-20260915.json)
was launched from frozen revision `3b35907` with the original absolute plan and `--resume`.
It reuses the qualified first unit and rechecks inputs before advancing. It survives assistant
shutdown/logout through the existing user service configuration; computer shutdown still
interrupts it. No final source result or FineWeb completion is claimed by this launch record.
