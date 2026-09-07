# 178 — Web rights evidence is ready for human decision

## Evidence collected

The exact-revision cards for Ultra-FineWeb, FineWeb-Edu, DCLM baseline, and FineWeb base are preserved
with hashes, alongside the Ultra-FineWeb Apache-2.0 text, ODC-By 1.0, CC-BY 4.0, and the Common Crawl
terms retrieved on 2026-09-07. This is an operational evidence packet, not legal advice or automated
acceptance.

The source-level labels do not close the whole chain. Ultra-FineWeb's pinned card says its English
data derives from FineWeb and explicitly directs users to each upstream license. ODC-By grants rights
over a database and expressly does not govern each item of content. DCLM declares CC-BY-4.0, but its
card also repeatedly describes research-only intended use and says it is not intended for
production-ready models. All four sources substantially derive from Common Crawl.

## Decision boundary

Common Crawl's current terms make use a binding acceptance event, leave page-owner terms and rights
with the user, disclaim lawfulness and non-infringement, and include an indemnification clause that
expressly names developing, training, or deploying AI systems. The terms also include liability,
arbitration, governing-law, and future-modification provisions. A named human authority must decide
whether those terms and the residual page-content risk are acceptable for the actual research,
publication, model-release, and any commercial scope.

The packet proposes concrete controls: immutable per-record lineage, dataset-level notices and
citations, a named removal contact, a URL/hash/domain deny ledger, rebuildable shards, term-change and
takedown tracking, and a separate no-assumption decision about corpus redistribution. Acceptance must
record each approved source, evidence hashes, scope, attribution placement, redistribution policy,
removal owner, date, and accountable approver. Until then, all four web successors remain blocked.

Artifact:

- [Human review packet](../results/data/web-rights-review-20260907.json)
