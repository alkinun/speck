# 204 — Model release and guarded source-use policy are frozen

On 2026-09-09 the SpeckLabs project owner selected MIT for repository source code and Apache-2.0 for
the first flagship model weights. Training data remains internal: neither source text nor derived
packed shards will be redistributed. The paper, model card, and checked manifests instead publish
source names, revisions, filters, mixture statistics, citations, attribution, and removal policy.

All 30 technically qualified tokenizer sources are approved for guarded internal training, paper
publication, public model-weight release, and commercial-capable downstream use. The approval accepts
the documented Common Crawl/underlying-page, code-license metadata, generator/seed lineage,
paper-license metadata, attribution, share-alike, and jurisdiction residual risks. It requires source
metadata retention, citations/notices, deny ledgers, a public removal contact before release, rebuilds
after verified corrections, and a new human decision if explicit prohibitive terms or material terms
changes are discovered.

This is a project risk and release decision, not legal advice or a warranty that dataset-level terms
clear every underlying record. It closes the human source-use gate only. The real 20B operations
rehearsal, production firewall, tokenizer decision, packed corpora, and data-launch receipt remain
mandatory and unexecuted.

Artifacts:

- [Release and data-use policy](../research/flagship/release_and_data_use_policy_v1.json)
- [Source registry successor](../research/flagship/source_registry_v2.json)
- [Human acceptance record](../research/flagship/source_rights_acceptance_v1.json)
- [Data rehearsal successor](../research/flagship/data_rehearsal_plan_v2.json)
- [Data launch successor](../research/flagship/data_launch_plan_v2.json)
