# E1/E3 recipes: discussion requested before freeze

## Decisions

The project owner chose **discuss recipe changes** when offered a concrete initial recipe set.
No recipe freeze follows from that exchange. The independent SQLite-policy binding work can proceed.

In the subsequent discussion, the owner requested review of MiniCPM4 and Ultra-FineWeb,
UltraData-Code, UltraData-Math, and Ultra-FineWeb-L3 before a recipe freeze. The
[paper and corpus review](../literature/50_minicpm4_ultradata.md) now records the evidence and concrete
schema questions. New code/math tiers remain candidates for a budgeted revision, not approved inputs.

## Context

The proposed set was category substitutions at the fixed 55/15/10/10/5/5 prior, with FineWeb-Edu,
Stack-Edu, FineMath-4+, Cosmopedia v2, peS2o v3, and FineWiki as the E3/E1-background incumbent.
The proposed web arms were Ultra-FineWeb v1.4, FineWeb-Edu, DCLM, and a 35/30/25/10 blend of those
three plus FineWeb base. Proposed specialist blends were 50/50 between their two primary candidates.
These are discussion inputs, not selected execution recipes.

## Work performed

Reviewed the data protocol, source survey, registry roles, current source-use record, and conditional
capacity analysis to identify which proposed defaults still need scientific justification.

## Evidence and links

- [Data protocol](../flagship/DATA.md)
- [Source survey](../literature/surveys/sources.md)
- [Source registry](../flagship/source_registry_v2.json)
- [Current source-use record](../flagship/source_rights_acceptance_v1.json)
- [Capacity review](../flagship/SCREEN_CAPACITY.md)

## Open questions

1. **E1's question.** Fixed-background category substitution directly measures the effect of replacing
   one component of a complete training mixture. Pure-category training answers a different question.
   Decide this before calculating final per-arm source quotas.
2. **E3's incumbent.** One fixed mixture must support the repetition study within its current budget.
   An education-heavy incumbent and a broader prospective production blend can have different
   repetition behavior. Choose for relevance to the research question, rather than availability of
   the existing operations sample. peS2o remains the documented primary science source.
3. **Web blend.** The source survey proposes the three primary web sources plus a small FineWeb-base
   diversity component, but the 35/30/25/10 numbers were tokenizer-sampling proportions. They are not
   an established LM-training optimum. An equal-primary-source control or a deliberately production-
   weighted blend needs its own stated rationale.
4. **Specialist blends.** A two-source 50/50 mix is a simple pre-results control. The older code survey
   proposed 60/30/10 with code prose, while PEP material is marked tokenizer-only in the active source
   registry. Resolve that scope mismatch explicitly before putting code prose into a training arm.
5. **Source coverage.** L1-HQ appears in the registry/survey but is absent from the current approved-source
   list. The operations bank's Common Pile code/science sources do not silently become scientific
   incumbents. Capacity must be checked for whichever recipes are actually selected.

## Next actions

Discuss the comparison question, E3 incumbent, and blend rationale before freezing recipes. Maintain
the four web/three-per-specialist funnel, fixed category weights for the chosen comparison,
replication requirements, token horizons, and compute ceilings. Recipe choices remain open in the
maintained status record until the discussion resolves them.
