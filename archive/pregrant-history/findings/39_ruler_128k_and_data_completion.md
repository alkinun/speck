# 39 — RULERv1 128K and all-length data qualification

## Result

The final 128K stage passes the frozen case-generation gate. All 13 task files reproduce byte for byte
across two complete generations; all 1,300 retained rows pass schema and full context accounting; and
the application-layer guard records zero network attempts. The case-stream identity is
`f14b9372cd2bb15e3254d4f3bccd0ad56b7fb9b53754e520854df8e852b668f8`.

The largest accounted cases reach exactly 131,072 tokens. HotpotQA reaches 130,967 and completes under
the same hashed control-flow repair. The retained 128K first pass and logs occupy 564MB. Across 4K,
8K, 16K, 32K, 64K, and 128K, the local case cache occupies 1.1GB.

## Aggregate qualification

All 78 declared RULER task/length cells are now case-qualified: 13 tasks, six lengths, and 100 retained
cases per cell, for 7,800 retained cases. Qualification actually generated 15,600 cases because every
length was built twice and compared by full file hash. Every length passed:

- exact row count and exported schema;
- task length plus the five-token base placeholder and 50-token NeMo reserve at or below its ceiling;
- byte-identical task hashes across two complete generations;
- source, package, generator, patch, scorer, tokenizer, and lockfile identity checks; and
- a positive network-denial self-test followed by zero generator network attempts.

The mixed-rights case text and logs remain outside Git. Checked reports retain all identities, hashes,
counts, and extrema.

## Decision

RULER source and case-data preparation is complete. This closes the data-generation scope of SPE-99,
but it is not a RULER model result. External execution readiness remains false: every frozen candidate
needs an attested export at the evaluated context ceiling, then all 78 cells must be executed and scored
through the pinned NeMo-Skills adapter/scorer. No architecture or release claim changes from this data
qualification alone.

## Artifact

- [128K case qualification report](../results/Speck-Architecture-Promotion-v1/ruler-cases-131072-qualified.json)
