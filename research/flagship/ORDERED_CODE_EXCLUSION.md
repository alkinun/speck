# Ordered Stack-Edu acquisition to excluded stock

The finite ordered worker preserves source order and separately archived units. Its result is
acquisition evidence, not a fully excluded stock. `scripts.prepare_ordered_stack_edu_stock` connects
a complete result to the existing source-stock exclusion pipeline without fetching content again.

An explicit preparation input must use format `speck_ordered_stack_edu_exclusion_preparation`, version
1, source `stack_edu`, a hash-bound `ordered_result`, a new `output_directory`, and
`training_authority: false`. No actual input can be bound until acquisition publishes its completed
result. A partial or pre-exclusion language-shortfall result is rejected. The handoff inherits the
original source-use, metadata, tokenizer, filtering, reference exclusion and SQLite policy; final
per-language targets come from that finite acquisition's own contract. The older larger stock target
cannot silently replace those targets. Neither target becomes a new selected-base recipe.

The adapter checks unit order and non-overlapping physical row ranges, original metadata/configuration
and payload identities, both predecessor/successor locations, complete language coverage and counts.
The existing importer copies only the output, security report, configuration and manifest into a new
work directory. Original fetch journals, unscanned content, raw cache, failures and archives stay at
their hash-bound owners; the copied subset is explicitly labeled and cannot stand in for the full
acquisition history. The standard grouping/exclusion/counting path then measures actual usable tokens,
including every language's final headroom. A failed capacity gate remains a measured shortfall.

Ten focused tests check a real fixture handoff through the existing grouping function, exact byte
order, unchanged original files, no new network fetch, finite target inheritance, rejection of
incomplete/changed/reordered/duplicate inputs, and protected output locations. They do not establish
real full-corpus completion or throughput. The new adapter needs a frozen runtime revision, an actual
completed result, an explicit new plan and a storage preflight before execution.

Wait for the existing FineWeb exclusion queue before starting another large exclusion pass on the
same disk. Consider available NVMe workspace for the next SQLite pass after accounting for preserved
code units, copied/grouped text, reference restoration, candidate output, database/WAL and the real
free-space floor. A faster-storage run would be an operating measurement, not a controlled speedup
against a different source or workload. Tokenization and complete cache verification follow successful
text qualification; joint eligibility and scientific launch remain separate.
