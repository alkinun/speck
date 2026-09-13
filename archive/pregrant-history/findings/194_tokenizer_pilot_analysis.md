# 194 — The seven-run tokenizer pilot analysis is frozen

The matched 60M pilot now has a complete pre-results analyzer. Seed 42 screens Mistral plus the
compression and compact static endpoints. It selects one custom by fixed-document equal-category BPB,
active time, vocabulary size, then ID. Confirmation reuses seed 42 and adds seeds 43/44 for Mistral
and that custom, totaling seven runs.

Fixed-document evaluation uses the identical raw stream ending at 1.2B Mistral tokens. Fixed-FLOP
evaluation uses Mistral seed 42's analytic endpoint for every tokenizer. A custom must keep the upper
paired 95% bound within +0.01 macro BPB and +0.02 in every category in both views. Document IDs and
UTF-8 byte counts must pair exactly. Ten thousand deterministic bootstrap replicates average each
document's delta across seeds before resampling. Fixed wall-clock is reported secondarily and is never
selected as a stopping rule.

Compute-to-quality uses the Mistral seed-42 fixed-document final macro BPB, defined without inspecting
custom curves. The candidate must reach it in all seeds; median analytic FLOPs rank it against
Mistral, with smaller vocabulary breaking an exact tie. Parameters, throughput, peak memory, active
seconds, per-category BPB, and seed dispersion are retained.

Fixture tests cover a passing faster custom, category-regression fallback, mismatched documents,
incomplete run matrix, systems views, and the D5 handoff. The analyzer only drafts a two-finalist
`D5_tokenizer` opening request; it cannot open the audit or select D5. No real LM run or audit opening
occurred.

Artifacts: [checked result](../results/data/tokenizer-pilot-analysis-fixture-20260907.json) and
[pilot plan](../research/flagship/tokenizer_pilot_plan.json).
