# 197 — The corrected tokenizer candidate is now a formal v2 contract

The local whitespace-piece finding is promoted into a versioned pre-results contract without
modifying the original plan, policy, or fixture results. Tokenizer v2 compares Mistral 32K with
custom BPE vocabularies of exactly 32,000, 32,768, and 40,960 pieces. Every custom candidate explicitly
sets SentencePiece `allow_whitespace_only_pieces=true`. The 49,152-piece v1 candidate remains in its
historical evidence but is excluded from v2 because its parameter cost is no longer an informative
formal treatment.

The executable tokenizer pipeline accepts the new setting as an optional, hash-bound field. Old
configs that omit it remain byte-for-byte normalized as before, so their plan fingerprints and
historical artifacts are not reinterpreted. Fixture training verifies that the explicit setting
reaches SentencePiece, preserves IDs 0/1/2 and exact spaces/newlines/tabs, and exports identical model
bytes with hash-bound chat metadata.

The v2 static policy retains the two Pareto endpoints. Applying it to translated local metrics reports
40,960 as the compression endpoint and exact 32,000 as the compact endpoint; 32,768 remains visible on
the frontier. That application has no advancement authority and does not select D5.

The active v2 contract still requires the authorized 600/60 MB sample, real candidate training,
static qualification, seven matched 60M LM runs, and the single `D5_tokenizer` opening. Mistral remains
the fallback throughout.

Artifacts: [v2 qualification](../results/data/tokenizer-v2-contract-20260908.json),
[v2 plan](../research/flagship/tokenizer_plan_v2.json), and
[v2 nomination policy](../research/flagship/tokenizer_static_nomination_policy_v2.json).
