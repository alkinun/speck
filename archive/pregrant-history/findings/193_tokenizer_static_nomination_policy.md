# 193 — Static tokenizer finalists have a frozen pre-results rule

If all three custom tokenizers pass hard gates, static evaluation may leave all three on the
compression/parameter Pareto frontier. The rule is now frozen before real outputs: nominate the
lowest-tokens/KiB custom candidate as the compression endpoint, then the lowest-parameter remaining
Pareto candidate as the compact endpoint. Ties prefer the other objective, then smaller vocabulary,
then candidate ID. A hard-gate failure or domination removes a candidate; fewer than two valid
distinct Pareto candidates stops the process rather than inviting substitution.

The generated fixture places all three custom sizes on the frontier and therefore reports 49,152 as
the compression endpoint and 32,768 as the compact endpoint. This is only a plumbing demonstration:
`advancement_authority=false`, because custom tokenizers saw the generated distribution and Mistral
did not. The real report will apply the same policy to the authorized 600/60 MB sample.

Static nomination never selects D5. The matched 60M pilot and one-opening tokenizer audit decide;
Mistral remains fallback.

Artifacts: [checked fixture](../results/data/tokenizer-static-nomination-fixture-20260907.json) and
[frozen policy](../research/flagship/tokenizer_static_nomination_policy.json).
