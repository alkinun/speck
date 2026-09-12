# 216 — MiniCPM5-2B strengthens the data-stage and release thesis, not the mechanism backlog

MiniCPM5-2B is a credible new release-era comparator: Apache-2.0, standard Llama runtime, 2.517B total
parameters, 1.982B non-embedding parameters, aggressive 16:2 GQA, and 128K configured context. Its
independent Artificial Analysis result confirms unusual agentic strength below 4B, while also exposing
knowledge, terminal-coding, long-context, abstention, and output-token trade-offs hidden by the vendor's
53.9 aggregate.

The most relevant controlled result is data scheduling. OpenBMB reports +1.49 aggregate points from
sequential L1→L2→L3 stages versus mixing the same 120B tokens and domain proportions in a 1.2B model,
with late reasoning/math/code gains but small commonsense regressions. This supports the already-frozen
Speck stable/decay allocation and category guardrails; it does not require another grant-1 arm.

The final model is also evidence that post-training and release engineering can dominate perception of
a small model. OpenBMB publishes Base, Midtrain, SFT, and final RL+OPD checkpoints, attributes large
gains to specialist RL plus 16-expert OPD, and ships mainstream and edge formats. Speck should preserve
stage checkpoints, evaluate agent/tool behavior and output-token cost, and make R15 export parity real.
It should not import OPD, DSpark, specialist RL, or a standard dense architecture into the fixed grant.

Finally, MiniCPM's 130,560-row untied vocabulary consumes exactly 534,773,760 parameters—21.25% of the
model. That strengthens, rather than weakens, Speck's tied-head D5 accounting and total-versus-
non-embedding disclosure.

Literature record: [MiniCPM5-2B release audit](../papers/49_minicpm5_2b_release.md). Comparator identity:
[`minicpm5_comparator_v1.json`](../research/flagship/minicpm5_comparator_v1.json).
