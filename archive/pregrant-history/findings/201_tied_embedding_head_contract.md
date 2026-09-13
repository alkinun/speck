# 201 — The flagship keeps physically tied token embeddings

Exact scale accounting exposed a contradiction: planning text said the input embedding and LM head
were untied, while every native Speck model and Transformers export physically shared one parameter.
The launch default is now frozen to the inherited tied implementation rather than adding an
unsupported experiment axis.

Missing and explicit `tie_word_embeddings=true` configs normalize identically; explicit false is
rejected. Native checkpoints retain two equal aliases backed by one parameter, contradictory aliases
fail strict loading, Transformers exports store one tensor and restore physical sharing, and the
shared tensor has one AdamW no-decay optimizer membership.

The decision avoids 16.39M parameters at 60M and 65.54M at 1.2B relative to the previously described
untied counterfactual. Exact target parameters and analytic projection FLOPs do not change because the
implementation was already tied. Tokenizer v3 and scale accounting v2 now price physical vocabulary
cost as `V*E`; all predecessor plans and accounting remain preserved.

This is a compatibility/accounting decision, not a GPU quality result. D5 and every launch authority
remain open.

Artifacts: [embedding/head contract](../research/flagship/embedding_head_contract_v1.json),
[tokenizer v3](../research/flagship/tokenizer_plan_v3.json), and
[scale accounting v2](../research/flagship/targets/accounting-v2.json).
