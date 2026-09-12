# 214 — The tokenizer pilot now has one fixed 1.200007B-token document stream

The passed firewall-excluded corpus yields one immutable whole-document stream with 1,106,243
documents and 1,200,007,273 Mistral tokens. The 7,273-token overshoot comes only from retaining the
terminal document in each frozen 55/15/10/10/5/5 category quota. It is now fixed before model outputs.

Exactly the same ordered document identities are packed under all three pilot tokenizers. The 40,960
candidate uses 1,108,037,579 tokens (7.66% fewer than Mistral), and exact-32K uses 1,142,621,116
(4.78% fewer). Every source, tokenizer, index, and packed-shard hash reverifies.

This closes stream construction, not the pilot. Exact 60M model, optimizer, schedule, evaluation, and
seven-run manifests remain required. D5 is unopened and no tokenizer or model-training authority is
issued.

Artifact: [fixed-stream result](../results/data/tokenizer-pilot-stream-20260911.json).
