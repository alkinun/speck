# 211 — Formal tokenizer static evaluation advances the 40,960 and exact-32K endpoints

The R4 production partitions are adopted without repartitioning: 600,078,199 training bytes and
60,035,301 evaluation bytes across six equal categories. Deterministic one-thread SentencePiece
training produces exact 32,000, 32,768, and 40,960-piece custom models; the pinned Mistral baseline
matches its declared revision, 32,000-piece size, and SHA-256.

All four tokenizers pass uint16-with-chat, zero-unknown, and exact probe-roundtrip gates. Macro static
tokens/KiB are 279.637 for Mistral, 262.728 for custom 32K, 261.896 for 32,768, and 255.305 for 40,960.
Under the frozen two-objective policy, all three custom models remain Pareto-efficient. The 40,960
model advances as the compression endpoint and exact-32K advances as the compact endpoint.

This is nomination, not the D5 decision. The matched 60M pilot must select one custom finalist and
produce a provisional ranking against Mistral before the sealed D5 audit can be opened once. The audit
remains unopened. The SentencePiece TSV writer warns about newline-containing pieces; binary model
files are authoritative and all exact roundtrip gates pass.

Artifact: [formal static tokenizer result](../results/data/formal-tokenizer-static-20260910.json).
