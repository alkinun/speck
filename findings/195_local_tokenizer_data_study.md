# 195 — Whitespace-piece support fixes the local custom-tokenizer deficit

A bounded CPU-only study trained custom tokenizers on 60.38 MB of balanced six-category text and
evaluated every treatment on the same 6.61 MB held-out payload. The initial custom tokenizers were
misleadingly weak: the 32,768-piece BPE scored 292.97 equal-category tokens/KiB against Mistral's
285.51, with large regressions on raw code and markup-heavy FineWiki text.

The deficit was not fixed by tripling balanced data, paragraph chunking, doubling the unique code
share at the same total bytes, matching Mistral's dummy prefix, or switching to unigram. The isolated
cause was SentencePiece's `allow_whitespace_only_pieces`. Mistral enables it; the original Speck
trainer did not. The original custom vocabulary learned no ordinary multi-space pieces and spent
262,332 held-out token occurrences representing whitespace. The corrected 32,000-piece model learned
15 multi-space pieces and used 83,493 whitespace-piece occurrences while preserving exact roundtrip.

At exactly 32,000 pieces, the corrected custom BPE scores **264.88 tokens/KiB**, 7.23% below Mistral,
with identical 131.08M Shape-A embedding/head parameters. It improves all six categories and all 30
sources. Its category deltas range from -9.41 tokens/KiB on web to -35.40 on code. A paired
document-bootstrap diagnostic remains entirely favorable. After collapsing all whitespace, the
custom model still leads by 8.23 tokens/KiB, so domain fit and much lower rare-byte fallback also
contribute; the result is not only indentation compression.

The corrected 32,768-piece model reproduced with identical training-stream and model hashes. Moving
to 40,960 pieces improves another 1.57% but adds 33.55M embedding/head parameters, making exact 32K
the attractive operating point for a later matched-LM test.

This is static, local, hypothesis-generating evidence. It does not select D5, authorize flagship
training, replace the formal sample, justify opening `D5_tokenizer`, or establish language-model
quality. The formal tokenizer plan now has a [versioned v2 successor](197_tokenizer_v2_contract.md)
that carries exact-32K BPE with explicit whitespace-only-piece support alongside Mistral; the
original setting remains the failed control.

Artifact: [checked study result](../results/data/tokenizer-local-study-20260907.json).
