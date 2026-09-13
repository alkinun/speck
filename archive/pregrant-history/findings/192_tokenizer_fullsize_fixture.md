# 192 — All tokenizer sizes and Mistral qualify on generated fixtures

The exact Mistral-7B-v0.1 tokenizer is pinned at revision `27d67f1`, SHA-256 `dadfd56d…`, 493,443
bytes, 32,000 pieces, and special IDs `<unk>/BOS/EOS = 0/1/2`. Separately, a blocked input manifest now
binds all 30 technically qualified real successor files and their exact 100/10 MB per-category quotas.
Every runtime file and parent report revalidates, but the manifest cannot emit or act as an executable
tokenizer experiment while rights, production operations, and real firewall partitions remain open.

The first generated fixture intentionally failed rather than shrinking the requested vocabulary: its
repetitive text supported at most 621 pieces, so hard 32,768 training was rejected. A committed
generator then produced richer deterministic alphabetic text across all six categories. It yielded
18.001 MB train and 1.801 MB evaluation fixture text.

All requested candidates train at their exact sizes: 32,768, 40,960, and 49,152. A second independent
fixture directory produces identical category sample hashes and identical model hashes for every
candidate. All candidates and Mistral pass uint16-with-three-chat-tokens, zero-unknown, and exact probe
roundtrip gates. Shape-A embedding/head cost rises from 131.08M parameters for Mistral to 134.23M,
167.78M, and 201.34M for the three custom sizes.

The custom models use 348.40, 343.55, and 340.48 tokens/KiB versus Mistral's 494.76 on this fixture.
That is expected fixture overfitting: the custom models were trained on deliberately random alphabetic
fixture words while Mistral was not. These numbers have no scientific selection authority, advance no
candidate, and do not predict the real comparison.

Meaningful selection still requires the rights-authorized 600/60 MB sample, static evaluation on its
held-out split, the matched 60M LM pilot, and the one-opening D5 audit. No real candidate training was
performed or authorized.

Artifacts: [checked fixture result](../results/data/tokenizer-fullsize-fixture-20260907.json),
[blocked real inputs](../research/flagship/tokenizer_inputs_blocked_v1.json), and
[Mistral baseline](../research/flagship/mistral_tokenizer_baseline.json).
