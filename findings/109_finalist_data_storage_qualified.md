# 109 — Finalist materialization, data, and storage qualification

## Qualified gates

All 84 generated configs revalidate against their manifest, and the twelve resolved experiments have
unique checkpoint targets under the frozen root. That checkpoint root, the result root, target lock,
and final analysis path all remain absent.

The two 1,539,833,856-token data windows are `[0, 1,539,833,856)` and
`[1,610,612,736, 3,150,446,592)`, leaving a 70,778,880-token gap. Each window is crossed with seeds
42/43/44. At the start, three quartiles, and endpoint of both windows, direct-offset and resumed loaders
produce identical cursor states, input bytes, and target bytes: ten exact replay points under the pinned
packed-data manifest.

The dedicated ext4 device UUID remains correct with `rw,nosuid,nodev,noexec`; its user-owned parent is
mode 0700. It has 5,643,926,896,640 bytes free versus the 25,769,803,776-byte finalist floor. No artifact
was moved or deleted and cleanup contributes zero capacity.

## Remaining blockers

This qualifies materialization, output absence, data windows, and storage only. The exact finalized
configs still need a CUDA/BF16 runtime preflight and the finalist collector/analysis implementation must
be tested against the frozen df=5 contract. RULER v2, NoLiMa, and HELMET remain unqualified or
unexecuted. Training and automatic launch remain false.

## Artifact

- [Finalist qualification v1](../results/Speck-Paper1/finalist-qualification-v1.json)
