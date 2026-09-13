# 119 — HELMET archive path inventory qualified

## Reproduced inspection

Two complete streaming passes over the 11.27 GB gzip archive produce the same member identity
`23d404071444d52686c917ab9827b604594f2faa23502d1e9b45fa9c558bbd10`. All 175 members are safe
regular files/directories: no absolute path, traversal, link, or device member is accepted. The archive
contains 154 files totaling 35,631,220,823 uncompressed bytes.

Every one of the 52 config-declared archive-local paths is present. The independent runtime audit is
confirmed: 50 configured entries still load external datasets rather than archive files.

## Rights boundary

The archive contains zero small license, terms, notice, citation, or README metadata files. Therefore
archive completion cannot establish component dataset rights. Extraction remains false until rights are
resolved from authoritative external sources; runtime-loaded data, the gated tokenizer, proprietary
judges, and exact candidate evaluation remain separate blockers.

This safe inventory does not affect finalist training authority beyond confirming that the acquisition
service has ended. No payload was extracted.

## Artifact

- [HELMET archive inspection](../results/Speck-Architecture-Promotion-v1/helmet-archive-inspection.json)
