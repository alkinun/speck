# 208 — The 2B production-data path passes; 150B fits and 500B does not yet

The rights-bound six-category calibration completes every declared gate over 2.000008B packed tokens.
It downloads 12.287 GB, emits 9.784 GB filtered text, examines 2,316,094 records, retains 2,307,620,
removes 6,077 normalized exact and 2,397 verified near duplicates, builds a 2.907 GB SQLite index,
and packs 4.000 GB at 6.90M tokens/s. Peak recorded global-dedup RSS is 675 MB. Source identity,
Gitleaks/contamination filtering, raw cleanup, exact and near deduplication, packing, interruption/resume,
and training/holdout disjointness all pass. All six stage-result hashes reverify.

Acquisition removes 20,357 benchmark-contaminated, 17,641 raw email/IP, 4,647 duplicate-line, five
adult-host, and 2,167 Gitleaks-affected records; the redacted scanner reports contain 3,117 findings.
The resume probe retains exact cleaned output, removals, counts, and logical SQLite table identities.
Physical SQLite page hashes remain unequal and non-authoritative as qualified in finding 207.

Measured linear byte/rate projection gives:

| Target | Live working bytes | Acquisition | Global dedup | Packing | Serial total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20B | 264.8 GB | 18.5 h | 41.3 h | 0.8 h | 60.6 h |
| 150B | 1.99 TB | 138.5 h | 309.7 h | 6.0 h | 454.2 h |
| 500B | 6.62 TB | 461.6 h | 1,032.4 h | 20.1 h | 1,514.1 h |

The dedup projection uses the conservative exact-equivalent batched resume segment, including state
verification and finalization; it is not an isolated steady-state benchmark. Source exhaustion,
non-linear candidate growth, and filesystem behavior remain unmeasured above 2B.

The 20B path fits but is not worth repeating before D5. The 150B branch fits the current 5.0 TiB
volume and is operationally plausible if E3 authorizes repetition, acquired intermediates are cleaned
after dedup, and the long preparation runs under monitored checkpoints. The current 500B branch exceeds
the volume and takes about 63 serial days; it remains blocked pending E3 plus a cleanup/parallelization
or storage successor.

Recommended disposition: accept this calibration as the P0 non-GPU operations fallback through the
150B branch, do not resume the full 20B rehearsal now, and require a new operations/storage plan if E3
selects 500B unique tokens. This recommendation is not itself the project-owner fallback decision and
does not issue production-operations or training authority.

Artifact: [2B calibration analysis](../results/data/production-data-calibration-2b-20260910.json).
