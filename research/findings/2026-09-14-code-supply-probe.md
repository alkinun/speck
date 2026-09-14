# Added-file code supply measured

The [bound v2 probe](../flagship/code_supply_probe_v2.json) completed 1,920 seeded samples across
all fifteen added files. Content/security/Gitleaks checks retained 1,212 documents and 712,507
Mistral tokens. The [result](../../results/data/code-supply-probe-v2-20260914.json) binds the
completed metadata, indices, samples, cache inputs, content checks and verified 159,016,960-byte
archival copy. Complete reopen passed. The original qualification/attempts remain intact.

Indexing took 26.57 seconds and fetching 21.34 seconds. Those stages exclude subsequent content,
security and archival processing, and include current cache/network/storage conditions. This was
a supply probe using the already-qualified fetcher; no interruption/warm benchmark was repeated
and no controlled speedup is claimed.

The [combined analysis](../../results/data/code-supply-comparison-20260914.json) adds each original
first-file estimate to its disjoint added-file estimate. Each file-position stratum retains its
own population size; sampling variances are summed. Rounded estimates before full exclusion:

| Language | Estimated M tokens | Sampling SE, M tokens | E1S headroom target, M tokens |
| --- | ---: | ---: | ---: |
| Python | 80.315 | 8.565 | 90 |
| C++ | 69.960 | 3.545 | 54 |
| C | 23.374 | 2.121 | 18 |
| Java | 45.826 | 1.566 | 54 |
| JavaScript | 48.507 | 4.887 | 54 |
| TypeScript | 31.683 | 3.525 | 36 |
| Rust | 36.393 | 3.638 | 18 |
| Go | 23.996 | 2.529 | 18 |
| Shell | 294.447 | 37.549 | 3.6 |
| SQL | 55.936 | 7.034 | 3.6 |
| Markdown | 50.756 | 10.462 | 10.8 |

These are **sample estimates, not measured whole-source capacity**. Full reference/candidate
exclusion can reduce supply. They do not justify a paper capacity-table entry or simultaneous
confidence claims. Both complete Java and TypeScript views remain below the E1S headroom target
in the point estimates; JavaScript/Python still have further released files. The larger incumbent
background requirement is a separate, harder constraint.

Proceed with capacity resolution before bulk stock fetching: additional complete JavaScript/
Python metadata can expand those languages; Java/TypeScript need a concrete source/language or
background decision because no further files exist at the pinned revision. Preserve all existing
contracts until a recorded pre-results successor is adopted. No quality/license filter relaxation,
headroom reduction, source substitution, tokenizer reopening or model launch follows from this probe.
