# FineMath recovered successfully; nominal capacity passes but headroom is short

The [completed eight-shard result](../../results/data/finemath-stock-preparation-20260914.json)
(SHA-256 `5aee0d7a544d00df3e6bab4dd84e7c4e2b4d42b3743bd3deb3939c34ca5c3013`)
binds the unchanged [first preparation plan](../flagship/finemath_stock_preparation_v1.json).
The 837,440 physical rows / 2,289,547,838 compressed bytes yielded 567,002 acquisition candidates.
Full exclusion retained **545,996 documents**, **2,661,929,096 UTF-8 text bytes**, and
**814,103,172 frozen-Mistral tokens**, including BOS/EOS.

The **800M nominal requirement passes** by 14,103,172 tokens (1.76%). The **960M preparation target
fails** by 145,896,828 tokens. The result correctly reports
`prepared_text_requires_capacity_or_storage_followup`, with capacity false and storage true.
This is successful recovery with insufficient headroom. It is not a token cache or training manifest.

Verified output/index/removal identities, reference preservation, exact/near positive controls and
zero final exact candidate/reference overlap passed. Candidate removals comprise 14,238 exact
reference matches, 766 verified-near reference matches and 6,002 verified-near candidate duplicates.
These counts exclude the injected positive controls.

The declared natural-domain policy retained 49,300 known hosts and no missing-host records. Largest
retained-text byte shares are physicsforums.com 4.40%, gmatclub.com 3.07%, nrich.maths.org 2.73%, and
math.stackexchange.com 2.67%; known-host byte HHI is 0.006335. These are distribution diagnostics,
not learning-quality findings or a reason to reweight the source after observing outputs.

The [outage snapshot](../../results/systems/finemath-powerloss-snapshot-20260914.json),
[initial resume](../../results/systems/finemath-powerloss-resume-20260914.json), and
[physical-index migration](../../results/systems/finemath-recovery-index-20260914.json) remain part
of the operating provenance. The final indexed service completed with exit status 0 after
49m 36.539s wall / 12m 14.454s CPU. The last resumed exclusion invocation took 2,858.52 seconds
with a 369.24 MiB observed WAL peak. Original interrupted elapsed time and WAL peak remain
incomplete; the stopped unindexed recovery is a separate attempt. These are not total original
costs, a controlled same-storage speedup, or GH200 throughput evidence.

Next, pin additional complete shards in a successor, reuse only configuration-identical verified
acquisition units, and rerun combined exclusion without relaxing filters or the 960M target.
The expanded bank replaces this count; overlapping stocks are not added. Then build and verify its
document-token cache. Joint background/treatment eligibility, ordering and launch authority remain
separate. No preparation job or FineMath token-cache job remains running at this completion record.
