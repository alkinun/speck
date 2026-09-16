# Stack-Edu resumes from a verified, preserved acquisition prefix

The [launch receipt](../../results/systems/stack-edu-ordered-recovery-launch-20260916.json) binds the
explicit [v2 successor](../flagship/stack_edu_e1s_ordered_acquisition_v2.json) and complete runtime
inventory. All 1,856 completed units, their four acquisition payloads, existing archive tar hashes and
archive inventories reopened successfully. Their totals agree with the stopped durable progress:
668,366 documents and 436,048,483 frozen-tokenizer tokens **before full exclusion**. Verification took
2,660.90 wall seconds on the shared local drive; this is an audit cost, not acquisition throughput.

The original attempt stopped at its conservative storage guard. It was not treated as completed stock
or restarted into the same workspace. Recovery implementation `15e314f` preserves the original units,
cache, archive, owner, failed invocation and logs; a new workspace owns only the remaining acquisition.
It rechecks exact unit configurations and the original ordered prefix before fetching anything new.
The original metadata, filters, language quotas/order and 24GiB working / 12GiB cache / 64GiB actual
free-space guards are unchanged. Historical bytes remain physically present and count against real
free space; the per-workspace cap is not an aggregate historical-storage claim.

`speck-stack-edu-ordered-recovery-20260916.service` was launched from the clean frozen checkout. It has
a six-hour service ceiling and preserves partial work on failure. Completion and actual post-exclusion
capacity remain pending at this launch observation. No competing full-exclusion worker was started
while FineWeb's existing finite queue remains active. The old E1S names identify retained preparation
work; they neither restore the retired experiment matrix nor substitute Stack-Edu for the selected
restricted Stack v3 base source.

Fifteen focused ordered-acquisition/recovery tests passed, including byte parity with uninterrupted
acquisition and rejection of changed guards, paths, input bytes and missing archives. Full quality at
the implementation commit passed 1,282 tests; subsequent independent preparation additions bring the
latest full pass to 1,314. The previous bound status and implementations are preserved in the
[history snapshot](../history/2026-09-16-stack-recovery/manifest.json). The
[readiness successor](../../results/systems/local-preparation-readiness-20260916.json) changes only the
`lc_data` and `stack_edu_stock` status rows and rechecks the unchanged selected contracts. The source-pin
checker flags the intentional status update; that explicit snapshot and successor account for it.
