# Finite local preparation followups

The [bound sequence](local_preparation_followups_v2.json) implements the owner's request to keep
preparation moving without another message. It is a local job sequence, not a model-launch receipt.

1. Wait for the existing frozen FineMath successor to finish successfully. Require its exact plan
   and implementation identities, 960M capacity and storage gates, preserved references, zero exact
   reference overlap, and exact/near positive controls. Bind the completed result into a runtime
   token plan, verify the text/reference inputs, build the token cache and verify complete reopen.
2. Apply the same process to the Cosmopedia stock, also targeting 960M.
3. After both source jobs have finished, execute the already-bound three-file FineWeb E1S tranche.
   Require its 1.32B target and the same relevant checks, then build and verify its token cache.

The data filesystem UUID is checked before creating the working directory and before each new
work stage. The sequence runs from a frozen clean checkout through a user systemd service. It
can survive logout; a machine shutdown still interrupts it. Existing source services retain their
own ownership and recovery rules. No second large exclusion pass is launched by this sequence
until the current FineMath and Cosmopedia services have completed.

Each token plan binds the future result's actual SHA-256 after successful completion. Plans,
progress, failures and the final sequence receipt are preserved under
`/mnt/speck-data/speck/local-preparation-followups-v2-20260914`. Result receipts are published under
`results/data/`. The script performs no Git mutation; completed evidence and maintained status
are reviewed and committed separately. Token caches remain source-specific, not jointly eligible
experimental datasets or final training manifests.

A source failure, capacity/storage shortfall, missing service without a completed publication, changed identity, missing control,
unmounted data disk or 24-hour dependency-wait deadline stops dependent work with a failure record.
There is no automatic retry into unfinished output or changed scientific contract. The deliberate
Stack-Edu pause is unaffected; its supply/recipe review remains separate. No new source-use approval,
model training, sealed-evaluation opening or post-training work is authorized by this sequence.

The original waiting-only sequence was intentionally interrupted and preserved before any cache
or dependent text job began. Its [replacement receipt](../../results/systems/local-followup-replacement-20260914.json)
records the reason: systemd garbage-collects successful transient services. An unloaded service
now hands its existing publication to the same strict execution/capacity/exclusion checks, and
records that the service exit status is unavailable. A missing result or invalid publication
still stops dependent work. The FineMath/Cosmopedia source services were not stopped.

As of 2026-09-15, FineMath and Cosmopedia text/cache stages are complete and reviewed. The
same frozen sequence is processing the FineWeb-Edu E1S text tranche; its capacity and cache
remain pending. Do not launch duplicate source/cache jobs.
