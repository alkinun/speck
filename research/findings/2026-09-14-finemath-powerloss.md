# FineMath recovery after the reported electricity outage

The owner reported an electricity outage and computer restart. The harness cancelled the running
FineMath shell when its server restarted. On reconnection, the 5.5 TB data drive was detected but
unmounted. The owner mounted its existing ext4 filesystem at `/mnt/speck-data`; its saved data was
then available. No new empty preparation was started under the unmounted directory.

## Verified state before resume

All eight complete raw files, retained acquisition outputs, security reports and acquisition
configuration identities pass their original hashes. Acquisition had finished with **567,002 retained
records**. Exclusion state was committed at **checkpoint 72**, after **360,000 candidate records**,
with 648,872 records seen including the 288,872-reference prefix.

The [snapshot receipt](../../results/systems/finemath-powerloss-snapshot-20260914.json) records exact
copies of the interrupted staging directory, including SQLite main/WAL/SHM files, state, output and
removal tails, plus execution/progress/configuration records. Source and snapshot hashes were compared
before SQLite was opened by the recovery process. The snapshot remains under
`/mnt/speck-data/speck/recovery/finemath-powerloss-20260914`.

The completed Math L2, FineWiki and expanded peS2o token caches also passed post-outage raw-parent,
tokenizer, shard/index hash and contiguous document-coverage checks.

## Frozen execution recovery

The resume uses original revision `8e9ae3b89ead03d2cf1e479284606dc2797830ce` in a detached recovery
checkout, with the original absolute plan path and hash. A compatibility symlink for
`research/flagship/web_contamination_v1.json` points to the identical archived input. This is necessary
because the original code's fallback resolver cannot reroot an absent absolute path belonging to a
different checkout. Its resolved path and hash match the original acquisition configuration.

The [launch receipt](../../results/systems/finemath-powerloss-resume-20260914.json) records the systemd
user service, invocation, output path and log. The service runs independently of the assistant-server
process. The maintained checkout stays available for documentation while the scientific execution
uses the original code and checkpoint contract.

Final publication, exclusion controls, output/index hashes and usable-token capacity must still pass
after resume. The lost original exclusion timing/WAL peak must remain explicit; resumed timings are
not complete original-run timings. This is evidence from one unplanned outage, not a controlled
power-loss experiment or a general guarantee about future storage failures.

Follow-up: the initial resume exposed an [unindexed cascade-cleanup bottleneck](2026-09-14-indexed-recovery.md).
An explicitly recorded physical-index migration removed that scan, and the new indexed service has
advanced beyond checkpoint 72. Its migration is part of the recovery lineage; final publication is
still required before reporting complete recovery success.

## Completion follow-up

The indexed service subsequently exited successfully. The [completed result](../../results/data/finemath-stock-preparation-20260914.json)
passed final artifact and exclusion checks, retaining 814,103,172 tokens. Recovery succeeded; the
960M headroom target remains short by 145,896,828 tokens. See the [completion finding](2026-09-14-finemath-stock.md).
The progress/pending statements above describe the earlier observation, not the current service state.
