# 202 — Pre-access training and Slurm failure boundaries are explicit

The 4xGH200 operations layer now validates immutable wave manifests, clean tracked Git state, exact
config/data/authority identities, one-GPU arrays, four-GPU jobs, mechanical dependencies, bounded
same-manifest retries, `sacct` accounting, and separate 4,111-hour mandatory and 889-hour protected
reserve pools. It cannot automate scientific promotion or reserve spending, and account/partition
values remain site inputs.

Slurm preemption no longer relies on a copied training loop. A small subclass hook observes the
signal sentinel after an optimizer boundary, synchronizes the request across ranks, writes one partial
checkpoint, emits a distinct non-final summary, and exits 99 for bounded exact-latest requeue. Partial
or milestone checkpoints cannot be exported as completed models.

Base validation and final parameter publication reject non-finite values. Resume compilation is
charged to startup rather than steady time. Distributed rank/local-rank/GPU geometry and per-rank
manifest/tokenizer identity fail closed, distinct token milestones cannot collapse silently, and data
launch receipts now require a clean tracked tree identity.

CPU failure injection is complete. SFT requeue, multi-node Slurm, live scheduler signals, NCCL,
GH200 kernels, storage, and actual resume behavior remain hardware gates.

Artifacts: [launch-risk successor](../results/launch-risk-hardening-successor-20260908.json),
[Slurm guide](../docs/slurm.md), and
[combined integration record](../results/preaccess-engineering-integration-20260908.json).
