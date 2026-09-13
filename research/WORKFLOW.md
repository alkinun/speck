# Research workflow

## Define and freeze

State the question, control, metrics, seeds, stopping rule, compute envelope, and fallback.
Develop draft plans in Git; freeze exact model, data, tokenizer, implementation, and analysis
identities before consequential outputs. The catalog identifies the selected contract for each role.

Protocol changes after outputs need explicit successors with the predecessor retained. Technical
fixtures qualify plumbing; they do not establish model quality or production throughput.

## Execute and retain

Run from a clean, identified implementation with qualified inputs and checkpoint/resume behavior.
Record actual compute, data order, hardware/software, checkpoint parents, outcomes, and external
artifact identities. Retain failed attempts and interruptions within the experiment history.

Monitor mechanical health during confirmatory work. Arm selection, stopping, and analysis follow
the frozen rules. W&B and Linear supplement the durable record.

## Analyze and communicate

Analyze the declared set, including failures, uncertainty, per-domain results, and measured cost.
Publish a coherent result record and a finding when there is a durable conclusion. Routine software
changes are explained in commits and behavioral tests.

Update `research/status.json` after a meaningful transition. Update `paper/claims.json` when the
evidence changes a claim. Figures and tables should read checked results and record their generating
command and input identities.

## Close and archive

Move completed collections out of the working directories with their original paths, hashes, and
reproduction revisions recorded. Historical verification uses the original implementation identity;
new execution binds the maintained implementation. Preserve negative and unresolved evidence.

Follow the [artifact policy](DATA_MANAGEMENT.md) for runtime storage, backups, and publication.
Validate current references and archive integrity with `make quality`.
