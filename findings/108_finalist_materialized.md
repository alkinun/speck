# 108 — Six-pair finalist configs materialized

## Result

The deterministic materializer created exactly 84 generated JSON configs plus one manifest under
`experiments/Speck-Paper1-Finalist-131M`. The family has two SHA-pinned arm templates and twelve unique
run configs across the frozen six seed/data-order pairs. The manifest records the expected SHA-256 of
every generated config and revalidation passes with no missing, changed, or unexpected JSON.

Each run resolves to 1,539,833,856 training tokens, 23,496 optimizer steps, 5,874-step validation
quartiles, one final checkpoint, and one unique checkpoint path under the dedicated volume. Candidate
and control share seed, data offset, batch, optimizer, tokenizer, data, and training recipe within every
pair. Parameter identities remain 153,977,088 dense and 153,958,938 candidate.

## Boundary

No checkpoint, result, target-lock, or finalist-analysis directory was created. The manifest explicitly
records `training_authorized=false`. Data-window replay, current storage, exact runtime preflight,
collector/analysis behavior, and release dependencies still require separate qualification.

## Artifacts

- [Finalist materialization manifest](../experiments/Speck-Paper1-Finalist-131M/finalist_materialization.json)
- [Materialization contract](../research/paper-1/finalist_materialization_v1.json)
- [Materializer](../speck/paper_finalist.py)
