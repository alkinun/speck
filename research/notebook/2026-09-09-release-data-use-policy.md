# 2026-09-09 — Release and data-use policy

## Context

The technically qualified 30-source set remained blocked on a human decision about commercial-capable
training, public model weights, corpus redistribution, and residual upstream rights risk. Treating every
web page or imperfect metadata field as independently cleared was not practical for the planned model.

## Work performed

Reviewed the six checked rights evidence packets and confirmed the intended release boundary with the
project owner. Created an append-only source-registry successor, completed source-level attribution,
non-redistribution, removal, condition, and rationale records, and generated a human acceptance record.
Created successor rehearsal, launch, and tokenizer plans that remove only the completed rights blocker.

## Decisions

- Repository source code remains MIT.
- First flagship model weights use Apache-2.0.
- Training and public weight release are commercial-capable.
- Source text and packed shards are not redistributed.
- All 30 selected sources are accepted for guarded internal training under their documented terms and
  residual risks.
- Source metadata, citations, notices, known limitations, deny ledgers, removal contact, and rebuild
  behavior remain mandatory.
- Explicit future prohibitions or material terms changes require a new human decision.

## Evidence and links

- [`release_and_data_use_policy_v1.json`](../flagship/release_and_data_use_policy_v1.json)
- [`source_registry_v2.json`](../flagship/source_registry_v2.json)
- [`source_rights_acceptance_v1.json`](../flagship/source_rights_acceptance_v1.json)
- [`findings/204_release_and_source_use_policy.md`](../../findings/204_release_and_source_use_policy.md)

## Open questions

- Which public email or form will receive removal requests before release?
- Which exact paper open-access license will the selected venue permit?
- Do any source terms materially change before production acquisition or release?

## Next actions

- Freeze and execute the real 20B data rehearsal from the approved source registry.
- Materialize real firewall partitions after production operations qualify.
- Carry all citations, notices, and limitations into the paper and model-card generators.
