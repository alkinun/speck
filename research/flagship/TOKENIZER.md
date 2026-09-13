# Tokenizer protocol

D5 compares the pinned Mistral 32K tokenizer with custom SentencePiece BPE candidates. Final selection
requires measured language-model quality and cost; static compression alone cannot select a tokenizer.

## Selected contracts

- [Tokenizer plan](tokenizer_plan_v7.json): candidate geometry, balanced sample, static evaluation,
  and nomination.
- [Pilot plan](tokenizer_pilot_plan_v10.json): paired document/compute views, screening, confirmation,
  and the sealed D5 decision.
- [Embedding/head contract](embedding_head_contract_v1.json): physically tied parameter accounting.
- [Source registry](source_registry_v2.json) and [firewall](firewall_plan_v4.json): source identities,
  quotas, and disjoint selection/audit inputs.

The catalog selects these versions explicitly. Their predecessor chains, executable input records,
and qualification outputs are retained in the [archive](../../archive/README.md).

## Design

Sample the six categories—web, code, math, synthetic, science, and reference—under the frozen quotas
and source identities. Keep tokenizer training, static evaluation, LM selection, and sealed audits
disjoint. Train candidates deterministically and report category-level compression and byte fallback.

The corrected candidate set is 32,000, 32,768, and 40,960 pieces. Static Pareto nomination advances
the compression endpoint and a distinct compact endpoint to the paired LM pilot. Mistral remains
the baseline and fallback. The uint16 packing and reserved chat-token budget constrain vocabulary size.

The LM pilot uses one whole-document stream tokenized under each candidate and an unchanged backbone
with physically tied embedding/head accounting. It reports both a fixed-document endpoint and a
fixed-analytic-FLOP endpoint, category-level bits per UTF-8 byte, uncertainty, learning curves,
throughput, and memory. The exact stops, guardrails, seed matrix, and selection rule are frozen in
the selected pilot contract.

## Execution and decision

Use [current status](../status.json) for completed arms and next actions. The existing three-arm
screen is bound to its original execution checkout. Restore it with:

```bash
python -m scripts.archive restore /path/to/tokenizer-screen-checkout
```

Run its preserved execution records with the original runtime inputs. The detailed commands and
qualification lineage are in the [original tokenizer guide](../../archive/pregrant-history/research/flagship/TOKENIZER.md).

After the three screen arms complete, reconcile actual overhead before the eligible confirmation
runs. Open D5 only according to the frozen rule, then freeze the selected tokenizer artifact and
update corpus, model-accounting, and evaluation manifests. A failure keeps the declared fallback;
the cleanup does not alter the decision design.

Use the [measured screen handoff](TOKENIZER_SCREEN.md) to reconcile all-attempt GPU spending and
project the four confirmation runs from the three completed screens. The maintained CPU analyzer
records the budget disposition without issuing confirmation or audit-opening authority.
