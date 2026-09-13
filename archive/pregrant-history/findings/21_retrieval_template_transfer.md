# 21 — Retrieval answer transfer and template failure

## Question

Finding 19 established distractor-controlled exact lookup through 128K after 4K adaptation with
50% original-language replay. That result used the same archive prose in training and evaluation,
and every answer was one token from `A`–`J`. This audit separates two possible failure modes:

1. dependence on the ten training answers;
2. dependence on the training prompt grammar.

The completed KDA replay checkpoint is evaluated without further training. All conditions use eight
records, 30 held-out seeds, exact 4,096-token examples, target mutation, and unrelated-distractor
mutation. The new phrase set contains ten arbitrary two-token pairs with unique first tokens. Exact
match requires both tokens; candidate and specificity metrics use the ten unique first tokens.

## Conditions

| Template | Answer set | What changes from training |
| --- | --- | --- |
| Archive | Two-token phrases | Answers only |
| Registry | `A`–`J` | Surface syntax only |
| Registry | Two-token phrases | Surface syntax and answers |

The archive condition uses the prose seen during adaptation. The registry condition expresses the
same bindings as `ID[...] :: PAYLOAD[...]` and queries them with `LOOKUP ID[...]`.

## Result

| Template / answers | Exact | Token accuracy | Candidate | Target score | Distractor score | Specificity accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Archive / phrases | **96.7%** | **98.3%** | **100%** | 7.806 | 0.123 | **100%** |
| Registry / letters | 6.7% | 6.7% | 6.7% | 0.341 | 0.341 | 46.7% |
| Registry / phrases | 0.0% | 8.3% | 10.0% | 0.016 | 0.020 | 46.7% |

The phrase result is strong answer-format transfer. The model retrieves and emits unseen two-token
values under the learned archive grammar. The registry result is a decisive failure at the shortest
length: candidate accuracy is statistically indistinguishable from 10% chance, and association
specificity is indistinguishable from 50% direction chance (`p = 0.708`). In the registry/letter
condition, every target mutation moves the logits, but an unrelated mutation moves them by the same
amount. This is exactly the nonspecific sensitivity that the distractor control was designed to
expose.

## Interpretation

The earlier 128K statement is valid only as **template-conditioned associative lookup**. It proves
that a relation learned at 4K can remain usable at 128K and can bind unseen multi-token values, while
50% replay preserves language loss. It does not prove general retrieval across surface forms.

No 32K or 128K registry run is warranted because the independent template already fails at 4K. The
next adaptation must train on multiple surface forms and reserve at least one entirely held-out
template. Seed replication of the single-template recipe would only replicate the wrong target.

## Decision

- Retain KDA as the lead mechanism, but remove “general retrieval” from the evidence ledger.
- Make template split part of the task contract, alongside seed and record-count splits.
- Require held-out-template candidate and specificity accuracy before any MQA architecture spend.
- Keep two-token exact emission in the standard evaluation; answer-vocabulary memorization is not
  the observed limitation.

## Artifacts

- Evaluator implementation: `speck/long_context.py` and `scripts/structured_retrieval_eval.py`
- Raw reports:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/transfer](../results/SpeckLC-150M-StructuredRetrievalAdapt/transfer)
- Source checkpoint: `SpeckLC-150M-StructuredRetrievalAdapt-kda-replay50`, step 200,
  model SHA-256 `dd49b20afc9bc1a15692a94f5f0ea77433a1776186adc90023e2698d834bc066`

