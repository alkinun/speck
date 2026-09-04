# 23 — Symbolic two-hop composition

## Question

The earlier natural-language two-hop task failed after 6,400 answer-supervised examples, but that
failure did not reveal whether the model could not retrieve each edge or could not compose them. This
experiment introduces a shared symbolic graph with ten one-token intermediate nodes (`K`–`T`) and
three supervised queries over the same two-record prompts:

- route: start key → intermediate node;
- payload: intermediate node → value;
- composition: start key → value through both edges.

The intermediate vocabulary is separate from the `A`–`J` value vocabulary. Counterfactual route
mutations use an unused node, so changing a distractor cannot silently change the queried route.

## Protocol

The seed-42 post-32K KDA/sigmoid/NoPE checkpoint is adapted for 400 steps at 4K, batch 4,
accumulation 4, AdamW `1e-4`, and 50% original-language replay. The 3,200 retrieval examples cycle
evenly across route, payload, and composition queries. Training and validation seeds are disjoint.
Validation during training contains composition queries only.

## Result

The final 30-case composition validation remains below gate: 46.7% exact, 50.0% candidate, and 56.7%
specificity. A separate 100-case evaluation of every view makes the failure unambiguous:

| Query | Exact | Candidate | Specificity accuracy | Specificity score |
| --- | ---: | ---: | ---: | ---: |
| Route edge | **99.0%** | **99.0%** | **100%** | 9.287 |
| Payload edge | **100%** | **100%** | **100%** | 8.790 |
| Two-hop composition | 43.0% | 43.0% | 45.0% | −0.139 |

The composed prompt has a large positive target-change score, but its distractor-change score is
slightly larger. The model reacts to changed values without selecting the chain named by the query.
This is a composition failure, not a failure to store or retrieve either constituent edge.

## Decision

- Keep the symbolic graph as the calibrated two-hop diagnostic.
- Do not interpret more answer-only composition examples as evidence of reasoning; they already
  coexist with perfect edge lookup and still fail specificity.
- Add a training-only chain target that emits the intermediate node followed by the final value.
  Continue to validate direct final-value composition with no revealed intermediate.
- Mixed language replay remains mandatory.

## Artifacts

- Training report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/kda-symbolic-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/kda-symbolic-seed42.json)
- Auxiliary evaluation:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/eval-kda-symbolic-seed42-auxiliaries-4k-n100.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/eval-kda-symbolic-seed42-auxiliaries-4k-n100.json)
- Checkpoint: `SpeckLC-150M-StructuredRetrievalAdapt-kda-symbolic-twohop-seed42`, step 400
- Model / metadata / optimizer SHA-256:
  `26e9e242464b9adf51803f0974e7dbdc8524a770ab196d407185dbf18490a1f6`,
  `3ec7874c1058aeee4a33814252944b1db03b5794074f85553ff81722bcdb0d10`,
  `e3e477e66c4381dbeb29b7f35be1960204e8f17c5f4e10eae3ef73c606573dc6`

