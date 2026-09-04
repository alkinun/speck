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

## Chain-supervision curriculum

A second 200-step stage starts from the edge-qualified checkpoint, lowers the learning rate to
`5e-5`, and mixes route, payload, explicit chain (`node value`), and direct composition targets. The
validation prompt still requests only the final value.

Stream 42 appears successful: final fixed validation reaches 86.7% exact/candidate and 80.0%
specificity. On 100 independent cases it retains 82.0% exact, 83.0% candidate, and 86.0%
specificity. However, the same intervention with two disjoint training streams does not replicate:

| Chain-stage stream | Exact | Candidate | Specificity accuracy | Specificity score |
| ---: | ---: | ---: | ---: | ---: |
| 42 | **86.7%** | **86.7%** | **80.0%** | 1.785 |
| 43 | 63.3% | 63.3% | 50.0% | −0.089 |
| 44 | 43.3% | 43.3% | 56.7% | 0.046 |

The seed-42 checkpoint also fails to transfer from single-token nodes to the original random
multi-token destination identifiers: 50.0% exact/candidate and 56.7% specificity on 30 cases.

Chain supervision can unlock composition, but simultaneous mixing of edge, chain, and direct
objectives is not a stable recipe. The next experiment temporally separates chain learning from
direct-answer distillation while validating direct composition throughout.

## Staged chain-to-direct distillation

A two-phase 200-step run uses only explicit chain targets for steps 1–100, then only direct
composition targets for steps 101–200. It starts from the same edge-qualified parent, uses a new
training stream, `5e-5` peak learning rate, and 50% language replay.

The chain phase lowers its task loss to nearly zero but direct candidate accuracy falls to 33.3% at
the switch. The direct phase recovers only to 60.0% exact/candidate and 53.3% specificity. Temporal
separation therefore does not stabilize the seed-42 mixed-objective success.

The curriculum branch is closed. The durable result is the decomposition itself: constituent edges
are almost perfectly retrievable, while composition needs an objective or mechanism that explicitly
binds the two changes rather than more sequencing of answer-only losses.

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
- Chain reports and transfer controls:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop](../results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop)
- Chain seed-42 model / metadata / optimizer SHA-256:
  `7a8f5ec2ed447b0a1be941a0f4124f6dc9ff2493c36fb1996947bd4f12568541`,
  `6c0a080094be36a623d6386c082be0c71760cbffc8943010f0ed3fdba3a66652`,
  `0b332cf78f4851009aec1a34460513d8f879817ea7ae80de55eaf09208ab3090`
- Chain seed-43 model / metadata / optimizer SHA-256:
  `8c27c87ccb0bf9e29ffdf662e171cbc9f4dd84886ba48e6f07d7b133d50f6c2d`,
  `8b4070982c151938e1f8a30fa1d3141a3d31ebd8ef3e1113c031a6aa331f00fd`,
  `6ec4e6f69362edfba7763b64ec86225264c9b390c6fb8e72265a149c152fe99b`
- Chain seed-44 model / metadata / optimizer SHA-256:
  `0778695306813f7f7e76f30e2f8e8b110347846651b4260ff8a882e68ddee6e0`,
  `da50cd4be54c9d6ca2e972089e912b06da330dbeb9aaaa418c38771c0a1a02c1`,
  `78382395f46f3072c2985fbb951fcff1fb15f45cab629acb0659615ef764d65d`
- Staged-distillation report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/kda-staged-chain-to-direct-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/two-hop/kda-staged-chain-to-direct-seed42.json)
- Staged model / metadata / optimizer SHA-256:
  `76ae991856ff6f6d3fc6ed49ca03f68baec1bba5cf799352499ad335cca1403e`,
  `41e0dcb31e204daa030caa08b7a0b5edf05c3a57744ed505ad4890106baa8561`,
  `34692564996f5008083fab4b18b67dfcee4d47fcbef3e7dd61a16344de209a2e`
