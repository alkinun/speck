# 19 — Retrieval specificity, exact completion, and language replay

> **Interpretation update (Finding 21):** a held-out-template audit shows that this checkpoint
> transfers to unseen two-token answers but fails when the archive prose is replaced with registry
> syntax. The result below is template-conditioned associative lookup, not general retrieval.

## Why the old metric was insufficient

Findings 16–18 used a factual/counterfactual passkey pair. KDA responded strongly when the distant
answer token changed, but the prompt contained only one answer-bearing record. This proved causal
sensitivity without proving that the model selected the record named by the query.

The strengthened evaluation adds:

- eight-record associative recall;
- two-hop lookup through six distractor chains;
- unconstrained exact next-token emission;
- ten-way candidate accuracy;
- a target mutation and an unrelated distractor mutation for every case;
- **association specificity**: target-change score minus distractor-change score.

A model passes specificity only when changing the queried association affects its prediction more
than changing an irrelevant association. Every result uses 30 deterministic held-out cases.

## Baseline failure

Neither post-32K checkpoint has target-specific retrieval before task adaptation, even at 4K.

| Model/task | 4K target score | 4K distractor score | Specificity accuracy | Candidate accuracy |
| --- | ---: | ---: | ---: | ---: |
| GDN multi-key | 0.129 | 0.133 | 40.0% | 16.7% |
| KDA multi-key | 0.452 | 0.379 | 43.3% | 16.7% |
| GDN two-hop | 0.248 | 0.291 | 36.7% | 10.0% |
| KDA two-hop | 0.497 | 0.598 | 40.0% | 10.0% |

KDA's raw target-change score remains positive through 128K, but its specificity is not significant.
The earlier result must therefore be described as **long-range content sensitivity**, not retrieval.
The first version of the structured curves omitted the distractor mutation; those artifacts are
preserved under `diagnostics/structured-retrieval-no-distractor-control` rather than overwritten.

## Position-wise loss

The new unreduced-loss evaluator measures eight absolute-position bins and the trailing 2K without
materializing sequence-wide vocabulary logits.

| Metric at 32K | GDN/RoPE | KDA/NoPE | KDA minus GDN |
| --- | ---: | ---: | ---: |
| Mean loss | 2.615756 | 2.627110 | +0.011354 |
| Trailing 2K | 2.700115 | 2.722401 | **+0.022286** |
| Wikimedia trailing 2K | 2.285833 | 2.340352 | **+0.054519** |

The gap is not monotonic across position bins, but the trailing region is materially worse than the
average suggests. Aggregate loss alone understated KDA's late-context weakness.

## Task adaptation protocol

Both models start from their completed seed-42 32K checkpoints. Adaptation uses 4K prompts, AdamW at
`1e-4`, batch 4, accumulation 4, and one supervised answer token per example. Training and validation
seeds are disjoint. The model is never trained at 32K or 128K during this stage.

### Hard first pilot

KDA sees 3,200 examples mixing eight-record lookup and six-chain two-hop lookup. It does not pass:

| Task | Exact | Candidate | Specificity accuracy |
| --- | ---: | ---: | ---: |
| Eight-record | 20.0% | 23.3% | 66.7% |
| Six-chain two-hop | 13.3% | 16.7% | 63.3% |

### Easy matched calibration

The load is reduced to two records/two chains while retaining both tasks and 3,200 total examples.

| Model/task | Exact | Candidate | Specificity accuracy |
| --- | ---: | ---: | ---: |
| KDA two-record | **100%** | **100%** | **100%** |
| GDN two-record | 46.7% | 46.7% | 46.7% |
| KDA two-hop | 50.0% | 53.3% | 36.7% |
| GDN two-hop | 53.3% | 50.0% | 26.7% |

KDA learns genuine key/value binding under a protocol where GDN learns answer sensitivity without
target selection. This is the first natural-language-formatted associative-retrieval result that
separates the mixers cleanly.

Two-hop-only training doubles supervision to 6,400 examples. Both architectures still fail:

| Model | Candidate | Specificity accuracy | Specificity score |
| --- | ---: | ---: | ---: |
| KDA | 33.3% | 50.0% | −0.290 |
| GDN | 36.7% | 36.7% | −0.165 |

The present 150M system can learn direct binding but not held-out compositional lookup from this
supervision. Lower training loss does not rescue specificity.

## Exact length and load generalization

The easy KDA checkpoint was trained only at 4K with two-record cases. These measurements retain the
training template; Finding 21 separately tests surface-form transfer.

| Evaluation load | 4K exact | 32K exact | 128K exact | Specificity accuracy |
| --- | ---: | ---: | ---: | ---: |
| 2 records | 100% | 100% | 100% | 100% at all lengths |
| 8 records, unseen | 100% | 100% | 96.7% | 100% at all lengths |

The matched GDN checkpoint has no significant specificity at any length or load. Its exact accuracy
fluctuates because of answer priors and cannot be interpreted as retrieval.

However, full-model retrieval-only adaptation damages general language modeling:

| Model | Parent 4K loss | Adapted 4K loss | Regression |
| --- | ---: | ---: | ---: |
| KDA | 2.799689 | 3.607802 | +0.808112 |
| GDN | 2.805224 | 7.442082 | +4.636859 |

The exact KDA result proves capability after specialization, not a deployable training recipe.

## Mixed language replay

Motivated by Nemotron 3 Nano's mixed short/long continuation, a second KDA adaptation interleaves two
retrieval and two original-language microbatches in every optimizer step. It sees 1,600 two-record
retrieval examples, 6.55M retrieval prompt tokens, and 6.55M original-language replay tokens.

It reaches 100% held-out exact and specificity while preserving language loss:

| Model state | Original-corpus 4K loss |
| --- | ---: |
| Parent | 2.799689 |
| 50% replay adapter | **2.799237** |

Length/load evaluation:

| Load | 4K exact | 32K exact | 128K exact | 128K specificity score |
| --- | ---: | ---: | ---: | ---: |
| 2 records | 100% | 100% | 100% | 11.501 |
| 8 records, unseen | 96.7% | 96.7% | 96.7% | 11.088 |

All six specificity accuracies are 100%. This is genuine distractor-controlled, unconstrained exact
lookup within the trained template after 4K task training, extrapolating to 128K without measurable
language forgetting.

## Decision

KDA/NoPE has demonstrated a real long-context advantage over GDN/RoPE for direct associative lookup
within the trained template. The claim remains narrow:

- one seed;
- answer transfer succeeds on a held-out set of two-token values;
- the checkpoint fails a held-out registry template at 4K;
- task adaptation is required;
- two-hop composition still fails.

Future architecture promotion must use exact answer and specificity gates. Raw factual/
counterfactual direction is retained only as a cheap sensitivity diagnostic. Mixed original-language
replay becomes mandatory for retrieval or long-context task adaptation.

## Artifacts

- Experiment contract:
  [experiments/SpeckLC-150M-StructuredRetrievalAdapt](../experiments/SpeckLC-150M-StructuredRetrievalAdapt)
- Machine-readable summary:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/summary.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/summary.json)
- Exact length/load curves:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/length](../results/SpeckLC-150M-StructuredRetrievalAdapt/length)
- Position-loss curves:
  [results/SpeckLC-150M-KimiContext32K/position-loss](../results/SpeckLC-150M-KimiContext32K/position-loss)
