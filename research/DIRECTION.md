# SpeckLabs research direction

## Mission

Build open language models that maximize intelligence per unit of training compute, serving compute,
and resident memory. Every flagship should ship as a useful model, establish reusable scientific
knowledge, and leave a continuation checkpoint and evidence ladder for the next allocation.

“Frontier” means a measured Pareto frontier under a declared scale and hardware envelope. Grant 1 does
not claim absolute capability parity with laboratories training hundreds of billions of parameters on
tens of trillions of tokens.

## Research principle: allocation, not mechanism accumulation

Under fixed resources, model quality depends on where computation and memory are allocated:

- **training data:** high-information tokens, repetition, mixture, and decay;
- **sequence computation:** recurrent processing versus periodic exact access;
- **model capacity:** parameters versus token horizon and, later, conditional width;
- **serving state:** fixed state versus length-growing state, cache precision, and persistence;
- **test-time computation:** prompt prefill, token decode, and generated reasoning tokens.

New mechanisms enter a flagship only after isolated correctness, causal evidence, scale transfer, and
realized hardware benefit. A public release report is prior evidence, not permission to copy its full
operator stack.

## Grant 1: prove the operating method

Grant 1 uses 5,000 GH200 GPU-hours to train one fixed 1.2B-class dense-width KDA/global model over a
400B-token target, with a 320B-token throughput fallback. The paper asks whether high-information data
and periodic exact memory produce complementary quality/compute/state gains that transfer, compose,
and survive scale.

The four paper claims are:

1. selected data improves equal-category quality per training FLOP without hiding a domain regression;
2. a recurrent/global hybrid preserves dense quality with lower training compute and length-growing
   state;
3. the data and architecture effects transfer, compose, and survive scale and a mature-token-horizon
   sentinel;
4. the released 1.2B system realizes the predicted frontier on named hardware.

The model is the held-out consequence of the evidence program, not a configuration chosen after its
results. Negative and unresolved outcomes narrow these claims without changing the paper question.

## Grant 2: conditional width

Begin from the public pre-decay dense-width checkpoint. Sparse-upcycle selected feed-forward layers,
compare with continued dense training at matched data and wall clock, and qualify expert-parallel training,
checkpointing, export, and serving. This is the path to more total knowledge capacity without charging
every parameter on every token.

## Grant 3: asymmetric context computation

Study prompt/decode asymmetry and compressed global memory as one integrated program:

- lower causal encoding of complete prompts;
- periodically refreshed global representations;
- more frequent depth-specific selection or reads;
- an exact local or recurrent path;
- quantization-aware global cache storage;
- explicit runtime versus persistent-cache economics;
- trained and measured recovery after cache misses.

DeepSeek-V4.1-Flash is the current large-scale reference for this direction. Speck will not transplant
its CED/CSA2/mHC/Engram/DSpark package into grant 1.

## Long-term lab standard

Each generation should improve at least one measured frontier without silently weakening another:

- quality per training FLOP and GPU-hour;
- quality per total and active parameter;
- quality per runtime and persistent byte;
- quality per prefill and decode second;
- quality per generated token and completed task;
- reproducibility from public code, contracts, small evidence, and immutable model artifacts.

The public research object includes model weights, code, experiment and analysis contracts, losing
arms, checked results, source metadata, model cards, and an independently regenerated paper. When data
bytes cannot be redistributed, publish enough identity, transformation, and attribution metadata to
make the boundary explicit rather than describing the release as fully open data.
