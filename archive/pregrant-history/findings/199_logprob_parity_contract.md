# 199 — R13 now has a pre-access log-probability parity contract

The R13/SPE-116 comparator table remains incomplete, but it now has a small correctness gate for the
failure class highlighted by Magic's reported vLLM/SGLang baseline issues. The offline comparator
consumes one declared trusted native or Hugging Face per-token record and one or more optimized-backend
records. It requires
exact evaluation-manifest, model, tokenizer, canonical payload, case order, position, and token-ID
identity before measuring maximum and mean absolute log-probability error under dtype-specific
thresholds.

Missing, non-finite, unhashed, and misaligned data are validation failures, not numerical failures.
Reports retain all input artifact hashes, per-backend/dtype checks, the shared payload/token identity,
and an explicit local-files-only network declaration. Existing report paths cannot be overwritten.

This does not modify the local OpenAI-compatible evaluation endpoint, which still rejects log
probabilities, or the hash-pinned architecture-promotion external manifest. Backend-native collectors
remain separate and must be reviewed before real use. This separation avoids silently treating an API
adapter as a numerical reference while retaining the existing external-suite evidence boundary.

The checked vLLM- and SGLang-named records are deterministic synthetic fixtures only. Neither package
was installed, no model was evaluated, and the provisional thresholds estimate no production error.
The fixture result has no authority to complete R13, qualify a backend, select a comparator, support a
quality result, or make a serving claim. A real run requires a pre-results successor that pins backend
revisions, collector code, hardware/software, exact payload selection, and reviewed dtype thresholds.

Artifacts: [non-authoritative plan](../research/flagship/logprob_parity_plan.json),
[synthetic fixture report](../results/evaluation/logprob-parity-fixture-20260908.json), and
[record fixtures](../tests/fixtures/logprob_parity/).
