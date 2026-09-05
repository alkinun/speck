# 40 — HELMET native Speck adapter qualification

## Question

Can the pinned HELMET native Hugging Face path load and run a real Speck export in a locked environment
without installing a duplicate CUDA stack, enabling compilation, changing prompt/scorer behavior, or
accessing the network?

## Environment and source

The `helmet-adapter` dependency group locks the model-path requirements, including Accelerate 1.10.1,
Datasets 5.0.1, SentencePiece 0.2.2, Setuptools 80.9.0, Transformers 5.1.0, and CPU Torch 2.9.1. It
also includes the dependencies needed to execute HELMET's RULER recall and QA post-processing. The
qualification uses HELMET commit `af609c4` and hashes `model_utils.py`, `data.py`, `utils.py`,
`eval.py`, `arguments.py`, and the upstream requirements file.

The model is the existing parity-attested `Speck2-140M-Instruct-endpoint-current` export, with complete
directory identity `08685c3046fec1e9c918cf57b281f4779dbfc55b5443682e95e4b4b4a6ddaa90`.

## Result

The unmodified pinned `model_utils.HFModel` successfully:

- loads the remote-code `SpeckForCausalLM` class with `device_map="auto"` on CPU FP32;
- records eager attention and confirms that the returned model is not torch-compiled;
- sets the slow Speck tokenizer to left padding and left truncation as HELMET requires;
- maps the official synthetic RULER row to a 78-token prompt;
- truncates a 2,589-character synthetic context to a 309-character prefix and exactly 124 prompt
  tokens under a four-token generation reserve;
- produces the same raw output dictionary twice, including the deterministic text `111`;
- returns 1.0 versus 0.5 official RULER recall for full versus partial answer coverage; and
- parses `Answer: 111` to `111` with exact-match and substring-exact-match success.

The worker runs with Hugging Face/Datasets offline flags and inherited IPv4/IPv6 socket denial. A
deliberate connection self-test is rejected, then the complete qualification records zero network
attempts.

## Boundary

This qualifies the native CPU/eager adapter protocol, not HELMET capability, GPU correctness, or
serving performance. The synthetic scorer smoke uses HELMET's actual RULER and QA post-processing.
`pytrec_eval` is an unused eager import on those paths and is stubbed because its PyPI source build
performs an unverified nested GitHub download; reproducible reranking remains a full-suite environment
blocker. The 34GB dataset, component license audit, and candidate-specific context-ceiling exports also
remain blocked.

## Decision

SPE-100's native-adapter smoke is complete. HELMET remains execution-blocked by SPE-103 and by the
full dataset/scorer environment. No architecture or capability claim changes.

## Artifacts

- [Adapter qualification](../results/Speck-Architecture-Promotion-v1/helmet-adapter-qualification.json)
- [HELMET contract](../research/architecture-promotion-v1/external/helmet.json)
- [Qualification runner](../scripts/helmet_adapter_qualify.py)
