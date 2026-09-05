# Research programs

The active [Speck Paper 1 program](paper-1/) defines the thesis, novelty gate, claim ledger,
tri-axis experiment sequence, manuscript structure, and paper-scale pretraining gate.

## Architecture promotion

This directory contains the versioned decision contract for selecting Speck architecture
components. It sits between the literature notes in `papers/`, the experiment definitions in
`experiments/`, and the observed results in `results/`.

The active contract is [`architecture-promotion-v1`](architecture-promotion-v1/):

- [`policy.json`](architecture-promotion-v1/policy.json) defines the statistical design and the
  gates a component must pass.
- [`cost_envelopes.json`](architecture-promotion-v1/cost_envelopes.json) defines named training and
  serving workloads on the current research hardware.
- [`evaluation_manifest.json`](architecture-promotion-v1/evaluation_manifest.json) freezes internal
  evaluation settings and upstream revisions. An upstream pin is not a qualification result.
- [`evidence_matrix.json`](architecture-promotion-v1/evidence_matrix.json) records what is retained,
  rejected, proposed, or still unresolved.
- [`internal/structured_retrieval_v2.json`](architecture-promotion-v1/internal/structured_retrieval_v2.json)
  freezes the 200-case, two/eight-record, held-out-template and held-out-answer protocol.
- [`internal/symbolic_composition_v2.json`](architecture-promotion-v1/internal/symbolic_composition_v2.json)
  freezes route, payload, and direct-composition views over a tokenizer-qualified 100-way route
  vocabulary.
- [`external/`](architecture-promotion-v1/external/) contains source-, revision-, file-, data-,
  license-, and model-adapter contracts for RULERv1, NoLiMa, and HELMET.

Validate the complete cross-file contract with:

```bash
uv run --extra cpu python -m scripts.research_contract_validate \
  research/architecture-promotion-v1 \
  --tokenizer-experiment experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope
```

Preflight every declared validation case without loading a checkpoint:

```bash
uv run --extra cpu python -m scripts.promotion_case_preflight \
  research/architecture-promotion-v1/internal/structured_retrieval_v2.json \
  --tokenizer-experiment experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope
```

The checked [preflight summary](../results/Speck-Architecture-Promotion-v1/protocol-case-preflight.json)
contains the resulting case-stream hashes for both internal protocols.

Qualify an external source checkout without downloading its benchmark data:

```bash
uv run --extra cpu python -m scripts.external_suite_qualify \
  research/architecture-promotion-v1/external/helmet.json \
  --checkout helmet_pipeline=<pinned-checkout>
```

The checked
[external-source qualification](../results/Speck-Architecture-Promotion-v1/external-source-qualification.json)
proves upstream commit and source-file identity. The separate
[endpoint qualification](../results/Speck-Architecture-Promotion-v1/evaluation-endpoint-qualification.json)
binds an attested instruction export to the exact serialized OpenAI request shapes used by RULER's
pinned NeMo-Skills adapter and NoLiMa. It does not qualify long-context capability or serving speed.
RULER remains blocked on its transitive data pins; NoLiMa requires acceptance of a non-commercial
research license and a data manifest; HELMET remains blocked on a separate 34GB data volume and its
own model-adapter smoke.

Reproduce the endpoint protocol check for an attested instruction export with:

```bash
uv run --extra cpu --group transformers python -m scripts.evaluation_endpoint_qualify \
  <instruction-export> --device cpu --dtype bfloat16 --output <qualification.json>
```

Run one protocol-bound adapter. The protocol supplies every scientific setting, including the
training/validation split, synthetic streams, replay source, optimizer, and sample counts:

```bash
uv run --extra gpu --extra linear python -m scripts.structured_retrieval_adapt \
  experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope \
  --protocol research/architecture-promotion-v1/internal/structured_retrieval_v2.json \
  --seed 42 \
  --checkpoint-dir <parent-checkpoint> \
  --output-dir <adapter-checkpoint> \
  --report <adapter-report.json>
```

Then run the protocol's exact 4K/32K/128K condition grid:

```bash
uv run --extra gpu --extra linear python -m scripts.structured_retrieval_eval \
  experiments/SpeckLC-150M-KimiTransfer131M/kda-sigmoid-nope \
  --protocol research/architecture-promotion-v1/internal/structured_retrieval_v2.json \
  --protocol-length 4096 \
  --checkpoint-dir <adapter-checkpoint> \
  --step 400 \
  --output <length-report.json>
```

Run 32K only after the paired 4K gate passes, and 128K only after the paired 32K gate passes.
`--protocol-length` must name exactly one length declared by the protocol, preventing one invocation
from charging through a failed shorter-length gate.

Operational arguments select paths, device, and compilation. When `--protocol` is present, the runner
replaces all scientific CLI values with the pinned protocol and records its absolute path and SHA-256
in the checkpoint and report. A copied or edited protocol is rejected unless the active evaluation
manifest pins that exact path and hash.

## Decision principle

Speck does not combine quality and cost into one opaque score. Quality and correctness are hard
constraints. Among candidates that pass them, the selected point must improve at least one declared
training or serving profile without violating the others. Claims are attached to a specific model
scale, training horizon, context length, numerical format, runtime, batch/load profile, and hardware
configuration.

The policy uses paired one-sided confidence bounds for non-inferiority. NIST's engineering statistics
guidance motivates the paired comparison, one-sided threshold test, and prospective power calculation:

- <https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm>
- <https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm>

Training cost is measured as wall-clock time and physical resources required to reach a locked quality
target, following the central MLPerf Training principle:

- <https://mlcommons.org/benchmarks/training/>

Serving profiles report TTFT, TPOT/inter-token latency, and token throughput rather than a context-free
"speedup":

- <https://mlcommons.org/2024/03/mlperf-llama2-70b/>
- <https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md>
