# Architecture promotion research

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

Validate the complete cross-file contract with:

```bash
uv run --extra cpu python -m scripts.research_contract_validate \
  research/architecture-promotion-v1
```

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

