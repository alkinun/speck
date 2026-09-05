# 25 — Architecture promotion policy and cost envelopes

## Question

How should Speck decide that a cheaper component is genuinely quality-preserving, and what does
"cheap" mean on the hardware and workloads available to the project?

The Reader Attention frontier exposed the need for this contract. Three caches had a real state and
high-batch decode benefit, yet small language-loss differences, one failed seed-level candidate gate,
and a symbolic route-edge failure prevented promotion. Treating an unresolved mean difference as
equivalence would have selected the wrong architecture.

## Decision

The active machine-readable contract is
[`research/architecture-promotion-v1`](../research/architecture-promotion-v1/). It establishes:

- paired candidate/control comparisons;
- a default `0.01`-nat language-loss non-inferiority margin, equivalent to a maximum `1.01005×`
  perplexity ratio;
- one-sided 95% confidence bounds instead of “no significant difference” reasoning;
- one-seed discovery, three-pair proxy confirmation, a six-pair crossed finalist design, and
  three-pair medium-scale transfer;
- separate correctness, quality, capability, systems, scale, runtime/export, and interaction gates;
- a minimum 10% realized primary systems gain for simple components and 20% when a custom runtime is
  required;
- named RTX 3090 training, interactive, throughput, and 128K serving envelopes;
- frozen upstream revisions for RULER, NoLiMa, and HELMET, while explicitly recording that their
  integrations remain incomplete;
- a live evidence matrix containing retained, rejected, proposed, blocked, and research-only
  components, including a conventional dense global-attention reference.

The measured `0.00965`-nat seed range is not the new margin. It estimates variance for prospective
power and replication planning. The `0.01`-nat margin is a versioned engineering choice: a candidate
may cost at most about one percent in perplexity before its efficiency benefit becomes irrelevant to
this policy. Changing that choice requires a new policy version.

## Retrospective audit of the current frontier

The checked
[`retrospective-audit.json`](../results/Speck-Architecture-Promotion-v1/retrospective-audit.json)
applies the v1 loss calculation to existing results without granting them promotion authority:

| Comparison | Mean candidate penalty | Upper one-sided 95% bound | `0.01`-nat pass |
| --- | ---: | ---: | --- |
| Five-cache KDA/NoPE versus GDN/RoPE | `0.006909` | `0.014244` | no |
| Three-cache Reader versus five caches | `0.005783` | `0.015815` | no |

Both point estimates lie inside the old seed range, but both upper confidence bounds cross the new
non-inferiority margin. This is the operational difference between “unresolved” and “equivalent.”

## Statistical design

For loss, define each paired observation as candidate loss minus control loss under the same
initialization seed, packed-data order, token budget, sequence length, optimizer schedule, and
evaluation manifest. A candidate is non-inferior only when the upper one-sided 95% confidence bound
is no more than `0.01` nats. Per-source loss also has a `0.02`-nat guardrail.

Bounded benchmark metrics use example-level paired resampling where raw predictions exist. The
default score margin is two absolute percentage points, with a five-point guardrail on every critical
task and an independent absolute capability floor. Multiple candidates sharing a control are frozen
before confirmation and use Holm correction. Exploratory metrics remain reportable but have no
promotion authority.

Three paired runs are intentionally a confirmation screen, not final evidence. Before an architecture
freeze, finalists require a `3 × 2` crossed design over initialization seed and packed-data order and
at least ten training tokens per parameter unless a preregistered scaling-law analysis demands more.
A single 1.2B pair is only a costly reversal/integration sentinel; it cannot establish population-level
equivalence.

The paired, thresholded, one-sided design and prospective power rule follow NIST's engineering
statistics guidance:

- <https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm>
- <https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm>

## Cost design

Training efficiency is time and physical resources to a locked quality target, not tokens per second
alone. This follows the central MLPerf Training measurement principle:

- <https://mlcommons.org/benchmarks/training/>

Serving reports latency and throughput as a curve under a declared request shape and load. The v1
profiles use TTFT, TPOT/inter-token latency, maximum resident batch, prompt/output lengths, persistent
state, peak allocation, and output tokens per GPU-second. These metrics align with the public
MLCommons LLM methodology and the vLLM serving benchmark interface:

- <https://mlcommons.org/2024/03/mlperf-llama2-70b/>
- <https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md>

Dollar cost is derived only when a result records hardware amortization excluding energy, measured
power, and electricity price. Otherwise Speck reports time, energy, and memory separately rather than
inventing a portable dollar figure. Five thermally interleaved blocks are the minimum for paired
microbenchmarks; p99 serving claims require at least 1,000 online requests.

## Current consequences

1. Five-cache KDA/sigmoid/NoPE remains the conservative research control, not a promoted architecture.
2. Three-cache Reader Attention remains research-only. Its failed promotion result blocks downstream
   reader combinations even though the prerequisite experiment is administratively complete.
3. The next valid cache-compression comparison uses five independent memories: GQA3 versus MQA1 versus
   NoPE MLA.
4. Sparse long-context attention remains blocked on genuine dependency data and qualified evaluation
   integrations.
5. MoE remains blocked until the dense sequence architecture and medium-scale hardware envelope are
   stable.
6. No new multi-arm checkpoint family should launch with the research volume at 99% utilization.

## Deliberately incomplete work

- The 200-case `structured_retrieval_v2` and `symbolic_composition_v2` protocols are frozen, their
  answer/route vocabularies qualify against the pinned tokenizer, and both adaptation and multi-length
  runners enforce their exact path and SHA-256. A complete CPU preflight builds and hashes all 3,000
  factual/counterfactual/distractor cases at the strictest 4K geometry. The first protocol-bound GPU
  executions remain pending.
- RULERv1, NoLiMa, and HELMET source commits and required files are qualified. Data and execution are
  still blocked: RULER has unpinned transitive downloads and no supported Speck server; NoLiMa requires
  non-commercial license acceptance and a compatible endpoint; HELMET requires a separate 34GB volume
  and locked-environment model smoke.
- The medium-scale hardware, parallelism, and absolute cost envelope are not yet named.
- A production serving runtime profile is required in policy v2 before architecture freeze.

The validator rejects cross-file policy drift:

```bash
uv run --extra cpu python -m scripts.research_contract_validate \
  research/architecture-promotion-v1
```
