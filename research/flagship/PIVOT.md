# Selected pivot: efficient long-context intelligence

On 2026-09-15 the owner explicitly asked to pivot allocation, paper and model to the preceding
long-context vision, including substantial scope changes. [pivot_decision_v1.json](pivot_decision_v1.json)
records this decision. Strategic approval is complete; execution qualification remains separate.

## Changes

- General-purpose 1.2B model distinguished by document/history understanding, with short/general retention.
- One replicated architecture × dependency-supervision study, one focused transfer check and a measured
  flagship quality-cost frontier. Three new planned paper claims replace four broader claims.
- Research 1,236 -> 750 hours; context/instruction work 330 -> 700; final evaluation/serving 120 -> 236.
  Base stays 2,425, reserve stays 889, total stays 5,000.
- Base default 320B; 400B is a measured-fit stretch. Useful 32K first, 64K measured, 128K targeted.
- Context feasibility moves to R0. Corpus readiness is independently costed; paid bounded hardware
  tests do not wait for production supply, while scientific runs still require their own inputs.
- Native/Transformers plus one accelerated serving path required; CPU/GGUF optional.
- Public near-footprint/larger comparisons and simple retrieval alternatives enter the release evidence.

## Retired from this allocation

Old E1W/E1S/E2/E3/E4 source and mixture funnels; separate C0/D2/D3/D7/D8 component selection;
old D4/D6 schedule (bounded qualification moves to R0/R1); I1/I2/I3 assembled-allocation study;
generic scaling ladder and old S2. Their scientific claims are retired with them, not transferred
without evidence. The old 29-slot/192-hour first wave and preparation quotas no longer govern new work.

## Preserved

All measured results and historical JSON bytes; published models; frozen Mistral/token caches; accepted
source-use boundaries and exclusions; checkpoint/runtime evidence; negative results; and existing
finite source-preparation jobs. This edit does not stop or launch jobs, download sources, open tests,
change training code or send external messages.

Mutable predecessor documents/registries were copied byte-for-byte into
[the scope snapshot](../history/2026-09-15-allocation-thesis/manifest.json) with hashes and Git revision.
Old JSON contracts remain at their existing paths. Detailed source-preparation notes now carry a scope
banner; their numerical quotas describe the former proposal until rebound under [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md).

The earlier `DATA.md` is itself bound by a checked capacity result and remains byte-identical at its
original path. [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md) is its active prose successor.

## Next concrete work

Use [FIRST_WAVE.md](FIRST_WAVE.md) and [PREGRANT.md](PREGRANT.md): reconcile retained artifacts, prepare
coherent-unit/task pilots, qualify hardware, cost and freeze the paired study, then materialize new
launch inputs. Readiness is recorded honestly in [status](../status.json); selecting a design is not
reporting that its experiments or tooling already ran.
