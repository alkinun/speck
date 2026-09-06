# 128 — HELMET model-judge reproducibility and data boundary

## Exact runtime correction

The pinned NarrativeQA and summarization scripts instantiate `gpt-4o-2024-05-13` at temperature
0.1. Although the earlier runtime inventory recorded no explicit seed at those constructor sites,
`OpenAIModel` defaults to seed 42 and includes it in every Batch Chat Completions request. This finding
corrects that field without mutating the historical audit.

Seed does not qualify reproducibility. Official OpenAI documentation now lists the snapshot as
deprecated and describes Chat Completions seed behavior as beta, deprecated, best-effort, and not
guaranteed. `system_fingerprint` is returned by HELMET's adapter, but neither judge preserves it in the
scored JSON or rejects mixed fingerprints. Each score is sampled once with an unpinned SDK.

## Denominator and schema failures

NarrativeQA excludes absent/unparseable outputs from its mean, while summarization drops an example if
any of its three judge calls fails. Neither path freezes the successful denominator or validates JSON
keys, numeric types, rubric ranges, or recall/precision denominators. Long-QA also omits raw judge
reasoning from its scored JSON. Thus identical model outputs can yield an incomparable aggregate after
backend, parse, or availability changes.

## Data-handling boundary

The workflow uploads full questions, reference answers or summaries, key points, and candidate outputs
as a Batch API input file. Official data controls say API content is not used for training unless the
customer opts in, but default abuse-monitoring retention can be 30 days; `/v1/files` and `/v1/batches`
retain application state until deletion and are not ZDR-eligible. HELMET never deletes the remote file
or batch. Dataset transmission authority, project controls, region, deletion, and cost are unqualified.

## Decision

Historical source identity and the corrected seed path are qualified; reproducibility, fixed-sample
inference, data handling, cost, and any replacement are not. No API request, data upload, newer-model
substitution, local-judge substitution, candidate scoring, or manifest change is authorized. A future
protocol must retain every raw response/fingerprint, hard-fail denominator/schema drift, repeat the
judge, and calibrate against a released blinded human set.

## Artifact

- [Model-judge readiness](../results/Speck-Architecture-Promotion-v1/helmet-model-judge-readiness.json)

Validate it offline with:

```bash
python -m scripts.helmet_model_judge_validate \
  --readiness results/Speck-Architecture-Promotion-v1/helmet-model-judge-readiness.json \
  --runtime-protocol research/architecture-promotion-v1/helmet_runtime_dependencies_v1.json \
  --runtime-audit results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json \
  --helmet-contract research/architecture-promotion-v1/external/helmet.json
```
