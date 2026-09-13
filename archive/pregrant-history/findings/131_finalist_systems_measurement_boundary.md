# 131 — Finalist systems-measurement and temporal-confounding boundary

## What the finalist preserves

The frozen trainer records analytic FLOPs/token, parameters, CUDA-synchronized optimizer and steady
training seconds, active/evaluation/checkpoint/startup time, five validation-time points, final
checkpoint hashes, and peak `torch.cuda` allocation. Steady time excludes the first ten startup steps
and validation/checkpoint intervals. Every launch requires the same RTX 3090 UUID, an idle GPU, and a
start temperature at most 50°C.

## What it does not preserve

There is no board-power or energy sampler, idle-power baseline, temperature/clock/power-limit/P-state/
throttle/fan trajectory, host-load trace, or calendar timestamp per timing interval. Successor launches
enforce the live gate but discard its returned values instead of writing per-run gate artifacts.
Energy, carbon, and monetary cost are therefore unmeasured.

## Block-order confounding

All six controls run before all six candidates because the target must be locked from controls first.
Across a forecast 121.23 steady GPU hours, architecture is consequently aliased with calendar block.
Pairing seeds and data orders protects language comparisons but cannot remove ambient, boost-clock,
throttling, display/background, host-load, or multi-day runtime drift from timing comparisons. Same GPU
and start-temperature gates reduce this risk; they do not eliminate it.

The finalist can report per-run physical timing, peak allocation, and order-stratified secondary
summaries descriptively. It cannot support causal architecture-only speedup, energy-to-quality,
dollars, production serving, hardware-normalized performance, or systems-based promotion.

## Next systems gate

After the language sequence, freeze new systems-only measurements with at least five thermally
interleaved randomized/balanced AB/BA blocks, 1 Hz power/thermal/clock telemetry, matched idle blocks,
missing-sample bounds, host/background accounting, and per-block live-gate artifacts. Do not inject
telemetry or change ordering mid-sequence.

## Artifact

- [Systems measurement audit](../results/Speck-Paper1/finalist-systems-measurement-audit-v1.json)

Validate it offline with:

```bash
python -m scripts.paper_finalist_systems_audit_validate \
  results/Speck-Paper1/finalist-systems-measurement-audit-v1.json
```
