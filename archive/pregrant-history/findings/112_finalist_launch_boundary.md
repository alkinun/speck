# 112 — Multi-day finalist launch boundary

## Release-gate separation

As frozen for the proxy in Finding 55, RULER/NoLiMa/HELMET consume a trained checkpoint. Their missing
or failed status must remain a failed promotion/release gate, but requiring their execution before
checkpoint production would be circular. The finalist boundary preserves that separation without
weakening any final evidence requirement.

## Fixed execution

The order is all six dense controls, a control-only target lock, all six candidates, then one twelve-cell
analysis. Continuation may use only one transient training service and one final-summary path trigger at
a time, with 15-minute gaps, no polling, no quality branch, no automatic retry, and a commit after every
qualified result. Any failure stops the chain and retains evidence.

Linear extrapolation from the three-pair proxy gives 10.63 steady GPU-hours per dense run and 9.57 per
candidate: 121.23 total steady GPU-hours, or about 5.05 days before validation/checkpoint/cooldown
overhead. The planning contingency is seven elapsed days. No energy or monetary number is inferred
without measurement.

Every launch must recheck a clean validated repository, absent unfinished outputs, idle/cool RTX 3090,
12 GiB host memory, the named ext4 device with 25.77 GB free, and an inactive HELMET transfer. Final-
only checkpoint retention means most interrupted runs have no recovery point; no cell may be skipped.

## Decision

Automation implementation and live qualification are authorized, but the initial launch is not: the
automation source/hash does not exist yet and HELMET acquisition is active. No attribution, promotion,
novelty, release, or paper-scale authority follows.

## Artifact

- [Finalist launch v1](../research/paper-1/finalist_launch_v1.json)
