# 121 — Finalist control 0 operator interruption

## What happened

The first v2 dense control started at 03:33:58, emitted step-0 validation at 03:34:51 and one optimizer
step at 03:35:01, then was manually stopped at 03:35:10. The stop was an operator error: I incorrectly
believed the crossed-factor correction had been made after launch.

Commit timestamps prove the opposite. Analysis v2 was frozen at 03:10, automation v2 at 03:16, the v2
program at 03:17, and the live gate at 03:33—all before the service started. No analysis, threshold, or
configuration changed after launch.

## Preserved evidence

The observed dense values were step-0 validation 10.40856 and step-1 training loss 10.40802 after
65,536 tokens. They have no analysis authority. The git-ignored W&B directory and all five file hashes
are retained. Journal exit status is 143/SIGTERM. There was no model, hardware, data, numerical, or
analysis failure.

No checkpoint/result root was created: model, optimizer, metadata, timing, completion, normalized
result, target-lock, and final-analysis files all remain absent. Only a complete step-23,496 checkpoint
can enter the collector.

## Decision

This attempt cannot count as a result and cannot be silently retried. The frozen recovery rule requires
a preregistered restart of the identical cell from step 0, followed by a fresh live gate. All scientific
settings remain unchanged and training is false until that contract exists.

## Artifact

- [Failed attempt record](../results/Speck-Paper1/finalist-failed-attempt-control-pair0-v1.json)
