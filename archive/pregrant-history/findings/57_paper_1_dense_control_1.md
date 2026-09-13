# 57 — Paper 1 dense control 1

## Run

The second new Paper 1 dense control completed the same fixed 2,000 optimizer steps and 131,072,000
training tokens. This cell uses initialization seed 43 and the disjoint packed-data window beginning at
token 536,870,912.

The final 19,988,480-token validation loss is 2.839127 nats. All six validation boundaries and all
eleven source losses are retained, and no non-finite step occurred. This result is 0.005481 nats above
control 0, but the cells differ in both initialization and data order; the difference is a descriptive
control-variation observation, not an architecture comparison.

Steady training took 3,260.22 seconds, within 0.25 seconds of control 0. Optimizer time was 3,284.89
seconds, evaluation took 353.02 seconds, and peak allocated device memory was 12,340,162,048 bytes.
The lower 24.67-second startup optimizer overhead versus control 0's 115.15 seconds is consistent with
reuse of the compiled CUDA environment and must not be folded into steady training time.

## Decision

Dense control 1 is complete and qualified. Two of three controls now exist; neither the
time-to-quality target nor any candidate record may be created until dense control 2 is also complete
and qualified.

## Artifact

- [Qualified control result](../results/Speck-Paper1/runs/Speck-Paper1-Baselines-131M-pair-1-seed-43-order-536870912-dense_global_param_match.json)
