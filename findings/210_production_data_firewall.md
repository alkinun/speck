# 210 — The production evaluation firewall is materialized and remains closed to model training

The calibrated six-category construction produces five mutually content-disjoint destinations with
equal per-category targets: 600.08 MB tokenizer training, 60.04 MB tokenizer evaluation, 120.13 MB
selection held-out, 60.05 MB sealed D5 audit, and 120.03 MB sealed E2 audit. Each selection category
contains at least 10.00 MB from a technically qualified unseen source, exactly two declared source
groups, and one group absent from tokenizer training.

All declared input identity, target, content-disjointness, unseen-source, held-out-group, and separate
audit-identity gates pass. The tokenizer-training, tokenizer-evaluation, and selection consumer checks
reverify all 18 readable file hashes. A model-training consumer is rejected. All 12 audit files have
mode zero and remain unopened; the tracked result publishes only their hashes and ordered content
commitments, not text.

This closes R4 but does not make the eventual model-training corpus disjoint. Calibration-primary
firewall records were sampled from otherwise eligible source files. Final corpus preparation must give
every firewall record precedence and remove exact plus verified-near matches before packing and data
launch. Consumer-path denial is not a substitute for that removal receipt.

Artifact: [production firewall result](../results/data/production-data-firewall-20260910.json).
