# SnapKV: observation-window prompt-cache eviction

- **Paper:** [arXiv:2404.14469v2](https://arxiv.org/abs/2404.14469v2)
- **Version reviewed:** v2, 17 June 2024; NeurIPS 2024
- **Code:** [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV)
- **Primary topic:** post-prefill KV eviction using attention from the prompt's trailing observation
  window

## Mechanism

SnapKV splits a prompt into an older prefix and a trailing observation window. For each attention head,
it computes attention from observation queries to prefix keys, aggregates across the query dimension,
applies stride-one 1D pooling over prefix positions, and retains the highest-scoring prefix positions.
The selected prefix KV and the complete observation-window KV are concatenated for generation.

Pooling is intended to retain neighborhoods around high-attention features rather than isolated token
positions. The paper's pseudocode sums observation-query weights before pooling. It performs prompt
compression once after prefill and reports no further update during generation.

## Evidence boundary

The initial attention-pattern study uses filtered UltraChat prompts longer than 3K and responses longer
than 512. It reports that the final prompt window identifies positions similar to later generation
windows, while different instructions over the same document change which positions are important.
High hit rate across question placement is not the same as question-agnostic compression: the question
is still present somewhere in the prompt used to form the trailing observation queries.

Hyperparameters vary by evaluation. The LWM needle test uses a 1,024-token prompt cache, observation
window 16, and max-pool kernel 5. The four-model LongBench comparison uses cache sizes 1,024/2,048/4,096,
window 32, and kernel 7. Command-R uses cache 4,096, window 64, and kernel 13. These are transferred
settings, not one universal geometry.

On a single A100-80GB, the LWM system at 16K and batch two is reported around 3.6x faster in decode and
extends the OOM boundary from 16K to 131K (described as 8.2x memory efficiency). Those are post-hoc
inference-system results, not training savings or Speck hardware measurements. LongBench results are
mixed by task and capacity; pooling is ablated on synthetic LongEval-Lines rather than every task.

## What matters for Speck

Observation-window salience is a necessary strong baseline for all-required-source recall, but it is
query-conditioned whenever the instruction influences the prompt states used for scoring. Speck must
separate three modes: question-visible trailing-window compression, question-visible but question-
excluded scoring, and genuinely question-agnostic prefix compression before the question exists.

Budget accounting must count the always-retained observation window and selected prefix separately in
physical KV slots. Pooling order, reducer, padding, causal mask, selection ties, and duplicate positions
must be frozen. A dense-attention probe must be qualified independently because Speck's production fused
attention path does not expose weights.

## Bottom line

SnapKV occupies trailing-observation attention voting plus pooled token clustering for prompt eviction.
It does not justify copying a window or kernel size, treating question-visible results as prefix-cache
evidence, or changing Speck's fixed ring cache without correctness and realized-systems qualification.
