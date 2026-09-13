# MiniCPM5-2B release and tiered-data audit

## Sources

- Final model card and files: [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B),
  revision `62b9b3bd4308e72905c5bce38c1d6689549c525d`.
- Intermediate releases: Base `a85398542aea7c5cb424cd9b6a5b2bcb6d858b4e`, Midtrain
  `8dc5f6055b90fe4b9422340810b270b9569f37f3`, and SFT
  `54f7a31935a03431d5d7ad75ea8ce1a68c95877c`.
- Wang et al., [*Data Science and Technology Towards AGI Part I: Tiered Data
  Management*](https://arxiv.org/abs/2602.09003), arXiv v1.
- [Artificial Analysis release evaluation](https://artificialanalysis.ai/articles/openbmb-releases-minicpm5-2b),
  September 7, 2026.

## Reported system

The released config is a standard dense `LlamaForCausalLM`: 42 layers, hidden size 2,048, SwiGLU
intermediate size 6,144, 16 query heads, two KV heads, head dimension 128, 131,072 configured context,
RoPE theta 5,000,000, BF16, and no RoPE scaling. It has 2,516,756,480 total parameters but only
1,981,982,720 non-embedding parameters. Its 130,560-row untied input/output vocabulary accounts for
the exact 534,773,760-parameter difference. The weights and repository are Apache-2.0.

OpenBMB releases Base, Midtrain, SFT, and final RL+OPD checkpoints plus BF16, GGUF, MLX, GPTQ, LiteRT,
and speculative-decoding variants. The final card reports 400B deep-thinking SFT tokens, specialist RL
teachers for several domains, and on-policy distillation of 16 experts (five agentic). It attributes
10.96 average reasoning/general points and 6.96 agentic points to RL+OPD, but does not provide enough
training accounting on the card to transplant these gains or costs. The model card describes
full-vocabulary reverse KL, while the current repository prose describes a top-k union approximation;
that implementation boundary requires a pinned code audit before reproduction.

The tiered-data paper tests a controlled 1.2B model trained for 120B tokens under the same domain
distribution. Sequential 40B-token L1→L2→L3 stages score 31.66 versus 30.17 for a one-stage equal
mixture, a reported +1.49 points. Later-stage gains are strongest in reasoning, math, and code, while
several broad commonsense tasks decline slightly. Its efficient verification recipe uses a 10B-token
two-stage anneal, 30% candidate data, WSD, sequence length 4,096, weight decay 0.1, clipping 1.0, and
global batch 512 sequences.

OpenBMB's vendor table reports a 53.9 average over its chosen suite. Independent Artificial Analysis
reports Intelligence Index 15, highest among measured open-weight models below 4B total parameters,
but identifies weaker knowledge, coding-terminal, and some long-context results. Its non-hallucination
score is aided by attempting only 29% of omniscience questions. Reported output use is comparatively
efficient at 19K tokens per task. These are release-level observations, not controlled architecture
evidence.

## Speck interpretation

1. The strongest transferable evidence supports Speck's existing E2/E4 thesis: preserve broad L1-like
   coverage early, concentrate selected/refined data late, and retain category guardrails for breadth.
2. Base→Midtrain→SFT→Final checkpoint publication makes stage attribution inspectable. Speck should
   retain and release equivalent milestone identities even when only one final model is promoted.
3. The large post-training stack plausibly explains much of the viral final-model capability. Do not
   compare its final chat benchmark scores directly with Speck base-pretraining architecture arms.
4. The 130K untied vocabulary consumes 21.25% of total parameters. This reinforces Speck's tied-head,
   explicit tokenizer-cost accounting rather than motivating a larger vocabulary.
5. Aggressive GQA and a standard Llama runtime make 128K deployment broadly accessible. Speck's hybrid
   must earn its added runtime complexity through measured state/quality gains and export parity.
6. Tool/agent evaluations, output-token cost, abstention, and repetition/runaway rates belong in the
   released-system table. A single benchmark average is insufficient.
7. OPD, specialist RL, DSpark, and architectural changes remain outside grant 1. The release is a named
   comparator and post-training lesson, not permission to reopen the fixed mechanism scope.
