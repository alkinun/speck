# The MiniMax-M2 Series and the Return to Full Attention

- **Paper:** [arXiv:2605.26494](https://arxiv.org/pdf/2605.26494)
- **Version reviewed:** v2, 30 July 2026
- **Companion note:**
  [Why Did M2 End Up as a Full Attention Model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)
- **Primary topic:** negative hybrid evidence, agentic MoE, and evaluation methodology

## Central claim

MiniMax-M2 returns from MiniMax-01's 7:1 Lightning/full hybrid to GQA full attention in every layer.
MiniMax reports that efficient-attention variants looked competitive on standard small-scale benchmarks
but developed deficits in complex multi-hop reasoning and long-context agent tasks at larger scale. The
company chose quality and mature production behavior over the theoretical savings.

This is one of the most important negative results in the collection because it comes from a team that had
already trained and deployed a very large linear hybrid.

## Architecture

- 62 decoder blocks, hidden size 3,072, vocabulary 200,064.
- 229.9B total and 9.8B active parameters.
- Full attention in all layers with 48 query and eight KV heads, plus RoPE.
- 256 fine-grained experts, eight active, sigmoid routing with learned expert biases.
- One multi-token-prediction module during base training, expanded by weight copying to three during
  continued training for speculative decoding.
- 29.2T pretraining tokens and a native 192K context, progressively extended from 8K to 32K and
  ultimately 192K.

The broader report focuses on agent data, the Forge RL system, persistent interleaved reasoning, and
on-policy capability growth from M2 to M2.7. Those advances explain final agent quality and should not be
attributed to full attention alone.

## The negative attention result

The paper reports an M2-scale full-attention versus hybrid-SWA study:

| Metric | Full attention | Hybrid SWA |
| --- | ---: | ---: |
| HELMET ICL | 75.8 | 72.7 |
| RULER 128K CWE | 90.0 | 72.0 |
| RULER 128K MQ | 99.0 | 93.0 |
| MTOB Korean→English BLEURT | 60.0 | 45.0 |
| MTOB English→Korean ChrF | 44.8 | 27.2 |

At 32K, both variants score 99 on the listed RULER tasks. This is exactly the hidden-cliff pattern that
maximum-window or short-context averages miss.

After SFT, SWA is mixed at or below 32K—sometimes winning IFBench or shorter agent tasks—but is worse
on most long-horizon agent evaluations, including SWE-Verified, Terminal-Bench, BrowseComp-zh, and one
TauBench domain. The paper attributes this to restricted attention coverage.

The companion engineering note adds two details:

- retrieval and induction attention patterns form early during pretraining, so converting a checkpoint to
  hybrid SWA during continued pretraining cannot reliably relearn every necessary global head;
- linear/sparse serving still has open work in low-precision state storage, prefix caching, and speculative
  decoding. A theoretical crossover at a few thousand tokens does not guarantee a production crossover.

## Evaluation lesson

MiniMax's earlier MMLU, BBH, MATH, LongBench, and small-scale results did not predict the large-scale
multi-hop deficit. Once the failure was known, proxy metrics could be built, but their correlation at the
next scale and on new data remained uncertain. Architecture evaluation therefore needs capability probes
that are difficult enough to fail throughout scaling, plus repeated end-to-end checks after SFT and RL.

## What matters for Speck

Every hybrid promotion should require:

- multi-query and multi-hop retrieval at each proxy scale, not just next-token loss;
- evaluation beyond 32K before calling a local/global mix equivalent;
- from-scratch and conversion experiments kept separate;
- prefix-cache, low-precision state, and speculative-decode tests before claiming serving savings;
- short and long agent tasks after post-training, because pretraining parity may not survive SFT/RL.

The result does not prove all hybrids fail. Kimi Linear trains its hybrid from scratch and DeepSeek's
sparse models use different mechanisms. It proves that missing capability is easy to hide and costly to
discover late.

## Limitations and cautions

- The paper does not publish the full SWA architecture grid, token counts for every arm, or statistical
  intervals.
- The negative study is primarily hybrid SWA, not KDA/GDN plus periodic global attention.
- M2's final quality bundles vastly more than attention: data, MoE design, MTP, agent SFT, and large-scale
  RL dominate many headline scores.
- Full attention with GQA at 192K still has substantial cache and prefill cost; the decision optimizes the
  team's then-current quality/systems frontier, not an eternal architecture rule.

## Bottom line

MiniMax-M2 is Speck's release-gate warning. A hybrid should not ship because it matches loss, NIAH, or
32K RULER. It must preserve multi-hop reasoning after scale-up and post-training, and its promised
systems advantage must survive the production cache and precision stack.
