# Related work for the long-context pivot

Reviewed 2026-09-15 from primary reports and official repositories during the strategy discussion.
This is a focused positioning map, not exhaustive novelty clearance or local reproduction.
Experiments still require immutable release/scorer/data manifests.

| Source | Established territory | Required Speck distinction |
| --- | --- | --- |
| [Olmo Hybrid](https://arxiv.org/abs/2604.03444) | Controlled hybrid/dense comparisons and scaling evidence | Hybrid efficiency or scale transfer itself is not new |
| [Token-level hybrid comparison](https://arxiv.org/abs/2606.20936) | Architecture advantages vary across content and repeated tokens | Aggregate loss alone cannot establish evidence use |
| [Kimi Linear](https://arxiv.org/abs/2510.26692) | KDA, 3:1 hybrid and controlled comparisons | Operator and ratio are inherited; measure local costs |
| [ProLong](https://github.com/princeton-nlp/ProLong) | Long-context continuation recipes | Longer documents alone are not the new intervention |
| [LongPO](https://github.com/DAMO-NLP-SG/LongPO) | Short-to-long preference optimization | Targeted post-training alone is not novelty |
| [LongReward](https://arxiv.org/abs/2410.21252) | AI-feedback-based context improvement | No first claim for feedback improving long context |
| [HELMET](https://arxiv.org/abs/2410.02694) | Diverse applications and limits of needle tests | Include natural applications beyond retrieval |
| [LongBench Pro](https://arxiv.org/abs/2601.02872) | Natural tasks and effective context distinctions | Report task and length scope |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | History reasoning, updates and abstention | Pin exact version and use broad history skills |

The official [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) advertises later
updates, so the name alone does not identify the payload. Qualified subsets must be labeled.

## Candidate question and remaining review

Does dependency-requiring supervision have different effects under dense versus recurrent/global
memory, and does this relationship yield useful task-quality versus state/latency operating points?
Three paired seed blocks and one transfer check can bound this question; they cannot establish a
universal scaling law or full-horizon 1.2B dense equivalence.

Before R2, review contemporary task-generation/RL work, including
[LoongRL](https://arxiv.org/abs/2510.19363) and
[Beyond Reward Engineering](https://arxiv.org/abs/2606.18831), for direct overlap. These two are
screened references, not fully audited methods here. Complete a closest-work table covering the
intervention, controls, resources, model/horizon and conclusion. Narrow or revise the hypothesis
before confirmation if it is already answered. No novelty finding is claimed by this plan.

## Practical baselines

[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B),
[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),
[LFM2.5](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai), and
[SmolLM3](https://huggingface.co/blog/smollm3) motivate strong public comparisons. Their teacher,
token and post-training investments differ; deployed-system comparisons do not isolate architecture.
Retrieval plus a short-context answerer is a practical baseline. An advantage for full-context
processing on dispersed evidence is a hypothesis, not an assumed result.
