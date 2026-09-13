# Kimi K3 official release: Stable LatentMoE source audit

## Pinned primary sources

- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3/tree/3cb39dfd32e51c3328e2e4b4af21341247d06c43)
  at commit `3cb39dfd32e51c3328e2e4b4af21341247d06c43` contains the technical report,
  README, assets, and Kimi K3 License.
- [Kimi K3 technical report](https://arxiv.org/abs/2607.24653) specifies the architecture and
  training algorithms. The official-repository PDF has SHA-256
  `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c`.
- [Official Hugging Face release](https://huggingface.co/moonshotai/Kimi-K3/tree/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721)
  at commit `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` provides immutable config and
  custom inference code.

## Specification now available

The report makes Stable LatentMoE implementation-specific enough for clean-room reference work.
For a full-width token `x`, routed experts receive one shared latent projection `W_down x`; their
raw-score-weighted top-k aggregate is RMS-normalized once and mapped back by `W_up`. Full-width shared
experts process `x` directly and are summed with the routed path. K3 uses two shared experts, a 3,584-
wide routed latent, 896 routed experts, and top-16 activation at hidden width 7,168. Those production
dimensions are evidence about K3, not a Speck optimum.

SiTU-GLU smoothly caps both the gate's linear factor and the up branch. The report fixes gate cap
`beta_1=4` and up cap `beta_2=25`, proves first-order agreement with SwiGLU near zero, recovers SwiGLU
as both caps grow, and bounds each output coordinate by 100.

Quantile Balancing uses sigmoid router scores. Selection uses raw score plus expert bias, while mixture
weights use only selected raw scores normalized by their sum. A top-(k+1) biased cutoff yields each
token threshold. The next expert bias is the negative `(1-k/n)` quantile of raw-score-minus-cutoff
margins, mean-centered, and applied only to the next step.

The production estimator histograms required biases over the full global training step. It uses a
per-step range derived from current bias extrema, 1,000 uniform bins, microbatch-local accumulation,
one integer all-reduce, target rank `m*k/n`, within-bin interpolation, and an error bounded by one bin
width. The report describes an EMA as a possible refinement, not an unambiguous required K3 setting;
Speck must not silently add it.

## What the official code does and does not provide

The pinned Hugging Face code implements SiTU-GLU and the inference path for sigmoid routing, latent
down-projection, routed experts, post-aggregate RMSNorm, latent up-projection, and shared experts. The
released config pins the dimensions and activation caps above.

It is not a training reference. `KimiSparseMoeBlock` explicitly raises `NotImplementedError` in
training mode, and the released gate has no Quantile Balancing or histogram update. Its correction
bias is loaded for inference. Consequently, official inference semantics may inform a local reference,
but training dispatch, QB state/update, accumulation, distributed reduction, backward behavior, and
rescue policy still require local implementation and qualification.

## License boundary

The Kimi K3 License permits use, copying, modification, distribution, and derivative work subject to
notice, legal-compliance, large Model-as-a-Service commercial-agreement, and very-large-product
attribution conditions. Speck's current internal research use is allowed by the text, but any future
distribution or commercial deployment must re-evaluate the applicable conditions.

## Speck decision

Primary specification and inference-shape uncertainty are closed. CPU clean-room references for
normalized LatentMoE, SiTU-GLU, exact QB, and a histogram estimator are now authorized as isolated
primitives. Conventional MoE remains first in the causal order. No upstream training code is available,
no local training integration is authorized, and expert-parallel hardware/parent/resource blockers
remain.
