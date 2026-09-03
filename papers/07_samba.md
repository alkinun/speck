# Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

- **Paper:** [arXiv:2406.07522](https://arxiv.org/pdf/2406.07522)
- **Version reviewed:** v3, 28 February 2025; published at ICLR 2025
- **Code:** [microsoft/Samba](https://github.com/microsoft/Samba)
- **Primary topic:** Mamba plus sliding-window attention without a growing global cache

## Central claim

Samba repeats Mamba, MLP, sliding-window attention, and MLP layers to combine a fixed recurrent state
with exact access to the recent 2,048 tokens. Because it contains no global-attention layer, both compute
and resident sequence state remain linear or bounded as context grows.

The division of labor is explicit: Mamba compresses long history, SWA provides high-resolution recent
memory, and SwiGLU MLPs transform and store factual features. This is the cheapest serious hybrid arm in
the collection.

## Architecture

- The main SWA window is 2,048 with RoPE base 10,000 and FlashAttention-2.
- Mamba uses a kernel-4 short depthwise convolution, expansion `2 × d_model`, low-rank selection
  projection, state dimension 16, and a SiLU output gate.
- At about 1.7B parameters, the paper compares pure Mamba, Mamba+MLP, Mamba+SWA+MLP, Llama-style
  full attention, Mistral-style SWA, and Samba under a shared 230B-token recipe.
- Released/scaled models cover 421M, 1.3B, 1.7B, and 3.8B parameters; the largest sees 3.2T tokens.

## Evidence

On the matched 1.7B/230B-token study, Samba has the best reported average over 15 short-context tasks
(`54.33`), ahead of Mamba+SWA+MLP (`53.77`), pure Mamba (`52.31`), and the Llama/Mistral controls
(`51.17`/`51.12`). This supports the particular ordering and the presence of separate MLPs, although it
does not isolate every component with multiple seeds.

The long-context claims need to be separated carefully:

- A model pretrained at 4K shows decreasing Proof-Pile perplexity as evaluation context grows to 1M.
  This demonstrates use of additional context under next-token loss, not exact recall at 1M.
- After only 500 instruction-tuning steps at length 4K, Samba reports perfect single-passkey retrieval to
  256K. The pure SWA model cannot retrieve beyond its local range.
- On the harder multi-pair Phonebook task, Samba improves substantially and extrapolates better than the
  full-attention comparison beyond the training length, but it does not make fixed-state recall lossless.
- At a 128K prompt, the paper reports `3.73×` higher throughput than a GQA Transformer. When generating
  64K tokens in streaming mode it reports `3.64×` speedup.

## What matters for Speck

Samba should be a standing control whenever a more complex global layer is proposed. If a finite-state
mixer plus 2K–4K SWA gives sufficient real-task quality, it removes the entire context-growing global KV
cache. It is especially relevant for CPU and edge deployment where random access and cache bandwidth
dominate.

A fair Speck arm should keep total parameters, training tokens, position treatment, and local window
fixed while comparing:

- KDA/GDN + SWA only;
- the same model with one final global layer;
- the best two-placement global design;
- a 3:1 global hybrid.

Evaluate high-load MQAR and multi-hop document tasks. A single key can fit into a compressed state even
when many simultaneous associations or evidence chains cannot.

## Limitations and cautions

- “Unlimited context” is an asymptotic systems property, not unlimited effective memory.
- Perfect 256K passkey performance follows a short instruction-tuning run and a highly synthetic task.
- The 1M result is perplexity extrapolation, not 1M RULER, NoLiMa, or HELMET.
- The largest quality comparisons include different public baselines and training histories; the 1.7B
  architecture study is the cleaner evidence.
- The paper uses Mamba-1-era state geometry. GDN/KDA may change how much SWA is needed.

## Bottom line

Samba defines the minimum-state hybrid worth beating. It proves that local exact attention can repair
many recurrent weaknesses without a global cache, while leaving open the central Speck question: where
does compressed history fail on multi-hop and high-load retrieval?
