# SqueezeAttention: layer-wise KV budget allocation

- **Paper:** [arXiv:2404.04793](https://arxiv.org/pdf/2404.04793)
- **Version reviewed:** v2, 10 October 2024
- **Code:** [hetailang/SqueezeAttention](https://github.com/hetailang/SqueezeAttention)
- **Primary topic:** combine layer-wise budget allocation with sequence-wise KV eviction

## Importance signal and allocation

SqueezeAttention measures cosine similarity between each token's hidden representation immediately
before and after self-attention. High similarity is treated as low layer importance. It averages this
signal over prompt tokens, clusters layers into three K-means groups, protects a small group of special
layers, reduces the budget of the least-changing group by a factor `p`, and redistributes the saved slots
to the other groups. The selected sequence-wise eviction policy then runs independently at each layer's
new budget during decode.

The paper reports that first/last layers are often special and earlier layers often change
representations more, but the pattern varies by model and task. The method therefore derives allocation
from each prompt rather than fixing a universal layer pattern.

## Evidence boundary

Layer-importance observations use four models and 200 prompts. Accuracy/memory studies pair the
allocator with task-selected sequence compressors. Throughput uses Mistral-7B and Llama2-70B on eight
A100-40GB GPUs with 512+1,024 and 256+512 input/output shapes. Batch-one throughput is essentially tied
to full cache; reported gains grow because smaller caches allow much larger batches. Prefill cosine plus
clustering adds a reported 6.3% in one Mistral-7B 8K profile.

The paper's own limitation is structural: SqueezeAttention assumes an underlying sequence-wise eviction
policy can meet the task's accuracy tolerance. If that compressor fails, layer-wise reallocation may also
fail.

## What matters for Speck

Layer importance from before/after-attention cosine and joint layer/sequence budget allocation are not
novel. Both are required baselines for a role-grounded placement or state-budget law.

Speck N1 must predict from-scratch operator placement and multi-role failures beyond a prompt-specific
hidden-change clustering heuristic. It must also distinguish batch-one latency from capacity-enabled
throughput, because SqueezeAttention illustrates how memory gains can yield large saturation wins with
no batch-one speedup.

## Transfer cautions

- Cosine change is an association with layer importance, not a causal proof.
- The best sequence compressor is selected per task in parts of the evaluation.
- Full-cache comparisons and compressed baselines answer different cost questions.
- Reported multi-GPU throughput is shape-, batch-, model-, and hardware-specific.
- The method is post-training eviction, not native architecture or training evidence.

## Bottom line

SqueezeAttention is a direct layer-budget and hidden-change baseline. Speck must beat it predictively and
causally rather than rename layer roles or capacity-driven throughput.
