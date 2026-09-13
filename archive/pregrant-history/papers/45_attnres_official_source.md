# Attention Residuals official source and K3 implementation audit

## Pinned primary sources

- [Official Attention Residuals repository](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6)
  at commit `85e22310fe5ee860b4a023de312d791de8a5a5e6` contains the 21-page report and
  README. It contains no code or license file.
- [Attention Residuals](https://arxiv.org/abs/2603.15031) specifies Full and Block AttnRes. The
  official-repository PDF has SHA-256
  `e5831b0db1347606453b5176b0142115a18887b6a9c2e1d05a266d4805a26b2f`.
- [Kimi K3 official Hugging Face code](https://huggingface.co/moonshotai/Kimi-K3/blob/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/modeling_kimi_linear.py)
  at commit `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` contains a released Block
  AttnRes inference path under the Kimi K3 License.

## Report semantics

Full AttnRes treats the embedding and every completed residual sublayer output as separate values.
Each receiving sublayer has one learned width-sized pseudo-query. Keys are independently RMS-normalized
source values; scores are unscaled query-key dot products; a single softmax over depth weights the
unnormalized values. The pseudo-query must initialize to exactly zero, producing uniform initial source
weights.

Block AttnRes sums sublayer updates within a block. The token embedding is source zero. The first
sublayer of a new block attends only to the embedding and completed block sums; later sublayers also
receive the current block's partial sum. The final output attends over the completed blocks plus any
final partial block.

The report's pseudocode explicitly says block size counts attention and MLP sublayers, while its block
transition is evaluated at a Transformer-layer boundary using half that sublayer count. Therefore a
block cannot end between one logical block's attention and FFN. This constraint was missing from
Speck's v1 readiness design.

## Released K3 implementation

K3 sets `attn_res_block_size=12` over 93 decoder layers. Each decoder layer has separate projection and
RMSNorm objects before attention and before MoE/MLP; the model also has an output projection/norm. The
helper concatenates completed block sums with the current partial sum, computes key RMSNorm and scores
in float32, applies source-axis softmax, multiplies the unnormalized values, and casts back to the input
dtype.

At a decoder-layer index divisible by 12, the implementation moves the prior partial sum into the
completed-block tensor before adding the current attention update. This yields eight logical-layer
blocks across 93 layers and at most nine output sources including the embedding and final partial block.
It confirms that K3's configuration counts decoder layers, equivalent to 24 residual sublayers per full
block.

The helper itself is differentiable, but the released full K3 sparse-MoE forward rejects training. The
generic model initializer draws every linear weight from a normal distribution and contains no special
zeroing for AttnRes projections; loading the released checkpoint overwrites those values. Consequently,
the code is a checkpoint inference semantic reference, not evidence that a new training run will honor
the report's mandatory zero-query initialization. The release also provides no Full AttnRes path,
two-phase online-softmax implementation, pipeline cache, training parity test, or isolated test suite.

## Consequence for Speck's 40-module graph

Speck's 20 logical blocks still contain 40 ordered residual modules, so Full AttnRes has at most 40
sources at the last module. But eight blocks of five residual modules are invalid because every other
boundary falls between attention and FFN. Eight boundary-aligned blocks over 20 logical blocks use
logical boundaries `[0,2,5,7,10,12,15,17,20]`, giving alternating residual-module sizes 4 and 6.
The final maximum remains nine sources: embedding, seven completed blocks, and current partial sum.

The block-count successor must partition the 20 logical blocks first and multiply boundaries by two.
Four blocks have 10 residual modules each; eight have alternating 4/6; twelve have 2/4. Arbitrary
`floor(i*40/N)` module boundaries are not source faithful when they are odd.

## License and decision boundary

The standalone Attention Residuals repository has no license file and no code. Its equations and
pseudocode can define a clean-room contract, but there is no separately licensed upstream implementation
to reuse. K3's implementation is covered by the Kimi K3 License and may inform internal research under
its conditions.

This audit authorizes an append-only correction to the readiness design only. Sequence-parent
selection, a clean-room reference, zero-initialization proof, forward/backward/chunk/decode parity,
training, and promotion remain blocked.
