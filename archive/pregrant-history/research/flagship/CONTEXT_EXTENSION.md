# Flagship context extension: data and training plan

Status: current approximate design, 2026-09-11. Coordination: [SPE-52](https://linear.app/openspecklabs/issue/SPE-52)
and [SPE-53](https://linear.app/openspecklabs/issue/SPE-53). This defines the intended recipe, not
prepared data or runnable experiments. Source weights and token counts are planning defaults; exact
manifests, batch, learning rate, and endpoints follow final-tokenizer data yield and GH200 measurements.
The context envelope is 200 GPU-hours inside P6. See [EXECUTION.md](EXECUTION.md) for budget boundaries.

## 1. Place in the training pipeline

```text
4K final base -> 32K continuation -> 128K continuation -> instruction post-training
```

Context extension is continued next-token pretraining on coherent documents, with ordinary language
modeling loss. It preserves the selected architecture and optimizer state. [Instruction post-training](POST_TRAINING.md)
then teaches the assistant to answer, extract, summarize, and follow instructions using that context.
The extension corpus and instruction datasets are separate products with a shared provenance system.

Use the final post-decay base as the default parent. Preserve the pre-decay checkpoint separately for
recovery and future continuation. Keep each completed length checkpoint so a later regression does not
erase an earlier usable model.

## 2. Candidate documents

Prefer sources already present in the flagship [source registry](source_registry_v2.json). Their pinned
revisions and existing source-use decisions are the starting identities; long-unit reconstruction and
length-specific quality still need checking. A source name does not establish a supply of 128K units.

| Document group | Primary candidates | How to use them |
| --- | --- | --- |
| Books and educational long prose | `common-pile/project_gutenberg_filtered`; `common-pile/pressbooks_filtered`; `common-pile/libretexts_filtered` | complete books or coherent chapters; preserve original order; group pages only with real book/chapter identity |
| Papers and technical documents | `allenai/peS2o` v3; `HuggingFaceFW/finepdfs-edu` English; `common-pile/arxiv_papers_filtered`; `common-pile/pubmed_filtered` | complete retained text, section order, equations/tables where available; reject broken extraction and abstract-only records |
| Repository code and documentation | restricted permissive `HuggingFaceCode/stack-v3-train`; repository-reconstructable `HuggingFaceTB/stack-edu` | coherent repository snapshots or subtrees with paths, README/docs, source and relevant tests |
| Long web and reference | long retained pages from the E1/E2-selected web treatment: FineWeb-Edu, DCLM, or Ultra-FineWeb | retain genuine long pages/manuals; connected pages require explicit provenance, not topical concatenation |
| General-domain replay | the frozen E2 stable mixture | replay all six categories at their selected weights inside the replay allocation |

Stack-Edu files without enough repository/version identity remain file-level material, not fabricated
repository context. Do not recover a repository by combining unrelated files with similar topics.
Use the original indexed text and metadata: packed shards alone cannot reliably reconstruct books,
paper versions, page order, or repository trees. Preserve these records before raw-file cleanup.

### Approximate source mixture

Shares below are **processed training tokens**, not document counts or assistant target tokens.
Each window belongs to one primary allocation; replay is a separate sampling stream even when the
same source also contributes long documents. Account for repeated text across streams and stages.

| Allocation | 32K stage | 128K stage |
| --- | ---: | ---: |
| Books / educational long prose | 25% | 35% |
| Papers / technical documents | 25% | 20% |
| Repository code / documentation | 25% | 25% |
| Long web / reference | 10% | 5% |
| General-domain replay | 15% | 15% |
| Total | 100% | 100% |

The 128K prior favors sources more likely to have coherent long units. Measure the joint source/length
yield before finalizing either mixture. If a source cannot supply its share, move it to a shorter view
and explicitly redistribute among qualified long sources before the stage begins. Do not pad its quota
with unrelated documents and call that long-context supervision. These are development choices, not
new source-mixture ablations or claimed optima.

## 3. Preparation pipeline

1. **Retain source structure.** Store document/work/repository ID, revision, source terms, section/file
   order, original length, and source hashes. Remove navigation, duplicate headers, generated/vendor
   files, and extraction artifacts while preserving useful structure.
2. **Build coherent units.** A unit is one book, paper, manual, or repository snapshot/subtree. Prefer
   complete units; for oversized units, take contiguous, structure-aware windows and record offsets.
   Repository ordering is deterministic: README/build context first where appropriate, then stable
   path/subtree order with explicit file separators. Dependency-aware ordering is optional, not assumed.
3. **Deduplicate and partition by family.** Group editions, paper versions, related pages, repository
   forks, and generated task variants. Exclude evaluation/audit material. Reuse of pretraining documents
   in extension is allowed and counted; reuse of held-out documents is not.
4. **Tokenize with the selected tokenizer.** Measure actual unit lengths and retained yield at 8K, 32K,
   64K, and 128K. Dataset character counts or a different tokenizer's length field are not sufficient.
5. **Make 32K and 128K views.** The main long stream uses windows within coherent units. Shorter units
   feed shorter views, replay, or later SFT. Preserve source and window indices next to packed tokens.
6. **Inspect and package.** Check a small source/length-stratified sample, document continuity, useful
   content across the sequence, and held-out separation; emit immutable stage manifests and hashes.

Aim initially to curate **roughly 2-4B unique retained tokens across the shared collection**, including
about **0.5-1B tokens available in coherent 128K-or-longer units**. These are preparation targets, not
measured availability. The length views may overlap; do not count them as separate unique corpora.
Track repeated windows and per-source exposure instead of silently cycling a small set of long books.
Prepare incrementally and stop acquiring excess material once the stage budget and coverage are met.

### Packing and replay semantics

The initial recipe uses one sequence length per stage. A main-stream window should remain inside one
coherent long unit; arbitrary concatenation does not satisfy this requirement. The 15% replay stream
may contain shorter base documents packed with their boundaries retained; its purpose is retention,
and it does not count toward the long-dependency coverage target.

The current loader consumes a flat stream and does not itself guarantee coherent-unit windows.
Document/window-aware sampling or equivalent boundary-aligned preprocessing is therefore required.
Do not describe BOS/EOS markers as recurrent-state resets or attention isolation. If independent
example packing/reset semantics are introduced, qualify both the recurrent and attention paths.
Actual interleaved 4K training steps are not assumed in this fixed-length recipe.

## 4. Two continuation runs

| Stage | Parent | Sequence | Initial token target | Planning range | Initial GPU-hour allowance |
| --- | --- | ---: | ---: | ---: | ---: |
| LC1 | final 4K base | 32,768 | 3B | 2-4B | 70 |
| LC2 | retained LC1 checkpoint | 131,072 | 1B | 0.5-1.5B | 110 |
| Engineering preflight / triggered repair | exact-shape qualification and an existing stage | stage-dependent | measured | within overall ceiling | 20 |
| Total | | | about 4B | not a guarantee | 200 |

Token targets are deliberately approximate and are not forecasts of what 200 hours can buy. Freeze
each endpoint using measured end-to-end throughput including startup, recomputation, saving, and
data delivery. Record input tokens, useful tokens, source exposures, elapsed time, and GPU-hours.
Unused hours stay unspent until there is a documented need within this continuation work.

Use BF16, the inherited Muon/AdamW optimizer state, a low learning rate, a short warmup, and smooth
decay. Exact LR and batch follow the frozen base recipe and numerical/memory qualification. Enable
activation checkpointing and efficient loss calculation where needed. The selected NoPE default needs
no RoPE scaling; if D2 instead selects partial RoPE, specify its extension settings before continuation.
Do not change the positional design or mixer topology as a late extension shortcut.

The repository supports DDP, not a qualified context-parallel trainer. Each GPU must fit its own
full training example; four GPUs do not pool memory for one 128K sequence. A successful exact-shape
forward/backward/resume rehearsal is required before LC2. If 128K cannot fit or learn adequately within
the envelope, preserve the qualified 32K model, document the failure, and revise the effective-context
claim. A new context-parallel implementation or extra tokens require their own costed decision.

## 5. Evaluation that guides continuation

Use a compact checkpoint dashboard rather than a new experiment matrix:

- original-4K held-out loss, source breakdown, and core short-context capabilities;
- position-binned and trailing-token loss on held-out coherent long documents;
- retrieval with distractors and varied evidence positions, using existing internal/RULER tooling;
- a small independent set of document comprehension/composition tasks;
- memory, prefill/decode cost, and training throughput at the actual sequence length.

Evaluate the parent, LC1, and LC2. Define acceptable retention and useful-context criteria before each
run; do not choose thresholds after observing results. A needle score alone does not establish useful
context. Existing flagship evaluation pays for benchmark runs from the separate 60-hour P6 quality
envelope; training and engineering checks are charged to the 200-hour context envelope once.

Recheck context after each [post-training](POST_TRAINING.md) stage and on exported weights. Report
trained, effective, and hardware-usable lengths separately. Later instruction tuning cannot silently
inherit the extended base's length claim if it regresses.

## 6. Practical readiness

Defined now: candidate sources, coherent-unit policy, two length stages, approximate mixtures and token
targets, replay, the 200-hour envelope, and milestone evaluation.

Still to materialize: actual indexed documents and length yield; the final tokenizer-bound views;
coherent-window sampling; exact parent/config hashes; measured 1.2B/32K and 1.2B/128K training cost;
LR/batch/endpoints; and the stage-specific retention criteria. The corpus is not marked prepared until
those artifacts exist. Source selection and reconstruction can start before final tokenization.

Technical reference: [docs/long_context.md](../../docs/long_context.md),
[docs/training.md](../../docs/training.md#progressive-context-continuation), and the existing
`scripts.context_stage_prepare` / `scripts.context_budget` tools.
