# Context capability stages

Context capability is central to the flagship. [capability_plan_v1.json](capability_plan_v1.json)
allocates L1=140 hours to 32K continuation and L2=180 to the 128K target/length fallback, within the
700-hour capability program. The former 200-hour total extension envelope is superseded.

## Training path

4K base -> coherent 32K continuation -> qualified 128K continuation (64K evaluation) -> broad SFT ->
mixed-length evidence/history instruction -> optional targeted preference/distillation -> final audit.
Retain each checkpoint. If 128K fails, use a qualified 64K or 32K parent; later stages cannot inherit an
unverified context claim.

Initial continuation targets remain approximately 3B processed tokens at 32K and 1B at 128K. They are
cost hypotheses; use exact-shape end-to-end measurements to freeze attainable endpoints before each
stage. Preserve optimizer state, use a low LR, short warmup and smooth decay. No late operator or
position change is an extension shortcut. The NoPE default does not require RoPE scaling.

## Data

Use coherent books/chapters, papers/manuals, histories and identifiable repository snapshots from
qualified sources; retain order and family metadata. Aim for a measured 2-4B unique long-unit pool
with 0.5-1B tokens available in 128K units. Missing sources stay in shorter bins; arbitrary concatenation
cannot fill a coherent-document quota. Mix roughly 15% broad general replay in natural-document
continuation as a pre-results starting point. Final length/domain weights follow measured supply and
are frozen before stage outputs, not described as optimized.

Boundary-aligned preprocessing or document-aware sampling is still required. Flat streams and BOS/EOS
alone do not isolate examples or reset recurrent state. Qualify any new packing/masking semantics.
The supervision treatment is separate from shared natural-document continuation; see [STUDY.md](STUDY.md).

## Fit and useful-length gates

R0 must test the actual 1.2B 128K forward/backward/resume. DDP replicates the model and does not pool
memory for one sequence. Failure means preserve the 32K path, or cost one 64K successor inside L2;
no new context-parallel trainer or million-token program is implied.

Before each expensive stage, require a passing preceding-length dashboard and frozen quality floors:
both document/history families, short/general retention, position/trailing loss, distractor retrieval,
composition, update handling and missing-evidence behavior. Floors are calibrated on disjoint pilots
before confirmation. If fit, quality or budget fails, retain the earlier checkpoint and report it.

L6 owns 40 hours of integration and stage diagnostics; V1 owns final benchmark and comparator scoring.
Report actual input/target tokens, sources/exposures, wall time, peak memory and failed attempts. Recheck
quality after SFT and on exported weights; advertise only passing measured useful length.
