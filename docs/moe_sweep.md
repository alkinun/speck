# SpeckLabs 1B-token MoE sweep

This sweep compares `SpeckLabs-1B-D0`, `SpeckLabs-1B-M1`,
`SpeckLabs-1B-M2`, and `SpeckLabs-1B-M3` on one fixed packed DCLM-Edu
stream. It intentionally excludes LLAL, external validation corpora, 2B-token
runs, FineWeb-Edu, and Ultra-FineWeb.

## Fixed inputs and schedule

All arms inherit byte-identical data, tokenizer, and training recipes from
`experiments/SpeckLabs-1B-shared`. The DCLM-Edu revision is pinned, rows must
be English with `edu_int_score >= 3`, and global exact deduplication occurs
before deterministic packing. Preparation requests 1B training tokens and
holds out 20M validation tokens. Preparing any one arm publishes the same
manifest and token order used by every arm.

The global batch is 65,536 tokens at sequence length 2,048. The 1B request is
15,259 optimizer steps and 1,000,013,824 actual tokens. Muon uses learning rate
0.0015, cosine decay to 5%, 102 warmup steps, weight decay 0.1, and gradient
clipping at 1.0. Validation runs every 1,526 steps and at the requested 50M,
500M, and 1B milestones.

RTX 3090 qualification selected device batches 16/8/8/2 for D0/M1/M2/M3,
giving accumulation counts 2/4/4/16. The measurements and rejected candidates
are recorded in `experiments/SpeckLabs-1B-shared/qualification.json`.

## Run sequence

1. Prepare the corpus once:

   ```bash
   uv run --group dataset-build python -m scripts.data_prepare experiments/SpeckLabs-1B-D0
   ```

2. Qualify every arm at the 50M requested-token boundary:

   ```bash
   for arm in D0 M1 M2 M3; do
     uv run --extra gpu python -m scripts.base_train \
       "experiments/SpeckLabs-1B-${arm}" --stop-at-tokens 50000000
   done
   ```

3. Before acceptance, inspect finite total/LM/auxiliary losses, router entropy
   and utilization, zero-load experts, per-expert weight/gradient norms,
   throughput, peak reserved VRAM, optimizer-role metadata, and the loader
   cursor. Reload step 763 and verify that the next batch and learning rate are
   identical to an uninterrupted replay.

4. If code, architecture, data, tokenizer, or training recipe changes, archive
   the four checkpoint directories and restart every arm from step zero. Do not
   mix pre-change and post-change arms. If all checks pass without changes,
   resume each accepted run:

   ```bash
   for arm in D0 M1 M2 M3; do
     uv run --extra gpu python -m scripts.base_train \
       "experiments/SpeckLabs-1B-${arm}" --resume 763
   done
   ```

5. At completed steps 7,630 (500M requested) and 15,259 (1B requested), export
   locally, require native/Transformers logits and parameter-count parity, then
   run the pinned Open SLM stages and routed-expert masking analysis. Keep each
   model/step in its own output directory.

The implementation qualification uses only synthetic tokens. These commands
are the user-run workflow; repository verification does not download the full
corpus or launch any 1B-token arm.
