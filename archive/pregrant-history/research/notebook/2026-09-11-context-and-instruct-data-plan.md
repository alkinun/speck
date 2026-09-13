# 2026-09-11 — Consolidated context-extension and Instruct data plans

## Context

The owner selected Instruct for grant 1, asked for guided post-training rather than extensive
experiments, and requested clean documentation of approximate long-context and instruction datasets
and training plans. Existing context tooling had no complete flagship corpus or per-stage recipe.

## Work performed

Documented a coherent long-unit collection from registered books/educational prose, papers/technical
documents, repository code, and long web, with general-domain replay. Defined approximate 32K/128K
mixtures, 3B/1B initial token targets, a 200-hour envelope, preparation semantics, and hardware-dependent
settings. Explicitly separated coherent long supervision from packed short-document replay.

Created a concise current Instruct plan: broad SFT, high-quality mixed-length finishing SFT, and
preference tuning. Preserved the earlier proposal with a superseded banner; its grid and RL comparison
are historical discussion, not mandatory work. Added source/capability/length tables, approximate
data sizes, development evaluation cadence, and local-GPU generation accounting. Linked both plans
from the repository, compendium, flagship, and technical guides.

## Decisions

- Base + Instruct remains the owner-selected release shape; Think/dual mode is deferred.
- Use two continued-pretraining length stages followed by three Instruct-development stages.
- Plan around 200 context + 130 post-training + 60 evaluation + 60 serving = 450 GPU-hours.
- The 130-hour planning split replaces the old 80-hour annealing/merge plus 50-hour SFT intent.
- Preserve the active machine-readable predecessor and its 5,000/4,111/889-hour boundaries; its work
  list must be reconciled through an executable successor before stage launches.
- No source-yield, corpus-prepared, hardware-qualified, or model-quality result is asserted.

## Evidence and links

- [Context extension](../flagship/CONTEXT_EXTENSION.md)
- [Instruct post-training](../flagship/POST_TRAINING.md)
- [Dataset survey](../flagship/POST_TRAINING_DATA_SURVEY.md)
- [Source registry](../flagship/source_registry_v2.json)
- [Execution view](../flagship/EXECUTION.md)
- [SPE-52](https://linear.app/openspecklabs/issue/SPE-52)
- [SPE-53](https://linear.app/openspecklabs/issue/SPE-53)
- [SPE-176](https://linear.app/openspecklabs/issue/SPE-176)

## Open questions

- Actual coherent English 128K-unit yield, especially technical documents and repository snapshots.
- Exact per-stage batch, learning rate, memory, throughput, and affordable token endpoint on GH200.
- Final source-component acceptance, family-level overlap, and per-message SFT target support.

## Next actions

- Preserve document structure and index metadata during acquisition and cleanup.
- Prepare bounded source views and measure length/quality yield with the selected tokenizer.
- Materialize coherent-window sampling and source-specific instruction adapters.
- Create exact runnable stage and budget successors after data/hardware measurements.

## Pre-commit cleanup

Removed the full superseded proposal from the active flagship directory before its first commit.
The initial design notebook retains the discussion history, and links now resolve to the current
Instruct recipe. The source survey and its revision receipt remain separate supporting research.
