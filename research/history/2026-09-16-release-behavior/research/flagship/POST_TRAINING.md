# General-purpose instruction model with long-context strength

Selected [capability plan](capability_plan_v1.json): L3=120 hours for broad SFT, L4=160 for mixed-length
finishing, L5=60 for one optional justified intervention. L6=40 covers integration and stage diagnostics.
These are within the 700-hour capability ceiling, not additional to context continuation.

## External work handoff

Post-training experiments are underway in another codebase. This pivot does not claim to have audited
them. Before importing a recipe, reconcile code revision, tokenizer/chat format, loss masks, weights,
optimizer lineage, teacher/data identities, all attempted costs, evaluations and export compatibility.
Use that evidence to avoid duplicating experiments. No thinking-only response mode is frozen.

## Broad SFT

Use assistant-target loss on a diverse mixture covering ordinary assistance, writing, precise
instructions/structured output, math, code, document transformations and appropriate uncertainty.
Measure processed input-plus-target tokens separately from supervised tokens. Maintain short and
multi-turn tasks, meaningful system instructions, and all general-retention guardrails.

## Mixed-length finishing

Use the R2-supported standard or dependency-requiring training recipe. The starting structure keeps
70% broad general replay and 30% the controlled standard/targeted slice by assistant-target tokens.
Freeze actual 4K/32K/64K/128K processed-token shares and source/family exposures before each stage.
Length allocation must fit measured hardware and the inherited checkpoint's qualified context.

Targeted examples cover evidence combination, applying separated rules, explicit corrections,
conflicting evidence and missing evidence. Retain supporting text/turn IDs. Verify labels and use
held-out templates/source families. Long reasoning traces are not inherently higher quality; measure
correctness, answer grounding, output tokens, latency and termination together.

If targeted supervision fails the study's selection rule, L4 develops standard supervision and the
paper reports the boundary. A final-model development change is disclosed as development, never
retroactively described as the confirmatory intervention.

## Optional preference or distillation

Choose at most one method based on observed errors and existing evidence. Record teacher inference,
verification/rejection, student training and external costs. RL infrastructure, a multi-teacher fleet,
branch-merging search and a compulsory Think model are outside this budget. Unused L5 stays headroom
unless a costed phase-local change is frozen before affected outputs.

## Release gate

Retain model identities after every stage. Re-evaluate useful context, both primary workload families,
short/general capability, response length and exports. Failed retention keeps the earlier qualified
model. The release should be a broadly useful assistant with measured long-context strength; its
ability to generate long answers is not the headline.
