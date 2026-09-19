# SpeckLabs: current decisions and next work

Updated 2026-09-19. This is the status and work order for the first flagship program.
Read the [program overview](docs/program.md) for the connected design, data, compute and release
outline. [Main-data plan.json](experiments/main-data/plan.json) owns working numeric targets;
experiment configurations and verified receipts own actual run settings and measured results.
Update these in place. Historical proposals and failures remain in Git and their original receipts.

## Goal and selected decisions

Pretrain SpeckLabs' own useful, competitive and efficient model from scratch, then release the
base and an always-thinking assistant with an open technical report. Primary uses are agentic
coding, normal coding, math reasoning and tools; general usefulness remains a regression check.
External models are comparators or qualified teachers, not our base initialization.

- **Model selected:** 1,195,884,576 total/active parameters; 24 layers, width 2048; three KDA
  recurrent blocks then one global NoPE GQA block, repeated six times. Dense SwiGLU throughout,
  intermediate width 5120, tied embeddings, sigmoid KDA gates. No MoE investigation or model sweep.
- **Tokenizer fixed:** Mistral 32K; 32,003 embedding rows include three assistant role IDs.
- **Compute envelope confirmed by the user:** 5,000 total GPU-hours across four GH200s, equivalent
  to 1,250 hours with all four allocated. Site/access and GH200 throughput remain unconfirmed.
- **Behavior selected:** one always-thinking protocol with brief/deep reasoning and tool actions;
  no supported non-thinking response mode. Preserve separate base and assistant checkpoints.
- **Context target:** start at 4K, extend toward approximately 128K after measured qualification.
- **Future ambition:** 50,000 GH200-hours and larger models; no future allocation is assumed funded.

The [architecture decision](docs/architecture-program.md) fixes one model. Remaining experiments
concern implementation, data and training recipes. Report a measured need for a structural departure
before changing that choice. Competitive quality and useful 128K context are targets, not results.

## Working recipes, not launch settings

| Area | Preparation target | Still to resolve |
| --- | --- | --- |
| Base training | 320B desired / 400B stretch; 35% code, 25% math, 40% supporting material | Qualified source supply and measured cost; 100B remains the present budget-fit scenario |
| Context extension | Proposed 8B tokens: 2B up to 16K, 2B up to 32K, 4B toward 128K, including short replay | Memory, useful-context learning, positional behavior and stage costs |
| Thinking SFT | 1.5M unique qualified conversations, within a 1–2M range | Correctness, source-family deduplication, complete long examples and supervised/context token totals |
| Reward training | Conditional verifiable math/code rewards after useful SFT | Trainer, rollout integration, verifiers, recovery and affordable measured benefit |
| Data comparison | One bounded natural-code versus checked-exercise comparison | Useful common base, enough qualified examples, frozen endpoints and all-in cost |

The [main data plan](experiments/main-data/README.md) specifies candidate sources and counting rules.
Retained stock, admitted data and training exposure are different quantities. No small preview or
publisher quality label establishes flagship-scale supply. No automatic repetition fills a gap.

## Completed evidence

| Work | What is established | Limits / evidence |
| --- | --- | --- |
| Runtime rehearsal | Full-size single-H100 base/SFT restart, loader/RNG recovery, numerical checks, generation and export | [H100 receipt](experiments/qualification/h100-result.json); ARM64 GH200, four-worker execution and scheduler recovery remain open |
| Engineering pilot | 800 steps / 104,857,600 tokens; 2.265 trainer hours, 12,859 full-trainer tokens/s, 13,595 steady optimizer tokens/s, 19.3 GiB peak allocated memory | [Execution receipt](experiments/pilot/h100-run.json); engineering evidence, not an optimized ceiling |
| Development scoring | 2,619 tasks in 2.12 evaluation hours: GSM8K strict 0/253, compiled code pass@1 0/33, IFEval strict prompts 12/101, ARC normalized 44/222, HellaSwag normalized 530/2,010 | [Results](experiments/pilot/development-result.json); weak base capability, custom subsets, no final-partition scoring |
| Backup closeout | Eight model/optimizer checkpoints, both exports, evaluation outputs and recovery logs retained locally | [Backup receipt](experiments/pilot/backup-result.json); pilot closed, no further rental work needed; provider billing/stop state not verified |
| Natural-code audit | 16 exact upstream matches, 13 with license-notice evidence; ten files clear preliminary lineage/syntax/benchmark checks | [Cohort receipt](experiments/corpus-audit/natural-code-cohort.json); zero new training admissions |
| Other data preparation | 6.799B retained source tokens before joint eligibility; 500K assistant rows inventoried; finite tool-aware SFT rehearsal | [Supply](experiments/pilot/supply.json), [assistant contract](docs/assistant.md); not main-run qualified supply |

The [timing measurements](experiments/qualification/timing-result.json) additionally cover saves,
validation, restart, SFT and prefill/decode. Microbatch four showed about 20% higher steady throughput
in a short probe; sustained operation and recovery need qualification before adoption.
SFT padded positions/s are not supervised tokens/s. Earlier estimates and failed attempts remain
historical; do not repeat the completed pilot because a preparation document still contains commands.

## Immediate order of work

1. **Finish the bounded practical-code checks on CPU.** Independently test the exception-handling,
   dice-parsing and entropy candidates in the existing sandbox. Pin dependencies, retain notices
   and parent identities. Freeze broader coding-benchmark exclusions and family splits before
   exercise derivation. The 16-row UltraData-Code L3 preview remains held for unresolved lineage.
2. **Qualify the data recipe.** Audit pinned natural Ultra-FineWeb against FineWeb-Edu; keep DCLM
   as an independent coverage candidate and synthetic L3 separate. Review math correctness and
   source overlap, then assistant reasoning/tool outcomes and missing long-example tails. Work
   one bounded packet at a time; close code supply feasibility before bulk packing.
3. **Prepare the single-model GH200 packet before access.** Bind current source and exact inputs,
   checks, workload sizes, measurements and stop conditions. Rebuild the historical transfer bundle;
   do not treat its old source commit as the current release. No paid run starts from this outline.
4. **On access, qualify one worker then four.** Measure actual topology/ARM64 dependencies, kernels,
   batching, optimizer, communication, sustained throughput, restart/export and scheduler behavior.
   Declare every allocated GPU, including idle devices. Use results to freeze the main schedule,
   batch, token horizon, data manifests and budget. Resolve the 320–400B feasibility gap explicitly.
5. **Train, extend context, post-train, evaluate and release through gates.** Preserve continuation
   checkpoints and source-wise learning curves. Run the bounded data comparison only when its base,
   supply and cost gates pass. Qualify context stages before long SFT. Promote reward training only
   for measured gains. Pin final endpoints/comparators before selection; keep final tests untouched.

Runtime and data preparation are the active tracks. Long-context and assistant designs can be
prepared now; their expensive execution depends on a useful base and measured costs. Use milestone
checks and bounded supervisors, not frequent manual progress polling or duplicate workers.

## Compute

The [numeric plan](experiments/main-data/plan.json) reserves 70 GPU-hours for runtime qualification,
50 for engineering pilot work, 91 for the bounded data comparison, 2,300 for base training, 800 for
context extension, 800 for post-training and 889 for protected evaluation/recovery: **5,000 total**.
The [overview](docs/program.md#compute-and-allocation) proposes subdivisions of the existing
post-training and protected envelopes; these do not add budget or promise that every stage fits.
Completed external rental costs are recorded separately from the future grant ledger. The 50-hour
pilot ceiling is not an instruction to repeat the H100 run; reconcile unused reservations at launch.

At the measured H100 full-trainer rate, 2,300 GPU-hours corresponds to about 106.5B base tokens.
320B/400B requires about 38.6K/48.3K effective tokens/s per allocated GPU within that reservation.
Four GPUs increase aggregate speed while consuming four GPU-hours per elapsed hour; they do not
close the per-GPU efficiency gap. No GH200 or distributed speedup is assumed. The rough 90-calendar-day
application window is not 90 days of continuous four-GPU funding. No GH200 jobs have launched.

## What success means

| Capability | Evidence to collect |
| --- | --- |
| Code and agentic coding | Standard executable code tasks plus held-out repository repair, regressions and actual task completion |
| Math and general usefulness | Checked answers, source-wise loss, knowledge, instruction compliance and representative language tasks |
| Thinking and tools | Correctness versus reasoning budget, valid calls, use of observations, error recovery and bounded loops |
| Context | Positional retrieval, related-prefix benefit, multi-file/document reasoning and short-task retention |
| Efficiency | All-in GPU-hours, tokens/FLOPs to useful quality, prefill/decode latency, memory and cost including failed attempts |
| Openness | Identified checkpoints, source/config manifests, processing recipes, curves, costs, failure analysis and executable evaluation |

The [competitive strategy](docs/competitive.md) and [report outline](docs/report.md) define the
comparison and release evidence. No architecture superiority claim follows from an engineering
pilot without a matching control. Release claims must describe measured capability and limitations.
