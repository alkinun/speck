# MiniCPM4 and the current UltraData corpus candidates

## Sources and scope

Checked on 2026-09-13 against the PDF of [MiniCPM4, arXiv:2506.07900v2](https://arxiv.org/pdf/2506.07900v2)
(2025-09-04), pinned dataset cards, and four bounded Parquet-footer inspections. The
[source receipt](surveys/openbmb_source_review_20260913.json) records revisions, card/PDF identities,
inspected files, and schemas. Large sources remain in the runtime literature store.

- Ultra-FineWeb: `02c85641e3d19a854be2e09139c25adaa9518063`.
- UltraData-Code: `85182d829f2ce7ea07cca72ebfc509deea1d9f5f`.
- UltraData-Math: `fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87`.
- Ultra-FineWeb-L3: `bc3b1ba986fcaef6871b9790a413b16267c2de0f`.

Numbers below are author-reported, not Speck replications. Current Code/Math releases were absent
from the proposed first-wave matrix and warrant review before its freeze. Ultra-FineWeb v1.4 and
Ultra-FineWeb-L3 Multi-Style already appear in that proposal and the existing source survey.

## What MiniCPM4 establishes

Table 8 reports **1T training tokens for MiniCPM4-0.5B** and **8T for MiniCPM4-8B**. The introduction
gives 8.3T for the 8B model, split into 7T stable and 1.3T annealing tokens; section 5.1 instead describes
7T plus 1T. Preserve that reporting discrepancy rather than inventing a reconciled exact total.
The headline 22% comparison refers to roughly 8T versus the paper's listed 36T Qwen3 baseline.

The 0.5B model's Table-8 average is 52.99 versus 44.93 for Qwen3-0.6B, combining English, Chinese,
reasoning, and code tasks. MiniCPM4-0.5B reports MMLU 55.55 versus 42.95, while MATH500 is 29.60 versus
50.20 and GSM8K is 52.08 versus 61.71. It is a strong compact-model result, not uniform superiority.
These are released-system comparisons after their training pipeline.

The report bundles UltraClean filtering/generation, WSD, hyperparameter transfer/search, multi-token
prediction, FP8 engineering, sparse long-context attention, post-training, and inference systems.
Its model-level comparison does not isolate data or architecture. Table 2 is more direct data evidence:
at about 100B tokens using a 1.2B architecture, reported English averages are 42.28 for FineWeb,
44.56 for FineWeb-Edu, and 45.89 for UltraFineWeb.

The HTML conversion duplicates some math numerals (`8` appears as `88`). The counts above were checked
against the PDF and the MiniCPM4-0.5B model card rather than copied from those conversion artifacts.

## The validation method matters

UltraClean's efficient verification starts from a roughly 1.1T-token pretrained 1B model, then uses
a 10B-token annealing comparison with a 30% candidate-data component and 70% background. Its reported
110 GPU-hours compare with 1,200 GPU-hours for 100B from scratch. The pretrained parent and upstream
generation/classification work are separate costs; those GPU-hours are not GH200 rates. The report
also includes 100B-token from-scratch data comparisons, which are distinct experiments.

ModelTunnel v2 uses task-conditioned likelihood through ScalingBench because tiny models can remain
near chance on direct downstream scores. This supports capability-relevant development signals
alongside neutral held-out loss. Speck must keep final benchmarks/audits independent: benchmark-derived
development indicators cannot silently become independent final evidence.

## Current corpus evidence

| Corpus/tier | What it supplies | Author-reported evidence and role |
| --- | --- | --- |
| Ultra-FineWeb / L2 | Selected natural web, around 1T English tokens | Direct 1.2B/100B comparison above; retain as a serious web contender |
| UltraData-Code L2 | About 400B selected algorithmically relevant tokens in 11 programming languages | Controlled 1B/10B continual training reports +4.37 EvalPlus and +3.05 MultiPL-E points over Stack-Edu |
| UltraData-Code L3 | About 150B implementation-grounded programming-exercise tokens | A 50/50 L2/L3 mix reportedly adds +8.42 EvalPlus and +8.07 MultiPL-E points over L2 alone in the multilingual-training comparison |
| UltraData-Math L2-preview | Card reports 33.7B quality-selected web-math tokens | Serious natural-math candidate; distinguish the exposed preview from an assumed complete L2 release |
| UltraData-Math L3 | Card reports 88B tokens across QA, conversation, multi-style, and textbook/exercise forms | Tier comparisons use a 1.2B model pretrained for 1.3T tokens, followed by about 100B at 30% target / 70% general data; establish the exact published subconfig before transferring the result |
| Ultra-FineWeb L3 | General-knowledge QA and multi-style rewrites; about 245B English QA + 164B English Multi-Style tokens | Card reports stronger late-stage results in 100B-token tests and use in MiniCPM5's decay phase |

Code's technical report is marked coming soon; its controlled numbers currently come from the card.
Math's [full-comparison figure](https://huggingface.co/datasets/openbmb/UltraData-Math/resolve/fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87/assets/ultradata-math-full-comparison.png)
labels its winning row simply `UltraData-Math (Ours)`: MATH500 37.02 and GSM8K 61.79, versus
33.40/58.45 for Nemotron-CC 4plus. The row does not identify a single released L2/L3 subconfig; do not
attribute those headline scores to an arbitrary L3 subset or invented mixture.
Token counts use upstream accounting conventions, not measured Speck-token capacity. L1/L2/L3 are
related views/derivatives and must not be summed as an independent unique corpus.

These releases span different generations: Code was released in September 2026, Math in February
2026, and the web-L3 card describes a May 2026 release using MiniCPM4/Qwen generators. Those dates and
names do not establish that today's exact artifacts were MiniCPM4's original 2025 training mixture.
Current Code and web-L3 cards explicitly connect to MiniCPM5.

## What the inspected schemas imply

Pinned footer inspection, without downloading full shards, found:

- Code L2: `uuid`, `repo_name`, `relative_path`, `content`, role/category, relevance and quality scores.
  No explicit repository revision or license field appears in the inspected schema. The card requires
  source-repository license compliance; the project Apache label does not replace source mapping for
  Speck's restricted-code policy.
- Code L3: `uuid`, `content`, serialization labels, `raw_content`, `task`, `analysis`, `solution`,
  `test`, and `full_content`. Default `content` and all-fields serialization are different treatments.
  A `test` field contains generated test candidates, not proof every solution passed them.
- Math L2-preview: `content` and `quality_label`; no explicit language, URL, or parent identifier in
  the inspected shard. Math L3 QA: `uid` and `content`. Establish label semantics, English yield, and
  available lineage before materializing a training treatment.

Cards list English and Chinese. Programming-language splits do not guarantee English prose/comments.
L3 derivatives require parent/problem/repository-family handling: a paraphrase can evade lexical
deduplication while sharing evaluation ancestry. Math lists FineMath, MegaMath and Nemotron math among
seed sources; web L3 derives from Ultra-FineWeb. These relationships matter for clean comparisons.

The live viewer reports 13,707,851 Math L2-preview rows while the card lists 14.98M. The viewer is an
unversioned convenience observation; reconcile pinned inventory before using either number as an
operations capacity assertion. Full shard digests/content and dataset-wide coverage were not verified
by the footer inspection.

## Speck interpretation and proposed response

1. Revisit first-wave choices before freezing. UltraData-Code and UltraData-Math deserve explicit
   candidate consideration; earlier natural-source blends should not exclude stronger new evidence.
2. Organize the question around **selected natural data versus refined/synthetic data**, with a strong
   incumbent control. A candidate ladder is incumbent → UltraData L2 → matched L2/L3 mixture. Fit a
   revision within existing screen/decay budgets rather than silently adding another matrix.
3. Keep domain and synthetic-origin accounting explicit. Code/math L3 and general-web L3 have different
   roles; assign primary categories once so tokens are not double-counted.
4. Treat tiny from-scratch runs as screens, retaining later-stage/scale confirmation. Much of the new
   evidence uses mature parents and larger budgets than proposed E1S runs. Efficient decay checks
   inform E4 and continuation assets, while the mature parent's training remains a real cost.
5. Public refined data could provide reasoning-rich pretraining while the other codebase explores
   thinking-only post-training. These stages are complementary; longer output alone does not establish
   stronger reasoning.
6. Keep the backbone question and grant ceiling stable while improving the shortlist. The 0.5B model's
   1T horizon is 2.5 times Speck's 400B target. Source-level evidence is a stronger basis for action
   than expecting its full-model scores to transfer directly.

This is a literature/candidate review. New source-use decisions, English/provenance qualification,
serialization choices, budgeted recipe revisions, and local controlled evidence remain pending.

Follow-up: the [revision-checked content intake](../findings/2026-09-13-ultradata-intake.md) inspects
672 real records across seven views and turns these questions into a conditional preparation proposal.
It is an intake observation, not a replication of the authors' model-quality results.
