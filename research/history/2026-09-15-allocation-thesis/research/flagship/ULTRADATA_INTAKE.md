# UltraData content intake

The [frozen intake plan](ultradata_intake_v1.json) examines seven views: Python Code L2/L3, Math
L2-preview, and Math L3 QA/conversation/multi-style/textbook-exercise. It uses 32-row windows at the
beginning, middle, and end of each dataset-viewer split, excluding duplicate indices and truncated rows
from content statistics. This is a bounded intake review, not preparation of training data.

Every response must carry an `x-revision` header matching the pinned dataset commit. Raw response bytes
and their hashes are retained in the runtime store. A partial dataset view, mismatched revision, missing
row, or oversized response fails that view explicitly. The viewer protocol does not independently
verify complete source-shard hashes; final corpus acquisition still requires the normal input gates.

The review records:

- complete/truncated/missing-content counts and string sizes;
- English diagnostics on Python comments/docstrings, code-task prose and analysis, or math content;
- released serialization labels and exact containment of task/analysis/solution/test fields;
- quality-label values, JSON structure, exposed metadata fields and sample identifier overlaps;
- SPDX-marker presence as an observation, without inferring source-use permission.

Language classification is capped and diagnostic. These beginning/middle/end windows are not a uniform
or corpus-weighted sample, so their proportions are not production yield forecasts. Missing ID overlap
cannot establish unrelated ancestry. Source rights, original-parent mapping, decontamination, answer
correctness, and full usable capacity remain separate qualification tasks.

```bash
uv run --no-sync python -m scripts.ultradata_intake \
  research/flagship/ultradata_intake_v1.json \
  results/data/ultradata-intake-20260913.json
```

The outcome should drive a concrete subset/serialization decision and a revised budgeted recipe
proposal. It grants no training authority and executes no downloaded code.

## Completed intake and proposed disposition

The [checked intake](../../results/data/ultradata-intake-20260913.json) covers all seven views and
672 complete rows. The [finding](../findings/2026-09-13-ultradata-intake.md) records language, schema,
serialization and provenance observations. The [candidate revision](first_wave_candidate_revision_v2.json)
places natural L2 challengers into existing specialist slots and keeps L3 refinement for the existing
decay budget. Source admission and final recipe freeze remain open.

Follow-up: the owner approved natural Math L2-preview under the existing guarded-use scope, and the
[first real stock preparation](ADMITTED_MATH.md) now covers its nominal reference-token requirement.
Code and L3 remain outside that source-use extension.
