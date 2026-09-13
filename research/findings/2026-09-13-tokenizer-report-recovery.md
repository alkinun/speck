# The completed 40,960-tokenizer screen report is recovered

The [recovery record](../../results/data/tokenizer-pilot-40960-recovery-20260913.json) verifies the
terminal step-17,174 checkpoint, model/optimizer payload identities, finite model tensors, all bound
evaluation boundaries, and the original corrected-FLOP input. Training and evaluation are not replayed.

The original service completed its training endpoint but failed while constructing the final report
because repository reorganization had removed an absolute metadata path. The
[failure record](../../results/systems/tokenizer-pilot-40960-finalization-interruption-20260913.json)
is retained. A compatibility symlink now serves the identical archived accounting JSON to frozen runs.

Recovered fixed-document macro BPB is 1.109140944; fixed-FLOP macro BPB is 1.107121722. Recorded active
time through the final evaluation is 16,521.759 seconds. CUDA peak allocation had only existed in the
terminated process, so schema-v2 run/summary records explicitly store it as unavailable (`null`). No
system-RAM measurement, zero, or later observation substitutes for it.

This closes report publication for one screen arm, not the tokenizer decision. The compact 32K seed-42
screen was separately [launched](../../results/systems/tokenizer-pilot-32000-launch-20260913.json) after
input hash verification, from the frozen pre-cleanup checkout. Ranking, the measured 30-hour budget
review, any eligible confirmation, and D5 follow their existing order after the full screen completes.
