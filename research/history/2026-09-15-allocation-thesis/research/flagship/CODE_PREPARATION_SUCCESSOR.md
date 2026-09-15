# Current code preparation decision — 2026-09-15

The [selected decision](code_preparation_decision_v2.json) replaces the v2 first-wave preparation
background and v1 code-language prior **before all first-wave model outputs**. This is an agent
preparation choice under the owner's delegated instruction to decide and continue. Existing human
source-use approvals remain unchanged; it is not a model nomination or training-launch receipt.

Use [first-wave proposal v3](first_wave_proposal_v3.json), its
[compiled preparation](first_wave_preparation_v3.json), [language prior v2](code_language_preparation_v2.json)
and [compiled language requirements](code_language_requirements_v2.json) for future preparation.
The existing source-bound acquisition/qualification plans and completed attempts remain immutable.

## Selected recipe

Use **restricted Stack v3 as the shared code background**. Stack-Edu remains an E1S specialist
treatment and one half of the equal blend. All three code arms remain: restricted Stack v3,
Stack-Edu, and 50/50 within each language. Becoming the preparation background does not win the
source comparison. Other source assignments, including natural Math L2, are unchanged.

The former Stack-Edu background required 1.2B nominal / 1.44B headroom. Its measured sample estimates
do not support that envelope, particularly Java and TypeScript whose complete released metadata
has been inspected. Retaining that background with only modest language changes would leave this
constraint unresolved. Restricted Stack v3 has working complete-file acquisition and bounded
metadata discovery paths; its additional capacity must still be prepared and verified.

The revised allocation reduces Java and TypeScript, retains every language and redistributes nine
percentage points across Python, C++, Rust, Shell and SQL. It keeps application and systems
languages prominent without selecting weights from model losses. Apply these exact shares
independently to both sources, each half of the blend, shared backgrounds and repetition pools.

| Language | Previous share | Selected share | Stack-Edu headroom | Stack v3 full-wave headroom |
| --- | ---: | ---: | ---: | ---: |
| Python | 25% | 28% | 100,800,000 | 403,200,000 |
| C++ | 15% | 16% | 57,600,000 | 230,400,000 |
| C | 5% | 5% | 18,000,000 | 72,000,000 |
| Java | 15% | 10% | 36,000,000 | 144,000,000 |
| JavaScript | 15% | 15% | 54,000,000 | 216,000,000 |
| TypeScript | 10% | 6% | 21,600,000 | 86,400,000 |
| Rust | 5% | 8% | 28,800,000 | 115,200,000 |
| Go | 5% | 5% | 18,000,000 | 72,000,000 |
| Shell | 1% | 2% | 7,200,000 | 28,800,000 |
| SQL | 1% | 2% | 7,200,000 | 28,800,000 |
| Markdown | 3% | 3% | 10,800,000 | 43,200,000 |

Stack-Edu now requires **300M nominal / 360M headroom** over the full first wave. Restricted Stack
v3 requires **1.2B nominal / 1.44B headroom** over the full first wave. For the first 2B-token E1S
dataset milestone, either complete code source needs only **300M nominal / 360M headroom**;
prepare that smaller stage before expanding the background for E1W/E3. The full source envelope
remains **17.7B**, before headroom, and is not measured globally unique text.

## Evidence and remaining risk

Existing Stack-Edu estimates under the unchanged content/security filters give:

| Language | Estimated tokens before exclusion | Sampling SE | Selected treatment headroom |
| --- | ---: | ---: | ---: |
| Python | 123.387M | 9.852M | 100.8M |
| C++ | 69.960M | 3.545M | 57.6M |
| C | 23.374M | 2.121M | 18M |
| Java | 45.826M | 1.566M | 36M |
| JavaScript | 71.800M | 5.564M | 54M |
| TypeScript | 31.683M | 3.525M | 21.6M |
| Rust | 36.393M | 3.638M | 28.8M |
| Go | 23.996M | 2.529M | 18M |
| Shell | 294.447M | 37.549M | 7.2M |
| SQL | 55.936M | 7.034M | 7.2M |
| Markdown | 50.756M | 10.462M | 10.8M |

All point estimates exceed the selected per-language treatment targets. They are **not** measured
excluded stock, simultaneous confidence bounds, or a guarantee against exclusion losses. Further
redistribution does not happen automatically if an actual count fails. A deficit stops the affected
materialization and requires new supply or another explicit pre-results successor.

Restricted Stack v3 currently has **41,383,315 content/security-passing tokens** in fourteen
complete files. Full reference/candidate exclusion and joint eligibility remain pending, and the
selected larger background remains unfulfilled. The previous cached sample and these acquisition
units overlap; do not add their counts. This decision resolves what to prepare, not readiness to
launch or a forecast that larger supply will finish by a particular date.

Preserve the 55/15/10/10/5/5 category prior, 29 logical runs, seeds and confirmations, 130B logical
exposure, 192-GPU-hour data ceiling, 38-GPU-hour D4/D6 dependency and 230-GPU-hour P1 total. E3's
nested pools, 1x/2x/4x exposure and explicit repetition implementation remain required. The
flagship remains 1.2B / 400B with the existing 320B throughput fallback. Mistral is frozen; D5
remains unopened. No source filtering, rights or attribution requirement is relaxed.

## Next execution steps

1. Bind a Stack-Edu acquisition successor to all 28 verified metadata files, the qualified ordered
   fetch path, the selected language targets, preserved cache/attempt reuse and explicit storage
   limits. The original paused stock worker must not be blindly resumed.
2. Expand restricted Stack v3 supply through bounded metadata-directed complete-file tranches,
   preserving file and unit identities. Reuse the 130 completed content units; unchanged language
   membership and content policies permit reuse, but each successor must verify config parity.
3. Measure excluded per-language capacity against the 360M E1S target first, then assemble one
   complete 2B mixture with shared-background precedence and fixed document/order/seed identities.
4. Finish the independent heldout extraction, global identity/leakage and run-analysis contracts,
   and prepare the GH200 qualification package. No model training is launched by this decision.
