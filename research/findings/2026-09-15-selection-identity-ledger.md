# Real selection-heldout identity ledger materialized

The [bound preparation](../flagship/heldout_selection_identity_v1.json) produced a reusable
identity ledger and source-input row locators for **all 28,604 real selection-heldout documents**.
It uses the existing production firewall's six committed selection files and passes the existing
`data_selection` authorizer. No document selection or size change occurred, and no sealed audit
payload was read or opened.

| Category | Documents |
| --- | ---: |
| Web | 4,860 |
| Code | 4,452 |
| Math | 5,878 |
| Synthetic | 6,554 |
| Science | 613 |
| Reference | 6,247 |

Every source/document hash and normalized-content hash was checked. The ledger preserves source
identity, category and original per-category order and reproduces each firewall commitment, exact
UTF-8 byte count and document count. A separate locator file retains source-input ID, original
input row and provenance hash for reconstruction from preserved source artifacts. These are
locators into firewall inputs, not a claim that upstream raw-document reconstruction has already
been completed.

The [result](../../results/data/heldout-selection-identity-20260915.json) binds the original
firewall manifest, heldout plan, preparation plan and execution revision. The
[independent review](../../results/systems/heldout-selection-identity-verification-20260915.json)
reopened both output files, recomputed every ledger field directly from the authorized production
selection rows, reproduced every locator and checked every category commitment/count/byte total.
The ledger is at `/mnt/speck-data/speck/heldout-selection-identity-v1/selection_heldout.jsonl`;
source text remains in the original firewall files.

This is one real component of the parser-independent evaluation bundle. It does not qualify an
independent extractor, create the aggregate training identity ledger or audit commitment ledgers,
pass additional leakage checks, supply model/backend score bindings, or authorize model selection
or training. The next evaluation work is to resolve the source locators against preserved raw
inputs and bind genuinely separate extraction implementations. Re-serializing the production
text alone would not establish parser independence. The original fixture plan and its failed or
pending requirements remain preserved.
