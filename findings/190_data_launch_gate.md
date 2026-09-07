# 190 — Marked flagship training now fails before model construction

The data launch preflight binds the checked-out full Git commit, exact data/tokenizer/model/train
files, human source-rights acceptance, production operations qualification, human-reviewed deny
ledger, production firewall, selected tokenizer, and packed dataset. It cross-checks the firewall's
rights/operations hashes, the operations ledger hash and disjointness gate, and the packed dataset's
selected-tokenizer fingerprint before exclusively creating a non-overwriting receipt.

Training configs explicitly marked `requires_data_launch_authority=true` require
`--data-authority PATH`. The trainer verifies packed shards first, then rehashes the receipt's Git,
experiment, rights, operations, deny, firewall, tokenizer, and packed artifacts before model
construction. Historical configs default to false and retain their existing behavior; all future
flagship configs must set the marker.

Fixture tests fail each missing authority independently, reject changed external records after receipt
issuance, reject launches without the required CLI receipt, and construct no model. No real receipt
exists because rights, the 20B operations qualification, real firewall, tokenizer decision, packed
data, and flagship experiment manifests remain absent. This gate authorizes nothing by itself.

Artifacts: [checked result](../results/data/data-launch-gate-20260907.json) and
[launch plan](../research/flagship/data_launch_plan.json).
