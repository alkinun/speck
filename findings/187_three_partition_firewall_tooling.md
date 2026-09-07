# 187 — Three-partition firewall tooling qualifies on fixtures

The executable firewall now constructs `tokenizer_sample`, `selection_heldout`, and `sealed_audit`
without reading any real qualified source during tooling validation. Tokenizer train/evaluation are
separate subpartitions. All destinations use distinct frozen seeds and global normalized-content
deduplication. Equal-category byte targets, source allocations, unseen-source bytes, domain diversity,
and domains absent from tokenizer training are mandatory rather than inferred after results.

The sealed partition has separate `D5_tokenizer` and `E2_mixture` identities. D5 requires the two
ranked tokenizer finalists and falls back to Mistral; E2 requires all three ranked mixture finalists
and falls back to the balanced prior. Generic consumers cannot access either. Opening requires a
hash-bound ranking frozen before audit; an exclusive receipt is fsync'd before files become readable.
If payload verification then fails, the receipt still prevents a retry.

Consumer checks deny unknown paths, deny every firewall file to model training, and restrict tokenizer
training, tokenizer static evaluation, and data selection to their exact declared files. Fixture mode
cannot authorize any real consumer. Production construction additionally requires immutable external
records containing named human approval for every source and passing global exact/near deduplication,
cleanup, and interruption/resume gates.

No real held-out/audit byte sizes were previously frozen, so this work deliberately does not invent
them or materialize real data. File permissions and receipts are cooperative operational controls,
not an adversarial secrecy claim against the data owner. Real targets and inputs must be frozen in a
successor before outputs; no tokenizer or model training is authorized.

Artifacts: [checked tooling result](../results/data/data-firewall-tooling-20260907.json) and
[firewall plan](../research/flagship/firewall_plan.json).
