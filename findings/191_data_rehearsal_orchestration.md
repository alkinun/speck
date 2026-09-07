# 191 — The 20B rehearsal is orchestrated but has not run

The fixture-qualified runner freezes six stages: source identity, acquisition, global deduplication,
packing, resume/cleanup, and firewall/training disjointness. Each stage binds its command, environment,
working directory, and inputs, then records immutable result/stdout/stderr hashes plus elapsed time,
cumulative child RSS, free-space change, and staging-size change.

State is durable after each stage. A resumed run rehashes and validates every completed result and log
before skipping it; injected interruption proves completed commands are not repeated, while changed
results or logs fail closed. Finalization requires every exact/near dedup, packing, cleanup, resume,
and firewall-disjointness gate plus acquisition, filtering, record, SQLite, memory, packing, storage,
packed-token, and unique-token metrics. Packed tokens must reach the declared target.

Fixture mode cannot issue production operations authority. Production mode requires exactly 20B
tokens and a hash-bound all-approved human rights record; authority issuance revalidates every stage,
log, result, gate, metric, rights record, and deny ledger. The positive authority path is exercised
only with temporary fixtures.

No real command, qualified source record, or persistent operations record was produced. Real source
acceptance, commands, storage paths, and interruption schedule remain unfrozen, and all production
measurements remain unknown. This is orchestration readiness—not the 20B rehearsal result.

Artifacts: [checked result](../results/data/data-rehearsal-tooling-20260907.json) and
[orchestration plan](../research/flagship/data_rehearsal_plan.json).
