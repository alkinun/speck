# Bounded ordered code-fetch qualification

The [qualification plan](ordered_code_fetch_qualification_v1.json) tests the revised execution
path without restarting the paused stock or changing code-source policy. It uses the eleven
original verified metadata files; the separate [metadata successor](stack_edu_metadata_acquisition_v2.json)
adds supply for deficient languages and does not select a new recipe.

Metadata eligibility is indexed once in physical source order, retaining original rows and
checksummed source/configuration identities. The index accounts for every input row, combining
coarse score/license-type rejections explicitly instead of claiming identical rejection labels
to the old per-row loop. It preserves the accepted metadata set under the frozen policy.

The probe selects 32 eligible records in each of four equally sized eligible-order strata per
language, with seed 42. This is 128 records per language, 1,408 in total. The selected records and
sampling populations are retained. Yield estimates are stratified, pre-full-exclusion estimates
for these individual first files. Sampling standard errors do not cover source ancestry,
post-exclusion loss or an unobserved additional shard. No per-language capacity pass is declared.

A 64-item window feeds 32 workers across eligible inputs; completed results publish in original
selected order. Duplicate blob IDs in the live window share a future. The window bounds memory
and speculative work, so unusually slow earlier requests can still exert backpressure. Missing
404s are recorded; exhausted transient retries fail the invocation without silently skipping.
Verified original cached inputs remain at their existing paths, and new blobs use a separate
NVMe cache. No old cache migration or deletion occurs.

The cache has a conservative 12GiB reservation ceiling including possible partial retries and
receipts. Each language index is limited to 512MiB. Startup requires 20GiB of working headroom
above a 64GiB free-space floor, and fetches check that floor during operation. The working path
is outside the repository on NVMe; all resulting files receive a checked archival tar copy on
the data drive. Keep both copies and their identities. This is a bounded probe, not permission
for an unbounded SSD source stock.

The real test deliberately interrupts after 64 ordered records, resumes from its checksummed
journal, compares with a clean warm-cache replay and checks completed-result reopening. It then
runs the existing content, prose, security, benchmark, Gitleaks and Mistral checks. Retain all
failed and prefetched attempts. Fetch timing includes persistence and intentional interruption;
a warm replay is not a separate network benchmark or a controlled storage speedup.

Passing this probe qualifies its bounded components. Integrating the index/queue into a full
source-stock successor, full exclusion, joint eligibility and training manifests remain separate
work. The paused frozen stock is not silently converted into this execution.
