# 42 — NoLiMa license and metadata decision gate

## Question

What exactly must be approved before Speck may materialize and use the pinned NoLiMa dataset, and did
the metadata audit itself fetch or expose restricted payloads?

## Exact terms

The code repository and dataset revision contain the same Adobe Research License with SHA-256
`8638b5a5beb5e1cdf06a09512d23268358b64f646775581d8276e47329e0aa06`. The license is accepted by
exercising rights. If acceptance is on behalf of an entity, the individual represents that they have
authority to bind that entity.

The permission is limited to academic research and teaching. It expressly excludes commercial
licensing or distribution, commercial product development, and any activity resulting in commercial
gain. Redistribution must remain for noncommercial research, include the Adobe license, and preserve
copyright notices and disclaimers. The grant is revocable and terminates automatically on material
breach. This inventory is not legal advice.

## Haystack rights

The pinned `haystack/LICENSES.md` hash is
`0a295ce17544dc60964bfb7d817388c6f775680c2f4014dc991f9efeb5400b42`. It identifies ten underlying
works: six prohibit commercial use, five impose ShareAlike, two are identified as public domain in the
US, and two use attribution-only licenses. The noncommercial ShareAlike works span versions 2.5, 3.0,
and 4.0. The file does not map the five shuffled or five long-shuffled outputs to individual works.
Because those outputs appear derived from a mixed corpus, local policy conservatively treats every
haystack as noncommercial, attribution-bearing, non-redistributable through the MIT repository, and
subject to unresolved cross-license compatibility review.

## Metadata and cache audit

Dataset commit `378115b` declares ten shuffled haystacks and six needle sets totaling 8,342,694 bytes.
The optional original-book archive remains an unmaterialized 10,023,582-byte Git-LFS object with SHA-256
`43daa2c4c11c608743129cbd688cb5ab06245872a0551fb98d33f7a061220266`.

The partial clone has an empty worktree, and this audit fetched zero new objects. However, its existing
promisor object cache already contains all 16 restricted haystack/needle blobs. Their prior acquisition
is recorded rather than mislabeled as a metadata-only state. The audit does not inspect or materialize
their content.

## Decision

NoLiMa remains blocked. Before use, an authorized representative must confirm that the entity accepts
the pinned Adobe Research License solely for noncommercial academic research/teaching, authorizes local
evaluation-data use, and forbids redistribution through Speck's MIT artifacts. After approval, a
separate step must materialize only the declared files outside Git and compute SHA-256 for every payload.

If the entity cannot make that attestation, NoLiMa must be removed from the required evaluation policy
through a reviewed contract version and replaced by a legally compatible independent benchmark.

## Artifacts

- [Machine-readable decision packet](../results/Speck-Architecture-Promotion-v1/nolima-license-decision.json)
- [NoLiMa contract](../research/architecture-promotion-v1/external/nolima.json)
- [No-download audit runner](../scripts/nolima_license_audit.py)
