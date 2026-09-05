# 50 — HELMET TREC rights and provenance decision

## Question

Does public availability of the TREC question-classification files provide affirmative authority to
download, evaluate, redistribute, or use CogComp's labeled collection in the intended organizational
scope?

## Authoritative evidence

The pinned [CogComp question-classification page](https://cogcomp.seas.upenn.edu/Data/QA/QC/) offers
the experimental files and names their authors and a contact address. Neither that page nor its
[parent QA corpus index](https://cogcomp.seas.upenn.edu/Data/QA/) contains a license, copyright grant,
or terms of use. The pinned Hugging Face card independently labels the license `unknown`.

The collection is not a single-origin raw file. Its metadata attributes questions to USC, manually
constructed rare-class examples, TREC 8/9, and TREC 10, with CogComp's manual fine/coarse labels. This
mixed provenance matters because availability from one mirror does not demonstrate authority over
every source or the labeled derivative.

The [NIST TREC FAQ](https://trec.nist.gov/faq.html) says topics and relevance judgments are generally
downloadable while document collections commonly require licensing from an authorized distributor.
That general access statement does not identify a license for this separately assembled and labeled
CogComp classification dataset.

## Evidence versioning

V1 correctly stopped before a decision because it froze raw NIST HTML bytes. Cloudflare changed only
obfuscated email-link tokens and injected scripts between capture and execution. V1 is preserved as a
failed transport-identity artifact; none of its partial metadata was overwritten or deleted.

V2 freezes canonical visible text while continuing to byte-pin the commit-addressed Hugging Face
source and the stable CogComp pages. NIST's raw transport changed again during v2, but its 3,158-byte
canonical visible text retained SHA-256 `a6fe07c1…`. This demonstrates that the normalization removes
only delivery wrappers, not rights-relevant content.

## Decision

The metadata audit qualifies, but authority does not. Payload acquisition, evaluation use,
redistribution, and commercial scope remain blocked. No TREC Parquet or original label files were
downloaded, and no external contact was attempted.

The next valid options are:

1. obtain written terms or permission from a rights-bearing source for the exact labeled collection
   and intended organizational scope; or
2. freeze a new HELMET manifest version removing TREC before any candidate results exist.

Until then, HELMET ICL cannot qualify even though Banking77, NLU Evaluation Data, and CLINC150 have
immutable technical paths. This operational gate is not legal advice or a final ownership judgment.

## Artifacts

- [Failed v1 protocol](../research/architecture-promotion-v1/helmet_trec_rights_v1.json)
- [Failed v1 result](../results/Speck-Architecture-Promotion-v1/helmet-trec-rights-v1-failed.json)
- [Executed v2 protocol](../research/architecture-promotion-v1/helmet_trec_rights_v2.json)
- [Blocked v2 decision](../results/Speck-Architecture-Promotion-v1/helmet-trec-rights-decision.json)
- [Rights audit runner](../scripts/helmet_trec_rights_audit.py)
