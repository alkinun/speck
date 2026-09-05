# 33 — RULERv1 offline source-bundle qualification

## Question

Can the pinned `rulerv1-ns` pipeline generate promotion evidence without cloning moving branches,
installing unversioned packages, downloading mutable datasets, or silently redistributing upstream
content under Speck's MIT license?

## Transitive audit

The pinned NeMo-Skills preparer at `f4a3fd8` does more than its top-level contract originally showed:

- runs an unversioned `pip install wonderwords html2text tenacity`;
- clones `NVIDIA/RULER` without a revision when four JSON inputs are absent;
- downloads Paul Graham pages and repository files from moving URLs;
- downloads SQuAD and HotpotQA without checking content hashes;
- requires a Git-LFS English-word payload that an ordinary filtered checkout leaves as a pointer; and
- triggers mutable NLTK `punkt` and `punkt_tab` downloads from import-time code.

Any one of these can change generated prompts while the nominal RULER and NeMo-Skills commits remain
fixed.

## Offline bundle

The new `ruler` dependency group pins the complete source-preparation environment. The retained local
bundle is 99MB and contains:

- 169 raw Paul Graham HTML responses plus parser-canonicalized essay bodies;
- 49 essay files from `LLMTest_NeedleInAHaystack` commit `021385d`;
- the exact RULER consolidated `PaulGrahamEssays.json`;
- SQuAD v2 development data;
- HotpotQA distractor development data;
- the materialized RULER English-word Git-LFS object;
- all four Wonderwords 3.0.1 word-list assets; and
- the NLTK `punkt` and `punkt_tab` archives.

The HTML raw responses contain dynamic server timestamps. Qualification therefore retains both the
raw bytes and the canonical `<font>` body produced by the exact BeautifulSoup/html2text versions used
to build the consolidated essay artifact. Repeated checks use the retained bytes and never redownload.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| Paul Graham aggregate | 3,108,621 | `8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd` |
| SQuAD v2 dev | 4,370,528 | `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` |
| HotpotQA distractor dev | 61,065,698 | `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` |
| English-word payload | 8,564,991 | `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca` |

The complete 218-source plus package/data identity is
`9a32cac9a3d6e02024930b2eb34173a565189c7ef15abfea8fa9661a2222ba00`.

## Rights and release boundary

SQuAD v2 and HotpotQA are CC-BY-SA-4.0. RULER code is Apache-2.0 and the pinned Needle repository is
MIT, but neither repository license is treated as relicensing the underlying essay text. Paul Graham
payloads and generated prompts containing them remain in the local research cache. Only hashes,
provenance, code, and aggregate measurements may enter the MIT repository or release artifacts.

## Decision

RULER's transitive source inputs are now qualified for offline use. This closes the mutable-download
part of SPE-99 but not the issue itself. Official cases for all 13 tasks and six declared lengths still
need to be generated with the Speck tokenizer, hashed, and proven network-free. Candidate-specific
long-context exports and benchmark execution also remain blocked.

No RULER capability score is claimed.

## Artifacts

- [RULER source contract](../research/architecture-promotion-v1/external/ruler_v1.json)
- [Provenance manifest](../results/Speck-Architecture-Promotion-v1/ruler-source-manifest.json)
- [Offline bundler](../scripts/ruler_source_prepare.py)
