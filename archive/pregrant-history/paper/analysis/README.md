# Paper analysis

Put deterministic figure and table generation code here. Scripts read immutable checked result records
and may read external artifacts only through a hash-bound manifest.

Each script must:

- name the claim and output IDs it serves;
- reject missing or unexpected arms, seeds, sources, and tasks;
- preserve preregistered statistical rules;
- write outputs create-only or through an auditable successor;
- emit a sidecar manifest containing input hashes, Git revision, command, and output hash;
- have fixture tests that exercise failure as well as success.

Exploratory analysis belongs in a dated notebook entry or isolated scratch area outside Git. It is
promoted here only after its role and output contract are frozen.
