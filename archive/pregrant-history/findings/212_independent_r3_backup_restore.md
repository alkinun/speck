# 212 — A representative R3 tree restores exactly from a second physical device

The complete 2B packed calibration dataset and stage manifests, complete production firewall, and
formal tokenizer model/static artifacts were copied from the Toshiba data volume to the separate NVMe
root volume. The backup contains 85 files and 5,080,799,226 bytes. Every backup hash passes; the entire
tree was then restored to the data volume, every restored hash passed, and only the generated restore
target was removed.

All twelve sealed audit files remain mode zero in both locations. The trusted backup process temporarily
granted itself read permission only to byte-copy and hash those files, did not parse or emit payload
text, and restored mode zero in `finally` handling. This is not an evaluation opening under the project's
non-adversarial security boundary, but it is an access-control relaxation and must not be described as
cryptographic sealing. A successor builder should create the second sealed copy before publication.

This closes the local independent-device backup/restore rehearsal, not all of R16. Both copies remain on
one host; off-site backup, the DOI-minting archive, final model/artifact destinations, and credential
checks remain pending. Git metadata through `2187c00` is independently present on GitHub.

Artifact: [backup and restore result](../results/release/independent-r3-backup-restore-20260910.json).
