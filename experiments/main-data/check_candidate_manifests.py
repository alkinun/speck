"""Validate design-only candidate manifests without admitting any source.

This check catches drift between the shared stage-conditioned contract and the
four source-specific candidate manifests.  Per-source identity, inventory, gates
and blockers belong to the source-readiness matrix; a manifest names the source
and adds only its domain evidence.  The check does not infer rights,
correctness, eligible supply, or training authority.

Run from the repository root::

    python experiments/main-data/check_candidate_manifests.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speck.provenance.io import check_reference

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FIELDS = {
    "stage",
    "source_family",
    "transformation_type",
    "quality_vector",
    "coverage_strata",
    "dependency_metadata",
    "contamination_status",
    "lineage",
    "unique_tokens",
    "exposure_tokens",
    "repetition_count",
}
READINESS = "experiments/main-data/source-readiness.json"
# Fields the readiness matrix owns; a manifest copy would drift from it.
READINESS_FIELDS = {
    "role",
    "roles",
    "identity",
    "inventory",
    "gate_status",
    "gates",
    "admitted",
    "blockers",
}
MANIFESTS = (
    "experiments/main-data/natural-web-candidate-manifest.json",
    "experiments/main-data/natural-code-candidate-manifest.json",
    "experiments/main-data/math-candidate-manifest.json",
    "experiments/main-data/post-training-candidate-manifest.json",
)


def _verify_receipts(manifest: dict, manifest_path: Path) -> int:
    receipts = manifest.get("source_of_truth")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError(f"{manifest_path}: missing source_of_truth receipts")
    for receipt in receipts.values():
        check_reference(receipt)
    return len(receipts)


def _validate_manifest(path: Path, source_ids: set[str]) -> tuple[int, int]:
    manifest = json.loads(path.read_text())
    if not manifest.get("format", "").startswith("speck_"):
        raise ValueError(f"{path}: unsupported format")
    if manifest.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported format version")
    if manifest.get("status") != "candidate_manifest_not_admitted":
        raise ValueError(f"{path}: candidate must remain non-admitted")
    if (
        not manifest.get("admission_boundary")
        or "does not admit" not in manifest["admission_boundary"]
    ):
        raise ValueError(f"{path}: missing non-admission boundary")

    contract = manifest.get("common_contract")
    if not isinstance(contract, dict) or set(contract) != REQUIRED_FIELDS:
        raise ValueError(f"{path}: common contract fields drifted")
    strata = contract["coverage_strata"]
    if (
        not isinstance(strata, list)
        or not strata
        or any(not isinstance(value, str) or not value.strip() for value in strata)
    ):
        raise ValueError(f"{path}: coverage strata must be nonempty strings")
    for field in REQUIRED_FIELDS - {"coverage_strata"}:
        if not isinstance(contract[field], str) or not contract[field].strip():
            raise ValueError(f"{path}: empty contract field {field}")

    candidate_count = 0
    candidates = manifest.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"{path}: candidates must be a list")
    seen = set()
    for candidate in candidates:
        candidate_count += 1
        if candidate.get("id") not in source_ids:
            raise ValueError(
                f"{path}: candidate is not in the readiness matrix: {candidate.get('id')}"
            )
        if candidate["id"] in seen:
            raise ValueError(f"{path}: duplicate candidate id")
        seen.add(candidate["id"])
        if READINESS_FIELDS & candidate.keys():
            raise ValueError(f"{path}: {candidate['id']} copies readiness-owned fields")

    return candidate_count, _verify_receipts(manifest, path)


def validate(paths: tuple[str, ...] = MANIFESTS) -> dict:
    readiness = json.loads((ROOT / READINESS).read_text())
    source_ids = {source["id"] for source in readiness["sources"]}
    checked_candidates = 0
    checked_receipts = 0
    for name in paths:
        candidates, receipts = _validate_manifest(ROOT / name, source_ids)
        checked_candidates += candidates
        checked_receipts += receipts
    contract = json.loads((ROOT / "experiments/main-data/data-design-contract.json").read_text())
    fields = set(contract["manifest_schema"]["required_fields"])
    # The prose candidate contracts enumerate strata; the record schema names
    # the individual stratum. Keep that spelling mapping explicit.
    fields.remove("coverage_stratum")
    fields.add("coverage_strata")
    if fields != REQUIRED_FIELDS:
        raise ValueError("shared manifest schema drifted from candidate checker")
    return {
        "manifests": len(paths),
        "candidates": checked_candidates,
        "receipts_checked": checked_receipts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=MANIFESTS)
    args = parser.parse_args()
    print(json.dumps(validate(tuple(args.paths)), indent=2, sort_keys=True))
