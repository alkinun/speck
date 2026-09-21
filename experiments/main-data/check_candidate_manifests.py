"""Validate design-only candidate manifests without admitting any source.

This check catches drift between the shared stage-conditioned contract and the
four source-specific candidate manifests.  It verifies local receipt hashes and
the explicit non-admission boundary; it does not infer rights, correctness,
eligible supply, or training authority.

Run from the repository root::

    python experiments/main-data/check_candidate_manifests.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speck.provenance.io import file_sha256

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
MANIFESTS = (
    "experiments/main-data/natural-web-candidate-manifest.json",
    "experiments/main-data/natural-code-candidate-manifest.json",
    "experiments/main-data/math-candidate-manifest.json",
    "experiments/main-data/post-training-candidate-manifest.json",
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify_receipts(manifest: dict, manifest_path: Path) -> int:
    receipts = manifest.get("source_of_truth")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError(f"{manifest_path}: missing source_of_truth receipts")
    checked = 0
    for name, receipt in receipts.items():
        if not isinstance(receipt, dict) or not receipt.get("path") or not receipt.get("sha256"):
            raise ValueError(f"{manifest_path}: incomplete receipt {name}")
        path = _path(receipt["path"])
        if not path.is_file():
            raise ValueError(f"{manifest_path}: missing receipt target {path}")
        if file_sha256(path) != receipt["sha256"]:
            raise ValueError(f"{manifest_path}: receipt checksum mismatch {path}")
        checked += 1
    return checked


def _validate_manifest(path: Path) -> tuple[int, int]:
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
        if not candidate.get("id") or candidate.get("admitted") is not False:
            raise ValueError(f"{path}: candidate must have an id and admitted=false")
        if candidate["id"] in seen:
            raise ValueError(f"{path}: duplicate candidate id")
        seen.add(candidate["id"])
        gates = candidate.get("gate_status")
        if not isinstance(gates, dict) or set(gates) != {
            "source_use",
            "family_partition",
            "correctness",
            "finite_supply",
            "runtime",
        }:
            raise ValueError(f"{path}: incomplete gate set for {candidate.get('id')}")
        if not candidate.get("blockers"):
            raise ValueError(f"{path}: candidate has no blockers: {candidate['id']}")

    return candidate_count, _verify_receipts(manifest, path)


def validate(paths: tuple[str, ...] = MANIFESTS) -> dict:
    checked_candidates = 0
    checked_receipts = 0
    for name in paths:
        path = _path(name)
        candidates, receipts = _validate_manifest(path)
        checked_candidates += candidates
        checked_receipts += receipts
    preflight = _path("experiments/main-data/candidate-manifest-preflight.json")
    preflight_value = json.loads(preflight.read_text())
    summary = preflight_value["summary"]
    if summary["admitted_count"] != 0 or summary["complete_required_manifest_count"] != 0:
        raise ValueError("candidate preflight must remain non-admitting and incomplete")
    if (
        type(summary["eligible_unique_tokens_established"]) is not int
        or summary["eligible_unique_tokens_established"] != 0
    ):
        raise ValueError("candidate preflight cannot claim eligible tokens")
    contract = json.loads(_path("experiments/main-data/data-design-contract.json").read_text())
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
        "admitted_candidates": 0,
        "eligible_unique_tokens_established": summary["eligible_unique_tokens_established"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=MANIFESTS)
    args = parser.parse_args()
    print(json.dumps(validate(tuple(args.paths)), indent=2, sort_keys=True))
