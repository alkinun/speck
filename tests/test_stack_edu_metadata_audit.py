"""The vectorized census predicate must reject exactly what the historical row predicate did."""

import importlib.util
from pathlib import Path

import pyarrow as pa

SPEC = importlib.util.spec_from_file_location(
    "stack_edu_metadata_audit",
    Path(__file__).resolve().parents[1] / "experiments/corpus-audit/audit_stack_edu_metadata.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

POLICY = {
    "filters": {
        "accepted_detected_licenses": ["MIT", "Apache-2.0"],
        "accepted_encodings": ["UTF-8"],
        "min_file_bytes": 100,
        "max_file_bytes": 1000,
    },
    "excluded_path_components": ["vendor"],
}
GOOD = {
    "blob_id": "a" * 40,
    "repo_name": "owner/repo",
    "path": "/src/module.py",
    "src_encoding": "UTF-8",
    "length_bytes": 500,
    "detected_licenses": ["MIT"],
    "license_type": "permissive",
}
CASES = [
    ({}, True),
    ({"detected_licenses": ["MIT", "Apache-2.0"]}, True),
    ({"detected_licenses": ["MIT", "GPL-3.0"]}, False),
    ({"detected_licenses": []}, False),
    ({"license_type": "no_license"}, False),
    ({"src_encoding": "Latin-1"}, False),
    ({"length_bytes": 99}, False),
    ({"length_bytes": 1001}, False),
    ({"path": "/lib/Vendor/x.py"}, False),
    ({"path": "\\lib\\vendor\\x.py"}, False),
    ({"path": "/lib/vendored/x.py"}, True),
    ({"repo_name": "  "}, False),
    ({"blob_id": "A" * 40}, False),
]


def test_mask_matches_historical_row_predicate():
    rows = [GOOD | change for change, _ in CASES]
    table = pa.Table.from_pylist(rows)
    assert audit.file_mask(table, POLICY).tolist() == [expected for _, expected in CASES]
