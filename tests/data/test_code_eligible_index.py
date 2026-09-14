import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.data.code_eligible_index import build_eligible_index, sample_index
from speck.data.stack_edu_stock import metadata_rejection
from speck.provenance.io import file_sha256


def test_index_matches_scalar_eligibility_and_preserves_physical_rows(tmp_path):
    root = Path(__file__).resolve().parents[2]
    policy = {
        "filters": json.loads(
            (
                root / "archive/pregrant-history/research/flagship/stack_edu_sample_v1.json"
            ).read_text()
        )["filters"],
        "excluded_path_components": ["vendor"],
    }
    rows = [
        dict(
            blob_id=f"{i:040x}",
            language="Rust",
            repo_name="example/repo",
            path="src/lib.rs",
            src_encoding="UTF-8",
            length_bytes=200,
            int_score=4 if i % 3 else 2,
            score=4.5,
            detected_licenses=["MIT"],
            license_type="permissive",
        )
        for i in range(30)
    ]
    rows[5]["path"] = "vendor/file.rs"
    rows[7]["detected_licenses"] = ["GPL-3.0"]
    raw = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw, row_group_size=7)
    unit = {
        "metadata_path": str(raw),
        "language": "Rust",
        "start_row": 0,
        "stop_row": 30,
        "expected_file_rows": 30,
        "raw": {"sha256": file_sha256(raw), "bytes": raw.stat().st_size},
    }
    report = build_eligible_index(unit, policy, tmp_path / "index", maximum_index_bytes=1000000)
    indexed = [json.loads(line) for line in Path(report["output"]["path"]).read_text().splitlines()]
    assert [r["source_row"] for r in indexed] == [
        i for i, row in enumerate(rows) if metadata_rejection(row, "Rust", policy) is None
    ]
    assert report["eligible_rows"] + sum(report["metadata_rejections"].values()) == 30
    sample = sample_index(report, per_stratum=2, strata=4, seed=42)
    assert sample == sample_index(report, per_stratum=2, strata=4, seed=42)
    assert len(sample) == 8 and len({r["eligible_ordinal"] for r in sample}) == 8
    assert (
        build_eligible_index(unit, policy, tmp_path / "index", maximum_index_bytes=1000000)
        == report
    )
    modified = copy.deepcopy(policy)
    modified["excluded_path_components"] = []
    with pytest.raises(ValueError, match="configuration"):
        build_eligible_index(unit, modified, tmp_path / "index", maximum_index_bytes=1000000)
    Path(report["output"]["path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        sample_index(report, per_stratum=2, strata=4, seed=42)
