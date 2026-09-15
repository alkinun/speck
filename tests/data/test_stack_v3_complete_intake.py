import hashlib
from pathlib import Path

import pytest

from scripts.acquire_stack_v3_targeted_raw import assemble_verified
from speck.data.parquet_ranges import RecordedRangeFile
from tests.data.test_parquet_ranges import contract, getter


def test_complete_assembly_resumes_preserved_ranges_without_refetch(tmp_path):
    data = bytes(range(251)) * 3
    spec = contract(len(data), sha256=hashlib.sha256(data).hexdigest(), maximum_range_bytes=128)
    cache = tmp_path / "ranges"
    with RecordedRangeFile(cache, spec, get=getter(data)) as source:
        source.fetch(0, 128)
    first = (cache / "attempt-00000/payload.bin").read_bytes()
    requests = []

    def remaining(url, **kwargs):
        requests.append(kwargs["headers"]["Range"])
        return getter(data)(url, **kwargs)

    destination = tmp_path / "complete.parquet"
    with RecordedRangeFile(cache, spec, resume=True, get=remaining) as source:
        assemble_verified(source, destination)
        assert source.reserved_bytes == len(data) + 6
    assert destination.read_bytes() == data
    assert "bytes=0-127" not in requests
    assert (cache / "attempt-00000/payload.bin").read_bytes() == first
    with pytest.raises(FileExistsError):
        with RecordedRangeFile(cache, spec, resume=True) as source:
            assemble_verified(source, destination)


def test_range_receipts_cannot_substitute_for_full_file_hash(tmp_path):
    data = b"different complete bytes"
    spec = contract(
        len(data), sha256=hashlib.sha256(b"expected").hexdigest(), maximum_range_bytes=8
    )
    destination = tmp_path / "assembly-00000.parquet"
    with RecordedRangeFile(tmp_path / "ranges", spec, get=getter(data)) as source:
        with pytest.raises(ValueError, match="complete file identity mismatch"):
            assemble_verified(source, destination)
    assert destination.read_bytes() == data
    assert len(list(Path(tmp_path / "ranges").glob("attempt-*/result.json"))) == 3
