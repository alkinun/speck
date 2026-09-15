import io
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speck.data.parquet_ranges import RecordedRangeFile, metadata_ranges


class Response:
    def __init__(self, data, start, length, *, status=206, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {
            "Content-Range": f"bytes {start}-{start + length - 1}/{len(data)}",
            "Content-Length": str(length),
        }
        self.raw = io.BytesIO(data[start : start + length] if payload is None else payload)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()


def contract(size, **changes):
    return {
        "url": "https://example.invalid/pinned/file.parquet",
        "sha256": "a" * 64,
        "size": size,
        "maximum_transfer_bytes": 2**20,
        "maximum_requests": 100,
        "maximum_range_bytes": 2**20,
        **changes,
    }


def getter(data, **changes):
    def get(url, headers, stream, timeout):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        assert stream and headers["Accept-Encoding"] == "identity"
        return Response(data, start, end - start + 1, **changes)

    return get


def test_real_parquet_projection_avoids_content_and_reopens_without_network(tmp_path):
    rng = random.Random(42)
    table = pa.table({"repo": range(8), "content": [rng.randbytes(200000) for _ in range(8)]})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=2, compression="NONE")
    data = sink.getvalue().to_pybytes()
    directory = tmp_path / "ranges"
    with RecordedRangeFile(directory, contract(len(data)), get=getter(data)) as source:
        parquet = pq.ParquetFile(source)
        ranges = metadata_ranges(parquet.metadata, {"repo"}, gap=0, maximum=2**20)
        source.prefetch(ranges, workers=2)
        assert parquet.read(columns=["repo"]).equals(table.select(["repo"]))
        assert source.reserved_bytes < len(data) // 10
        attempts = len(source.attempts)

    def no_network(*args, **kwargs):
        pytest.fail("completed projection unexpectedly requested network")

    with RecordedRangeFile(directory, contract(len(data)), resume=True, get=no_network) as source:
        assert pq.ParquetFile(source).read(columns=["repo"]).equals(table.select(["repo"]))
        assert len(source.attempts) == attempts
        source.seek(0)
        with pytest.raises(ValueError, match="budget"):
            source.read()  # Accidental full-text access fails closed.


@pytest.mark.parametrize(
    "changes",
    [
        {"status": 200},
        {"headers": {"Content-Range": "bytes 0-3/999"}},
        {"headers": {"Content-Range": "bytes 0-3/10", "Content-Encoding": "gzip"}},
        {"payload": b"abc"},
        {"payload": b"abcde"},
    ],
)
def test_invalid_range_response_is_preserved_and_never_reused(tmp_path, changes):
    data = b"0123456789"
    with RecordedRangeFile(
        tmp_path / "ranges", contract(10), get=getter(data, **changes)
    ) as source:
        with pytest.raises(ValueError):
            source.fetch(0, 4)
    result = tmp_path / "ranges/attempt-00000/result.json"
    prior = result.read_bytes()
    assert json.loads(prior)["status"] == "failed_range_preserved"
    with RecordedRangeFile(
        tmp_path / "ranges", contract(10), resume=True, get=getter(data)
    ) as source:
        assert source.fetch(0, 4) == data[:4]
        assert source.reserved_bytes == 10  # Failed request still consumes its reservation.
    assert result.read_bytes() == prior


def test_resume_rejects_contract_and_completed_payload_changes(tmp_path):
    data = b"0123456789"
    directory = tmp_path / "ranges"
    with RecordedRangeFile(directory, contract(10), get=getter(data)) as source:
        source.fetch(0, 4)
    with pytest.raises(ValueError, match="identical contract"):
        RecordedRangeFile(directory, contract(11), resume=True)
    (directory / "attempt-00000/payload.bin").write_bytes(b"xxxx")
    with pytest.raises(ValueError, match="payload or receipt changed"):
        RecordedRangeFile(directory, contract(10), resume=True)


def test_prefetch_reserves_budget_before_concurrent_requests(tmp_path):
    data = b"0123456789"
    with RecordedRangeFile(
        tmp_path / "ranges", contract(10, maximum_transfer_bytes=5), get=getter(data)
    ) as source:
        with pytest.raises(ValueError, match="budget"):
            source.prefetch([(0, 4), (4, 4)], workers=2)
        assert len(source.attempts) == 1
        assert source.reserved_bytes == 5


def test_interrupted_body_keeps_partial_payload_and_retries_in_a_new_attempt(tmp_path):
    class InterruptedBody(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("injected interrupted body")
            return super().read(2)

    def interrupted(*args, **kwargs):
        response = Response(b"0123456789", 0, 4)
        response.raw = InterruptedBody(b"0123")
        return response

    directory = tmp_path / "ranges"
    with RecordedRangeFile(directory, contract(10), get=interrupted) as source:
        with pytest.raises(OSError, match="injected"):
            source.fetch(0, 4)
    assert (directory / "attempt-00000/payload.bin").read_bytes() == b"01"
    failure = json.loads((directory / "attempt-00000/result.json").read_text())
    assert failure["received_bytes"] == 2
    with RecordedRangeFile(
        directory, contract(10), resume=True, get=getter(b"0123456789")
    ) as source:
        assert source.fetch(0, 4) == b"0123"
        assert len(source.attempts) == 2
    assert (directory / "attempt-00000/payload.bin").read_bytes() == b"01"
