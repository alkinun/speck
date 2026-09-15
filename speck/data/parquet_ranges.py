"""Bounded, recorded HTTP ranges for Parquet metadata discovery, never raw-file qualification."""

import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from speck.provenance.io import durable_json


def merge_ranges(ranges, *, gap, maximum):
    """Coalesce nearby metadata chunks without crossing a large excluded content chunk."""
    merged = []
    for start, length in sorted(ranges):
        end = start + length
        if start < 0 or length <= 0 or length > maximum:
            raise ValueError("invalid or oversized metadata range")
        if merged and start <= merged[-1][1] + gap and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [(start, end - start) for start, end in merged]


class RecordedRangeFile(io.RawIOBase):
    """Seekable read-only view with bounded transfers and independently hashed range payloads.

    The caller supplies an immutable URL and declared whole-file identity. That full SHA is
    provenance, not a hash this partial reader verifies. Failed/interrupted attempts survive
    resume; only payloads with a completed, matching receipt may be reused.
    """

    def __init__(self, directory, contract, *, resume=False, get=None):
        super().__init__()
        self.directory = Path(directory)
        self.contract = contract
        self.position = 0
        self.lock = threading.Lock()
        self.get = get or requests.get
        self.chunks = []
        self.attempts = []
        self.reserved_bytes = 0
        self.get_count = 0
        for key in ("size", "maximum_transfer_bytes", "maximum_requests", "maximum_range_bytes"):
            if type(contract[key]) is not int or contract[key] <= 0:
                raise ValueError("range contract limits must be positive integers")
        if self.directory.exists():
            if not resume or json.loads((self.directory / "contract.json").read_text()) != contract:
                raise ValueError("preserve range attempts; resume requires identical contract")
            for directory in sorted(self.directory.glob("attempt-*")):
                request = json.loads((directory / "request.json").read_text())
                self.reserved_bytes += request["length"] + 1
                self.attempts.append(directory)
                receipt = directory / "result.json"
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    if result["status"] == "verified_range_not_full_file":
                        payload = (directory / "payload.bin").read_bytes()
                        if (
                            len(payload) != request["length"]
                            or hashlib.sha256(payload).hexdigest() != result["sha256"]
                            or result["start"] != request["start"]
                            or result["length"] != request["length"]
                        ):
                            raise ValueError("completed range payload or receipt changed")
                        self.chunks.append((request["start"], payload))
        else:
            if resume:
                raise ValueError("cannot resume missing range execution")
            self.directory.mkdir(parents=True)
            durable_json(self.directory / "contract.json", contract)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError("unsupported seek origin")
        position = (0, self.position, self.contract["size"])[whence] + offset
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        length = max(0, self.contract["size"] - self.position)
        if size is not None and size >= 0:
            length = min(length, size)
        if not length:
            return b""
        data = self.fetch(self.position, length)
        self.position += len(data)
        return data

    def fetch(self, start, length):
        self._checkClosed()
        with self.lock:
            for offset, payload in self.chunks:
                if offset <= start and start + length <= offset + len(payload):
                    return payload[start - offset : start - offset + length]
            if (
                start < 0
                or length <= 0
                or start + length > self.contract["size"]
                or length > self.contract["maximum_range_bytes"]
                or self.reserved_bytes + length + 1 > self.contract["maximum_transfer_bytes"]
                or len(self.attempts) >= self.contract["maximum_requests"]
            ):
                raise ValueError("range bounds or transfer/request budget exceeded")
            directory = self.directory / f"attempt-{len(self.attempts):05d}"
            directory.mkdir()
            durable_json(directory / "request.json", {"start": start, "length": length})
            self.attempts.append(directory)
            self.reserved_bytes += length + 1
            self.get_count += 1
        started = time.perf_counter()
        received = 0
        try:
            # Separate cache keys prevent stale proxy responses to distinct Range headers.
            separator = "&" if "?" in self.contract["url"] else "?"
            url = self.contract["url"] + f"{separator}speck_range={start}-{start + length - 1}"
            with self.get(
                url,
                headers={
                    "Range": f"bytes={start}-{start + length - 1}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=(15, 60),
            ) as response:
                expected = f"bytes {start}-{start + length - 1}/{self.contract['size']}"
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != expected
                    or response.headers.get("Content-Encoding", "identity") != "identity"
                    or (
                        response.headers.get("Content-Length") is not None
                        and response.headers["Content-Length"] != str(length)
                    )
                ):
                    raise ValueError("server did not return the exact identity-encoded range")
                # Read one extra byte to reject overlong bodies; reservation includes that byte.
                with (directory / "payload.bin").open("xb") as handle:
                    try:
                        while received < length + 1:
                            chunk = response.raw.read(min(65536, length + 1 - received))
                            if not chunk:
                                break
                            handle.write(chunk)
                            received += len(chunk)
                    finally:
                        handle.flush()
                        os.fsync(handle.fileno())
                if received != length:
                    raise ValueError("truncated or overlong range body")
                payload = (directory / "payload.bin").read_bytes()
            durable_json(
                directory / "result.json",
                {
                    "status": "verified_range_not_full_file",
                    "start": start,
                    "length": length,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            with self.lock:
                self.chunks.append((start, payload))
            return payload
        except BaseException as error:
            durable_json(
                directory / "result.json",
                {
                    "status": "failed_range_preserved",
                    "error_type": type(error).__name__,
                    "received_bytes": received,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            raise

    def prefetch(self, ranges, *, workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda span: self.fetch(*span), ranges))


def metadata_ranges(metadata, columns, *, gap, maximum):
    """Locate complete compressed chunks for exactly the named physical Parquet columns."""
    ranges = []
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        found = set()
        for column in (group.column(i) for i in range(group.num_columns)):
            if column.path_in_schema not in columns:
                continue
            found.add(column.path_in_schema)
            offsets = [column.data_page_offset, column.dictionary_page_offset]
            ranges.append(
                (min(x for x in offsets if x is not None and x >= 0), column.total_compressed_size)
            )
        if found != set(columns):
            raise ValueError("required metadata column missing")
    return merge_ranges(ranges, gap=gap, maximum=maximum)
