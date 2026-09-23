"""Dataset packing components."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class TokenShardWriter:
    """Write token IDs (or a parallel uint8 stream) into bounded, checksummed shards."""

    def __init__(
        self, directory, split, shard_tokens, *, shards=None, total_tokens=0, dtype="<u2", name=None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.name = name or split
        self.dtype = np.dtype(dtype)
        self.shard_tokens = shard_tokens
        self.shards = list(shards or [])
        self.total_tokens = total_tokens
        if sum(shard["tokens"] for shard in self.shards) != self.total_tokens:
            raise ValueError(f"invalid restored {split} shard totals")
        self._array: np.memmap | None = None
        self._path: Path | None = None
        self._position = 0
        self._hasher: Any = None

    def _open(self):
        index = len(self.shards)
        self._path = self.directory / f"{self.name}_{index:05d}.bin.tmp"
        self._array = np.memmap(self._path, mode="w+", dtype=self.dtype, shape=(self.shard_tokens,))
        self._position = 0
        self._hasher = hashlib.sha256()

    def write(self, token_ids):
        values = np.asarray(token_ids, dtype=np.int64)
        if values.size == 0:
            return 0
        if values.min() < 0 or values.max() > np.iinfo(self.dtype).max:
            raise ValueError(f"values must fit in {self.dtype}")
        written = 0
        while written < values.size:
            if self._array is None:
                self._open()
            assert self._array is not None and self._hasher is not None
            count = min(values.size - written, self.shard_tokens - self._position)
            chunk = values[written : written + count].astype(self.dtype, copy=False)
            self._array[self._position : self._position + count] = chunk
            self._hasher.update(chunk.tobytes())
            self._position += count
            self.total_tokens += count
            written += count
            if self._position == self.shard_tokens:
                self._close()
        return written

    def _close(self):
        if self._array is None:
            return
        assert self._path is not None and self._hasher is not None
        self._array.flush()
        del self._array
        self._array = None
        with self._path.open("r+b") as handle:
            handle.truncate(self._position * self.dtype.itemsize)
        final_path = self._path.with_suffix("")
        self._path.replace(final_path)
        self.shards.append(
            {
                "path": final_path.name,
                "tokens": self._position,
                "sha256": self._hasher.hexdigest(),
            }
        )
        self._path = None
        self._position = 0
        self._hasher = None

    def finish(self):
        self._close()
        return self.shards


class BestFitRows:
    """Pack whole records into rows of exactly `length` tokens, best fit first.

    A record is never split or truncated. It goes to the open row with the least room that still
    holds it, or opens a new row. Beyond `open_rows` open rows the fullest one closes. Closed rows
    are padded with `pad_id` under a zero mask. Placement depends only on record order.
    """

    def __init__(self, length, open_rows, pad_id):
        self.length = length
        self.open_rows = open_rows
        self.pad_id = pad_id
        self.rows = []

    def add(self, tokens, mask, record):
        """Place one record and return the rows this closes."""

        if not 0 < len(tokens) == len(mask) <= self.length:
            raise ValueError("a packed record must be non-empty, masked and fit one row")
        fitting = [row for row in self.rows if self.length - len(row["tokens"]) >= len(tokens)]
        if fitting:
            row = min(fitting, key=lambda row: self.length - len(row["tokens"]))
        else:
            row = {"tokens": [], "mask": [], "records": []}
            self.rows.append(row)
        row["records"].append((record, len(row["tokens"]), len(tokens)))
        row["tokens"].extend(tokens)
        row["mask"].extend(mask)
        if len(self.rows) <= self.open_rows:
            return []
        fullest = min(self.rows, key=lambda row: self.length - len(row["tokens"]))
        return [self._close(fullest)]

    def flush(self):
        """Close every open row in the order it was opened."""

        closed = [self._close(row) for row in list(self.rows)]
        return closed

    @property
    def pending_tokens(self):
        return len(self.rows) * self.length

    def _close(self, row):
        self.rows.remove(row)
        padding = self.length - len(row["tokens"])
        return (
            row["tokens"] + [self.pad_id] * padding,
            row["mask"] + [0] * padding,
            row["records"],
        )
