"""Dataset packing components."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class TokenShardWriter:
    """Write token IDs into bounded, checksummed uint16 shards."""

    def __init__(self, directory, split, shard_tokens, *, shards=None, total_tokens=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
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
        self._path = self.directory / f"{self.split}_{index:05d}.bin.tmp"
        self._array = np.memmap(self._path, mode="w+", dtype="<u2", shape=(self.shard_tokens,))
        self._position = 0
        self._hasher = hashlib.sha256()

    def write(self, token_ids):
        values = np.asarray(token_ids, dtype=np.int64)
        if values.size == 0:
            return 0
        if values.min() < 0 or values.max() > np.iinfo(np.uint16).max:
            raise ValueError("token IDs must fit in uint16")
        written = 0
        while written < values.size:
            if self._array is None:
                self._open()
            assert self._array is not None and self._hasher is not None
            count = min(values.size - written, self.shard_tokens - self._position)
            chunk = values[written : written + count].astype("<u2", copy=False)
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
            handle.truncate(self._position * np.dtype("<u2").itemsize)
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
