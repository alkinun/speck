"""Single-owner bounded SWH cache accounting for production ordered acquisition."""

import re
import shutil
import threading
from pathlib import Path

from speck.data.swh_cache import fetch_cached_blob, read_cached_blob


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


class StockBlobStore:
    """Reserve worst-case retry space per live fetch, then charge actual retained attempts.

    One process owns the new cache. Ordered prefetch deduplicates live IDs; this store also
    rejects duplicate concurrent ownership so accounting cannot silently double-spend space.
    Old caches are read-only fallbacks in declared priority order, verified before reuse.
    """

    def __init__(self, cache, fallbacks, settings, *, maximum_bytes, minimum_free_bytes):
        self.cache = Path(cache).resolve()
        self.fallbacks = [Path(p).resolve() for p in fallbacks]
        if any(
            p == self.cache or p.is_relative_to(self.cache) or self.cache.is_relative_to(p)
            for p in self.fallbacks
        ):
            raise ValueError("stock cache overlaps a preserved fallback")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.maximum_bytes = maximum_bytes
        self.minimum_free_bytes = minimum_free_bytes
        self.maximum_per_blob = settings["attempts"] * (
            settings["maximum_compressed_bytes"] + 65536
        )
        self.used = directory_bytes(self.cache)
        self.reserved = 0
        self.live = set()
        self.lock = threading.Lock()
        if self.used > maximum_bytes:
            raise ValueError("existing cache exceeds its bound")

    def fetch(self, item):
        blob = item["blob_id"]
        if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise ValueError("invalid stock blob identity")
        for base in (self.cache, *self.fallbacks):
            directory = base / blob[:2] / blob
            if (directory / "manifest.json").exists():
                _, receipt = read_cached_blob(directory, blob, self.settings)
                return receipt
        directory = self.cache / blob[:2] / blob
        before = directory_bytes(directory)
        with self.lock:
            if blob in self.live:
                raise ValueError("duplicate live stock-cache owner")
            if (
                self.used + self.reserved + self.maximum_per_blob > self.maximum_bytes
                or shutil.disk_usage(self.cache).free
                < self.minimum_free_bytes + self.reserved + self.maximum_per_blob
            ):
                raise ValueError("stock blob cache reservation/free-space bound reached")
            self.reserved += self.maximum_per_blob
            self.live.add(blob)
        try:
            _, receipt = fetch_cached_blob(blob, self.cache, self.settings)
            return receipt
        finally:
            after = directory_bytes(directory)
            with self.lock:
                self.used += after - before
                self.reserved -= self.maximum_per_blob
                self.live.remove(blob)
                if after - before > self.maximum_per_blob or self.used > self.maximum_bytes:
                    raise ValueError("retained fetch exceeded its reservation")
