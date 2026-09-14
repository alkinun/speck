"""Bounded concurrent blob fetching with ordered, checksummed checkpoint publication."""

import hashlib
import json
import os
import re
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from speck.data.acquisition_units import _digest
from speck.data.swh_cache import fetch_cached_blob, read_cached_blob
from speck.provenance.io import durable_json, file_sha256


def ordered_prefetch(items, function, *, workers, window, key):
    """Keep a bounded window in flight; duplicate keys share one live future."""
    if type(workers) is not int or type(window) is not int or not 1 <= workers <= window <= 256:
        raise ValueError("invalid bounded prefetch concurrency")
    iterator, pending, shared = iter(items), deque(), {}
    with ThreadPoolExecutor(max_workers=workers) as executor:

        def submit():
            try:
                item = next(iterator)
            except StopIteration:
                return False
            identity = key(item)
            if identity not in shared:
                shared[identity] = [executor.submit(function, item), 0]
            shared[identity][1] += 1
            pending.append((item, identity, shared[identity][0]))
            return True

        for _ in range(window):
            if not submit():
                break
        while pending:
            item, identity, future = pending.popleft()
            result = future.result()
            shared[identity][1] -= 1
            if not shared[identity][1]:
                del shared[identity]
            # Refill before yielding to keep workers active during consumer processing.
            submit()
            yield item, result


def fetch_targets(
    targets,
    output,
    cache,
    fallback,
    settings,
    *,
    workers=32,
    window=64,
    maximum_working_bytes,
    minimum_free_bytes,
    resume=False,
    interrupt_after=None,
):
    output, cache, fallback = (
        Path(output).resolve(),
        Path(cache).resolve(),
        Path(fallback).resolve(),
    )
    if cache == fallback or cache.is_relative_to(fallback) or fallback.is_relative_to(cache):
        raise ValueError("new cache must be separate from the preserved fallback")
    if any(
        not isinstance(row.get("blob_id"), str) or not re.fullmatch(r"[0-9a-f]{40}", row["blob_id"])
        for row in targets
    ):
        raise ValueError("invalid ordered fetch blob identity")
    config = {
        "format": "speck_ordered_blob_fetch",
        "format_version": 1,
        "targets_sha256": _digest(targets),
        "targets": len(targets),
        "cache": str(cache),
        "fallback": str(fallback),
        "settings": settings,
        "workers": workers,
        "window": window,
        "maximum_working_bytes": maximum_working_bytes,
        "minimum_free_bytes": minimum_free_bytes,
    }
    if output.exists():
        if not resume or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("ordered fetch resume requires its original configuration")
    else:
        if resume:
            raise ValueError("cannot resume absent ordered fetch")
        output.mkdir(parents=True)
        durable_json(output / "config.json", config)
    cache.mkdir(parents=True, exist_ok=True)
    # Conservative whole-invocation reservation also covers retried/partial payloads and receipts.
    maximum_per_blob = settings["attempts"] * (settings["maximum_compressed_bytes"] + 65536)
    missing = {
        row["blob_id"]
        for row in targets
        if not any(
            (base / row["blob_id"][:2] / row["blob_id"] / "manifest.json").exists()
            for base in (cache, fallback)
        )
    }
    reservation = len(missing) * maximum_per_blob
    used = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    if (
        used + reservation > maximum_working_bytes
        or shutil.disk_usage(cache).free < minimum_free_bytes + reservation
    ):
        raise ValueError("ordered fetch exceeds its declared cache/free-space envelope")
    state_path, journal = output / "state.json", output / "fetches.jsonl"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"next_index": 0, "output_bytes": 0, "output_sha256": hashlib.sha256(b"").hexdigest()}
    )
    if not 0 <= state["next_index"] <= len(targets):
        raise ValueError("ordered fetch checkpoint index outside target list")
    completed_path = output / "manifest.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else None
    started = time.perf_counter()

    def fetch(item):
        if shutil.disk_usage(cache).free < minimum_free_bytes + workers * maximum_per_blob:
            raise ValueError("ordered fetch free-space floor reached")
        blob = item["blob_id"]
        previous = fallback / blob[:2] / blob
        if (previous / "manifest.json").exists():
            _, receipt = read_cached_blob(previous, blob, settings)
        else:
            _, receipt = fetch_cached_blob(blob, cache, settings)
        return receipt

    hasher = hashlib.sha256()
    with journal.open("r+b" if journal.exists() else "w+b") as handle:
        remaining = state["output_bytes"]
        while remaining:
            chunk = handle.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("ordered fetch journal truncated")
            hasher.update(chunk)
            remaining -= len(chunk)
        if hasher.hexdigest() != state["output_sha256"]:
            raise ValueError("ordered fetch journal checksum mismatch")
        handle.seek(0)
        for index in range(state["next_index"]):
            row = json.loads(handle.readline())
            if row["index"] != index or row["blob_id"] != targets[index]["blob_id"]:
                raise ValueError("ordered fetch journal order mismatch")
            receipt_path = Path(row["manifest"]["path"])
            expected = {
                base / row["blob_id"][:2] / row["blob_id"] / "manifest.json"
                for base in (cache, fallback)
            }
            if (
                receipt_path not in expected
                or file_sha256(receipt_path) != row["manifest"]["sha256"]
            ):
                raise ValueError("ordered fetch input manifest changed")
            read_cached_blob(receipt_path.parent, row["blob_id"], settings)
        if handle.tell() != state["output_bytes"]:
            raise ValueError("ordered fetch cursor does not match committed journal")
        if completed is not None:
            if (
                completed.get("status") != "complete_verified_fetches_not_text_stock"
                or completed["config_sha256"] != _digest(config)
                or state["next_index"] != len(targets)
                or completed["journal"]["sha256"] != file_sha256(journal)
            ):
                raise ValueError("completed ordered fetch receipt mismatch")
            return completed
        if journal.stat().st_size > state["output_bytes"]:
            tail = output / f"uncommitted-tail-{time.time_ns()}.jsonl"
            with tail.open("xb") as dest:
                shutil.copyfileobj(handle, dest)
                dest.flush()
                os.fsync(dest.fileno())
            durable_json(
                tail.with_suffix(".receipt.json"), {"path": str(tail), "sha256": file_sha256(tail)}
            )
        handle.truncate(state["output_bytes"])
        handle.seek(state["output_bytes"])

        def checkpoint():
            handle.flush()
            os.fsync(handle.fileno())
            state.update(output_bytes=handle.tell(), output_sha256=hasher.hexdigest())
            durable_json(state_path, state)

        stream = ordered_prefetch(
            targets[state["next_index"] :],
            fetch,
            workers=workers,
            window=window,
            key=lambda item: item["blob_id"],
        )
        try:
            for item, receipt in stream:
                row = {
                    "index": state["next_index"],
                    "blob_id": item["blob_id"],
                    "manifest": receipt,
                }
                payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                handle.write(payload)
                hasher.update(payload)
                state["next_index"] += 1
                if state["next_index"] % 32 == 0:
                    checkpoint()
                    print(f"ordered fetch: {state['next_index']} / {len(targets)}", flush=True)
                if interrupt_after is not None and state["next_index"] >= interrupt_after:
                    checkpoint()
                    raise RuntimeError("injected ordered fetch interruption")
            checkpoint()
        finally:
            stream.close()  # Wait for bounded in-flight fetches; their cache attempts remain durable.
    report = {
        "format": "speck_ordered_blob_fetch_result",
        "format_version": 1,
        "status": "complete_verified_fetches_not_text_stock",
        "config_sha256": _digest(config),
        "journal": {"path": str(journal), "sha256": file_sha256(journal)},
        "targets": len(targets),
        "invocation_elapsed_seconds": time.perf_counter() - started,
        "cache_reservation_bytes": reservation,
        "training_authority": False,
    }
    durable_json(output / "manifest.json", report)
    return report
