"""Retain bounded Software Heritage blob fetches, including failed request attempts."""

import gzip
import hashlib
import io
import json
import os
import re
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

from speck.provenance.io import durable_json, file_sha256

_SESSIONS = threading.local()


def _get(url, timeout):
    if not hasattr(_SESSIONS, "session"):
        _SESSIONS.session = requests.Session()
    return _SESSIONS.session.get(url, timeout=timeout, stream=True)


def _decode(payload, maximum):
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("SWH blob exceeds the declared uncompressed envelope")
    return raw


def read_cached_blob(directory, blob_id, settings):
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("blob_id") != blob_id
        or manifest.get("base_url") != settings["base_url"]
        or manifest.get("format") != "speck_swh_blob_cache"
        or manifest.get("status") not in ("available", "missing")
    ):
        raise ValueError("SWH cache ownership/status mismatch")
    entry = manifest["terminal_attempt"]
    attempt_path = directory / entry["path"]
    if attempt_path.parent != directory or file_sha256(attempt_path) != entry["sha256"]:
        raise ValueError("SWH terminal attempt identity mismatch")
    attempt = json.loads(attempt_path.read_text())
    if (
        attempt.get("blob_id") != blob_id
        or attempt.get("base_url") != settings["base_url"]
        or attempt.get("http_status") != (200 if manifest["status"] == "available" else 404)
        or "error_type" in attempt
    ):
        raise ValueError("SWH terminal attempt status mismatch")
    if manifest["status"] == "available" and attempt.get("payload") != manifest["payload"]:
        raise ValueError("SWH terminal payload mismatch")
    raw = None
    if manifest["status"] == "available":
        entry = manifest["payload"]
        payload_path = directory / entry["path"]
        if payload_path.parent != directory or file_sha256(payload_path) != entry["sha256"]:
            raise ValueError("SWH compressed payload identity mismatch")
        if (
            payload_path.stat().st_size != entry["bytes"]
            or entry["bytes"] > settings["maximum_compressed_bytes"]
        ):
            raise ValueError("SWH compressed payload exceeds envelope")
        raw = _decode(payload_path.read_bytes(), settings["maximum_blob_bytes"])
        if (
            hashlib.sha1(raw).hexdigest() != blob_id
            or hashlib.sha256(raw).hexdigest() != manifest["raw_sha256"]
            or len(raw) != manifest["raw_bytes"]
        ):
            raise ValueError("SWH decompressed payload identity mismatch")
    return raw, {"path": str(path.resolve()), "sha256": file_sha256(path)}


def fetch_cached_blob(blob_id, cache, settings):
    if not isinstance(blob_id, str) or not re.fullmatch(r"[0-9a-f]{40}", blob_id):
        raise ValueError("invalid Software Heritage SHA-1 blob identifier")
    directory = Path(cache) / blob_id[:2] / blob_id
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return read_cached_blob(directory, blob_id, settings)
    # A caller dispatches each unique ID at most once concurrently. Successful
    # and 404 receipts are stable inputs; transient failures never become omissions.
    for attempt in range(settings["attempts"]):
        name = "attempt-" + uuid4().hex
        started = time.perf_counter()
        record = {
            "blob_id": blob_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "base_url": settings["base_url"],
        }
        manifest = None
        payload = None
        try:
            with _get(settings["base_url"] + blob_id, settings["timeout_seconds"]) as response:
                record["http_status"] = response.status_code
                if response.status_code == 404:
                    manifest = {"status": "missing"}
                elif response.status_code == 200:
                    payload = directory / (name + ".gz")
                    size = 0
                    with payload.open("xb") as handle:
                        for chunk in response.iter_content(chunk_size=65536):
                            if size + len(chunk) > settings["maximum_compressed_bytes"]:
                                raise ValueError("SWH response exceeds compressed envelope")
                            handle.write(chunk)
                            size += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    record["payload"] = {
                        "path": payload.name,
                        "sha256": file_sha256(payload),
                        "bytes": size,
                    }
                    raw = _decode(payload.read_bytes(), settings["maximum_blob_bytes"])
                    if hashlib.sha1(raw).hexdigest() != blob_id:
                        raise ValueError("Software Heritage content SHA-1 mismatch")
                    manifest = {
                        "status": "available",
                        "payload": record["payload"],
                        "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "raw_bytes": len(raw),
                    }
                else:
                    record["error_type"] = "HTTPStatus"
        except (OSError, ValueError, EOFError, zlib.error, requests.RequestException) as error:
            record["error_type"] = type(error).__name__
        finally:
            if payload is not None and payload.exists():
                record["payload"] = {
                    "path": payload.name,
                    "sha256": file_sha256(payload),
                    "bytes": payload.stat().st_size,
                }
            record["elapsed_seconds"] = time.perf_counter() - started
            durable_json(directory / (name + ".json"), record)
        if manifest is not None:
            manifest.update(
                format="speck_swh_blob_cache",
                format_version=1,
                blob_id=blob_id,
                base_url=settings["base_url"],
                terminal_attempt={
                    "path": name + ".json",
                    "sha256": file_sha256(directory / (name + ".json")),
                },
            )
            durable_json(directory / "manifest.json", manifest)
            return read_cached_blob(directory, blob_id, settings)
        if attempt + 1 < settings["attempts"]:
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"SWH fetch failed after recorded attempts: {blob_id}")
