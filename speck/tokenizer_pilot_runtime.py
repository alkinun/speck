"""Consume corrected tokenizer-pilot token streams with exact resumable cursors."""

import bisect
import json
from pathlib import Path

import numpy as np
import torch

from speck.io import file_sha256
from speck.tokenizer_pilot_runs import _fingerprint
from speck.validation import positive_integer, require_keys


def validate_pilot_run_manifest(value):
    """Validate a corrected, execution-blocked v2 run before reading its data."""

    required = {
        "format",
        "format_version",
        "status",
        "run_id",
        "run_fingerprint",
        "tokenizer_id",
        "repository_revision",
        "training_data",
        "settings",
        "stops",
        "authority",
    }
    if not isinstance(value, dict):
        raise ValueError("tokenizer pilot run must be an object")
    require_keys(value, required, "tokenizer pilot run")
    if (
        value["format"] != "speck_tokenizer_pilot_corrected_screen_run"
        or value["format_version"] != 2
        or value["status"] != "corrected_materialized_not_started_execution_blocked"
    ):
        raise ValueError("unsupported corrected tokenizer pilot run")
    payload = {key: item for key, item in value.items() if key != "run_fingerprint"}
    if value["run_fingerprint"] != _fingerprint(payload):
        raise ValueError("tokenizer pilot run fingerprint mismatch")
    authority = value["authority"]
    if authority != {
        "v2_run_materialization": True,
        "screen_execution": False,
        "confirmation_execution": False,
        "D5_opening": False,
        "final_selection": False,
        "flagship_training": False,
    }:
        raise ValueError("tokenizer pilot run authority is invalid")
    settings = value["settings"]
    for key in ("sequence_length", "device_batch_size", "batch_tokens", "accumulation"):
        positive_integer(settings.get(key), key.replace("_", " "))
    if settings["batch_tokens"] != (
        settings["sequence_length"] * settings["device_batch_size"] * settings["accumulation"]
    ):
        raise ValueError("tokenizer pilot run batch geometry is inconsistent")
    stops = value["stops"]
    for key in (
        "fixed_document_tokens",
        "fixed_document_step",
        "fixed_flop_token_stop",
        "final_step",
        "run_stop_aligned_tokens",
    ):
        positive_integer(stops.get(key), key.replace("_", " "))
    if (
        stops["run_stop_aligned_tokens"] != stops["final_step"] * settings["batch_tokens"]
        or stops["fixed_document_step"]
        != (stops["fixed_document_tokens"] + settings["batch_tokens"] - 1)
        // settings["batch_tokens"]
    ):
        raise ValueError("tokenizer pilot run stopping geometry is inconsistent")
    training = value["training_data"]
    require_keys(
        training,
        {"fixed_stream", "continuation", "fixed_shards", "continuation_shards"},
        "tokenizer pilot training data",
    )
    shards = [*training["fixed_shards"], *training["continuation_shards"]]
    if not shards:
        raise ValueError("tokenizer pilot run has no packed shards")
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "tokens", "category"}:
            raise ValueError("tokenizer pilot packed shard declaration is invalid")
        positive_integer(shard["tokens"], "packed shard tokens")
        if not isinstance(shard["path"], str) or not isinstance(shard["sha256"], str):
            raise ValueError("tokenizer pilot packed shard identity is invalid")
    fixed_tokens = sum(shard["tokens"] for shard in training["fixed_shards"])
    available = sum(shard["tokens"] for shard in shards)
    if fixed_tokens != stops["fixed_document_tokens"]:
        raise ValueError("fixed-document stop differs from packed shard totals")
    if available <= stops["run_stop_aligned_tokens"]:
        raise ValueError("tokenizer pilot stream lacks the final next-token target")
    return value


def load_pilot_run_manifest(path):
    return validate_pilot_run_manifest(json.loads(Path(path).read_text()))


class PilotTokenStream:
    """Expose fixed and continuation shards as one immutable uint16 stream."""

    def __init__(self, run, *, verify_hashes=True):
        self.run = validate_pilot_run_manifest(run)
        self.shard_manifests = [
            *run["training_data"]["fixed_shards"],
            *run["training_data"]["continuation_shards"],
        ]
        self.shards = []
        self.ends = []
        total = 0
        for shard in self.shard_manifests:
            path = Path(shard["path"])
            expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"invalid tokenizer pilot shard size: {path}")
            if verify_hashes and file_sha256(path) != shard["sha256"]:
                raise ValueError(f"tokenizer pilot shard checksum mismatch: {path}")
            self.shards.append(np.memmap(path, mode="r", dtype="<u2"))
            total += shard["tokens"]
            self.ends.append(total)
        self.total_tokens = total
        self.fixed_tokens = sum(shard["tokens"] for shard in run["training_data"]["fixed_shards"])
        self.maximum_offset = run["stops"]["run_stop_aligned_tokens"]

    def read(self, start, count, dtype=np.int64):
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or start < 0
            or count < 0
            or start + count > self.total_tokens
        ):
            raise IndexError("tokenizer pilot token read is out of range")
        if count == 0:
            return np.empty(0, dtype=dtype)
        pieces = []
        position = start
        remaining = count
        while remaining:
            index = bisect.bisect_right(self.ends, position)
            shard_start = 0 if index == 0 else self.ends[index - 1]
            offset = position - shard_start
            take = min(remaining, len(self.shards[index]) - offset)
            pieces.append(self.shards[index][offset : offset + take])
            position += take
            remaining -= take
        values = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return np.array(values, dtype=dtype, copy=True)


class PilotBatchLoader:
    """Yield contiguous microbatches and exact next-batch resume state."""

    def __init__(self, stream, *, device="cpu", resume_state=None):
        self.stream = stream
        settings = stream.run["settings"]
        self.sequence_length = settings["sequence_length"]
        self.batch_size = settings["device_batch_size"]
        self.stride = self.sequence_length * self.batch_size
        self.device = torch.device(device)
        self.offset = 0
        if resume_state is not None:
            self.load_state_dict(resume_state)

    def __iter__(self):
        return self

    def __next__(self):
        required = self.stride + 1
        if (
            self.offset >= self.stream.maximum_offset
            or self.offset + required > self.stream.total_tokens
        ):
            raise StopIteration
        state = self.state_dict()
        dtype = np.uint16 if self.device.type == "cuda" else np.int64
        flat = torch.from_numpy(self.stream.read(self.offset, required, dtype=dtype))
        if self.device.type == "cuda":
            flat = flat.pin_memory().to(self.device, dtype=torch.int64, non_blocking=True)
        else:
            flat = flat.to(self.device)
        rows = flat.unfold(0, self.sequence_length + 1, self.sequence_length)
        self.offset += self.stride
        return rows[:, :-1], rows[:, 1:], state

    def state_dict(self):
        return {
            "format_version": 1,
            "contract": "tokenizer_pilot_batch_start",
            "run_fingerprint": self.stream.run["run_fingerprint"],
            "token_offset": self.offset,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or state.get("format_version") != 1
            or state.get("contract") != "tokenizer_pilot_batch_start"
            or state.get("run_fingerprint") != self.stream.run["run_fingerprint"]
            or state.get("sequence_length") != self.sequence_length
            or state.get("batch_size") != self.batch_size
        ):
            raise ValueError("tokenizer pilot resume state differs from the run")
        offset = state.get("token_offset")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset % self.stride
            or offset > self.stream.maximum_offset
        ):
            raise ValueError("tokenizer pilot resume offset is invalid")
        self.offset = offset
