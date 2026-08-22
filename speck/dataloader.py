"""Load distributed batches from packed uint16 token shards."""

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from speck.common import dist_info
from speck.dataset import default_data_dir, load_manifest


class PackedTokenSplit:
    """Expose packed token shards as one contiguous memory-mapped split."""

    def __init__(self, data_dir, split, manifest):
        self.data_dir = Path(data_dir)
        split_manifest = manifest["splits"][split]
        self.shards = []
        self.ends = []
        total = 0
        for shard in split_manifest["shards"]:
            path = self.data_dir / shard["path"]
            expected_bytes = shard["tokens"] * np.dtype("<u2").itemsize
            if not path.exists() or path.stat().st_size != expected_bytes:
                raise ValueError(f"invalid packed token shard: {path}")
            self.shards.append(np.memmap(path, mode="r", dtype="<u2"))
            total += shard["tokens"]
            self.ends.append(total)
        if total != split_manifest["tokens"] or not self.shards:
            raise ValueError(f"invalid {split} token count in manifest")
        self.total_tokens = total

    def read(self, start, count):
        if start < 0 or start + count > self.total_tokens:
            raise IndexError("packed token read is out of range")
        pieces = []
        position = start
        remaining = count
        while remaining:
            shard_index = bisect.bisect_right(self.ends, position)
            shard_start = 0 if shard_index == 0 else self.ends[shard_index - 1]
            offset = position - shard_start
            take = min(remaining, len(self.shards[shard_index]) - offset)
            pieces.append(self.shards[shard_index][offset : offset + take])
            position += take
            remaining -= take
        return np.array(
            pieces[0] if len(pieces) == 1 else np.concatenate(pieces), dtype=np.int64, copy=True
        )

    def shard_at(self, position):
        return bisect.bisect_right(self.ends, min(position, self.total_tokens - 1))


def manifest_fingerprint(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def packed_loader(
    tokenizer,
    batch_size,
    sequence_length,
    split,
    device: str | torch.device = "cuda",
    resume_state_dict=None,
    data_dir=None,
):
    """Yield losslessly packed batches and each batch's exact start cursor."""
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    data_dir = Path(data_dir or default_data_dir / "packed")
    manifest = load_manifest(data_dir)
    tokenizer_manifest = manifest["tokenizer"]
    if tokenizer.vocab_size != tokenizer_manifest["vocab_size"]:
        raise ValueError("packed dataset vocabulary does not match tokenizer")
    if tokenizer.fingerprint() != tokenizer_manifest["fingerprint"]:
        raise ValueError("packed dataset was created with a different tokenizer")

    packed = PackedTokenSplit(data_dir, split, manifest)
    rank, _, world_size = dist_info()
    global_stride = batch_size * sequence_length * world_size
    required = batch_size * sequence_length + 1
    dataset_hash = manifest_fingerprint(manifest)
    global_offset = 0
    epoch = 0
    if resume_state_dict is not None:
        if resume_state_dict.get("manifest") != dataset_hash:
            raise ValueError("cannot resume with a different packed dataset")
        if (
            resume_state_dict.get("sequence_length") != sequence_length
            or resume_state_dict.get("batch_size") != batch_size
        ):
            raise ValueError("cannot resume with different batch geometry")
        if resume_state_dict.get("world_size") != world_size:
            raise ValueError("cannot resume with a different world size")
        global_offset = resume_state_dict["global_offset"]
        epoch = resume_state_dict["epoch"]

    device = torch.device(device)
    while True:
        if global_offset + global_stride + 1 > packed.total_tokens:
            global_offset = 0
            epoch += 1
        rank_offset = global_offset + rank * batch_size * sequence_length
        if global_stride + 1 > packed.total_tokens:
            raise ValueError("packed split is smaller than one distributed batch")

        state = {
            "format_version": 1,
            "manifest": dataset_hash,
            "global_offset": global_offset,
            "epoch": epoch,
            "shard": packed.shard_at(rank_offset),
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        flat = torch.from_numpy(packed.read(rank_offset, required))
        rows = flat.unfold(0, sequence_length + 1, sequence_length)
        inputs = rows[:, :-1].contiguous()
        targets = rows[:, 1:].contiguous()
        if device.type == "cuda":
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        yield inputs, targets, state
        global_offset += global_stride
